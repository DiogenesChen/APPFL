import copy
import time
import logging
import numpy as np
import torch.nn as nn
from mpi4py import MPI
from typing import Any
from appfl.algorithm import *
from appfl.misc import validation, save_model_iteration
from appfl.comm.mpi import MpiCommunicator
from appfl.compressor import Compressor
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def run_server(
    cfg: DictConfig,
    comm: MPI.Comm,
    model: nn.Module,
    loss_fn: nn.Module,
    num_clients: int,
    test_dataset: Dataset = Dataset(),
    dataset_name: str = "MNIST",
    metric: Any = None,
):
    """
    run_server:
        Run PPFL server that updates the global model parameter in a synchronous way.

    Args:
        cfg: the configuration for the FL experiment
        comm: MPI communicator
        model: neural network model to train
        loss_fn: loss function
        num_clients: the number of clients used in PPFL simulation
        test_dataset: optional testing data. If given, validation will run based on this data
        dataset_name: optional dataset name
        metric: evaluation metric function
    """
    device = "cpu"  # server aggregation happens on CPU
    communicator = MpiCommunicator(comm, Compressor(cfg) if cfg.enable_compression else None)

    logger = logging.getLogger(__name__)
    logger = create_custom_logger(logger, cfg)
    cfg.logginginfo.comm_size = comm.Get_size()
    cfg.logginginfo.DataSet_name = dataset_name

    ## Using tensorboard to visualize the test loss
    if cfg.use_tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(
            comment=cfg.fed.args.optim + "_clients_nums_" + str(cfg.num_clients)
        )

    ## Run validation if test data is given and the configuration is enabled.
    if cfg.validation == True and len(test_dataset) > 0:
        test_dataloader = DataLoader(
            test_dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.test_data_batch_size,
            shuffle=cfg.test_data_shuffle,
        )
    else:
        cfg.validation = False

    ## Collect the number of data from each client and compute the weights for each client
    num_data = communicator.gather(0, dest=0)
    total_num_data = 0
    for num in num_data:
        total_num_data += num
    weights = [num / total_num_data for num in num_data]
    communicator.scatter(weights, 0)

    ## Synchronous federated learning server
    server = eval(cfg.fed.servername)(
        weights[1:], copy.deepcopy(model), loss_fn, num_clients, device, **cfg.fed.args
    )

    start_time = time.time()
    test_loss, test_accuracy, best_accuracy = 0.0, 0.0, 0.0
    for iter in range(cfg.num_epochs):
        global_broadcast_start = time.time()
        per_iter_start = time.time()
        server.model.to("cpu")
        global_state = server.model.state_dict()

        communicator.broadcast_global_model(global_state, {"done": False})

        local_states = [None for _ in range(num_clients)]
        model_copy = copy.deepcopy(server.model)
        global_broadcast_end = time.time()
        print(f"{communicator.current_time()} [APPFL] Server global_broadcast_time: {global_broadcast_end - global_broadcast_start}, start at {global_broadcast_start}, end at {global_broadcast_end}", flush=True)
        global_recv_start = time.time()
        for _ in range(num_clients):
            client_idx, model = communicator.recv_local_model_from_client(model_copy)
            local_states[client_idx] = model
        global_recv_end = time.time()
        print(f"{communicator.current_time()} [APPFL] Server global_recv_time: {global_recv_end - global_recv_start}, start at {global_recv_start}, end at {global_recv_end}", flush=True)

        cfg.logginginfo.LocalUpdate_time = time.time() - per_iter_start

        global_update_start = time.time()
        server.update(local_states)
        cfg.logginginfo.GlobalUpdate_time = time.time() - global_update_start
        print(f"{communicator.current_time()} [APPFL] Server global_update_time: {time.time() - global_update_start}, start at {global_update_start}, end at {time.time()}", flush=True)

        validation_start = time.time()
        if cfg.validation == True:
            test_loss, test_accuracy = validation(server, test_dataloader, metric)
            if cfg.use_tensorboard:
                writer.add_scalar("server_test_accuracy", test_accuracy, iter)
                writer.add_scalar("server_test_loss", test_loss, iter)
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
        cfg.logginginfo.Validation_time = time.time() - validation_start
        cfg.logginginfo.PerIter_time = time.time() - per_iter_start
        cfg.logginginfo.Elapsed_time = time.time() - start_time
        cfg.logginginfo.test_loss = test_loss
        cfg.logginginfo.test_accuracy = test_accuracy
        cfg.logginginfo.BestAccuracy = best_accuracy

        server.logging_iteration(cfg, logger, iter)

        ## Saving model
        if (iter + 1) % cfg.checkpoints_interval == 0 or iter + 1 == cfg.num_epochs:
            if cfg.save_model == True:
                save_model_iteration(iter + 1, server.model, cfg)

        if np.isnan(test_loss) == True:
            break

    ## Notify the clients about the end of the learning
    global_broadcast_start = time.time()
    communicator.cleanup()
    communicator.broadcast_global_model(args={"done": True})
    global_broadcast_end = time.time()
    print(f"{communicator.current_time()} [APPFL] Server global_broadcast_time for ending: {global_broadcast_end - global_broadcast_start}, start at {global_broadcast_start}, end at {global_broadcast_end}", flush=True)

    ## Summary
    server.logging_summary(cfg, logger)


def run_client(
    cfg: DictConfig,
    comm: MPI.Comm,
    model: nn.Module,
    loss_fn: nn.Module,
    train_data: Dataset,
    test_data: Dataset = Dataset(),
    metric: Any = None,
):
    """
    run_client:
        Run PPFL simulation clients, each of which updates its own local parameters of model.

    args:
        cfg: the configuration for the FL experiment
        comm: MPI communicator
        model: neural network model to train
        loss_fn: loss function
        train_data: training data
        test_data: validation data
        metric: evaluation metric function
    """
    client_idx = comm.Get_rank() - 1
    communicator = MpiCommunicator(comm, Compressor(cfg) if cfg.enable_compression else None)

    ## log for clients
    output_filename = cfg.output_filename + "_client_%s" % (client_idx)
    outfile = client_log(cfg.output_dirname, output_filename)

    num_data = len(train_data[client_idx])
    communicator.gather(num_data, dest=0)
    weight = None
    weight = communicator.scatter(weight, source=0)

    batchsize = cfg.train_data_batch_size
    if cfg.batch_training == False:
        batchsize = len(train_data[client_idx])

    ## Run validation if test data is given or the configuration is enabled
    if cfg.validation == True and len(test_data) > 0:
        test_dataloader = DataLoader(
            test_data,
            num_workers=cfg.num_workers,
            batch_size=cfg.test_data_batch_size,
            shuffle=cfg.test_data_shuffle,
        )
    else:
        cfg.validation = False
        test_dataloader = None

    client = eval(cfg.fed.clientname)(
        client_idx,
        weight,
        copy.deepcopy(model),
        loss_fn,
        DataLoader(
            train_data[client_idx],
            num_workers=cfg.num_workers,
            batch_size=batchsize,
            shuffle=True,
            pin_memory=True,
        ),
        cfg,
        outfile,
        test_dataloader,
        metric,
        **cfg.fed.args,
    )

    while True:
        local_recv_start = time.time()
        model = communicator.recv_global_model_from_server(source=0)
        local_recv_end = time.time()
        print(f"{communicator.current_time()} [APPFL] Client {comm.Get_rank()} recv_local_model_from_server: {local_recv_end - local_recv_start}, start at {local_recv_start}, end at {local_recv_end}", flush=True)
        local_update_start = time.time()
        if isinstance(model, tuple):
            model, done = model[0], model[1]["done"]
        else:
            done = False
        if done:
            break
        client.model.load_state_dict(model)
        local_model = client.update()
        local_update_end = time.time()
        print(f"{communicator.current_time()} [APPFL] Client {comm.Get_rank()} client update: {local_update_end - local_update_start}, start at {local_update_start}, end at {local_update_end}", flush=True)
        local_send_start = time.time()
        communicator.send_local_model_to_server(local_model, dest=0)
        local_send_end = time.time()
        print(f"{communicator.current_time()} [APPFL] Client {comm.Get_rank()} send_local_model_to_server: {local_send_end - local_send_start}, start at {local_send_start}, end at {local_send_end}", flush=True)
    outfile.close()
