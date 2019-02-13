# ******************************************************************************
# This file is part of dlplay
# 
# Copyright (C) Luigi Freda <luigi dot freda at gmail dot com>
# 
# dlplay is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# dlplay is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with dlplay. If not, see <http://www.gnu.org/licenses/>.
# ******************************************************************************
from multiprocessing import current_process
import torch
from typing import Callable, Optional, Union, Dict, Any

from enum import Enum

from dlplay.models.model_helpers import dump_optimized_model_layers
from dlplay.core.learning_rate import learning_rate_factory, LearningRateSchedulerType
from dlplay.utils.device import resolve_device
from dlplay.core.metric_logger import MetricLogger
from dlplay.core.smoothed_value import SmoothedValue
from dlplay.core.evaluation import Evaluator, TaskType
from dlplay.utils.device import resolve_device

from abc import ABC, abstractmethod


# -------------------------------------------------------------------------
# Simple training functions
# -------------------------------------------------------------------------


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    device: str,
    print_freq: int,
):
    """
    (Simple) Train the model for one epoch.
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % print_freq == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_one_epoch(
    model: torch.nn.Module,
    loss_fn: Callable,
    dataloader: torch.utils.data.DataLoader,
    device: str,
):
    """
    (Simple) Test the model on one epoch.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def train_model_simple(
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 10,
    device: str | None = None,
    print_freq: int = 10,
):
    """
    (Simple) Train the model for a number of epochs.

    Args:
        train_dataloader: The training data loader.
        test_dataloader: The test data loader.
        model: The model to train.
        loss_fn: The loss function.
        optimizer: The optimizer.
        epochs: The number of epochs to train for.
        device: The device to train on.
    """
    device = resolve_device(device)

    for t in range(num_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_one_epoch(
            dataloader=train_dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            print_freq=print_freq,
        )
        test_one_epoch(
            dataloader=test_dataloader,
            model=model,
            loss_fn=loss_fn,
            device=device,
        )
    print("Training Done!")


# -------------------------------------------------------------------------
# Train a model using SGD optimizer
# -------------------------------------------------------------------------


def train_model_SGD(
    model: torch.nn.Module,
    data_loader_train: torch.utils.data.DataLoader,
    data_loader_test: torch.utils.data.DataLoader,
    train_one_epoch_fn: Callable,
    post_epoch_callback: Callable = None,
    device: str = None,
    num_epochs: int = 2,
    optimizer_dict: dict = {
        "lr": 0.005,
        "momentum": 0.9,
        "weight_decay": 0.0005,
    },
    lr_scheduler_dict: dict = {
        "step_size": 3,
        "gamma": 0.1,
    },
    lr_scheduler_type: LearningRateSchedulerType = LearningRateSchedulerType.COSINE,
    print_freq: int = 10,  # print every 10 iterations,
):
    """
    Train a model using SGD optimizer.
    """

    device = resolve_device(device)

    # Init optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    lr = optimizer_dict.get("lr", 0.005)
    momentum = optimizer_dict.get("momentum", 0.9)
    weight_decay = optimizer_dict.get("weight_decay", 0.0005)
    optimizer = torch.optim.SGD(
        params, lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    # Check which layers are being optimized
    dump_optimized_model_layers(model)

    # Learning rate scheduler
    lr_scheduler_type = lr_scheduler_dict.get("type", LearningRateSchedulerType.COSINE)
    step_size = lr_scheduler_dict.get("step_size", 3)
    gamma = lr_scheduler_dict.get("gamma", 0.1)
    lr_scheduler = learning_rate_factory(
        optimizer, num_epochs, lr_scheduler_type, step_size=step_size, gamma=gamma
    )

    # Train the model

    for epoch in range(num_epochs):
        # train for one epoch, printing every print_freq iterations
        train_one_epoch_fn(
            model=model,
            optimizer=optimizer,
            data_loader=data_loader_train,
            device=device,
            epoch=epoch,
            print_freq=print_freq,
        )

        if post_epoch_callback:
            post_epoch_callback(
                model=model,
                data_loader_test=data_loader_test,
                device=device,
            )

        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

    print("Training is done!")


# -------------------------------------------------------------------------
# Train a model using AdamW optimizer
# -------------------------------------------------------------------------


def train_model_AdamW(
    model: torch.nn.Module,
    data_loader_train: torch.utils.data.DataLoader,
    data_loader_test: torch.utils.data.DataLoader,
    train_one_epoch_fn: Callable,
    post_epoch_callback: Callable = None,
    device: str = None,
    num_epochs: int = 2,
    optimizer_dict: dict = {
        "lr": 0.0001,
        "weight_decay": 0.01,
    },
    lr_scheduler_dict: dict = {
        "step_size": 3,
        "gamma": 0.1,
    },
    lr_scheduler_type: LearningRateSchedulerType = LearningRateSchedulerType.COSINE,
    print_freq: int = 10,  # print every 10 iterations
):
    """
    Train a model using AdamW optimizer.
    """

    device = device or (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )

    # Init optimizer with better hyperparameters
    params = [p for p in model.parameters() if p.requires_grad]
    lr = optimizer_dict.get("lr", 0.0001)
    weight_decay = optimizer_dict.get("weight_decay", 0.01)
    optimizer = torch.optim.AdamW(
        params, lr=lr, weight_decay=weight_decay
    )  # Lower LR, better optimizer

    # Check which layers are being optimized
    dump_optimized_model_layers(model)

    # Learning rate scheduler
    lr_scheduler_type = lr_scheduler_dict.get("type", LearningRateSchedulerType.COSINE)
    step_size = lr_scheduler_dict.get("step_size", 3)
    gamma = lr_scheduler_dict.get("gamma", 0.1)
    lr_scheduler = learning_rate_factory(
        optimizer, num_epochs, lr_scheduler_type, step_size=step_size, gamma=gamma
    )

    # Train the model

    for epoch in range(num_epochs):
        # train for one epoch, printing every print_freq iterations
        train_one_epoch_fn(
            model=model,
            optimizer=optimizer,
            data_loader=data_loader_train,
            device=device,
            epoch=epoch,
            print_freq=print_freq,
        )

        if post_epoch_callback:
            post_epoch_callback(
                model=model,
                data_loader_test=data_loader_test,
                device=device,
            )

        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

    print("Training is done!")
