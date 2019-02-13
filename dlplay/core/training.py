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
import torch
from typing import Callable, Optional, Union, Dict, Any

from enum import Enum

from dlplay.models.model_helpers import dump_optimized_model_layers
from dlplay.core.learning_rate import learning_rate_factory, LearningRateSchedulerType
from dlplay.utils.device import resolve_device
from dlplay.core.metric_logger import MetricLogger
from dlplay.core.smoothed_value import SmoothedValue
from dlplay.core.evaluation import Evaluator, TaskType

from abc import ABC, abstractmethod
import traceback


class OptimizerType(Enum):
    SGD = "sgd"
    SGD_MOMENTUM = "sgd_momentum"
    ADAM = "adam"
    ADAMW = "adamw"


def optimizer_factory(
    model: torch.nn.Module,
    optimizer_type: OptimizerType,
    **kwargs,  # expected {lr, momentum, weight_decay}
):
    lr = kwargs.get("lr", 0.005)
    momentum = kwargs.get("momentum", 0.9)
    weight_decay = kwargs.get("weight_decay", 0.0005)

    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_type == OptimizerType.SGD:
        return torch.optim.SGD(params, lr=lr)
    elif optimizer_type == OptimizerType.SGD_MOMENTUM:
        return torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer_type == OptimizerType.ADAM:
        return torch.optim.Adam(params, lr=lr)
    elif optimizer_type == OptimizerType.ADAMW:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


class TrainerBase(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        dataloader_test: torch.utils.data.DataLoader,
        optimizer_type: OptimizerType,
        lr_scheduler_type: LearningRateSchedulerType,
        loss_fn: Callable,
        task_type: TaskType,
        device: str,
        num_epochs: int,
        print_freq: int,  # print every print_freq batches
        evaluate_freq: int = 1,  # evaluate every evaluate_freq epochs
        **kwargs,  # for optimizer and lr_scheduler, expected:
        # optimizer: {lr, momentum, weight_decay}
        # lr_scheduler: {step_size, gamma, patience, factor}
    ):
        self.model = model
        self.device = resolve_device(device)
        self.num_epochs = num_epochs
        self.print_freq = print_freq
        self.evaluate_freq = evaluate_freq
        self.dataloader = dataloader
        self.dataloader_test = dataloader_test
        self.loss_fn = loss_fn
        self.task_type = task_type
        self.optimizer_type = optimizer_type
        self.lr_scheduler_type = lr_scheduler_type
        # Set up the optimizer
        optimizer_kwargs = kwargs.get(
            "optimizer", {}
        )  # expected {lr, momentum, weight_decay}
        self.optimizer = optimizer_factory(
            self.model, self.optimizer_type, **optimizer_kwargs
        )
        # Set up the learning rate scheduler
        lr_scheduler_kwargs = kwargs.get(
            "lr_scheduler", {}
        )  # expected {step_size, gamma, patience, factor}
        self.lr_scheduler = learning_rate_factory(
            self.optimizer,
            self.num_epochs,
            self.lr_scheduler_type,
            **lr_scheduler_kwargs,
        )
        # Set up the metric logger and the metrics to be logged
        self.metric_logger = MetricLogger(delimiter="  ")
        self.metric_logger.add_meter(
            "lr", SmoothedValue(window_size=1, fmt="{value:.6f}")
        )
        self.metric_logger.add_meter(
            "current", SmoothedValue(window_size=1, fmt="{value:>5d}")
        )
        self.metric_logger.add_meter(
            "size", SmoothedValue(window_size=1, fmt="{value:>5d}")
        )
        # Set up the callbacks
        self.current_epoch = 0
        self.evaluate_callback = None
        self.post_epoch_callback = None  # e.g., save the model

    @abstractmethod
    def train_one_epoch(self):
        """
        Train the model for one epoch.
        """
        pass

    # Generic training loop
    def train_model(self):
        """
        Train the model for a number of epochs.
        """

        # Check which layers are being optimized
        dump_optimized_model_layers(self.model)

        # Train the model

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # train for one epoch, printing every print_freq batches
            try:
                self.train_one_epoch()
            except Exception as e:
                print(f"Error in train_one_epoch: {e}")
                print(traceback.format_exc())
                raise e

            # evaluate the model
            if self.evaluate_callback and self.current_epoch % self.evaluate_freq == 0:
                try:
                    self.evaluate_callback()
                except Exception as e:
                    print(f"Error in evaluate callback: {e}")
                    print(traceback.format_exc())
                    raise e

            # post-epoch callback (e.g., save the model)
            if self.post_epoch_callback:
                try:
                    self.post_epoch_callback()
                except Exception as e:
                    print(f"Error in post-epoch callback: {e}")
                    print(traceback.format_exc())
                    raise e

            # update the learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        print("Training is done!")


class SimpleTrainer(TrainerBase):
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        dataloader_test: torch.utils.data.DataLoader,
        optimizer_type: OptimizerType,
        lr_scheduler_type: LearningRateSchedulerType,
        loss_fn: Callable,
        task_type: TaskType,
        device: str,
        num_epochs: int,
        print_freq: int,  # print every print_freq batches
        evaluate_freq: int = 1,  # evaluate every evaluate_freq epochs
        **kwargs,  # for optimizer and lr_scheduler, expected:
        # optimizer: {lr, momentum, weight_decay}
        # lr_scheduler: {step_size, gamma, patience, factor}
    ):
        super().__init__(
            model,
            dataloader,
            dataloader_test,
            optimizer_type,
            lr_scheduler_type,
            loss_fn,
            task_type,
            device,
            num_epochs,
            print_freq,
            evaluate_freq,
            **kwargs,
        )
        self.evaluator = Evaluator(
            self.model, self.dataloader_test, self.device, self.loss_fn, self.task_type
        )
        self.evaluate_callback = self.evaluator.evaluate

    def train_one_epoch(self):
        """
        (Simple) Train the model for one epoch.
        """
        size = len(self.dataloader.dataset)
        self.model.train()

        header = "Epoch: [{}]".format(self.current_epoch)
        for batch, (X, y) in self.metric_logger.log_every(
            self.dataloader, self.print_freq, header=header
        ):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            with torch.amp.autocast("cuda"):
                pred = self.model(X)
                loss = self.loss_fn(pred, y)

            if not torch.isfinite(loss):
                print(f"Loss is {loss}, stopping training")
                raise ValueError("Loss is not finite")

            # Backpropagation
            loss.backward()
            self.optimizer.step()  # update the model parameters
            self.optimizer.zero_grad()  # zero the gradients

            # update the metric logger
            self.metric_logger.update(loss=loss.item())
            self.metric_logger.update(
                lr=(
                    self.lr_scheduler.get_last_lr()[0]
                    if self.lr_scheduler is not None
                    else getattr(self.optimizer, "defaults", {}).get(
                        "lr", 0.001
                    )  # Safe fallback
                )
            )
            self.metric_logger.update(current=(batch + 1) * len(X), size=size)
