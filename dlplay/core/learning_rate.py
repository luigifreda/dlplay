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
from enum import Enum


class LearningRateSchedulerType(Enum):
    COSINE = "cosine"
    STEP = "step"
    PLATEAU = "plateau"
    NONE = "none"


def learning_rate_factory(
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    lr_scheduler_type: LearningRateSchedulerType,
    **kwargs,  # expected {step_size, gamma, patience, factor}
):
    """
    Factory function for creating learning rate schedulers.
    """
    # For StepLR
    step_size = kwargs.get("step_size", 10)
    gamma = kwargs.get("gamma", 0.1)

    # For ReduceLROnPlateau
    patience = kwargs.get("patience", 10)
    factor = kwargs.get("factor", 0.1)

    if lr_scheduler_type == LearningRateSchedulerType.COSINE:
        # CosineAnnealingLR:
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
        # T_max: maximum number of iterations.
        # eta_min: minimum learning rate.
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif lr_scheduler_type == LearningRateSchedulerType.STEP:
        # StepLR:
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
        # step_size: number of epochs to wait before updating the learning rate.
        # gamma: factor by which the learning rate is multiplied.
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif lr_scheduler_type == LearningRateSchedulerType.PLATEAU:
        # ReduceLROnPlateau:
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
        # mode: "min" or "max".
        # factor: factor by which the learning rate is multiplied.
        # patience: number of epochs to wait before reducing the learning rate.
        # verbose: if True, prints a message to the console when the learning rate is reduced.
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=factor, patience=patience, verbose=True
        )
    elif lr_scheduler_type == LearningRateSchedulerType.NONE:
        return None
