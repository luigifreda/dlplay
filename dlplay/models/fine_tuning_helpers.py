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
import torch.nn as nn


# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------
def set_trainable(module: torch.nn.Module, requires_grad: bool):
    """Enable/disable gradient updates for all parameters in a module."""
    for p in module.parameters():
        p.requires_grad = requires_grad


def set_trainable_only_head(
    model: torch.nn.Module, head_modules: torch.nn.Module | list[torch.nn.Module]
):
    """
    Freeze all parameters in the model, except for the given head modules.
    Useful when you want to keep the backbone fixed and only train the heads.
    """
    set_trainable(model, False)  # freeze entire model
    if isinstance(head_modules, (list, tuple)):
        for m in head_modules:
            set_trainable(m, True)  # unfreeze head(s)
    else:
        set_trainable(head_modules, True)
