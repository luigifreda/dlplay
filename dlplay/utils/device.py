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
from typing import Optional, Union
import platform

# NOTE: MPS may introduce instabilities in the training code. 
#       With MPS, float64 is not supported and float32 is used instead, so we disable it by default
ENABLE_MPS = False
ENABLE_ACCELERATOR = True
if platform.system() == "Darwin":
    ENABLE_ACCELERATOR = ENABLE_MPS # MacOS does return "mps" as the accelerator type


def resolve_device(
    device: Optional[Union[str, torch.device]] | None = None,
) -> torch.device:
    if device is None:
        # Keep compatibility with user's accelerator helper if present
        if ENABLE_ACCELERATOR:
            try:  # type: ignore[attr-defined]
                if hasattr(torch, "accelerator") and torch.accelerator.is_available():
                    return torch.device(torch.accelerator.current_accelerator().type)
            except Exception:
                pass
        # CUDA
        if torch.cuda.is_available():
            return torch.device("cuda")
        # Apple MPS
        if ENABLE_MPS:
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(v, device) for v in obj)
    return obj


def empty_device_cache(device: Optional[Union[str, torch.device]] | None = None):
    if device is None:
        device = resolve_device()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    elif device.type == "mps":
        torch.mps.empty_cache()
        print(f"MPS memory: {torch.mps.memory_allocated() / 1024**3:.2f} GB")
    elif device.type == "cpu":
        pass
    else:
        raise ValueError(f"Unsupported device: {device.type}")
