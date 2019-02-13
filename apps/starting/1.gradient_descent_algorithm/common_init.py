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
from typing import List

import os
import numpy as np
import torch
import jax
import jax.numpy as jnp

from dlplay.utils.types import ArrayLike
from dlplay.core.tensor_backend import JAX_ENABLE_X64

from typing import List, Callable, Tuple

DEFAULT_START_POINTS = [(1, 1), (3, 3), (5, 5), (7, 7), (9, 9)]


X_RANGE = (0, 10)
Y_RANGE = (0, 10)
GRID_RESOLUTION = 100


def init_start_points(
    framework: str = "numpy",
    start_points: List[ArrayLike] = None,
    device: str = None,
) -> List[ArrayLike]:
    """
    Get starting points for optimization in the specified framework format.

    Args:
        framework: 'numpy', 'pytorch', or 'jax'

    Returns:
        List of starting points in the appropriate format
    """
    if framework == "numpy":
        return [np.array(x, dtype=np.float64) for x in start_points]
    elif framework == "torch":
        # Use float32 for MPS devices, float64 for others
        if device and hasattr(device, 'type') and device.type == 'mps':
            dtype = torch.float32
        else:
            dtype = torch.float64
        
        return [
            torch.tensor(x, dtype=dtype, device=device) for x in start_points
        ]
    elif framework == "jax":
        try:
            return [
                jnp.array(x, dtype=jnp.float64 if JAX_ENABLE_X64 else jnp.float32)
                for x in start_points
            ]
        except Exception as e:
            print(f"JAX initialization failed: {e}")
            print("Falling back to CPU-only mode...")
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
            return [jnp.array(x, dtype=jnp.float64) for x in start_points]
    else:
        raise ValueError(f"Unsupported framework: {framework}")


def create_grid_points(
    x_range: Tuple[float, float] = X_RANGE,
    y_range: Tuple[float, float] = Y_RANGE,
    grid_resolution: int = GRID_RESOLUTION,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a grid of points for visualization.

    Returns:
        Tuple of (xx, yy, zz) where xx, yy are meshgrid coordinates
        and zz is the function values (to be computed by framework-specific code)
    """
    xgrid = np.linspace(x_range[0], x_range[1], grid_resolution)
    ygrid = np.linspace(y_range[0], y_range[1], grid_resolution)
    xx, yy = np.meshgrid(xgrid, ygrid)
    return xx, yy, None  # zz will be computed by framework-specific code
