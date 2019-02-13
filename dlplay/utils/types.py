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
from typing import Union, Tuple, Dict, List
import numpy as np
import torch
import jax
import tensorflow as tf

# Type hints for better code clarity
ArrayLike = Union[np.ndarray, "torch.Tensor", "jax.Array", "tf.Tensor"]

OptimizationPathData = Tuple[
    np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]
]  # (path, objective_function_values, grad_norms, metrics)


TensorOrList = Union[torch.Tensor, List[torch.Tensor]]
