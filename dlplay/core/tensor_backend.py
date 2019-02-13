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
import numpy as np
import torch

import jax
from jax import config

JAX_ENABLE_X64 = False
if JAX_ENABLE_X64:
    config.update("jax_enable_x64", True)
import jax.numpy as jnp

# NEW: Optional TensorFlow import (kept at top for typing/static tools)
try:
    import tensorflow as tf

    _TF_AVAILABLE = True
except Exception:
    _TF_AVAILABLE = False

from typing import Literal, Tuple
from dlplay.utils.types import ArrayLike


class TensorBackend:
    """
    A unified interface to basic numerical operations across NumPy, PyTorch, JAX, and TensorFlow.
    This class is used to allow the same code to work with different array backends.
    """

    def __init__(
        self,
        array_type: Literal["numpy", "torch", "jax", "tensorflow", "tf"] | None = None,
        device: str = None,
        verbose: bool = False,
    ):
        self.array_type = array_type
        self.verbose = verbose
        self.device = device if array_type is not None else None
        if array_type is not None:
            if device is None:
                raise ValueError("device cannot be None")
            self.init(array_type, device)

    def init(
        self,
        array_type: Literal["numpy", "torch", "jax", "tensorflow", "tf"],
        device: str,
    ):
        if array_type is None:
            raise ValueError("array_type cannot be None")
        if device is None:
            raise ValueError("device cannot be None")

        # normalize TF alias
        if array_type == "tf":
            array_type = "tensorflow"

        self.array_type = array_type
        self.device = device

        if self.verbose:
            print(
                f"TensorBackend initialized with array_type: {self.array_type}, device: {self.device}"
            )

        # ============================== numpy ===============================
        if self.array_type == "numpy":
            self._array = np.array
            self._ndim = np.ndim
            self._print_info = lambda x, msg: print(
                f"{msg} | Numpy array: shape {x.shape}, type: {x.dtype}"
            )
            self._abs = np.abs
            self._exp = np.exp
            self._clip = np.clip
            self._mean = lambda x, axis=None, keepdims=False: np.mean(
                x, axis=axis, keepdims=keepdims
            )
            self._log = np.log
            self._norm = np.linalg.norm
            self._random_normal = lambda shape: np.random.randn(*shape)
            self._unsqueeze = lambda x: x[np.newaxis, :]
            self._squeeze = lambda x: x.squeeze()
            self._flatten = lambda x: x.flatten()
            self._to_numpy = lambda x: x
            self._copy_and_convert_to_float = lambda x, dtype: x.copy().astype(dtype)
            self._zeros_like = lambda x: np.zeros_like(x)
            self._one_hot = lambda y, num_classes: np.eye(num_classes)[y]
            self._homogeneous = lambda x: np.concatenate(
                [x, np.ones((x.shape[0], 1))], axis=1
            )
            self._max = np.max
            self._maximum = lambda x, y: np.maximum(x, y)
            self._min = np.min
            self._minimum = lambda x, y: np.minimum(x, y)
            self._argmax = lambda x, axis=None: np.argmax(x, axis=axis)
            self._unique = lambda x: np.unique(x)
            self._transpose = lambda x: x.T
            self._sum = lambda x, axis=None, keepdims=False: np.sum(
                x, axis=axis, keepdims=keepdims
            )
            self._broadcast = lambda x, axis: np.broadcast_to(
                x[None, :], (axis, x.shape[0])
            )
            self._einsum = np.einsum
            self._cast = lambda x, dtype: x.astype(dtype)
            self._scalar = lambda x, dtype: x  # no need to cast to scalar

        # ============================== torch ===============================
        elif self.array_type == "torch":
            if (
                self.device == "cuda" or self.device == "gpu"
            ) and torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif self.device == "mps" and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

            if self.verbose:
                print(f"Torch device: {self.device}")

            def torch_array(data):
                if isinstance(data, list) and len(data) > 0:
                    converted_data = []
                    for item in data:
                        if isinstance(item, torch.Tensor):
                            converted_data.append(item)
                        elif isinstance(item, np.ndarray):
                            converted_data.append(
                                torch.from_numpy(item).float().to(self.device)
                            )
                        else:
                            converted_data.append(
                                torch.tensor(item, dtype=torch.float32).to(self.device)
                            )
                    return torch.stack(converted_data).to(self.device)
                else:
                    return torch.tensor(data, dtype=torch.float32).to(self.device)

            self._array = torch_array
            self._ndim = lambda x: x.ndim
            self._print_info = lambda x, msg: print(
                f"{msg} | Torch array: shape {x.shape}, type: {x.dtype}, device: {x.device}"
            )
            self._abs = torch.abs
            
            # MPS-compatible exp function with clipping
            def _exp_torch(x):
                if self.device.type == 'mps':
                    # Clip values to prevent overflow on MPS
                    x = torch.clamp(x, min=-88, max=88)  # exp(-88) ≈ 1e-39, exp(88) ≈ 1e38
                return torch.exp(x)
            self._exp = _exp_torch
            
            self._clip = torch.clamp
            self._mean = lambda x, axis=None, keepdims=False: torch.mean(
                x, dim=axis, keepdim=keepdims
            )
            
            # MPS-compatible log function with clipping
            def _log_torch(x):
                if self.device.type == 'mps':
                    # Add small epsilon to prevent log(0) and ensure numerical stability
                    x = torch.clamp(x, min=1e-8)
                return torch.log(x)
            self._log = _log_torch
            
            self._norm = torch.norm
            self._random_normal = lambda shape: torch.randn(
                shape, dtype=torch.float32
            ).to(self.device)
            self._unsqueeze = lambda x: x.unsqueeze(0)
            self._squeeze = lambda x: x.squeeze()
            self._flatten = lambda x: x.flatten()

            def _to_numpy_torch(x):
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().numpy()
                elif isinstance(x, (list, tuple)) and all(
                    isinstance(t, torch.Tensor) for t in x
                ):
                    return np.array([t.detach().cpu().numpy() for t in x])
                else:
                    raise TypeError(f"Unsupported type for to_numpy: {type(x)}")

            self._to_numpy = _to_numpy_torch

            def _copy_and_convert_to_float_torch(x, dtype):
                # Convert Python float to PyTorch dtype
                if dtype == float:
                    dtype = torch.float64
                
                # Handle MPS devices - they don't support float64
                # Check both the tensor's current device and target device
                is_mps_tensor = hasattr(x, 'device') and hasattr(x.device, 'type') and x.device.type == 'mps'
                is_mps_target = hasattr(self.device, 'type') and self.device.type == 'mps'
                
                if (is_mps_tensor or is_mps_target) and dtype == torch.float64:
                    dtype = torch.float32
                
                return x.clone().detach().to(dtype).to(self.device)

            self._copy_and_convert_to_float = _copy_and_convert_to_float_torch
            self._zeros_like = lambda x: x.new_zeros(x.shape).to(self.device)
            self._one_hot = lambda y, num_classes: torch.eye(
                num_classes, device=self.device
            )[y]
            self._homogeneous = lambda x: torch.cat(
                [x, torch.ones((x.shape[0], 1), device=self.device)], dim=1
            ).to(self.device)
            self._max = lambda x, axis=None, keepdims=False: (
                torch.max(x, dim=axis, keepdim=keepdims).values
                if axis is not None
                else torch.max(x)
            )
            self._maximum = lambda x, y: torch.maximum(x, y)
            self._min = lambda x, axis=None, keepdims=False: (
                torch.min(x, dim=axis, keepdim=keepdims).values
                if axis is not None
                else torch.min(x)
            )
            self._minimum = lambda x, y: torch.minimum(x, y)
            self._argmax = lambda x, axis=None: x.argmax(dim=axis)
            self._unique = lambda x: torch.unique(x)
            self._transpose = lambda x: x.T
            self._sum = lambda x, axis=None, keepdims=False: torch.sum(
                x, dim=axis, keepdim=keepdims
            )
            self._broadcast = lambda x, axis: torch.broadcast_to(
                x[None, :], (axis, x.shape[0])
            )
            self._einsum = torch.einsum
            def _cast_torch(x, dtype):
                """Cast tensor to dtype with MPS compatibility."""
                # Convert Python float to PyTorch dtype
                if dtype == float:
                    dtype = torch.float64
                # Handle MPS devices - they don't support float64
                if self.device.type == 'mps' and dtype == torch.float64:
                    dtype = torch.float32
                return x.to(dtype)
            self._cast = _cast_torch
            def _scalar_torch(x, dtype):
                if isinstance(x, torch.Tensor):
                    return x.detach().clone().to(dtype=dtype, device=self.device)
                else:
                    return torch.tensor(x, dtype=dtype, device=self.device)
            self._scalar = _scalar_torch

        # ============================== jax ===============================
        elif self.array_type == "jax":
            self._array = jnp.array
            self._ndim = lambda x: x.ndim
            self._print_info = lambda x, msg: print(
                f"{msg} | Jax array: shape {x.shape}, type: {x.dtype}, device: {x.device}"
            )
            self._abs = jnp.abs
            self._exp = jnp.exp
            self._clip = jnp.clip
            self._mean = lambda x, axis=None, keepdims=False: jnp.mean(
                x, axis=axis, keepdims=keepdims
            )
            self._log = jnp.log
            self._norm = jnp.linalg.norm
            self._key = jax.random.PRNGKey(0)
            self._unsqueeze = lambda x: x[jnp.newaxis, :]
            self._squeeze = lambda x: x.squeeze()
            self._flatten = lambda x: x.flatten()
            self._to_numpy = lambda x: np.array(x)
            self._copy_and_convert_to_float = lambda x, dtype: jnp.array(
                x,
                dtype=jnp.float64 if JAX_ENABLE_X64 and dtype == float else jnp.float32,
            )
            self._zeros_like = lambda x: jnp.zeros_like(x)
            self._one_hot = lambda y, num_classes: jnp.eye(num_classes)[y]
            self._homogeneous = lambda x: jnp.concatenate(
                [x, jnp.ones((x.shape[0], 1))], axis=1
            )
            self._max = jnp.max
            self._maximum = lambda x, y: jnp.maximum(x, y)
            self._min = jnp.min
            self._minimum = lambda x, y: jnp.minimum(x, y)
            self._argmax = lambda x, axis=None: x.argmax(axis=axis)
            self._unique = lambda x: jnp.unique(x)
            self._transpose = lambda x: x.T
            self._sum = lambda x, axis=None, keepdims=False: jnp.sum(
                x, axis=axis, keepdims=keepdims
            )
            self._broadcast = lambda x, axis: jnp.broadcast_to(
                x[None, :], (axis, x.shape[0])
            )
            self._einsum = jnp.einsum
            self._cast = lambda x, dtype: jnp.array(x, dtype=dtype)
            self._scalar = lambda x, dtype: x  # no need to cast to scalar

            def jax_random_normal(shape):
                self._key, subkey = jax.random.split(self._key)
                return jax.random.normal(subkey, shape)

            self._random_normal = jax_random_normal

        # ============================== tensorflow ===============================
        elif self.array_type == "tensorflow":
            if not _TF_AVAILABLE:
                raise ImportError(
                    "TensorFlow is not available. Please install tensorflow>=2.x to use this backend."
                )

            tf.experimental.numpy.experimental_enable_numpy_behavior()

            # Resolve a TF device string
            # Accepts "cpu", "gpu", "/CPU:0", "/GPU:0"
            def _normalize_tf_device(dev_str: str) -> str:
                s = dev_str.lower()
                if "gpu" in s:
                    return "/GPU:0"
                if "tpu" in s:
                    return "/TPU:0"
                return "/CPU:0"

            self._tf_device_str = _normalize_tf_device(self.device)

            if self.verbose:
                print(f"TensorFlow device target: {self._tf_device_str}")

            def tf_array(data):
                # convert and place on device
                with tf.device(self._tf_device_str):
                    if isinstance(data, list) and len(data) > 0:
                        converted = []
                        for item in data:
                            if isinstance(item, tf.Tensor):
                                converted.append(tf.cast(item, tf.float32))
                            elif isinstance(item, np.ndarray):
                                converted.append(
                                    tf.convert_to_tensor(item, dtype=tf.float32)
                                )
                            else:
                                converted.append(
                                    tf.convert_to_tensor(item, dtype=tf.float32)
                                )
                        return tf.stack(converted, axis=0)
                    else:
                        return tf.convert_to_tensor(data, dtype=tf.float32)

            self._array = tf_array
            self._ndim = lambda x: x.ndim if hasattr(x, "ndim") else len(x.shape)
            self._print_info = lambda x, msg: print(
                f"{msg} | TF tensor: shape {tuple(x.shape)}, type: {x.dtype.name}, device: {x.device}"
            )
            self._abs = tf.math.abs
            self._exp = tf.math.exp
            self._clip = lambda x, min_val, max_val: tf.clip_by_value(
                x, min_val, max_val
            )
            self._mean = lambda x, axis=None, keepdims=False: tf.reduce_mean(
                x, axis=axis, keepdims=keepdims
            )
            self._log = tf.math.log
            self._norm = lambda x: tf.norm(x)
            self._random_normal = lambda shape: tf.random.normal(
                shape, dtype=tf.float32
            )
            self._unsqueeze = lambda x: tf.expand_dims(x, axis=0)
            self._squeeze = lambda x: tf.squeeze(x)
            self._flatten = lambda x: tf.reshape(x, [-1])
            self._to_numpy = lambda x: (
                x.numpy() if isinstance(x, tf.Tensor) else np.array(x)
            )
            self._copy_and_convert_to_float = lambda x, dtype: tf.cast(
                tf.identity(x), dtype
            )
            self._zeros_like = tf.zeros_like
            self._one_hot = lambda y, num_classes: tf.one_hot(
                self._cast(y, int), int(num_classes)
            )
            self._homogeneous = lambda x: tf.concat(
                [x, tf.ones((tf.shape(x)[0], 1), dtype=x.dtype)], axis=1
            )
            # Reductions
            self._max = lambda x, axis=None, keepdims=False: tf.reduce_max(
                x, axis=axis, keepdims=keepdims
            )
            self._maximum = tf.maximum
            self._min = lambda x, axis=None, keepdims=False: tf.reduce_min(
                x, axis=axis, keepdims=keepdims
            )
            self._minimum = tf.minimum
            self._argmax = lambda x, axis=None: tf.argmax(x, axis=axis)
            # tf.unique is 1-D only; flatten first
            self._unique = lambda x: tf.unique(tf.reshape(x, [-1])).y
            self._transpose = lambda x: tf.transpose(x)
            self._sum = lambda x, axis=None, keepdims=False: tf.reduce_sum(
                x, axis=axis, keepdims=keepdims
            )
            self._broadcast = lambda x, axis: tf.broadcast_to(
                tf.expand_dims(x, 0), [axis, x.shape[0]]
            )
            self._einsum = tf.einsum

            def _dtype_tensorflow(dtype):
                if dtype == int:
                    return "int32"
                else:
                    return dtype

            self._cast = lambda x, dtype: tf.cast(x, _dtype_tensorflow(dtype))
            self._scalar = lambda x, dtype: tf.convert_to_tensor(x, dtype=dtype)

        else:
            raise ValueError(f"Unsupported array type: {self.array_type}")

    # --------------------------- unified API ---------------------------
    def array(self, data):
        return self._array(data)

    def ndim(self, x):
        return self._ndim(x)

    def abs(self, x):
        return self._abs(x)

    def exp(self, x):
        return self._exp(x)

    def clip(self, x, min_val, max_val):
        return self._clip(x, min_val, max_val)

    def mean(self, x, axis=None, keepdims=False):
        return self._mean(x, axis=axis, keepdims=keepdims)

    def log(self, x):
        return self._log(x)

    def norm(self, x):
        return self._norm(x)

    def random_normal(self, shape: Tuple[int, ...]):
        return self._random_normal(shape)

    def unsqueeze(self, x):
        return self._unsqueeze(x)

    def squeeze(self, x):
        return self._squeeze(x)

    def flatten(self, x):
        return self._flatten(x)

    def to_numpy(self, x):
        return self._to_numpy(x)

    def copy_and_convert_to_float(self, x, dtype=float):
        return self._copy_and_convert_to_float(x, dtype)

    def zeros_like(self, x):
        return self._zeros_like(x)

    def one_hot(self, y, num_classes):
        return self._one_hot(y, num_classes)

    def homogeneous(self, x):
        return self._homogeneous(x)

    def max(self, x, axis=None, keepdims=False):
        return self._max(x, axis=axis, keepdims=keepdims)

    def maximum(self, x, y):
        return self._maximum(x, y)

    def min(self, x, axis=None, keepdims=False):
        return self._min(x, axis=axis, keepdims=keepdims)

    def minimum(self, x, y):
        return self._minimum(x, y)

    def argmax(self, x, axis=None):
        return self._argmax(x, axis=axis)

    def unique(self, x):
        return self._unique(x)

    def transpose(self, x):
        return self._transpose(x)

    def sum(self, x, axis=None, keepdims=False):
        return self._sum(x, axis=axis, keepdims=keepdims)

    def broadcast(self, x, axis):
        return self._broadcast(x, axis)

    def einsum(self, equation, *operands):
        return self._einsum(equation, *operands)

    def cast(self, x, dtype):
        return self._cast(x, dtype)

    def scalar(self, x, dtype):
        return self._scalar(x, dtype)

    def print_info(self, x, msg: str = ""):
        try:
            self._print_info(x, msg)
        except Exception as e:
            print(f"{msg} | {x}, type: {type(x)}")

    # ------------------------ helpers / detection ------------------------
    @staticmethod
    def _is_jax_array(x) -> bool:
        try:
            return isinstance(x, jax.Array)
        except AttributeError:
            return hasattr(
                x, "__array_priority__"
            ) and x.__class__.__module__.startswith("jax")

    @staticmethod
    def get_array_type(x: ArrayLike) -> str:
        """Get the type of the array-like object."""
        device = "cpu"
        if isinstance(x, np.ndarray):
            return "numpy", device
        if isinstance(x, torch.Tensor):
            if x.is_cuda:
                device = "cuda"
            elif hasattr(x.device, 'type') and x.device.type == 'mps':
                device = "mps"
            else:
                device = "cpu"
            return "torch", device
        if TensorBackend._is_jax_array(x):
            device = (
                "gpu"
                if getattr(getattr(x, "device", None), "platform", "") == "gpu"
                else "cpu"
            )
            return "jax", device
        # NEW: TensorFlow detection
        if _TF_AVAILABLE and isinstance(x, tf.Tensor):
            dev = x.device or ""
            device = (
                "gpu"
                if "GPU" in dev.upper()
                else ("tpu" if "TPU" in dev.upper() else "cpu")
            )
            return "tensorflow", device
        raise ValueError(f"Unsupported type: {type(x)}")
