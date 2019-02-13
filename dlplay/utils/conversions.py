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
import jax.numpy as jnp
from typing import Union, Optional, Any, TYPE_CHECKING
from dlplay.utils.types import ArrayLike
import PIL

try:
    import tensorflow as tf

    _TF_AVAILABLE = True
except Exception:
    tf = None  # type: ignore
    _TF_AVAILABLE = False

if TYPE_CHECKING:
    import tensorflow as tf  # for type checkers only


class JaxUtils:
    @staticmethod
    def jax_has_gpu() -> bool:
        try:
            return len(jax.devices("gpu")) > 0
        except Exception:
            return False

    @staticmethod
    def jax_device_from_str(device: Optional[str]) -> Optional[jax.Device]:
        if device is None:
            return None
        devs = jax.devices(device)  # "cpu" | "gpu" | "tpu"
        return devs[0] if devs else None

    @staticmethod
    def is_jax_array(x) -> bool:
        # Works across JAX versions
        try:
            return isinstance(x, jax.Array)
        except AttributeError:
            return hasattr(
                x, "__array_priority__"
            ) and x.__class__.__module__.startswith("jax")


class TensorflowUtils:
    # Helper to create/relocate on device without changing dtype
    @staticmethod
    def on_dev(make_tensor, dev: Optional[str] = None):
        if dev is None:
            return make_tensor()
        with tf.device(dev):  # type: ignore[attr-defined]
            return make_tensor()

    @staticmethod
    def normalize_tf_device(
        device: Optional[Union[str, "tf.DeviceSpec"]]
    ) -> Optional[str]:
        if not device:
            return None
        if hasattr(tf, "DeviceSpec") and isinstance(device, tf.DeviceSpec):  # type: ignore[attr-defined]
            return str(device)
        s = str(device).strip()
        low = s.lower()
        if low in {"cpu", "cpu:0", "/cpu:0"}:
            return "/CPU:0"
        if low in {"gpu", "cuda", "gpu:0", "/gpu:0"}:
            return "/GPU:0"
        if "tpu" in low:
            return "/TPU:0"
        # Assume caller passed a valid TensorFlow device string like "/GPU:1"
        return s


TORCH_DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
JAX_DEFAULT_DEVICE = "gpu" if JaxUtils.jax_has_gpu() else "cpu"


def to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert to numpy array (host)."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if JaxUtils.is_jax_array(x):
        return np.asarray(x)  # host copy
    # fall back for lists/tuples or array-likes
    try:
        return np.asarray(x)
    except Exception as e:
        raise ValueError(f"Unsupported type: {type(x)}") from e


def to_torch(
    x: ArrayLike, device: Union[str, torch.device] = TORCH_DEFAULT_DEVICE
) -> torch.Tensor:
    """Convert to torch tensor (moves to `device`)."""
    dev = torch.device(device)
    if isinstance(x, torch.Tensor):
        return x.to(dev, non_blocking=True)
    if isinstance(x, np.ndarray):
        # Handle MPS device compatibility - convert float64 to float32
        if dev.type == 'mps' and x.dtype == np.float64:
            x = x.astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)).to(dev, non_blocking=True)
    if JaxUtils.is_jax_array(x):
        # Handle MPS device compatibility for JAX arrays too
        arr = np.asarray(x)
        if dev.type == 'mps' and arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        return torch.from_numpy(arr).to(dev, non_blocking=True)
    # last resort
    arr = np.asarray(x)
    if dev.type == 'mps' and arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    return torch.from_numpy(arr).to(dev, non_blocking=True)


def to_jax(
    x: ArrayLike,
    device: Optional[Union[str, jax.Device]] = JAX_DEFAULT_DEVICE,
) -> jax.Array:
    """Convert to jax array (optionally place on a specific device)."""
    if JaxUtils.is_jax_array(x):
        arr = x
    elif isinstance(x, np.ndarray):
        arr = jnp.asarray(x)
    elif isinstance(x, torch.Tensor):
        arr = jnp.asarray(x.detach().cpu().numpy())
    else:
        arr = jnp.asarray(np.asarray(x))

    # place on device if specified
    target_dev = (
        JaxUtils.jax_device_from_str(device) if isinstance(device, str) else device
    )
    return jax.device_put(arr, device=target_dev)


def to_tensorflow(
    x: Any,
    device: Optional[Union[str, "tf.DeviceSpec"]] = None,
    dtype: Optional[Union[str, "tf.dtypes.DType"]] = None,
) -> "tf.Tensor":
    """
    Convert `x` to a TensorFlow EagerTensor and (optionally) place it on `device`.
    - Does NOT change dtype (lets TF infer from input).
    - Supports NumPy, PyTorch, JAX, Python scalars/lists, and TF tensors.
    - `device` may be 'cpu', 'gpu', '/CPU:0', '/GPU:0', or a tf.DeviceSpec. If None, uses TF default.
    - `dtype` may be 'float32', 'float64', 'int32', 'int64', etc. If None, uses TF default.
    """
    if not _TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is not available. Install tensorflow>=2.x to use to_tensorflow()."
        )

    if dtype is None:
        # infer from x
        dtype = x.dtype if isinstance(x, tf.Tensor) else type(x)

    dev = TensorflowUtils.normalize_tf_device(device)

    # Already TF tensor: optionally relocate via identity on target device
    if isinstance(x, tf.Tensor):
        if dev is None:
            return x
        return TensorflowUtils.on_dev(lambda: tf.identity(x, dtype=dtype))

    # NumPy
    if isinstance(x, np.ndarray):
        return TensorflowUtils.on_dev(lambda: tf.convert_to_tensor(x, dtype=dtype))

    # PyTorch
    if isinstance(x, torch.Tensor):
        return TensorflowUtils.on_dev(
            lambda: tf.convert_to_tensor(x.detach().cpu().numpy(), dtype=dtype)
        )

    # JAX
    if JaxUtils.is_jax_array(x):
        return TensorflowUtils.on_dev(
            lambda: tf.convert_to_tensor(np.asarray(x), dtype=dtype)
        )

    # Python scalars / sequences / other array-likes
    try:
        return TensorflowUtils.on_dev(
            lambda: tf.convert_to_tensor(np.asarray(x), dtype=dtype)
        )
    except Exception:
        # Last resort: let TF interpret directly
        return TensorflowUtils.on_dev(lambda: tf.convert_to_tensor(x, dtype=dtype))


# ======================================================
# Image conversions
# ======================================================


def to_numpy_uint_image(x: ArrayLike, scale_factor: float = 255.0) -> np.ndarray:
    # from normalized float in [0, 1] to uint8 in [0, 255]
    if isinstance(x, torch.Tensor):
        numpy_img = (x.squeeze().cpu().numpy() * scale_factor).astype("uint8")
        if numpy_img.ndim == 2:
            # set shape from (h,w) to (h,w,1)
            numpy_img = numpy_img[..., np.newaxis]
        elif numpy_img.ndim == 3:
            # set shape from (c,h,w) to (h,w,c)
            numpy_img = np.transpose(numpy_img, (1, 2, 0))
        return numpy_img
    if isinstance(x, np.ndarray):
        return (x * scale_factor).astype("uint8")
    if JaxUtils.is_jax_array(x):
        return (np.asarray(x) * scale_factor).astype("uint8")
    if isinstance(x, PIL.Image.Image):
        return np.array(x.convert("RGB"))
    raise ValueError(f"Unsupported type: {type(x)}")
