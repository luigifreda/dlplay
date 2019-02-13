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
import tensorflow as tf
import numpy as np

if __name__ == "__main__":

    # Create tensors
    a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)  # Constant tensor
    b = tf.ones((2, 2))  # All ones
    c = tf.zeros((2, 2))  # All zeros
    d = tf.random.uniform((2, 2), minval=0, maxval=10)  # Random values
    e = tf.convert_to_tensor(np.array([[1, 2], [3, 4]]))  # From NumPy array
    f = tf.eye(2)

    print("a:\n", a.numpy())  # Convert to NumPy for printing

    # Basic arithmetic
    print("Add:\n", (a + b).numpy())
    print("Multiply element-wise:\n", (a * 2).numpy())
    print("Matrix multiply:\n", tf.matmul(a, b).numpy())

    # Reshaping
    e = tf.reshape(a, (4,))  # Flatten to 1D
    print("Reshaped e:", e.numpy())

    # Indexing & slicing
    print("First row:", a[0].numpy())
    print("Element at (1, 0):", a[1, 0].numpy())

    # Reductions
    print("Sum of all elements:", tf.reduce_sum(a).numpy())
    print("Mean of all elements:", tf.reduce_mean(a).numpy())
    print("Max element:", tf.reduce_max(a).numpy())

    # Concatenation & stacking
    f = tf.concat([a, b], axis=0)  # Stack along rows
    g = tf.stack([a, b], axis=0)  # New dimension
    print("Concatenated:\n", f.numpy())
    print("Stacked shape:", g.shape)

    # Type casting
    a_int = tf.cast(a, tf.int32)
    print("a as int:\n", a_int.numpy())

    # Tensor ↔ NumPy
    np_array = a.numpy()  # Tensor → NumPy
    tensor_from_np = tf.convert_to_tensor(np_array)  # NumPy → Tensor
    print("Tensor from NumPy:\n", tensor_from_np.numpy())
