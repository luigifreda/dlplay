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
import jax.numpy as jnp

if __name__ == "__main__":
    #  Tensor Reshaping and Views

    x = jnp.arange(12)

    reshaped = x.reshape(3, 4)
    transposed = reshaped.T

    print("Original:", x)
    print("Reshaped:\n", reshaped)
    print("Transposed:\n", transposed)

    #  Note: JAX doesn't support view() for reshaping like PyTorch
    #  Use reshape() instead
    print("Note: JAX's view() method doesn't support reshaping like PyTorch")
    print("Use reshape() for reshaping operations in JAX")

    #  Reshape
    reshaped = x.reshape(3, 4)
    print("Reshaped:\n", reshaped)
