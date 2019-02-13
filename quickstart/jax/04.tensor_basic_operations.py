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
    #  Tensor Basic Operations

    a = jnp.array([1.0, 2.0, 3.0])
    b = jnp.array([4.0, 5.0, 6.0])

    #  Add, Mul, Dot, MatMul
    print("Add:", a + b)
    print("Mul:", a * b)
    print("Dot:", jnp.dot(a, b))
    print("MatMul:", jnp.matmul(a[None, :], b[:, None]))

    #  Element-wise operations
    print("Element-wise max:", jnp.maximum(a, b))
    print("Element-wise min:", jnp.minimum(a, b))
    print("Element-wise abs:", jnp.abs(a))
    print("Element-wise sign:", jnp.sign(a))
    print("Element-wise sqrt:", jnp.sqrt(a))

    #  Reduction operations
    print("Sum:", jnp.sum(a))
    print("Mean:", jnp.mean(a))
    print("Std:", jnp.std(a))
    print("Max:", jnp.max(a))
    print("Min:", jnp.min(a))
    print("Argmax:", jnp.argmax(a))
    print("Argmin:", jnp.argmin(a))

    print("L2 Norm:", jnp.linalg.norm(a))
