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
from jax import grad


#  Define a function
def f(x):
    return x**3 + 2 * x


if __name__ == "__main__":
    #  Tensor Autodiff Basics

    #  Compute the gradient of the function
    df = grad(f)
    # df/dx = 3x^2 + 2

    print("df(2.0):", df(2.0))
    # df(2.0) = 3*2^2 + 2 = 12 + 2 = 14
