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
#!/usr/bin/env python3
import numpy as np
np.random.seed(0)  # seed for reproducibility

x1 = np.random.randint(10, size=6)  # One-dimensional array
x2 = np.random.randint(10, size=(3, 4))  # Two-dimensional array
x3 = np.random.randint(10, size=(3, 4, 5))  # Three-dimensional array

print("x3 ndim: ", x3.ndim)
print("x3 shape:", x3.shape)
print("x3 size: ", x3.size)
print("dtype:", x3.dtype)

print(x2)
# [[3  5  2  4]
#  [ 7  6  8  8]
#  [ 1  6  7  7]]

x2_sub_copy = x2[:2, :2].copy()
print(x2_sub_copy)
# [[3  5]
#  [ 7  6]]

x2_sub_copy[0, 0] = 42
print(x2_sub_copy)
# [[42  5]
#  [ 7  6]]

print(x2)
# [[3  5  2  4]
#  [ 7  6  8  8]
#  [ 1  6  7  7]]
