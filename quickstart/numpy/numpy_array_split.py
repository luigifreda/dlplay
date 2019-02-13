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

"""
The opposite of concatenation is splitting, which is implemented by the functions np.split, np.hsplit, and np.vsplit. 
"""


x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])
print(x1, x2, x3)
# [1 2 3] [99 99] [3 2 1]


"""
Notice that N split-points, leads to N + 1 subarrays. The related functions np.hsplit and np.vsplit are similar:
"""

grid = np.arange(16).reshape((4, 4))
# grid
# array([[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11],
#        [12, 13, 14, 15]])
upper, lower = np.vsplit(grid, [2])
print(upper)
# [[0 1 2 3]
#  [4 5 6 7]]
print(lower)
# [[ 8  9 10 11]
#  [12 13 14 15]]

left, right = np.hsplit(grid, [2])
print(left)
# [[ 0  1]
#  [ 4  5]
#  [ 8  9]
#  [12 13]]
print(right)
# [[ 2  3]
#  [ 6  7]
#  [10 11]
#  [14 15]]