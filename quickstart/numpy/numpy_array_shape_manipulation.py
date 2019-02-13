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
Changing the shape of an array
An array has a shape given by the number of elements along each axis:
"""

a = np.floor(10*np.random.random((3,4)))
print(a)
# array([[ 2.,  8.,  0.,  6.],
#        [ 4.,  5.,  1.,  1.],
#        [ 8.,  9.,  3.,  6.]])
print(a.shape)
# (3, 4)

"""
The shape of an array can be changed with various commands. 
Note that the following three commands all return a modified array, but do not change the original array:
"""

b = a.ravel()  # returns the array, flattened
print(b)
# array([ 2.,  8.,  0.,  6.,  4.,  5.,  1.,  1.,  8.,  9.,  3.,  6.])

c = a.reshape(6,2)  # returns the array with a modified shape
print(c)
# array([[ 2.,  8.],
#        [ 0.,  6.],
#        [ 4.,  5.],
#        [ 1.,  1.],
#        [ 8.,  9.],
#        [ 3.,  6.]])

at = a.T  # returns the array, transposed
print(at)
# array([[ 2.,  4.,  8.],
#        [ 8.,  5.,  9.],
#        [ 0.,  1.,  3.],
#        [ 6.,  1.,  6.]])

print(a.T.shape)
# (4, 3)

print(a.shape)
# (3, 4)

"""
The reshape function returns its argument with a modified shape, whereas the ndarray.resize method modifies the array itself:
"""

a = np.array([[ 2.,  8.,  0.,  6.],
             [ 4.,  5.,  1.,  1.],
             [ 8.,  9.,  3.,  6.]])
b = a.resize((2,6))
print(b)
# array([[ 2.,  8.,  0.,  6.,  4.,  5.],
#        [ 1.,  1.,  8.,  9.,  3.,  6.]])

"""
If a dimension is given as -1 in a reshaping operation, the other dimensions are automatically calculated:
"""

c = a.reshape(3,-1)
print(c)
# array([[ 2.,  8.,  0.,  6.],
#        [ 4.,  5.,  1.,  1.],
#        [ 8.,  9.,  3.,  6.]])