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
Concatenation, or joining of two arrays in NumPy, is primarily accomplished using the routines np.concatenate, np.vstack, and np.hstack. 
"""

x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
np.concatenate([x, y])
# array([1, 2, 3, 3, 2, 1])


#--------------------

"""
You can also concatenate more than two arrays at once:
"""

z = [99, 99, 99]
print(np.concatenate([x, y, z]))
# [ 1  2  3  3  2  1 99 99 99]

#--------------------

"""
It can also be used for two-dimensional arrays:
"""

grid = np.array([[1, 2, 3],
                 [4, 5, 6]])
# concatenate along the first axis
np.concatenate([grid, grid])
# array([[1, 2, 3],
#        [4, 5, 6],
#        [1, 2, 3],
#        [4, 5, 6]])

# concatenate along the second axis (zero-indexed)
np.concatenate([grid, grid], axis=1)
# array([[1, 2, 3, 1, 2, 3],
#        [4, 5, 6, 4, 5, 6]])


#--------------------

"""
For working with arrays of mixed dimensions, it can be clearer to use the np.vstack (vertical stack) and np.hstack (horizontal stack) functions:
"""

x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7],
                 [6, 5, 4]])

# vertically stack the arrays
np.vstack([x, grid])
# array([[1, 2, 3],
#        [9, 8, 7],
#        [6, 5, 4]])

# horizontally stack the arrays
y = np.array([[99],
              [99]])
np.hstack([grid, y])
# array([[ 9,  8,  7, 99],
#        [ 6,  5,  4, 99]])

#Similary, np.dstack will stack arrays along the third axis.


"""
The function column_stack stacks 1D arrays as columns into a 2D array. It is equivalent to hstack only for 2D arrays:
"""

from numpy import newaxis

a = np.floor(10*np.random.random((2,2)))
# array([[ 8.,  8.],
#        [ 0.,  0.]])
b = np.floor(10*np.random.random((2,2)))
# array([[ 1.,  8.],
#        [ 0.,  4.]])

np.column_stack((a,b))     # with 2D arrays
# array([[ 8.,  8.,  1.,  8.],
#        [ 0.,  0.,  0.,  4.]])
a = np.array([4.,2.])
b = np.array([3.,8.])
np.column_stack((a,b))     # returns a 2D array
# array([[ 4., 3.],
#        [ 2., 8.]])

np.hstack((a,b))           # the result is different
# array([ 4., 2., 3., 8.])

a[:,newaxis]               # this allows to have a 2D columns vector
# array([[ 4.],
    #    [ 2.]])

np.column_stack((a[:,newaxis],b[:,newaxis]))
# array([[ 4.,  3.],
#        [ 2.,  8.]])

np.hstack((a[:,newaxis],b[:,newaxis]))   # the result is the same
# array([[ 4.,  3.],
#        [ 2.,  8.]])