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

a = np.array([[1.0, 2.0], [3.0, 4.0]])
print(a)
# [[ 1.  2.]
#  [ 3.  4.]]

a.transpose()
# array([[ 1.,  3.],
#        [ 2.,  4.]])

np.linalg.inv(a)
# array([[-2. ,  1. ],
#        [ 1.5, -0.5]])

u = np.eye(2) # unit 2x2 matrix; "eye" represents "I"
print(u)
# array([[ 1.,  0.],
#        [ 0.,  1.]])

j = np.array([[0.0, -1.0], [1.0, 0.0]])

print(j @ j)        # matrix product
# array([[-1.,  0.],
#        [ 0., -1.]])

print(np.trace(u))  # trace
# 2.0

y = np.array([[5.], [7.]])
np.linalg.solve(a, y)
# array([[-3.],
#        [ 4.]])


"""
np.linalg.eig(j)
Parameters:
    square matrix
Returns
    The eigenvalues, each repeated according to its multiplicity.
    The normalized (unit "length") eigenvectors, such that the
    column ``v[:,i]`` is the eigenvector corresponding to the
    eigenvalue ``w[i]`` .
"""
np.linalg.eig(j)
# (array([ 0.+1.j,  0.-1.j]), array([[ 0.70710678+0.j        ,  0.70710678-0.j        ],
#        [ 0.00000000-0.70710678j,  0.00000000+0.70710678j]]))


"""
numpy.linalg.svd(a, full_matrices=True, compute_uv=True, hermitian=False)[source]
Singular Value Decomposition.

When a is a 2D array, it is factorized as u @ np.diag(s) @ vh = (u * s) @ vh, 
where u and vh are 2D unitary arrays and s is a 1D array of a’s singular values. 
When a is higher-dimensional, SVD is applied in stacked mode as explained below.

https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html
"""

print('SVD')
a = np.random.randn(4, 5)
print('a = ',a)

# Reconstruction based on full SVD, 2D case:

u, s, vt = np.linalg.svd(a, full_matrices=True)
u.shape, s.shape, vt.shape
print('u = ',u)
print('s = ',s)
print('vt = ',vt)
