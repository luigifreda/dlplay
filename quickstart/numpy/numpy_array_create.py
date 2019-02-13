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
import math 

a = np.zeros((2,2))   # Create an array of all zeros
print(a)              # Prints "[[ 0.  0.]
                      #          [ 0.  0.]]"

b = np.ones((1,2))    # Create an array of all ones
print(b)              # Prints "[[ 1.  1.]]"

c = np.full((2,2), 7)  # Create a constant array
print(c)               # Prints "[[ 7.  7.]
                       #          [ 7.  7.]]"

d = np.eye(2)         # Create a 2x2 identity matrix
print(d)              # Prints "[[ 1.  0.]
                      #          [ 0.  1.]]"

e = np.random.random((2,2))  # Create an array filled with random values
print(e)                     # Might print "[[ 0.91940167  0.08143941]
                             #               [ 0.68744134  0.87236687]]"

f= np.arange( 10, 30, 5 )
print(f) 
# array([10, 15, 20, 25])

g= np.arange( 0, 2, 0.3 )                 # it accepts float arguments
print(g) 
# array([ 0. ,  0.3,  0.6,  0.9,  1.2,  1.5,  1.8])


k = np.linspace( 0, 2, 9 )    # 9 numbers from 0 to 2
# array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75,  2.  ])

w = np.linspace( 0, 2*math.pi, 100 )        # useful to evaluate function at lots of points
fun = np.sin(w)


test = np.zeros((10, 0))
print('test: ', test )
print('test.shape: ', test.shape )