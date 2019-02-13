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
When operating and manipulating arrays, their data is sometimes copied into a new array and sometimes not. 
This is often a source of confusion for beginners. There are three cases:
"""

#--------------------
# No Copy at All
"""
Simple assignments make no copy of array objects or of their data.
"""


a = np.arange(12)
b = a            # no new object is created
print(b is a)    # a and b are two names for the same ndarray object
# True

b.shape = 3,4    # changes the shape of a
print(a.shape)
# (3, 4)

"""
Python passes mutable objects as references, so function calls make no copy.
"""

def f(x):
    print(id(x))

print(id(a))                           # id is a unique identifier of an object
# 148293216
f(a)
# 148293216


#--------------------
# View or Shallow Copy
"""
Different array objects can share the same data. The view method creates a new array object that looks at the same data.
"""

c = a.view()
print(c is a)
# False
print(c.base is a)                        # c is a view of the data owned by a
# True
print(c.flags.owndata)
# False

c.shape = 2,6                      # a's shape doesn't change
print(a.shape)
#(3, 4)
c[0,4] = 1234                      # a's data changes
print(a)
# array([[   0,    1,    2,    3],
#        [1234,    5,    6,    7],
#        [   8,    9,   10,   11]])

"""
Slicing an array returns a view of it:
"""

s = a[ : , 1:3]     # spaces added for clarity; could also be written "s = a[:,1:3]"
s[:] = 10           # s[:] is a view of s. Note the difference between s=10 and s[:]=10
print(a)
# array([[   0,   10,   10,    3],
#        [1234,   10,   10,    7],
#        [   8,   10,   10,   11]])

#--------------------
# Deep Copy
"""
The copy method makes a complete copy of the array and its data.
"""


d = a.copy()                          # a new array object with new data is created
print(d is a)
# False
print(d.base is a)                           # d doesn't share anything with a
# False

d[0,0] = 9999
print(a)
# array([[   0,   10,   10,    3],
#        [1234,   10,   10,    7],
#        [   8,   10,   10,   11]])