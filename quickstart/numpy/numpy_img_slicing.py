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

delta = 2
start = np.array([[2,7],
 [5,3],
 [1,4]],dtype=np.intp)
print('start:',start)
end = start + delta
print('end:',end)
ranges = np.linspace(start,end,3,dtype=np.intp)[:,:].T 
print('ranges:',ranges)
elem = ranges[:,0]
print('elem:',elem)
img = np.arange(100).reshape(10,10)
print('\n img:',img)
print('\n img[elem[0],elem[1]]:',img[elem[0][:,np.newaxis],elem[1]])
#print('\n img[elem]:',img[elem])