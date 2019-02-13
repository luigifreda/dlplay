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
import numpy as np
import cv2
from matplotlib import pyplot as plt

"""
Use the function cv2.imread() to read an image. The image should be in the working directory or a full path of image should be given.
Second argument is a flag which specifies the way image should be read.
- cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. It is the default flag.
- cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
- cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel
"""

img = cv2.imread('data/messi5.jpg',cv2.IMREAD_COLOR)
print('image shape: ', img.shape)
#plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.imshow(img[:,:,[0,1,2]], cmap = 'gray', interpolation = 'bicubic') # invert RGB fiels for a correct visualization 
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()



