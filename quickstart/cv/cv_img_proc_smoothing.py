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
import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
1. Averaging

This is done by convolving image with a normalized box filter. 
It simply takes the average of all the pixels under kernel area and replace the central element. 
This is done by the function cv2.blur() or cv2.boxFilter(). Check the docs for more details about the kernel. 
We should specify the width and height of kernel. A 3x3 normalized box filter would look like below:

K =  \frac{1}{9} \begin{bmatrix} 1 & 1 & 1  \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}

Note If you don’t want to use normalized box filter, use cv2.boxFilter(). Pass an argument normalize=False to the function.

"""    

"""
2. Gaussian Blurring
In this, instead of box filter, gaussian kernel is used. It is done with the function, cv2.GaussianBlur(). 
We should specify the width and height of kernel which should be positive and odd. 
We also should specify the standard deviation in X and Y direction, sigmaX and sigmaY respectively. 
If only sigmaX is specified, sigmaY is taken as same as sigmaX. If both are given as zeros, they are calculated from kernel size. 
Gaussian blurring is highly effective in removing gaussian noise from the image.

If you want, you can create a Gaussian kernel with the function, cv2.getGaussianKernel().

The above code can be modified for Gaussian blurring:
"""

img = cv2.imread('data/opencv_logo.png')

kernel_size = 51  # must be odd 

blur = cv2.blur(img,(kernel_size,kernel_size))
blur2 = cv2.GaussianBlur(img,(kernel_size,kernel_size),0)

plt.subplot(131),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(blur2),plt.title('GaussianBlur')
plt.xticks([]), plt.yticks([])
plt.show()