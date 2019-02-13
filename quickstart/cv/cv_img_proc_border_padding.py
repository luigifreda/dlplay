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
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# from https://docs.opencv.org/trunk/d3/df2/tutorial_py_basic_ops.html#gsc.tab=0

"""
borderType - Flag defining what kind of border to be added. It can be following types:
cv.BORDER_CONSTANT - Adds a constant colored border. The value should be given as next argument.
cv.BORDER_REFLECT - Border will be mirror reflection of the border elements, like this : fedcba|abcdefgh|hgfedcb
cv.BORDER_REFLECT_101 or cv.BORDER_DEFAULT - Same as above, but with a slight change, like this : gfedcb|abcdefgh|gfedcba
cv.BORDER_REPLICATE - Last element is replicated throughout, like this: aaaaaa|abcdefgh|hhhhhhh
cv.BORDER_WRAP - Can't explain, it will look like this : cdefgh|abcdefgh|abcdefg
"""

BLUE = [255,0,0]
img1 = cv.imread('data/opencv-logo.png')

top_border = 50
bottom_border = 50
left_border = 60
right_border = 70

replicate = cv.copyMakeBorder(img1,top_border,bottom_border,left_border,right_border,cv.BORDER_REPLICATE)
reflect = cv.copyMakeBorder(img1,top_border,bottom_border,left_border,right_border,cv.BORDER_REFLECT)
reflect101 = cv.copyMakeBorder(img1,top_border,bottom_border,left_border,right_border,cv.BORDER_REFLECT_101)
wrap = cv.copyMakeBorder(img1,top_border,bottom_border,left_border,right_border,cv.BORDER_WRAP)
constant= cv.copyMakeBorder(img1,top_border,bottom_border,left_border,right_border,cv.BORDER_CONSTANT,value=BLUE)

plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

plt.show()