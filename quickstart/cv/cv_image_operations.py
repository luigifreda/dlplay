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
Image Addition

Note There is a difference between OpenCV addition and Numpy addition. 
OpenCV addition is a saturated operation while Numpy addition is a modulo operation.
For example, consider below sample:
"""

x = np.uint8([250])
y = np.uint8([10])

print(cv2.add(x,y)) # 250+10 = 260 => 255
#[[255]]

print(x+y)          # 250+10 = 260 % 256 = 4
#[4]

"""
Image Blending

This is also image addition, but different weights are given to images so that it gives a feeling of blending or transparency. Images are added as per the equation below:
g(x) = (1 - \alpha)f_{0}(x) + \alpha f_{1}(x)
"""

img1 = cv2.imread('data/left.jpg')
img2 = cv2.imread('data/right.jpg')

dst = cv2.addWeighted(img1,0.7,img2,0.3,0)

cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()