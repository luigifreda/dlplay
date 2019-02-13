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

img = cv2.imread('data/messi5.jpg')

"""
Accessing properties
"""

# Shape of image is accessed by img.shape. It returns a tuple of number of rows, columns and channels (if image is color):
print(img.shape)

# Total number of pixels is accessed by img.size:
print(img.size)
# 562248

# Image datatype is obtained by img.dtype:
print(img.dtype)
# uint8

px = img[100,100]
print(px)
#[157 166 200]

# accessing only blue pixel
blue = img[100,100,0] # first x,y then r,g,b
print(blue)
# 157

"""
Warning Numpy is a optimized library for fast array calculations. 
So simply accessing each and every pixel values and modifying it will be very slow and it is discouraged.
Note Above mentioned method is normally used for selecting a region of array, say first 5 rows and last 3 columns like that. 
For individual pixel access, Numpy array methods, array.item() and array.itemset() is considered to be better. 
But it always returns a scalar. So if you want to access all B,G,R values, you need to call array.item() separately for all.
"""

# accessing RED value
print(img.item(10,10,2))
#59

# modifying RED value
img.itemset((10,10,2),100)
print(img.item(10,10,2))
# 100

"""
Image ROI
"""
# Sometimes, you will have to play with certain region of images. 
# For eye detection in images, first face detection is done all over the image and when face is obtained, 
# we select the face region alone and search for eyes inside it instead of searching whole image. 
# It improves accuracy (because eyes are always on faces :D ) and performance (because we search for a small area)

# ROI is again obtained using Numpy indexing. Here I am selecting the ball and copying it to another region in the image:

img2 = img.copy()
ball = img[280:340, 330:390]
img2[273:333, 100:160] = ball

cv2.imshow('image ',img)
cv2.imshow('image2',img2)
cv2.waitKey(0)
