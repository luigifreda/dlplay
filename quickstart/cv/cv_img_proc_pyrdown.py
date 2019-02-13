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
Theory

Normally, we used to work with an image of constant size. But in some occassions, we need to work with images of different resolution of the same image. 
For example, while searching for something in an image, like face, we are not sure at what size the object will be present in the image. 
In that case, we will need to create a set of images with different resolution and search for object in all the images. 
These set of images with different resolution are called Image Pyramids 
(because when they are kept in a stack with biggest image at bottom and smallest image at top look like a pyramid).

There are two kinds of Image Pyramids. 
1) Gaussian Pyramid 
and 
2) Laplacian Pyramids

Higher level (Low resolution) in a Gaussian Pyramid is formed by removing consecutive rows and columns in Lower level (higher resolution) image. 
Then each pixel in higher level is formed by the contribution from 5 pixels in underlying level with gaussian weights. 
By doing so, a M \times N image becomes M/2 \times N/2 image. So area reduces to one-fourth of original area. 
It is called an Octave. The same pattern continues as we go upper in pyramid (ie, resolution decreases). 
Similarly while expanding, area becomes 4 times in each level. We can find Gaussian pyramids using cv2.pyrDown() and cv2.pyrUp() functions.

"""
img = cv2.imread('data/messi5.jpg')
img_down = cv2.pyrDown(img)
img_down_up = cv2.pyrUp(img_down)

cv2.imshow('img',img)
cv2.imshow('img_down',img_down)
cv2.imshow('img_down_up',img_down_up)
cv2.waitKey(0)
cv2.destroyAllWindows()