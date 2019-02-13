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

b,g,r = cv2.split(img)
img2 = cv2.merge((b,g,r))

cv2.imshow('image ',img)
cv2.imshow('b image ',b)
cv2.imshow('g image ',g)
cv2.imshow('r image ',r)
cv2.imshow('image2',img2)
cv2.waitKey(0)
