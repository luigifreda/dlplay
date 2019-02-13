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


img = cv2.imread('data/butterfly.jpg',0)

# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
#surf = cv2.SURF(400)
surf = cv2.xfeatures2d.SURF_create(10000)

# In actual cases, it is better to have a value 300-500
#surf.hessianThreshold = 50000

# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(img,None)

print(len(kp))

img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)

cv2.namedWindow('dst',cv2.WINDOW_KEEPRATIO)
cv2.imshow('dst',img2)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()