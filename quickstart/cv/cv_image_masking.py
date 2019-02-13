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
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html
import cv2
import numpy as np
import sys 


# load the image
img = cv2.imread('data/opencv_logo.png')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if img is None: 
    print('img is empty!')

# detect circles
gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 5)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=32, minRadius=0, maxRadius=0)
if circles is None: 
    print('circles is empty!')
    sys.exit(0)
    
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(gray,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(gray,(i[0],i[1]),2,(0,0,255),3)
    cv2.imshow('circles',gray)    


# draw mask
mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8)  # mask is only 
for i in circles[0, :]:
    cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), -1)
cv2.imshow('mask',mask)    

# get first masked value (foreground)
fg = cv2.bitwise_or(img, img, mask=mask)
cv2.imshow('fg',fg)    

# get second masked value (background) mask must be inverted
mask = cv2.bitwise_not(mask)
background = np.full(img.shape, 255, dtype=np.uint8)
bk = cv2.bitwise_or(background, background, mask=mask)
cv2.imshow('bk',bk)    

# combine foreground+background
final = cv2.bitwise_or(fg, bk)
cv2.imshow('final',final)

cv2.waitKey(0)
cv2.destroyAllWindows()