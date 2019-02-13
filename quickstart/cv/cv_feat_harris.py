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


# apply Non-Maxima Suppression (NMS) to an image that represents a score map 
def nms_from_map(score_map, size):
    kernel = np.ones((size,size),np.uint8)
    score_map = score_map * (score_map == cv2.dilate(score_map, kernel))  # cv2.dilate is a maximum filter 
    return score_map


filename = './data/objects.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
"""
The function runs the Harris edge detector on the image. Similarly to cornerMinEigenVal() and cornerEigenValsAndVecs() , 
for each pixel (x, y) it calculates a 2x2 gradient covariance matrix M^{(x,y)} over a blockSizexblockSize neighborhood. 
Then, it computes the following characteristic:

\texttt{dst} (x,y) =  \mathrm{det} M^{(x,y)} - k  \cdot \left ( \mathrm{tr} M^{(x,y)} \right )^2
"""

# apply NMS 
dst = nms_from_map(dst, size=5)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

cv2.namedWindow('dst',cv2.WINDOW_KEEPRATIO)
cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()