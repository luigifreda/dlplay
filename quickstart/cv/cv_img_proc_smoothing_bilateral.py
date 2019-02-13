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

# concept: http://www.coldvision.io/wp-content/uploads/2016/01/bilateral_filter.jpg

"""
4. Bilateral Filtering
cv2.bilateralFilter() is highly effective in noise removal while keeping edges sharp. But the operation is slower compared to other filters. 
We already saw that gaussian filter takes the a neighbourhood around the pixel and find its gaussian weighted average. 
This gaussian filter is a function of space alone, that is, nearby pixels are considered while filtering. 
It doesn’t consider whether pixels have almost same intensity. It doesn’t consider whether pixel is an edge pixel or not. 
So it blurs the edges also, which we don’t want to do.

Bilateral filter also takes a gaussian filter in space, but one more gaussian filter which is a function of pixel difference. 
Gaussian function of space make sure only nearby pixels are considered for blurring while gaussian function of intensity difference 
make sure only those pixels with similar intensity to central pixel is considered for blurring. 
So it preserves the edges since pixels at edges will have large intensity variation.

Below samples shows use bilateral filter (For details on arguments, visit docs).
"""

img = cv2.imread('data/messi5.jpg')
# convert to BGR 
img = img[..., [2,1,0]]  # invert RGB fiels for a correct visualization 

blur = cv2.bilateralFilter(img,9,75,75)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Bilateral')
plt.xticks([]), plt.yticks([])
plt.show()