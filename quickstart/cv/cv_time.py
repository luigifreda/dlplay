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
Measuring Performance with OpenCV
cv2.getTickCount function returns the number of clock-cycles after a reference event 
(like the moment machine was switched ON) to the moment this function is called. 
So if you call it before and after the function execution, you get number of clock-cycles used to execute a function.

cv2.getTickFrequency function returns the frequency of clock-cycles, or the number of clock-cycles per second. 
So to find the time of execution in seconds, you can do following:
"""

e1 = cv2.getTickCount()
# your code execution
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()

# testing a particular example 

img1 = cv2.imread('data/messi5.jpg')

e1 = cv2.getTickCount()

for i in range(5,49,2):
    print(i)
    img1 = cv2.medianBlur(img1,i)
e2 = cv2.getTickCount()
t = (e2 - e1)/cv2.getTickFrequency()
print(t)
