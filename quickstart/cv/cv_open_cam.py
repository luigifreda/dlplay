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

"""
To capture a video, you need to create a VideoCapture object. 
Its argument can be either the device index or the name of a video file. 
Device index is just the number to specify which camera. Normally one camera will be connected (as in my case). 
So I simply pass 0 (or -1). You can select the second camera by passing 1 and so on. 
After that, you can capture frame-by-frame. But at the end, don’t forget to release the capture.
"""
cam = 1
print('opening cam: ', cam)
cap = cv2.VideoCapture(cam)

while(True):
    # Capture frame-by-frame
    # cap.read() returns a bool (True/False). If frame is read correctly, it will be True. So you can check end of the video by checking this return value.    
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()