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

from dlplay.paths import DATA_DIR, RESULTS_DIR

if __name__ == "__main__":

    img = cv2.imread("data/home.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # sift = cv2.SIFT()
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)
    """
    sift.detect() function finds the keypoint in the images. You can pass a mask if you want to search only a part of image. 
    Each keypoint is a special structure which has many attributes like its (x,y) coordinates, size of the meaningful neighbourhood, 
    angle which specifies its orientation, response that specifies strength of keypoints etc.
    """

    # img=cv2.drawKeypoints(gray,kp)
    cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    """
    OpenCV also provides cv2.drawKeyPoints() function which draws the small circles on the locations of keypoints. 
    If you pass a flag, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS to it, 
    it will draw a circle with size of keypoint and it will even show its orientation. See below example.
    """

    # Since you already found keypoints, you can call sift.compute() which computes the descriptors from the keypoints we have found.
    # If you didn’t find keypoints, directly find keypoints and descriptors in a single step with the function, sift.detectAndCompute().
    kp, des = sift.compute(gray, kp)

    cv2.imwrite(f"{RESULTS_DIR}/sift_keypoints.jpg", img)

    cv2.namedWindow("sift_keypoints", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("sift_keypoints", img)
    if cv2.waitKey(0) & 0xFF == 27:
        cv2.destroyAllWindows()
