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

    img = cv2.imread("data/objects.jpg", 0)
    img2 = img.copy()
    img3 = img.copy()

    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create()
    # find and draw the keypoints
    kp = fast.detect(img, None)
    img2 = cv2.drawKeypoints(
        img,
        kp,
        None,
        color=(255, 0, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    # Print all default params
    print("Threshold: {}".format(fast.getThreshold()))
    print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
    print("neighborhood: {}".format(fast.getType()))
    print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))
    cv2.imwrite(f"{RESULTS_DIR}/fast_true.png", img2)

    # Disable nonmaxSuppression
    fast.setNonmaxSuppression(0)
    kp = fast.detect(img, None)
    print("Total Keypoints without nonmaxSuppression: {}".format(len(kp)))
    img3 = cv2.drawKeypoints(
        img,
        kp,
        None,
        color=(255, 0, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    cv2.imwrite(f"{RESULTS_DIR}/fast_false.png", img3)

    cv2.namedWindow("img2", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("img2", img2)

    cv2.namedWindow("img3", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("img3", img3)

    if cv2.waitKey(0) & 0xFF == 27:
        cv2.destroyAllWindows()
