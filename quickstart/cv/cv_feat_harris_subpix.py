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

from dlplay.paths import DATA_DIR, RESULTS_DIR


if __name__ == "__main__":

    filename = "data/objects.jpg"
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    """
    computes the connected components labeled image of boolean image image with 4 or 8 way connectivity - returns N, 
    the total number of labels [0, N-1] where 0 represents the background label. ltype specifies the output label image type, 
    an important consideration based on the total number of labels or alternatively the total number of pixels in the source image.
    """

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    # Now draw them
    res = np.hstack((centroids, corners))
    res = np.int32(
        res
    )  # Integer used for indexing (same as C ssize_t; normally either int32 or int64)
    img[res[:, 1], res[:, 0]] = [0, 0, 255]  # red centroids
    img[res[:, 3], res[:, 2]] = [0, 255, 0]  # green corners

    cv2.imwrite(f"{RESULTS_DIR}/subpixel5.png", img)
    cv2.namedWindow("subpixel5", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("subpixel5", img)
    if cv2.waitKey(0) & 0xFF == 27:
        cv2.destroyAllWindows()
