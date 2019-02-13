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
import numpy as np, sys

from dlplay.paths import DATA_DIR, RESULTS_DIR


if __name__ == "__main__":

    A = cv2.imread("data/apple.jpg")
    B = cv2.imread("data/orange.jpg")

    # generate Gaussian pyramid for A
    G = A.copy()
    gpA = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpA.append(G)

    # generate Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpB.append(G)

    # generate Laplacian Pyramid for A
    lpA = [gpA[5]]
    for i in range(5, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)

    # generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in range(5, 0, -1):
        GE = cv2.pyrUp(gpB[i])
        L = cv2.subtract(gpB[i - 1], GE)
        lpB.append(L)

    # Now add left and right halves of images in each level
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        ls = np.hstack((la[:, 0 : round(cols / 2)], lb[:, round(cols / 2) :]))
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1, 6):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])

    # image with direct connecting each half
    real = np.hstack((A[:, : round(cols / 2)], B[:, round(cols / 2) :]))

    cv2.imwrite(f"{RESULTS_DIR}/Pyramid_blending2.jpg", ls_)
    cv2.imwrite(f"{RESULTS_DIR}/Direct_blending.jpg", real)

    cv2.imshow("Pyramid_blending2", ls_)
    cv2.imshow("Direct_blending", real)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
