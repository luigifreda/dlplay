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

from dlplay.paths import DATA_DIR, RESULTS_DIR


if __name__ == "__main__":

    """
    Use the function cv2.imread() to read an image. The image should be in the working directory or a full path of image should be given.
    Second argument is a flag which specifies the way image should be read.
    - cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. It is the default flag.
    - cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
    - cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel
    """

    # Load an color image in grayscale
    img_color = cv2.imread("data/messi5.jpg", cv2.IMREAD_COLOR)
    print("img_color type: ", img_color.dtype)

    # convert to BGR
    image_color_bgr = img_color[..., [2, 1, 0]]

    img_gray = cv2.imread("data/messi5.jpg", cv2.IMREAD_GRAYSCALE)
    print("img_gray type: ", img_gray.dtype)

    cv2.namedWindow(
        "image color", cv2.WINDOW_KEEPRATIO
    )  # in order to have the image resizable
    cv2.imshow("image color", img_color)

    cv2.imshow("image image_color_bgr", image_color_bgr)
    cv2.imshow("image gray", img_gray)

    """
    cv2.waitKey() is a keyboard binding function. Its argument is the time in milliseconds. T
    he function waits for specified milliseconds for any keyboard event. If you press any key in that time, the program continues. 
    If 0 is passed, it waits indefinitely for a key stroke. 
    It can also be set to detect specific key strokes like, if key a is pressed etc which we will discuss below.
    """
    k = cv2.waitKey(0)
    if k == 27:  # wait for ESC key to exit
        print("just exit")
    elif k == ord("s"):  # wait for 's' key to save and exit
        print("saving gray image")
        cv2.imwrite(f"{RESULTS_DIR}/messi_gray.png", img_gray)

    cv2.destroyAllWindows()
