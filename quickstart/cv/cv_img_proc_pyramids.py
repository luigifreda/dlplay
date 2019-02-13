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
import sys
import cv2

# from https://docs.opencv.org/3.4.9/d4/d1f/tutorial_pyramids.html


def main(argv):
    print(
        """
    Zoom In-Out demo
    ------------------
    * [i] -> Zoom [i]n
    * [o] -> Zoom [o]ut
    * [ESC] -> Close program
    """
    )

    # Load the image
    src = cv2.imread("data/messi5.jpg")
    # Check if image is loaded fine
    if src is None:
        print("Error opening image!")
        return -1

    while 1:
        rows, cols, _channels = map(int, src.shape)

        cv2.imshow("Pyramids Demo", src)

        k = cv2.waitKey(0)
        if k == 27:
            break

        elif chr(k) == "i":
            src = cv2.pyrUp(src, dstsize=(2 * cols, 2 * rows))
            print("** Zoom In: Image x 2")

        elif chr(k) == "o":
            src = cv2.pyrDown(src, dstsize=(cols // 2, rows // 2))
            print("** Zoom Out: Image / 2")

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
