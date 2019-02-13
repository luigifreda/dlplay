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
import os
from torchvision.io import decode_image

# A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.
# Take a look at this implementation; the FashionMNIST images are stored in a directory img_dir,
# and their labels are stored separately in a CSV file annotations_file.
from dlplay.datasets.custom_dataset import CustomImageDataset
from dlplay.utils.conversions import to_numpy_uint_image
from dlplay.paths import DATA_DIR, RESULTS_DIR

import numpy as np
import cv2


# from https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
if __name__ == "__main__":

    dataset = CustomImageDataset(
        annotations_file=f"{RESULTS_DIR}/datasets/fashion_mnist_cv2/labels.csv",
        img_dir=f"{RESULTS_DIR}/datasets/fashion_mnist_cv2",
    )

    num_samples = len(dataset)
    print(f"num_samples: {num_samples}")

    # print the first 10 images
    for i in range(num_samples):
        img, label = dataset[i]
        cv2_img = to_numpy_uint_image(img, scale_factor=1.0)
        cv2.imshow(f"img", cv2_img)
        print(f"label: {label}")
        cv2.waitKey(40)

    cv2.destroyAllWindows()
