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
from torch.utils.data import DataLoader

# A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.
# Take a look at this implementation; the FashionMNIST images are stored in a directory img_dir,
# and their labels are stored separately in a CSV file annotations_file.
from dlplay.datasets.custom_dataset import CustomImageDataset
from dlplay.utils.conversions import to_numpy_uint_image
from dlplay.paths import DATA_DIR, RESULTS_DIR

import numpy as np
import cv2


# from https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#preparing-your-data-for-training-with-dataloaders
if __name__ == "__main__":
    dataset = CustomImageDataset(
        annotations_file=f"{DATA_DIR}/datasets/fashion_mnist_cv2/labels.csv",
        img_dir=f"{DATA_DIR}/datasets/fashion_mnist_cv2",
    )

    # The Dataset retrieves our dataset’s features and labels one sample at a time.
    # While training a model, we typically want to pass samples in “minibatches”,
    # reshuffle the data at every epoch to reduce model overfitting, and use Python’s multiprocessing to speed up data retrieval.

    # The torch.utils.data.DataLoader is an iterator which provides methods to
    # access the samples.

    # create a DataLoader
    # NOTE: Setting shuffle=True in a DataLoader means:
    # - At the beginning of each new epoch, the indices of the dataset are reshuffled before batching.
    # - So the training dataset will be presented in a new random order at every epoch.
    # - This behavior is standard practice for training — it prevents the model from overfitting to the fixed order of the data.
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # We have loaded that dataset into the DataLoader and can iterate through the dataset as needed.
    # Each iteration below returns a batch of train_features and train_labels
    # (containing batch_size features and labels respectively).
    # Because we specified shuffle=True, after we iterate over all batches the data is shuffled

    for i in range(len(dataloader)):
        batch_imgs, batch_labels = next(iter(dataloader))
        # print(f"batch: {batch}")
        print(f"batch size: {len(batch_imgs)}")  # batch_size = 10
        print(f"batch_imgs shape: {batch_imgs.shape}")  # (10, 28, 28, 1)
        for j in range(len(batch_imgs)):
            img = batch_imgs[j]  # (10, 28, 28, 1)
            label = batch_labels[j]
            cv2_img = to_numpy_uint_image(img, scale_factor=1.0)
            cv2.imshow(f"img", cv2_img)
            # print(f"label: {label}")
            cv2.waitKey(10)

    cv2.destroyAllWindows()
