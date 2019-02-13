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
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from dlplay.utils.conversions import to_numpy_uint_image
from dlplay.paths import DATA_DIR, RESULTS_DIR

import cv2


this_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(RESULTS_DIR, "datasets", "fashion_mnist_cv2")
annotations_file_path = os.path.join(target_dir, "labels.csv")

show_images = True  # show or not the images in a window
save_loaded_images = False  # save or not loaded samples to disk

if __name__ == "__main__":

    if save_loaded_images:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        # create the file handle
        annotations_file = open(annotations_file_path, "w")

    # PyTorch provides two data primitives:
    # torch.utils.data.DataLoader
    # torch.utils.data.Dataset
    # These allow you to use pre-loaded datasets as well as your own data.
    # Dataset stores the samples and their corresponding labels, and
    # DataLoader wraps an iterable around the Dataset to enable easy access to the samples.

    training_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )

    # test_data = datasets.FashionMNIST(
    #     root="data", train=False, download=True, transform=ToTensor()
    # )

    # We can index Datasets manually like a list: training_data[index].
    # We use matplotlib to visualize some samples in our training data.

    # NOTE: This is the mapping of the labels to the actual clothing items.
    # We have 10 classes, and each class is a different clothing item.
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

    # let's explore the dataset by sampling random images
    figure = plt.figure(figsize=(10, 10))
    cols, rows = 10, 10  # 10x10 grid of sample images

    num_samples = len(training_data)
    print(f"num_samples: {num_samples}")

    # let's explore the dataset by sampling random images
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(num_samples, size=(1,)).item()
        img, label = training_data[sample_idx]  # img is a torch.Tensor

        if show_images:
            # used for building the custom dataset

            # convert to cv2 image: from normalized float in [0, 1] to uint8 in [0, 255]
            cv2_img = to_numpy_uint_image(img)
            print(f"cv2_img.shape: {cv2_img.shape}")

            cv2.imshow(f"img", cv2_img)
            cv2.waitKey(5)

            if save_loaded_images:
                # save to jpg
                dst_file_name = f"img_{i}.png"
                dst_file_path = os.path.join(target_dir, dst_file_name)
                print(f"saving image with label {label} to {dst_file_path}")
                cv2.imwrite(dst_file_path, cv2_img)
                annotations_file.write(f"{dst_file_name}, {label}\n")

            figure.add_subplot(rows, cols, i)
            plt.title(labels_map[label])
            plt.axis("off")
            plt.imshow(cv2_img, cmap="gray")

    if show_images:
        plt.tight_layout()
        plt.show()  # show the figure and wait for the user to close it

    if save_loaded_images:
        annotations_file.close()
        cv2.destroyAllWindows()
