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
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import decode_image


# A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.
# Take a look at this implementation; the FashionMNIST images are stored in a directory img_dir,
# and their labels are stored separately in a CSV file annotations_file.
# from https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
# NOTE:
# The only specificity that we require is that the dataset __getitem__ should return a tuple:
# image: torchvision.tv_tensors.Image of shape [3, H, W], a pure tensor, or a PIL Image of size (H, W)
# target: a dict containing the following fields
# boxes, torchvision.tv_tensors.BoundingBoxes of shape [N, 4]: the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
# labels, integer torch.Tensor of shape [N]: the label for each bounding box. 0 represents always the background class.
# image_id, int: an image identifier. It should be unique between all the images in the dataset, and is used during evaluation
# area, float torch.Tensor of shape [N]: the area of the bounding box. This is used during evaluation with the COCO metric, to separate the metric scores between small, medium and large boxes.
# iscrowd, uint8 torch.Tensor of shape [N]: instances with iscrowd=True will be ignored during evaluation.
# (optionally) masks, torchvision.tv_tensors.Mask of shape [N, H, W]: the segmentation masks for each one of the objects
class CustomImageDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
        """
        Args:
            annotations_file (string): path to the csv file with annotations.
            img_dir (string): directory with all the images.
            transform (callable, optional): transform to apply to a sample.
            target_transform (callable, optional): transform to apply to a target.

        The labels.csv file looks like:

        tshirt1.jpg, 0
        tshirt2.jpg, 0
        ...
        ankleboot999.jpg, 9
        """
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        print(f"img_path: {img_path}")
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
