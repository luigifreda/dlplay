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

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


# from https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html
class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        self.indices = list(range(len(self.imgs)))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = read_image(img_path)
        mask = read_image(mask_path)
        # Instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # First id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # Split the color-encoded mask into a set of binary masks.
        # masks, torchvision.tv_tensors.Mask of shape [N, H, W]: the segmentation masks for each one of the objects
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # Get bounding box coordinates for each mask.
        # boxes, torchvision.tv_tensors.BoundingBoxes of shape [N, 4]:
        # the coordinates of the N bounding boxes in [x0, y0, x1, y1] format,
        # ranging from 0 to W and 0 to H
        boxes = masks_to_boxes(masks)

        # There is only one class.
        # labels, integer torch.Tensor of shape [N]: the label for each bounding box.
        # NOTE: 0 represents always the background class.
        labels = torch.ones((num_objs,), dtype=torch.int64)

        # image_id, int: an image identifier. It should be unique between all the images in the dataset, and is used during evaluation
        image_id = idx

        # area, float torch.Tensor of shape [N]: the area of the bounding box.
        # This is used during evaluation with the COCO metric, to separate the metric
        # scores between small, medium and large boxes.
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Suppose all instances are not crowd.
        # iscrowd, uint8 torch.Tensor of shape [N]: instances with iscrowd=True will be ignored during evaluation.
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        # image: torchvision.tv_tensors.Image of shape [3, H, W], a pure tensor, or a PIL Image of size (H, W)
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(
            boxes, format="XYXY", canvas_size=F.get_size(img)
        )
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
