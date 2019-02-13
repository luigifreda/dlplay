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
from dlplay.datasets.pennfudan_dataset import PennFudanDataset
from dlplay.utils.conversions import to_numpy_uint_image
from dlplay.paths import DATA_DIR, RESULTS_DIR

import numpy as np
import cv2


# from https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
# NOTE: before running this script, you need to download the dataset using the following command:
# data/download_pennfudanped_dataset.sh
if __name__ == "__main__":

    max_num_masks = 100
    # create a color palette
    color_palette = np.random.randint(0, 255, (max_num_masks, 3)).astype(np.uint8)
    # print(f"color_palette: {color_palette}")

    max_out_img_width = 0
    max_out_img_height = 0

    dataset = PennFudanDataset(
        root=f"{RESULTS_DIR}/datasets/PennFudanPed",
        transforms=None,
    )

    num_samples = len(dataset)
    print(f"num_samples: {num_samples}")

    # print the first 10 images
    for i in range(num_samples):
        img, target = dataset[i]
        boxes = target["boxes"]
        masks = target["masks"]
        labels = target["labels"]
        image_id = target["image_id"]
        area = target["area"]
        iscrowd = target["iscrowd"]

        rgb_img = to_numpy_uint_image(img, scale_factor=1.0)
        # reverse the image
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        # print(f"rgb_img.shape: {rgb_img.shape}")

        masks_img = to_numpy_uint_image(masks, scale_factor=1.0)
        # NOTE: each mask instance corresponds to a different mask image channel, from zero to N, where
        # N is the number of instances.
        # print(f"masks_img.shape: {masks_img.shape}")

        # VIZ
        # create a full image with the given width and height

        max_out_img_width = max(max_out_img_width, rgb_img.shape[1], masks_img.shape[1])
        max_out_img_height = max(
            max_out_img_height, rgb_img.shape[0], masks_img.shape[0]
        )
        out_imgs = np.full(
            (max_out_img_height, 2 * max_out_img_width, 3), 255, dtype=np.uint8
        )

        # 1. visualize the boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.numpy().astype(int)
            # Convert color_palette[i] to a tuple and use BGR format for OpenCV
            color = tuple(color_palette[i].tolist())
            rgb_img = cv2.rectangle(
                rgb_img, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA
            )

        # 2. visualize the masks
        # create a sum of all masks with the given color palette
        num_masks = masks_img.shape[2]
        sum_mask_img = np.zeros(
            (masks_img.shape[0], masks_img.shape[1], 3), dtype=np.uint8
        )
        for i in range(num_masks):
            mask_img = masks_img[:, :, i]
            # Reshape mask to (H, W, 1) so it can broadcast with (3,) color palette
            mask_img = mask_img[..., np.newaxis]
            sum_mask_img += mask_img * color_palette[i]

        out_imgs[: rgb_img.shape[0], : rgb_img.shape[1], :] = rgb_img
        out_imgs[
            : sum_mask_img.shape[0],
            rgb_img.shape[1] : rgb_img.shape[1] + sum_mask_img.shape[1],
            :,
        ] = sum_mask_img

        cv2.imshow("out_imgs", out_imgs)
        print(f"label: {labels.numpy()}")

        print("Press any key to continue...")
        key = cv2.waitKey(0)
        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()
