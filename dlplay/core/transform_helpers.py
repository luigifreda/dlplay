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
import torch
from torchvision.transforms import v2 as T

from torchvision.tv_tensors import BoundingBoxes, Mask, Image
from torchvision.transforms.v2 import functional as F  # v2 functional


# NOTE:
# Reference:
# new transform API https://docs.pytorch.org/vision/stable/transforms.html
# https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html#sphx-glr-auto-examples-transforms-plot-transforms-illustrations-py


# Helper functions for data augmentation / transformation


def get_transforms_basic(train: bool):
    """
    Basic transform that includes horizontal flip.
    """
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def get_transforms_basic_color(train: bool):
    """
    Transform that includes horizontal flip, affine, color jitter.
    """
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToTensor())
    transforms.append(
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    )  # normalize to mean=0, std=1
    return T.Compose(transforms)


def get_transforms_affine(train: bool):
    """
    Affine transform that includes horizontal flip, affine, color jitter.
    """
    ops = []
    if train:
        ops += [
            T.RandomHorizontalFlip(0.5),
            T.RandomAffine(degrees=10, translate=(0.02, 0.02), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ]
    ops += [T.ToDtype(torch.float, scale=True), T.ToPureTensor()]
    return T.Compose(ops)


def get_transforms_affine_color(train: bool):
    """
    Affine transform that includes horizontal flip, affine, color jitter.
    """
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(
            T.RandomAffine(degrees=10, translate=(0.02, 0.02), scale=(0.9, 1.1))
        )
        transforms.append(
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        )
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToTensor())
    transforms.append(
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    )  # normalize to mean=0, std=1
    return T.Compose(transforms)


def get_transforms_enhanced(train: bool):
    """
    Enhanced transform that includes color jitter, grayscale, and posterize.
    """
    ops = []
    if train:
        ops += [
            T.RandomHorizontalFlip(0.5),
            # Optional: adds safer crops that keep boxes visible
            # T.RandomIoUCrop(min_scale=0.3, max_scale=1.0, sampler_options=[0.0, 0.1, 0.3, 0.5, 0.7]),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            T.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),  # Add color jitter
            T.RandomGrayscale(p=0.1),  # Add grayscale
            T.RandomPosterize(
                bits=4, p=0.1
            ),  # Add posterize, i.e. reduce the number of bits in each color channel of the image
        ]
    # SanitizeBoundingBoxes (v2) removes/clips degenerate boxes created by crops/affines. Keep it after spatial transforms and before tensor conversion.
    ops += [
        T.SanitizeBoundingBoxes(
            min_size=1
        ),  # drops zero-area / out-of-bounds boxes and aligns labels/masks
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.ToPureTensor(),
    ]
    return T.Compose(ops)
