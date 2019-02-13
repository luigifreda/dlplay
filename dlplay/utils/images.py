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
import torchvision.transforms.functional as F


def square_crop_resize(img: torch.Tensor, size: int) -> torch.Tensor:
    """
    Crop and resize an image to a square of the given size.
    """
    # crop along the longer side
    # assume the input image is a CHW tensor
    h, w = img.shape[-2:]
    crop_size = min(h, w)
    if h > w:
        img = img[:, :w, :w]
    else:
        img = img[:, :h, :h]
    if crop_size != size:
        img = F.resize(img, (size, size))
    return img
