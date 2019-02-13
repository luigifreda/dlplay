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
"""
Data helpers for demo images used by the visualization examples.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

from torchvision.io import decode_image
import torch

from dlplay.paths import DATA_DIR
from dlplay.utils.images import square_crop_resize

_DEMO_DIR = Path(DATA_DIR) / "torch_data"


def load_dog_images() -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    Returns:
        dog1_int, dog2_int, dog_list (uint8 CHW tensors)
    """
    dog1_int = decode_image(str(_DEMO_DIR / "dog1.jpg"))
    dog2_int = decode_image(str(_DEMO_DIR / "dog2.jpg"))
    dogs_sheeps_man_int = decode_image(str(_DEMO_DIR / "dogs_sheeps_man.jpg"))

    min_width = min(dog1_int.shape[2], dog2_int.shape[2], dogs_sheeps_man_int.shape[2])
    min_height = min(dog1_int.shape[1], dog2_int.shape[1], dogs_sheeps_man_int.shape[1])
    min_size = min(min_width, min_height)

    image_list = [
        square_crop_resize(img, min_size)
        for img in [dog1_int, dog2_int, dogs_sheeps_man_int]
    ]
    return image_list


def load_person_image() -> torch.Tensor:
    """Return the 'person1.jpg' demo image as a uint8 CHW tensor."""
    return decode_image(str(_DEMO_DIR / "person1.jpg"))
