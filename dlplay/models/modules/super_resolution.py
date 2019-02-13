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
# start from importing some stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from einops import rearrange, reduce, asnumpy, parse_shape
from einops.layers.torch import Rearrange, Reduce


class SuperResolutionNetOld(nn.Module):
    def __init__(self, upscale_factor):
        super(SuperResolutionNetOld, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(
            1, 64, (5, 5), (1, 1), (2, 2)
        )  # 1 input channel, 64 output channels, 5x5 kernel, 1 stride, 2 padding
        self.conv2 = nn.Conv2d(
            64, 64, (3, 3), (1, 1), (1, 1)
        )  # 64 input channels, 64 output channels, 3x3 kernel, 1 stride, 1 padding
        self.conv3 = nn.Conv2d(
            64, 32, (3, 3), (1, 1), (1, 1)
        )  # 64 input channels, 32 output channels, 3x3 kernel, 1 stride, 1 padding
        self.conv4 = nn.Conv2d(
            32, upscale_factor**2, (3, 3), (1, 1), (1, 1)
        )  # 32 input channels, upscale_factor**2 output channels, 3x3 kernel, 1 stride, 1 padding
        self.pixel_shuffle = nn.PixelShuffle(
            upscale_factor
        )  # upscale_factor is the factor by which the image is upsampled

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x


class SuperResolutionNetEinops(nn.Module):
    """
    A new super resolution network using einops.

    Here is the difference:
    - no need in special instruction pixel_shuffle (and result is transferrable between frameworks)
    - output doesn't contain a fake axis (and we could do the same for the input)
    - inplace ReLU used now, for high resolution pictures that becomes critical and saves us much memory
    - and all the benefits of nn.Sequential again
    """

    def __init__(self, upscale_factor):
        super(SuperResolutionNetEinops, self).__init__()
        self.upscale_factor = upscale_factor
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, upscale_factor**2, kernel_size=3, padding=1),
            Rearrange(
                "b (h2 w2) h w -> b (h h2) (w w2)", h2=upscale_factor, w2=upscale_factor
            ),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # test the model
    model = SuperResolutionNetEinops(upscale_factor=2)
    print(model)
    x = torch.randn(1, 1, 28, 28)
    print(model(x).shape)
