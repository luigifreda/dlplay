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


from collections import OrderedDict


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


def conv3x3(in_channels, out_channels, stride=1, groups=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,
        bias=bias,
    )


def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        groups=groups,
        bias=bias,
    )


def _make_divisible(v, divisor):
    return int(math.ceil(v / divisor) * divisor)


class ShuffleUnit(nn.Module):
    def __init__(
        self, in_channels, out_channels, groups=3, grouped_conv=True, combine="add"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.combine = combine

        if combine == "add":
            self.depthwise_stride = 1
            self._combine_func = self._add
        elif combine == "concat":
            self.depthwise_stride = 2
            self._combine_func = self._concat
            self.out_channels -= self.in_channels
            if self.out_channels <= 0:
                raise ValueError("For 'concat', out_channels must be > in_channels.")
        else:
            raise ValueError(f'Invalid combine="{combine}"')

        # bottleneck rounded up to a multiple of groups
        self.bottleneck_channels = _make_divisible(
            max(1, self.out_channels // 4), self.groups
        )

        self.first_1x1_groups = self.groups if grouped_conv else 1

        # Compress 1x1
        self.g_conv_1x1_compress = self._make_grouped_conv1x1(
            in_channels=self.in_channels,
            out_channels=self.bottleneck_channels,
            groups=self.first_1x1_groups,
            batch_norm=True,
            relu=True,
        )

        # Depthwise 3x3
        self.depthwise_conv3x3 = conv3x3(
            self.bottleneck_channels,
            self.bottleneck_channels,
            stride=self.depthwise_stride,
            groups=self.bottleneck_channels,
        )
        self.bn_after_depthwise = nn.BatchNorm2d(self.bottleneck_channels)

        # Expand 1x1
        self.g_conv_1x1_expand = self._make_grouped_conv1x1(
            self.bottleneck_channels,
            self.out_channels,
            self.groups,
            batch_norm=True,
            relu=False,
        )

        # --- NEW: projection for add when channels differ ---
        self.need_proj = self.combine == "add" and self.in_channels != self.out_channels
        if self.need_proj:
            self.proj = nn.Sequential(
                conv1x1(self.in_channels, self.out_channels, groups=1, bias=False),
                nn.BatchNorm2d(self.out_channels),
            )

    @staticmethod
    def _add(x, out):
        # residual connection
        return x + out

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def _make_grouped_conv1x1(
        self, in_channels, out_channels, groups, batch_norm=True, relu=False
    ):

        modules = OrderedDict()
        conv = conv1x1(in_channels, out_channels, groups=groups)
        modules["conv1x1"] = conv

        if batch_norm:
            modules["batch_norm"] = nn.BatchNorm2d(out_channels)
        if relu:
            modules["relu"] = nn.ReLU()
        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv

    def forward(self, x):
        residual = x
        if self.combine == "concat":
            residual = F.avg_pool2d(residual, kernel_size=3, stride=2, padding=1)
        elif self.need_proj:
            residual = self.proj(residual)

        out = self.g_conv_1x1_compress(x)
        out = channel_shuffle(out, self.groups)
        out = self.depthwise_conv3x3(out)
        out = self.bn_after_depthwise(out)
        out = self.g_conv_1x1_expand(out)

        out = self._combine_func(residual, out)
        return F.relu(out)


class ShuffleUnitEinops(nn.Module):
    def __init__(
        self, in_channels, out_channels, groups=3, grouped_conv=True, combine="add"
    ):
        super().__init__()
        first_1x1_groups = groups if grouped_conv else 1

        # determine right-branch output channels depending on combine mode
        if combine == "add":
            right_out_channels = out_channels
            depthwise_stride = 1
            # projection needed if channels differ
            need_proj = in_channels != out_channels
            if need_proj:
                self.left = nn.Sequential(
                    conv1x1(in_channels, out_channels, groups=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.left = Rearrange("...->...")  # identity
        elif combine == "concat":
            depthwise_stride = 2
            self.left = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            # right branch must produce only the extra channels
            right_out_channels = out_channels - in_channels
            if right_out_channels <= 0:
                raise ValueError("For 'concat', out_channels must be > in_channels.")
        else:
            raise ValueError(f'Invalid combine="{combine}"')

        # bottleneck rounded up to a multiple of groups
        bottleneck_channels = _make_divisible(max(1, right_out_channels // 4), groups)

        self.combine = combine
        self.right = nn.Sequential(
            # 1x1 reduce
            conv1x1(in_channels, bottleneck_channels, groups=first_1x1_groups),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            # channel shuffle
            Rearrange("b (c1 c2) h w -> b (c2 c1) h w", c1=groups),
            # 3x3 depthwise
            conv3x3(
                bottleneck_channels,
                bottleneck_channels,
                stride=depthwise_stride,
                groups=bottleneck_channels,
            ),
            nn.BatchNorm2d(bottleneck_channels),
            # 1x1 expand to right_out_channels
            conv1x1(bottleneck_channels, right_out_channels, groups=groups),
            nn.BatchNorm2d(right_out_channels),
        )

    def forward(self, x):
        if self.combine == "add":
            combined = self.left(x) + self.right(x)
        else:  # "concat"
            combined = torch.cat([self.left(x), self.right(x)], dim=1)
        return F.relu(combined, inplace=True)


if __name__ == "__main__":
    model = ShuffleUnitEinops(
        in_channels=3, out_channels=21, groups=3, grouped_conv=True, combine="add"
    )
    print(model)
    x = torch.randn(1, 3, 28, 28)
    print(model(x).shape)
