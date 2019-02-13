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
Drawing bounding boxes with torchvision.utils.draw_bounding_boxes

Source (adapted from):
https://docs.pytorch.org/vision/main/auto_examples/others/plot_visualization_utils.html#sphx-glr-auto-examples-others-plot-visualization-utils-py
"""

from __future__ import annotations
import torch
from torchvision.utils import draw_bounding_boxes

from data import load_dog_images
from dlplay.viz.plotting import show_images


def demo_basic() -> None:
    image_list = load_dog_images()
    dog1_int = image_list[0]

    boxes = torch.tensor([[50, 50, 100, 200], [210, 150, 350, 430]], dtype=torch.float)
    colors = ["blue", "yellow"]
    result = draw_bounding_boxes(dog1_int, boxes, colors=colors, width=5)
    show_images(result)


if __name__ == "__main__":
    demo_basic()
