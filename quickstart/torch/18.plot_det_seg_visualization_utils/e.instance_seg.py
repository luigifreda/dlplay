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
Instance segmentation with Mask R-CNN and mask visualization.

Source (adapted from):
https://docs.pytorch.org/vision/main/auto_examples/others/plot_visualization_utils.html#sphx-glr-auto-examples-others-plot-visualization-utils-py
"""

from __future__ import annotations
import numpy as np
import torch
from typing import List
import matplotlib.pyplot as plt

from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes

from dlplay.viz.plotting import show_images
from dlplay.segmentation.color_segmentation import (
    make_label_palette,
    distinct_instance_colors,
    colors_from_labels,
)
from data import load_dog_images


class InstanceSegmentation:
    """
    Instance segmentation with Mask R-CNN and mask visualization.
    """

    def __init__(self):
        self.weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn(weights=self.weights)
        self.model = self.model.eval()
        self.transforms = self.weights.transforms()
        self.categories = self.weights.meta["categories"]
        self.label_palette = make_label_palette(len(self.categories), seed=42)

    def run(self, image_list: List[torch.Tensor]):
        images = [self.transforms(d) for d in image_list]

        with torch.no_grad():
            output = self.model(images)  # list of dicts
            # NOTE: output is a list of dicts, each containing:
            # - masks: (N, 1, H, W) tensor of masks where
            #           N is the number of instances,
            #           1 a single foreground probability channel per instance,
            #           H and W are the height and width of the image,
            #           the mask is a float in [0,1] (sigmoid probabilities)
            # - boxes: (N, 4) tensor of bounding boxes
            # - labels: (N,) tensor of labels
            # - scores: (N,) tensor of scores
            # - image_id: (N,) tensor of image ids (not used)
            # - area: (N,) tensor of areas (not used)
            # - iscrowd: (N,) tensor of iscrowd (not used)
        return output


def demo(
    prob_threshold: float = 0.5,
    score_threshold: float = 0.75,
    color_mode: str = "instance",  # "label" or "instance"
) -> None:
    """
    Args:
        prob_threshold: threshold for the mask probability
        score_threshold: threshold for the score
        color_mode: "label" or "instance"
    """
    dog_list = load_dog_images()  # list of 3xHxW uint8 CPU tensors
    model = InstanceSegmentation()
    outputs = model.run(dog_list)

    # Move model outputs to CPU (safer for drawing utilities)
    outputs = [{k: v.detach().cpu() for k, v in o.items()} for o in outputs]

    # palettes

    vis_images = []
    for img, out in zip(dog_list, outputs):
        # Per-image tensors
        # N = number of instances
        m = out["masks"].squeeze(
            1
        )  # (N, H, W), float probs in [0,1], , squeeze to remove instance channel dimension
        b = out["boxes"]  # (N, 4)
        l = out["labels"]  # (N,)
        s = out["scores"]  # (N,)

        # Filter instances by score
        keep = s > score_threshold
        m = m[keep] > prob_threshold  # boolean (N_kept, H, W)
        b = b[keep]
        l = l[keep]
        s = s[keep]

        if m.numel() == 0:
            vis_images.append(img)  # nothing to draw
            continue

        # Colors
        if color_mode == "label":
            inst_colors = colors_from_labels(
                l, model.label_palette
            )  # same color for same class
        else:  # "instance"
            inst_colors = distinct_instance_colors(
                len(l), seed=123
            )  # unique per instance

        # Overlay masks (colors length must == number of kept masks)
        img_masks = draw_segmentation_masks(img, masks=m, alpha=0.5, colors=inst_colors)

        # Text labels & box colors (match the same scheme used for masks)
        text_labels = [
            f"{model.categories[int(li)]}: {float(si):.2f}" for li, si in zip(l, s)
        ]
        box_colors = inst_colors  # keep consistent with the masks

        img_final = draw_bounding_boxes(
            img_masks, boxes=b, width=2, colors=box_colors, labels=text_labels
        )
        vis_images.append(img_final)

    show_images(vis_images, title="Mask R-CNN: boxes + masks", plt_show=False)
    plt.show()


if __name__ == "__main__":
    demo()
