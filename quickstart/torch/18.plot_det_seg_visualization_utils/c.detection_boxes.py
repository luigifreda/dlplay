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
Run Faster R-CNN on demo images and visualize high-score detections.

Source (adapted from):
https://docs.pytorch.org/vision/main/auto_examples/others/plot_visualization_utils.html#sphx-glr-auto-examples-others-plot-visualization-utils-py
"""

from __future__ import annotations
from typing import List, Dict

import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)

from data import load_dog_images
from dlplay.viz.plotting import show_images


# def run_model(dog_list: List[torch.Tensor]):
#     weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
#     transforms = weights.transforms()
#     images = [transforms(d) for d in dog_list]

#     model = fasterrcnn_resnet50_fpn(weights=weights, progress=False)
#     model = model.eval()

#     with torch.no_grad():
#         outputs = model(images)
#     return outputs, weights


class DetectionModel:
    """
    Faster R-CNN model with a ResNet-50 backbone from the Faster R-CNN paper.
    The output of the detection model is a dictionary with the following keys:
    - boxes: a tensor of shape (num_boxes, 4) containing the bounding boxes in [x1, y1, x2, y2] format
    - labels: a tensor of shape (num_boxes,) containing the labels of the boxes
    - scores: a tensor of shape (num_boxes,) containing the scores of the boxes
    """

    def __init__(self):
        self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=self.weights)
        self.model = self.model.eval()
        self.transforms = self.weights.transforms()
        self.categories = self.weights.meta["categories"]

    def run(self, image_list: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        images = [self.transforms(d) for d in image_list]
        with torch.no_grad():
            outputs = self.model(images)
        return outputs, self.weights


def demo(model: DetectionModel, score_threshold: float = 0.8) -> None:
    dog_list = load_dog_images()
    outputs, _ = model.run(dog_list)

    print(f"categories: {model.categories}")

    # Let’s plot the boxes detected by our model. We will only plot the boxes with
    # a score greater than a given threshold.
    dogs_with_boxes = []
    for dog_int, out in zip(dog_list, outputs):
        # Filter boxes by score threshold
        mask = out["scores"] > score_threshold
        filtered_boxes = out["boxes"][mask]
        filtered_scores = out["scores"][mask]
        filtered_labels = out["labels"][mask]

        # Create labels with scores
        labels = [
            f"{model.categories[label]}: {score:.2f}"
            for label, score in zip(filtered_labels, filtered_scores)
        ]

        dogs_with_boxes.append(
            draw_bounding_boxes(
                dog_int,
                boxes=filtered_boxes,
                labels=labels,
                width=4,
                colors="yellow",
            )
        )

    show_images(dogs_with_boxes)


if __name__ == "__main__":
    model = DetectionModel()
    demo(model)
