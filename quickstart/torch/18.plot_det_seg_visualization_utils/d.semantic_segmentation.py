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
Semantic segmentation with FCN-ResNet50 and mask visualization utilities.

Source (adapted from):
https://docs.pytorch.org/vision/main/auto_examples/others/plot_visualization_utils.html#sphx-glr-auto-examples-others-plot-visualization-utils-py
"""

from __future__ import annotations
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Iterable

from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision.ops import masks_to_boxes

from data import load_dog_images
from dlplay.viz.plotting import show_images
from dlplay.segmentation.img_segmentation import get_boxes_from_masks


class SemanticSegmentation:
    """
    Fully-Convolutional Network model with a ResNet-50 backbone from the Fully Convolutional Networks for Semantic Segmentation paper.

    The output of the segmentation model is a tensor of shape (batch_size, num_classes, H, W).
    Each value is a non-normalized score (logits), and we can normalize them into [0, 1] by using a softmax.
    After the softmax, we can interpret each value as a probability indicating how likely
    a given pixel is to belong to a given class.
    """

    def __init__(self):
        self.weights = FCN_ResNet50_Weights.DEFAULT
        self.model = fcn_resnet50(weights=self.weights)
        self.model = self.model.eval()
        self.transforms = self.weights.transforms(resize_size=None)
        self.categories = self.weights.meta["categories"]
        # generate a palette of colors to each class
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, (len(self.categories), 3))
        self.colors = [tuple(color) for color in self.colors]

    def run(
        self, image_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = torch.stack([self.transforms(d) for d in image_list])
        with torch.no_grad():
            output = self.model(batch)
            logits = output["out"]  # (batch_size, num_classes, H, W)

        return logits


def demo_main_classes(model: SemanticSegmentation, show_boxes: bool = False) -> None:
    """
    Overlay masks for "dog" and "boat" on the original images.
    """

    dog_list = load_dog_images()
    logits = model.run(dog_list)

    # Normalize logits to probabilities
    normalized_masks = torch.nn.functional.softmax(logits, dim=1)
    categories = model.categories
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(categories)}

    # Show per-class probabilities for "dog" and "boat"
    dog_and_boat_masks = [
        normalized_masks[img_idx, sem_class_to_idx[cls]]
        for img_idx in range(len(dog_list))
        for cls in ("dog", "boat")
    ]
    show_images(
        dog_and_boat_masks,
        title="dog and boat masks (per-class probabilities)",
        subtitles=[cls for img_idx in range(len(dog_list)) for cls in ("dog", "boat")],
        plt_show=False,
    )

    # Boolean dog masks (argmax over classes equals "dog")
    # NOTE:
    # dim 0: batch size (multiple images)
    # dim 1: number of classes (e.g., dog, boat, person, etc.)
    # dim 2: image height
    # dim 3: image width

    class_dim = 1  # indicates the class dimension is at index 1
    boolean_dog_masks = normalized_masks.argmax(class_dim) == sem_class_to_idx["dog"]
    show_images(
        [m.float() for m in boolean_dog_masks],
        title="boolean dog masks (argmax over classes equals 'dog')",
        subtitles=["dog"] * len(boolean_dog_masks),
        plt_show=False,
    )

    # Overlay boolean masks on original images
    # NOTE:
    # The draw_segmentation_masks() function can be used to plots those masks on top of the original image.
    # This function expects the masks to be boolean masks, but our masks above contain probabilities in [0, 1].
    # To get boolean masks, we can do the following:
    dogs_with_masks = [
        draw_segmentation_masks(img, masks=mask, alpha=0.7, colors="yellow")
        for img, mask in zip(dog_list, boolean_dog_masks)
    ]
    show_images(
        dogs_with_masks,
        title="dogs with masks (boolean dog masks overlayed)",
        subtitles=["dog"] * len(dog_list),
        plt_show=False,
    )


def demo_all_classes_overlay(model: SemanticSegmentation) -> None:
    """
    Overlay all classes masks on the original images.
    """

    # NOTE:
    # We can plot more than one mask per image!
    # Remember that the model returned as many masks as there are classes.
    # Let’s ask the same query as above, but this time for all classes, not just the dog class:
    # “For each pixel and each class C, is class C the most likely class?”

    dog_list = load_dog_images()
    dog1_int = dog_list[0]
    logits = model.run(dog_list)
    normalized_masks = torch.nn.functional.softmax(logits, dim=1)

    all_boxes, all_scores, all_labels = get_boxes_from_masks(normalized_masks)

    # as explained above the model output has shape (batch_size, num_classes, H, W)
    num_classes = normalized_masks.shape[1]

    print(f"number of classes: {num_classes}")
    print(f"model categories: {model.categories}")
    assert num_classes == len(model.categories)

    # Single image version (dog1)
    if False:
        dog1_masks = normalized_masks[0]  # select the first image in the batch

        # now the class dimension is at index 0 since we extracted the first image in the batch and
        # removed the batch dimension
        class_dim = 0

        # get the boolean mask for each class (e.g., dog, boat, person, etc.)
        dog1_all_classes_masks = (
            dog1_masks.argmax(class_dim) == torch.arange(num_classes)[:, None, None]
        )

        dog1_with_all_masks = draw_segmentation_masks(
            dog1_int, masks=dog1_all_classes_masks, alpha=0.6
        )

        show_images(
            dog1_with_all_masks,
            title="dog1 with all classes masks (overlayed)",
            plt_show=False,
        )

    # Batch version
    class_dim = 1  # now the class dimension is at index 1 since we are using the full batch output
    all_classes_masks = (
        normalized_masks.argmax(class_dim)
        == torch.arange(num_classes)[:, None, None, None]
    )
    all_classes_masks = all_classes_masks.swapaxes(0, 1)

    # First, draw segmentation masks
    dogs_with_masks = [
        draw_segmentation_masks(img, masks=mask, alpha=0.6, colors=model.colors)
        for img, mask in zip(dog_list, all_classes_masks)
    ]

    # Then, draw bounding boxes on the images with masks
    if True:
        # Draw bounding boxes on the images with masks
        # We need to map each box to its corresponding class color
        dogs_with_boxes_and_masks = []
        for img, boxes, labels, scores in zip(
            dogs_with_masks, all_boxes, all_labels, all_scores
        ):
            if len(boxes) > 0:
                # Get the color for each box based on its class label
                box_colors = [model.colors[label.item()] for label in labels]
                text_labels = [
                    f"{model.categories[label.item()]}: {score:.2f}"
                    for label, score in zip(labels, scores)
                ]
                img_with_boxes = draw_bounding_boxes(
                    img,
                    boxes=boxes,
                    width=2,
                    colors=box_colors,
                    labels=text_labels,
                )
            else:
                img_with_boxes = img
            dogs_with_boxes_and_masks.append(img_with_boxes)

    else:
        dogs_with_boxes_and_masks = dogs_with_masks

    show_images(
        dogs_with_boxes_and_masks,  # Use the final result
        title="dogs with all classes masks (overlayed)",
        plt_show=False,
    )


if __name__ == "__main__":
    model = SemanticSegmentation()
    # demo_main_classes(model)
    demo_all_classes_overlay(model)

    plt.show()
