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
Keypoint visualization with Keypoint R-CNN and custom connectivity.

Source (adapted from):
https://docs.pytorch.org/vision/main/auto_examples/others/plot_visualization_utils.html
"""

from __future__ import annotations
from typing import List, Tuple
import torch
import matplotlib.pyplot as plt

from torchvision.models.detection import (
    keypointrcnn_resnet50_fpn,
    KeypointRCNN_ResNet50_FPN_Weights,
)
from torchvision.utils import draw_keypoints

from dlplay.viz.plotting import show_images
from data import load_person_image

# ---- COCO keypoint names (order used by torchvision models) ----
COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# ---- Skeleton connectivity (pairs of keypoint indices) ----
# This is a reasonable subset; you can add cross-links like (5,6) shoulders or (11,12) hips if you want.
CONNECT_SKELETON: List[Tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
    # Optional extras often used in COCO-style visualizations:
    # (5, 6),   # left_shoulder <-> right_shoulder
    # (11, 12), # left_hip <-> right_hip
]


class BodyKeypoints:
    """Thin wrapper around torchvision's Keypoint R-CNN with the default weights."""

    def __init__(self):
        self.weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = keypointrcnn_resnet50_fpn(weights=self.weights).eval()
        # The weights specify the preprocessing pipeline (resize/normalize) expected by the model
        self.transforms = self.weights.transforms()

    def run(self, image_list: List[torch.Tensor]):
        """
        Args:
            image_list: list of CPU uint8 tensors (C, H, W) in [0, 255].
        Returns:
            outputs: list[dict] with keys: 'boxes', 'labels', 'scores', 'keypoints', 'keypoints_scores' (if available)
        """
        images = [self.transforms(d) for d in image_list]  # to float, normalized, etc.
        with torch.no_grad():
            outputs = self.model(images)  # one dict per input image
        # NOTE:
        # As we see the output contains a list of dictionaries.
        # The output list is of length batch_size.
        # Each entry in the list corresponds to an input image, and it is a dict
        # with keys boxes, labels, scores, keypoints and keypoint_scores.
        # Each value associated to those keys has num_instances elements in it.
        return outputs


def demo_detect_and_draw(detect_threshold: float = 0.75) -> None:
    """
    Detect people and draw keypoints first (points only), then with skeleton connectivity.

    detect_threshold filters *instances* by their detection scores (outputs[i]['scores']).
    """
    person_int = load_person_image()  # expected uint8 (C,H,W) CPU tensor
    model = BodyKeypoints()
    outputs = model.run([person_int])

    # Move to CPU for drawing utilities (draw_* expects CPU tensors)
    outputs = [
        {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in o.items()}
        for o in outputs
    ]

    # ---- Extract detections for the single image ----
    kpts = outputs[0]["keypoints"]  # (N, K, 3) -> (x, y, visibility) per keypoint
    det_scores = outputs[0]["scores"]  # (N,)

    # Guard for empty results (avoid indexing errors)
    if kpts.numel() == 0:
        show_images([person_int], title="No detections", plt_show=False)
        plt.show()
        return

    # NOTE:
    # The KeypointRCNN model detects there are two instances in the image.
    # If you plot the boxes by using draw_bounding_boxes() you would recognize they are the person and the surfboard.
    # If we look at the scores, we will realize that the model is much more confident about the person than surfboard.
    # We could now set a threshold confidence and plot instances which we are confident enough.
    # Let us set a threshold of 0.75 and filter out the keypoints corresponding to the person.

    # Boolean mask of instances to keep
    keep = det_scores > detect_threshold
    if keep.sum() == 0:
        show_images(
            [person_int],
            title=f"No detections above {detect_threshold:.2f}",
            plt_show=False,
        )
        plt.show()
        return

    # Slice to the kept instances
    keypoints_kept = kpts[keep]  # (N_kept, K, 3)
    coords, visibility = keypoints_kept.split([2, 1], dim=-1)
    coords = coords  # (N_kept, K, 2)
    visibility = visibility.bool().squeeze(-1)  # (N_kept, K), True means "visible"

    # ---- 1) Draw only keypoints (points) ----
    # If you pass visibility, draw_keypoints will hide keypoints where visibility is False.
    img_points = draw_keypoints(
        person_int,
        coords,  # (N, K, 2)
        visibility=visibility,  # (N, K) bool
        colors="blue",  # or a list of colors (one per instance)
        radius=3,
    )
    show_images([img_points], title="Keypoints", plt_show=False)

    # ---- 2) Draw with skeleton connectivity ----
    # What if we are interested in joining the keypoints? This is especially useful in creating
    # pose detection or action recognition. We can join the keypoints easily using the connectivity parameter.
    # A close observation would reveal that we would need to join the points in below order
    # to construct human skeleton.
    img_skel = draw_keypoints(
        person_int,
        coords,
        visibility=visibility,  # hide undetected/invisible keypoints and the lines connected to them
        connectivity=CONNECT_SKELETON,  # list of (i, j) keypoint index pairs
        colors="blue",
        radius=4,
        width=3,
    )
    show_images([img_skel], title="Keypoints + Skeleton", plt_show=False)


def demo_visibility_example() -> None:
    """
    Demonstrate using explicit visibility to hide undetected keypoints.

    `prediction` contains a single instance (N=1) with 17 keypoints (K=17).
    Last channel is 0/1 visibility. Coordinates are (x, y) in image space.
    """
    person_int = load_person_image()

    prediction = torch.tensor(
        [
            [
                [208.0176, 214.2409, 1.0],
                [0.0000, 0.0000, 0.0],
                [197.8246, 210.6392, 1.0],
                [0.0000, 0.0000, 0.0],
                [178.6378, 217.8425, 1.0],
                [221.2086, 253.8591, 1.0],
                [160.6502, 269.4662, 1.0],
                [243.9929, 304.2822, 1.0],
                [138.4654, 328.8935, 1.0],
                [277.5698, 340.8990, 1.0],
                [153.4551, 374.5145, 1.0],
                [0.0000, 0.0000, 0.0],
                [226.0053, 370.3125, 1.0],
                [221.8081, 455.5516, 1.0],
                [273.9723, 448.9486, 1.0],
                [193.6275, 546.1933, 1.0],
                [273.3727, 545.5930, 1.0],
            ]
        ],
        dtype=torch.float32,
    )
    img1 = draw_keypoints(
        person_int,
        prediction,
        connectivity=CONNECT_SKELETON,
        colors="blue",
        radius=4,
        width=3,
    )
    show_images([img1], title="Custom visibility + skeleton", plt_show=False)
    # NOTE:
    # What happened there? The model, which predicted the new keypoints, can’t detect the three points
    # that are hidden on the upper left body of the skateboarder. More precisely,
    # the model predicted that (x, y, vis) = (0, 0, 0) for the left_eye, left_ear, and left_hip.
    # So we definitely don’t want to display those keypoints and connections, and you don’t have to.
    # Looking at the parameters of draw_keypoints(), we can see that we can pass a visibility
    # tensor as an additional argument. Given the models’ prediction, we have the visibility
    # as the third keypoint dimension, we just need to extract it. Let’s split the prediction
    # into the keypoint coordinates and their respective visibility, and pass both of them
    # as arguments to draw_keypoints().

    coords, vis = prediction.split([2, 1], dim=-1)
    vis = vis.bool().squeeze(-1)  # (1, 17)

    img2 = draw_keypoints(
        person_int,
        coords,  # (1, 17, 2)
        visibility=vis,  # (1, 17)
        connectivity=CONNECT_SKELETON,
        colors="blue",
        radius=4,
        width=3,
    )
    show_images([img2], title="Custom visibility + skeleton 2", plt_show=False)
    # NOTE:
    # We can see that the undetected keypoints are not draw and the invisible keypoint connections were skipped.
    # This can reduce the noise on images with multiple detections, or in cases like ours,
    # when the keypoint-prediction model missed some detections.
    # Most torch keypoint-prediction models return the visibility for every prediction, ready for you to use it.
    # The keypointrcnn_resnet50_fpn() model, which we used in the first case, does so too.


if __name__ == "__main__":
    demo_detect_and_draw()
    demo_visibility_example()
    plt.show()
