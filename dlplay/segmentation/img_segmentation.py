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
from typing import Iterable, Tuple, List, Sequence
import numpy as np
import colorsys
import torch
from torchvision.ops import masks_to_boxes


def get_boxes_from_masks(
    masks: torch.Tensor,
    scores: torch.Tensor | None = None,
    prob_thresh: float = 0.5,
    min_area: int = 25,
    exclude_labels: Iterable[int] = (0,),  # skip background by default
    score_reduce: str = "mean",  # "mean" | "max" | "median"
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Convert masks (bool or probabilities) to bounding boxes, scores, and labels.

    Args:
        masks: (B, C, H, W) or (C, H, W). Bool or probs in [0, 1].
        scores: (B, C) or (C,), optional per-class scores (overrides pixel-prob scoring).
        prob_thresh: threshold used if masks are probabilities.
        min_area: minimum number of True pixels to keep a box.
        exclude_labels: class indices to skip (default: (0,) background).
        score_reduce: how to aggregate per-pixel probabilities inside the mask.

    Returns:
        boxes_per_image:  list len B, tensors (Ni, 4)
        scores_per_image: list len B, tensors (Ni,)
        labels_per_image: list len B, tensors (Ni,)
    """
    # Ensure batch dimension
    if masks.dim() == 3:
        masks = masks.unsqueeze(0)
        if scores is not None and scores.dim() == 1:
            scores = scores.unsqueeze(0)

    device = masks.device

    # Preserve per-pixel probabilities if provided
    if masks.dtype == torch.bool:
        pixel_probs = None
        bin_masks = masks
    else:
        pixel_probs = masks.clamp(0, 1).float()  # (B, C, H, W)
        bin_masks = masks > prob_thresh

    B, C, H, W = bin_masks.shape

    if scores is not None:
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        assert scores.shape[:2] == (B, C), "scores must match (B, C)"

    skip = set(int(x) for x in exclude_labels)

    all_boxes: list[list[torch.Tensor]] = [[] for _ in range(B)]
    all_scores: list[list[torch.Tensor]] = [[] for _ in range(B)]
    all_labels: list[list[torch.Tensor]] = [[] for _ in range(B)]

    for b in range(B):
        for c in range(C):
            if c in skip:
                continue

            mask = bin_masks[b, c]
            area = int(mask.sum().item())
            if area < min_area:
                continue

            # Box
            box = masks_to_boxes(mask.unsqueeze(0).to(torch.uint8)).squeeze(0)  # (4,)

            # Score
            if scores is not None:
                score_val = scores[b, c]
            elif pixel_probs is not None:
                vals = pixel_probs[b, c][mask]  # per-pixel probs within the kept region
                if score_reduce == "max":
                    score_val = vals.max()
                elif score_reduce == "median":
                    score_val = vals.median()
                else:  # "mean"
                    score_val = vals.mean()
            else:
                # Fallback: fraction of image covered (size proxy, not confidence)
                score_val = mask.float().mean()

            all_boxes[b].append(box.to(device=device, dtype=torch.float32))
            all_scores[b].append(
                score_val
                if isinstance(score_val, torch.Tensor)
                else torch.as_tensor(score_val, device=device, dtype=torch.float32)
            )
            all_labels[b].append(torch.tensor(c, device=device, dtype=torch.long))

    boxes_per_image = [
        (
            torch.stack(b)
            if b
            else torch.empty((0, 4), device=device, dtype=torch.float32)
        )
        for b in all_boxes
    ]
    scores_per_image = [
        (torch.stack(s) if s else torch.empty((0,), device=device, dtype=torch.float32))
        for s in all_scores
    ]
    labels_per_image = [
        (torch.stack(l) if l else torch.empty((0,), device=device, dtype=torch.long))
        for l in all_labels
    ]
    return boxes_per_image, scores_per_image, labels_per_image
