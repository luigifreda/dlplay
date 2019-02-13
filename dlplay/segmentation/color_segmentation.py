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


def make_label_palette(num_classes: int, seed: int = 0) -> List[Tuple[int, int, int]]:
    """Deterministic per-class colors."""
    rng = np.random.default_rng(seed)
    cols = rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)
    return [tuple(map(int, c)) for c in cols]


def distinct_instance_colors(n: int, seed: int = 0) -> List[Tuple[int, int, int]]:
    """N visually distinct colors via HSV wheel."""
    rng = np.random.default_rng(seed)
    hues = np.linspace(0, 1, num=max(n, 1), endpoint=False)
    rng.shuffle(hues)
    out = []
    for h in hues[:n]:
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 1.0)
        out.append((int(255 * r), int(255 * g), int(255 * b)))
    return out


def colors_from_labels(
    labels: torch.Tensor, class_palette: Sequence[Tuple[int, int, int]]
) -> List[Tuple[int, int, int]]:
    """Map each instance to its class color."""
    return [class_palette[int(l)] for l in labels.tolist()]


# ======================================================
# Color Segmentation
# ======================================================

# 60 qualitative colors: Tab20 + Tab20b + Tab20c (hex)
# Source palettes are widely used and reasonably color-blind friendly.
_TAB60_HEX = [
    # tab20 (20)
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
    # tab20b (20)
    "#393b79",
    "#5254a3",
    "#6b6ecf",
    "#9c9ede",
    "#637939",
    "#8ca252",
    "#b5cf6b",
    "#cedb9c",
    "#8c6d31",
    "#bd9e39",
    "#e7ba52",
    "#e7cb94",
    "#843c39",
    "#ad494a",
    "#d6616b",
    "#e7969c",
    "#7b4173",
    "#a55194",
    "#ce6dbd",
    "#de9ed6",
    # tab20c (20)
    "#3182bd",
    "#6baed6",
    "#9ecae1",
    "#c6dbef",
    "#e6550d",
    "#fd8d3c",
    "#fdae6b",
    "#fdd0a2",
    "#31a354",
    "#74c476",
    "#a1d99b",
    "#c7e9c0",
    "#756bb1",
    "#9e9ac8",
    "#bcbddc",
    "#dadaeb",
    "#636363",
    "#969696",
    "#bdbdbd",
    "#d9d9d9",
]


def _hex_to_rgb01(hex_str: str):
    hex_str = hex_str.lstrip("#")
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    return r, g, b


# Precompute as a CPU float32 tensor (we'll move/cast to the right device/dtype on use)
_TAB60_RGB = torch.tensor([_hex_to_rgb01(h) for h in _TAB60_HEX], dtype=torch.float32)


def segment_labels_to_colors(
    segment_labels: torch.Tensor, num_classes: int = 50
) -> torch.Tensor:
    """
    Map integer labels -> reproducible, high-contrast RGB colors in [0,1].

    Args:
        segment_labels: (N,) int tensor of labels on any device.
        num_classes: total number of classes (used only to warn/validate).

    Returns:
        colors: (N,3) float tensor on same device as segment_labels.
    """
    if segment_labels.numel() == 0:
        return torch.empty((0, 3), dtype=torch.float32, device=segment_labels.device)

    labels = segment_labels.to(torch.long).clamp(min=0)
    device = segment_labels.device

    # Build a color table of shape (K,3), where K >= max_label+1.
    max_label = int(labels.max().item())
    k = max(max_label + 1, num_classes)

    # Tile/cycle the 60-color palette if needed
    reps = (k + len(_TAB60_RGB) - 1) // len(_TAB60_RGB)
    color_table = _TAB60_RGB.repeat(reps, 1)[:k, :]  # (k,3)
    color_table = color_table.to(device=device)  # keep float32

    # Optional: reserve color 0 for "unlabeled" or background by forcing a neutral gray
    # if you have such a convention. Comment out if not needed.
    # color_table[0] = torch.tensor([0.6, 0.6, 0.6], device=device)

    colors = color_table[labels]
    return colors  # float32 [0,1]
