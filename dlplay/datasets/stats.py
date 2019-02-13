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
import os
import sys
import time
import math

import torch


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2
    )
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("==> Computing mean and std..")
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def get_category_stats(dataset) -> dict[str, int]:
    """Get statistics about categories in the dataset."""
    stats = {}

    # Get category mapping from dataset if available
    category_names = None
    if hasattr(dataset, "categories"):
        category_names = dataset.categories
    elif hasattr(dataset, "id_to_category"):
        category_names = list(dataset.id_to_category.values())

    for i in range(len(dataset)):
        data = dataset[i]
        # Check if category attribute exists and is not dummy
        if hasattr(data, "category") and data.category != "dummy":
            category = data.category

            # Handle PyTorch tensors
            if hasattr(category, "item"):
                # Single value tensor
                category = category.item()
            elif hasattr(category, "numpy"):
                # Convert to numpy and get first element
                category = category.numpy().item()
            elif isinstance(category, (list, tuple)) and len(category) > 0:
                # Handle list/tuple with single element
                category = category[0]

            # Handle integer category indices
            if isinstance(category, int):
                if category_names and 0 <= category < len(category_names):
                    category = category_names[category]
                else:
                    category = f"category_{category}"  # Fallback for unknown indices
            elif isinstance(category, str) and category.isdigit():
                # Handle string representations of integers
                cat_idx = int(category)
                if category_names and 0 <= cat_idx < len(category_names):
                    category = category_names[cat_idx]
                else:
                    category = f"category_{cat_idx}"

            stats[category] = stats.get(category, 0) + 1
        else:
            # Handle dummy data or missing category
            stats["dummy"] = stats.get("dummy", 0) + 1
    return stats
