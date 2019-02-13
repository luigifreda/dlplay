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

import os.path as osp

import torch
import torch.nn.functional as F
from dlplay.models.space3d.point_transformer_classifier import (
    TransformerBlock,
    TransitionDown,
)
from torchmetrics.functional import jaccard_index

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from dlplay.datasets.shapenet import ShapeNet
from dlplay.datasets.point3d_datasets_helpers import PointDatasetType

from dlplay.models.space3d.pointnet2_segmenter import PointNet2Segmenter
from dlplay.models.space3d.point_transformer_segmenter import PointTransformerSegmenter

from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, knn_graph, knn_interpolate
from torch_geometric.typing import WITH_TORCH_CLUSTER
from torch_geometric.utils import scatter

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")


def model_prediction(model: torch.nn.Module, data):
    if isinstance(model, PointNet2Segmenter):
        return model(data)
    elif isinstance(model, PointTransformerSegmenter):
        return model(data.x, data.pos, data.batch)
    else:
        return model(data)


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_accumulation_steps: int = 1,
    scaler: torch.amp.GradScaler = None,
    print_every: int = 10,
):
    """
    Train the Point Segmenter.
    """
    model.train()

    global_loss = 0
    global_correct_nodes = 0
    global_total_nodes = 0

    total_loss = correct_nodes = total_nodes = 0
    for i, data in enumerate(train_loader):
        data = data.to(device)

        # Use mixed precision if scaler is provided
        if scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                out = model_prediction(model, data)
        else:
            out = model_prediction(model, data)

        # Handle ignore labels (-1) for ScanNet dataset
        valid_mask = data.y >= 0

        if valid_mask.sum() > 0:
            valid_out = out[valid_mask]
            valid_y = data.y[valid_mask].long()

            loss = F.nll_loss(valid_out, valid_y)
            loss = loss / gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            correct_nodes += valid_out.argmax(dim=1).eq(valid_y).sum().item()
            total_nodes += valid_mask.sum().item()
        else:
            continue

        if (i + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item() * gradient_accumulation_steps
        else:
            total_loss += loss.item() * gradient_accumulation_steps

        global_loss += loss.item() * gradient_accumulation_steps
        global_correct_nodes += correct_nodes
        global_total_nodes += total_nodes

        if (i + 1) % print_every == 0:
            print(
                f"[{i+1}/{len(train_loader)}] Loss: {total_loss / print_every:.4f} "
                f"Train Acc: {correct_nodes / total_nodes:.4f}"
            )
            total_loss = correct_nodes = total_nodes = 0

    return global_loss / len(train_loader), global_correct_nodes / global_total_nodes


@torch.no_grad()
def test(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dataset_type: PointDatasetType,
):
    """
    Test the Point Segmenter.
    """
    model.eval()

    ious, categories = [], []
    for data in loader:
        data = data.to(device)
        outs = model_prediction(model, data)

        sizes = (
            data.ptr[1:] - data.ptr[:-1]
        ).tolist()  # calculate the number of nodes in each point cloud of the batch

        for out, y in zip(
            outs.split(sizes),  # split the output of each point cloud of the batch
            data.y.split(
                sizes
            ),  # split the ground truth of each point cloud of the batch
        ):
            # Handle ignore labels (-1) for ScanNet dataset
            valid_mask = y >= 0

            if valid_mask.sum() > 0:
                valid_out = out[valid_mask]
                valid_y = y[valid_mask]

                # Check if this is ShapeNet (has category attribute) or ScanNet
                # if hasattr(data, "category") and data.category is not None:
                if dataset_type == PointDatasetType.ShapeNet:
                    # ShapeNet: category-based evaluation with local part mapping
                    category = data.category.tolist()
                    for i, (out_single, y_single, cat) in enumerate(
                        zip(valid_out.split(1), valid_y.split(1), category)
                    ):
                        # Get category name from index
                        category_names = list(ShapeNet.seg_classes.keys())
                        category_name = category_names[cat]
                        part_labels = ShapeNet.seg_classes[category_name]
                        part_labels = torch.tensor(part_labels, device=device)

                        # Create mapping from global labels to local part labels
                        y_map = torch.zeros(
                            loader.dataset.num_classes, device=device, dtype=torch.long
                        )
                        y_map[part_labels] = torch.arange(
                            len(part_labels), device=device
                        )

                        # Map ground truth labels to local part indices
                        y_local = y_map[y_single.squeeze()]

                        # Get predictions for the relevant parts only
                        out_parts = out_single.squeeze()[part_labels]

                        iou = jaccard_index(
                            out_parts.argmax(dim=-1),
                            y_local,
                            num_classes=len(part_labels),
                            task="multiclass",
                        )
                        ious.append(iou)
                        categories.append(cat)
                else:
                    # ScanNet: direct global semantic segmentation evaluation
                    iou = jaccard_index(
                        valid_out.argmax(dim=-1),
                        valid_y,
                        num_classes=loader.dataset.num_classes,
                        task="multiclass",
                    )
                    ious.append(iou)
                    categories.append(0)  # Use 0 as default category for ScanNet

    iou = torch.tensor(ious, device=device)
    category = torch.tensor(categories, device=device)

    if (
        hasattr(loader.dataset, "seg_classes")
        and loader.dataset.seg_classes is not None
    ):
        # ShapeNet: Per-category IoU
        mean_iou = scatter(iou, category, reduce="mean")
        return float(mean_iou.mean())  # Global IoU
    else:
        # ScanNet: Direct mean IoU
        return float(iou.mean())
