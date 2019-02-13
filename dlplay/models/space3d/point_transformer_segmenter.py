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

from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, knn_graph, knn_interpolate
from torch_geometric.typing import WITH_TORCH_CLUSTER
from torch_geometric.utils import scatter

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")


class TransitionUp(torch.nn.Module):
    """Reduce features dimensionality and interpolate back to higher
    resolution and cardinality.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp_sub = MLP([in_channels, out_channels], plain_last=False)
        self.mlp = MLP([out_channels, out_channels], plain_last=False)

    def forward(self, x, x_sub, pos, pos_sub, batch=None, batch_sub=None):
        # transform low-res features and reduce the number of features
        x_sub = self.mlp_sub(x_sub)

        # interpolate low-res feats to high-res points
        x_interpolated = knn_interpolate(
            x_sub, pos_sub, pos, k=3, batch_x=batch_sub, batch_y=batch
        )

        x = self.mlp(x) + x_interpolated

        return x


class PointTransformerSegmenter(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim_model, k=16):
        super().__init__()
        self.k = k

        # dummy feature is created if there is none given
        in_channels = max(in_channels, 1)

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]], plain_last=False)

        self.transformer_input = TransformerBlock(
            in_channels=dim_model[0],
            out_channels=dim_model[0],
        )

        # backbone layers
        self.transformers_up = torch.nn.ModuleList()
        self.transformers_down = torch.nn.ModuleList()
        self.transition_up = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        for i in range(0, len(dim_model) - 1):

            # Add Transition Down block followed by a Point Transformer block
            self.transition_down.append(
                TransitionDown(
                    in_channels=dim_model[i], out_channels=dim_model[i + 1], k=self.k
                )
            )

            self.transformers_down.append(
                TransformerBlock(
                    in_channels=dim_model[i + 1], out_channels=dim_model[i + 1]
                )
            )

            # Add Transition Up block followed by Point Transformer block
            self.transition_up.append(
                TransitionUp(in_channels=dim_model[i + 1], out_channels=dim_model[i])
            )

            self.transformers_up.append(
                TransformerBlock(in_channels=dim_model[i], out_channels=dim_model[i])
            )

        # summit layers
        self.mlp_summit = MLP(
            [dim_model[-1], dim_model[-1]], norm=None, plain_last=False
        )

        self.transformer_summit = TransformerBlock(
            in_channels=dim_model[-1],
            out_channels=dim_model[-1],
        )

        # class score computation
        self.mlp_output = MLP([dim_model[0], 64, out_channels], norm=None)

    def forward(self, x, pos, batch=None):

        # add dummy features in case there is none
        if x is None:
            x = torch.ones((pos.shape[0], 1)).to(pos.get_device())

        out_x = []
        out_pos = []
        out_batch = []

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # save outputs for skipping connections
        out_x.append(x)
        out_pos.append(pos)
        out_batch.append(batch)

        # backbone down : #reduce cardinality and augment dimensionnality
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)
            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

            out_x.append(x)
            out_pos.append(pos)
            out_batch.append(batch)

        # summit
        x = self.mlp_summit(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_summit(x, pos, edge_index)

        # backbone up : augment cardinality and reduce dimensionnality
        n = len(self.transformers_down)
        for i in range(n):
            x = self.transition_up[-i - 1](
                x=out_x[-i - 2],
                x_sub=x,
                pos=out_pos[-i - 2],
                pos_sub=out_pos[-i - 1],
                batch_sub=out_batch[-i - 1],
                batch=out_batch[-i - 2],
            )

            edge_index = knn_graph(out_pos[-i - 2], k=self.k, batch=out_batch[-i - 2])
            x = self.transformers_up[-i - 1](x, out_pos[-i - 2], edge_index)

        # Class score
        out = self.mlp_output(x)

        return F.log_softmax(out, dim=-1)


# def train(
#     model: torch.nn.Module,
#     train_loader: DataLoader,
#     optimizer: torch.optim.Optimizer,
#     device: torch.device,
#     gradient_accumulation_steps: int = 1,
#     scaler: torch.amp.GradScaler = None,
#     print_every: int = 10,
# ):
#     """
#     Train the PointTransformer Segmenter.
#     """
#     model.train()

#     global_loss = 0
#     global_correct_nodes = 0
#     global_total_nodes = 0

#     total_loss = correct_nodes = total_nodes = 0
#     for i, data in enumerate(train_loader):
#         data = data.to(device)

#         # Use mixed precision if scaler is provided
#         if scaler is not None:
#             with torch.amp.autocast(device_type=device.type):
#                 out = model(data.x, data.pos, data.batch)
#         else:
#             out = model(data.x, data.pos, data.batch)

#         # Handle ignore labels (-1) for ScanNet dataset
#         valid_mask = data.y >= 0

#         if valid_mask.sum() > 0:
#             valid_out = out[valid_mask]
#             valid_y = data.y[valid_mask].long()

#             loss = F.nll_loss(valid_out, valid_y)
#             loss = loss / gradient_accumulation_steps

#             if scaler is not None:
#                 scaler.scale(loss).backward()
#             else:
#                 loss.backward()

#             correct_nodes += valid_out.argmax(dim=1).eq(valid_y).sum().item()
#             total_nodes += valid_mask.sum().item()
#         else:
#             continue

#         if (i + 1) % gradient_accumulation_steps == 0:
#             if scaler is not None:
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 optimizer.step()
#             optimizer.zero_grad()
#             total_loss += loss.item() * gradient_accumulation_steps
#         else:
#             total_loss += loss.item() * gradient_accumulation_steps

#         global_loss += loss.item() * gradient_accumulation_steps
#         global_correct_nodes += correct_nodes
#         global_total_nodes += total_nodes

#         if (i + 1) % print_every == 0:
#             print(
#                 f"[{i+1}/{len(train_loader)}] Loss: {total_loss / print_every:.4f} "
#                 f"Train Acc: {correct_nodes / total_nodes:.4f}"
#             )
#             total_loss = correct_nodes = total_nodes = 0

#     return global_loss / len(train_loader), global_correct_nodes / global_total_nodes


# @torch.no_grad()
# def test(
#     model: torch.nn.Module,
#     loader: DataLoader,
#     device: torch.device,
#     dataset_type: PointDatasetType,
# ):
#     """
#     Test the PointTransformer Segmenter.
#     """
#     model.eval()

#     ious, categories = [], []
#     for data in loader:
#         data = data.to(device)
#         outs = model(data.x, data.pos, data.batch)

#         sizes = (
#             data.ptr[1:] - data.ptr[:-1]
#         ).tolist()  # calculate the number of nodes in each point cloud of the batch

#         for out, y in zip(
#             outs.split(sizes),  # split the output of each point cloud of the batch
#             data.y.split(
#                 sizes
#             ),  # split the ground truth of each point cloud of the batch
#         ):
#             # Handle ignore labels (-1) for ScanNet dataset
#             valid_mask = y >= 0

#             if valid_mask.sum() > 0:
#                 valid_out = out[valid_mask]
#                 valid_y = y[valid_mask]

#                 # Check if this is ShapeNet (has category attribute) or ScanNet
#                 # if hasattr(data, "category") and data.category is not None:
#                 if dataset_type == PointDatasetType.ShapeNet:
#                     # ShapeNet: category-based evaluation with local part mapping
#                     category = data.category.tolist()
#                     for i, (out_single, y_single, cat) in enumerate(
#                         zip(valid_out.split(1), valid_y.split(1), category)
#                     ):
#                         # Get category name from index
#                         category_names = list(ShapeNet.seg_classes.keys())
#                         category_name = category_names[cat]
#                         part_labels = ShapeNet.seg_classes[category_name]
#                         part_labels = torch.tensor(part_labels, device=device)

#                         # Create mapping from global labels to local part labels
#                         y_map = torch.zeros(
#                             loader.dataset.num_classes, device=device, dtype=torch.long
#                         )
#                         y_map[part_labels] = torch.arange(
#                             len(part_labels), device=device
#                         )

#                         # Map ground truth labels to local part indices
#                         y_local = y_map[y_single.squeeze()]

#                         # Get predictions for the relevant parts only
#                         out_parts = out_single.squeeze()[part_labels]

#                         iou = jaccard_index(
#                             out_parts.argmax(dim=-1),
#                             y_local,
#                             num_classes=len(part_labels),
#                             task="multiclass",
#                         )
#                         ious.append(iou)
#                         categories.append(cat)
#                 else:
#                     # ScanNet: direct global semantic segmentation evaluation
#                     iou = jaccard_index(
#                         valid_out.argmax(dim=-1),
#                         valid_y,
#                         num_classes=loader.dataset.num_classes,
#                         task="multiclass",
#                     )
#                     ious.append(iou)
#                     categories.append(0)  # Use 0 as default category for ScanNet

#     iou = torch.tensor(ious, device=device)
#     category = torch.tensor(categories, device=device)

#     if (
#         hasattr(loader.dataset, "seg_classes")
#         and loader.dataset.seg_classes is not None
#     ):
#         # ShapeNet: Per-category IoU
#         mean_iou = scatter(iou, category, reduce="mean")
#         return float(mean_iou.mean())  # Global IoU
#     else:
#         # ScanNet: Direct mean IoU
#         return float(iou.mean())
