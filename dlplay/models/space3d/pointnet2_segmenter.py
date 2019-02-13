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

from dlplay.models.space3d.pointnet2_classifier import GlobalSAModule, SAModule
from torchmetrics.functional import jaccard_index

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from dlplay.datasets.shapenet import ShapeNet
from dlplay.datasets.point3d_datasets_helpers import PointDatasetType

from torch_geometric.nn import MLP, knn_interpolate
from torch_geometric.typing import WITH_TORCH_CLUSTER
from torch_geometric.utils import scatter

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")

# from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_segmentation.py


class FPModule(torch.nn.Module):
    """
    Feature Propagation Module, from PointNet++ paper.
    """

    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        # KNN Interpolation
        # Interpolate the features of the skipped points to the new points.
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class PointNet2Segmenter(torch.nn.Module):
    """
    PointNet2Segmenter class from PointNet++ paper.
    """

    def __init__(self, num_classes):
        super().__init__()

        # Input channels account for both `pos` and node features.
        # Set Abstraction Module 1
        # 0.2 means the ratio of the points to be sampled
        # 0.2 means the radius of the ball query
        # MLP([3 + 3, 64, 64, 128]) means the input is 3D points and the output is 128D features.
        #                       [3+3,64,64,128] means 3D points -> 64D features -> 64D features -> 128D features.
        self.sa1_module = SAModule(0.2, 0.2, MLP([3 + 3, 64, 64, 128]))
        # Set Abstraction Module 2
        # 0.25 means the ratio of the points to be sampled
        # 0.4 means the radius of the ball query
        # MLP([128 + 3, 128, 128, 256]) means the input is 128D features and 3D points and the output is 256D features.
        #                               [128+3,128,128,256] means 128D features -> 128D features -> 128D features -> 256D features.
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        # Global Set Abstraction Module
        # MLP([256 + 3, 256, 512, 1024]) means the input is 256D features and 3D points and the output is 1024D features.
        #                               [256+3,256,512,1024] means 256D features -> 256D features -> 512D features -> 1024D features.
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        # Feature Propagation Module 3
        # 1 means the number of nearest neighbors
        # MLP([1024 + 256, 256, 256]) means the input is 1024D features and 256D features and the output is 256D features.
        #                               [1024+256,256,256] means 1024D features -> 256D features -> 256D features.
        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        # Feature Propagation Module 2
        # 3 means the number of nearest neighbors
        # MLP([256 + 128, 256, 128]) means the input is 256D features and 128D features and the output is 128D features.
        #                               [256+128,256,128] means 256D features -> 256D features -> 128D features.
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        # Feature Propagation Module 1
        # 3 means the number of nearest neighbors
        # MLP([128 + 3, 128, 128, 128]) means the input is 128D features and 3D points and the output is 128D features.
        #                               [128+3,128,128,128] means 128D features -> 128D features -> 128D features -> 128D features.
        self.fp1_module = FPModule(3, MLP([128 + 3, 128, 128, 128]))

        # Final MLP for segmentation
        # MLP([128, 128, 128, num_classes], dropout=0.5, norm=None) means the input is 128D features and the output is num_classesD features.
        #                               [128,128,128,num_classes] means 128D features -> 128D features -> 128D features -> num_classesD features.
        self.mlp = MLP([128, 128, 128, num_classes], dropout=0.5, norm=None)

        # self.lin1 = torch.nn.Linear(128, 128)
        # self.lin2 = torch.nn.Linear(128, 128)
        # self.lin3 = torch.nn.Linear(128, num_classes)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)  # Set Abstraction Module 1
        sa2_out = self.sa2_module(*sa1_out)  # Set Abstraction Module 2
        sa3_out = self.sa3_module(*sa2_out)  # Global Set Abstraction Module

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)  # Feature Propagation Module 3
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)  # Feature Propagation Module 2
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)  # Feature Propagation Module 1

        return self.mlp(x).log_softmax(dim=-1)


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
#     Train the PointNet++ Segmenter.
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
#                 out = model(data)
#         else:
#             out = model(data)

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
#     Test the PointNet++ Segmenter.
#     """
#     model.eval()

#     ious, categories = [], []
#     for data in loader:
#         data = data.to(device)
#         outs = model(data)

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
