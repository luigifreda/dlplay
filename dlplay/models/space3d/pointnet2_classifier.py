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

import torch_geometric.transforms as T
from torch_geometric.nn import MLP, fps, global_max_pool, radius
from torch_geometric.typing import WITH_TORCH_CLUSTER

from dlplay.models.space3d.point_conv import PointNetConv

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")

# from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py
# Reference paper: "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space"
# https://arxiv.org/abs/1706.02413
# In this work, the Authors introduce a hierarchical neural network that applies PointNet recursively
# on a nested partitioning of the input point set. By exploiting metric space distances,
# the network is able to learn local features with increasing contextual scales.
# With further observation that point sets are usually sampled with varying densities,
# which results in greatly decreased performance for networks trained on uniform densities,
# the Authors propose novel set learning layers to adaptively combine features from multiple scales.
# Experiments show that the network called PointNet++ is able to learn deep point set features
# efficiently and robustly. In particular, results significantly better than state-of-the-art
# have been obtained on challenging benchmarks of 3D point clouds.


class SAModule(torch.nn.Module):
    """
    Set Abstraction Module class from PointNet++ paper.
    It performs sampling and grouping operation, and then applies a PointNet convolution layer.

    A set abstraction level takes an N x (d + C) matrix as input that is from N points with
    d-dim coordinates and C-dim point feature. It outputs an N x (d + C') matrix of N subsampled
    points with d-dim coordinates and new C'-dim feature vectors summarizing local context.
    """

    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(
            nn, add_self_loops=False
        )  # pointnet++ convolution layer

    def forward(self, x, pos, batch):
        # x: N x C
        # pos: N x d  (d=3 for 3D point clouds)
        # batch: N

        # Sampling and grouping operation
        # Given input points {x1,x2,...,xn},we use iterative farthest point sampling (FPS) to choose a
        # subset of points {x_i1,x_i2,...,x_im} such that x_ij is the most distant point (in metric distance)
        # from the set {x_i1,...,x_i(j-1)} with regard to the rest points. Compared with random sampling,
        # it has better coverage of the entire point set given the same number of centroids.
        # The output of FPS is a set of indices of the points that are the centroids.
        idx = fps(pos, batch, ratio=self.ratio)

        # Grouping operation
        # The input to this layer is a point set of size N x (d + C) and the coordinates of a set of centroids
        # of size N' x d. The output are groups of point sets of size N' x K x (d+C), where each group
        # corresponds to a local region and K is the number of points in the neighborhood of centroid points.
        # Note that K varies across groups.
        # Given the subset of points {x_i1,x_i2,...,x_im}, we use a ball query to find all points (from the input
        # point set) within a distance r.
        # The output is a tuple of two tensors: row, col. row is the indices of the points that are
        # the centroids and col is the indices of the points in the neighborhood of the centroids.
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64
        )
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]

        # PointNet convolution layer
        # In this layer, the input are N' local regions of points with data size N' x K x (d+C).
        # Each local region in the output is abstracted by its centroid and local feature
        # that encodes the centroid’s neighborhood. Output data size is N' x (d + C').
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    """
    Global Set Abstraction Module class from PointNet++ paper.
    This is the final layer for performing classification.
    It performs global feature aggregation and then applies a MLP for classification.
    """

    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        # Global feature aggregation.
        # Returns batch-wise graph-level-outputs by taking the channel-wise
        # maximum across the node dimension.
        x = global_max_pool(x, batch)
        # Reset the position and batch of the global feature
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNet2Classifier(torch.nn.Module):
    """
    PointNet2Classifier class from PointNet++ paper.
    """

    def __init__(self):
        super().__init__()

        # Input channels account for both `pos` and node features.
        # Set Abstraction Module 1
        # 0.5 means the ratio of the points to be sampled
        # 0.2 means the radius of the ball query
        # MLP([3, 64, 64, 128]) means the input is 3D points and the output is 128D features.
        #                       [3,64,64,128] means 3D points -> 64D features -> 64D features -> 128D features.
        # The MLP is used to transform the aggregate features/messages of the local regions.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        # Set Abstraction Module 2
        # 0.25 means the ratio of the points to be sampled
        # 0.4 means the radius of the ball query
        # MLP([128 + 3, 128, 128, 256]) means the input is 128D features and 3D points and the output is 256D features.
        #                               [128+3,128,128,256] means 128D features -> 128D features -> 128D features -> 256D features.
        # The MLP is used to transform the aggregate features/messages of the local regions.
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        # Global Set Abstraction Module
        # MLP([256 + 3, 256, 512, 1024]) means the input is 256D features and 3D points and the output is 1024D features.
        #                               [256+3,256,512,1024] means 256D features -> 256D features -> 512D features -> 1024D features.
        # The MLP is used to transform the aggregate features/messages of the local regions.
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        # Final MLP for classification
        # MLP([1024, 512, 256, 10]) means the input is 1024D features and the output is 10D features.
        #                         [1024,512,256,10] means 1024D features -> 512D features -> 256D features -> 10D features.
        self.mlp = MLP([1024, 512, 256, 10], dropout=0.5, norm=None)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)  # Set Abstraction Module 1
        sa2_out = self.sa2_module(*sa1_out)  # Set Abstraction Module 2
        sa3_out = self.sa3_module(*sa2_out)  # Global Set Abstraction Module
        x, pos, batch = sa3_out

        return self.mlp(x).log_softmax(dim=-1)


def train(epoch, model, device, optimizer, train_loader):
    """
    Train the PointNet++ Classifier.
    """
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def test(loader, model, device):
    """
    Test the PointNet++ Classifier.
    """
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)
