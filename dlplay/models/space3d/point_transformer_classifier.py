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
from torch.nn import Linear as Lin

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    MLP,
    PointTransformerConv,
    fps,
    global_mean_pool,
    knn,
    knn_graph,
)
from torch_geometric.typing import WITH_TORCH_CLUSTER
from torch_geometric.utils import scatter

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")


class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Lin(in_channels, in_channels)
        self.lin_out = Lin(out_channels, out_channels)

        self.pos_nn = MLP([3, 64, out_channels], norm=None, plain_last=False)

        self.attn_nn = MLP(
            [out_channels, 64, out_channels], norm=None, plain_last=False
        )

        self.transformer = PointTransformerConv(
            in_channels, out_channels, pos_nn=self.pos_nn, attn_nn=self.attn_nn
        )

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x


class TransitionDown(torch.nn.Module):
    """Samples the input point cloud by a ratio percentage to reduce
    cardinality and uses an mlp to augment features dimensionnality.
    """

    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], plain_last=False)

    def forward(self, x, pos, batch):
        # FPS sampling (arthest point sampling)
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(
            pos, pos[id_clusters], k=self.k, batch_x=batch, batch_y=sub_batch
        )

        # transformation of features through a simple MLP
        x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out = scatter(
            x[id_k_neighbor[1]],
            id_k_neighbor[0],
            dim=0,
            dim_size=id_clusters.size(0),
            reduce="max",
        )

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


class PointTransformerClassifier(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim_model, k=16):
        super().__init__()
        self.k = k

        # dummy feature is created if there is none given
        in_channels = max(in_channels, 1)

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]], plain_last=False)

        self.transformer_input = TransformerBlock(
            in_channels=dim_model[0], out_channels=dim_model[0]
        )
        # backbone layers
        self.transformers_down = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        for i in range(len(dim_model) - 1):
            # Add Transition Down block followed by a Transformer block
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

        # class score computation
        self.mlp_output = MLP([dim_model[-1], 64, out_channels], norm=None)

    def forward(self, x, pos, batch=None):

        # add dummy features in case there is none
        if x is None:
            x = torch.ones((pos.shape[0], 1), device=pos.get_device())

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # backbone
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)

            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

        # GlobalAveragePooling
        x = global_mean_pool(x, batch)

        # Class score
        out = self.mlp_output(x)

        return F.log_softmax(out, dim=-1)


def train(model, optimizer, train_dataset, train_loader, device):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.pos, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_dataset)


@torch.no_grad()
def test(model, loader, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.pos, data.batch).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)
