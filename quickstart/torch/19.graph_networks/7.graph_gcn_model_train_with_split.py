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

from torch_geometric.datasets import IMDB
from dlplay.paths import DATA_DIR
from dlplay.utils.device import resolve_device

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# This class defines a GCN model with two GCNConv layers which get called in the forward pass of our network.
# Note that the non-linearity is not integrated in the conv calls and hence needs to be applied
# afterwards (something which is consistent across all operators in PyG).
# Here, we chose to use ReLU as our intermediate non-linearity and finally output a softmax
# distribution over the number of classes.
# NOTE: The GCN layer implements the graph convolution operation from the paper:
# "Semi-Supervised Classification with Graph Convolutional Networks"
#  X' = D^-1/2 A D^-1/2 X W
# where:
# A is the modified adjacency matrix with inserted self-loops and
# D is the corresponding diagonal degree matrix.
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


# https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
if __name__ == "__main__":
    # It’s time to implement our first graph neural network!

    # We first need to load the Cora dataset:
    dataset = IMDB(root=f"{DATA_DIR}/datasets/IMDB")
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset: {dataset[0]}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Number of node features: {dataset.num_node_features}")
    print(f"Number of edges: {dataset.edge_index.shape[1]}")
    print(f"Number of node features: {dataset.num_node_features}")

    device = resolve_device()
    model = GCN().to(device)

    # randomly split it into 80%/10%/10% training, validation and test graphs.
    dataset = dataset.shuffle()
    train_dataset = dataset[: int(len(dataset) * 0.8)]
    val_dataset = dataset[int(len(dataset) * 0.8) : int(len(dataset) * 0.9)]
    test_dataset = dataset[int(len(dataset) * 0.9) :]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
