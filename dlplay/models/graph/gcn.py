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

import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


# A general message passing layer is  is typically expressed as a neighborhood aggregation or message passing scheme.
# With x_i^k in IR^F denoting node features of node i in layer k and e_ij in IR^D denoting (optional) edge features
# from node i to node j, message passing graph neural networks can be described as
#
#   x_i^k = gamma^k( x_i^{k-1}, Aggr_{j in N(i) U {i}} phi^k(x_i^{k-1}, x_j^{k-1}, e_ij) )
#
# where
# - N(i) denotes the neighborhood of node i.
# - phi^k() is a differentiable message function that transforms the node features and edge features into a message.
# - Aggr_{} is the aggregation function that aggregates the messages from a set of nodes.
#           It is assumed to be a differentiable and permutation invariant function
# - gamma^k() is a differentiable function that aggregate the node features and the aggregated messages.
#
#
# The GCN layer is a specific message passing layer that is defined as
#
#   x_i^k = sum_{j in N(i) U {i}}  1 / sqrt(deg(i) * deg(j)) * (W^T x_j^{k-1}) + b
#
#   f(X, A) = (D_hat^-1/2 A_hat D_hat^-1/2) (X W + b)
#
# where neighboring node features are first transformed by a weight matrix W,
# normalized by their degree, and finally summed up. Lastly, we apply the bias vector b
# to the aggregated output. This formula can be divided into the following steps:
#
# 1. Add self-loops to the adjacency matrix = > A_hat = A + I
# 2. Linearly transform node feature matrix = > X_hat = X W
# 3. Compute normalization coefficients = > D_hat^-1/2 A_hat D_hat^-1/2
# 4. Normalize node features in phi = > phi = D_hat^-1/2 X_hat
# 5. Sum up neighboring node features ("add" aggregation).
# 6. Apply a final bias vector.


class GCNConv(MessagePassing):
    """
    Graph Convolutional Network (GCN) layer.
    All the logic of the layer takes place in its forward() method.
    Args:
        in_channels: Number of input features.
        out_channels: Number of output features.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="add")  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        # A = A + I
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out = out + self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
