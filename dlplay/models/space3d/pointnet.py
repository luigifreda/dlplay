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
from torch.nn import Sequential, Linear, ReLU
import torch_geometric.transforms as T
from torch_geometric.nn import MessagePassing, global_max_pool

# from https://pytorch-geometric.readthedocs.io/en/latest/tutorial/point_cloud.html
#
# PointNet++ is a pioneering work that proposes a Graph Neural Network architecture for point cloud
# classification and segmentation. PointNet++ processes point clouds iteratively by following a
# simple grouping, neighborhood aggregation and downsampling scheme:
#
# https://pytorch-geometric.readthedocs.io/en/latest/_images/point_cloud4.png
#
# 1. The grouping phase constructs a graph with k-nearest neighbors or ball queries.
# 2. The neighborhood aggregation phase executes a GNN layer that, for each point,
#    aggregates information from its direct neighbors (given by the graph constructed
#    in the previous phase). This allows PointNet++ to capture local context at different scales.
# 3. The downsampling phase implements a pooling scheme suitable for point clouds with potentially
#    different sizes. Due to simplicity, we will ignore this phase for now.
#
# We recommend to take a look at examples/pointnet2_classification.py on guidance to how to implement this step.


# Neighborhood Aggregation
# The PointNet++ layer follows a simple neural message passing scheme defined via
#
#  h_i^{l+1} = max_{j in N(i)} MLP(h_j^{l}, p_j - p_i)
#
# where
# - h_j^{l} in IR^d denotes the hidden features of point j in layer l, d is the dimension of the hidden features
# - N(i) denotes the set of direct neighbors of point i (1-hop neighbors)
# - p_j denotes the position of point i
# - MLP is a multi-layer perceptron
# The max_{j in N(i)} operation is used to aggregate the information from the direct neighbors.
#
# We can make use of the MessagePassing interface in PyG to implement this layer from scratch.
# The MessagePassing interface helps us in creating message passing graph neural networks by automatically
# taking care of message propagation. Here, we only need to define its message() function and which aggregation
# scheme we want to use, e.g., aggr="max" (see here for the accompanying tutorial):
class PointNetLayer(MessagePassing):
    """
    Basic implementation of PointNet layer class from PointNet paper.
    """

    def __init__(self, in_channels: int, out_channels: int):
        # Message passing with "max" aggregation.
        super().__init__(aggr="max")

        # Initialize an MLP that takes care of transforming node features of neighbors and the spatial
        # relation between source and destination nodes to a (trainable) message.
        # Here, the number of input features correspond to the hidden
        # node dimensionality plus point dimensionality (=3).
        self.mlp = Sequential(
            Linear(in_channels + 3, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
        )

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        # Start propagating messages.
        # In the forward() function, we can start propagating messages based on edge_index,
        # and pass in everything needed in order to create messages. In the message() function,
        # we can now access neighbor and central node information via *_j and *_i suffixes,
        # respectively, and return a message for each edge.
        return self.propagate(edge_index, h=h, pos=pos)

    def message(
        self,
        h_j: torch.Tensor,
        pos_j: torch.Tensor,
        pos_i: torch.Tensor,
    ) -> torch.Tensor:
        # h_j: The features of neighbors as shape [num_edges, in_channels]
        # pos_j: The position of neighbors as shape [num_edges, 3]
        # pos_i: The central node position as shape [num_edges, 3]

        # concat the features of neighbors and the spatial relation between source and destination nodes,
        # dim=-1 means concatenate the last dimension => output shape [num_edges, in_channels + 3]
        edge_feat = torch.cat([h_j, pos_j - pos_i], dim=-1)
        return self.mlp(edge_feat)


# We can make use of above PointNetLayer to define our network architecture
# (or use its equivalent torch_geometric.nn.conv.PointNetConv directly integrated in PyG).
# With this, our overall PointNet architecture looks as follows:
class PointNet(torch.nn.Module):
    """
    PointNet Classifier, basic implementation from PointNet paper.
    """

    def __init__(self, num_classes):
        super().__init__()

        # Two-layers of message passing:
        # The first operator takes in 3 input features (the positions of nodes) and maps them to 32 output features.
        self.conv1 = PointNetLayer(3, 32)
        # The second operator takes in 32 input features (the output features of the first operator) and maps
        # them to 32 output features.
        self.conv2 = PointNetLayer(32, 32)
        # The classifier takes in 32 input features (the output features of the second operator) and maps
        # them to the number of classes.
        self.classifier = Linear(32, num_classes)

    def forward(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:

        # In the forward() method, we apply two graph-based convolutional operators and enhance them by ReLU non-linearities.
        # The first operator takes in 3 input features (the positions of nodes) and maps them to 32 output features.
        # The second operator takes in 32 input features (the output features of the first operator) and maps
        # them to 32 output features.
        # After that, each point holds information about its 2-hop neighborhood, and should already be able to
        # distinguish between simple local shapes.

        # Perform two-layers of message passing:
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()

        # Next, we apply a global graph readout function, i.e., global_max_pool(), which takes the maximum value
        # along the node dimension for each example. In order to map the different nodes to their corresponding examples,
        # we use the batch vector which will be automatically created for use when using the
        # mini-batch torch_geometric.loader.DataLoader.
        # In other words, global_max_pool() returns batch-wise graph-level-outputs by taking
        # the channel-wise maximum across the node dimension.
        # Global Pooling:
        h = global_max_pool(h, batch)  # [num_examples, hidden_channels]

        # Last, we apply a linear classifier to map the global 32 features per point cloud to one of the M classes.
        # Classifier:
        return self.classifier(h)


def train(model, optimizer, criterion, train_loader, device):
    model.train()

    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        logits = model(data.pos, data.edge_index, data.batch)
        loss = criterion(logits, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(model, test_loader, device):
    model.eval()

    total_correct = 0
    for data in test_loader:
        data = data.to(device)
        logits = model(data.pos, data.edge_index, data.batch)
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())

    return total_correct / len(test_loader.dataset)
