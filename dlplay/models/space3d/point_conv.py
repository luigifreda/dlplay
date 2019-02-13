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

from typing import Callable, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairOptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import add_self_loops, remove_self_loops


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
class PointNetConv(MessagePassing):
    r"""The PointNet set layer from the `"PointNet: Deep Learning on Point Sets
    for 3D Classification and Segmentation"
    <https://arxiv.org/abs/1612.00593>`_ and `"PointNet++: Deep Hierarchical
    Feature Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ papers.

    .. math::
        \mathbf{x}^{\prime}_i = \gamma_{\mathbf{\Theta}} \left( \max_{j \in
        \mathcal{N}(i) \cup \{ i \}} h_{\mathbf{\Theta}} ( \mathbf{x}_j,
        \mathbf{p}_j - \mathbf{p}_i) \right),

    where :math:`\gamma_{\mathbf{\Theta}}` and :math:`h_{\mathbf{\Theta}}`
    denote neural networks, *i.e.* MLPs, and
    :math:`\mathbf{P} \in \mathbb{R}^{N \times D}` defines the position of
    each point.

    Args:
        local_nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` and
            relative spatial coordinates :obj:`pos_j - pos_i` of shape
            :obj:`[-1, in_channels + num_dimensions]` to shape
            :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
        global_nn (torch.nn.Module, optional): A neural network
            :math:`\gamma_{\mathbf{\Theta}}` that maps aggregated node features
            of shape :obj:`[-1, out_channels]` to shape :obj:`[-1,
            final_out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          positions :math:`(|\mathcal{V}|, 3)` or
          :math:`((|\mathcal{V_s}|, 3), (|\mathcal{V_t}|, 3))` if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    def __init__(
        self,
        local_nn: Optional[Callable] = None,
        global_nn: Optional[Callable] = None,
        add_self_loops: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "max")
        super().__init__(**kwargs)

        self.local_nn = local_nn
        self.global_nn = global_nn
        self.add_self_loops = add_self_loops

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.local_nn)
        reset(self.global_nn)

    def forward(
        self,
        x: Union[OptTensor, PairOptTensor],
        pos: Union[Tensor, PairTensor],
        edge_index: Adj,
    ) -> Tensor:

        if not isinstance(x, tuple):
            x = (x, None)

        if isinstance(pos, Tensor):
            pos = (pos, pos)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(
                    edge_index, num_nodes=min(pos[0].size(0), pos[1].size(0))
                )
            elif isinstance(edge_index, SparseTensor):
                edge_index = torch_sparse.set_diag(edge_index)

        # propagate_type: (x: PairOptTensor, pos: PairTensor)
        out = self.propagate(edge_index, x=x, pos=pos)

        if self.global_nn is not None:
            # apply the global neural network to the output
            out = self.global_nn(out)

        return out

    def message(self, x_j: Optional[Tensor], pos_i: Tensor, pos_j: Tensor) -> Tensor:
        msg = pos_j - pos_i
        if x_j is not None:
            # concat the features of neighbors and the spatial relation between source and destination nodes,
            # dim=-1 means concatenate the last dimension => output shape [num_edges, in_channels + 3]
            msg = torch.cat([x_j, msg], dim=1)
        if self.local_nn is not None:
            # apply the local neural network to the message
            msg = self.local_nn(msg)
        return msg

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(local_nn={self.local_nn}, "
            f"global_nn={self.global_nn})"
        )
