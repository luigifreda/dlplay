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
from torch_geometric.data import Data


"""
From https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html

A graph is used to model pairwise relations (edges) between objects (nodes). 
A single graph in PyG is described by an instance of torch_geometric.data.Data, 
which holds the following attributes by default:
- data.x: Node feature matrix with shape [num_nodes, num_node_features]
- data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
- data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
- data.y: Target to train against (may have arbitrary shape), e.g., node-level targets 
          of shape [num_nodes, *] or graph-level targets of shape [1, *]
- data.pos: Node position matrix with shape [num_nodes, num_dimensions]

We show a simple example of an unweighted and undirected graph with three nodes and 
four edges. Each node contains exactly one feature.


[0 (x=-1)] -- [1 (x=0)] -- [2 (x=1)]

"""
if __name__ == "__main__":
    # fmt: off
    edge_index = torch.tensor([[0, 1, 1, 2], 
                               [1, 0, 2, 1]], dtype=torch.long)
    # fmt: on
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    print(f"Data: {data}")

    # NOTE: Note that edge_index, i.e. the tensor defining the source and target nodes
    # of all edges, is not a list of index tuples. If you want to write your indices
    # this way, you should transpose and call contiguous on it before passing them
    # to the data constructor:

    # fmt: off
    edge_index = torch.tensor([[0, 1], 
                               [1, 0], 
                               [1, 2], 
                               [2, 1]], dtype=torch.long)
    # fmt: on
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data2 = Data(x=x, edge_index=edge_index.t().contiguous())
    print(f"Data2: {data2}")

    # NOTE: Although the graph has only two edges, we need to define four index tuples
    # to account for both directions of a edge.

    # NOTE: Note that it is necessary that the elements in edge_index only hold indices in
    # the range { 0, ..., num_nodes - 1}. This is needed as we want our final data representation
    # to be as compact as possible, e.g., we want to index the source and destination
    # node features of the first edge (0, 1) via x[0] and x[1], respectively.
    # You can always check that your final Data objects fulfill these requirements by running validate():

    data.validate(raise_on_error=True)
    data2.validate(raise_on_error=True)

    # Besides holding a number of node-level, edge-level or graph-level attributes,
    # Data provides a number of useful utility functions, e.g.:

    print(data.keys())
    # >>> ['x', 'edge_index']

    print(data["x"])
    # >>> tensor([[-1.0],
    #             [0.0],
    #             [1.0]])

    for key, item in data:
        print(f"{key} found in data")
    # >>> x found in data
    # >>> edge_index found in data

    print("edge_attr" in data)
    # >>> False

    print(data.num_nodes)
    # >>> 3

    print(data.num_edges)
    # >>> 4

    print(data.num_node_features)
    # >>> 1

    print(data.has_isolated_nodes())
    # >>> False

    print(data.has_self_loops())
    # >>> False

    print(data.is_directed())
    # >>> False

    # Transfer data object to GPU.
    device = torch.device("cuda")
    data = data.to(device)
