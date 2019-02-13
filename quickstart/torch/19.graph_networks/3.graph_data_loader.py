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

from torch_geometric.utils import scatter
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from dlplay.paths import DATA_DIR, RESULTS_DIR


"""_summary_
from https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

Neural networks are usually trained in a batch-wise fashion. PyG achieves parallelization 
over a mini-batch by creating sparse block diagonal adjacency matrices (defined by edge_index) 
and concatenating feature and target matrices in the node dimension. This composition allows 
differing number of nodes and edges over examples in one batch:

PyG contains its own torch_geometric.loader.DataLoader, which already takes care of this concatenation 
process. Let's learn about it in an example:

A = [A_1,            ]
    [    A_2,        ]
    [        ...     ]
    [            A_N ]

X = [X_1]
    [X_2]
    [...]
    [X_N]
    
Y = [Y_1]
    [Y_2]
    [...]
    [Y_N]
"""


def example_1():
    dataset = TUDataset(
        root=f"{DATA_DIR}/datasets/ENZYMES", name="ENZYMES", use_node_attr=True
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for i, batch in enumerate(loader):
        print(f"Batch {i}:")
        print(batch)
        # >>> DataBatch(batch=[1082], edge_index=[2, 4066], x=[1082, 21], y=[32])

        print(f"Batch {i}.num_graphs: {batch.num_graphs}")
        # >>> 32

        # NOTE: torch_geometric.data.Batch inherits from torch_geometric.data.Data and contains
        # an additional attribute called batch.

        # batch is a column vector which maps each node to its respective graph in the batch:
        print(f"Batch {i}.batch: {batch.batch}")


def example_2():
    # NOTE: batch is a column vector which maps each node to its respective graph in the batch:
    # You can use it to, e.g., average node features in the node dimension for each graph individually:
    dataset = TUDataset(
        root=f"{DATA_DIR}/datasets/ENZYMES", name="ENZYMES", use_node_attr=True
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for data in loader:
        print(f"data: {data}")
        # >>> DataBatch(batch=[1082], edge_index=[2, 4066], x=[1082, 21], y=[32])

        print(f"data.num_graphs: {data.num_graphs}")
        # >>> 32

        # https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html#torch_scatter.scatter
        x = scatter(data.x, data.batch, dim=0, reduce="mean")
        print(f"x.size(): {x.size()}")
        # >>> torch.Size([32, 21])


if __name__ == "__main__":
    example_1()
    example_2()
