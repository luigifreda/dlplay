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
from torch_geometric.datasets import TUDataset, Planetoid

from dlplay.paths import DATA_DIR, RESULTS_DIR

"""_summary_
from https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

PyG contains a large number of common benchmark datasets, e.g., all Planetoid datasets 
(Cora, Citeseer, Pubmed), all graph classification datasets from TUDatasets and their 
cleaned versions, the QM7 and QM9 dataset, and a handful of 3D mesh/point cloud datasets 
like FAUST, ModelNet10/40 and ShapeNet.

Initializing a dataset is straightforward. An initialization of a dataset will automatically 
download its raw files and process them to the previously described Data format. E.g., to load 
the ENZYMES dataset (consisting of 600 graphs within 6 classes), type:
"""


def example_1():

    dataset = TUDataset(root=f"{DATA_DIR}/datasets/ENZYMES", name="ENZYMES")
    print(dataset)
    # >>> ENZYMES(600)

    print(len(dataset))
    # >>> 600

    print(dataset.num_classes)
    # >>> 6

    print(dataset.num_node_features)
    # >>> 3

    # We now have access to all 600 graphs in the dataset:

    # Check the first graph in the dataset
    print(dataset[0])
    # >>> Data(edge_index=[2, 168], x=[37, 3], y=[1])
    print(dataset[0].is_undirected())
    # >>> True

    # We can see that the first graph in the dataset contains 37 nodes, each one having 3 features.
    # There are 168/2 = 84 undirected edges and the graph is assigned to exactly one class.
    # In addition, the data object is holding exactly one graph-level target.

    # We can even use slices, long or bool tensors to split the dataset. E.g., to create a 90/10 train/test split, type:

    train_dataset = dataset[:540]
    print(train_dataset)
    # >>> ENZYMES(540)

    test_dataset = dataset[540:]
    print(test_dataset)
    # >>> ENZYMES(60)

    # If you are unsure whether the dataset is already shuffled before you split,
    # you can randomly permute it by running:
    dataset = dataset.shuffle()
    print(dataset)
    # >>> ENZYMES(600)

    # This is equivalent of doing:
    perm = torch.randperm(len(dataset))
    dataset = dataset[perm]
    print(dataset)
    # >> ENZYMES(600)


def example_2():

    # Let’s try another one! Let’s download Cora, the standard benchmark dataset for semi-supervised
    # graph node classification:

    dataset = Planetoid(root=f"{DATA_DIR}/datasets/Cora", name="Cora")
    print(dataset)
    # >>> Cora()

    print(f"len(dataset): {len(dataset)}")
    # >>> 1

    print(f"dataset.num_classes: {dataset.num_classes}")
    # >>> 7

    print(f"dataset.num_node_features: {dataset.num_node_features}")
    # >>> 1433

    # Here, the dataset contains only a single, undirected citation graph:
    data = dataset[0]
    print(f"data=dataset[0]: {data}")
    # >>> Data(edge_index=[2, 10556], test_mask=[2708], train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])

    print(f"data.is_undirected(): {data.is_undirected()}")
    # >>> True

    print(f"data.train_mask.sum().item(): {data.train_mask.sum().item()}")
    # >>> 140

    print(f"data.val_mask.sum().item(): {data.val_mask.sum().item()}")
    # >>> 500

    print(f"data.test_mask.sum().item(): {data.test_mask.sum().item()}")
    # >>> 1000

    # This time, the Data objects holds a label for each node, and additional node-level attributes:
    # train_mask, val_mask and test_mask, where
    # - train_mask denotes against which nodes to train (140 nodes),
    # - val_mask denotes which nodes to use for validation, e.g., to perform early stopping (500 nodes),
    # - test_mask denotes against which nodes to test (1000 nodes).


if __name__ == "__main__":

    example_1()
    example_2()
