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

from torch_geometric import transforms as T

import torch
import numpy as np

from dlplay.paths import DATA_DIR

from dlplay.datasets.shapenet import ShapeNet

from dlplay.viz.view3d import viz_graph_with_edges, viz_point_cloud
from dlplay.viz.viz_process import VizProcess
from dlplay.datasets.stats import get_category_stats
import open3d as o3d

"""
Transforms are a common way in torchvision to transform images and perform augmentation. 
PyG comes with its own transforms, which expect a Data object as input and return a new transformed Data object. 
Transforms can be chained together using torch_geometric.transforms.Compose and are applied before saving a processed
dataset on disk (pre_transform) or before accessing a graph in a dataset (transform).

Let's look at an example, where we apply transforms on the ShapeNet dataset (containing 17,000 3D shape point clouds and per point labels from 16 shape categories).
"""
if __name__ == "__main__":

    # We can convert the point cloud dataset into a graph dataset by generating nearest neighbor graphs
    # from the point clouds via transforms.
    # NOTE: We use the pre_transform to convert the data before saving it to disk (leading to faster loading times).
    # Note that the next time the dataset is initialized it will already contain graph edges, even if you do not pass
    # any transform. If the pre_transform does not match with the one from the already processed dataset, you will
    # be given a warning.

    categories = ["Airplane", "Car", "Chair"]
    force_reload = False

    # Get ShapeNet dataset
    dataset = ShapeNet(
        root=f"{DATA_DIR}/datasets/ShapeNet",
        categories=categories,
        pre_transform=T.KNNGraph(k=6),
        force_reload=force_reload,
    )

    # Show first point cloud with edges
    point_cloud = dataset[0]
    print(f"point_cloud: {point_cloud}")
    print(f"Number of nodes: {point_cloud.pos.shape[0]}")
    print(f"Number of edges: {point_cloud.edge_index.shape[1]}")
    # Print category statistics
    stats = get_category_stats(dataset)
    print(f"Category statistics: {stats}")

    viz1a = VizProcess(
        viz_point_cloud,
        {"data": point_cloud, "title": "ShapeNet point cloud"},
        process_name="ShapeNet point cloud",
    )

    # Visualize with edges
    viz1b = VizProcess(
        viz_graph_with_edges,
        {"data": point_cloud, "title": "ShapeNet Graph with KNN Edges (k=6)"},
        process_name="ShapeNet Graph with KNN Edges (k=6)",
    )

    # NOTE: In addition, we can use the transform argument to randomly augment a Data object,
    # e.g., translating each node position by a small number:

    dataset2 = ShapeNet(
        root=f"{DATA_DIR}/datasets/ShapeNet",
        categories=categories,
        pre_transform=T.KNNGraph(k=6),
        transform=T.RandomJitter(0.01),
        force_reload=force_reload,
    )

    point_cloud2 = dataset2[0]
    print(f"point_cloud2: {point_cloud2}")
    print(f"Number of nodes: {point_cloud2.pos.shape[0]}")
    print(f"Number of edges: {point_cloud2.edge_index.shape[1]}")

    # Visualize the jittered version with edges

    viz2a = VizProcess(
        viz_point_cloud,
        {"data": point_cloud2, "title": "ShapeNet point cloud with Jittered Points"},
        process_name="ShapeNet point cloud with Jittered Points",
    )
    viz2b = VizProcess(
        viz_graph_with_edges,
        {
            "data": point_cloud2,
            "title": "ShapeNet Graph with Jittered Points and Edges",
        },
        process_name="ShapeNet Graph with Jittered Points and Edges",
    )
