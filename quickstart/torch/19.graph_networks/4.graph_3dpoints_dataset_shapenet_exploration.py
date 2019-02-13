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
from torch_geometric.loader import DataLoader

from dlplay.paths import DATA_DIR

from dlplay.datasets.shapenet import ShapeNet
from dlplay.datasets.stats import get_category_stats

import open3d as o3d


# NOTE: The following commented code does not work. It seems the shapenet url is broken.
# if __name__ == "__main__":

#     dataset = ShapeNet(
#         root=f"{DATA_DIR}/datasets/ShapeNet",
#         categories=["Airplane"],
#         pre_transform=T.KNNGraph(k=6),
#     )

#     print(dataset[0])
#     # >>> Data(edge_index=[2, 15108], pos=[2518, 3], y=[2518])


"""
 ShapeNet dataset (containing 17,000 3D shape point clouds and per point labels from 16 shape categories).
"""
if __name__ == "__main__":

    categories = ["Airplane", "Car", "Chair"]
    force_reload = False

    dataset = ShapeNet(
        root=f"{DATA_DIR}/datasets/ShapeNet",
        categories=categories,
        force_reload=force_reload,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Number of node features: {dataset.num_node_features}")

    # Print first item
    print(f"First item: {dataset[0]}")

    # Print category statistics
    stats = get_category_stats(dataset)
    print(f"Category statistics: {stats}")

    # Test with DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    if False:
        for batch in loader:
            print(f"Batch: {batch}")
            print(f"Number of graphs: {batch.num_graphs}")
            print(f"Node positions shape: {batch.pos.shape}")
            print(f"Edge index shape: {batch.edge_index.shape}")
            print(f"Labels: {batch.y}")
            break  # Just show first batch

    # Show first point cloud with open3d

    point_cloud = dataset[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud.pos)
    o3d.visualization.draw_geometries([pcd])
