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

from torch_geometric.datasets import GeometricShapes
import torch_geometric.transforms as T

from dlplay.paths import DATA_DIR
from dlplay.viz.viz_process import VizProcess
from dlplay.viz.view3d import viz_point_cloud, viz_mesh, viz_graph_with_edges


# from https://pytorch-geometric.readthedocs.io/en/latest/tutorial/point_cloud.html
# NOTE: Although point clouds do not come with a graph structure by default, we can utilize
# PyG transformations to make them applicable for the full suite of GNNs available in PyG.
# The key idea is to create a synthetic graph from point clouds, from which we can learn meaningful
# local geometric structures via a GNN’s message passing scheme. These point representations
# can then be used to, e.g., perform point cloud classification or segmentation.
if __name__ == "__main__":

    # NOTE: A first example of synthetic graph creation from point clouds by using the KNNGraph transform
    #       is given in quickstart/torch/19.graph_networks/5.graph_3dpoints_dataset_kaggle_shapenet_transform.py

    # PyG provides several point cloud datasets, such as the:
    # - PCPNetDataset  -> from torch_geometric.datasets import PCPNetDataset
    # - S3DIS dataset -> from torch_geometric.datasets import S3DIS
    # - ShapeNet dataset -> from torch_geometric.datasets import ShapeNet
    # - GeometricShapes dataset -> from torch_geometric.datasets import GeometricShapes

    # To get started, we also provide the GeometricShapes dataset, which is a toy dataset that contains
    # various geometric shapes such cubes, spheres or pyramids. Notably, the GeometricShapes dataset
    # contains meshes instead of point clouds by default, represented via pos and face attributes,
    # which hold the information of vertices and their triangular connectivity, respectively:

    dataset = GeometricShapes(root=f"{DATA_DIR}/datasets/GeometricShapes")
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset: {dataset[0]}")

    data = dataset[0]
    print(f"original dataset[0]: {data}")
    # >>> Data(pos=[32, 3], face=[3, 30], y=[1])

    viz = VizProcess(
        viz_point_cloud,
        {"data": data, "title": "GeometricShapes - point cloud"},
        process_name="GeometricShapes",
    )

    viz2 = VizProcess(
        viz_mesh,
        {
            "data": data,
            "title": "GeometricShapes - wired mesh",
            "wired": True,
            "shaded": True,
        },
        process_name="GeometricShapes",
    )

    # Since we are interested in point clouds, we can transform our meshes into points via the usage of
    # torch_geometric.transforms. In particular, PyG provides the SamplePoints transformation, which will
    # uniformly sample a fixed number of points on the mesh faces according to their face area.
    # We can add this transformation to the dataset by simply setting it via
    # dataset.transform = SamplePoints(num=...). Each time an example is accessed from the dataset,
    # the transformation procedure will get called, converting our mesh into a point cloud (faces will be removed).
    # Note that sampling points is stochastic, and so you will receive a new point cloud upon every access:

    dataset.transform = T.SamplePoints(
        num=256
    )  # number of points to sample, the faces will be removed
    data = dataset[0]
    print(f"sampled dataset[0]: {data}")
    # >>> Data(pos=[256, 3], y=[1])

    viz3 = VizProcess(
        viz_point_cloud,
        {"data": data, "title": "GeometricShapes - sampled point cloud"},
        process_name="GeometricShapes - sampled point cloud",
    )

    # Finally, let’s convert our point cloud into a graph. Since we are interested in learning local geometric structures,
    # we want to construct a graph in such a way that nearby points are connected. Typically, this is either done via
    # nearest neighbor search or via ball queries (which connect all points that are within a certain radius to the query point).
    # PyG provides utilities for such graph generation via the KNNGraph and RadiusGraph transformations, respectively.

    dataset.transform = T.Compose([T.SamplePoints(num=256), T.KNNGraph(k=6)])

    data = dataset[0]
    print(f"graph dataset[0]: {data}")
    # >>> Data(pos=[256, 3], edge_index=[2, 1536], y=[1])

    # You can see that the data object now also contains an edge_index representation, holding 1536 edges in total,
    # 6 edges for every of the 256 points. We can confirm that our graph looks good via the following visualization:

    viz4 = VizProcess(
        viz_graph_with_edges,
        {"data": data, "title": "GeometricShapes - graph with edges"},
        process_name="GeometricShapes - graph with edges",
    )
