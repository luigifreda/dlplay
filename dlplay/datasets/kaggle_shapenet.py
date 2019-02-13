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

import os
import math
import numpy as np
from abc import ABC, abstractmethod

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.nn import knn_graph
import kagglehub
from typing import List, Optional, Dict, Any

from dlplay.paths import DATA_DIR
from dlplay.datasets.kaggle_basedataset import KaggleBaseDataset


def explore_shapenet_kaggle_shapenet_dataset():
    """
    Explore the dataset structure. A didactic/debugging function.
    """
    print("=== EXPLORING KAGGLE DATASET STRUCTURE ===")
    kaggle_path = kagglehub.dataset_download("mitkir/shapenet")
    print(f"Dataset downloaded to: {kaggle_path}")

    if os.path.exists(kaggle_path):
        print("\nContents of dataset root:")
        for item in os.listdir(kaggle_path):
            item_path = os.path.join(kaggle_path, item)
            if os.path.isdir(item_path):
                print(f"  📁 Directory: {item}")
                try:
                    subitems = os.listdir(item_path)
                    print(f"     Contains {len(subitems)} items")
                    # Show first few items
                    for subitem in subitems[:5]:
                        print(f"     - {subitem}")
                    if len(subitems) > 5:
                        print(f"     ... and {len(subitems) - 5} more")
                except Exception as e:
                    print(f"     Error: {e}")
            else:
                print(f"  📄 File: {item}")

    # Let's examine a sample .npz file to understand the data structure
    print("\n=== EXAMINING SAMPLE .NPZ FILE ===")
    sample_file = os.path.join(
        kaggle_path,
        "shapenetcore_partanno_segmentation_benchmark_v0_normal",
        "02691156",
        "1021a0914a7207aff927ed529ad90a11_8x8.npz",
    )

    if os.path.exists(sample_file):
        try:
            data = np.load(sample_file)
            print(f"Keys in .npz file: {list(data.keys())}")
            for key in data.keys():
                print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
                if data[key].size < 20:  # Show small arrays
                    print(f"    Sample values: {data[key]}")
        except Exception as e:
            print(f"Error examining .npz file: {e}")


class ShapeNetKaggleDataset(KaggleBaseDataset):
    """
    Specific implementation for ShapeNet dataset from Kaggle.

    This class handles the conversion of ShapeNet 3D point cloud data
    from Kaggle into PyTorch Geometric format.
    """

    def __init__(
        self,
        root: str,
        categories: Optional[List[str]] = None,
        max_points: Optional[int] = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload: bool = False,
        train: bool = True,
    ):
        """
        Initialize ShapeNet dataset from Kaggle.

        Args:
            root: Root directory where dataset should be saved
            categories: List of ShapeNet categories to include (e.g., ['Airplane', 'Car'])
            max_points: Maximum number of points per point cloud (for memory efficiency)
            transform: Optional transform to apply to data
            pre_transform: Optional transform to apply before saving
            pre_filter: Optional filter to apply before saving
            force_reload: If True, force reload even if processed files exist
        """
        self.categories_all = [
            "Airplane",
            "Bag",
            "Cap",
            "Car",
            "Chair",
            "Earphone",
            "Guitar",
            "Knife",
            "Lamp",
            "Laptop",
            "Motorbike",
            "Mug",
            "Pistol",
            "Rocket",
            "Skateboard",
            "Table",
        ]
        self.categories = categories or self.categories_all
        self.max_points = max_points
        self.train = train
        # Load category mapping from the actual dataset
        self.category_to_id = {}
        self.id_to_category = {}
        self.category_to_label = {}

        super().__init__(
            root=root,
            kaggle_dataset_name="mitkir/shapenet",
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )

    @property
    def num_classes(self) -> int:
        return 50  # Total number of segmentation classes

    @property
    def num_node_features(self) -> int:
        return 3  # 3D coordinates

    @property
    def raw_file_names(self) -> List[str]:
        return [f"shapenet_raw_{'train' if self.train else 'test'}.pt"]

    @property
    def processed_file_names(self) -> List[str]:
        return [f"shapenet_processed_{'train' if self.train else 'test'}.pt"]

    def download_kaggle_data(self) -> str:
        """Download ShapeNet dataset from Kaggle."""
        return kagglehub.dataset_download(self.kaggle_dataset_name)

    def _load_category_mapping(self, kaggle_path: str):
        """Load category mapping from synsetoffset2category.txt file."""
        mapping_file = os.path.join(
            kaggle_path,
            "shapenetcore_partanno_segmentation_benchmark_v0_normal",
            "synsetoffset2category.txt",
        )

        print(f"Category mapping file: {mapping_file}")

        if not os.path.exists(mapping_file):
            print(f"Warning: Category mapping file not found at {mapping_file}")
            return

        with open(mapping_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and "\t" in line:
                    category_name, category_id = line.split("\t")
                    self.category_to_id[category_name] = category_id
                    self.id_to_category[category_id] = category_name

        # Create label mapping for the requested categories
        for i, category in enumerate(self.categories):
            if category in self.category_to_id:
                self.category_to_label[category] = i
            else:
                print(f"Warning: Category '{category}' not found in dataset")

        print(f"Loaded {len(self.category_to_id)} categories from mapping file")
        print(f"Requested categories: {self.categories}")
        print(f"Available categories: {list(self.category_to_id.keys())}")

    def load_raw_data(self) -> List[Dict[str, Any]]:
        """Load and parse ShapeNet data from Kaggle."""
        kaggle_path = self.download_kaggle_data()
        data_list = []

        print(f"Loading ShapeNet data from: {kaggle_path}")

        # Load category mapping
        self._load_category_mapping(kaggle_path)

        # Navigate to the correct subdirectory
        shapenet_path = os.path.join(
            kaggle_path, "shapenetcore_partanno_segmentation_benchmark_v0_normal"
        )

        if not os.path.exists(shapenet_path):
            print(f"Error: ShapeNet data directory not found at {shapenet_path}")
            return data_list

        print(f"Processing data from: {shapenet_path}")

        # Load train/test split information
        train_file_paths = self._load_split_files(
            shapenet_path, "train" if self.train else "test"
        )

        # Process each requested category
        for category in self.categories:
            if category not in self.category_to_id:
                print(f"Warning: Category {category} not found in mapping")
                continue

            category_id = self.category_to_id[category]
            category_path = os.path.join(shapenet_path, category_id)

            if not os.path.exists(category_path):
                print(f"Warning: Category directory {category_path} not found")
                continue

            print(f"Processing category: {category} (ID: {category_id})")

            # Correct logic: iterate through all files and check if their full path is in the training split
            for file_name in os.listdir(category_path):
                if not file_name.endswith(".npz"):
                    continue

                relative_file_path = os.path.join(category_id, file_name)
                # Remove both .npz and _8x8 from the relative path for comparison since JSON paths don't have them
                relative_file_path_no_suffix = relative_file_path.replace(
                    ".npz", ""
                ).replace("_8x8", "")

                print(f"Checking file: {file_name}")
                print(f"Relative path: {relative_file_path}")
                print(f"Relative path no suffix: {relative_file_path_no_suffix}")
                print(
                    f"Is in train paths: {relative_file_path_no_suffix in train_file_paths}"
                )

                # Only process files that are in the training split
                if relative_file_path_no_suffix in train_file_paths:
                    print(f"MATCH FOUND!")
                    file_path = os.path.join(category_path, file_name)

                    try:
                        points = self._load_npz_file(file_path)

                        if points is None or len(points) == 0:
                            print(f"Warning: Empty or invalid data in {file_name}")
                            continue

                        if self.max_points and len(points) > self.max_points:
                            indices = np.random.choice(
                                len(points), self.max_points, replace=False
                            )
                            points = points[indices]

                        data_list.append(
                            {
                                "points": points,
                                "label": self.category_to_label.get(category, -1),
                                "category": category,
                                "file_name": file_name,
                                "file_path": file_path,
                            }
                        )

                    except Exception as e:
                        print(f"Error loading {file_name}: {e}")
                        continue

            # This line is for demonstration purposes only. Remove in production.
            if len(data_list) > 10:
                break

        print(f"Loaded {len(data_list)} items from {len(self.categories)} categories")
        return data_list

    def _load_split_files(self, shapenet_path: str, split: str) -> set:
        """Load file list from train/test split JSON files."""
        split_file = os.path.join(
            shapenet_path, "train_test_split", f"shuffled_{split}_file_list.json"
        )

        # This part of the code is also incorrect. The JSON contains paths relative to the `shapenet_path`,
        # but the `kagglehub` library downloads to a temporary path, so the `split_file` path
        # will not exist. We need to load the JSON file from the correct Kaggle path.

        # Correct path to the JSON file
        correct_split_path = os.path.join(
            shapenet_path,  # Changed from kaggle_path to shapenet_path
            "train_test_split",
            f"shuffled_{split}_file_list.json",
        )

        if not os.path.exists(correct_split_path):
            print(f"Warning: Split file not found: {correct_split_path}")
            return set()

        # The JSON file contains paths like:
        # "shapenetcore_partanno_segmentation_benchmark_v0_normal/02691156/1000b0b8c5a4734790636a282f483c_9.npz"
        # We need to strip the prefix to match the relative paths.
        prefix = "shape_data/"

        try:
            import json

            with open(correct_split_path, "r") as f:
                file_list = json.load(f)

            # Convert the full paths to relative paths (e.g., "02691156/1000b0b8c5a4734790636a282f483c_9.npz")
            relative_paths = {path.replace(prefix, "") for path in file_list}
            print(f"Loaded {len(relative_paths)} files from {split} split")
            print(f"First 5 original paths: {list(file_list)[:5]}")
            print(f"First 5 relative paths: {list(relative_paths)[:5]}")
            airplane_relative = [
                path for path in relative_paths if path.startswith("02691156")
            ]
            print(f"Airplane relative paths: {len(airplane_relative)}")
            if airplane_relative:
                print(f"First 3 airplane paths: {airplane_relative[:3]}")
            return relative_paths
        except Exception as e:
            print(f"Error loading split file {correct_split_path}: {e}")
            return set()

    def _load_npz_file(self, file_path: str) -> Optional[np.ndarray]:
        """Load 3D point cloud from .npz file."""
        try:
            data = np.load(file_path)

            # Based on the examination, the point cloud data is in the 'pc' key
            if "pc" in data:
                points = data["pc"]
                print(f"Loaded point cloud from 'pc' key: shape={points.shape}")
            else:
                # Fallback to other possible keys
                possible_keys = ["points", "vertices", "coords", "pos", "xyz"]
                points = None
                for key in possible_keys:
                    if key in data:
                        points = data[key]
                        print(
                            f"Loaded point cloud from '{key}' key: shape={points.shape}"
                        )
                        break

                # If no standard key found, try the first array in the file
                if points is None and len(data.keys()) > 0:
                    first_key = list(data.keys())[0]
                    points = data[first_key]
                    print(
                        f"Using key '{first_key}' for points in {file_path}: shape={points.shape}"
                    )

            if points is not None:
                # Ensure points is 2D with 3 columns (x, y, z)
                if points.ndim == 1:
                    points = points.reshape(-1, 3)
                elif points.ndim == 2 and points.shape[1] != 3:
                    # If it's 2D but not 3 columns, take first 3 columns
                    points = points[:, :3]

                return points.astype(np.float32)
            else:
                print(f"No point data found in {file_path}")
                return None

        except Exception as e:
            print(f"Error loading .npz file {file_path}: {e}")
            return None

    def _explore_dataset_structure(self, kaggle_path: str):
        """Explore and print the actual structure of the Kaggle dataset."""
        print(f"Dataset root: {kaggle_path}")

        if not os.path.exists(kaggle_path):
            print(f"Error: Dataset path does not exist: {kaggle_path}")
            return

        print("Contents of dataset root:")
        for item in os.listdir(kaggle_path):
            item_path = os.path.join(kaggle_path, item)
            if os.path.isdir(item_path):
                print(f"  Directory: {item}")
                # Show first few files in each directory
                try:
                    files = os.listdir(item_path)[:5]  # Show first 5 files
                    for file in files:
                        print(f"    - {file}")
                    if len(os.listdir(item_path)) > 5:
                        print(
                            f"    ... and {len(os.listdir(item_path)) - 5} more files"
                        )
                except Exception as e:
                    print(f"    Error listing contents: {e}")
            else:
                print(f"  File: {item}")

    def create_data_object(self, raw_item: Dict[str, Any]) -> Data:
        """Convert raw ShapeNet item to PyTorch Geometric Data object."""
        points = torch.tensor(raw_item["points"], dtype=torch.float)
        category = raw_item["category"]

        # Generate per-point segmentation labels based on category
        # Since we don't have actual segmentation labels, we'll generate them
        num_points = len(points)

        # Get the segmentation classes for this category
        seg_classes = {
            "Airplane": [0, 1, 2, 3],
            "Bag": [4, 5],
            "Cap": [6, 7],
            "Car": [8, 9, 10, 11],
            "Chair": [12, 13, 14, 15],
            "Earphone": [16, 17, 18],
            "Guitar": [19, 20, 21],
            "Knife": [22, 23],
            "Lamp": [24, 25, 26, 27],
            "Laptop": [28, 29],
            "Motorbike": [30, 31, 32, 33, 34, 35],
            "Mug": [36, 37],
            "Pistol": [38, 39, 40],
            "Rocket": [41, 42, 43],
            "Skateboard": [44, 45, 46],
            "Table": [47, 48, 49],
        }

        if category in seg_classes:
            # Get the part labels for this category
            part_labels = seg_classes[category]
            # Randomly assign parts to points
            point_labels = torch.tensor(
                [part_labels[i % len(part_labels)] for i in range(num_points)],
                dtype=torch.long,
            )
        else:
            # Fallback: assign all points to class 0
            point_labels = torch.zeros(num_points, dtype=torch.long)

        # Convert category name to category index
        category_names = list(seg_classes.keys())
        if category in category_names:
            category_idx = category_names.index(category)
        else:
            category_idx = 0  # Fallback

        # Create PyTorch Geometric Data object
        # For PointNet2, we need 6 features: 3 for position + 3 for features
        # Since we only have coordinates, we'll use them for both
        # The PointNetConv will concatenate x and pos internally to get 6 features
        data = Data(
            x=points,  # Node features (3D coordinates)
            pos=points,  # Node positions (3D coordinates)
            y=point_labels,  # Per-point segmentation labels
            category=category_idx,  # Category index (integer)
            file_name=raw_item["file_name"],  # Original file name
        )

        # Only create KNN graph if no pre_transform is provided
        # if self.pre_transform is not None:
        #     data = self.pre_transform(data)

        return data

    def get_category_stats(self) -> Dict[str, int]:
        """Get statistics about categories in the dataset."""
        stats = {}
        for i in range(len(self)):
            data = self[i]
            # Check if category attribute exists and is not dummy
            if hasattr(data, "category") and data.category != "dummy":
                category = data.category
                stats[category] = stats.get(category, 0) + 1
            else:
                # Handle dummy data or missing category
                stats["dummy"] = stats.get("dummy", 0) + 1
        return stats
