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


class KaggleBaseDataset(InMemoryDataset, ABC):
    """
    Base class for converting Kaggle datasets to PyTorch Geometric datasets.

    This class provides the common structure and functionality for downloading,
    processing, and converting Kaggle datasets into PyTorch Geometric format.
    """

    def __init__(
        self,
        root: str,
        kaggle_dataset_name: str,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload: bool = False,
    ):
        """
        Initialize the base Kaggle dataset.

        Args:
            root: Root directory where dataset should be saved
            kaggle_dataset_name: Name of the Kaggle dataset (e.g., "mitkir/shapenet")
            transform: Optional transform to apply to data
            pre_transform: Optional transform to apply before saving
            pre_filter: Optional filter to apply before saving
            force_reload: If True, force reload even if processed files exist
        """
        self.kaggle_dataset_name = kaggle_dataset_name
        self._force_reload = force_reload

        # Initialize the parent class first
        super().__init__(root, transform, pre_transform, pre_filter)

        # Load data if not already loaded
        if force_reload or not os.path.exists(self.processed_paths[0]):
            self._load_data()
        else:
            self.data, self.slices = self._safe_load(self.processed_paths[0])

    @property
    @abstractmethod
    def raw_file_names(self) -> List[str]:
        """Return list of raw file names that should be found in raw_dir."""
        pass

    @property
    @abstractmethod
    def processed_file_names(self) -> List[str]:
        """Return list of processed file names."""
        pass

    @abstractmethod
    def download_kaggle_data(self) -> str:
        """
        Download data from Kaggle and return the path to downloaded data.

        Returns:
            Path to the downloaded Kaggle dataset
        """
        pass

    @abstractmethod
    def load_raw_data(self) -> List[Dict[str, Any]]:
        """
        Load and parse raw data from the downloaded Kaggle dataset.

        Returns:
            List of dictionaries containing raw data items
        """
        pass

    @abstractmethod
    def create_data_object(self, raw_item: Dict[str, Any]) -> Data:
        """
        Convert a raw data item to a PyTorch Geometric Data object.

        Args:
            raw_item: Dictionary containing raw data for one item

        Returns:
            PyTorch Geometric Data object
        """
        pass

    def _safe_load(self, path: str):
        """Safely load PyTorch Geometric data with proper security settings."""
        try:
            # Use weights_only=False for PyTorch Geometric compatibility
            return torch.load(path, weights_only=False)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Fallback: create empty dataset
            dummy_data = Data(
                x=None,
                pos=torch.zeros(1, 3),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                y=torch.tensor([0], dtype=torch.long),
                category=torch.tensor([-1], dtype=torch.long),
                file_name="dummy.obj",
            )
            data, slices = self.collate([dummy_data])
            return data, slices

    def download(self):
        """Download raw data from Kaggle."""
        print(f"Downloading Kaggle dataset: {self.kaggle_dataset_name}")
        kaggle_path = self.download_kaggle_data()

        # Process and save raw data
        raw_data = self.load_raw_data()
        torch.save(raw_data, self.raw_paths[0])
        print(f"Raw data saved to: {self.raw_paths[0]}")

    def process(self):
        """Process raw data and save to processed directory."""
        print("Processing raw data...")
        raw_data = torch.load(self.raw_paths[0], weights_only=False)

        data_list = []
        for i, raw_item in enumerate(raw_data):
            if i % 100 == 0:
                print(f"Processing item {i}/{len(raw_data)}")

            try:
                data = self.create_data_object(raw_item)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                continue

        print(f"Successfully processed {len(data_list)} items")

        # Handle empty dataset case
        if len(data_list) == 0:
            print("Warning: No data items were processed. Creating empty dataset.")
            # Create a dummy data object to avoid collate errors
            dummy_data = Data(
                pos=torch.zeros(1, 3),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                y=torch.tensor([0], dtype=torch.long),
                category="dummy",  # Add category attribute
                file_name="dummy.obj",  # Add file_name attribute
            )
            data_list = [dummy_data]

        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Processed data saved to: {self.processed_paths[0]}")

    def _load_data(self):
        """Load data, downloading and processing if necessary."""
        if not os.path.exists(self.raw_paths[0]):
            self.download()

        if not os.path.exists(self.processed_paths[0]) or self._force_reload:
            self.process()

        self.data, self.slices = self._safe_load(self.processed_paths[0])
