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

import copy
from typing import Sequence, Union, Optional

import torch
from torch.utils.data import Dataset, Subset  # standard import path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class CustomSubset(Dataset):
    """
    Custom subset of a dataset with optional per-subset transforms.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        indices_or_subset: Union[Subset, Sequence[int], torch.Tensor],
        transform=None,
        target_transform=None,
    ):
        # Shallow copy so we can safely tweak attributes.
        self.base_dataset = copy.copy(base_dataset)

        if transform is not None or target_transform is not None:
            # TorchVision datasets typically expose these attrs.
            for attr in ("transform", "target_transform", "transforms"):
                if (
                    hasattr(self.base_dataset, attr)
                    and getattr(self.base_dataset, attr) is not None
                ):
                    # Avoid noisy prints in library code; drop or use logging if preferred.
                    setattr(self.base_dataset, attr, None)

        # Extract and normalize the indices
        if isinstance(indices_or_subset, Subset):
            self.indices = torch.as_tensor(indices_or_subset.indices, dtype=torch.long)
        else:
            self.indices = torch.as_tensor(indices_or_subset, dtype=torch.long)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.indices.numel()

    def __getitem__(self, i):
        idx = int(self.indices[i])
        sample = self.base_dataset[idx]

        # Support common dataset return shapes
        if isinstance(sample, tuple):
            if len(sample) < 2:
                raise ValueError("Expected dataset to return at least (x, y).")
            x, y = sample[0], sample[1]
        elif isinstance(sample, dict):
            # Try common dict keys
            try:
                x, y = sample["image"], sample["label"]
            except KeyError:
                raise ValueError("Dict sample must have 'image' and 'label' keys.")
        else:
            raise ValueError("Unsupported sample type; expected tuple or dict.")

        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y


class DatasetSplitter:
    """
    Split a dataset into train/test, optionally stratified and with per-split transforms.
    """

    def __init__(
        self,
        full_dataset: Dataset,
        test_size: float = 0.2,
        train_transform=None,
        train_target_transform=None,
        test_transform=None,
        test_target_transform=None,
        random_state: int = 42,
        stratify: bool = False,  # Boolean flag; when True we compute labels and pass to sklearn
    ):
        self.full_dataset = full_dataset
        self.test_size = test_size
        self.random_state = random_state
        self.train_transform = train_transform
        self.train_target_transform = train_target_transform
        self.test_transform = test_transform
        self.test_target_transform = test_target_transform
        self.stratify = stratify

    def _collect_targets(self) -> Sequence[int]:
        # Try fast attributes used by many torchvision datasets
        if hasattr(self.full_dataset, "targets"):
            return list(self.full_dataset.targets)
        if hasattr(self.full_dataset, "labels"):
            return list(self.full_dataset.labels)

        # Fallback: iterate without shuffling; try not to break if y is list/ndarray/tensor/scalar
        targets = []
        loader = DataLoader(self.full_dataset, batch_size=64, shuffle=False)
        for batch in loader:
            # Expect (x, y, ...) structure
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                batch_targets = batch[1]
            elif isinstance(batch, dict) and "label" in batch:
                batch_targets = batch["label"]
            else:
                raise ValueError("Cannot infer targets from dataset batches.")

            if isinstance(batch_targets, torch.Tensor):
                targets.extend(batch_targets.detach().cpu().tolist())
            else:
                # list/tuple/ndarray/scalar
                try:
                    targets.extend(list(batch_targets))
                except TypeError:
                    targets.append(int(batch_targets))
        return targets

    def split(self):
        dataset_indices = list(range(len(self.full_dataset)))

        targets: Optional[Sequence[int]] = None
        if self.stratify:
            targets = self._collect_targets()

        train_indices, test_indices = train_test_split(
            dataset_indices,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=targets,
        )

        train_dataset = CustomSubset(
            self.full_dataset,
            train_indices,
            self.train_transform,
            self.train_target_transform,
        )
        test_dataset = CustomSubset(
            self.full_dataset,
            test_indices,
            self.test_transform,
            self.test_target_transform,
        )
        return train_dataset, test_dataset
