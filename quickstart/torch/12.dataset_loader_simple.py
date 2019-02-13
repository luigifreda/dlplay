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
# A dataset composed by two tensors
import torch
from torch.utils.data import DataLoader


if __name__ == "__main__":

    # create a dataset composed by two tensors
    num_samples = 1000
    num_features = 3
    output_dim = 1
    dataset = torch.utils.data.TensorDataset(
        torch.randn(num_samples, num_features),
        torch.randn(num_samples, output_dim),
    )

    # The data loader provides shuffling and mini-batching
    batch_size = 32

    # NOTE: Setting shuffle=True in a DataLoader means:
    # - At the beginning of each new epoch, the indices of the dataset are reshuffled before batching.
    # - So the training dataset will be presented in a new random order at every epoch.
    # - This behavior is standard practice for training — it prevents the model from overfitting to the fixed order of the data.
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}")
        print("-" * 10)

        # Iterating over mini-batches (one epoch)
        for xb, yb in dataloader:
            # xb has shape (batch_size, num_features)
            # yb has shape (batch_size, output_dim)
            print(f"xb shape: {xb.shape}, yb shape: {yb.shape}")
