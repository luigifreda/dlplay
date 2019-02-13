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
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

from dlplay.paths import DATA_DIR, RESULTS_DIR
from dlplay.models.space3d.point_transformer_classifier import (
    PointTransformerClassifier,
    train,
    test,
)
from dlplay.utils.device import resolve_device
from dlplay.datasets.point3d_datasets_helpers import (
    get_datasets,
    PointDatasetType,
    PointTask,
)


if __name__ == "__main__":

    dataset_type = PointDatasetType.ModelNet10
    task = PointTask.Classification
    checkpoint_name = f"point_transformer_classifier_{dataset_type.value.lower()}.pth"

    num_epochs = 200
    batch_size = 32
    dataset_force_reload = False
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)

    train_dataset, test_dataset, validation_dataset = get_datasets(
        dataset_type, task, dataset_force_reload, transform, pre_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = resolve_device()
    print(f"Using device: {device}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Number of classes: {train_dataset.num_classes}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Train loader size: {len(train_loader)}")
    print(f"Test loader size: {len(test_loader)}")
    print(f"Batch size: {batch_size}")

    model = PointTransformerClassifier(
        0,
        train_dataset.num_classes,
        dim_model=[32, 64, 128, 256, 512],
        k=16,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(1, num_epochs + 1):
        loss = train(model, optimizer, train_dataset, train_loader, device)
        test_acc = test(model, test_loader, device)
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}")
        scheduler.step()

    # save the model
    os.makedirs(osp.join(RESULTS_DIR, "saved_models"), exist_ok=True)
    torch.save(
        model.state_dict(),
        osp.join(RESULTS_DIR, "saved_models", checkpoint_name),
    )
