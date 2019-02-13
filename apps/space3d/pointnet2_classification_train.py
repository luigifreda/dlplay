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

import os.path as osp
import os

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius
from torch_geometric.typing import WITH_TORCH_CLUSTER

from dlplay.datasets.modelnet import ModelNet
from dlplay.utils.device import resolve_device
from dlplay.viz.view3d import viz_point_cloud
from dlplay.models.space3d.pointnet2_classifier import PointNet2Classifier, train, test

from dlplay.paths import DATA_DIR, RESULTS_DIR

from dlplay.datasets.point3d_datasets_helpers import (
    get_datasets,
    PointDatasetType,
    PointTask,
)


if __name__ == "__main__":

    dataset_type = PointDatasetType.ModelNet10
    task = PointTask.Classification
    checkpoint_name = f"pointnet2_classifier_{dataset_type.value.lower()}.pth"

    num_epochs = 200
    batch_size = 32
    num_workers = 6
    dataset_force_reload = False

    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    train_dataset, test_dataset, validation_dataset = get_datasets(
        dataset_type, task, dataset_force_reload, transform, pre_transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    device = resolve_device()
    print(f"Using device: {device}")
    model = PointNet2Classifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, num_epochs + 1):
        train_loss = train(epoch, model, device, optimizer, train_loader)
        test_acc = test(test_loader, model, device)
        print(
            f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.4f}"
        )

    # save the model
    os.makedirs(osp.join(RESULTS_DIR, "saved_models"), exist_ok=True)
    torch.save(
        model.state_dict(),
        osp.join(RESULTS_DIR, "saved_models", checkpoint_name),
    )
