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
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.typing import WITH_TORCH_CLUSTER

from dlplay.datasets.modelnet import ModelNet


if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")

from dlplay.models.space3d.pointnet2_segmenter import PointNet2Segmenter
from dlplay.models.space3d.point_segmentation_train_test import train, test
from dlplay.utils.device import resolve_device, empty_device_cache
from dlplay.paths import DATA_DIR, RESULTS_DIR

from dlplay.datasets.point3d_datasets_helpers import (
    get_datasets,
    PointDatasetType,
    PointTask,
)


if __name__ == "__main__":

    dataset_type = PointDatasetType.ScanNet  # Use ScanNet or ShapeNet
    task = PointTask.Segmentation
    checkpoint_name = f"pointnet2_segmenter_{dataset_type.value.lower()}"

    num_epochs = 31
    batch_size = 1 if dataset_type == PointDatasetType.ScanNet else 12
    gradient_accumulation_steps = 2  # Accumulate gradients over 2 batches
    num_workers = (
        1 if dataset_type == PointDatasetType.ScanNet else 2
    )  # Reduce to save memory
    continue_training = False
    dataset_force_reload = False

    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")

    transform = T.Compose(
        [
            T.RandomJitter(0.01),
            T.RandomRotate(15, axis=0),
            T.RandomRotate(15, axis=1),
            T.RandomRotate(15, axis=2),
        ]
    )
    pre_transform = T.NormalizeScale()

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
    empty_device_cache(device)

    model = PointNet2Segmenter(train_dataset.num_classes)
    if continue_training:
        model_path = osp.join(RESULTS_DIR, "saved_models", f"{checkpoint_name}.pth")
        if not osp.exists(model_path):
            raise FileNotFoundError(
                f"Model path {model_path} does not exist, you need to train the model first"
            )
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Add mixed precision training
    scaler = torch.amp.GradScaler(device=device)

    best_iou = 0
    for epoch in range(1, 1 + num_epochs):
        train(
            model,
            train_loader,
            optimizer,
            device,
            gradient_accumulation_steps,
            scaler,
        )
        iou = test(model, test_loader, device, dataset_type)
        print(f"Epoch: {epoch:02d}, Test IoU: {iou:.4f}")
        if iou > best_iou:
            best_iou = iou
            torch.save(
                model.state_dict(),
                osp.join(
                    RESULTS_DIR,
                    "saved_models",
                    f"{checkpoint_name}_epoch{epoch}_iou{iou:.4f}.pth",
                ),
            )

        # Clear cache after each epoch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # save the model
    os.makedirs(osp.join(RESULTS_DIR, "saved_models"), exist_ok=True)
    torch.save(
        model.state_dict(),
        osp.join(RESULTS_DIR, "saved_models", f"{checkpoint_name}.pth"),
    )
