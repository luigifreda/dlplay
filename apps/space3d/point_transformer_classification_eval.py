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

import torch
import torch.nn.functional as F
import random

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from dlplay.paths import DATA_DIR, RESULTS_DIR
from dlplay.models.space3d.point_transformer_classifier import (
    PointTransformerClassifier,
    test,
)
from dlplay.utils.device import resolve_device
from dlplay.viz.view3d import viz_point_cloud

from dlplay.datasets.point3d_datasets_helpers import (
    get_datasets,
    PointDatasetType,
    PointTask,
    get_categories_names,
)


if __name__ == "__main__":

    dataset_type = PointDatasetType.ModelNet10
    task = PointTask.Classification
    checkpoint_name = f"point_transformer_classifier_{dataset_type.value.lower()}.pth"

    batch_size = 32
    num_workers = 6
    dataset_force_reload = False
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)

    test_dataset, test_dataset, validation_dataset = get_datasets(
        dataset_type,
        task,
        dataset_force_reload,
        transform,
        pre_transform,
        only_test=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    device = resolve_device()
    print(f"Using device: {device}")

    model = PointTransformerClassifier(
        0,
        test_dataset.num_classes,
        dim_model=[32, 64, 128, 256, 512],
        k=16,
    )
    model_path = osp.join(
        RESULTS_DIR,
        "saved_models",
        checkpoint_name,
    )
    if not osp.exists(model_path):
        raise FileNotFoundError(
            f"Model path {model_path} does not exist, you need to train the model first"
        )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    test_acc = test(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Test the model on some samples
    # get some random samples from the test dataset
    num_samples = 5
    random_samples = random.sample(list(test_dataset), num_samples)

    # Define the categories for ModelNet10
    categories = get_categories_names(dataset_name)

    # Create a batch from the random samples
    batch_data = Batch.from_data_list(random_samples).to(device)

    with torch.no_grad():
        out = model(batch_data, batch_data.pos)
        gt = batch_data.y

        for i in range(num_samples):
            # Get the prediction for this sample
            sample_out = out[i]
            sample_gt = gt[i]

            # get the category name
            gt_category_name = categories[sample_gt.item()]
            pred_category_name = categories[sample_out.argmax().item()]
            print(
                f"Sample {i+1} - Predicted: {pred_category_name}, Ground Truth: {gt_category_name}"
            )

            # Visualize individual point cloud
            sample_data = random_samples[i].to(device)
            viz_point_cloud(
                data_dict={"data": sample_data, "title": f"Point Cloud {i+1}"}
            )
