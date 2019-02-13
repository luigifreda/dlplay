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
import random

import torch
import torch.nn.functional as F
from point_transformer_classification import TransformerBlock, TransitionDown
from torchmetrics.functional import jaccard_index

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from dlplay.models.space3d.point_transformer_segmenter import PointTransformerSegmenter
from dlplay.models.space3d.point_segmentation_train_test import train, test
from dlplay.segmentation.color_segmentation import (
    segment_labels_to_colors,
)
from dlplay.utils.device import resolve_device
from dlplay.viz.viz_process import VizProcess, viz_point_cloud
from dlplay.datasets.point3d_datasets_helpers import (
    PointDatasetType,
    PointTask,
    get_datasets,
)


if __name__ == "__main__":

    dataset_type = PointDatasetType.ShapeNet
    task = PointTask.Segmentation
    checkpoint_name = f"point_transformer_segmenter_{dataset_type.value.lower()}.pth"

    dataset_force_reload = False
    transform = T.Compose(
        [
            T.RandomJitter(0.01),
            T.RandomRotate(15, axis=0),
            T.RandomRotate(15, axis=1),
            T.RandomRotate(15, axis=2),
        ]
    )
    pre_transform = T.NormalizeScale()

    # NOTE: Shapenet provides object part segmentation data.
    test_dataset, test_dataset, validation_dataset = get_datasets(
        dataset_type,
        task,
        dataset_force_reload,
        transform,
        pre_transform,
        only_test=True,
    )

    test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=6)

    device = resolve_device()
    print(f"Using device: {device}")
    model = PointTransformerSegmenter(
        in_channels=3,
        out_channels=test_dataset.num_classes,
        dim_model=[32, 64, 128, 256, 512],
        k=16,
    )

    iou = test(model, test_loader, device)
    print(f"Test IoU: {iou:.4f}")

    # Test the model on some samples
    # get some random samples from the test dataset
    num_samples = 5
    random_samples = random.sample(list(test_dataset), num_samples)

    # Create a batch from the random samples
    batch_data = Batch.from_data_list(random_samples).to(device)

    with torch.no_grad():
        out = model(batch_data)
        gt = batch_data.y

        # Calculate cumulative point counts for each sample
        point_counts = [len(sample.pos) for sample in random_samples]
        cumsum_counts = torch.cumsum(torch.tensor([0] + point_counts), dim=0)

        for i in range(num_samples):
            # Get the correct slice for this sample
            start_idx = cumsum_counts[i].item()
            end_idx = cumsum_counts[i + 1].item()

            sample_out = out[start_idx:end_idx]  # [num_points, num_classes]
            sample_gt = gt[start_idx:end_idx]  # [num_points]

            # get the predicted segment labels
            pred_segment_labels = sample_out.argmax(dim=-1)
            # get the ground truth segment labels
            gt_segment_labels = sample_gt

            # transform the segment labels into colors
            pred_segment_colors = segment_labels_to_colors(
                pred_segment_labels, test_dataset.num_classes
            )
            gt_segment_colors = segment_labels_to_colors(
                gt_segment_labels, test_dataset.num_classes
            )

            # Visualize individual point cloud
            sample_data = random_samples[i].to(device)

            # Convert tensors to CPU and numpy before passing to VizProcess
            sample_data_cpu = sample_data.cpu()
            pred_segment_colors_cpu = pred_segment_colors.cpu().numpy()
            gt_segment_colors_cpu = gt_segment_colors.cpu().numpy()

            viz_process = VizProcess(
                viz_point_cloud,
                data_dict={
                    "data": sample_data_cpu,
                    "title": f"Point Cloud {i+1} - Predicted",
                    "colors": pred_segment_colors_cpu,
                },
            )

            # Also create a ground truth visualization
            viz_process_gt = VizProcess(
                viz_point_cloud,
                data_dict={
                    "data": sample_data_cpu,
                    "title": f"Point Cloud {i+1} - Ground Truth",
                    "colors": gt_segment_colors_cpu,
                },
            )

            # Wait for both visualizations to be closed and then continue
            # Click on the windows and press 'q' to close them
            VizProcess.wait_all([viz_process, viz_process_gt])
