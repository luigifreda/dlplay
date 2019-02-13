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
import torch
import torchvision

from torchvision.transforms import v2 as T

from dlplay.datasets.pennfudan_dataset import PennFudanDataset
import dlplay.detection.utils as detection_utils
import dlplay.detection.coco_engine as detection_engine
from dlplay.models.model_helpers import (
    dump_optimized_model_layers,
    finetuned_model_name,
)
from dlplay.paths import DATA_DIR, RESULTS_DIR
from dlplay.models.fine_tuning_rcnn import build_maskrcnn_resnet50_fpn
from dlplay.core.transform_helpers import (
    get_transforms_basic,
    get_transforms_affine,
    get_transforms_basic_color,
    get_transforms_affine_color,
)
from dlplay.core.training_functions import train_model_SGD, train_model_AdamW
from dlplay.datasets.custom_subset import DatasetSplitter
from dlplay.utils.device import resolve_device

from dlplay.core.training import (
    SimpleTrainer,
    OptimizerType,
    LearningRateSchedulerType,
)
from dlplay.core.evaluation import TaskType

# from https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# For this tutorial, we will be finetuning a pre-trained Mask R-CNN model
# on the Penn-Fudan Database for Pedestrian Detection and Segmentation.
# It contains 170 images with 345 instances of pedestrians, and we will use
# it to illustrate how to use the new features in torchvision in order
# to train an object detection and instance segmentation model on a custom dataset.
# NOTE: One note on the labels. The model considers class 0 as background.
#       If your dataset does not contain the background class, you should not have 0 in your labels.


if __name__ == "__main__":
    # In this tutorial, we will be using Mask R-CNN, which is based on top of Faster R-CNN.
    # Faster R-CNN is a model that predicts both bounding boxes and class scores for potential objects in the image.

    # NOTE: There are two common situations where one might want to modify one of the available models in TorchVision Model Zoo.
    # 1. The first is when we want to start from a pre-trained model, and just finetune the last layer.
    # 2. The other is when we want to replace the backbone of the model with a different one (for faster predictions, for example).

    # You can explore the Penn-Fudan Database for Pedestrian Detection
    # by running quickstart/torch/11.dataset_custom_pennfudan_exploring.py

    # In our case, we want to finetune from a pre-trained model, given that our dataset is very small,
    # so we will be following approach number 1.

    # 1 - Finetuning from a pretrained model

    # our dataset has two classes only - background and person
    num_classes = 2  # 1 class (person) + background
    just_finetune_head = False

    # If we want to fine-tune the model for object detection, we can use the get_model_to_finetune_for_object_detection() function.
    # model, backbone_model, head_model = get_model_to_finetune_for_object_detection(
    #     num_classes, just_finetune_head
    # )

    # If we want to fine-tune the model for instance segmentation, we can use the get_model_to_finetune_for_instance_segmentation() function.
    model, backbone_model, head_model = build_maskrcnn_resnet50_fpn(
        num_classes, just_finetune_head
    )

    saved_model_name = finetuned_model_name(
        model, backbone_model, head_model, just_finetune_head, num_classes
    )
    print(f"model_name: {saved_model_name}")

    # train on the GPU or on the CPU, if a GPU is not available
    device = resolve_device()
    model.to(device)

    # NOTE: Before iterating over the dataset, it’s good to see what the model expects during training
    # and inference time on sample data.

    # For Training
    # images, targets = next(iter(data_loader))
    # images = list(image for image in images)
    # targets = [{k: v for k, v in t.items()} for t in targets]
    # output = model(images, targets)  # Returns losses and detections
    # print(output)

    # For inference
    # model.eval()
    # x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    # predictions = model(x)  # Returns predictions
    # print(predictions[0])

    get_transforms = get_transforms_basic
    # get_transforms = get_transforms_affine_color

    # 2 - Init dataset and dataloader

    split_dataset = True  # This is just didactic/educational. In the original tutorial, the finetuning was done without applying any proper dataset split
    print(f"split dataset: {split_dataset}")
    if split_dataset:
        # The original tutorial didn't properly split the dataset, so we need to do it here
        dataset_full = PennFudanDataset(
            root=f"{DATA_DIR}/datasets/PennFudanPed",
            transforms=None,
        )

        # split the dataset in train and test set
        splitter = DatasetSplitter(
            dataset_full,
            test_size=0.1,
            train_transform=get_transforms(train=True),
            test_transform=get_transforms(train=False),
        )
        dataset_train, dataset_test = splitter.split()
    else:
        # You can check here the peformance of the original tutorial without splitting the dataset
        dataset_train = PennFudanDataset(
            root=f"{DATA_DIR}/datasets/PennFudanPed",
            transforms=get_transforms(train=True),
        )
        dataset_test = PennFudanDataset(
            root=f"{DATA_DIR}/datasets/PennFudanPed",
            transforms=get_transforms(train=False),
        )

    print(f"num_train: {len(dataset_train)}, num_test: {len(dataset_test)}")

    data_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=2,
        shuffle=True,
        collate_fn=detection_utils.collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        collate_fn=detection_utils.collate_fn,
    )

    num_epochs = 20
    train_model = train_model_SGD

    # define a post-epoch callback (called after each epoch) to evaluate and save the best model
    class PostEpochCallback:
        def __init__(self, saved_model_name):
            self.saved_model_name = saved_model_name
            self.best_map = 0.0

        def __call__(self, model, data_loader_test, device):

            if self.saved_model_name:
                # evaluate and save best model
                coco_evaluator = detection_engine.evaluate(
                    model, data_loader_test, device=device
                )

                map_50 = coco_evaluator.coco_eval["bbox"].stats[1]  # mAP@0.5
                if map_50 > self.best_map:
                    self.best_map = map_50
                    torch.save(
                        model.state_dict(),
                        f"{RESULTS_DIR}/saved_models/{self.saved_model_name}_best.pth",
                    )

    train_model(
        model=model,
        data_loader_train=data_loader,
        data_loader_test=data_loader_test,
        train_one_epoch_fn=detection_engine.train_one_epoch,
        post_epoch_callback=PostEpochCallback(saved_model_name),
        device=device,
        optimizer_dict={"lr": 0.005, "momentum": 0.9, "weight_decay": 0.0005},
        lr_scheduler_dict={"step_size": 3, "gamma": 0.1},
        lr_scheduler_type=LearningRateSchedulerType.STEP,
        print_freq=10,
        num_epochs=num_epochs,
    )

    # 5 - Save the model
    torch.save(model.state_dict(), f"{RESULTS_DIR}/saved_models/{saved_model_name}.pth")

    # save the train/test permutation of the dataset
    with open(f"{RESULTS_DIR}/saved_models/{saved_model_name}_test_set.txt", "w") as f:
        if split_dataset:
            for i in dataset_test.indices:
                f.write(f"{dataset_test.base_dataset.imgs[i]}\n")
        else:
            for i in dataset_test.indices:
                f.write(f"{dataset_test.imgs[i]}\n")

    print(f"Model {saved_model_name} saved!")
