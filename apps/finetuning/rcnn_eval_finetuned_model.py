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
from dlplay.models.fine_tuning_rcnn import (
    build_maskrcnn_resnet50_fpn,
    build_maskrcnn_mobilenetv2_fpn,
)
from dlplay.models.model_helpers import finetuned_model_name
from dlplay.core.transform_helpers import get_transforms_basic, get_transforms_affine
from dlplay.utils.device import resolve_device

from dlplay.paths import DATA_DIR, RESULTS_DIR
import matplotlib.pyplot as plt
import random
import os


# from https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# Here we load a finetuned model and evaluate it on the test set


build_model_dict = {
    "maskrcnn_resnet50_fpn": (
        build_maskrcnn_resnet50_fpn,
        {"finetune_head_only": False},  # args for build_model
    ),
    "maskrcnn_mobilenetv2_fpn": (
        build_maskrcnn_mobilenetv2_fpn,
        {"finetune_head_only": False},
    ),
}


if __name__ == "__main__":
    # In this tutorial, we evaluate the finetuned model on the test set.

    # False: evaluate on test set, True: just evaluate on a random sample of images
    evaluate_on_random_images = True

    # our dataset has two classes only - background and person
    num_classes = 2  # 1 class (person) + background

    model_to_eval_name = "maskrcnn_resnet50_fpn"
    # model_to_eval_name = "maskrcnn_mobilenetv2_fpn"  # an experiment just for fun
    build_model, build_model_args = build_model_dict[model_to_eval_name]

    model, backbone_model, head_model = build_model(num_classes, **build_model_args)

    device = resolve_device()

    # Load the model
    saved_model_name = finetuned_model_name(
        model,
        backbone_model,
        head_model,
        build_model_args["finetune_head_only"],
        num_classes,
    )
    print(f"Loading model: {saved_model_name}")
    model_checkpoint_path = f"{RESULTS_DIR}/saved_models/{saved_model_name}.pth"
    if not os.path.exists(model_checkpoint_path):
        print(f"Model checkpoint not found at {model_checkpoint_path}")
        print(
            f"You first need to train the model by running the rcnn_finetuning_replace_XXX.py script"
        )
        exit()
    model.load_state_dict(torch.load(model_checkpoint_path))
    model.to(device)

    # load index permutation of images in the test set
    with open(f"{RESULTS_DIR}/saved_models/{saved_model_name}_test_set.txt", "r") as f:
        test_image_names = [line.strip() for line in f.readlines()]
        test_image_paths = [
            f"{DATA_DIR}/datasets/PennFudanPed/PNGImages/{image_name}"
            for image_name in test_image_names
        ]

    get_transforms = get_transforms_basic

    # Init dataset and dataloader
    dataset_full = PennFudanDataset(
        root=f"{DATA_DIR}/datasets/PennFudanPed",
        transforms=get_transforms(train=False),
    )

    test_images_indices = []
    for name in test_image_names:
        # search for the index of the image in the dataset
        index = dataset_full.imgs.index(name)
        test_images_indices.append(index)

    # for comparison, let's add the image of the tutorial to the test set
    tutorial_test_image = "FudanPed00046.png"
    tutorial_test_image_index = dataset_full.imgs.index(tutorial_test_image)
    test_image_names.append(tutorial_test_image)
    test_image_paths.append(
        f"{DATA_DIR}/datasets/PennFudanPed/PNGImages/{tutorial_test_image}"
    )
    test_images_indices.append(tutorial_test_image_index)

    dataset_test = torch.utils.data.Subset(dataset_full, test_images_indices)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=detection_utils.collate_fn,
    )

    num_images_w = 5
    num_images_h = 5

    if evaluate_on_random_images:

        plt.figure(figsize=(15, 15))

        # select random images from the test set
        num_test_images = len(test_image_names)
        num_samples = min(num_images_w * num_images_h, num_test_images)
        random_ids = random.sample(range(num_test_images), num_samples)

        # replace the last image with the tutorial image
        random_ids[-1] = len(dataset_test) - 1

        for i in range(num_samples):
            random_id = random_ids[i]
            random_image_name = test_image_names[random_id]
            random_image_path = test_image_paths[random_id]
            print(f"Evaluating image: {random_image_name}")
            image = torchvision.io.read_image(random_image_path).to(device)
            eval_transform = get_transforms(train=False)

            model.eval()
            with torch.no_grad():
                x = eval_transform(image)
                # convert RGBA -> RGB and move to device
                x = x[:3, ...].to(device)
                predictions = model(
                    [
                        x,
                    ]
                )
                pred = predictions[0]

            image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(
                torch.uint8
            )
            image = image[:3, ...]
            pred_labels = [
                f"pedestrian: {score:.3f}"
                for label, score in zip(pred["labels"], pred["scores"])
            ]
            pred_boxes = pred["boxes"].long()
            output_image = torchvision.utils.draw_bounding_boxes(
                image, pred_boxes, pred_labels, colors="red"
            )

            masks = (pred["masks"] > 0.7).squeeze(1)
            output_image = torchvision.utils.draw_segmentation_masks(
                output_image, masks, alpha=0.5, colors="blue"
            )

            plt.subplot(num_images_h, num_images_w, i + 1)
            plt.imshow(output_image.permute(1, 2, 0))
            plt.title(random_image_name)
            plt.axis("off")

        plt.tight_layout()
        plt.show()
    else:
        # evaluate on the test dataset
        print("Evaluating on the full test dataset")
        detection_engine.evaluate(model, data_loader_test, device=device)
