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
from datasets import load_dataset, Image

import torch
from torchvision.transforms import Compose, ColorJitter, ToTensor
from torch.utils.data import DataLoader

import albumentations
import numpy as np

# from https://huggingface.co/docs/datasets/en/quickstart#vision
if __name__ == "__main__":

    # Image datasets are loaded just like text datasets. However, instead of a tokenizer,
    # you’ll need a feature extractor to preprocess the dataset. Applying data augmentation
    # to an image is common in computer vision to make the model more robust against overfitting.
    # You’re free to use any data augmentation library you want, and then you can apply the
    # augmentations with 🤗 Datasets. In this quickstart, you’ll load the Beans dataset and get
    # it ready for the model to train on and identify disease from the leaf images.

    # 1. Load the Beans dataset by providing the load_dataset() function with the dataset name and a dataset split:
    dataset = load_dataset("AI-Lab-Makerere/beans", split="train")

    # Most image models work with RBG images. If your dataset contains images in a different mode,
    # you can use the cast_column() function to set the mode to RGB:
    # (The Beans dataset contains only RGB images, so this step is unnecessary here.)
    dataset = dataset.cast_column("image", Image(mode="RGB"))

    # 2. Now you can add some data augmentations with any library (Albumentations, imgaug, Kornia) you like.
    # Here, you’ll use torchvision to randomly change the color properties of an image:
    jitter = Compose([ColorJitter(brightness=0.5, hue=0.5), ToTensor()])

    # 3. Create a function to apply your transform to the dataset and generate the model input: pixel_values.
    def transforms(data_examples):
        data_examples["pixel_values"] = [
            jitter(image.convert("RGB")) for image in data_examples["image"]
        ]
        return data_examples

    # 4. Use the with_transform() function to apply the data augmentations on-the-fly:
    # dataset = dataset.with_transform(transforms)
    dataset = dataset.map(transforms, batched=True)

    # 5. Set the dataset format according to the machine learning framework you’re using.
    dataset.set_format(type="torch", columns=["pixel_values", "labels"])

    # ===============================
    # PyTorch
    # ===============================

    def collate_fn(examples):
        # Extract the pixel_values and labels from the examples
        # and stack them into a single tensor
        images = []
        labels = []
        for example in examples:
            images.append((example["pixel_values"]))
            labels.append(example["labels"])

        pixel_values = torch.stack(images)
        labels = torch.tensor(labels)
        return {"pixel_values": pixel_values, "labels": labels}

    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=4)
