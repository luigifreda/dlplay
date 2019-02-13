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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from einops import rearrange, reduce, asnumpy, parse_shape
from einops.layers.torch import Rearrange, Reduce


# from https://einops.rocks/pytorch-examples.html


class SimpleConvNet(nn.Module):
    """
    Simple convolutional neural network.
    The output is a log-softmax of the 10 classes.
    """

    def __init__(self, num_classes: int = 10):
        super(SimpleConvNet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(
            1, 10, kernel_size=5
        )  # 1 input channel, 10 output channels
        self.conv2 = nn.Conv2d(
            10, 20, kernel_size=5
        )  # 10 input channels, 20 output channels
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)  # 320 input features, 50 hidden units
        self.fc2 = nn.Linear(50, num_classes)  # 10 classes

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)  # flatten the output of the last conv layer
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class SimpleConvNetEinops(nn.Module):
    """
    Reason to prefer einops version:
    - in the original code (to the left) if input size is changed and batch size is divisible by 16
      (that's usually so), we'll get something senseless after reshaping
    - new code will explicitly raise an error in this case
    - we won't forget to use dropout with flag self.training with new version
    - code is straightforward to read and analyze
    - sequential makes printing / saving / passing trivial. And there is no need in your code to
      load a model (which also has a number of benefits)
    - don't need logsoftmax? Now you can use self.model[:-1]. One more reason to prefer nn.Sequential
    - ... and we could also add inplace for ReLU
    """

    def __init__(self, num_classes: int = 10):
        super(SimpleConvNetEinops, self).__init__()
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),  # 1 input channel, 10 output channels
            nn.MaxPool2d(kernel_size=2),  # 2x2 max pooling
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),  # 10 input channels, 20 output channels
            nn.MaxPool2d(kernel_size=2),  # 2x2 max pooling
            nn.ReLU(),
            nn.Dropout2d(),
            Rearrange(
                "b c h w -> b (c h w)"
            ),  # flatten the output of the last conv layer
            nn.Linear(320, 50),  # 320 input features, 50 hidden units
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


# def train_convnet(convnet, data_loader, optimizer, device, epochs=10):
#     convnet.train()
#     for epoch in range(epochs):
#         for batch_idx, (data, target) in enumerate(data_loader):
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
#             output = convnet(data)
#             loss = F.nll_loss(output, target)
#             loss.backward()
#             optimizer.step()
#             if batch_idx % 100 == 0:
#                 print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f}")
#     return loss.item()


# Simple training of a model on the MNIST dataset
if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torchvision.transforms import ToTensor

    from dlplay.core.training import (
        SimpleTrainer,
        OptimizerType,
        LearningRateSchedulerType,
    )
    from dlplay.utils.device import resolve_device
    from dlplay.core.evaluation import TaskType
    from dlplay.paths import DATA_DIR, RESULTS_DIR

    # We want to be able to train our model on an accelerator such as CUDA, MPS, MTIA, or XPU.
    # If the current accelerator is available, we will use it. Otherwise, we use the CPU.
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # model = SimpleConvNet().to(device)
    model = SimpleConvNetEinops().to(device)
    print(f"Model structure: {model}")

    # To train a model, we need a 1. loss function and 2. an optimizer.
    loss_fn = nn.CrossEntropyLoss()  # cross-entropy loss

    # Load the training and test datasets.
    training_data = datasets.FashionMNIST(
        root=f"{DATA_DIR}/datasets", train=True, download=True, transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root=f"{DATA_DIR}/datasets", train=False, download=True, transform=ToTensor()
    )

    # Create data loaders (for shuffling and batching)
    batch_size = 64

    # NOTE: Setting shuffle=True in a DataLoader means:
    # - At the beginning of each new epoch, the indices of the dataset are reshuffled before batching.
    # - So the training dataset will be presented in a new random order at every epoch.
    # - This behavior is standard practice for training — it prevents the model from overfitting to the fixed order of the data.
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    # NOTE: On the other hand, during testing, usually you want deterministic ordering in the test set,
    # so you’d normally set shuffle=False.
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    num_epochs = 10
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = SimpleTrainer(
        model=model,
        dataloader=train_dataloader,
        dataloader_test=test_dataloader,
        optimizer_type=OptimizerType.SGD,
        lr_scheduler_type=LearningRateSchedulerType.COSINE,
        loss_fn=loss_fn,
        task_type=TaskType.CLASSIFICATION,
        device=device,
        num_epochs=num_epochs,
        print_freq=100,
    )
    trainer.train_model()

    # A common way to save a model is to serialize the internal state dictionary (containing the model parameters).
    if False:
        filename = f"{RESULTS_DIR}/saved_models/trained_mnist_model.pth"
        torch.save(model.state_dict(), filename)
        print(f"Saved torch model state to {filename}")
