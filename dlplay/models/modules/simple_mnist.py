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
from torch import nn


# Neural networks comprise of layers/modules that perform operations on data.
# The torch.nn namespace provides all the building blocks you need to build your own neural network.
# Every module in PyTorch subclasses the nn.Module. A neural network is a module itself that consists
# of other modules (layers). This nested structure allows for building and managing complex architectures easily.


# Define model for the MNIST dataset
# To define a neural network in PyTorch, we create a class that inherits from nn.Module.
# We define the layers of the network in the __init__ function and specify how data will pass
# through the network in the forward function. To accelerate operations in the neural network,
# we move it to the accelerator such as CUDA, MPS, MTIA, or XPU.
# If the current accelerator is available, we will use it. Otherwise, we use the CPU.
class NeuralNetworkMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        # We initialize the nn.Flatten layer to convert each 2D 28x28 image into a contiguous array
        # of 784 pixel values (the minibatch dimension (at dim=0) is maintained).
        self.flatten = nn.Flatten()
        # We define a sequence of layers that will be used to process the input data.
        # The sequence is defined using the nn.Sequential container.
        # The nn.Sequential container is a convenient way to combine a sequence of layers
        # into a single module, where the output from one layer is used as the input to the next layer.
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=512),  # input layer
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),  # hidden layer
            nn.Linear(in_features=512, out_features=10),  # output layer
        )
        # Calling the model on the input returns a 2-dimensional tensor with dim=0 corresponding to each output of 10
        # raw predicted values for each class, and dim=1 corresponding to the individual values of each output.
        # We get the prediction probabilities by passing it through an instance of the nn.Softmax module.

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


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

    model = NeuralNetworkMNIST().to(device)
    print(f"Model structure: {model}")

    # To train a model, we need a 1. loss function and 2. an optimizer.
    loss_fn = nn.CrossEntropyLoss()  # cross-entropy loss
    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=1e-3
    # )  # stochastic gradient descent

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
        torch.save(
            model.state_dict(), f"{RESULTS_DIR}/saved_models/trained_mnist_model.pth"
        )
        print("Saved PyTorch Model State to model.pth")
