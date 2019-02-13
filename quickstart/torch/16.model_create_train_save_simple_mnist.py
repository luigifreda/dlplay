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
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import random_split

from dlplay.paths import DATA_DIR, RESULTS_DIR
from dlplay.datasets.custom_subset import CustomSubset, DatasetSplitter

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


# In a single training loop, the model makes predictions on the training dataset (fed to it in batches),
# and backpropagates the prediction error to adjust the model’s parameters.
# The prediction error is computed using the loss function, and the optimizer adjusts the model’s parameters
# to minimize the loss.
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# We also check the model’s performance against the test dataset to ensure it is learning.
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


# from https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#optimizing-the-model-parameters
if __name__ == "__main__":

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
    optimizer = torch.optim.SGD(
        model.parameters(), lr=1e-3
    )  # stochastic gradient descent

    use_dataset_splitter = True

    if use_dataset_splitter:
        # Load all the data.
        full_dataset = datasets.FashionMNIST(
            root=f"{DATA_DIR}/datasets",
            train=True,
            download=True,
            transform=None,
        )

        splitter = DatasetSplitter(
            full_dataset,
            test_size=0.2,
            train_transform=ToTensor(),
            test_transform=ToTensor(),
        )
        training_dataset, test_dataset = splitter.split()
    else:
        training_dataset = datasets.FashionMNIST(
            root=f"{DATA_DIR}/datasets",
            train=True,
            download=True,
            transform=ToTensor(),
        )
        test_dataset = datasets.FashionMNIST(
            root=f"{DATA_DIR}/datasets",
            train=False,
            download=True,
            transform=ToTensor(),
        )

    # Create data loaders (for shuffling and batching)
    batch_size = 64

    # NOTE: Setting shuffle=True in a DataLoader means:
    # - At the beginning of each new epoch, the indices of the dataset are reshuffled before batching.
    # - So the training dataset will be presented in a new random order at every epoch.
    # - This behavior is standard practice for training — it prevents the model from overfitting to the fixed order of the data.
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    # NOTE: On the other hand, during testing, usually you want deterministic ordering in the test set,
    # so you’d normally set shuffle=False.
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # The training process is conducted over several iterations (epochs).
    # During each epoch, the model learns parameters to make better predictions.
    # We print the model’s accuracy and loss at each epoch;
    # we’d like to see the accuracy increase and the loss decrease with every epoch.
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    # A common way to save a model is to serialize the internal state dictionary (containing the model parameters).
    torch.save(
        model.state_dict(), f"{RESULTS_DIR}/saved_models/trained_mnist_model.pth"
    )
    print("Saved PyTorch Model State to model.pth")
