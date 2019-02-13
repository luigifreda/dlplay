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

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


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


# from https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#creating-models
#      https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
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

    # List of all parameters of the model. For non-trainable parameters, this will be empty.
    print(f"Model parameters:")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    # List of all buffers of the model. For non-trainable parameters, this will be empty.
    print(f"Model buffers:")
    for name, buffer in model.named_buffers():
        print(f"Buffer: {name} | Size: {buffer.size()} | Values : {buffer[:2]} \n")

    # List of all children modules of the model.
    print(f"Model children:")
    for name, child in model.named_children():
        print(f"Child: {name} | Type: {type(child)}")

    # To use the model, we pass it the input data. This executes the model’s forward,
    # along with some background operations. Do not call model.forward() directly!
    # The last linear layer of the neural network returns logits - raw values in [-infty, infty] -
    # which are passed to the nn.Softmax module. The logits are scaled to values [0, 1] representing
    # the model’s predicted probabilities for each class. dim parameter indicates the dimension
    # along which the values must sum to 1.
    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred.squeeze().cpu().numpy()}")
