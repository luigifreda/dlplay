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


if __name__ == "__main__":
    #  Autograd Basics
    x = torch.tensor(
        [2.0], requires_grad=True
    )  # Set requires_grad=True tells PyTorch to track computations involving x, so you can compute gradients later.
    y = x**3 + 2 * x
    # y(x=2) = 2^3 + 2*2 = 8 + 4 = 12

    y.backward()  # Compute the gradient of y with respect to x
    # “It will compute dy/dx and it will store that in x.grad, because x is the variable
    # you're differentiating with respect to.”

    # dy/dx = 3*x^2 + 2
    # dy/dx(x=2) = 3*2^2 + 2 = 12 + 2 = 14 (chain rule)
    print("x.grad:", x.grad)

    # NOTE: .grad is stored in the variable you're differentiating with respect to.
    # So the gradient is stored in x, because that's the variable you're differentiating with respect to.
