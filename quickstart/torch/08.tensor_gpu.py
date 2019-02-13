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
    # Using GPUs

    #  Check if GPU is available
    print("GPU available:", torch.cuda.is_available())

    #  Get available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    #  Create a tensor on the device
    x = torch.rand((2, 2)).to(device)
    print("x on", x.device)

    #  Move tensors to GPU if available
    if torch.cuda.is_available():
        print("moving to GPU")
        x = x.to("cuda")
        print("x on", x.device)

    #  Move tensors to CPU
    print("moving to CPU")
    x = x.to("cpu")
    print("x on", x.device)
