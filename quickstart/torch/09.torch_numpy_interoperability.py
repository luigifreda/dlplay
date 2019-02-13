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
import numpy as np


if __name__ == "__main__":
    #  PyTorch and NumPy Interoperability

    #  Convert NumPy array to PyTorch tensor
    np_arr = np.array([1, 2, 3])
    tensor = torch.from_numpy(np_arr)
    print("Numpy → Tensor:", tensor)

    #  Convert PyTorch tensor to NumPy array
    back_to_np = tensor.numpy()
    print("Tensor → Numpy:", back_to_np)
