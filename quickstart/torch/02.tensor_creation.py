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

    # From lists
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    # With shape and initialization
    zeros = torch.zeros((2, 3))
    ones = torch.ones((3, 3))
    rand = torch.rand((2, 2))
    eye = torch.eye(3)

    # Like another tensor
    like_a = torch.ones_like(a)

    print("a:", a)
    print("b:", b)
    print("zeros:", zeros)
