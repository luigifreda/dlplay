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
    # Basic operations
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])

    #  Add, Mul, Dot, MatMul
    print("Add:", a + b)
    print("Mul:", a * b)
    print("Dot:", torch.dot(a, b))
    print("MatMul:", torch.matmul(a.unsqueeze(0), b.unsqueeze(1)))

    # Element-wise operations
    print("Element-wise max:", torch.maximum(a, b))
    print("Element-wise min:", torch.minimum(a, b))
    print("Element-wise abs:", torch.abs(a))
    print("Element-wise sign:", torch.sign(a))
    print("Element-wise sqrt:", torch.sqrt(a))
    print("Element-wise exp:", torch.exp(a))
    print("Element-wise log:", torch.log(a))
    print("Element-wise sin:", torch.sin(a))
    print("Element-wise cos:", torch.cos(a))

    #  Reduction operations
    print("Sum:", torch.sum(a))
    print("Mean:", torch.mean(a))
    print("Std:", torch.std(a))
    print("Max:", torch.max(a))
    print("Min:", torch.min(a))
    print("Argmax:", torch.argmax(a))
    print("Argmin:", torch.argmin(a))
