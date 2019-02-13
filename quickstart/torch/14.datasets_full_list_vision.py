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
import torchvision

# The full list of available datasets is available at:
# https://docs.pytorch.org/vision/stable/datasets.html
if __name__ == "__main__":
    print("Full list of available vision datasets:")
    datasets_list = []
    for dataset in torchvision.datasets.__all__:
        try:
            # check if the dataset is a subclass of torch.utils.data.Dataset
            if hasattr(torchvision.datasets, dataset) and issubclass(
                getattr(torchvision.datasets, dataset), torch.utils.data.Dataset
            ):
                datasets_list.append(dataset)
        except Exception as e:
            pass
    print(datasets_list)
