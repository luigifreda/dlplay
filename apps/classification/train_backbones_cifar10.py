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
"""Train CIFAR10 with PyTorch.
From https://github.com/kuangliu/pytorch-cifar/blob/master/main.py

## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
| [RegNetX_200MF](https://arxiv.org/abs/2003.13678)     | 94.24%      |
| [RegNetY_400MF](https://arxiv.org/abs/2003.13678)     | 94.29%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [SimpleDLA](https://arxiv.org/abs/1707.064)           | 94.89%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |
| [DLA](https://arxiv.org/pdf/1707.06484.pdf)           | 95.47%      |



"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from dlplay.models.backbones import *
from dlplay.io.progress_bar import ProgressBar
from dlplay.paths import DATA_DIR, RESULTS_DIR

from dlplay.core.training import (
    SimpleTrainer,
    OptimizerType,
    LearningRateSchedulerType,
)
from dlplay.core.evaluation import TaskType


# Training
def train(trainloader, epoch, best_acc):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    progress_bar = ProgressBar()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar.step(
            batch_idx,
            len(trainloader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (
                train_loss / (batch_idx + 1),
                100.0 * correct / total,
                correct,
                total,
            ),
        )

    # Save checkpoint.
    acc = 100.0 * correct / total
    if acc > best_acc:
        print("Saving..")
        state = {
            "net": net.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        target_folder = f"{RESULTS_DIR}/cifar10/checkpoint"
        if not os.path.isdir(target_folder):
            os.mkdir(target_folder)
        torch.save(state, f"{target_folder}/ckpt.pth")
        best_acc = acc

    return best_acc


def test(testloader, epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    progress_bar = ProgressBar()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar.step(
                batch_idx,
                len(testloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )


# from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument(
        "--resume", "-r", action="store_true", help="resume from checkpoint"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print("==> Preparing data..")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=f"{DATA_DIR}/datasets",
        train=True,
        download=True,
        transform=transform_train,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root=f"{DATA_DIR}/datasets",
        train=False,
        download=True,
        transform=transform_test,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    # Model
    print("==> Building model..")
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    net = SimpleDLA()
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    results_dir = f"{RESULTS_DIR}/saved_models/train_backbones_cifar"

    if args.resume:
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        assert os.path.isdir(
            f"{results_dir}/checkpoint"
        ), "Error: no checkpoint directory found!"
        checkpoint = torch.load(f"{results_dir}/checkpoint/ckpt.pth")
        net.load_state_dict(checkpoint["net"])
        best_acc = checkpoint["acc"]
        start_epoch = checkpoint["epoch"]

    num_epochs = 200
    criterion = nn.CrossEntropyLoss()

    # two ways to train the model:
    # 1) using the classical way
    # 2) using the SimpleTrainer class
    use_simple_trainer = False
    if not use_simple_trainer:
        optimizer = optim.SGD(
            net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )

        for epoch in range(start_epoch, start_epoch + num_epochs):
            best_acc = train(trainloader, epoch, best_acc)
            test(testloader, epoch)
            scheduler.step()

    else:
        trainer = SimpleTrainer(
            model=net,
            dataloader=trainloader,
            dataloader_test=testloader,
            optimizer_type=OptimizerType.SGD_MOMENTUM,
            lr_scheduler_type=LearningRateSchedulerType.COSINE,
            loss_fn=criterion,
            task_type=TaskType.CLASSIFICATION,
            device=device,
            num_epochs=num_epochs,
            print_freq=100,
            evaluate_freq=1,
            optimizer={"lr": args.lr, "momentum": 0.9, "weight_decay": 5e-4},
            # lr_scheduler={"step_size": 10, "gamma": 0.5},
        )
        trainer.train_model()
