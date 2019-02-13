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
import torchvision.models
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.backends.cudnn as cudnn

from dlplay.models.backbones.resnet import ResNet18
from dlplay.models.backbones.mobilenetv2 import MobileNetV2

from dlplay.core.training import (
    SimpleTrainer,
    OptimizerType,
    LearningRateSchedulerType,
)
from dlplay.utils.device import resolve_device
from dlplay.core.evaluation import TaskType
from dlplay.paths import DATA_DIR, RESULTS_DIR

from dlplay.core.initialize import init_module_params
from dlplay.core.transform_helpers import (
    get_transforms_basic_color,
    get_transforms_affine_color,
)


class BottleneckBlock(torch.nn.Module):
    """
    Bottleneck with dropout.
    """

    def __init__(self, in_dim, dim_factor=4, p=0.01):
        super().__init__()
        self.in_dim = in_dim
        self.dim_factor = dim_factor
        self.out_dim = in_dim * dim_factor
        self.p = p
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, self.out_dim),
            torch.nn.GELU(),
            # torch.nn.ReLU(),
            torch.nn.Dropout(self.p),
            torch.nn.Linear(self.out_dim, in_dim),
        )

    def forward(self, x):
        return self.net(x)


# pre-normalized transform block
class PreNormalizedTransformerBlock(torch.nn.Module):
    """
    Pre-normalized transformer block.
    """

    def __init__(
        self,
        dim_embedding,
        num_heads=4,
        bottleneck_dim_factor=4,
        attn_dropout=0.01,
        bottleneck_dropout=0.01,
    ):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(dim_embedding)
        self.ln2 = torch.nn.LayerNorm(dim_embedding)
        # Use batch_first so inputs are (B, V, E)
        self.mha = torch.nn.MultiheadAttention(
            embed_dim=dim_embedding,
            num_heads=num_heads,
            batch_first=True,
            dropout=attn_dropout,
        )
        self.bottleneck = BottleneckBlock(
            dim_embedding, dim_factor=bottleneck_dim_factor, p=bottleneck_dropout
        )

    def forward(self, x):  # x: (B, V, E)
        x_norm = self.ln1(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)  # self-attention
        x1 = x + attn_out
        x2 = x1 + self.bottleneck(self.ln2(x1))
        return x2  # (B, V, E)


class MLPClassifierHead(torch.nn.Module):
    """
    MLP classifier head.
    """

    def __init__(self, in_dim, hidden_units=256, num_classes=10, p=0.01):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, hidden_units),
            torch.nn.ReLU(),
            torch.nn.Dropout(p),
            torch.nn.Linear(hidden_units, num_classes),
        )

    def forward(self, x):  # x: (B, E)
        return self.net(x)  # logits: (B, num_classes)


class ClassTokenHead(torch.nn.Module):
    def __init__(self, dim_embedding, num_heads=4, mlp_ratio=4, num_classes=10):
        super().__init__()
        self.cls = torch.nn.Parameter(torch.zeros(1, 1, dim_embedding))
        self.block = PreNormalizedTransformerBlock(
            dim_embedding, num_heads=num_heads, bottleneck_dim_factor=mlp_ratio
        )
        self.head = MLPClassifierHead(dim_embedding, num_classes=num_classes)

    def forward(self, z, key_padding_mask=None):  # z: (B, V, E)
        B = z.size(0)
        cls_tok = self.cls.expand(B, -1, -1)  # (B,1,E)
        z = torch.cat([cls_tok, z], dim=1)  # (B,1+V,E)
        # pad mask must pad only the views, not cls
        if key_padding_mask is not None:
            pad = F.pad(key_padding_mask, (1, 0), value=False)
        else:
            pad = None
        z = self.block(z, key_padding_mask=pad)
        cls_out = z[:, 0]  # (B,E)
        return self.head(cls_out)


class MeanPooling(torch.nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim  # dimension to mean over

    def forward(self, x):
        # input shape: (B, V, E) or (V, E)
        # output shape: (B, E) or (E) if dim=1
        return torch.mean(x, dim=self.dim)


class SimpleBackboneConvNet(torch.nn.Module):
    """
    Simple convolutional neural network.
    """

    def __init__(
        self,
        input_channels=1,
        cfg=[32, 64, 128],
        dim_embedding=128,
    ):
        super(SimpleBackboneConvNet, self).__init__()
        self.input_channels = input_channels
        self.cfg = cfg
        self.dim_embedding = dim_embedding
        self.model = self._make_layers(cfg, input_channels, dim_embedding)
        self.model.model_name = "simple_backbone_convnet"

    def _make_layers(self, cfg, input_channels=3, dim_embedding=128):
        """
        Make layers of a convolutional neural network.
        """
        layers = []
        in_channels = input_channels
        for x in cfg:
            layers += [
                torch.nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(x),
                torch.nn.ReLU(inplace=True),  # inplace=True is faster
            ]
            layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            in_channels = x

        # Make channel dimension fixed and ignore spatial dimensions
        layers += [
            torch.nn.AdaptiveAvgPool2d(1)
        ]  # global average pooling -> (B C, 1, 1)
        layers += [torch.nn.Flatten(start_dim=1)]  # flatten the output -> (B,C)
        layers += [torch.nn.Linear(cfg[-1], dim_embedding)]
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


"""
                        +--------------------+
        Input image --> | Data augmentations |
                        +--------------------+
                                |
        ----------------------------------------------
        |                       |                    |
        v                       v                    v
       +-----------+      +-----------+         +-----------+
       |  View 1   |      |  View 2   |         |  View 3   |
       +-----------+      +-----------+         +-----------+
             |                  |                    |
             v                  v                    v
   +-----------------+  +-----------------+  +-----------------+
   | Conv. backbone  |  | Conv. backbone  |  | Conv. backbone  |
   +-----------------+  +-----------------+  +-----------------+
         \                      |                        /
          \                     |                       /
           \--------------------|----------------------/
                                v
                       +------------------+
                       | Transformer block |
                       +------------------+
                                |
                       +------------------+
                       |   Pooling (avg)  |
                       +------------------+
                                |
                       +------------------+
                       | Classification   |
                       |      head        |
                       +------------------+
                                |
                             "Lizard"
"""


class MultiviewClassificationModel(torch.nn.Module):
    """
    Multiview classification model.
    This is a simple didactic example in multiview learning.
    """

    def __init__(
        self,
        dim_embedding=128,
        num_mha_heads=4,
        bottleneck_dim_factor=3,
        num_classes=10,
    ):
        super().__init__()

        # 1) Backbone:
        # ResNet without the final FC -> outputs embeddings
        # backbone = torchvision.models.resnet18(weights=None)
        # backbone_dim_embedding = backbone.fc.in_features
        # print(f"ResNet18: dim_embedding: {backbone_dim_embedding}")
        # backbone.fc = torch.nn.Linear(backbone_dim_embedding, dim_embedding)
        # self.backbone = backbone
        # self.backbone.model_name = "resnet18"

        # self.backbone = SimpleBackboneConvNet(
        #     input_channels=3, cfg=[32, 64, 128, 256], dim_embedding=dim_embedding
        # )

        # resnet = ResNet18()
        # self.backbone = torch.nn.Sequential(
        #     resnet,
        #     torch.nn.Linear(resnet.linear.out_features, dim_embedding),
        # )
        mobilenet = MobileNetV2()
        self.backbone = torch.nn.Sequential(
            mobilenet,
            torch.nn.Linear(mobilenet.linear.out_features, dim_embedding),
        )

        # 2) Classifier head: transformer -> mean over views -> MLP
        self.classifier = torch.nn.Sequential(
            PreNormalizedTransformerBlock(
                dim_embedding,
                num_heads=num_mha_heads,
                bottleneck_dim_factor=bottleneck_dim_factor,
            ),
            MeanPooling(dim=1),
            MLPClassifierHead(dim_embedding, num_classes=num_classes),
        )

    def make_views(self, x):
        # x: (B, C, H, W)
        # returns: (B, 4, C, H, W) with rotations on spatial dims
        v0 = x
        v1 = torch.rot90(x, 1, (2, 3))  # rotate 90 degrees clockwise
        v2 = torch.rot90(x, 2, (2, 3))  # rotate 180 degrees clockwise
        # v3 = torch.rot90(x, 3, (2, 3)).contiguous()  # rotate 270 degrees clockwise
        # out = torch.stack([v0, v1, v2, v3], dim=1)
        out = torch.stack([v0, v1, v2], dim=1)
        self.num_views = out.shape[1]  # V
        return out

    def forward(self, x):
        # x: (B, C, H, W)
        B, _, _, _ = x.shape

        # 4 rotated views
        x_views = self.make_views(x)  # (B, V, C, H, W)
        x_flat = x_views.flatten(0, 1)  # (B*V, C, H, W)
        embeds = self.backbone(x_flat)  # (B*V, E)

        # reshape back to (B, V, E)
        E = embeds.shape[-1]
        embeds = embeds.view(B, self.num_views, E)  # (B, V, E)

        # transformer + mean over views + MLP
        logits = self.classifier(embeds)  # (B, num_classes)
        return logits


def test_simple_backbone_convnet():
    model = SimpleBackboneConvNet(
        input_channels=3, cfg=[32, 64, 128], dim_embedding=128
    )
    print(model)

    x = torch.randn(10, 3, 32, 32)
    y = model(x)
    print(y.shape)


def test_resnet18_backbone():
    dim_embedding = 128
    resnet = ResNet18()
    print(f"ResNet18: dim_embedding: {resnet.linear.out_features}")
    model = torch.nn.Sequential(
        resnet, torch.nn.Linear(resnet.linear.out_features, dim_embedding)
    )
    print(model)

    x = torch.randn(10, 3, 32, 32)
    y = model(x)
    print(y.shape)


def test_multiview_classification_model():
    model = MultiviewClassificationModel(num_classes=10)
    print(model)

    x = torch.randn(
        10, 3, 32, 32
    )  # (B, C, H, W); if real images, scale+normalize accordingly
    y = model(x)
    print(y.shape)  # should be (10, 10)


def train_multiview_classification_model_cifar10():

    resume_training = False
    explicit_init_weights = False

    model = MultiviewClassificationModel(num_classes=10)

    if resume_training:
        resume_checkpoint = (
            f"{RESULTS_DIR}/saved_models/trained_multiview_classification_model.pth"
        )
        model.load_state_dict(torch.load(resume_checkpoint))
        print(f"Loaded torch model state from {resume_checkpoint}")
    else:
        # explicitly init weights
        if explicit_init_weights:
            init_module_params(model)

    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Using {device} device")
    print(f"Model structure: {model}")

    loss_fn = torch.nn.CrossEntropyLoss()

    transform_train = get_transforms_basic_color(train=True)
    transform_test = get_transforms_basic_color(train=False)

    # Load the training and test datasets.
    training_data = datasets.CIFAR10(
        root=f"{DATA_DIR}/datasets",
        train=True,
        download=True,
        transform=transform_train,
    )

    test_data = datasets.CIFAR10(
        root=f"{DATA_DIR}/datasets",
        train=False,
        download=True,
        transform=transform_test,
    )

    batch_size = 64
    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    num_epochs = 200

    trainer = SimpleTrainer(
        model=model,
        dataloader=train_dataloader,
        dataloader_test=test_dataloader,
        optimizer_type=OptimizerType.SGD_MOMENTUM,
        lr_scheduler_type=LearningRateSchedulerType.COSINE,
        loss_fn=loss_fn,
        task_type=TaskType.CLASSIFICATION,
        device=device,
        num_epochs=num_epochs,
        print_freq=100,
        optimizer={"lr": 0.05, "momentum": 0.9, "weight_decay": 5e-4},
        # lr_scheduler={"step_size": 10, "gamma": 0.5},
    )
    trainer.train_model()

    if True:
        filename = (
            f"{RESULTS_DIR}/saved_models/trained_multiview_classification_model.pth"
        )
        torch.save(model.state_dict(), filename)
        print(f"Saved torch model state to {filename}")


if __name__ == "__main__":
    # test_simple_backbone_convnet()
    # test_resnet18_backbone()
    # test_multiview_classification_model()
    train_multiview_classification_model_cifar10()
