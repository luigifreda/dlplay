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
import torch.nn as nn
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import FasterRCNN, MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


from dlplay.models.fine_tuning_helpers import set_trainable_only_head


# NOTE: see the following examples for how to use the functions included in this file
#      apps/finetuning/rcnn_finetuning_replace_head.py
#      apps/finetuning/rcnn_finetuning_replace_backbone.py
#      apps/finetuning/rcnn_eval_finetuned_model.py


# -------------------------------------------------------------------------
# Faster R-CNN (ResNet50-FPN backbone)
# Task: Object Detection
# -------------------------------------------------------------------------
def build_fasterrcnn_resnet50(num_classes: int, finetune_head_only: bool = False):
    """
    Build Faster R-CNN with a ResNet50-FPN backbone (pretrained on COCO).
    Replace the box predictor head with a new one for `num_classes`.
    """
    # Load pretrained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.model_name = "fasterrcnn_resnet50_fpn"

    # Replace the classification head with a new one adapted to `num_classes`
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    backbone_model = model.backbone
    backbone_model.model_name = "resnet50"
    head_model = model.roi_heads.box_predictor

    # Optionally freeze everything except the head
    if finetune_head_only:
        set_trainable_only_head(model, model.roi_heads.box_predictor)

    return model, backbone_model, head_model


# -------------------------------------------------------------------------
# Faster R-CNN (MobileNetV2 backbone)
# Task: Object Detection
# -------------------------------------------------------------------------
def build_fasterrcnn_mobilenetv2(num_classes: int, finetune_head_only: bool = False):
    """
    Build Faster R-CNN with a MobileNetV2 backbone (pretrained on ImageNet).
    This replaces the default ResNet backbone with a lighter MobileNet.
    """
    # Load MobileNetV2 pretrained backbone (features only)
    backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
    # ``FasterRCNN`` needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280  # output feature depth of MobileNetV2

    # Define anchor generator for Region Proposal Network (RPN)
    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),  # anchor box sizes
        aspect_ratios=((0.5, 1.0, 2.0),),  # anchor aspect ratios
    )

    # Define RoI (Region of Interest) pooling operation
    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # ``OrderedDict[Tensor]``, and in ``featmap_names`` you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"], output_size=7, sampling_ratio=2
    )

    # Put backbone + RPN + heads together into a Faster R-CNN
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )
    model.model_name = "fasterrcnn_mobilenetv2"

    backbone_model = model.backbone
    backbone_model.model_name = "mobilenetv2"
    head_model = model.roi_heads.box_predictor

    # Optionally freeze everything except the detection head
    if finetune_head_only:
        set_trainable_only_head(model, model.roi_heads.box_predictor)

    return model, backbone_model, head_model


# -------------------------------------------------------------------------
# Mask R-CNN (ResNet50-FPN backbone)
# Task: Instance Segmentation
# -------------------------------------------------------------------------
def build_maskrcnn_resnet50_fpn(num_classes: int, finetune_head_only: bool = False):
    """
    Build Mask R-CNN with a ResNet50-FPN backbone (pretrained on COCO).
    Replace both the box predictor and the mask predictor with new heads.
    """
    # Load pretrained model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    model.model_name = "maskrcnn_resnet50_fpn"

    # Replace the box classifier head
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # Replace the mask predictor head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels=in_features_mask,
        dim_reduced=256,  # hidden layer size in mask head
        num_classes=num_classes,
    )

    backbone_model = model.backbone
    backbone_model.model_name = "resnet50"
    head_model = model.roi_heads.mask_predictor

    # Optionally freeze everything except the box + mask heads
    if finetune_head_only:
        set_trainable_only_head(
            model, [model.roi_heads.box_predictor, model.roi_heads.mask_predictor]
        )

    return model, backbone_model, head_model


# -------------------------------------------------------------------------
# Mask R-CNN (MobileNetV2 backbone)
# Task: Instance Segmentation
# -------------------------------------------------------------------------
def build_maskrcnn_mobilenetv2(
    num_classes: int,
    finetune_head_only: bool = False,
):
    """
    Mask R-CNN with a MobileNetV2 backbone (ImageNet-pretrained, no FPN).
    Without FPN, the model is not able to learn the objectness of the anchors.
    This results in a model that is not able to detect objects. We are just
    using it for demonstration purposes.
    Returns (model, backbone_module, mask_head_module).
    """
    # 1) Backbone
    backbone: nn.Module = torchvision.models.mobilenet_v2(weights="DEFAULT").features
    backbone.out_channels = 1280  # MobileNetV2 final feature depth

    # 2) RPN anchors
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    )

    # 3) ROI poolers for boxes and masks (single feature map named "0")
    box_roi_pool = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"], output_size=7, sampling_ratio=2
    )
    # With a single-map backbone (“0”), it’s safer to explicitly pass a mask_roi_pool
    # (output size 14 is standard for masks). Some versions won’t default it correctly
    # for non-FPN backbones.
    mask_roi_pool = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"], output_size=14, sampling_ratio=2
    )

    # 4) Build model; predictors will be created automatically for num_classes
    model = MaskRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pool,
        mask_roi_pool=mask_roi_pool,
    )
    model.model_name = "maskrcnn_mobilenetv2"

    if finetune_head_only:
        # freeze everything
        for p in model.parameters():
            p.requires_grad = False
        # unfreeze heads (box + mask). Optionally also unfreeze RPN if desired.
        for p in model.roi_heads.box_predictor.parameters():
            p.requires_grad = True
        for p in model.roi_heads.mask_predictor.parameters():
            p.requires_grad = True
        # If you want RPN trainable too, uncomment:
        # If finetune_head_only=True, you probably also want the RPN to train (it learns anchors objectness).
        print(f"unfreezing RPN, to allow it to learn anchors objectness")
        for p in model.rpn.parameters():
            p.requires_grad = True

    backbone_module = model.backbone
    backbone_module.model_name = "mobilenetv2"
    mask_head_module = model.roi_heads.mask_predictor
    return model, backbone_module, mask_head_module


# -------------------------------------------------------------------------
# Mask R-CNN (MobileNetV2 backbone + FPN)
# Task: Instance Segmentation
# NOTE: Here, we can't finetune the head only, because the backbone includes an additional untrained FPN layer
# -------------------------------------------------------------------------
def build_maskrcnn_mobilenetv2_fpn(
    num_classes: int,
    fpn_out_channels: int = 256,
    finetune_head_only: bool = False,  # fake argument, just for compatibility with other build_model functions
):
    """
    For torchvision's MobileNetV2:
        features[2] → 1/4, 24 ch
        features[4] → 1/8, 32 ch
        features[7] → 1/16, 64 ch
        features[18] → 1/32, 1280 ch

    Loads ImageNet-pretrained weights for MobileNetV2.
    Everything we add on top is newly initialized:
    - FPN layers → random init
    - RPN → random init
    - Box & mask heads → random init (for your num_classes)

    Behavior with finetune_head_only:
    - finetune_head_only=True: backbone + FPN stay frozen; only RPN and heads train.
    - finetune_head_only=False: the whole model (including the pretrained backbone) trains.
    """

    # 1) Base MobileNetV2
    try:
        # newer torchvision
        from torchvision.models import MobileNet_V2_Weights

        mv2 = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    except Exception:
        mv2 = torchvision.models.mobilenet_v2(weights="DEFAULT")
    body = mv2.features

    # 2) Choose pyramid taps at strides 4,8,16,32
    return_layers = {"2": "0", "4": "1", "7": "2", "18": "3"}  # C2..C5

    # 3) Probe the actual channel sizes to avoid mismatches
    body_returner = IntermediateLayerGetter(body, return_layers=return_layers)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 256, 256)
        feats = body_returner(dummy)
    in_channels_list = [feats[k].shape[1] for k in ["0", "1", "2", "3"]]

    # 4) Wrap with FPN; add a coarse 'pool' level for the RPN
    backbone = BackboneWithFPN(
        body_returner,
        return_layers,
        in_channels_list,
        fpn_out_channels,
        extra_blocks=LastLevelMaxPool(),  # adds 'pool' (P6) for RPN
    )
    backbone.out_channels = fpn_out_channels

    # 5) RPN anchors: one tuple per returned FPN map (4 + pool = 5)
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

    # 6) ROIAlign over the 4 main pyramid maps (exclude 'pool')
    featmap_names = ["0", "1", "2", "3"]
    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=featmap_names, output_size=7, sampling_ratio=2
    )
    mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=featmap_names, output_size=14, sampling_ratio=2
    )

    # 7) Build Mask R-CNN
    model = MaskRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pool,
        mask_roi_pool=mask_roi_pool,
    )
    model.model_name = "maskrcnn_mobilenetv2_fpn"

    # if finetune_head_only:
    #     for p in model.parameters():
    #         p.requires_grad = False
    #     for m in [
    #         model.roi_heads.box_predictor,
    #         model.roi_heads.mask_predictor,
    #         model.rpn,
    #     ]:
    #         for p in m.parameters():
    #             p.requires_grad = True

    backbone_module = model.backbone
    backbone_module.model_name = "mobilenetv2"
    mask_head_module = model.roi_heads.mask_predictor
    return model, backbone_module, mask_head_module


# -------------------------------------------------------------------------
# Simple factory function for testing
# -------------------------------------------------------------------------
def build_model(kind: str, num_classes: int = 2, finetune_head_only: bool = False):
    """
    Factory to build a detection/segmentation model.
    - kind: one of {"fasterrcnn", "fasterrcnn_mbv2", "maskrcnn"}
    - num_classes: number of output classes (including background)
    - finetune_head_only: if True, only train the head layers
    """
    kind = kind.lower()
    if kind in {"fasterrcnn_resnet50", "fasterrcnn"}:
        return build_fasterrcnn_resnet50(num_classes, finetune_head_only)
    if kind in {"fasterrcnn_mobilenetv2", "fasterrcnn_mbv2", "mbv2"}:
        return build_fasterrcnn_mobilenetv2(num_classes, finetune_head_only)
    if kind in {"maskrcnn_resnet50", "maskrcnn"}:
        return build_maskrcnn_resnet50_fpn(num_classes, finetune_head_only)
    if kind in {"maskrcnn_mobilenetv2", "maskrcnn_mbv2", "mbv2"}:
        return build_maskrcnn_mobilenetv2(num_classes, finetune_head_only)
    if kind in {"maskrcnn_mobilenetv2_fpn", "maskrcnn_mbv2_fpn", "mbv2_fpn"}:
        return build_maskrcnn_mobilenetv2_fpn(num_classes, finetune_head_only)
    raise ValueError(f"Unknown model kind: {kind}")


# -------------------------------------------------------------------------
# Demo usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    for kind in [
        "fasterrcnn",
        "fasterrcnn_mbv2",
        "maskrcnn",
        "maskrcnn_mbv2",
        "maskrcnn_mbv2_fpn",
    ]:
        print("-" * 80)
        print(f"Building model: {kind}")
        model, backbone_model, head_model = build_model(
            kind, num_classes=2, finetune_head_only=True
        )
        print(model.__class__.__name__, "created with finetune_head_only finetuning")
        print(backbone_model.__class__.__name__)
        print(head_model.__class__.__name__)

        # quick sanity check
        model.eval()
        with torch.no_grad():
            x = [torch.randn(3, 480, 640)]
            out = model(x)  # list of dicts with 'boxes','labels','scores','masks'
            print(
                {
                    k: v.shape if hasattr(v, "shape") else type(v)
                    for k, v in out[0].items()
                }
            )
