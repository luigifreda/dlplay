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

import re
import unicodedata


def dump_optimized_model_layers(net: torch.nn.Module, print_func=print):
    """Check which layers are being optimized."""
    for name, param in net.named_parameters():
        if param.requires_grad:
            print_func(f"Optimizing layer: {name}")


def base_model_name(
    model: torch.nn.Module,
    backbone_model: torch.nn.Module,
    head_model: torch.nn.Module,
    num_classes: int,
) -> str:
    """
    Get the base name of the model.

    Args:
        model: The model to get the base name of.
        backbone_model: The backbone model to get the base name of.
        head_model: The head model to get the base name of.
        num_classes: The number of classes in the model.

    Returns:
        The base name of the model with information about the backbone, head, and RPN.
    """
    base_name = _to_slug(getattr(model, "model_name", model.__class__.__name__.lower()))
    backbone_name = _to_slug(getattr(backbone_model, "model_name", ""))

    # Backbone details
    bb_tag = _infer_backbone_tag(backbone_model)
    bb_bits = [backbone_name, bb_tag]
    if hasattr(backbone_model, "out_channels"):
        bb_bits.append(f"och{backbone_model.out_channels}")
    if hasattr(backbone_model, "num_layers"):
        bb_bits.append(f"layers{backbone_model.num_layers}")
    backbone_block = _fmt_block("backbone", *bb_bits)

    # Head details
    head_name = head_model.__class__.__name__.lower()
    head_block = _fmt_block("head", head_name, f"nclasses{num_classes}")

    # RPN details (optional)
    rpn_block = _rpn_block(model)

    name = base_name + "".join([backbone_block, head_block, rpn_block])
    return _safe_basename(name)


def finetuned_model_name(
    model, backbone_model, head_model, just_finetune_head, num_classes
):
    base = base_model_name(model, backbone_model, head_model, num_classes)
    finetune_block = _fmt_block("finetune", "head" if just_finetune_head else "all")
    return _safe_basename(base + finetune_block)


# ---------- other private and filename-safe helpers ----------

_VALID_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789._-+")
_RESERVED_WIN = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def _len_safe(x):
    """
    Safely get the length of an object.
    """
    try:
        return len(x)
    except Exception:
        return None


def _as_seq(x):
    """
    Convert a tensor/array to a Python list.
    """
    # convert tensors/arrays to Python lists if needed, keep tuples/lists as-is
    try:
        if isinstance(x, torch.Tensor):
            return x.tolist()
    except Exception:
        pass
    return list(x) if isinstance(x, (tuple, list)) else [x]


def _to_slug(s: str) -> str:
    """
    Convert a string to a filename-safe slug.
    """
    s = unicodedata.normalize("NFKC", s).lower().replace(" ", "_")
    s = "".join(ch if ch in _VALID_CHARS else "-" for ch in s)
    s = re.sub(r"[-_+.]{2,}", lambda m: m.group(0)[0], s)
    return s.strip("._-+") or "unnamed"


def _safe_basename(s: str, max_len: int = 200) -> str:
    """
    Make a filename-safe basename.
    """
    if len(s) > max_len:
        s = s[:max_len].rstrip("._-+")
    if s in _RESERVED_WIN:
        s = f"_{s}"
    return s


def _fmt_block(label: str, *parts: str) -> str:
    """
    Format a block of text with a label and parts.
    """
    parts = [_to_slug(p) for p in parts if p]
    if not parts:
        return ""
    return f"__{_to_slug(label)}[" + "_".join(parts) + "]"


def _infer_backbone_tag(backbone_model):
    """
    Return a compact tag like 'resnet50_fpn' even when the body is wrapped
    by torchvision's IntermediateLayerGetter.

    Strategy:
    - Take backbone_model.body if present, else backbone_model.
    - If the object has layer1/layer3 (like ResNet or IntermediateLayerGetter of a ResNet),
      infer depth by block type + number of blocks in layer3.
    - Otherwise fall back to class name.
    - Append 'fpn' if FPN is present.
    """
    # 1) pick the thing that actually has layers
    body = getattr(backbone_model, "body", backbone_model)

    # 2) detect a ResNet by structure, not by class name
    layer1 = getattr(body, "layer1", None)
    layer3 = getattr(body, "layer3", None)

    parts = []

    if layer1 is not None and layer3 is not None:
        # try to infer BasicBlock vs Bottleneck from the first block of layer1
        try:
            first_block = list(layer1)[0]
            block_name = first_block.__class__.__name__.lower()
            is_bottleneck = "bottleneck" in block_name
            n_l3 = len(list(layer3))

            if is_bottleneck:
                depth_map = {6: 50, 23: 101, 36: 152}
            else:
                depth_map = {2: 18, 6: 34}

            depth = depth_map.get(n_l3)
            if depth is not None:
                parts.append(f"d{depth}")
        except Exception:
            # structure looks like resnet but we couldn't read it cleanly
            pass
    else:
        # 3) not a resnet-ish container; fall back to the class name
        if not isinstance(
            body, torchvision.models.detection.backbone_utils.IntermediateLayerGetter
        ):
            parts.append(body.__class__.__name__.lower())

    # 4) FPN?
    has_fpn = (
        hasattr(backbone_model, "fpn")
        or "fpn" in backbone_model.__class__.__name__.lower()
    )
    if has_fpn:
        parts.append("fpn")

    return "_".join(parts)


def _find_anchor_generator(model):
    """
    Find the anchor generator in the model.
    """
    # Prefer the canonical location
    rpn = getattr(model, "rpn", None)
    if rpn is not None and getattr(rpn, "anchor_generator", None) is not None:
        return rpn.anchor_generator
    # Fallback (older/custom models)
    ag = getattr(model, "rpn_anchor_generator", None)
    if ag is not None:
        return ag
    return None


def _rpn_block(model):
    """
    Summarize the RPN anchor generator in the model.
    """
    ag = _find_anchor_generator(model)
    if ag is None:
        return ""  # no RPN / anchors found

    sizes = getattr(ag, "sizes", None)
    ratios = getattr(ag, "aspect_ratios", None)

    # Nothing to summarize
    if sizes is None or ratios is None:
        return ""

    # Normalize to list-of-lists shape (one entry per pyramid level)
    sizes = [tuple(_as_seq(s)) for s in _as_seq(sizes)]
    ratios = [tuple(_as_seq(r)) for r in _as_seq(ratios)]

    # Common FPN case: sizes like ((32,), (64,), (128,), (256,), (512,))
    # and ratios like ((0.5,1.0,2.0),) * 5
    n_levels = max(_len_safe(sizes), _len_safe(ratios)) or 1

    # If single-level (non-FPN) shape like ((32,64,128,256,512),) and ((0.5,1.0,2.0),)
    if n_levels == 1:
        ns = _len_safe(sizes[0]) or 0
        nr = _len_safe(ratios[0]) or 0
        tag = f"anchors{ns}x{nr}" if ns and nr else None
        return _fmt_block("rpn", tag) if tag else ""

    # Multi-level (FPN). We summarize compactly.
    # Many FPN configs have a constant number of ratios per level.
    # For sizes per level, we often have 1 (one scale per level).
    # We’ll emit levels + per-level counts.
    sizes_per_level = [(_len_safe(s) or 0) for s in sizes]
    ratios_per_level = [(_len_safe(r) or 0) for r in ratios]

    # If all levels share the same counts, show the uniform counts
    if len(set(sizes_per_level)) == 1 and len(set(ratios_per_level)) == 1:
        return _fmt_block(
            "rpn",
            f"l{n_levels}",
            f"s{sizes_per_level[0]}",
            f"r{ratios_per_level[0]}",
        )

    # Otherwise, provide a concise per-level summary like s[1,1,1,1,1]+r[3,3,3,3,3]
    sizes_str = ",".join(str(x) for x in sizes_per_level)
    ratios_str = ",".join(str(x) for x in ratios_per_level)
    return _fmt_block("rpn", f"levels{n_levels}", f"s[{sizes_str}]", f"r[{ratios_str}]")
