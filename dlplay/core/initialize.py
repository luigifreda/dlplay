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


def init_module_params_simple(net: torch.nn.Module):
    """
    Simple initialization of the parameters of a module.
    """
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)


def init_module_params(
    net: torch.nn.Module,
    *,  # force keyword-only arguments (no positional arguments) after net
    nonlinearity: str = "relu",  # relu, leaky_relu, tanh, ...
    mode: str = "fan_out",  # 'fan_in' or 'fan_out' for Kaiming
    only_trainable: bool = False,  # if True, skip params with requires_grad=False
    lstm_forget_bias: float = 1.0,  # common trick for LSTM
):
    """
    Initialize the parameters of a module.
    """
    gain = torch.nn.init.calculate_gain(nonlinearity)

    def should_init(m: torch.nn.Module) -> bool:
        if not only_trainable:
            return True
        # Initialize this module only if it has at least one trainable param of its own
        for p in m.parameters(recurse=False):
            if p.requires_grad:
                return True
        return False

    def init_fn(m: torch.nn.Module):
        if not should_init(m):
            return

        # Convs & Linear
        if isinstance(
            m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)
        ):
            if m.weight is not None:
                if nonlinearity in ("relu", "leaky_relu"):
                    torch.nn.init.kaiming_normal_(
                        m.weight, mode=mode, nonlinearity=nonlinearity
                    )
                elif nonlinearity in ("tanh", "sigmoid"):
                    torch.nn.init.xavier_uniform_(m.weight, gain=gain)
                else:
                    torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
            if getattr(m, "bias", None) is not None:
                torch.nn.init.zeros_(m.bias)

        # Normalization layers
        elif isinstance(
            m,
            (
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d,
                torch.nn.InstanceNorm1d,
                torch.nn.InstanceNorm2d,
                torch.nn.InstanceNorm3d,
                torch.nn.GroupNorm,
                torch.nn.LayerNorm,
            ),
        ):
            if getattr(m, "weight", None) is not None:
                torch.nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                torch.nn.init.zeros_(m.bias)

        # Embeddings
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

        # Recurrent layers
        elif isinstance(m, torch.nn.LSTM):
            for name, param in m.named_parameters(recurse=False):
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    torch.nn.init.orthogonal_(param)
                elif "bias_ih" in name:
                    torch.nn.init.zeros_(param)
                    # set forget gate bias
                    hidden = param.numel() // 4
                    param.data[hidden : 2 * hidden].fill_(lstm_forget_bias)
                elif "bias_hh" in name:
                    torch.nn.init.zeros_(param)
        elif isinstance(m, torch.nn.GRU):
            for name, param in m.named_parameters(recurse=False):
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    torch.nn.init.orthogonal_(param)
                elif "bias" in name:
                    torch.nn.init.zeros_(param)

        # MultiheadAttention (works for fused or qkv-separated variants)
        elif isinstance(m, torch.nn.MultiheadAttention):
            if hasattr(m, "in_proj_weight") and m.in_proj_weight is not None:
                torch.nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    torch.nn.init.zeros_(m.in_proj_bias)
            else:  # q_proj_weight / k_proj_weight / v_proj_weight style
                for attr in ("q_proj_weight", "k_proj_weight", "v_proj_weight"):
                    if hasattr(m, attr) and getattr(m, attr) is not None:
                        torch.nn.init.xavier_uniform_(getattr(m, attr))
                if hasattr(m, "in_proj_bias") and m.in_proj_bias is not None:
                    torch.nn.init.zeros_(m.in_proj_bias)
            if hasattr(m, "out_proj") and m.out_proj is not None:
                if m.out_proj.weight is not None:
                    torch.nn.init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    torch.nn.init.zeros_(m.out_proj.bias)

        # Fallback: any module with a >=2D weight gets Xavier; bias zeros
        else:
            w = getattr(m, "weight", None)
            if w is not None and isinstance(w, torch.Tensor) and w.dim() >= 2:
                torch.nn.init.xavier_uniform_(w, gain=gain)
            b = getattr(m, "bias", None)
            if b is not None and isinstance(b, torch.Tensor):
                torch.nn.init.zeros_(b)

    # Apply initialization function recursively to all modules including self
    net.apply(init_fn)
