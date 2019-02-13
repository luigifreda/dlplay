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
import torch.nn.functional as F
from enum import Enum
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, Tuple

from dlplay.utils.device import move_to_device


class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    GENERATION = "generation"
    OTHER = "other"


class MetricType(Enum):
    ACCURACY = "accuracy"
    MSE = "mse"
    MAP = "map"
    OTHER = "other"


class EvaluatorBase(ABC):
    def __init__(
        self,
        model,
        dataloader_test,
        device,
        loss_fn: Callable,
        task_type: TaskType,
        **kwargs,
    ):
        self.model = model
        self.dataloader_test = dataloader_test
        self.device = device
        self.loss_fn = loss_fn
        self.task_type = task_type
        self.kwargs = kwargs

        # Set up the evaluation parameters
        self.seg_ignore_index = self.kwargs.get("ignore_index", None)
        self.seg_num_classes = self.kwargs.get("num_classes", None)
        self.seg_outputs = self.kwargs.get("segmentation_outputs", "logits")
        self.seg_eps = self.kwargs.get("eps", 1e-7)

        # Initialize metrics based on task type
        self.metrics: Dict[
            str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = self._get_default_metrics(self.task_type)

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        pass

    def _get_default_metrics(self, task_type: TaskType):
        """Get default metrics for a given task type."""
        if task_type == TaskType.CLASSIFICATION:
            return {
                "loss": lambda pred, y: self.loss_fn(pred, y),
                "accuracy": lambda pred, y: (
                    (pred.argmax(dim=1) == _cls_targets(y)).float().mean()
                ),
            }
        elif task_type == TaskType.REGRESSION:
            return {
                "loss": lambda pred, y: self.loss_fn(pred, y),
                "mse": lambda pred, y: F.mse_loss(pred, y),
            }
        elif task_type == TaskType.DETECTION:
            return {
                "loss": lambda pred, y: self.loss_fn(pred, y),
                # TODO: add detection-specific metrics (e.g., mAP)
            }
        elif task_type == TaskType.SEGMENTATION:
            # TODO: add segmentation-specific metrics (e.g., pixel accuracy, mIoU, Dice)
            return {
                "loss": lambda pred, y: self.loss_fn(pred, y),
                "pixel_acc": lambda pred, y: _pixel_accuracy(
                    _seg_to_labels(pred),
                    _target_to_labels(y),
                    ignore_index=self.seg_ignore_index,
                ),
                "miou": lambda pred, y: _mean_iou(
                    _seg_to_labels(pred),
                    _target_to_labels(y),
                    num_classes=self.seg_num_classes,
                    ignore_index=self.seg_ignore_index,
                ),
                "dice": lambda pred, y: _mean_dice(
                    _seg_to_labels(pred),
                    _target_to_labels(y),
                    num_classes=self.seg_num_classes,
                    ignore_index=self.seg_ignore_index,
                ),
            }
        else:
            return {"loss": lambda pred, y: self.loss_fn(pred, y)}


class Evaluator(EvaluatorBase):
    def __init__(
        self,
        model,
        dataloader_test,
        device,
        loss_fn: Callable,
        task_type: TaskType,
        **kwargs,
    ):
        super().__init__(model, dataloader_test, device, loss_fn, task_type, **kwargs)

    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation over the test dataloader and compute the configured metrics.
        Returns a dict mapping metric name -> average value over batches.
        """
        size = len(getattr(self.dataloader_test, "dataset", []))
        num_batches = len(self.dataloader_test)

        # Check if dataloader is empty
        if num_batches == 0:
            print("Warning: Test dataloader is empty!")
            return {name: 0.0 for name in self.metrics.keys()}

        self.model.to(self.device)
        self.model.eval()

        metric_values = {name: 0.0 for name in self.metrics.keys()}

        with torch.no_grad():
            # Iterate over the test dataloader
            for batch in self.dataloader_test:
                X, y = _unpack_batch(batch)
                X, y = move_to_device(X, self.device), move_to_device(y, self.device)
                pred = self.model(X)

                # Compute all metrics
                for metric_name, metric_fn in self.metrics.items():
                    # Compute the metric value for the current batch
                    val = metric_fn(pred, y)
                    # Ensure tensor scalar then accumulate
                    if isinstance(val, torch.Tensor):
                        # Convert the tensor to a scalar
                        val = val.item()
                    metric_values[metric_name] += float(val)

        # Average metrics over batches (num_batches > 0 is guaranteed here)
        for metric_name in metric_values:
            metric_values[metric_name] /= num_batches

        # Print results
        _print_evaluation_results(metric_values, size)

        return metric_values


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------


def _cls_targets(y: torch.Tensor) -> torch.Tensor:
    """
    Convert one-hot encoded targets to indices.
    """
    # one-hot -> indices
    if y.ndim == 2 and y.shape[1] > 1:
        return y.argmax(dim=1)
    # squeeze (N,1) -> (N,)
    if y.ndim >= 2 and y.shape[1] == 1:
        return y.squeeze(1).long()
    return y.long()


def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Support (X, y) tuples/lists or dicts with 'input'/'target' keys."""
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        return batch[0], batch[1]
    elif isinstance(batch, dict):
        # Common key patterns
        for xk in ("input", "inputs", "image", "images", "x"):
            for yk in ("target", "targets", "label", "labels", "y"):
                if xk in batch and yk in batch:
                    return batch[xk], batch[yk]
    raise ValueError(
        "Unsupported batch format. Expected (X, y) or dict with input/target keys."
    )


def _print_evaluation_results(
    metric_values: Dict[str, float], dataset_size: int
) -> None:
    """Print evaluation results in a formatted way."""
    print(f"Test Results (Dataset size: {dataset_size}):")
    for metric_name, value in metric_values.items():
        if metric_name == "accuracy":
            print(f"  {metric_name.capitalize()}: {100.0 * value:0.1f}%")
        else:
            print(f"  {metric_name.capitalize()}: {value:0.6f}")
    print()


def _seg_to_labels(pred: torch.Tensor) -> torch.Tensor:
    """
    Convert model outputs to label maps (N,H,W).
    - Multiclass: argmax over channel dim.
    - Binary (C=1): threshold at 0 (logits) / 0.5 (probs). We use 0 assuming logits.
    """
    # in general,pred is (N,C,H,W)
    if pred.ndim == 4 and pred.shape[1] > 1:  # multiclass
        return pred.argmax(dim=1)
    if pred.ndim == 4 and pred.shape[1] == 1:  # binary
        return (pred[:, 0] > 0).long()
    # If already label maps (N,H,W)
    return pred.long()


def _target_to_labels(y: torch.Tensor) -> torch.Tensor:
    """Convert targets to label maps (N,H,W) from indices or one-hot."""
    # in general,y is (N,C,H,W)
    if y.ndim == 4 and y.shape[1] > 1:  # one-hot multiclass
        return y.argmax(dim=1)
    if y.ndim == 4 and y.shape[1] == 1:  # binary mask channel
        return y[:, 0].long()
    return y.long()  # (N,H,W) indices already


def _pixel_accuracy(
    pred_lab: torch.Tensor,
    tgt_lab: torch.Tensor,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """Compute the pixel accuracy."""
    # Create a boolean mask of valid pixels
    valid = torch.ones_like(tgt_lab, dtype=torch.bool)
    if ignore_index is not None:
        valid &= tgt_lab != ignore_index
    # Compute the number of correct pixels
    correct = (pred_lab == tgt_lab) & valid
    denom = valid.sum().clamp_min(1)
    return correct.sum().float() / denom.float()


def _mean_iou(
    pred_lab: torch.Tensor,
    tgt_lab: torch.Tensor,
    num_classes: int | None = None,
    ignore_index: int | None = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute the mean IoU."""
    # Create a boolean mask of valid pixels
    if ignore_index is not None:
        valid = tgt_lab != ignore_index
        pred_lab = pred_lab[valid]
        tgt_lab = tgt_lab[valid]
        # Check if we have any valid pixels after filtering
        if pred_lab.numel() == 0:
            return torch.tensor(0.0, device=pred_lab.device)

    if num_classes is None:
        num_classes = (
            int(torch.max(torch.stack([pred_lab.max(), tgt_lab.max()]))).item() + 1
        )
    ious = []
    for c in range(num_classes):
        pred_c = pred_lab == c
        tgt_c = tgt_lab == c
        inter = (pred_c & tgt_c).sum().float()
        union = (pred_c | tgt_c).sum().float()
        if union > 0:
            ious.append((inter + eps) / (union + eps))
    if len(ious) == 0:
        return torch.tensor(0.0, device=pred_lab.device)
    return torch.stack(ious).mean()


def _mean_dice(
    pred_lab: torch.Tensor,
    tgt_lab: torch.Tensor,
    num_classes: int | None = None,
    ignore_index: int | None = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute the mean Dice."""
    # Create a boolean mask of valid pixels
    if ignore_index is not None:
        valid = tgt_lab != ignore_index
        pred_lab = pred_lab[valid]
        tgt_lab = tgt_lab[valid]
        # Check if we have any valid pixels after filtering
        if pred_lab.numel() == 0:
            return torch.tensor(0.0, device=pred_lab.device)

    if num_classes is None:
        num_classes = (
            int(torch.max(torch.stack([pred_lab.max(), tgt_lab.max()]))).item() + 1
        )
    # Compute the Dice for each class
    dices = []
    for c in range(num_classes):
        p = pred_lab == c
        t = tgt_lab == c
        inter = (p & t).sum().float()
        denom = p.sum().float() + t.sum().float()
        if denom > 0:
            dices.append((2 * inter + eps) / (denom + eps))
    if len(dices) == 0:
        return torch.tensor(0.0, device=pred_lab.device)
    return torch.stack(dices).mean()
