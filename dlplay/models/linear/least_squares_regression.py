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
import numpy as np
from typing import Dict, List
import torch
import jax
import jax.numpy as jnp

from dlplay.optimization.optimization import ObjectiveFunction
from dlplay.utils.types import ArrayLike

from dlplay.core.tensor_backend import TensorBackend


def MSE(y, y_pre):
    """
    Mean Squared Error
    """
    return np.mean((y - y_pre) ** 2)


def RMSE(y, y_pre):
    """
    Root Mean Squared Error
    """
    return np.sqrt(MSE(y, y_pre))


def MAPE(y, y_pre):
    """
    Mean Absolute Percentage Error
    """
    return np.mean(np.abs((y - y_pre) / y))


def R2(y, y_pre):
    """
    R2 Score
    """
    u = np.sum((y - y_pre) ** 2)
    v = np.sum((y - np.mean(y_pre)) ** 2)
    return 1 - (u / v)


# A simple least squares regression model from scratch.
# It is a subclass of the ObjectiveFunction class so that we
# can use it in the optimization module.
#
# The model is defined by the following equation:
#
# y = X @ W
#
# where X is the input data, W is the weight matrix, and y is the predicted output.
#
class LeastSquaresRegression(ObjectiveFunction):
    def __init__(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_test: ArrayLike,
        y_test: ArrayLike,
        use_bias: bool = True,
        use_multiple_initializations: bool = False,
        num_initializations: int = 5,
    ):
        ObjectiveFunction.__init__(self, name="linear_regression")
        self.X_train: ArrayLike = X_train
        self.y_train: ArrayLike = y_train
        self.X_test: ArrayLike = X_test
        self.y_test: ArrayLike = y_test
        self.use_bias = use_bias
        self.use_multiple_initializations = use_multiple_initializations
        self.num_initializations = num_initializations

        array_type, device = TensorBackend.get_array_type(X_train)
        self.backend = TensorBackend(array_type, device=device)

        self.input_dim = X_train.shape[1]
        self.num_samples_train = len(y_train)
        self.num_samples_test = len(y_test)

        self.X_train_homogeneous = (
            self.backend.homogeneous(self.X_train) if use_bias else self.X_train
        )
        self.X_test_homogeneous = (
            self.backend.homogeneous(self.X_test) if use_bias else self.X_test
        )

        self.W_init = self._initialize_parameters(
            self.input_dim,
            scale=0.1,
            use_bias=use_bias,
            use_multiple_initializations=use_multiple_initializations,
            num_initializations=num_initializations,
        )

    # loss computed by using as input W and the self.X_train_homogeneous matrix
    def _loss_train_w(self, W: ArrayLike) -> ArrayLike:
        # Just for single sample
        preds = self._linear_forward(self.X_train_homogeneous, W)
        preds_flattened = self.backend.flatten(preds)
        y_train_flattened = self.backend.flatten(self.y_train)
        loss = self.mean_squared_error(preds_flattened, y_train_flattened)
        return loss

    # gradient (w.r.t. W) of the loss computed by using as input W and the self.X_train_homogeneous matrix
    def _loss_train_w_gradient(self, W: ArrayLike) -> ArrayLike:
        # Just for single sample
        preds = self._linear_forward(self.X_train_homogeneous, W)
        preds_flattened = self.backend.flatten(preds)
        y_train_flattened = self.backend.flatten(self.y_train)
        error = preds_flattened - y_train_flattened
        grad = self.X_train_homogeneous.T @ error / self.num_samples_train
        return grad

    def __call__(self, W: ArrayLike) -> ArrayLike:
        # NOTE: To be used just a training time
        if W.ndim == 1:
            return self._loss_train_w(W)
        else:
            # TODO: this is not efficient, we should use jax.vmap in the case of jax backend
            losses = []
            num_init_W = W.shape[0]
            for i in range(num_init_W):
                losses.append(self._loss_train_w(W[i]))
            return self.backend.array(losses)

    def gradient(self, W: ArrayLike) -> ArrayLike:
        # NOTE: To be used just a training time
        if W.ndim == 1:
            return self._loss_train_w_gradient(W)
        else:
            grads = []
            num_init_W = W.shape[0]
            for i in range(num_init_W):
                grads.append(self._loss_train_w_gradient(W[i]))
            return self.backend.array(grads)

    def _linear_forward(self, X: ArrayLike, W: ArrayLike) -> ArrayLike:
        # self.backend.print_info(X, "X")
        # self.backend.print_info(W, "W")
        return X @ W if W.ndim == 1 else X @ W.T

    def mean_squared_error(self, preds: ArrayLike, targets: ArrayLike) -> float:
        preds = preds.reshape(-1)
        targets = targets.reshape(-1)
        assert (
            preds.shape == targets.shape
        ), f"Shape mismatch: preds {preds.shape}, targets {targets.shape}"
        error = preds - targets
        return self.backend.mean(error**2)

    def _initialize_parameters(
        self,
        input_dim: int,
        scale: float = 1.0,
        use_bias: bool = True,
        use_multiple_initializations: bool = True,
        num_initializations: int = 5,
    ) -> ArrayLike:
        if use_bias:
            input_dim += 1
        if use_multiple_initializations:
            W = [
                self.backend.random_normal((input_dim,)) * scale
                for _ in range(num_initializations)
            ]
            W = self.backend.array(W)
        else:
            W = self.backend.random_normal((input_dim,)) * scale
        print(f"LeastSquaresRegression: Initialized W.shape: {W.shape}")
        return W

    def metrics(self, data: Dict[str, ArrayLike]) -> Dict[str, float]:
        w_path = data.get("path", None)
        if w_path is None:
            return {}
        return {
            "mse_train": self.mse_train(w_path),
            "mse_test": self.mse_test(w_path),
            "r2_score_train": self.r2_score_train(w_path),
            "r2_score_test": self.r2_score_test(w_path),
        }

    def r2_score(self, W: ArrayLike, X: ArrayLike, y: ArrayLike) -> float:
        preds = self._linear_forward(X, W)  # (N,) or (N, B)
        y_flat = y.reshape(-1)  # (N,)
        if W.ndim == 1:
            return self.r2_score_single(preds, y_flat)
        else:
            y_expanded = y_flat[:, None]  # (N, 1)
            ss_res = self.backend.sum((y_expanded - preds) ** 2, axis=0)  # (B,)
            ss_tot = self.backend.sum(
                (y_expanded - self.backend.mean(y_expanded, axis=0)) ** 2, axis=0
            )  # (B,)
            return 1 - ss_res / ss_tot  # (B,)

    def r2_score_single(self, preds: ArrayLike, targets: ArrayLike) -> float:
        """
        Computes the R2 score for a single sample.

        Args:
            preds: Array of predicted values, shape (N,)
            targets: Array of target values, shape (N,)

        Returns:
            float: R2 score

        R2 score is defined as:
            1 - (SS_res / SS_tot)
        where:
            SS_res = sum((targets - preds) ** 2)   <- that is the residual sum of squares
            SS_tot = sum((targets - self.backend.mean(targets)) ** 2)  <- that is the total variance of the data
        It is a measure of how well the model fits the data.
        In particular, it measures how much of the variance in the data is explained by the model.
        The higher the R2 score, the better the model fits the data.
        The R2 score is always between 0 and 1.
        The R2 score of 1 means that the model perfectly fits the data.
        The R2 score of 0 means that the model does not fit the data at all.
        The R2 score of -1 means that the model is worse than a constant model.
        """
        # preds: (N,)
        # targets: (N,)
        ss_res = self.backend.sum((targets - preds) ** 2)
        ss_tot = self.backend.sum((targets - self.backend.mean(targets)) ** 2)
        return 1 - ss_res / ss_tot

    def mse(self, W: ArrayLike, X: ArrayLike, y: ArrayLike) -> float:
        # W: (B, D) or (D,) if single
        # X: (N, D)
        # y: (N,)
        preds = self._linear_forward(X, W)  # (N, B) or (N,) if single
        if W.ndim == 1:
            # preds: (N,)
            # y: (N,)
            return self.mean_squared_error(preds, y)
        else:
            # preds: (N, B)
            # y: (N,)   -> broadcasted to (N, B)
            # print(f"preds.shape: {preds.shape}, y.shape: {y.shape}")
            y_expanded = y.reshape(-1, 1)  # reshape (N,) -> (N, 1) for broadcasting
            error = preds - y_expanded  # (N, B)
            return self.backend.mean(error**2, axis=0)  # (B,)

    def mse_train(self, W: ArrayLike) -> float:
        return self.mse(W, self.X_train_homogeneous, self.y_train)

    def mse_test(self, W: ArrayLike) -> float:
        return self.mse(W, self.X_test_homogeneous, self.y_test)

    def r2_score_train(self, W: ArrayLike) -> float:
        return self.r2_score(W, self.X_train_homogeneous, self.y_train)

    def r2_score_test(self, W: ArrayLike) -> float:
        return self.r2_score(W, self.X_test_homogeneous, self.y_test)
