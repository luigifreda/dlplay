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


# A simple binary logistic regression model from scratch.
# It is a subclass of the ObjectiveFunction class so that we
# can use it in the optimization module.
#
# The model is defined by the following equation:
#
# y = sigmoid(X @ W)
#
# where X is the input data, W is the weight matrix, and y is the predicted output.
#
class BinaryLogisticRegression(ObjectiveFunction):
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
        ObjectiveFunction.__init__(self, name="logistic_regression")
        self.X_train: ArrayLike = X_train  # cached input data
        self.y_train: ArrayLike = y_train  # cached target data
        self.X_test: ArrayLike = X_test  # cached input data
        self.y_test: ArrayLike = y_test  # cached target data
        self.use_bias = use_bias
        self.use_multiple_initializations = use_multiple_initializations
        self.num_initializations = num_initializations

        array_type, device = TensorBackend.get_array_type(X_train)
        self.backend = TensorBackend(array_type, device=device)

        # check this is a binary classification problem
        if len(self.backend.unique(y_train)) != 2:
            raise ValueError(
                "BinaryLogisticRegression only supports binary classification"
            )

        self.input_dim = X_train.shape[1]  # D
        self.num_classes = len(self.backend.unique(y_train))  # C
        self.num_samples_train = len(y_train)  # N_train
        self.num_samples_test = len(y_test)  # N_test

        self.y_train = self.backend.cast(y_train, float)
        self.y_test = self.backend.cast(y_test, float)

        # add bias term to the input data if use_bias is True (so we get an homogeneous
        # representation of the data)
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
        self.logits: ArrayLike = None  # cached logits
        self.preds: ArrayLike = None  # cached predictions
        self.loss: float = None  # cached loss

    # loss computed by using as input W and the self.X_train_homogeneous matrix
    def _loss_train_w(self, W: ArrayLike) -> ArrayLike:
        # Just for single sample
        self.logits = self._linear_forward(self.X_train_homogeneous, W)  # (N, 1)
        # self.preds = self._sigmoid(self.logits)  # (N, 1)
        # preds_flattened = self.backend.flatten(self.preds)
        # y_train_flattened = self.backend.flatten(self.y_train)
        # self.loss = self.binary_cross_entropy(
        #     preds_flattened, y_train_flattened
        # )
        logits_flattened = self.backend.flatten(self.logits)
        y_train_flattened = self.backend.flatten(self.y_train)
        self.loss = self.binary_cross_entropy_from_logits(
            logits_flattened, y_train_flattened
        )
        return self.loss

    # gradient (w.r.t. W) of the loss computed by using as input W and the self.X_train_homogeneous matrix
    def _loss_train_w_gradient(self, W: ArrayLike) -> ArrayLike:
        # Just for single sample
        self.logits = self._linear_forward(self.X_train_homogeneous, W)  # (N, 1)
        self.preds = self._sigmoid(self.logits)  # (N, 1)
        preds_flattened = self.backend.flatten(self.preds)
        y_train_flattened = self.backend.flatten(self.y_train)
        error = preds_flattened - y_train_flattened
        # error: (N,)
        # X_train_homogeneous: (N, D)
        # grad: (D, 1)
        # grad = self.X_train_homogeneous.T @ error / self.num_samples_train
        error = error[..., None]  # (N, 1)
        grad = (
            self.backend.einsum("nd,nc->dc", self.X_train_homogeneous, error)
            / self.num_samples_train
        )  # (D, 1)
        return self.backend.flatten(grad)  # (D,)

    def __call__(self, W: ArrayLike) -> ArrayLike:
        # NOTE: To be used just a training time
        if W.ndim == 1:
            # we have a single parameter W: (D,)
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
            # We have a single parameter W: (D,)
            return self._loss_train_w_gradient(W)
        else:
            # We have a batch of parameters W: (B, D)
            grads = []
            num_init_W = W.shape[0]
            for i in range(num_init_W):
                grads.append(self._loss_train_w_gradient(W[i]))
            return self.backend.array(grads)

    def _linear_forward(self, X: ArrayLike, W: ArrayLike) -> ArrayLike:
        if W.ndim == 1:
            # W.shape = (D,)
            # X: (N, D)
            W = W[..., None]  # from (D,) to (D, 1) , needed under tensorflow
            return X @ W  # (N, 1)
        else:
            # W.shape = (B, D)
            # X: (N, D)
            # return X @ W.T  # (N, B)
            return self.backend.einsum("nd,bd->nb", X, W)  # (N, B)

    def _sigmoid(self, z: ArrayLike) -> ArrayLike:
        max_val = 1e3
        z = self.backend.clip(z, -max_val, max_val)  # prevent overflow in exp
        return 1 / (1 + self.backend.exp(-z))

    def binary_cross_entropy(self, preds: ArrayLike, targets: ArrayLike) -> float:
        # preds: (N,)
        # targets: (N,)
        preds = self.backend.clip(preds, 1e-10, 1 - 1e-10)
        res = -self.backend.mean(
            targets * self.backend.log(preds)
            + (1 - targets) * self.backend.log(1 - preds)
        )
        return res

    def binary_cross_entropy_from_logits(
        self, logits: ArrayLike, targets: ArrayLike
    ) -> float:
        """
        Computes the binary cross-entropy loss directly from logits using a numerically stable formula.

        This function avoids computing the sigmoid explicitly by using a reformulation of BCE
        that incorporates the log-sum-exp trick for numerical stability.

        Standard BCE (from probabilities):
            BCE(y, p) = -y * log(p) - (1 - y) * log(1 - p)

        Rewritten in terms of logits z (where p = sigmoid(z)):
            BCE(y, z) = -y * log(sigmoid(z)) - (1 - y) * log(1 - sigmoid(z))

        Note that:
            log(1+e^(-z)) = log(e^0 + e^(-z)) = log( sum_i exp(z_i))
            where z0 = 0 and z_1 = -z

        Applying the log-sum-exp trick, this becomes:
            BCE(y, z) = max(z, 0) - y * z + log(1 + exp(-|z|))

        This formulation avoids overflow/underflow when z is large in magnitude,
        by:
        - rewriting log-sigmoid terms in terms of log(1 + exp(...))
        - guarding the exponential using abs(z)
        - ensuring stable addition using max(z, 0)

        Args:
            logits: Array of raw model outputs (before sigmoid), shape (N,)
            targets: Array of binary targets (0 or 1), shape (N,)

        Returns:
            Mean binary cross entropy loss over the batch.
        """
        # logits: (N,)
        # targets: (N,)
        zeros = self.backend.zeros_like(logits)
        res = (
            self.backend.maximum(logits, zeros)
            - logits * targets
            + self.backend.log(1 + self.backend.exp(-self.backend.abs(logits)))
        )
        return self.backend.mean(res)

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
            W = []
            for _ in range(num_initializations):
                W.append(self.backend.random_normal((input_dim,)) * scale)
            W = self.backend.array(W)
        else:
            W = self.backend.random_normal((input_dim,)) * scale
        print(f"BinaryLogisticRegression: Initialized W.shape: {W.shape}")
        return W

    def metrics(self, data: Dict[str, ArrayLike]) -> Dict[str, float]:
        """
        Compute metrics on the path of weights.

        Args:
            data: Dictionary containing the path of weights and the objective function values

        Returns:
            Dictionary containing the metrics

        Note:
            The data dictionary is expected to contain the following:
            data_for_metrics = {
                "path": path,
                "objective_function_values": objective_function_values,
            }
        """
        w_path = None
        # objective_function_values = None
        if "path" in data:
            w_path = data["path"]
        # if "objective_function_values" in data:
        #    objective_function_values = data["objective_function_values"]
        results = {}
        # print(f"metrics evaluation on w_path: {w_path.shape}")
        results["accuracy_train"] = self.accuracy_train(w_path)
        results["accuracy_test"] = self.accuracy_test(w_path)
        return results

    def accuracy(
        self, W: ArrayLike, X: ArrayLike, y: ArrayLike, decision_threshold: float = 0.5
    ) -> float:
        logits = self._linear_forward(X, W)  # (N, B) or (N,) if single
        preds = self._sigmoid(logits)  # same shape as logits
        targets = self.backend.flatten(y)  # (N,)
        targets_expanded = targets.reshape(-1, 1)  # (N, 1) — will broadcast

        assert preds.shape[0] == targets_expanded.shape[0], (
            f"preds.shape[0] ({preds.shape[0]}) != targets_expanded.shape[0]"
            f"({targets_expanded.shape[0]})"
        )

        if W.ndim == 1:
            # preds: (N,)
            # targets_expanded: (N, 1)
            num_samples = X.shape[0]
            matches = (preds > decision_threshold) == targets_expanded
            # Cast boolean to float before summing
            matches = self.backend.cast(matches, int)
            return self.backend.sum(matches) / num_samples
        else:
            # preds: (N, B),
            # targets_expanded: (N, 1) and is broadcasted along axis 1
            num_samples = X.shape[0]
            matches = (preds > decision_threshold) == targets_expanded
            # Cast boolean to float before summing
            matches = self.backend.cast(matches, int)
            return self.backend.sum(matches, axis=0) / num_samples

    def accuracy_train(self, W: ArrayLike) -> float:
        return self.accuracy(W, self.X_train_homogeneous, self.y_train)

    def accuracy_test(self, W: ArrayLike) -> float:
        return self.accuracy(W, self.X_test_homogeneous, self.y_test)


# A simple logistic regression model from scratch.
# It is a subclass of the ObjectiveFunction class so that we
# can use it in the optimization module.
#
# The model is defined by the following equation:
#
# y = softmax(X @ W)
#
# where X is the input data, W is the weight matrix, and y is the predicted output.
#
class MulticlassLogisticRegression(ObjectiveFunction):
    def __init__(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_test: ArrayLike,
        y_test: ArrayLike,
        use_bias: bool = True,
        use_multiple_initializations: bool = True,
        num_initializations: int = 5,
    ):
        ObjectiveFunction.__init__(self, name="logistic_regression")
        self.X_train: ArrayLike = X_train  # cached input data   (N_train, D)
        self.y_train: ArrayLike = y_train  # cached target data   (N_train,)
        self.X_test: ArrayLike = X_test  # cached input data   (N_test, D)
        self.y_test: ArrayLike = y_test  # cached target data   (N_test,)
        self.use_bias = use_bias
        self.use_multiple_initializations = use_multiple_initializations
        self.num_initializations = num_initializations

        array_type, device = TensorBackend.get_array_type(X_train)
        self.backend = TensorBackend(array_type, device=device)

        self.input_dim = X_train.shape[1]  # D
        self.num_classes = len(self.backend.unique(y_train))  # C
        self.num_samples_train = len(y_train)  # N_train
        self.num_samples_test = len(y_test)  # N_test

        self.y_train = self.backend.cast(y_train, float)
        self.y_test = self.backend.cast(y_test, float)

        self.y_train_one_hot = self.backend.one_hot(
            y_train, self.num_classes
        )  # (N_train, C)
        self.y_test_one_hot = self.backend.one_hot(
            y_test, self.num_classes
        )  # (N_test, C)

        # add bias term to the input data if use_bias is True (so we get an homogeneous
        # representation of the data)
        self.X_train_homogeneous = (
            self.backend.homogeneous(self.X_train) if use_bias else self.X_train
        )
        self.X_test_homogeneous = (
            self.backend.homogeneous(self.X_test) if use_bias else self.X_test
        )

        # W: (D, C)
        self.W_init = self._initialize_parameters(
            self.input_dim,
            self.num_classes,
            scale=0.1,
            use_bias=use_bias,
            use_multiple_initializations=use_multiple_initializations,
            num_initializations=num_initializations,
        )
        self.logits: ArrayLike = None  # cached logits
        self.preds: ArrayLike = None  # cached predictions
        self.loss: float = None  # cached loss

    # loss computed by using as input W and the self.X_train_homogeneous matrix
    def _loss_train_w(self, W: ArrayLike) -> ArrayLike:
        # Just for single sample
        # W: (D,C)
        # X_train_homogeneous: (N, D)
        # logits: (N,C)
        # preds: (N,C)
        # loss: float
        self.logits = self._linear_forward(self.X_train_homogeneous, W)  # (N, C)
        # self.preds = self._softmax(self.logits)  # (N, C)
        # self.loss = self.cross_entropy(self.preds, self.y_train_one_hot)
        self.loss = self.cross_entropy_from_logits(self.logits, self.y_train_one_hot)
        return self.loss

    # gradient (w.r.t. W) of the loss computed by using as input W and the self.X_train_homogeneous matrix
    def _loss_train_w_gradient(self, W: ArrayLike) -> ArrayLike:
        # Just for single sample
        # W: (D,C)
        # X_train_homogeneous: (N, D)
        # logits: (N,C)
        # preds: (N,C)
        # y_train_one_hot: (N, C)
        self.logits = self._linear_forward(self.X_train_homogeneous, W)  # (N, C)
        self.preds = self._softmax(self.logits)  # (N, C)
        error = self.preds - self.y_train_one_hot  # (N, C)
        # grad: (D, C)
        # grad = self.X_train_homogeneous.T @ error / self.num_samples_train  # (D, C)
        grad = (
            self.backend.einsum("nd,nc->dc", self.X_train_homogeneous, error)
            / self.num_samples_train
        )  # (D, C)
        return grad  # (D,)

    def __call__(self, W: ArrayLike) -> ArrayLike:
        # To be used just a training time
        if W.ndim == 2:
            # we have a single sample W: (D, C)
            return self._loss_train_w(W)  # float
        else:
            # batch mode W: (B, D, C)
            # TODO: this is not efficient, we should use jax.vmap in the case of jax backend
            losses = []
            num_samples = W.shape[0]
            for i in range(num_samples):
                losses.append(self._loss_train_w(W[i]))
            return self.backend.array(losses)  # (B,)

    def gradient(self, W: ArrayLike) -> ArrayLike:
        # To be used just a training time
        if W.ndim == 2:
            # we have a single sample W: (D, C)
            return self._loss_train_w_gradient(W)
        else:
            # batch mode W: (B, D, C)
            grads = []
            num_samples = W.shape[0]
            for i in range(num_samples):
                grads.append(self._loss_train_w_gradient(W[i]))
            return self.backend.array(grads)

    def _linear_forward(self, X: ArrayLike, W: ArrayLike) -> ArrayLike:
        if W.ndim == 2:
            # W.shape = (D, C)
            # X: (N, D)
            return X @ W  # (N, C)
        else:
            # W: (B, D, C)
            # X: (N, D)
            # res = X @ W  # (N, C, B)
            # return res.transpose(2, 0, 1)  # (B, N, C)
            res = self.backend.einsum("nd,bdc->bnc", X, W)  # (B, N, C)
            return res

    def _softmax(self, z: ArrayLike) -> ArrayLike:
        max_val = 1e3
        z = self.backend.clip(z, -max_val, max_val)  # prevent overflow in exp
        exp_z = self.backend.exp(z)
        return exp_z / self.backend.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy(self, preds: ArrayLike, targets_one_hot: ArrayLike) -> float:
        # preds: (N, C)
        # targets_one_hot: (N, C)
        if targets_one_hot.shape != preds.shape:
            raise ValueError(
                f"targets_one_hot.shape ({targets_one_hot.shape}) != preds.shape ({preds.shape})"
            )
        preds = self.backend.clip(preds, 1e-10, 1 - 1e-10)
        # targets_one_hot: (N, C)
        # preds: (N, C)
        res = -self.backend.mean(targets_one_hot * self.backend.log(preds))
        return res

    def cross_entropy_from_logits(
        self, logits: ArrayLike, targets_one_hot: ArrayLike
    ) -> float:
        """
        Computes the categorical cross-entropy loss directly from logits using a numerically stable formulation.

        This implementation avoids computing the softmax explicitly to prevent numerical instability
        when logits are large in magnitude.

        Standard cross-entropy loss (from probabilities):
            CE(y, p) = -sum_c y_c * log(p_c)

        With logits z (p = softmax(z)), this becomes:
            CE(y, z) = -sum_c y_c * log(softmax(z_c))
                    = -sum_c y_c * (z_c - log(sum_j exp(z_j)))
                    = log(sum_j exp(z_j)) - z_true

        This leads to a numerically stable implementation:
            1. Subtract max_logit per sample before computing exp (log-sum-exp trick)
            2. Compute log(sum_j exp(z_j - max_logit)) + max_logit
            3. Subtract the true class logit z_true

        Args:
            logits: Array of raw model outputs (before softmax), shape (N, C)
            targets_one_hot: One-hot encoded class labels, shape (N, C)

        Returns:
            float: Mean cross entropy loss over the batch.
        """
        if targets_one_hot.shape != logits.shape:
            raise ValueError(
                f"targets_one_hot.shape ({targets_one_hot.shape}) != logits.shape ({logits.shape})"
            )

        # (N, C)
        x = logits
        y = targets_one_hot

        # Compute per-sample max for numerical stability
        max_logits = self.backend.max(x, axis=1, keepdims=True)  # (N, 1)

        # Compute log(sum(exp(logits - max))) + max
        log_sum_exp = (
            self.backend.log(
                self.backend.sum(
                    self.backend.exp(x - max_logits), axis=1, keepdims=True
                )
            )
            + max_logits
        )  # shape: (N, 1)

        # Compute dot product: sum_c y_c * z_c => just select z_true (shape: (N, 1))
        z_true = self.backend.sum(y * x, axis=1, keepdims=True)

        # Final cross-entropy: log_sum_exp - z_true
        res = log_sum_exp - z_true  # shape: (N, 1)

        return self.backend.mean(res)

    def _initialize_parameters(
        self,
        input_dim: int,
        num_classes: int,
        scale: float = 1.0,
        use_bias: bool = True,
        use_multiple_initializations: bool = True,
        num_initializations: int = 5,
    ) -> ArrayLike:
        if use_bias:
            input_dim += 1
        if use_multiple_initializations:
            W = []
            for _ in range(num_initializations):
                W.append(self.backend.random_normal((input_dim, num_classes)) * scale)
            W = self.backend.array(W)
        else:
            W = self.backend.random_normal((input_dim, num_classes)) * scale
        print(f"MulticlassLogisticRegression: Initialized W.shape: {W.shape}")
        return W

    def metrics(self, data: Dict[str, ArrayLike]) -> Dict[str, float]:
        # It is expected that the data dictionary contains the following:
        # data_for_metrics = {
        #     "path": path,
        #     "objective_function_values": objective_function_values,
        # }
        w_path = None
        objective_function_values = None
        if "path" in data:
            w_path = data["path"]
        if "objective_function_values" in data:
            objective_function_values = data["objective_function_values"]
        results = {}
        results["accuracy_train"] = self.accuracy_train(w_path)
        results["accuracy_test"] = self.accuracy_test(w_path)
        return results

    def accuracy(self, W: ArrayLike, X: ArrayLike, y: ArrayLike) -> float:
        # W: (D, C) or (B, D, C)
        # X: (N, D) or (N, D, B)
        # y: (N,)
        logits = self._linear_forward(X, W)  # (B,N,C) or (N,C) if single
        preds = self._softmax(logits)  # same shape as logits
        preds_argmax = self.backend.argmax(preds, axis=-1)  # (B,N) or (N,) if single
        targets = self.backend.flatten(y)  # (N,)

        if W.ndim == 2:
            # Single weight matrix case W: (D, C)
            # preds_argmax: (N,), targets: (N,)
            assert preds_argmax.shape[0] == targets.shape[0], (
                f"preds_argmax.shape[0] ({preds_argmax.shape[0]}) != targets.shape[0] "
                f"({targets.shape[0]})"
            )
            num_samples = X.shape[0]
            matches = preds_argmax == targets
            matches = self.backend.cast(matches, int)
            return self.backend.sum(matches) / num_samples
        else:
            # Batched weights case W: (B, D, C)
            # preds_argmax: (B, N), targets: (N,)
            assert preds_argmax.shape[1] == targets.shape[0], (
                f"preds_argmax.shape[1] ({preds_argmax.shape[1]}) != targets.shape[0] "
                f"({targets.shape[0]})"
            )
            # Compare each batch prediction with targets
            # preds_argmax: (B, N)
            # targets: (N,)
            # targets_broadcasted: (B, N)
            targets_broadcasted = self.backend.broadcast(targets, preds_argmax.shape[0])
            # print(f"preds_argmax.shape: {preds_argmax.shape}")
            # print(f"targets_broadcasted.shape: {targets_broadcasted.shape}")
            num_samples = X.shape[0]
            matches = preds_argmax == targets_broadcasted
            matches = self.backend.cast(matches, int)
            return self.backend.sum(matches, axis=1) / num_samples

    def accuracy_train(self, W: ArrayLike) -> float:
        return self.accuracy(W, self.X_train_homogeneous, self.y_train)

    def accuracy_test(self, W: ArrayLike) -> float:
        return self.accuracy(W, self.X_test_homogeneous, self.y_test)
