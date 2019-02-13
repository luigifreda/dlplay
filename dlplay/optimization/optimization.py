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
"""
Common utilities for gradient descent optimization
-------------------------------------------------
This module provides shared functionality for gradient descent optimization
across different frameworks (NumPy, PyTorch, JAX).

It includes:
- Objective function definitions
- Gradient function definitions
- Gradient descent algorithms
- Visualization functions
- Common configuration
"""

import os

from dlplay.viz.plotting import plot_grad_descent_paths
from dlplay.utils.types import ArrayLike, OptimizationPathData
from dlplay.core.tensor_backend import TensorBackend

from typing import List, Tuple, Callable, Any, Union, Optional, Dict
from abc import ABC, abstractmethod


# Common configuration
DEFAULT_LEARNING_RATE = 1e-2
DEFAULT_MOMENTUM = 0.9
DEFAULT_STEPS = 100
DEFAULT_MAX_GRAD_NORM = 10.0
DEFAULT_GRADIENT_NORM_CONVERGENCE_THRESHOLD = 1e-3
DEFAULT_CONVERGENCE_COUNTER_THRESHOLD = 3


class ObjectiveFunction(ABC):
    """Abstract base class for objective functions."""

    def __init__(self, name: str = "objective_function"):
        self.name = name

    @abstractmethod
    def __call__(self, X: ArrayLike) -> ArrayLike:
        """Evaluate the objective function."""
        pass

    @abstractmethod
    def gradient(self, X: ArrayLike) -> ArrayLike:
        """Compute the gradient of the objective function."""
        pass

    # optional to include accuracy, and other metrics for better monitoring
    # the optimization process and the model fitting process
    def metrics(self, data: Dict[str, ArrayLike]) -> Dict[str, ArrayLike]:
        """Compute the metrics of the objective function."""
        return {}


class OptimizerBase(ABC):
    """
    Abstract base class for optimization algorithms.

    This class provides a unified interface for optimization algorithms
    across different frameworks (NumPy, PyTorch, JAX).
    """

    def __init__(
        self,
        objective_function: ObjectiveFunction,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        max_iterations: int = DEFAULT_STEPS,
        max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
        name: str = "optimizer",
    ):
        """
        Initialize the optimizer.

        Args:
            objective_function: The objective function to optimize
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum number of optimization steps
            name: Name of the optimizer
        """
        self.name = name
        self.objective_function = objective_function
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.history = []
        self.max_grad_norm = max_grad_norm
        self.convergence_counter = 0
        self.convergence_grad_norm_threshold = (
            DEFAULT_GRADIENT_NORM_CONVERGENCE_THRESHOLD
        )
        self.convergence_counter_threshold = DEFAULT_CONVERGENCE_COUNTER_THRESHOLD
        self.backend = TensorBackend()

    @abstractmethod
    def optimize(
        self,
        x0: ArrayLike,
        callback: Optional[Callable] = None,
    ) -> OptimizationPathData:
        """
        Run optimization.

        Args:
            x0: Initial point
            callback: Optional callback function called at each step

        Returns:
            Tuple of (path, objective_function_values, grad_norms)
        """
        pass

    def _compute_gradient(self, x: ArrayLike) -> ArrayLike:
        """Compute gradient at point x."""
        x_unsqueezed = self.backend.unsqueeze(x)
        grad = self.objective_function.gradient(x_unsqueezed)
        return grad[0] if grad.ndim > 1 else grad

    def _compute_grad_norm(self, grad: ArrayLike) -> float:
        """Compute gradient norm."""
        return self.backend.norm(grad)

    def has_converged(self, grad_norm: float) -> bool:
        """Check if the gradient norm has converged."""
        # check convergence, if the gradient has been below the threshold for a while
        if grad_norm < self.convergence_grad_norm_threshold:
            self.convergence_counter += 1
        else:
            self.convergence_counter = 0
        return self.convergence_counter > self.convergence_counter_threshold


class GradientDescentVanillaOptimizer(OptimizerBase):
    """
    Vanilla gradient descent optimizer.

    Implements standard gradient descent without momentum.
    """

    def __init__(
        self,
        objective_function: ObjectiveFunction,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        max_iterations: int = DEFAULT_STEPS,
        max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
        name: str = "vanilla_gd",
    ):
        super().__init__(
            objective_function, learning_rate, max_iterations, max_grad_norm, name
        )

    def optimize(
        self,
        x0: ArrayLike,
        callback: Optional[Callable] = None,
    ) -> OptimizationPathData:
        """
        Run vanilla gradient descent optimization.

        Args:
            x0: Initial point
            callback: Optional callback function called at each step

        Returns:
            Tuple of (path, objective_function_values, grad_norms)
        """
        array_type, device = self.backend.get_array_type(x0)
        self.backend.init(array_type, device)

        x = self.backend.copy_and_convert_to_float(x0)
        self.learning_rate = self.backend.scalar(self.learning_rate, x.dtype)

        path = [x]
        grad_norms = []
        self.convergence_counter = 0

        for step in range(self.max_iterations):
            grad = self._compute_gradient(x)
            grad_norm = self._compute_grad_norm(grad)

            # gradient clipping
            if grad_norm > self.max_grad_norm:
                grad = (grad / grad_norm) * self.max_grad_norm

            x = self._update_position(x, grad)

            # path.append(self.backend.to_numpy(x))
            path.append(x)

            grad_norms.append(grad_norm)
            if callback:
                callback(step, x, grad, path, grad_norms)

            if self.has_converged(grad_norm):
                print(f"Optimizer: {self.name} has converged at step {step}")
                break

        path = self.backend.array(path)
        objective_function_values = self.backend.to_numpy(self.objective_function(path))
        grad_norms = self.backend.to_numpy(grad_norms)
        data_for_metrics = {
            "path": path,
            "objective_function_values": objective_function_values,
        }
        metrics = self.objective_function.metrics(data_for_metrics)
        path = self.backend.to_numpy(path)
        for metric_name, metric_values in metrics.items():
            metrics[metric_name] = self.backend.to_numpy(metric_values)
        return (
            path,
            objective_function_values,
            grad_norms,
            metrics,
        )

    def _update_position(self, x: ArrayLike, grad: ArrayLike) -> ArrayLike:
        """Update position using vanilla gradient descent."""
        # self.backend.print_info(x, "x")
        # self.backend.print_info(grad, "grad")
        # self.backend.print_info(self.learning_rate, "learning_rate")
        return x - self.learning_rate * grad


class GradientDescentMomentumOptimizer(OptimizerBase):
    """
    Gradient descent optimizer with momentum.

    Implements gradient descent with momentum for faster convergence.
    """

    def __init__(
        self,
        objective_function: ObjectiveFunction,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        momentum: float = DEFAULT_MOMENTUM,
        max_iterations: int = DEFAULT_STEPS,
        max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
        name: str = "momentum_gd",
    ):
        """
        Initialize the momentum optimizer.

        Args:
            objective_function: The objective function to optimize
            learning_rate: Learning rate for gradient descent
            momentum: Momentum coefficient
            max_iterations: Maximum number of optimization steps
        """
        super().__init__(
            objective_function, learning_rate, max_iterations, max_grad_norm, name
        )
        self.momentum = momentum

    def optimize(
        self,
        x0: ArrayLike,
        callback: Optional[Callable] = None,
    ) -> OptimizationPathData:
        """
        Run gradient descent optimization with momentum.

        Args:
            x0: Initial point
            callback: Optional callback function called at each step

        Returns:
            Tuple of (path, objective_function_values, grad_norms)
        """
        array_type, device = self.backend.get_array_type(x0)
        self.backend.init(array_type, device)

        x = self.backend.copy_and_convert_to_float(x0)
        v = self.backend.zeros_like(x)

        self.learning_rate = self.backend.scalar(self.learning_rate, x.dtype)
        self.momentum = self.backend.scalar(self.momentum, x.dtype)

        path = [x]
        grad_norms = []
        self.convergence_counter = 0

        for step in range(self.max_iterations):
            grad = self._compute_gradient(x)
            grad_norm = self._compute_grad_norm(grad)

            # gradient clipping
            if grad_norm > self.max_grad_norm:
                grad = (grad / grad_norm) * self.max_grad_norm

            v = self._update_velocity(v, grad)
            x = self._update_position_momentum(x, v)

            # path.append(self.backend.to_numpy(x))
            path.append(x)

            grad_norms.append(grad_norm)
            if callback:
                callback(step, x, grad, path, grad_norms)

            if self.has_converged(grad_norm):
                print(f"Optimizer: {self.name} has converged at step {step}")
                break

        path = self.backend.array(path)
        objective_function_values = self.backend.to_numpy(self.objective_function(path))

        grad_norms = self.backend.to_numpy(grad_norms)
        data_for_metrics = {
            "path": path,
            "objective_function_values": objective_function_values,
        }
        metrics = self.objective_function.metrics(data_for_metrics)
        path = self.backend.to_numpy(path)
        for metric_name, metric_values in metrics.items():
            metrics[metric_name] = self.backend.to_numpy(metric_values)
        return (
            path,
            objective_function_values,
            grad_norms,
            metrics,
        )

    def _update_velocity(self, v: ArrayLike, grad: ArrayLike) -> ArrayLike:
        """Update velocity for momentum."""
        return self.momentum * v - self.learning_rate * grad

    def _update_position_momentum(self, x: ArrayLike, v: ArrayLike) -> ArrayLike:
        """Update position using momentum."""
        return x + v


class OptimizationRunner:
    """
    A class to manage and run an optimization.
    """

    def __init__(
        self,
        optimizer: OptimizerBase,
        start_points: List[ArrayLike],
        framework: str = "numpy",
    ):
        """
        Initialize the optimization runner.

        Args:
            optimizer: The gradient descent optimizer
            start_points: List of starting points
            framework: Framework being used ('numpy', 'pytorch', 'jax')
        """
        self.optimizer = optimizer
        self.start_points = start_points
        self.framework = framework
        self.results = {}

    def run(self) -> List[OptimizationPathData]:
        """
        Run optimization.

        Returns:
            List of (path, objective_function_values, grad_norms) tuples
        """
        results = []

        for i, x0 in enumerate(self.start_points):
            path, objective_function_values, grad_norms, metrics = (
                self.optimizer.optimize(x0)
            )
            results.append((path, objective_function_values, grad_norms, metrics))

        self.results[self.optimizer.name] = results
        return results

    def plot_results(self, title: str = "Optimization Results") -> None:
        """
        Plot all experiment results.
        """
        if not self.results:
            print("No results to plot. Run experiments first.")
            return

        for experiment_name, results in self.results.items():
            plot_grad_descent_paths(
                self.optimizer.objective_function,
                results,
                title=f"{title} - {experiment_name.replace('_', ' ').title()}",
                framework=self.framework,
            )


def print_optimization_summary(data: List[OptimizationPathData], title: str) -> None:
    """
    Print a summary of optimization results.

    Args:
        data: List of (path, objective_function_values, grad_norms) tuples
        title: Title for the summary
    """
    print(f"\n{title}")
    print("-" * len(title))

    for i, (path, objective_function_values, grad_norms, metrics) in enumerate(data):
        final_objective_function_value = objective_function_values[-1]
        final_grad_norm = grad_norms[-1]
        initial_objective_function_value = objective_function_values[0]
        initial_grad_norm = grad_norms[0]
        convergence_ratio = (
            final_grad_norm / initial_grad_norm if initial_grad_norm > 0 else 0
        )

        # print metrics
        metrics_str = ""
        for j, (metric_name, metric_values) in enumerate(metrics.items()):
            metrics_str += f" Initial/Final {metric_name}: {metric_values[0]:.4f}/{metric_values[-1]:.4f} | "

        print(
            f"Path {i+1}: "
            f"Steps={len(path)} | "
            f"Initial/Final objective funct. value: {initial_objective_function_value:.4f}/{final_objective_function_value:.4f} | "
            f"Initial/Final grad norm: {initial_grad_norm:.4f}/{final_grad_norm:.4f} | "
            f"Convergence ratio: {convergence_ratio:.4f} | "
            f"{metrics_str}"
        )
