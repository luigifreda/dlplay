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
Gradient Descent Optimization on a 2D Function using NumPy (Refactored)
-----------------------------------------------------------------------
This script uses the class-based approach to manage gradient descent optimization.

It includes:
- Vectorized implementation of the function
- Hard-coded gradient
- Gradient descent (with and without momentum)
- Visualization of optimization paths and gradient norms
"""

import numpy as np
import matplotlib.pyplot as plt

from dlplay.optimization.optimization import (
    ObjectiveFunction,
    GradientDescentVanillaOptimizer,
    GradientDescentMomentumOptimizer,
    OptimizationRunner,
    print_optimization_summary,
)
from dlplay.viz.plotting import plot_grad_descent_paths

from common_init import DEFAULT_START_POINTS, init_start_points, create_grid_points


class NumPyObjectiveFunction(ObjectiveFunction):
    """
    NumPy implementation of the objective function:
    f(x) = sin(x1)cos(x2) + sin(0.5x1)cos(0.5x2)
    """

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        f(x) = sin(x1)cos(x2) + sin(0.5x1)cos(0.5x2)
        """
        x1 = X[:, 0]
        x2 = X[:, 1]
        return np.sin(x1) * np.cos(x2) + np.sin(0.5 * x1) * np.cos(0.5 * x2)

    def gradient(self, X: np.ndarray) -> np.ndarray:
        """
        Gradient of f(x) = sin(x1)cos(x2) + sin(0.5x1)cos(0.5x2)
        df_dx1 = cos(x1)cos(x2) + 0.5cos(0.5x1)cos(0.5x2)
        df_dx2 = -sin(x1)sin(x2) - 0.5sin(0.5x1)sin(0.5x2)
        """
        x1 = X[:, 0]
        x2 = X[:, 1]
        df_dx1 = np.cos(x1) * np.cos(x2) + 0.5 * np.cos(0.5 * x1) * np.cos(0.5 * x2)
        df_dx2 = -np.sin(x1) * np.sin(x2) - 0.5 * np.sin(0.5 * x1) * np.sin(0.5 * x2)
        return np.stack([df_dx1, df_dx2], axis=1)


def main():
    """Run the NumPy gradient descent experiment."""

    # Create objective function
    objective_function = NumPyObjectiveFunction()

    # Create optimizer
    optimizer_vanilla = GradientDescentVanillaOptimizer(
        objective_function=objective_function,
        learning_rate=0.1,
        max_iterations=50,
    )
    optimizer_momentum = GradientDescentMomentumOptimizer(
        objective_function=objective_function,
        learning_rate=0.1,
        momentum=0.9,
        max_iterations=50,
    )

    # Get starting points
    start_points = init_start_points("numpy", DEFAULT_START_POINTS)

    # Create and run experiment
    optimizer_runner_vanilla = OptimizationRunner(
        optimizer=optimizer_vanilla, start_points=start_points, framework="numpy"
    )
    optimizer_runner_momentum = OptimizationRunner(
        optimizer=optimizer_momentum, start_points=start_points, framework="numpy"
    )

    # Run all experiments
    vanilla_results = optimizer_runner_vanilla.run()
    momentum_results = optimizer_runner_momentum.run()

    # Print summaries
    print_optimization_summary(vanilla_results, "Vanilla Gradient Descent (NumPy)")
    print_optimization_summary(
        momentum_results, "Gradient Descent with Momentum (NumPy)"
    )

    # Create grid points
    grid_points = create_grid_points()

    # Plot results
    plot_grad_descent_paths(
        objective_function,
        vanilla_results,
        grid_points=grid_points,
        title="Gradient Descent (NumPy)",
        framework="numpy",
    )
    plot_grad_descent_paths(
        objective_function,
        momentum_results,
        grid_points=grid_points,
        title="Gradient Descent with Momentum (NumPy)",
        framework="numpy",
    )

    plt.show()


if __name__ == "__main__":
    main()
