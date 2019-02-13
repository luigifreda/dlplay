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
import matplotlib.pyplot as plt
import numpy as np
import torch
import jax

from dlplay.models.linear.least_squares_regression import (
    LeastSquaresRegression,
    MSE,
    RMSE,
    MAPE,
    R2,
)
from dlplay.datasets.toydatasets_sklearn import (
    SklearnToyDatasetLoader,
    ToyDatasetType,
    ProblemType,
)
from dlplay.optimization.optimization import (
    GradientDescentVanillaOptimizer,
    GradientDescentMomentumOptimizer,
    OptimizationRunner,
    print_optimization_summary,
)
from dlplay.utils.conversions import to_torch, to_jax, to_numpy, JaxUtils, to_tensorflow
from dlplay.utils.device import resolve_device

import sklearn.linear_model


def sklearn_baseline(X_train, X_test, y_train, y_test, max_iter=1000):
    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    mse = MSE(y_test, preds)
    rmse = RMSE(y_test, preds)
    mape = MAPE(y_test, preds)
    print(
        f"Sklearn least squares regression - R2 score: {r2:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}"
    )


def main():
    dataset_name = ToyDatasetType.DIABETES
    problem_type = ProblemType.REGRESSION
    use_dataset_scaler = True

    dataset_loader = SklearnToyDatasetLoader(
        dataset_name=dataset_name,
        problem_type=problem_type,
        use_dataset_scaler=use_dataset_scaler,
    )

    X_train, y_train = dataset_loader.get_train_data()
    X_test, y_test = dataset_loader.get_test_data()

    print(f"dataset name: {dataset_name}")
    print(f"problem type: {problem_type}")
    print(f"use scaler: {use_dataset_scaler}")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    sklearn_baseline(X_train, X_test, y_train, y_test, max_iter=1000)

    use_multiple_initializations = True
    print(f"use multiple initializations: {use_multiple_initializations}")

    # Select backend and convert tensors to the backend
    backend = "torch"  # "numpy" | "torch" | "jax" | "tensorflow"
    if backend == "torch":
        torch_device = resolve_device() #"cuda" if torch.cuda.is_available() else "cpu"
        print(f"torch device: {torch_device}")
        X_train = to_torch(X_train, device=torch_device)
        X_test = to_torch(X_test, device=torch_device)
        y_train = to_torch(y_train, device=torch_device)
        y_test = to_torch(y_test, device=torch_device)
    elif backend == "jax":
        jax_device = "gpu" if JaxUtils.jax_has_gpu() else "cpu"
        print(f"jax device: {jax_device}")
        X_train = to_jax(X_train, device=jax_device)
        X_test = to_jax(X_test, device=jax_device)
        y_train = to_jax(y_train, device=jax_device)
        y_test = to_jax(y_test, device=jax_device)
    elif backend == "numpy":
        X_train = to_numpy(X_train)
        X_test = to_numpy(X_test)
        y_train = to_numpy(y_train)
        y_test = to_numpy(y_test)
    elif backend == "tensorflow":
        X_train = to_tensorflow(X_train, dtype="float32")
        X_test = to_tensorflow(X_test, dtype="float32")
        y_train = to_tensorflow(y_train, dtype="float32")
        y_test = to_tensorflow(y_test, dtype="float32")

    model = LeastSquaresRegression(
        X_train,
        y_train,
        X_test,
        y_test,
        use_multiple_initializations=use_multiple_initializations,
    )

    # optimization params
    max_iterations = 1000
    learning_rate = 1e-1

    # Run optimization
    optimizer_vanilla = GradientDescentVanillaOptimizer(
        name="least_squares_regression_vanilla_gd",
        objective_function=model,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
    )
    optimizer_momentum = GradientDescentMomentumOptimizer(
        name="least_squares_regression_momentum_gd",
        objective_function=model,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        momentum=0.9,
    )

    start_points = model.W_init if use_multiple_initializations else [model.W_init]
    runner_vanilla = OptimizationRunner(
        optimizer=optimizer_vanilla, start_points=start_points
    )
    runner_momentum = OptimizationRunner(
        optimizer=optimizer_momentum, start_points=start_points
    )

    results_vanilla = runner_vanilla.run()
    results_momentum = runner_momentum.run()

    runner_vanilla.plot_results()
    runner_momentum.plot_results()

    # Print optimization results
    print_optimization_summary(results_vanilla, "Vanilla Gradient Descent")
    print_optimization_summary(results_momentum, "Momentum Gradient Descent")

    print("Done")
    plt.show()


if __name__ == "__main__":
    main()
