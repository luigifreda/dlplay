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
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import numpy as np
import torch
import jax.numpy as jnp
import torchvision.transforms.functional as F

from typing import List, Callable, Tuple
from dlplay.utils.types import OptimizationPathData, TensorOrList


# NOTE: This is a global setting that affects all matplotlib figures.
# It ensures that the saved figure's bounding box is adjusted to tightly fit around the actual content without excessive whitespace around the edges.
plt.rcParams["savefig.bbox"] = "tight"


def show_images(
    imgs: TensorOrList,
    title: str = "",
    subtitles: List[str] = [],
    plt_show: bool = True,
) -> None:
    """
    Display one or multiple torch uint8 image tensors using matplotlib.

    Accepts a single CHW uint8 tensor or a list of them. Non-uint8 tensors
    are detached and converted via torchvision F.to_pil_image.
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    if title:
        fig.suptitle(title)
    for i, img in enumerate(imgs):
        if isinstance(img, torch.Tensor):
            img = img.detach()
            img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if subtitles:
            axs[0, i].set_title(subtitles[i])
    if plt_show:
        plt.show()
    return fig, axs


def plot_grad_descent_paths(
    f: Callable,
    data: List[OptimizationPathData],
    grid_points: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
    title: str = "Descent Paths",
    framework: str = "numpy",
    plt_show: bool = False,
) -> None:
    """
    Plot descent paths and gradient norms.

    Args:
        f: Function to evaluate for contour plot
        data: List of (path, grad_norms) tuples
        title: Plot title
        framework: Framework being used ('numpy', 'pytorch', 'jax')
        grid_points: Tuple of (xx, yy, zz) where xx, yy are meshgrid coordinates
        and zz is the function values (to be computed by framework-specific code)
    """

    # Setup colors
    colors = cm.tab10(np.linspace(0, 1, len(data)))

    # Let's configure how many subplots we need
    # If we have metrics, we need 2 subplots, otherwise 1
    num_subplots_rows = 2 if len(data[0][3]) > 0 else 1
    # If we have grid points, we need 3 subplots if the upper part otherwise 2
    num_subplots_upper = 3 if grid_points is not None else 2
    # The number of subplots in the lower part is equal to the number of available metrics
    num_subplots_lower = 0 if num_subplots_rows == 1 else len(data[0][3].keys())

    # Let's create the figure and the subplots
    fig = plt.figure(figsize=(20, 6))
    fig.suptitle(title)
    # print(f"plot_grad_descent_paths: num_subplots_rows: {num_subplots_rows}")
    # print(f"plot_grad_descent_paths: num_subplots_upper: {num_subplots_upper}")
    # print(f"plot_grad_descent_paths: num_subplots_lower: {num_subplots_lower}")
    height_ratios = [1.0] * num_subplots_rows if num_subplots_rows > 1 else [1.0]
    gs = gridspec.GridSpec(num_subplots_rows, 1, height_ratios=height_ratios)
    gs_top = gridspec.GridSpecFromSubplotSpec(1, num_subplots_upper, subplot_spec=gs[0])
    if num_subplots_lower > 0:
        gs_bottom = gridspec.GridSpecFromSubplotSpec(
            1, num_subplots_lower, subplot_spec=gs[1]
        )

    if grid_points is not None:
        ax1 = fig.add_subplot(gs_top[0])
        ax2 = fig.add_subplot(gs_top[1])
        ax3 = fig.add_subplot(gs_top[2])
    else:
        ax1 = None
        ax2 = fig.add_subplot(gs_top[0])
        ax3 = fig.add_subplot(gs_top[1])

    # Compute function values for contour plot for 2D functions
    if grid_points is not None:
        xx, yy, _ = grid_points

        # Compute function values for contour plot
        if framework == "numpy":
            grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
            zz = f(grid_points).reshape(xx.shape)
        elif framework == "torch":
            grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
            # Use float32 for MPS devices, float64 for others
            if hasattr(f, 'device') and hasattr(f.device, 'type') and f.device.type == 'mps':
                dtype = torch.float32
            else:
                dtype = torch.float64
            grid_points = torch.tensor(grid_points, dtype=dtype)
            zz = f(grid_points).cpu().numpy().reshape(xx.shape)
        elif framework == "jax":
            grid_points = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
            zz = np.array(f(grid_points)).reshape(xx.shape)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

        # Plot contour + trajectories with arrows
        contour = ax1.contourf(xx, yy, zz, levels=50, cmap="viridis")

    axs_lower = []

    for i, (path, objective_function_values, norms, metrics) in enumerate(data):
        num_steps = len(path)

        if grid_points is not None:
            ax1.plot(path[:, 0], path[:, 1], marker="o", markersize=3, color=colors[i])
            dx = path[1:, 0] - path[:-1, 0]
            dy = path[1:, 1] - path[:-1, 1]
            ax1.quiver(
                path[:-1, 0],
                path[:-1, 1],
                dx,
                dy,
                scale_units="xy",
                angles="xy",
                scale=1,
                color=colors[i],
                width=0.004,
                headwidth=4,
            )

        # Plot objective function values
        if objective_function_values is not None:
            num_objective_function_values = len(objective_function_values)
            ax2.plot(
                range(num_objective_function_values),
                objective_function_values,
                color=colors[i],
            )

        # Plot gradient norms
        if norms is not None:
            num_norms = len(norms)
            ax3.plot(range(num_norms), norms, color=colors[i])

        # Plot metrics if they exist
        if metrics:
            num_keys = len(metrics.keys())
            for j, (metric_name, metric_values) in enumerate(metrics.items()):
                # print(f"metric_name: {metric_name}, metric_values: {metric_values}")
                axs_set_pros = False
                if len(axs_lower) <= j:
                    ax_lower_j = fig.add_subplot(gs_bottom[j])
                    axs_lower.append(ax_lower_j)
                    axs_set_pros = True
                else:
                    ax_lower_j = axs_lower[j]
                num_metric_values = len(metric_values)
                ax_lower_j.plot(
                    range(num_metric_values), metric_values, color=colors[i]
                )
                if axs_set_pros:
                    ax_lower_j.set_title(f"{metric_name}")
                    ax_lower_j.set_xlabel("Step")
                    ax_lower_j.set_ylabel(metric_name)
                    ax_lower_j.grid(True)

    if ax1 is not None:
        ax1.set_title(title)
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        fig.colorbar(contour, ax=ax1, label="f(x)")

    ax2.set_title("Objective Function Values over Time")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Objective Function Value")
    ax2.grid(True)

    ax3.set_title("Gradient Norms over Time")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Gradient Norm")
    ax3.grid(True)

    plt.tight_layout()
    if plt_show:
        plt.show()
