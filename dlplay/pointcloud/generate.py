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


def generate_random_point_cloud(
    shape: str = "sphere", num_points: int = 1000, radius: float = 1.0
) -> np.ndarray:
    """
    Generate a point cloud with the specified shape.

    Args:
        shape: Shape type ("sphere", "cube", "random")
        num_points: Number of points to generate
        radius: Radius for sphere or size for cube

    Returns:
        numpy array of shape (num_points, 3) containing 3D coordinates
    """
    if shape == "sphere":
        # Generate points on a sphere using spherical coordinates
        phi = np.random.uniform(0, 2 * np.pi, num_points)
        costheta = np.random.uniform(-1, 1, num_points)
        u = np.random.uniform(0, 1, num_points)

        theta = np.arccos(costheta)
        r = radius * np.cbrt(u)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        return np.column_stack([x, y, z])

    elif shape == "cube":
        # Generate points inside a cube
        points = np.random.uniform(-radius, radius, (num_points, 3))
        return points

    elif shape == "random":
        # Generate random points in 3D space
        points = np.random.uniform(-radius, radius, (num_points, 3))
        return points

    else:
        raise ValueError(f"Unknown shape: {shape}")
