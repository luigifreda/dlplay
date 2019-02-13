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
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import torch
from typing import Tuple


def mesh_to_wireframe(mesh, color=[0.9, 0.9, 0.9], colors=None, debug=False):
    """
    Convert a mesh to a wireframe.
    """
    # Ensure the mesh has triangles
    if len(mesh.triangles) == 0:
        print("Warning: Mesh has no triangles, cannot create wireframe")
        return o3d.geometry.LineSet()

    lineset = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)

    # Check if lineset is empty
    if len(lineset.lines) == 0:
        if debug:
            print("Warning: LineSet is empty")
        return lineset

    # Use a more visible color and ensure it's applied correctly
    if colors is None:
        # Use a bright, visible color
        lineset.paint_uniform_color([1.0, 0.0, 0.0])  # Red color for better visibility
    else:
        # For now, just use a single color for all lines when colors are provided
        lineset.paint_uniform_color([1.0, 0.0, 0.0])  # Red color for better visibility

    return lineset


def viz_point_cloud(data_dict: dict):
    """
    Visualize a point cloud using Open3D.
    """
    data = data_dict.get("data", None)
    title = data_dict.get("title", "Point Cloud")
    colors = data_dict.get("colors", None)
    color = data_dict.get("color", None)  # single color for all points

    if data is None:
        raise ValueError("data is required")

    point_cloud = data
    pcd = o3d.geometry.PointCloud()

    # Handle both tensor and numpy array cases
    if hasattr(point_cloud, "pos"):
        # PyTorch Geometric Data object
        if isinstance(point_cloud.pos, torch.Tensor):
            points = point_cloud.pos.detach().cpu().numpy()
        else:
            points = point_cloud.pos
    else:
        # Direct tensor or numpy array
        if isinstance(point_cloud, torch.Tensor):
            points = point_cloud.detach().cpu().numpy()
        else:
            points = point_cloud

    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        if isinstance(colors, torch.Tensor):
            colors = colors.detach().cpu().numpy()

        # Debug: print colors info
        print(f"Colors shape: {colors.shape}, dtype: {colors.dtype}")
        print(f"Colors range: [{colors.min():.3f}, {colors.max():.3f}]")

        # Ensure colors are float64 for Open3D compatibility
        colors = colors.astype(np.float64)

        # Ensure colors are in valid range [0, 1]
        colors = np.clip(colors, 0.0, 1.0)

        # Ensure correct shape (N, 3)
        if colors.shape[1] != 3:
            raise ValueError(f"Colors must have shape (N, 3), got {colors.shape}")

        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        if color is not None:
            pcd.paint_uniform_color(color)
    o3d.visualization.draw_geometries([pcd], window_name=title)


def viz_mesh(data_dict: dict):
    """
    Visualize a mesh using Open3D.
    """
    data = data_dict.get("data", None)
    title = data_dict.get("title", "Mesh")
    wired = data_dict.get("wired", False)
    colors = data_dict.get("colors", None)
    color = data_dict.get("color", None)  # single color for all faces
    shaded = data_dict.get("shaded", False)
    debug = data_dict.get("debug", False)

    if data is None:
        raise ValueError("data is required")

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(data.pos.numpy())

    # Debug: print mesh info
    print(f"Mesh vertices: {len(mesh.vertices)}")

    if colors is not None:
        mesh.colors = o3d.utility.Vector3dVector(colors)
    else:
        if color is not None:
            mesh.paint_uniform_color(color)

    # Convert to int32, transpose to [num_faces, 3], and ensure contiguous array
    face_array = data.face.numpy().T.astype(np.int32)
    mesh.triangles = o3d.utility.Vector3iVector(face_array)

    # Debug: print triangle info
    print(f"Mesh triangles: {len(mesh.triangles)}")

    if wired:
        line_set = mesh_to_wireframe(
            mesh, color=[0.9, 0.9, 0.9], colors=colors, debug=debug
        )
        if debug:
            print(f"LineSet lines: {len(line_set.lines)}")
        if len(line_set.lines) == 0:
            print("Warning: LineSet is empty, falling back to solid mesh")
            o3d.visualization.draw_geometries([mesh], window_name=title)
        else:
            o3d.visualization.draw_geometries([line_set], window_name=title)
    else:
        if shaded:
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
        o3d.visualization.draw_geometries([mesh], window_name=title)


def viz_graph_with_edges(data_dict: dict):
    """
    Visualize a PyTorch Geometric graph with both points and edges using Open3D.

    Args:
        data: PyTorch Geometric Data object with pos and edge_index attributes
        title: Title for the visualization window
    """
    data = data_dict.get("data", None)
    title = data_dict.get("title", "Graph with Edges")
    colors = data_dict.get("colors", None)
    color = data_dict.get("color", None)  # single color for all points
    color_edges = data_dict.get(
        "color_edges", [0.0, 0.0, 1.0]
    )  # blue color for all edges
    colors_edges = data_dict.get("colors_edges", None)
    debug = data_dict.get("debug", False)

    if data is None:
        raise ValueError("data is required")

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data.pos.numpy())
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        if color is not None:
            pcd.paint_uniform_color(color)

    # Create line set for edges
    if hasattr(data, "edge_index") and data.edge_index.size(1) > 0:
        # Convert edge_index to line set
        edges = data.edge_index.T.numpy()  # Shape: [num_edges, 2]
        line_points = data.pos.numpy()  # Shape: [num_nodes, 3]

        # Create line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector(edges)

        if colors_edges is not None:
            line_set.colors = o3d.utility.Vector3dVector(colors_edges)
        else:
            # Color the edges (you can modify this)
            line_set.colors = o3d.utility.Vector3dVector(
                np.tile(color_edges, (edges.shape[0], 1))  # Blue color for edges
            )

        # Visualize both points and edges
        o3d.visualization.draw_geometries([pcd, line_set], window_name=title)
    else:
        # If no edges, just show points
        o3d.visualization.draw_geometries([pcd], window_name=title)


def create_lines_for_normals(pcd, length=0.05, color=[0, 0, 0]):
    """
    Create a LineSet to show the normals of a PointCloud.
    Args:
        pcd: PointCloud
        length: Length of the normal lines
        color: Color of the normal lines
    Returns:
        LineSet
    """
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    if len(points) == 0 or normals is None or len(normals) == 0:
        print("[WARNING] No point or normal found to create the normal lines.")
        return None

    norm_normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    start_points = points
    end_points = points + norm_normals * length

    if np.any(np.isnan(end_points)) or np.any(np.isinf(end_points)):
        print(
            "[WARNING] NaN or Inf found in the end points of the normals. Skipping the normal lines."
        )
        return None

    line_points = np.vstack((start_points, end_points))
    lines = [[i, i + len(points)] for i in range(len(points))]

    colors = np.tile(np.array(color).astype(np.float32), (len(lines), 1))

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def viz_geometries(data_dict: dict):
    """
    Visualize a list of geometries using Open3D.
    Args:
        data_dict: Dictionary containing the data
    Returns:
        None
    """

    window_name = data_dict.get("window_name", "Open3D")  # Name of the window
    list_geometry_colors = data_dict.get(
        "list_geometry_colors", []
    )  # List of geometries and colors
    list_labels = data_dict.get("list_labels", [])  # List of labels
    show_normals = data_dict.get("show_normals", False)  # Show normals
    show_coordinate_frame = data_dict.get(
        "show_coordinate_frame", True
    )  # Show coordinate frame
    axis_size = data_dict.get("axis_size", 0.2)  # Size of the axis
    no_shadow = data_dict.get("no_shadow", False)  # No shadow

    gui.Application.instance.initialize()

    window = gui.Application.instance.create_window(window_name, 1024, 768)
    scene_widget = gui.SceneWidget()
    window.add_child(scene_widget)
    scene_widget.scene = rendering.Open3DScene(window.renderer)

    # Add press 'q' event to close the window
    def _on_key(event: gui.KeyEvent):
        if event.type == gui.KeyEvent.Type.DOWN and (event.key == gui.KeyName.Q):
            # Close just this window; when the last window closes, app.run() returns.
            window.close()
            return True  # event handled
        return False

    # Attach to both the window and the SceneWidget (some keys go to the focused widget)
    window.set_on_key(_on_key)
    try:
        # Available on recent Open3D versions
        scene_widget.set_on_key(_on_key)
    except Exception as e:
        print(f"Warning: Exception {e}")

    scene_widget.scene.set_background([1.0, 1.0, 1.0, 1.0])

    if no_shadow:
        scene_widget.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.NO_SHADOWS, (0.5, 0.5, 0.5)
        )
    else:
        scene_widget.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, (1.0, 0.0, 0.0)
        )

    overall_bbox = o3d.geometry.AxisAlignedBoundingBox()
    geometry_added_flag = False

    pcd_origins = [
        geom.get_min_bound() for geom, _ in list_geometry_colors if geom is not None
    ]

    for geom, colors in list_geometry_colors:
        material = rendering.MaterialRecord()
        name = f"geometry_{id(geom)}"

        if isinstance(geom, o3d.geometry.PointCloud):
            if show_normals:
                if not geom.has_normals():
                    geom.estimate_normals()

            material.shader = "defaultUnlit"
            material.point_size = 2.0

            if colors is not None:
                # check if colors is a single color or a list of colors
                if isinstance(colors, list) and len(colors) == 3:
                    geom.paint_uniform_color(colors)
                else:
                    geom.colors = o3d.utility.Vector3dVector(colors)

            scene_widget.scene.add_geometry(name, geom, material)
            geometry_added_flag = True
            overall_bbox += geom.get_axis_aligned_bounding_box()

            if show_normals:
                normal_lines = create_lines_for_normals(geom)
                if normal_lines:
                    normal_material = rendering.MaterialRecord()
                    normal_material.shader = "unlitLine"
                    scene_widget.scene.add_geometry(
                        f"normals_{id(geom)}", normal_lines, normal_material
                    )

        elif isinstance(
            geom,
            (o3d.geometry.OrientedBoundingBox, o3d.geometry.AxisAlignedBoundingBox),
        ):
            material.shader = "unlitLine"
            if colors is not None:
                if isinstance(colors, list) and len(colors) == 3:
                    geom.color = colors  # Use .color instead of .paint_uniform_color()
                else:
                    geom.colors = o3d.utility.Vector3dVector(colors)
            scene_widget.scene.add_geometry(name, geom, material)
            geometry_added_flag = True
            overall_bbox += geom.get_axis_aligned_bounding_box()

        elif isinstance(geom, o3d.geometry.TriangleMesh):
            material.shader = "defaultLit"
            if colors is not None:
                if isinstance(colors, list) and len(colors) == 3:
                    material.base_color = [*colors, 1.0]
                    geom.paint_uniform_color(colors)
                else:
                    geom.vertex_colors = o3d.utility.Vector3dVector(colors)
            scene_widget.scene.add_geometry(name, geom, material)
            geometry_added_flag = True
            overall_bbox += geom.get_axis_aligned_bounding_box()
        else:
            print(f"[WARNING] Geometry type not supported: {type(geom)}. Skipping.")

    for label_text, label_pos in list_labels:
        scene_widget.add_3d_label(np.asarray(label_pos, dtype=np.float32), label_text)

    if show_coordinate_frame:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
        if len(pcd_origins) > 0:
            # Move it to center of point cloud
            axis.translate(pcd_origins[0])
            # Create material record for axis
            axis_material = rendering.MaterialRecord()
            axis_material.shader = "defaultUnlit"
            # White fallback
            axis_material.base_color = [1.0, 1.0, 1.0, 1.0]
        else:
            if geometry_added_flag and overall_bbox.volume() > 0:
                # Move it to center of bounding box
                axis.translate(overall_bbox.get_center())
            axis_material = rendering.MaterialRecord()
            axis_material.shader = "defaultLit"
            # White fallback
            axis_material.base_color = [1.0, 1.0, 1.0, 1.0]
        scene_widget.scene.add_geometry("coordinate_frame", axis, axis_material)

    if geometry_added_flag and overall_bbox.volume() > 0:
        # Setup camera to view the bounding box
        scene_widget.setup_camera(60.0, overall_bbox, overall_bbox.get_center())
    else:
        print("[INFO] No geometry added or bounding box empty. Setting default view.")
        scene_widget.setup_camera(
            60.0,
            o3d.geometry.AxisAlignedBoundingBox([-1, -1, -1], [1, 1, 1]),
            np.array([0, 0, 0]),
        )

    gui.Application.instance.run()


if __name__ == "__main__":
    # Test the viz_geometries function

    # generate random point cloud and colors
    pcd = o3d.geometry.PointCloud()
    num_points = 1000
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(num_points, 3))
    pcd.colors = o3d.utility.Vector3dVector(np.random.rand(num_points, 3))

    # generate random triangle mesh and colors
    mesh = o3d.geometry.TriangleMesh()
    num_triangles = 100
    mesh_vertices = np.random.rand(num_triangles, 3) + [-1.0, -1.0, -1.0]
    mesh_triangles = np.random.randint(0, num_triangles, (num_triangles, 3))
    mesh_vertex_colors = np.random.rand(num_triangles, 3)
    mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(mesh_triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_vertex_colors)

    # generate random axis aligned bounding box and colors
    bbox = o3d.geometry.AxisAlignedBoundingBox()
    bbox.min_bound = np.array([-1, -1, -1])
    bbox.max_bound = np.array([1, 1, 1])
    bbox.color = np.array([0.0, 0.0, 1.0])

    # compute centroids of the point cloud, triangle mesh, and axis aligned bounding box
    pcd_centroid = np.mean(pcd.points, axis=0)
    mesh_centroid = np.mean(mesh.vertices, axis=0)
    bbox_centroid = (bbox.get_min_bound() + bbox.get_max_bound()) / 2.0

    list_geometry_colors = [
        (pcd, None),
        (mesh, None),
        (bbox, None),
    ]
    list_labels = [
        ("Point Cloud", pcd_centroid),
        ("Triangle Mesh", mesh_centroid),
        ("Axis Aligned Bounding Box", bbox_centroid),
    ]
    data_dict = {
        "window_name": "Open3D",
        "list_geometry_colors": list_geometry_colors,
        "list_labels": list_labels,
        "show_normals": False,
        "show_coordinate_frame": True,
        "axis_size": 0.2,
        "no_shadow": False,
    }
    viz_geometries(data_dict)
