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
import multiprocessing
import time
from typing import Callable, List, Optional, Union, Any
import queue
import pickle
import inspect
import traceback

from dlplay.utils.types import TensorOrList


class VizProcess:
    """
    Generic visualization process that runs callbacks in separate processes.
    This allows for multiple independent visualization windows without blocking the main process.

    Usage:
        def my_callback(data_dict, shared_data):
            # Custom visualization logic
            # data_dict: any data_dict passed to the process
            # shared_data: shared dictionary for communication
            pass

        viz1 = VizProcess(my_callback, data1, "Window 1")
        viz2 = VizProcess(another_callback, data2, "Window 2")

        # Processes will run until explicitly closed
        viz1.join()  # This will block until process is closed
    """

    def __init__(
        self,
        callback: Callable,
        data_dict: dict | Any = None,
        process_name: str = "VizProcess",
        shared_data: Optional[dict] = None,
    ):
        """
        Initialize the visualization process.

        Args:
            callback: Callback function that receives (data_dict, shared_data) as arguments
            data_dict: Any data_dict to pass to the callback
            process_name: Name for the process
            shared_data: Optional shared dictionary for inter-process communication
        """
        self.callback = callback
        self.data_dict = data_dict
        self.process_name = process_name
        self.shared_data = shared_data or {}
        self.process = None
        self.is_running = False
        self._command_queue = multiprocessing.Queue()
        self._response_queue = multiprocessing.Queue()

        # automatially start the process
        self.start()

    def __del__(self):
        self.close()

    def is_alive(self):
        return self.process.is_alive() if self.process else False

    def _run_visualization(self, data_dict, shared_data, command_queue, response_queue):
        """Internal method to run the visualization in a separate process."""
        try:
            # Initialize shared data_dict in the process
            process_shared_data = shared_data.copy()
            process_shared_data["is_running"] = True

            # Call the user callback
            # check how many arguments the callback expects
            if len(inspect.signature(self.callback).parameters) == 1:
                try:
                    self.callback(data_dict)
                except Exception as e:
                    print(f"Error calling callback: {e}")
                    traceback.print_exc()
            elif len(inspect.signature(self.callback).parameters) == 2:
                try:
                    self.callback(data_dict, process_shared_data)
                except Exception as e:
                    print(f"Error calling callback: {e}")
                    traceback.print_exc()
            else:
                raise ValueError(
                    f"Callback {self.callback} expects {len(inspect.signature(self.callback).parameters)} arguments, but 1 or 2 are expected."
                )

            # Keep the process alive and handle commands [Concept]
            # while process_shared_data.get("is_running", True):
            #     try:
            #         # Check for commands from the main process
            #         command = command_queue.get(timeout=0.01)
            #         if command == "stop":
            #             process_shared_data["is_running"] = False
            #             response_queue.put("stopped")
            #             break
            #         elif command == "status":
            #             response_queue.put("running")
            #     except queue.Empty:
            #         pass

            #     time.sleep(0.01)  # Small delay to prevent excessive CPU usage

        except Exception as e:
            response_queue.put(f"error: {str(e)}")

    def start(self):
        """Start the visualization process in a non-blocking manner."""
        if self.process is None or not self.process.is_alive():
            self.process = multiprocessing.Process(
                target=self._run_visualization,
                args=(
                    self.data_dict,
                    self.shared_data,
                    self._command_queue,
                    self._response_queue,
                ),
                name=self.process_name,
            )
            self.process.start()
            self.is_running = True
            # Give the process a moment to start
            time.sleep(0.1)
        return self

    def send_command(self, command: str, timeout: float = 1.0) -> str:
        """Send a command to the visualization process and get a response."""
        if self.process and self.process.is_alive():
            self._command_queue.put(command)
            try:
                response = self._response_queue.get(timeout=timeout)
                return response
            except queue.Empty:
                return "timeout"
        return "not_running"

    def close(self):
        """Close the visualization process."""
        if self.process and self.process.is_alive():
            self.send_command("stop")
            self.process.join(timeout=2.0)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1.0)
        self.is_running = False

    def join(self):
        """Block until the visualization process is closed."""
        if self.process and self.process.is_alive():
            self.process.join()

    def is_open(self):
        """Check if the visualization process is still running."""
        return self.is_running and self.process and self.process.is_alive()

    @staticmethod
    def wait_all(
        processes: list["VizProcess"],
        timeout: float | None = None,
        poll_interval: float = 0.1,
    ) -> bool:
        """
        Wait for *all* VizProcess instances to exit, non-intrusively.

        Returns:
            True if all exited within timeout; False if any are still alive after timeout.
        """
        start = time.monotonic()
        while True:
            alive = [vp for vp in processes if vp.process and vp.process.is_alive()]
            if not alive:
                return True
            if timeout is not None and (time.monotonic() - start) >= timeout:
                return False
            # Nudge each with a short join; keeps signals responsive
            for vp in alive:
                vp.process.join(timeout=poll_interval)


if __name__ == "__main__":
    # NOTE: This is a simple example that generates two point clouds and visualizes them with open3d.

    # Test the VizProcess class with two point clouds
    print("Generating point clouds...")

    # Generate two different point clouds
    from dlplay.pointcloud.generate import generate_random_point_cloud

    sphere_points = generate_random_point_cloud("sphere", num_points=2000, radius=1.0)
    cube_points = generate_random_point_cloud("cube", num_points=1500, radius=1.5)

    print("Starting visualization processes...")

    # NOTE: This is a simple example that prints information about the point cloud and visualizes it with open3d.
    def visualize_point_cloud_callback(data_dict):
        """
        Callback function for visualizing point clouds.
        This is a simple example that prints information about the point cloud.
        In a real implementation, this would use a visualization library like Open3D, Matplotlib, etc.
        """
        point_cloud = data_dict.get("data", None)
        shape_name = data_dict.get("title", None)

        print(f"Visualizing {shape_name} point cloud with {len(point_cloud)} points")
        print(f"Point cloud shape: {point_cloud.shape}")
        print(
            f"Bounding box: min={point_cloud.min(axis=0)}, max={point_cloud.max(axis=0)}"
        )
        print(f"Mean position: {point_cloud.mean(axis=0)}")

        # visualize the point cloud with open3d
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        o3d.visualization.draw_geometries([pcd], window_name=shape_name)

    # Create two separate visualization processes
    viz1 = VizProcess(
        callback=visualize_point_cloud_callback,
        data_dict={"data": sphere_points, "title": "Sphere"},
        process_name="SphereViz",
    )

    viz2 = VizProcess(
        callback=visualize_point_cloud_callback,
        data_dict={"data": cube_points, "title": "Cube"},
        process_name="CubeViz",
    )
