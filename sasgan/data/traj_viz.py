"""Helper functions for Visualization of trajectories"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from nuscenes.utils.geometry_utils import (
    view_points,
    box_in_image,
    BoxVisibility,
    transform_matrix,
)
from nuscenes.nuscenes import NuScenes, view_points
from nuscenes.utils.data_classes import LidarPointCloud


# Defining Global variables
matplotlib.use("tkagg")
fig, axes = plt.subplots(figsize=(18, 9))
view = np.eye(4)
(ln,) = plt.plot([], [], "b.", markersize=1)
nusc = NuScenes(version="v1.0-mini", dataroot="nuScene-mini")


def sample_extractor(nusc, idx_scene):
    """returns sample frames from scene idx"""
    sample = nusc.get("sample", nusc.scene[idx_scene]["first_sample_token"])
    _frames = []
    while sample["next"] != "":
        _frames.append(nusc.get("sample", sample["token"]))
        sample = nusc.get("sample", sample["next"])

    # Get last frame
    _frames.append(nusc.get("sample", sample["token"]))

    return _frames


def update(frames):
    global data_path
    global boxes
    global corners
    axes.clear()

    for ann in frames["anns"]:
        data_path, boxes, _ = nusc.get_sample_data(
            frames["data"]["LIDAR_TOP"], selected_anntokens=[ann]
        )

        for box in boxes:
            c = np.array(get_color(box.name)) / 255.0
            box.render(axes, view=view, colors=(c, c, c))
            corners = _view_points(boxes[0].corners(), view)[:2, :]
            axes.set_xlim([np.min(corners[0, :]) - 10, np.max(corners[0, :]) + 10])
            axes.set_ylim([np.min(corners[1, :]) - 10, np.max(corners[1, :]) + 10])
            axes.axis("off")
            axes.set_aspect("equal")

    global frame
    frame = LidarPointCloud.from_file(data_path)
    frame.render_height(axes, view=np.eye(4))

    global data
    data = _view_points(frame.points[:3, :], np.eye(4))

    xdata = data[0, :]
    ydata = data[1, :]
    ln.set_data(xdata, ydata)
    return (ln,)


def _view_points(points, view):
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[: view.shape[0], : view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    return points


def init():
    axes.set_xlim(-20, 20)
    axes.set_ylim(-20, 20)
    ln.set_data([], [])
    return (ln,)


def get_color(category_name: str):
    """
        Provides the default colors based on the category names.
        This method works for the general nuScenes categories, as well as the nuScenes detection categories.
    """
    if "bicycle" in category_name or "motorcycle" in category_name:
        return 255, 61, 99  # Red
    elif "vehicle" in category_name or category_name in [
        "bus",
        "car",
        "construction_vehicle",
        "trailer",
        "truck",
    ]:
        return 255, 158, 0  # Orange
    elif "pedestrian" in category_name:
        return 0, 0, 230  # Blue
    elif "cone" in category_name or "barrier" in category_name:
        return 0, 0, 0  # Black
    else:
        return 255, 0, 255  # Magenta


# def _sample_annotations(nusc, instance):
#     sample_annotation = nusc.get("sample_annotation", instance)
#     annotation = []
#     while sample_annotation["next"] != "":
#         annotation.append(sample_annotation["token"])
#         sample_annotation = nusc.get("sample_annotation", sample_annotation["next"])

#     return annotation


def render_scene_lidar(root, nusc, idx_scene, save_path=None, blit=False):
    scene_data = sample_extractor(nusc, idx_scene)
    lidar = []
    for scene in scene_data:
        lidar.append(scene)

    ani = FuncAnimation(fig, update, frames=lidar, init_func=init, blit=blit)

    if save_path:
        ani.save(save_path + ".mp4")
    # commented for computational reasons
    # plt.show()
