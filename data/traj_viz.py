"""Helper functions for Visualization of trajectories"""
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud


matplotlib.use( 'tkagg' )
fig, ax = plt.subplots()
ln, = plt.plot([], [], "b.", markersize=1)


def scene_sensor_data(nusc, scene_number=0, sensor="LIDAR_TOP", only_tokens=True):
    """return information of a specific scene
    only_tokens: if set to True return the sensor tokens"""
    sample = nusc.get("sample", nusc.scene[scene_number]["first_sample_token"])
    scene_data = []
    scene_data.append(sample["data"][sensor])
    while sample["next"] != "":
        sample = nusc.get("sample", sample["next"])
        scene_data.append(sample["data"][sensor])

    scene_data = list(dict.fromkeys(scene_data))
    if only_tokens:
        return scene_data
    else:
        sensor_data = []
        for idx in range(len(scene_data)):
            sensor_data.append(nusc.get("sample_data", scene_data[idx]))

        return sensor_data


def update(frames):
    data = _view_points(frames.points[:3,:], np.eye(4))
    xdata = data[0, :]
    ydata = data[1, :]
    ln.set_data(xdata, ydata)
    return ln,


def _view_points(points, view):
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    return points

def init():
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    return ln,


def render_scene_lidar(root, nusc, idx_scene, save_path=None):
    scene_data = scene_sensor_data(nusc, idx_scene, only_tokens=False)
    lidar = []

    for idx in range(len(scene_data)):
        lidar.append(LidarPointCloud.from_file(os.path.join(root, scene_data[idx]["filename"])))


    ani = FuncAnimation(fig, update, frames=lidar, init_func=init, blit=False)
    if save_path:
        ani.save(save_path + ".mp4")

    plt.show()