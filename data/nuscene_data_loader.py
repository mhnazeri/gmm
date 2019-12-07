"""This module is responsible for loading of NuScene dataset
TODO:
1) read sensor data
2) vectorized data
3) each scene is a data sample
"""
# import required packages
import os
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion


logger = logging.getLogger(__name__)
matplotlib.use( 'tkagg' )


def load_dataset(_path, version='v1.0-mini', verbose=False):
    """Load Nuscene dataset"""
    nusc = NuScenes(version=version, dataroot=_path, verbose=verbose)
    logger.debug(str(nusc.list_scenes), exc_info=1)
    return nusc


def scene_sensor_data(nusc, scene_number=0, sensor="LIDAR_TOP", verbose=True):
    """return information of a specific scene
    verbose: if set to True return the sensor tokens"""
    sample = nusc.get("sample", nusc.scene[scene_number]["first_sample_token"])
    scene_data = []
    scene_data.append(sample["data"][sensor])
    while sample["next"] != "":
        sample = nusc.get("sample", sample["next"])
        scene_data.append(sample["data"][sensor])

    scene_data = list(dict.fromkeys(scene_data))
    if verbose:
        return scene_data
    else:
        sensor_data = []
        for idx in range(len(scene_data)):
            sensor_data.append(nusc.get("sample_data", scene_data[idx]))

        return sensor_data

def sample_token_extractor(nusc, idx_scene):
    """returns t sample frame from scene idx"""
    sample = nusc.get("sample", nusc.scene[idx_scene]["first_sample_token"])
    _frames = []
    while sample["next"] != "":
            _frames.append(nusc.get("sample", sample["token"]))
            sample = nusc.get("sample", sample["next"])

    # Get last frame
    _frames.append(nusc.get("sample", sample["token"]))

    return _frames


def update(frames):
    data = view_points(frames.points[:3,:], np.eye(4))
    xdata = data[0, :]
    ydata = data[1, :]
    ln.set_data(xdata, ydata)
    return ln,

def view_points(points, view):
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

def render_scene_lidar(idx_scene, fig, ax, ln, save_path=None):
    scene_data = scene_sensor_data(nusc, idx_scene, verbose=False)
    lidar = []

    for idx in range(len(scene_data)):
        lidar.append(LidarPointCloud.from_file(os.path.join(root, scene_data[idx]["filename"])))


    ani = FuncAnimation(fig, update, frames=lidar, init_func=init, blit=False)
    if save_path:
        ani.save(save_path + ".mp4")

    plt.show()


if __name__ == "__main__":
    root = "nuScene-mini"
    nusc = load_dataset(root, verbose=False)
    scene_data = scene_sensor_data(nusc, 1, verbose=False)
    samples = sample_token_extractor(nusc, 0)
    print(len(samples))
    print(nusc.get("sample_annotation", samples[0]["anns"][1]))
    # fig, ax = plt.subplots()
    # ln, = plt.plot([], [], "b.", markersize=1)
    # render_scene_lidar(0, fig, ax, ln, save_path="scene")
    # print(nusc.get("sample", nusc.scene[0]["first_sample_token"]))
    # nusc.render_annotation([nusc.get("sample_annotation", token) for token in nusc.get("sample", nusc.scene[0]["first_sample_token"])["anns"]][0]["token"])
    plt.show()


    # print(len(obs_frames), len(pred_frames), i)
    # print(obs_frames[0][0])
    # print(nusc.sample_annotation[0])
    # nusc.render_annotation(obs_frames[0][0]["anns"][0])
    # print([len(x) for x in obs_frames])
    # print([len(x) for x in pred_frames])
    # print(len(scene_data))
    # nusc.render_pointcloud_in_image(scene_data[0]["sample_token"])
    # nusc.render_scene(nusc.scene[1]["token"])
    # nusc.render_pointcloud_in_image(scene_data[3]["sample_token"])
    # nusc.render_pointcloud_in_image(scene_data[7]["sample_token"])
    # nusc.render_pointcloud_in_image(scene_data[11]["sample_token"])
    # lidar.render_height(ax=ax[0])
    # lidar.render_intensity(ax=ax[1])
    # print(len(scene_data))
    # print(lidar.points.shape)