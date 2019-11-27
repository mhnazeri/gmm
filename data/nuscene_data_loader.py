"""This module is responsible for loading of NuScene dataset
TODO:
1) read sensor data
2) vectorized data
3) each scene is a data sample
"""
# import required packages
import os
import math
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes


logger = logging.getLogger(__name__)


def load_dataset(_path, version='v1.0-mini', verbose=False):
    """Load Nuscene dataset"""
    nusc = NuScenes(version=version, dataroot=_path, verbose=verbose)
    logger.debug(str(nusc.list_scenes), exc_info=1)
    return nusc


def scene_info(nusc, scene_number=0, sensor="LIDAR_TOP"):
    """return information of a scene"""
    sample = nusc.get("sample", nusc.scene[scene_number]["first_sample_token"])
    scene_data = []
    scene_data.append(sample["data"][sensor])
    while sample["next"] != "":
        sample = nusc.get("sample", sample["next"])
        scene_data.append(sample["data"][sensor])

    return scene_data


if __name__ == "__main__":
    root = "nuScene-mini"
    nusc = load_dataset(root, verbose=False)
    # print(nusc.list_sample(nusc.scene[0]["first_sample_token"]))
    scene_data = scene_info(nusc, 1)
    # print(nusc.get("sample_annotations", scene_data[0])
    lidar = nusc.get("sample_data", scene_data[0])
    nusc.render_scene_channel(nusc.scene[1]["token"])
    # with open(root + "/" + lidar['filename'], 'rb') as f:
        # print(f.read())
    # nusc.render_sample_data(nusc.get("sample_data", scene_data[1]["LIDAR_TOP"]))
    # nusc.render_scene_channel(nusc.scene[0]["token"])
