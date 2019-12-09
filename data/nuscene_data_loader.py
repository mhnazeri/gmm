"""This module is responsible for loading of NuScene dataset
TODO:
1) read sensor data
2) vectorized data
3) each scene is a data sample
"""
import os
import logging
import json
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from traj_viz import render_scene_lidar



logger = logging.getLogger(__name__)


def load_dataset(_path, version='v1.0-mini', verbose=False):
    """Load Nuscene dataset"""
    nusc = NuScenes(version=version, dataroot=_path, verbose=verbose)
    logger.debug(str(nusc.list_scenes), exc_info=1)
    return nusc


def scene_sensor_data(nusc, scene_number=0, sensor="LIDAR_TOP", only_tokens=False):
    """return information of a specific scene
    verbose: if set to True return the sensor tokens"""
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


def sample_token_extractor(nusc, idx_scene):
    """returns sample frames from scene idx"""
    sample = nusc.get("sample", nusc.scene[idx_scene]["first_sample_token"])
    _frames = []
    while sample["next"] != "":
            _frames.append(nusc.get("sample", sample["token"]))
            sample = nusc.get("sample", sample["next"])

    # Get last frame
    _frames.append(nusc.get("sample", sample["token"]))

    return _frames


def extract_scene_data_as_json(nusc, scene_idx, path=None):
    """Write scene data as json file in to given path"""
    name = nusc.scene[scene_idx]["name"]
    scene_data = scene_sensor_data(nusc, scene_idx, only_tokens=False)
    print(len(scene_data))
    info_list = []
    for idx, sample in enumerate(scene_data):
        annotations = nusc.get("sample", sample["sample_token"])["anns"]
        info = {"id_sample": idx,
                "token":sample["token"],
                "ego_pose_translation": nusc.get("ego_pose", sample["ego_pose_token"])["translation"],
                "ego_pose_rotation": nusc.get("ego_pose", sample["ego_pose_token"])["rotation"],
                "timestamp": sample["timestamp"],
                "filename": sample["filename"],
                "channel": sample["channel"],
                "annotations": [{"token": nusc.get("sample_annotation", ann)["token"],
                                "sample_token": nusc.get("sample_annotation", ann)["sample_token"],
                                "category": nusc.get("sample_annotation", ann)["category_name"],
                                "translation": nusc.get("sample_annotation", ann)["translation"],
                                "rotation": nusc.get("sample_annotation", ann)["rotation"],
                                "size": nusc.get("sample_annotation", ann)["size"],} for ann in annotations]}

        info_list.append(info)

    if os.path.exists(path):
        with open(os.path.join(path, name) + ".json", "w") as f:
            f.write(json.dumps(info_list))
    else:
        os.mkdir(path)
        with open(os.path.join(path, name) + ".json", "w") as f:
            f.write(json.dumps(info_list))
    # print(nusc.field2token("scene", "first_sample_token", sample[0]["sample_token"]))
    # print(len(sample))


if __name__ == "__main__":
    root = "nuScene-mini"
    nusc = load_dataset(root, verbose=False)
    # scene_data = scene_sensor_data(nusc, 1, only_tokens=False)
    # print(nusc.get("calibrated_sensor", scene_data[1]["calibrated_sensor_token"]))
    # sample = nusc.get("sample", scene_data[1]["sample_token"])
    # l = nusc.get("sample_annotation", sample["anns"][25])
    # print(l)
    # print(scene_data[1])
    # samples = sample_token_extractor(nusc, 1)
    for idx in range(len(nusc.scene)):
        extract_scene_data_as_json(nusc, idx, "exported_json_data")
    # scene_info = nusc.scene[1]
    # print(nusc.get("sample_data", samples[0]["data"]["LIDAR_TOP"]))
    # print(nusc.get("sample_data", scene_data[0]))
    # print(scene_data[0])
    # ann = nusc.get("sample_annotation", samples[0]["anns"][10])
    # pos = []
    # sample_token = []
    # ann_tokens = []
    # while ann["next"] != "":
    #     pos.append(ann["translation"])
    #     ann_tokens.append(ann["token"])
    #     sample_token.append(nusc.get("sample", ann["sample_token"]))
    #     ann = nusc.get("sample_annotation", ann["next"])




    # pos = list(zip(pos[:-1], pos[1:]))
    # pos = np.array(pos, dtype=np.float64)
    # print(pos)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # plt.scatter(pos[:,0], pos[:, 1], pos[: 2])
    # for token in ann_tokens:
        # nusc.render_annotation(token)
    # print(render)
    # plt.scatter(pos[:,0,0], pos[:,1,2])
    # plt.show()
