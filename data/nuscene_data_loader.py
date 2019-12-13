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
from functools import lru_cache
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from traj_viz import render_scene_lidar



logger = logging.getLogger(__name__)


def load_dataset(_path, version='v1.0-mini', verbose=False):
    """Load Nuscene dataset"""
    nusc = NuScenes(version=version, dataroot=_path, verbose=verbose)
    logger.debug(str(nusc.list_scenes), exc_info=1)
    return nusc


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


@lru_cache(64)
def _sample_annotations(nusc, instance):
    sample_annotation = nusc.get("sample_annotation", instance)
    annotation = []
    while sample_annotation["next"] != "":
        movable = 0 if sample_annotation["category_name"] in \
         "barrier debris pushable_pul trafficcone bicycle_rack" else 1
        annotation.append({"translation": sample_annotation["translation"],
                        "rotation": sample_annotation["rotation"],
                        "size": sample_annotation["size"],
                        "visibility": sample_annotation["visibility_token"],
                        "category": sample_annotation["category_name"],
                        "instance_token": sample_annotation["instance_token"],
                        "sample_token": sample_annotation["sample_token"],
                        "timestamp": nusc.get("sample", sample_annotation["sample_token"])["timestamp"],
                        "movable": movable,
                        "velocity": nusc.box_velocity(sample_annotation["token"]).tolist()})

        sample_annotation = nusc.get("sample_annotation", sample_annotation["next"])

    return annotation


def extract_scene_data_as_json(nusc, scene_idx, path=None):
    """
    Write scene data as json file in to given path
    args:
        Nuscene nusc: Nuscene dataset
        int scene_idx: index of the desired scene
        str path: where to save the json file
    """
    name = nusc.scene[scene_idx]["name"]
    scene = nusc.scene[scene_idx]
    scene_data = sample_extractor(nusc, scene_idx)
    info_list = []
    extract = lambda table, token: nusc.get(table, token)
    instance_tokens = []
    agents = {}
    for j in range(len(scene_data)):
        total = 0
        for i, token in enumerate(scene_data[j]["anns"]):
            agent = _sample_annotations(nusc, token)
            if len(agent) == 0:
                continue
            # for value in agent:
            if agent[0]["instance_token"] in instance_tokens:
                continue
            else:
                instance_tokens.append(agent[0]["instance_token"])
                agents[f"agent_{total}"] = agent
                total += 1

    for idx, sample in enumerate(scene_data):
        sample_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        camera_front = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        info = {"frame_id": idx,
                "token": sample_data["token"],
                "sample_token": sample_data["sample_token"],
                "ego_pose_translation": extract("ego_pose", sample_data["ego_pose_token"])["translation"],
                "ego_pose_rotation": extract("ego_pose", sample_data["ego_pose_token"])["rotation"],
                "timestamp": sample_data["timestamp"],
                "lidar": sample_data["filename"],
                "camera": camera_front["filename"]}

        info_list.append(info)

    info_list.append(agents)
    if path:
        if os.path.exists(path):
            with open(os.path.join(path, name) + ".json", "w") as f:
                f.write(json.dumps(info_list))
        else:
            os.mkdir(path)
            with open(os.path.join(path, name) + ".json", "w") as f:
                f.write(json.dumps(info_list))

    else:
        return info_list


if __name__ == "__main__":
    root = "nuScene-mini"
    nusc = load_dataset(root, verbose=False)
    # render_scene_lidar(root, nusc, 0, save_path="demo", blit=True)
    for idx in range(len(nusc.scene)):
        extract_scene_data_as_json(nusc, idx, "exported_json_data")