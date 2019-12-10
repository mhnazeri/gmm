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

def _sample_annotations(nusc, instance):
    sample_annotation = nusc.get("sample_annotation", instance)
    annotation = []
    while sample_annotation["next"] != "":
        annotation.append({"translation": sample_annotation["translation"],
                        "rotation": sample_annotation["rotation"],
                        "size": sample_annotation["size"],
                        "visibility": sample_annotation["visibility_token"],
                        "category": sample_annotation["category_name"],
                        "instance_token": sample_annotation["instance_token"],
                        "sample_token": sample_annotation["sample_token"]})

        sample_annotation = nusc.get("sample_annotation", sample_annotation["next"])

    return annotation


def extract_scene_data_as_json(nusc, scene_idx, path=None):
    """Write scene data as json file in to given path
    args:
        Nuscene nusc: Nuscene dataset
        int scene_idx: index of the desired scene
        str path: where to save the json file"""
    name = nusc.scene[scene_idx]["name"]
    scene = nusc.scene[scene_idx]
    scene_data = sample_extractor(nusc, scene_idx)
    info_list = []
    extract = lambda table, token: nusc.get(table, token)
    instance_tokens = []
    agents = {}
    # print(scene_data)
    for i, token in enumerate(scene_data[0]["anns"]):
            agent = _sample_annotations(nusc, token)
            for value in agent:
                if value["instance_token"] not in instance_tokens:
                    instance_tokens.append(value["instance_token"])
                    agents[f"agent_{i}"] = agent

    for idx, sample in enumerate(scene_data):
        # all annotations of the sample
        # sample_annotations = sample["anns"]
        # retreive each agent position in the scene, the length may vary
        # agents = {f"agent_{i}": _sample_annotations(nusc, token) for i, token in enumerate(sample["anns"])}

        sample_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        info = {"id_sample": idx,
                "token": sample_data["token"],
                "sample_token": sample_data["sample_token"],
                "ego_pose_translation": extract("ego_pose", sample_data["ego_pose_token"])["translation"],
                "ego_pose_rotation": extract("ego_pose", sample_data["ego_pose_token"])["rotation"],
                "timestamp": sample_data["timestamp"],
                "filename": sample_data["filename"],
                "channel": sample_data["channel"]}

        info_list.append(info)
        # for info in info_list:
        #     for j in range(len(info_list)):
        #         if info[]
        # info_list.append(agents)
    info_list.append(agents)

    # info_list.append(agents)
    # for i in range(len(info_list)):
    #     for key, value in info_list[i][agents]:
    #         for token in value:
    #             if token["instance_token"]

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
    frames = sample_extractor(nusc, 1)
    # print(len(frames[0]["anns"]))
    # print(nusc.get("sample_data", frames[0]["data"]["LIDAR_TOP"]))
    for idx in range(len(nusc.scene)):
        extract_scene_data_as_json(nusc, idx, "exported_json_data")