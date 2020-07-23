"""This module is responsible for loading of NuScene dataset
"""
import os
import argparse
import logging
import ujson as json
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
import imageio
import torch
# from sasgan.data.traj_viz import render_scene_lidar


logger = logging.getLogger(__name__)


def load_dataset(_path, version="v1.0-mini", verbose=False):
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


def _sample_annotations(nusc, instance):
    sample_annotation = nusc.get("sample_annotation", instance)
    annotation = []
    movable = (
        0
        if sample_annotation["category_name"].split(".")[-1]
        in "barrier debris pushable_pul trafficcone bicycle_rack"
        else 1
    )

    if movable:
        while sample_annotation["next"] != "":
            annotation.append(
                {
                    "translation": sample_annotation["translation"],
                    "rotation": sample_annotation["rotation"],
                    "size": sample_annotation["size"],
                    "visibility": sample_annotation["visibility_token"],
                    "category": sample_annotation["category_name"],
                    "instance_token": sample_annotation["instance_token"],
                    "sample_token": sample_annotation["sample_token"],
                    "timestamp": nusc.get("sample", sample_annotation["sample_token"])[
                        "timestamp"
                    ],
                    "velocity": box_velocity(nusc, sample_annotation["token"]).tolist()
                }
            )

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
        info = {
            "frame_id": idx,
            "token": sample_data["token"],
            "sample_token": sample_data["sample_token"],
            "ego_pose_translation": extract("ego_pose", sample_data["ego_pose_token"])[
                "translation"
            ],
            "ego_pose_rotation": extract("ego_pose", sample_data["ego_pose_token"])[
                "rotation"
            ],
            "ego_pose_velocity": ego_velocity(
                nusc, sample["data"]["LIDAR_TOP"]
            ).tolist(),
            "calibrated_translation": extract("calibrated_sensor", sample_data["calibrated_sensor_token"])["translation"],
            "calibrated_rotation": extract("calibrated_sensor", sample_data["calibrated_sensor_token"])["rotation"],
            "timestamp": sample_data["timestamp"],
            "lidar": sample_data["filename"],
            "camera": camera_front["filename"],
        }

        info_list.append(info)

    info_list.append(agents)
    if path:
        if os.path.exists(path):
            with open(os.path.join(path, name) + ".json", "w") as f:
                f.write(json.dumps(info_list, indent=4, escape_forward_slashes=False))
        else:
            os.mkdir(path)
            with open(os.path.join(path, name) + ".json", "w") as f:
                f.write(json.dumps(info_list, indent=4, escape_forward_slashes=False))

    else:
        return info_list


def ego_velocity(
    nusc, sample_frame_token: str, max_time_diff: float = 1.5
) -> np.ndarray:
    """
        Estimate the velocity for ego-vehicle.
        If possible, we compute the centered difference between the previous and next frame.
        Otherwise we use the difference between the current and previous/next frame.
        If the velocity cannot be estimated, values are set to 0.
        :param sample_annotation_token: Unique sample_annotation identifier.
        :param max_time_diff: Max allowed time diff between consecutive samples that are used to estimate velocities.
        :return: <np.float: 3>. Velocity in x/y/z direction in m/s.
        """

    current = nusc.get("sample_data", sample_frame_token)
    has_prev = current["prev"] != ""
    has_next = current["next"] != ""

    # Cannot estimate velocity for a single annotation.
    if not has_prev and not has_next:
        return np.array([0.0, 0.0, 0.0])

    if has_prev:
        first = nusc.get("sample_data", current["prev"])
    else:
        first = current

    if has_next:
        last = nusc.get("sample_data", current["next"])
    else:
        last = current

    pos_last = np.array(nusc.get("ego_pose", last["ego_pose_token"])["translation"])
    pos_first = np.array(nusc.get("ego_pose", first["ego_pose_token"])["translation"])
    pos_diff = pos_last - pos_first

    time_last = 1e-6 * nusc.get("ego_pose", last["ego_pose_token"])["timestamp"]
    time_first = 1e-6 * nusc.get("ego_pose", first["ego_pose_token"])["timestamp"]
    time_diff = time_last - time_first

    if has_next and has_prev:
        # If doing centered difference, allow for up to double the max_time_diff.
        max_time_diff *= 2

    if time_diff > max_time_diff:
        # If time_diff is too big, return 0.
        return np.array([0.0, 0.0, 0.0])
    else:
        return pos_diff / time_diff


def box_velocity(nusc, sample_annotation_token: str, max_time_diff: float = 1.5) -> np.ndarray:
    """
    Estimate the velocity for an annotation.
    If possible, we compute the centered difference between the previous and next frame.
    Otherwise we use the difference between the current and previous/next frame.
    If the velocity cannot be estimated, values are set to 0.
    :param sample_annotation_token: Unique sample_annotation identifier.
    :param max_time_diff: Max allowed time diff between consecutive samples that are used to estimate velocities.
    :return: <np.float: 3>. Velocity in x/y/z direction in m/s.
    """

    current = nusc.get('sample_annotation', sample_annotation_token)
    has_prev = current['prev'] != ''
    has_next = current['next'] != ''

    # Cannot estimate velocity for a single annotation.
    if not has_prev and not has_next:
        return np.array([0.0, 0.0, 0.0])

    if has_prev:
        first = nusc.get('sample_annotation', current['prev'])
    else:
        first = current

    if has_next:
        last = nusc.get('sample_annotation', current['next'])
    else:
        last = current

    pos_last = np.array(last['translation'])
    pos_first = np.array(first['translation'])
    pos_diff = pos_last - pos_first

    time_last = 1e-6 * nusc.get('sample', last['sample_token'])['timestamp']
    time_first = 1e-6 * nusc.get('sample', first['sample_token'])['timestamp']
    time_diff = time_last - time_first

    if has_next and has_prev:
        # If doing centered difference, allow for up to double the max_time_diff.
        max_time_diff *= 2

    if time_diff > max_time_diff:
        # If time_diff is too big, return 0.
        return np.array([0.0, 0.0, 0.0])
    else:
        return pos_diff / time_diff


def move_samples(source: str, dest: str, portion: float, seed: int = 42):
    """randomly selects portion of data and move them to test directory
    args:
        str source: source folder
        str dest: destination folder to move selected samples
        float portion: how much of the data you want as test
        int seed: if you don't want reproducibility set seed to `None`
    """
    if seed:
        random.seed(seed)

    files = os.listdir(source)
    total = len(files)
    test_portion = int(total * portion)
    selected_files = []
    if not os.path.exists(dest):
        os.mkdir(dest)

    # selects samples randomly
    for _ in range(test_portion):
        file = random.choice(files)
        selected_files.append(file)
        # to prevent selecting the sample multiple times
        del files[files.index(file)]

    for file in selected_files:
        os.rename(os.path.join(source, file), os.path.join(dest, file))

    print(f"{test_portion} samples are selected as test samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('nuscenes', dest='nuscenes', type=str, help='set nuscenes directory')
    parser.add_argument('--source', dest='source', type=str, default="meta_data", help='source directory')
    parser.add_argument('--dest', dest='dest', type=str, default="meta_data_val", help='destination directory')
    parser.add_argument('--portion', dest='portion', type=float, default=0.01, help='what percentage of data should be used for testing')
    parser.add_argument('--seed', dest='seed', type=int, default=42, help='random seed')

    arguments = parser.parse_args()
    root = arguments.nuscenes
    nusc = load_dataset(root, verbose=False)
    print(f"Total scenes: {len(nusc.scene)}")
    print("Start converting to json")

    for idx in range(len(nusc.scene)):
        print(f"Convering scene {idx} from {len(nusc.scene)}")
        extract_scene_data_as_json(nusc, idx, "meta_data")

    print("Conversion is completed")
    move_samples(args.source, args.dest, args.portion, args.seed)
