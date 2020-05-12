"""This module is responsible for loading of NuScene dataset
TODO:
1) read sensor data
2) vectorized data
3) each scene is a data sample
"""
import os
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
                "movable": movable,
                "velocity": nusc.box_velocity(sample_annotation["token"]).tolist()
                if movable
                else torch.zeros(3, dtype=torch.float32),
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
                f.write(json.dumps(info_list, indent=4))
        else:
            os.mkdir(path)
            with open(os.path.join(path, name) + ".json", "w") as f:
                f.write(json.dumps(info_list, indent=4))

    else:
        return info_list


def ego_velocity(
    nusc, sample_frame_token: str, max_time_diff: float = 1.5
) -> np.ndarray:
    """
        Estimate the velocity for ego-vehicle.
        If possible, we compute the centered difference between the previous and next frame.
        Otherwise we use the difference between the current and previous/next frame.
        If the velocity cannot be estimated, values are set to np.nan.
        :param sample_annotation_token: Unique sample_annotation identifier.
        :param max_time_diff: Max allowed time diff between consecutive samples that are used to estimate velocities.
        :return: <np.float: 3>. Velocity in x/y/z direction in m/s.
        """

    current = nusc.get("sample_data", sample_frame_token)
    has_prev = current["prev"] != ""
    has_next = current["next"] != ""

    # Cannot estimate velocity for a single annotation.
    if not has_prev and not has_next:
        return np.array([np.nan, np.nan, np.nan])

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
        # If time_diff is too big, don't return an estimate.
        return np.array([np.nan, np.nan, np.nan])
    else:
        return pos_diff / time_diff


def backgraound_motion_detector(root: str, img_1: np.ndarray=None, img_2: np.ndarray = None)-> None:
    im = imageio.imread(os.path.join(root, "samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg"))
    im_2 = imageio.imread(os.path.join(root, "samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402928112460.jpg"))

    im = im / 255
    im_2 = im_2 / 255
    fig = plt.figure()
    a = fig.add_subplot(3, 1, 1)
    plt.imshow(im)
    # print("image 1: ", im.shape)
    # print("image 2: ", im_2.shape)
    fig.add_subplot(3, 1, 2)
    a.set_title("Frame 2")
    plt.imshow(im_2)
    a = fig.add_subplot(3, 1, 3)
    a.set_title("Motion")
    plt.imshow(im_2 - im, cmap="hot")
    plt.show()


if __name__ == "__main__":
    root = "nuScene-mini"
    nusc = load_dataset(root, verbose=False)
    for idx in range(len(nusc.scene)):
        extract_scene_data_as_json(nusc, idx, "exported_json_data")

    # sample = nusc.get("sample", "378a3a3e9af346308ab9dff8ced46d9c")
    # sample_ann = nusc.get("sample_annotation", sample["anns"][0])
    # # print(sample_ann)
    # sample_data = nusc.get("sample", sample_ann["sample_token"])
    # # print(sample_data)
    # lidar = nusc.get("sample_data", sample_data["data"]["LIDAR_TOP"])
    # # print(lidar)
    # # print(nusc.get("ego_pose", lidar["ego_pose_token"]))

    # print(nusc.get("calibrated_sensor", lidar["calibrated_sensor_token"]))
    # print(nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"]))

    # im = imageio.imread(os.path.join(root, "samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg"))
    # im_2 = imageio.imread(os.path.join(root, "samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402928112460.jpg"))

    # fig = plt.figure()
    # a = fig.add_subplot(2, 1, 1)
    # plt.imshow(im)
    # # print("image 1: ", im.shape)
    # # print("image 2: ", im_2.shape)
    # a = fig.add_subplot(2, 1, 2)
    # plt.imshow(im_2)
    # plt.show()
