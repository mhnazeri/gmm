"""General Helpers functions that can be used everywhere"""
import os
import argparse
import numpy as np
from PIL import Image
import ujson as json
import torch
from torchvision import transforms


def read_file(file: str, feature: str = None):
    """read json files"""
    with open(file, "r") as f:
        data = json.load(f)

    features = []
    for id in range(len(data[:-1])):
        features.append(data[id][feature])

    appears = []

    for i in range(len(data[-1])):
        # agent_traj = []
        for j in range(len(data) - 1):
            if data[j]["sample_token"] == data[-1][f"agent_{i}"][0]["sample_token"]:
                appears.append(
                    (
                        f"agent_{i}",
                        data[j]["frame_id"],
                        (data[j]["frame_id"] + len(data[-1][f"agent_{i}"])),
                    )
                )

    if feature:
        return data, features, appears
    print(appears)
    return data, appears


def create_feature_matrix(file, min_frames: int = 10):
    datum, timestamps, appears = read_file(file, "timestamp")
    num_frames = len(timestamps) if len(timestamps) < 40 else 40
    ego = []
    agents = np.zeros((len(datum[-1]), 520))
    # appears = (agent_num, start, stop)
    # sort by their number of visibilities in frames
    appears = sorted(appears, key=lambda x: x[2] - x[1], reverse=True)
    num = 0
    for key, start, stop in appears:
        if stop - start >= min_frames:
            for i in range(stop - start):
                # if datum[-1][key][i]["movable"] != 0:
                agent_data = datum[-1][key][i]["translation"]
                agent_data.extend(datum[-1][key][i]["rotation"])
                agent_data.extend(datum[-1][key][i]["velocity"])
                agent_data.extend(datum[-1][key][i]["size"])
                # agents[int(key.split("_")[-1]), (start * 14) + (i * 14): (start + 1) * 14 + (i * 14)] = np.array(agent_data, dtype=np.float64)
                agents[
                    num, (start * 13) + (i * 13) : (start + 1) * 13 + (i * 13)
                ] = np.array(agent_data)
            num += 1

    for id in range(num_frames):
        # ego vehicle location
        ego.extend(datum[id]["ego_pose_translation"])
        # ego vehicle rotation
        ego.extend(datum[id]["ego_pose_rotation"])
        ego.extend(datum[id]["ego_pose_velocity"])
        ego.extend([1.907, 4.727, 1.957])  # size of ego-vehicle
        # ego.extend([1])  # movable

    else:
        for i in range(40 - num_frames):
            ego.extend(torch.zeros(13, dtype=torch.float32))

    ego = torch.tensor(ego).reshape(-1, 520)
    agents = torch.from_numpy(agents).float()
    datum = torch.cat((ego, agents), 0)

    return datum


def create_feature_matrix_for_viz(file):
    """create feature matrix for visualization purposes"""
    datum, timestamps, appears = read_file(file, "timestamp")
    num_frames = len(timestamps) if len(timestamps) < 40 else 40
    ego = []
    agents = np.zeros((len(datum[-1]), 80))
    calibrated_features = []
    # appears = (agent_num, start, stop)
    # sort by their number of visibilities in frames
    appears = sorted(appears, key=lambda x: x[2] - x[1], reverse=True)
    num = 0
    for key, start, stop in appears:
        for i in range(stop - start):
            agent_data = datum[-1][key][i]["translation"][:2]
            agents[num, (start * 2) + (i * 2) : (start + 1) * 2 + (i * 2)] = np.array(
                agent_data
            )
        num += 1

    for id in range(num_frames):
        # ego vehicle location
        ego.extend(datum[id]["ego_pose_translation"][:2])
        calibrated_features.append(
            (
                datum[id]["calibrated_translation"][:2],
                datum[id]["calibrated_rotation"][:2],
            )
        )
    else:
        for i in range(40 - num_frames):
            ego.extend(torch.zeros(2, dtype=torch.float32))

    ego = np.array(ego).reshape(-1, 80)
    agents = agents
    datum = np.concatenate((ego, agents), 0)

    return datum, calibrated_features


def cal_distance(tensor: torch.Tensor) -> torch.Tensor:
    pivot = tensor[0]
    distance = (tensor - pivot).pow(2)
    _, indices = torch.sort(distance, dim=0)
    return indices[:, 0]


def save_train_samples(
    nuscenes_root: str,
    root_dir: str,
    source: str,
    save_dir: str,
    arch: str = "overfeat",
    min_frames: int = 10,
):
    """save each train sample on hdd"""
    if arch == "overfeat":
        transform = transforms.Compose(
            [
                transforms.Resize((231, 231)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=0, std=1),
            ]
        )
    elif arch == "vgg":
        transform = transforms.Compose(
            [
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=0, std=1),
            ]
        )

    json_files = os.path.join(root_dir, source)
    files = os.listdir(json_files)
    files = [os.path.join(json_files, _path) for _path in files]
    # check if save_dir exists, otherwise create one
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # read the list 13 element at a time
    num_features = [x for x in range(521) if x % 13 == 0]
    start_stop = list(zip(num_features[:], num_features[1:]))
    index = 0

    for file in files:
        # read lidar and camera data locations
        # lidar_address, camera_address = [], [] # read_file(file)
        with open(file, "r") as f:
            _data = json.load(f)

        ego = _data[:-1]
        # lidar_address = []
        camera_address = []

        for i in range(len(ego)):
            # lidar_address.append(ego[i]["lidar"])
            camera_address.append(ego[i]["camera"])
        # for each file (scene), creates corresponding matrix
        features = create_feature_matrix(file, min_frames=min_frames)
        # converting images to differences of consecutive frames
        images = [
            transform(Image.open(os.path.join(nuscenes_root, address)))
            for address in camera_address
        ]
        images = [
            image_1 - image_0 for image_0, image_1 in zip(images[:-1], images[1:])
        ]
        features_rel = features[:, 13:] - features[:, :-13]
        # skip first frame
        features = features[:, 13:]
        data = {}
        stamp = 0

        while stamp < 26:
            past = []
            past_rel = []
            future = []
            future_rel = []
            motion = []
            indices = cal_distance(
                features[:, start_stop[stamp][0] : start_stop[stamp][1]]
            )
            for j in range(4):
                # 4 frames in the past
                past.append(
                    features[:, start_stop[stamp + j][0] : start_stop[stamp + j][1]][
                        indices
                    ]
                )
                past_rel.append(
                    features_rel[
                        :, start_stop[stamp + j][0] : start_stop[stamp + j][1]
                    ][indices]
                )
                # each frame has an image
                motion.append(images[stamp + j])

            for j in range(4, 14):
                # 10 frames in the future (for future, choose only x and y)
                future.append(
                    features[:, start_stop[stamp + j][0] : start_stop[stamp + j][1]][
                        indices, :2
                    ]
                )
                future_rel.append(
                    features_rel[
                        :, start_stop[stamp + j][0] : start_stop[stamp + j][1]
                    ][indices, :2]
                )

            # copy agent's size from real past to relative past tensor
            for i in range(4):
                past_rel[i][:, 10:] = past[i][:, 10:]

            data["past"] = torch.stack(past, dim=0)
            # rel_past = torch.stack(rel_past, dim=0)
            data["rel_past"] = torch.stack(past_rel, dim=0)
            data["motion"] = motion
            data["future"] = torch.stack(future, dim=0)
            data["rel_future"] = torch.stack(future_rel, dim=0)

            # save data on hard
            torch.save(data, os.path.join(save_dir, f"{index}.pt"))
            print(f"sample {index} has been saved")
            index += 1
            data = {}
            stamp += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("nuscenes", type=str, help="set nuscenes directory")
    parser.add_argument(
        "--source",
        dest="source",
        type=str,
        default="meta_data",
        help="source directory",
    )
    parser.add_argument(
        "--save_dir",
        dest="save_dir",
        type=str,
        default="train_data",
        help="save directory",
    )
    parser.add_argument(
        "--arch",
        dest="arch",
        type=str,
        default="overfeat",
        help="feature extractor model architecture",
    )
    parser.add_argument(
        "--min_frames",
        dest="min_frames",
        type=int,
        default=10,
        help="minimum number of frames that an agent should be present",
    )
    args = parser.parse_args()
    save_train_samples(
        args.nuscenes, ".", args.source, args.save_dir, args.arch, args.min_frames
    )
    print("Saving samples is completed!")
