import os
import json
import numpy as np
from imageio import imread
import torch
from torch.utils.data import Dataset
from nuscenes.utils.data_classes import LidarPointCloud


def read_file(file: str, feature: str = None):
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
                appears.append((f"agent_{i}", data[j]["frame_id"],
                                 (data[j]["frame_id"] + len(data[-1][f"agent_{i}"]))))

    if feature:
        return data, features, appears

    return data, appears


# def collate_fn(batch):
#     # batch = torch.transpose(batch, 1, 0)
#     print(len(batch))
#     len_batch = 14
#     # for i in range(40):
#     #     samples.append(batch[])
#     # samples = (batch[::40,],)
#     return torch.tensor(batch), len_batch


def create_feature_matrix(file):
    datum, timestamps, appears = read_file(file, "timestamp")
    num_frames = len(timestamps) if len(timestamps) < 40 else 40
    ego = []
    agents = np.zeros((len(datum[-1]), 560), dtype=np.double)
    # appears = (agent_num, start, stop)
    # sort by their number of visibilities in frames
    appears = sorted(appears, key=lambda x: x[2] - x[1], reverse=True)
    num = 0
    for key, start, stop in appears:
        for i in range(stop - start):
            agent_data = datum[-1][key][i]["translation"]
            agent_data.extend(datum[-1][key][i]["rotation"])
            agent_data.extend(datum[-1][key][i]["velocity"])
            agent_data.extend(datum[-1][key][i]["size"])
            agent_data.extend([datum[-1][key][i]["movable"]])
            # print(f"{key}, start: {start}, stop: {stop}")
            # agents[int(key.split("_")[-1]), (start * 14) + (i * 14): (start + 1) * 14 + (i * 14)] = np.array(agent_data, dtype=np.float64)
            agents[num, (start * 14) + (i * 14): (start + 1) * 14 + (i * 14)] = np.array(agent_data, dtype=np.double)
        num += 1

    for id in range(num_frames):
        # ego vehicle location
        ego.extend(datum[id]["ego_pose_translation"])
        # ego vehicle rotation
        ego.extend(datum[id]["ego_pose_rotation"])
        ego.extend(datum[id]["ego_pose_velocity"])
        ego.extend([1.907, 4.727, 1.957])  # size of ego-vehicle
        ego.extend([1])  # movable

    else:
        for i in range(40 - num_frames):
            ego.extend([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    ego = torch.tensor(ego, dtype=torch.double).reshape(-1, 560)
    agents = torch.from_numpy(agents)
    datum = torch.cat((ego, agents), 0)
    # datum = torch.transpose(datum, 1, 0)

    return datum


class CAEDataset(Dataset):
    """Scene frames dataset."""

    def __init__(self, json_file, root_dir, transform=None):
        self.scene_frames = create_feature_matrix(json_file)
        # self.lidar_address, self.camera_address = self.read_file(json_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.scene_frames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.scene_frames[idx,:]
        # img_name = os.path.join(self.root_dir,
        #                         self.camera_address[idx])
        # lidar_name = os.path.join(self.root_dir,
        #                         self.lidar_address[idx])
        # image = torch.from_numpy(imread(img_name))
        # lidar = LidarPointCloud.from_file(lidar_name)
        # frames = self.scene_frames[idx, (start * 14) + (i * 14): (start + 1) * 14 + (i * 14)]
        # sample = {'frames': self.scene_frames, 'image': image, 'lidar': lidar}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def read_file(self, file: str, feature: str = None):
        with open(file, "r") as f:
            data = json.load(f)

        ego = data[:-1]
        lidar_address = []
        camera_address = []

        for i in range(len(ego)):
            lidar_address.append(ego[i]["lidar"])
            camera_address.append(ego[i]["camera"])

        return lidar_address, camera_address


# data = create_feature_matrix("exported_json_data/scene-1100.json")
# print(data.shape)
# data = CAEDataset("exported_json_data/scene-1100.json", "/home/nao/Projects/sasgan/data/nuScene-mini")
# print(len(data.lidar_address))
# print(data[0][14 * 1:14 * 2].reshape(-1, 14).shape)