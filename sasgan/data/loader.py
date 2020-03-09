import os
import numpy as np
import ujson as json
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from nuscenes.utils.data_classes import LidarPointCloud
from data_helpers import create_feature_matrix


def seq_collate(data):
    (feaures_list, rel_features_list, image_list, motion_list) = zip(*data)

    _len = [len(seq) for seq in features_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    features = torch.cat(features_list, dim=0).permute(2, 0, 1)
    rel_features = torch.cat(rel_features_list, dim=0).permute(2, 0, 1)
    image = torch.cat(image_list, dim=0).permute(2, 0, 1)
    motion = torch.cat(motion_list, dim=0).permute(2, 0, 1)
    out = [
        features, rel_features, image, motion
    ]

    return tuple(out)


class NuSceneDataset_copy(Dataset):
    """nuScenes dataset includes lidar data, camera images, positions, physical features
    """

    def __init__(self,  root_dir: str, transform=None):
        """str root_dir: json files directory"""
        self.root_dir = root_dir
        self.transform = transform
        json_files = os.path.join(root_dir, "exported_json_data")
        image_files = os.path.join(root_dir, "nuScene-mini")
        files = os.listdir(json_files)
        files = [os.path.join(json_files, _path) for _path in files]
        self.images = []
        self.lidar = []
        self.features = []

        for file in files:
            lidar_address, camera_address = self.read_file(file)
            for cam in camera_address:
                self.images.append(
                    transform(Image.open(os.path.join(image_files, cam))).squeeze())

            features = create_feature_matrix(file)
            dummy = torch.zeros(100 - len(features), 560)
            features = torch.cat((features, dummy), 0)
            self.features.append(features)

        self.features = torch.cat(self.features, dim=1)
        self.rel_features = self.features[:, 1:] - self.features[:, : -1]
        # read the list 14 element at a time
        num_features = list(range(self.features.shape[1]))
        self.start_stop = list(zip(num_features[::14], num_features[14::14]))
        # self.start_stop = [l[x[0]:x[1]] for x in t]

    def __len__(self):
        return len(self.start_stop)

    def __getitem__(self, idx):
        """
        return desired train data with index idx.
        :param int idx: train data index
        :return: row idx of train dataset
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}
        start, stop = self.start_stop[idx]
        # all agents (rows), specific columns (timestamps)
        sample["features"] = self.features[:, start: stop]
        sample["image"] = self.images[idx]
        if idx % 39 == 0:
            sample["rel_features"] = self.features[:, start: stop]
            sample["motion"] = torch.zeros_like(self.images[idx])
        else:
            sample["rel_features"] = self.rel_features[:, start: stop]
            sample["motion"] = self.images[idx] - self.images[idx - 1]

        return sample

    def read_file(self, file: str, feature: str = None):
        """
        read json file of a scene.
        separate different agent's lidar, camera features
        :param str file: file name of the scene.
        :param feature: desired feature. (not using this feature currently)
        :return:
        """
        with open(file, "r") as f:
            data = json.load(f)

        ego = data[:-1]
        lidar_address = []
        camera_address = []

        for i in range(len(ego)):
            lidar_address.append(ego[i]["lidar"])
            camera_address.append(ego[i]["camera"])

        return lidar_address, camera_address


# class NuSceneDataset(Dataset):
#     """nuScenes dataset includes lidar data, camera images, positions, physical features
#     """

#     def __init__(self,  root_dir: str, json_file: str, transform=None):
#         self.lidar_address, self.camera_address = self.read_file(json_file)
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.lidar_address)

#     def __getitem__(self, idx):
#         """
#         return desired train data with index idx.
#         :param int idx: train data index
#         :return: row idx of train dataset
#         """
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         sample = Image.open(osp.join(self.root_dir, self.camera_address[idx]))

#         # img_name = os.path.join(self.root_dir,
#         #                         self.camera_address[idx])
#         # lidar_name = os.path.join(self.root_dir,
#         #                         self.lidar_address[idx])
#         # image = torch.from_numpy(imread(img_name))
#         # lidar = LidarPointCloud.from_file(lidar_name)
#         # frames = self.scene_frames[idx, (start * 14) + (i * 14): (start + 1) * 14 + (i * 14)]
#         # sample = {'frames': self.scene_frames, 'image': image, 'lidar': lidar}

#         if self.transform:
#             sample = self.transform(sample).squeeze()

#         return sample

#     def read_file(self, file: str, feature: str = None):
#         """
#         read json file of a scene.
#         separate different agent's lidar, camera features
#         :param str file: file name of the scene.
#         :param feature: desired feature. (not using this feature currently)
#         :return:
#         """
#         with open(file, "r") as f:
#             data = json.load(f)

#         ego = data[:-1]
#         lidar_address = []
#         camera_address = []

#         for i in range(len(ego)):
#             lidar_address.append(ego[i]["lidar"])
#             camera_address.append(ego[i]["camera"])

#         return lidar_address, camera_address


class CAEDataset(Dataset):
    """inherited from pytorch dataset builder to facilitate reading, spliting, shuffling data.

    """

    def __init__(self, json_file: str, root_dir: str, transform=None):
        self.scene_frames = create_feature_matrix(json_file)
        self.root_dir = root_dir
        self.transform = transform
        dummy = torch.zeros(100 - len(self.scene_frames), 560)
        self.scene_frames = torch.cat((self.scene_frames, dummy), 0)
        # if you want read the list 5 element at a time
        t = list(zip(l[::28], l[28::28]))
        self.start_stop = [l[x[0]:x[1]] for x in t]

    def __len__(self):
        return len(self.scene_frames)

    def __getitem__(self, idx):
        """
        return desired train data with index idx.
        :param int idx: train data index
        :return: row idx of train dataset
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        start, stop = self.start_stop[idx]
        # all agents (rows), specific columns (timestamps)
        # sample = self.scene_frames[:, start: stop]
        sample = self.scene_frames[idx]
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
        """
        read json file of a scene.
        separate different agent's lidar, camera features
        :param str file: file name of the scene.
        :param feature: desired feature. (not using this feature currently)
        :return:
        """
        with open(file, "r") as f:
            data = json.load(f)

        ego = data[:-1]
        lidar_address = []
        camera_address = []

        for i in range(len(ego)):
            lidar_address.append(ego[i]["lidar"])
            camera_address.append(ego[i]["camera"])

        return lidar_address, camera_address


if __name__ == '__main__':
    transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    data = NuSceneDataset_copy("/home/nao/Projects/sasgan/sasgan/data/", transform=transforms)
    print(len(data))
