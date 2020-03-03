import os.path as osp
import numpy as np
import ujson as json
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from nuscenes.utils.data_classes import LidarPointCloud
from data_helpers import create_feature_matrix_for_cae
from utils import config


class NuSceneDataset(Dataset):
    """nuScenes dataset includes lidar data, camera images, positions, physical features
    """

    def __init__(self,  root_dir: str, json_file: str, transform=None):
        self.lidar_address, self.camera_address = self.read_file(json_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.lidar_address)

    def __getitem__(self, idx):
        """
        return desired train data with index idx.
        :param int idx: train data index
        :return: row idx of train dataset
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = Image.open(osp.join(self.root_dir, self.camera_address[idx]))

        # img_name = os.path.join(self.root_dir,
        #                         self.camera_address[idx])
        # lidar_name = os.path.join(self.root_dir,
        #                         self.lidar_address[idx])
        # image = torch.from_numpy(imread(img_name))
        # lidar = LidarPointCloud.from_file(lidar_name)
        # frames = self.scene_frames[idx, (start * 14) + (i * 14): (start + 1) * 14 + (i * 14)]
        # sample = {'frames': self.scene_frames, 'image': image, 'lidar': lidar}

        if self.transform:
            sample = self.transform(sample).squeeze()

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


class CAEDataset(Dataset):
    """inherited from pytorch dataset builder to facilitate reading, spliting, shuffling data.

    """

    def __init__(self, json_file: str, root_dir: str, transform=None):
        self.scene_frames = create_feature_matrix_for_cae(json_file)
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
    paths = config("Paths")
    data = create_feature_matrix_for_cae("exported_json_data/scene-1100.json")
    print(data.shape)
    transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    data = NuSceneDataset("/home/nao/Projects/sasgan/sasgan/data/nuScene-mini", "exported_json_data/scene-1100.json",
                         transform=transforms)
    print(len(data))
    print(data[0].shape)
    # print(type(data[0]))
    # print(data[0][14 * 1:14 * 2].reshape(-1, 14).shape)
