import os.path as osp
import numpy as np
import ujson as json
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from nuscenes.utils.data_classes import LidarPointCloud
from data_helpers import create_feature_matrix_for_cae


class NuSceneDataset(Dataset):
    """nuScenes dataset includes lidar data, camera images, positions, physical features
    """

    def __init__(self, json_file: str, root_dir: str, transform=None):
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


class CAEDataset(Dataset):
    """inherited from pytorch dataset builder to facilitate reading, spliting, shuffling data.

    """

    def __init__(self, json_file: str, root_dir: str, transform=None):
        self.scene_frames = create_feature_matrix_for_cae(json_file)
        # self.lidar_address, self.camera_address = self.read_file(json_file)
        self.root_dir = root_dir
        self.transform = transform

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
            sample = sample.transpose(2, 0, 1)

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
    data = create_feature_matrix_for_cae("exported_json_data/scene-1100.json")
    print(data.shape)
    transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    data = NuSceneDataset("exported_json_data/scene-1100.json", "/home/nao/Projects/sasgan/sasgan/data/nuScene-mini",
                         transform=transforms)
    print(len(data))
    print(data[0].shape)
    # print(data[0][14 * 1:14 * 2].reshape(-1, 14).shape)
