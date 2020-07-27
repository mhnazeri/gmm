import os
from typing import Dict
import numpy as np
import ujson as json
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from data.data_helpers import create_feature_matrix


class NuSceneDataset(Dataset):
    """nuScenes dataset loader
    each sample is a dictionary with keys: past, rel_past, future, motion
    """

    def __init__(self,  root_dir: str, test: bool = False):
        """str root_dir: train_data root directory"""
        if test:
            data = os.path.join(root_dir, "val_data")
        else:
            data = os.path.join(root_dir, "train_data")

        self.files = os.listdir(data)
        self.files = [os.path.join(data, _path) for _path in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        """
        return desired train data with index idx.
        :param int idx: train data index
        :return: train data with index idx
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        loaded_data = torch.load(self.files[idx])
        return loaded_data


class CFEXDataset(Dataset):
    """loading images for contextual feature extractor"""
    def __init__(self,  root_dir: str, test: bool = False):
        """str root_dir: train_data root directory"""
        if test:
            data = os.path.join(root_dir, "val_data")
        else:
            data = os.path.join(root_dir, "train_data")

        self.files = os.listdir(data)
        self.files = [os.path.join(data, _path) for _path in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        return desired train data with index idx.
        :param int idx: train data index
        :return: train data with index idx
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        loaded_data = torch.load(self.files[idx])
        # make the timestep as the channel of the image. each sample has 4 images, therefore the channel is always 4
        images = torch.cat(loaded_data["motion"], dim=0)
        return images


class CAEDataset(Dataset):
    """CAE dataloader separated from the model dataloader because of the input shape"""

    def __init__(self, root_dir: str):
        # self.scene_frames = create_feature_matrix(os.path.join(root_dir,json_file))
        self.root_dir = root_dir
        json_files = os.path.join(root_dir, "meta_data")
        files = os.listdir(json_files)
        files = [os.path.join(json_files, _path) for _path in files]

        self.features = []
        for file in files:
            l = create_feature_matrix(file)
            # print(l[0, :13], l[0, 13:26])
            # print(l[0, 26:39], l[0, 39:52])
            # break
            for i in range(38):
                self.features.append(l[:, (i+1)*13:(i+2)*13] - l[:, (i)*13:(i+1)*13])
                self.features[-1][:, 7:] = l[:, (i+1)*13+7:(i+2)*13]

        self.features = torch.cat(self.features, dim=0)
        self.features = self.features[torch.where(self.features.sum(axis=1) != 0)]
        self.features = (self.features - self.features.mean()) / self.features.std()

        # num_features = list(range(self.features.shape[1]))
        # self.start_stop = list(zip(num_features[::13], num_features[13::13]))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        return desired train data with index idx.
        :param int idx: train data index
        :return: row idx of train dataset
        """
        return self.features[idx]

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
    data = NuSceneDataset("/home/nao/Projects/sasgan/sasgan/data/")
    # data = CAEDataset("/home/nao/Projects/sasgan/sasgan/data/", "exported_json_data/scene-0061.json")
    # data = DataLoader(data, batch_size=1, shuffle=True, num_workers=2, drop_last=True)
    d = data.__getitem__(268)
    print(len(data))
    # print("len each data: ", len(d))
    # # print(d["past"])
    # print(type(d["motion"]))
    # print(len(d["motion"]))
    # print(type(d["motion"][0]))
    # print(d["motion"][0].shape)
    print(10*"-"+"past"+10*"-")
    print(type(d["past"]))
    print(len(d["past"]))
    print(type(d["past"][0]))
    print(d["past"][0].shape)
