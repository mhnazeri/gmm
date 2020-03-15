import os
import numpy as np
import ujson as json
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from nuscenes.utils.data_classes import LidarPointCloud
from data.data_helpers import create_feature_matrix


def seq_collate(data):
    (feaures_list, rel_features_list, image_list, motion_list) = zip(*data)

    _len = [len(seq) for seq in features_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    features = torch.cat(features_list, dim=0)
    rel_features = torch.cat(rel_features_list, dim=0)
    image = torch.cat(image_list, dim=0)
    motion = torch.cat(motion_list, dim=0)
    out = [
        features, rel_features, image, motion
    ]

    return tuple(out)


class NuSceneDataset(Dataset):
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
        images = []
        lidar = []
        features = []
        self.data = []
        for file in files:
            lidar_address, camera_address = self.read_file(file)
            for cam in camera_address:
                images.append(
                    transform(Image.open(os.path.join(image_files, cam))).squeeze()
                    )

            features = create_feature_matrix(file)
            dummy = torch.zeros(100 - len(features), 560)
            features = torch.cat((features, dummy), 0)

            # read the list 14 element at a time
            num_features = list(range(features.shape[1]))
            start_stop = list(zip(num_features[::14], num_features[14::14]))
            data = {}

            stamp = 0
            while stamp < 30:
                past = []
                future = []
                image = []
                # if stamp % 40 == 0:
                for j in range(4):
                    past.append(features[:, start_stop[stamp + j][0]: start_stop[stamp + j][1]])
                    image.append(images[stamp + j])

                for j in range(4, 14):
                    future.append(features[:, start_stop[stamp + j][0]: start_stop[stamp + j][1]])

                data["past"] = past
                data["future"] = future
                data["motion"] = image

                self.data.append(data)
                data = {}

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

        # start, stop = self.start_stop[idx]
        # # all agents (rows), specific column (timestamp)
        # # out = (features, rel_feature, images, motions)
        # if idx % 40 == 0:
        #     out = [
        #     self.features[:, start: stop], torch.zeros_like(self.features[:, start: stop]),]
        #     if not self.only_features:
        #         out.extend([self.images[idx], torch.zeros_like(self.images[idx].shape)])
        # else:
        #     out = [
        #     self.features[:, start: stop], self.rel_features[:, start: stop],]
        #     if not self.only_features:
        #         out.extend([self.images[idx], self.images[idx] - self.images[idx - 1]]
        #         )

        return self.data[idx]

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

    def __init__(self, root_dir: str):
        # self.scene_frames = create_feature_matrix(os.path.join(root_dir,json_file))
        self.root_dir = root_dir
        json_files = os.path.join(root_dir, "exported_json_data")
        files = os.listdir(json_files)
        files = [os.path.join(json_files, _path) for _path in files]
        self.features = []

        for file in files:
            features = create_feature_matrix(file)
            self.features.append(features)

        self.features = torch.cat(self.features, dim=0)

        num_features = list(range(self.features.shape[1]))
        self.start_stop = list(zip(num_features[::14], num_features[14::14]))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        return desired train data with index idx.
        :param int idx: train data index
        :return: row idx of train dataset
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # start, stop = self.start_stop[idx]
        # self.features[:, start: stop]
        # all agents (rows), specific columns (timestamps)
        # sample = self.features[:, start: stop]
        sample = self.features[idx]
        # img_name = os.path.join(self.root_dir,
        #                         self.camera_address[idx])
        # lidar_name = os.path.join(self.root_dir,
        #                         self.lidar_address[idx])
        # image = torch.from_numpy(imread(img_name))
        # lidar = LidarPointCloud.from_file(lidar_name)
        # frames = self.scene_frames[idx, (start * 14) + (i * 14): (start + 1) * 14 + (i * 14)]
        # sample = {'frames': self.scene_frames, 'image': image, 'lidar': lidar}

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
    data = NuSceneDataset("/home/nao/Projects/sasgan/sasgan/data/", transform=transforms)
    # print(len(data.__getitem__(0)))
    # data = CAEDataset("/home/nao/Projects/sasgan/sasgan/data/", "exported_json_data/scene-0061.json")
    # data = DataLoader(data, batch_size=1, shuffle=True, num_workers=2, drop_last=True)
    d = data.__getitem__(0)
    print(d)