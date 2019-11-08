"""Argoverse dataset loader. 5 seconds segments captured @10hz"""
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_forecasting_loader import (
    ArgoverseForecastingLoader,
)
from argoverse.visualization.visualize_sequences import viz_sequence


class ToTensor(object):
    """COnvert ndarrays to torch tensors"""

    def __call__(self, sample):
        sample.agent_traj = torch.from_numpy(sample.agent_traj)
        sample.seq_df = torch.from_numpy(sample.seq_df)

        return sample


class ArgoversDataset(Dataset):
    """Argoverse dataset"""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Path to the dataset directory
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = ArgoverseForecastingLoader(root_dir)
        self.map = ArgoverseMap()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def track_id_list(self, datum):
        """Return list of agent id's in the scene"""
        return datum.track_id_list

    def visualize_sequence(self, datum, show=True):
        """Visualize a five seconds segment"""
        viz_sequence(datum.seq_df, show=show)

    def visualize_candidate_centerlines_for_traj(datum, obs_len, viz=True):
        """Visualize candidate centerlines for given trajectory"""
        avm = ArgoverseMap()
        agent_obs_traj = datum.agent_traj[:obs_len]
        candidate_centerlines = avm.get_cadidate_centerlines_for_traj(
            agent_obs_traj, datum.city, viz=viz
        )

    def get_lane_direction(datum, viz=True):
        """Get the lane direction of the segment"""
        agent_traj = datum.agent_traj
        avm = ArgoverseMap()
        lane_direction = avm.get_lane_direction(
            agent_traj[0], datum.city, visualize=viz
        )


if __name__ == "__main__":
    root_dir = "./argoverse_train/"
    data = ArgoversDataset(root_dir)
    data.visualize_sequence(data[0])
