""" This file contains the implementation of the main models including:
    1. Encoder
    2. Contextual Feature Extractor
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os
from utils import *
import logging
from data.loader import CAEDataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# The logger used for debugging
logger = logging.getLogger(__name__)

# tensorboard logger
global tensorboard_logger
tensorboard_logger = Logger()

##################################################################################
#                                    Encoder
# ________________________________________________________________________________
class Encoder(nn.Module):
    def __init__(self, input_size: int = 7, hidden_size: int = 32, num_layers:int = 1, batch_size:int=64):
        """
        :param input_size: The size of each vector representing an agent in a frame containing all the features
        :param hidden_size: The size of the hidden dimension of LSTM layer. As bigger the hidden dimension as higher the
            capability of the encoder to model longer sequences
        :param num_layers: The depth of the LSTM networks, Set to one in SGAN original code
        """
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def initiate_hidden(self, batch_size):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
            torch.zeros(self.num_layers, batch_size, self.hidden_size)
        )

    def forward(self, inputs):
        """
        :param inputs: A tensor of the shape (sequence_length, batch, input_size)
        :return: A tensor of the shape (self.num_layers, batch, self.hidden_size)
        """

        # Check the integrity of the shapes
        logger.debug("The size of the inputs: " + str(inputs.size()))

        # batch_size = inputs.size(1)
        # embed = self.embedder(inputs.view(-1, 2))
        inputs = inputs.view(self.batch_size, 100, -1)

        states = self.initiate_hidden(self.batch_size)
        _, hidden_state, _ = self.lstm(inputs, states)

        return hidden_state


##################################################################################
#                        Contextual Feature Extractor
# ________________________________________________________________________________
class ContextualFeatures(nn.Module):
    """Extract contextual features from the environment
        Networks that can be used for feature extraction are:
            overfeat: returned matrix is 1024*12*12
    """
    def __init__(self, model_arch: str="overfeat"):
        super(ContextualFeatures, self).__init__()
        if model_arch == "overfeat":
            self.layer_1 = nn.Sequential(
                nn.Conv2d(in_channels=1, kernel_size=11, stride=4,
                                     out_channels=96),
                nn.MaxPool2d(kernel_size=2, stride=2)
                )
            # self.layer_1 = nn.Conv2d(in_channels=1, kernel_size=11, stride=4,
            #                          out_channels=96)
            # self.layer_1_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
            self.layer_2 = nn.Sequential(
                nn.Conv2d(in_channels=96, out_channels=256,
                                     kernel_size=5, stride=1),
                nn.MaxPool2d(kernel_size=2, stride=2)
                )
            self.layer_3 = nn.Conv2d(in_channels=256, out_channels=512,
                                     kernel_size=3, stride=1, padding=1)
            self.layer_4 = nn.Conv2d(in_channels=512, out_channels=1024,
                                     kernel_size=3, stride=1, padding=1)
            self.layer_5 = nn.Conv2d(in_channels=1024, out_channels=1024,
                                     kernel_size=3, stride=1, padding=1)
            # self.layer_5_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
            #self-attention method proposed in self-attention gan Zhang et al.
            self.frame_fx = nn.Conv2d(in_channels=1024, out_channels=1024,
                                 kernel_size=1, stride=1)
            self.frame_gx = nn.Conv2d(in_channels=1024, out_channels=1024,
                                 kernel_size=1, stride=1)
            self.frame_hx = nn.Conv2d(in_channels=1024, out_channels=1024,
                                 kernel_size=1, stride=1)
            self.frame_vx = nn.Conv2d(in_channels=1024, out_channels=1024,
                                 kernel_size=1, stride=1)

    def forward(self, frame: np.ndarray):
        batch = frame.size(0)
        frame = self.layer_1(frame)
        frame = self.layer_2(frame)
        frame = self.layer_3(frame)
        frame = self.layer_4(frame)
        frame = self.layer_5(frame)
        # self-attention gan
        frame_fx = self.frame_fx(frame)
        frame_gx = self.frame_gx(frame)
        frame_hx = self.frame_hx(frame)
        frame = nn.Softmax2d(frame_fx.transpose_(2, 1).matmul(frame_gx), dim=1)
        frame = frame_hx.matmul(frame)
        return self.frame_vx(frame).view(-1, 1024, 12, 12)

##################################################################################
#                               Fusion modules
# ________________________________________________________________________________
class Fusion(nn.Module):
    """Feature Pool and Fusion module"""
    def __init__(self, pool_dim=256, hidden_size=128):
        super(Fusion, self).__init__()
        self.hidden_size = hidden_size
        self.pool_dim = pool_dim
        # can be removed
        self.linear = nn.Sequential(nn.Linear(147_456, pool_dim),
                                    nn.MaxPool(kernel_size=2, stride=2),
                                    nn.ReLU())
        self.fuse = nn.LSTM(input_size=167, hidden_size=256)

    def initiate_hidden(self, batch, sequence_len):
        return (
            torch.zeros(sequence_len, batch, self.hidden_size),
            torch.zeros(sequence_len, batch, self.hidden_size)
        )

    def get_noise(self, shape, noise_type="gaussian"):
        if noise_type == 'gaussian':
            return torch.randn(*shape)
        elif noise_type == 'uniform':
            return torch.rand(*shape).sub_(0.5).mul_(2.0)
        raise ValueError('Unrecognized noise type "%s"' % noise_type)

    def forward(self, real_history, rel_history, pool, context_feature, agent_idx):
        """receives the whole feature matrix as input (max_agents * 56)
        28 is for 2 seconds input (each second is 2 frame, each frame has 14
        features)
        args:
            real_history: a matrix containing unmodified past locations (100,7)
            pool: modified past locations (100, 32)
            context_feature: tensor of size=(1024, 12, 12)=147456
            i: desired agent number to forecast future, if -1, it will predict all the agents at the same time
        """
        batch = pool.size(1)
        sequence_length = pool.size(0)
        if agent_idx == -1:
            agent = pool
            agent_rel = rel_history[:, :7]
            context_feature = context_feature.view(-1) # 147456 digits
            context_feature = context_feature.repeat(agent.size(0), 1)
            noise = self.get_noise((agent.size(0), 5))
        else:
            agent = pool[agent_idx] # a vector of size 32
            agent_rel = rel_history[agent_idx][:7] # vector of size 7
            real_history = real_history[agent_idx]
            noise = self.get_noise((5,))

        # agent = self.linear(agent)
        agent = torch.cat((agent_rel, agent), 1) # vector of 7 + 32 = 39
        context_feature = self.linear(context_feature) # vector of size 128

        cat_features = torch.cat((context_feature, agent), 1) # 167
        cat_features = cat_features.view(-1, batch, 167)

        fused_features_hidden, _ = self.fuse(cat_features,
                                                self.initiate_hidden(batch, sequence_length))

        # dim: 5 + 3 + 256 = 264
        fused_features_hidden = torch.cat(
            (noise, real_history, fused_features_hidden),
            1)
        return fused_features_hidden

##################################################################################
#                               Generator
# ________________________________________________________________________________
class Generator(nn.Module):
    """Trajectory generator"""
    def __init__(self, input_size=264, hidden_size=32, num_layers=1):
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.decoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers)
        self.hidden2pos = nn.Linear(hidden_size, 70)
        # self.hidden2pos = nn.Linear(h_dim, 3)

    def initiate_hidden(self, traj):
        return torch.zeros(traj.size(0), traj.size(1), self.hidden_size)

    def forward(self, traj, real_history, hidden_state):
        # batch = traj.size(0)
        hidden = (hidden_state, self.initiate_hidden(traj))
        # traj = traj.view(batch, traj.size(1), 264) # traj.size(1)=264
        traj, _ = self.decoder(traj, hidden)
        traj = traj.tolist()
        traj_pred = []
        for agent in traj:
            traj_pred.append(self.hidden2pos(
                torch.tensor(agent.view(batch, 70))
                ))
        # return a vector of size (seq_len, batch, input_size)
        return torch.cat(traj_pred, dim=0)


##################################################################################
#                               GAN Discriminator
# ________________________________________________________________________________

class TrajectoryDiscriminator(nn.Module):
    """GAN Discriminator"""
    def __init__(self, dropout=0):
        super(TrajectoryDiscriminator, self).__init__()
        self.encoder = Encoder(input_size=14, embedding_dimension=64, hidden_size=16, num_layers=1)
        self.contextual_features = ContextualFeatures()
        self.fusion = Fusion(pool_dim=64, hidden_size=128, batch_size=1)
        self.mpl = nn.Sequential(
            nn.Linear(264, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 1),
            nn.ReLU()
            )

    def forward(self, traj, image, real_history, agent_idx):
        traj = self.encoder(traj)
        image = self.contextual_features(image)
        encoded_traj = self.fusion(real_history, traj, image, agent_idx)
        score = self.mlp(encoded_traj)
        return score


##################################################################################
#                              Testing the modules
# ________________________________________________________________________________
path_for_json_data = "data/exported_json_data/"
dataset_dir = "data/nuScene-mini"

# This is the number of the layers which the LSTM cell contains ----> hyper
num_layers = 10

# This is the feature vector size ---> sure
input_size = 14

# Number of the frames of the observed trajectories ----> sure
sequence_length = 80

# The Batch size ------> hyperparameter
Batch_size = 128

# The number of the cells on each LSTM layer (This parameter acts like the width of the network) --> hyperparameter
hidden_layers = 2

# The size that the input features will be embedded to using a simple mlp    -----> sure
embedding_dimensions = 32

if __name__ == '__main__':
    pass
    # Still have some difficulty about the ouput of the previous module
