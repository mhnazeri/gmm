""" This file contains the implementation of the main models including:
    1. Encoder(Boh)
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
    def __init__(self, input_size: int = 14, embedding_dimension: int = 64, hidden_size: int = 16, num_layers:int = 1):
        """
        :param input_size: The size of each vector representing an agent in a frame containing all the features
        :param embedding_dimension: The size in which the input features will be embedded to
        :param hidden_size: The size of the hidden dimension of LSTM layer. As bigger the hidden dimension as higher the
            capability of the encoder to model longer sequences
        :param num_layers: The depth of the LSTM networks, Set to one in SGAN original code
        """
        super(Encoder, self).__init__()
        self.num_layers = 32
        self.hidden_size = hidden_size
        self.embedder = nn.Linear(input_size, embedding_dimension)
        self.embedding_dimension = embedding_dimension
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

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

        batch_size = inputs.size(1)
        embed = self.embedder(inputs.view(-1, 2))
        embed = embed.view(-1, batch_size, self.embedding_dimension)

        states = self.initiate_hidden(batch_size)
        _, hidden_state, _ = self.lstm(embed, states)

        return hidden_state


##################################################################################
#                        Contextual Feature Extractor
# ________________________________________________________________________________
class Contextual_Features(nn.Module):
    """Extract contextual features from the environment
        Networks that can be used for feature extraction are:
            overfeat: returned matrix is 1024*12*12
    """
    def __init__(self, model_arch: str="overfeat"):
        super(Contextual_Features, self).__init__()
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
                                 kernel_size=3, stride=1, padding=1)
            self.frame_gx = nn.Conv2d(in_channels=1024, out_channels=1024,
                                 kernel_size=3, stride=1, padding=1)
            self.frame_hx = nn.Conv2d(in_channels=1024, out_channels=1024,
                                 kernel_size=3, stride=1, padding=1)
            self.frame_vx = nn.Conv2d(in_channels=1024, out_channels=1024,
                                 kernel_size=3, stride=1, padding=1)


    def forward(self, frame_1: np.ndarray, frame_2: np.ndarray):
        frame = self.background_motion(frame_1, frame_2)
        frame = self.layer_1(frame)
        frame = self.layer_2(frame)
        frame = self.layer_3(frame)
        frame = self.layer_4(frame)
        frame = self.layer_5(frame)
        if frame.shape != (1024, 12, 12):
            frame = frame.view(1024, 12, 12)
        frame_fx = self.frame_fx(frame)
        frame_gx = self.frame_gx(frame)
        frame_hx = self.frame_hx(frame)
        frame = nn.Softmax2d(frame_fx.transpose_(2, 1).matmul(frame_gx), dim=1)
        frame = frame_hx.matmul(frame)
        return self.frame_vx(frame).view(1024, 12, 12)

    def background_motion(self, frame_1: np.ndarray, frame_2:np.ndarray) -> np.ndarray:
        """returns background motion between two consequtive frames"""
        return frame_2 - frame_1

##################################################################################
#                               Fusion modules
# ________________________________________________________________________________
class Fusion(nn.Module):
    """Feature Pool and Fusion module"""
    def __init__(self, pool_dim=64, hidden_size=128, batch_size=1):
        super(Fusion, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.pool_dim = pool_dim
        self.linear = nn.Linear(nn.Linear(56, pool_dim),
                                nn.MaxPool(kernel_size=2, stride=2))
        self.fuse = nn.LSTM(input_size=147619, hidden_size=256)

    def initiate_hidden(self):
        return (
            torch.zeros(1, self.batch_size, self.hidden_size),
            torch.zeros(1, self.batch_size, self.hidden_size)
        )

    def get_noise(self, shape, noise_type="gaussian"):
        if noise_type == 'gaussian':
            return torch.randn(*shape).cuda()
        elif noise_type == 'uniform':
            return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
        raise ValueError('Unrecognized noise type "%s"' % noise_type)

    def rel_distance(agent_1, agent_2):
        return torch.sqrt(((agent_1 - agent_2) ** 2).sum())

    def forward(self, real_history, pool, context_feature, i):
        """receives the whole feature matrix as input (max_agents * 56)
        28 is for 2 seconds input (each second is 2 frame, each frame has 14
        features)
        args:
            real_history: a matrix containing unmodified past locations
            pool: modified past locations
            context_feature: tensor of size=(1024, 12, 12)=147456
            i: desired agent number to forecast future
        """
        agent = pool[i] # a vector of size 56
        distances = []
        for j in range(len(pool)):
            if j != i:
                distances.append(rel_distance(agent, pool[j]))

        agent = self.linear(agent)
        agent = torch.cat((distances, agent), 1) # vector of 99 + 64 = 163
        context_feature = context_feature.view(-1) # 147456 digits
        cat_features = torch.cat((context_feature, agent), 0) # 147619
        cat_features = cat_features.view(-1, self.batch_size, self.pool_dim)

        _, fused_features_hidden, _ = self.fuse(cat_features,
                                                self.initiate_hidden())
        # dim: 2 + 6 + 256 = 264
        fused_features_hidden = torch.cat(
            (self.get_noise((2,)), real_history[i], fused_features_hidden),
            1)
        return fused_features_hidden

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
