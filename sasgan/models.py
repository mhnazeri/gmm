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


##################################################################################
#                                    Encoder
# ________________________________________________________________________________
class Generator_Encoder(nn.Module):
    def __init__(self,
                 input_size: int = 7,
                 hidden_size: int = 32,
                 num_layers: int = 1):
        """
        :param input_size: The size of each vector representing an agent in a frame containing all the features
        :param hidden_size: The size of the hidden dimension of LSTM layer. As bigger the hidden dimension as higher the
            capability of the encoder to model longer sequences
        :param num_layers: The depth of the LSTM networks, Set to one in SGAN original code
        """
        super(Generator_Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
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

        # batch_size = inputs.size(1)
        # embed = self.embedder(inputs.view(-1, 2))
        inputs = inputs.view(self.batch_size, 100, -1)

        states = self.initiate_hidden(self.batch_size)
        _, states = self.lstm(inputs, states)

        return states[0]


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
        return self.frame_vx(frame).view(-1, 1024, 12 * 12)

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
        self.linear = nn.Sequential(nn.Linear(147_456, 147_456 / 2),
                                    nn.MaxPool(kernel_size=2, stride=2),
                                    nn.ReLU(),
                                    nn.Linear(147_456 / 2, pool_dim),
                                    nn.MaxPool(kernel_size=2, stride=2),
                                    nn.ReLU())

        self.fuse_traj = nn.LSTM(input_size=39, hidden_size=128)
        self.fuse_context = nn.LSTM(input_size=144, hidden_size=128)

        self.fuse = nn.Sequential(nn.Linear(256, 256),
                                    nn.MaxPool(kernel_size=2, stride=2),
                                    nn.ReLU())

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

    def forward(self, real_history, rel_history, pool, context_feature, agent_idx=-1):
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
            # context_feature = context_feature.view(-1) # 147456 digits
            # context_feature = context_feature.repeat(agent.size(0), 1)
            noise = self.get_noise((agent.size(0), 5))
        else:
            agent = pool[agent_idx] # a vector of size 32
            agent_rel = rel_history[agent_idx][:7] # vector of size 7
            real_history = real_history[agent_idx]
            noise = self.get_noise((5,))

        # agent = self.linear(agent)
        agent = torch.cat((agent_rel, agent), 1) # vector of 7 + 32 = 39
        # context_feature = self.linear(context_feature) # vector of size 128

        # cat_features = torch.cat((context_feature, agent), 1) # 167
        agent = agent.view(-1, batch, 39)
        context_feature = torch.transpose(context_feature, 1, 0) # (1024, batch, 144)

        _, traj_hidden, _ = self.fuse_traj(agent,
                                                self.initiate_hidden(batch, sequence_length))

        _, context_hidden, _ = self.fuse_context(context_feature,
                                                self.initiate_hidden(batch, 1024))

        fused_features = torch.cat((traj_hidden, context_hidden), 1)
        fused_features = self.fuse(fused_features)
        # dim: 5 + 3 + 256 = 264
        fused_features_hidden = torch.cat(
            (noise, real_history, fused_features),
            1)
        return fused_features_hidden

##################################################################################
#                                    Decoder
# ________________________________________________________________________________
# Todo: for later
#      1. implement the pooling every timestep mechanism


class Decoder(nn.Module):
    def __init__(self,
                 seq_len:int = 10,
                 encoder=None,
                 embedding_dim:int = 7,
                 input_size:int = 13,
                 hidden_dim:int = 64,
                 num_layers:int = 1,
                 dropout:float = 0.0,
                 decoder_mlp_structure:list = [128],
                 decoder_mlp_activation:str = "Relu"
                 ):
        """
        The Decoder is responsible to forecast the whole sequence length of the future trajectories
        :param encoder: the cae encoder module passed to be used as the encoder
        :param embedding_dim: the embedding dimension which should be the same as the dae latent dim
        :param input_size: the inout size of the model
        :param hidden_dim: the hidden dimension of the LSTM module
        :param decoder_mlp_structure: the structure of mlp used to convert hidden to predictions
        """
        super(Decoder, self).__init__()

        self._seq_len = seq_len


        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = nn.Linear(input_size, embedding_dim)

        self.decoder = nn.LSTM(embedding_dim, hidden_dim,
                               num_layers, dropout=dropout)

        hidden2pos_struecture = [hidden_dim] + decoder_mlp_structure + [input_size]

        self.hidden2pos = make_mlp(
            layers=hidden2pos_struecture,
            activation=decoder_mlp_activation,
            dropout=dropout,
            batch_normalization=False
        )

    def forward(self, traj, traj_rel, last_pos):

        for _ in self._seq_len:
            pass




##################################################################################
#                               Generator
# ________________________________________________________________________________
class TrajectoryGenerator(nn.Module):
    """Trajectory generator"""
    def __init__(self,
                 input_size:int =264,
                 hidden_size=32,
                 num_layers=1):
        super(TrajectoryGenerator, self).__init__()

    def forward(self, traj, hidden_state):
        pass

##################################################################################
#                               GAN Discriminator
# ________________________________________________________________________________
class _Discriminator_Encoder(nn.Module):
    def __init__(self,
                 input_size: int = 13,
                 embedding_dimension: int = 64,
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 dropout: float = 0.0):
        """
        :param input_size: The dimension of the input data (number of features)
        :param hidden_size: The hidden size of LSTM module
        :param num_layers: Number of the layers for LSTM
        """
        super(_Discriminator_Encoder, self).__init__()

        self.linear = nn.Linear(input_size, embedding_dimension)
        self.encoder = nn.LSTM(embedding_dimension, hidden_size, num_layers, dropout=dropout)

    def initiate_hidden(self, batch_size):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
            torch.zeros(self.num_layers, batch_size, self.hidden_size)
        )

    def forward(self, trajectories):
        """
        :param trajectories: A Tensor of the shape (seq_length, batch, feature_size)
        :return: A Tensor of the shape (num_layers, batch, hidden_size)
        """
        batch_size = trajectories.shape[1]
        converted_trajectories = self.linear(trajectories)
        _, states = self.encoder(converted_trajectories, self.initiate_hidden(batch_size))
        return states[0]


class TrajectoryDiscriminator(nn.Module):
    def __init__(self,
                 input_size: int = 13,
                 embedding_dimension: int = 64,
                 encoder_num_layers: int = 1,
                 mlp_structure: list = [64, 128, 1],
                 mlp_activation: str = "Relu",
                 batch_normalization: bool = True,
                 dropout: float = 0.0):
        """
        Because the input for the discriminator is not supposed to be the output of the cae, then the implemented
            Encoder can not be used in this submodule, we may need to include the encoder inside discriminator manually

        :param embedding_dimension: The dimension the input data will be converted to,
        :param mlp_structure: A list defining the structure of the mlp in discriminator,
            Note: The first item in this list, defines the shape of the encoder's hidden state
        """
        super(TrajectoryDiscriminator, self).__init__()

        self.classifier = make_mlp(layers=mlp_structure,
                                   activation=mlp_activation,
                                   dropout=dropout,
                                   batch_normalization=batch_normalization)


        self.encoder = _Discriminator_Encoder(
            input_size=input_size,
            embedding_dimension= embedding_dimension,
            hidden_size=mlp_structure[0],
            num_layers=encoder_num_layers,
            dropout=dropout
        )

    def forward(self, traj):
        out = self.encoder(traj)
        scores = self.classifier(out)
        return scores

if __name__ == '__main__':
    pass
    # Still have some difficulty about the ouput of the previous module
