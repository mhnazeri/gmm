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
class Encoder(nn.Module):
    def __init__(self,
                 embedder = None,
                 input_size: int = 13,
                 embedding_dim: int = 7,
                 encoder_h_dim: int = 64,
                 dropout: float = 0.0,
                 num_layers: int = 1):
        super(Encoder, self).__init__()

        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = nn.Linear(input_size, embedding_dim)

        self._num_layers = num_layers
        self._input_size = input_size
        self._embedding_dim = embedding_dim
        self._hidden_size = encoder_h_dim
        self.encoder = nn.LSTM(embedding_dim, encoder_h_dim, num_layers, dropout=dropout)

    def initiate_hidden(self, batch_size):
        return (
            torch.zeros(self._num_layers, batch_size, self._hidden_size),
            torch.zeros(self._num_layers, batch_size, self._hidden_size)
        )

    def forward(self, inputs):
        """
        :param inputs: A tensor of the shape (sequence_length, batch, input_size)
        :return: A tensor of the shape (self.num_layers, batch, self.encoder_h_dim)
        """
        sequence_length = inputs.shape[0]
        batch = inputs.shape[1]

        # Check the integrity of the shapes
        logger.debug("The size of the inputs: " + str(inputs.size()))

        states = self.initiate_hidden(self.batch_size)

        # Embed the input data to the desired dimension using a cae encoder or a linear layer
        embedder_inputs = inputs.view(-1, self._input_size)
        embedded_features = self.embedder(embedder_inputs)

        # Return the shape of the inputs to the desired shapes for lstm layer to be encoded
        lstm_inputs = embedded_features.view(sequence_length, batch, self._embedding_dim)
        _, states = self.lstm(lstm_inputs, states)
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
#       1. implement the pooling every timestep mechanism
#       2. Not sure about adding for all the features

class Decoder(nn.Module):
    def __init__(self,
                 seq_len:int = 10,
                 embedder=None,
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
        :param embedder: the cae encoder module passed to be used as the encoder
        :param embedding_dim: the embedding dimension which should be the same as the cae latent dim
        :param input_size: the inout size of the model
        :param hidden_dim: the hidden dimension of the LSTM module
        :param decoder_mlp_structure: the structure of mlp used to convert hidden to predictions
        """
        super(Decoder, self).__init__()

        self._seq_len = seq_len
        self._embedding_dim = embedding_dim
        self._num_layers = num_layers

        if embedder is not None:
            # This is supposed to be the CAE encoder
            self.embedder = embedder
        else:
            self.embedder = nn.Linear(input_size, embedding_dim)

        self.decoder = nn.LSTM(embedding_dim, hidden_dim,
                               num_layers, dropout=dropout)

        hidden2pos_structure = [hidden_dim] + decoder_mlp_structure + [input_size]

        self.hidden2pos = make_mlp(
            layers=hidden2pos_structure,
            activation=decoder_mlp_activation,
            dropout=dropout,
            batch_normalization=False
        )

    def forward(self, last_features, last_features_rel, state_tuple):
        """

        :param state_tuple: A tuple of two states as the initial state of the LSTM where the first item
            contains the required info from the past, both in the shape (num_layers, batch_size, embedding_dim)
        :return: First item is of the shape (sequence_length, batch, input_size)
            while the second item is of the shape (num_layers, batch, embedding_dim)
        """
        predicted_traj = []
        batch = last_features.shape[0]
        decoder_input = self.embedder(last_features_rel).unsqueeze(0)

        for _ in range(self._seq_len):
            decoder_output, state_tuple = self.decoder(decoder_input, state_tuple)
            curr_rel = self.hidden2pos(decoder_output)

            # Todo: to add the every timestep mechanism here

            # not sure about just adding the other features or not
            current_features = last_features_rel + curr_rel
            predicted_traj.append(current_features)
            decoder_input = self.embedder(curr_rel).view(self._num_layers, batch, self._embedding_dim)

        predicted_traj = torch.stack(predicted_traj, dim=0)
        return predicted_traj, state_tuple[0]


##################################################################################
#                               Generator
# ________________________________________________________________________________
class TrajectoryGenerator(nn.Module):
    """Trajectory generator"""
    def __init__(self,
                 embedder = None,
                 embedding_dim: int = 7,
                 seq_length: int = 10,
                 input_size: int = 13,
                 decoder_hidden_dim: int = 64,
                 decoder_mlp_structure: list = [128],
                 decoder_mlp_activation: str = "Relu",
                 dropout: float = 0.0,
                 num_layers: int = 1):

        super(TrajectoryGenerator, self).__init__()

        # This module will be used to generate the future trajectory
        self.decoder = Decoder(
            embedder=embedder,
            embedding_dim=embedding_dim,
            seq_len=seq_length,
            input_size=input_size,
            hidden_dim=decoder_hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            decoder_mlp_activation=decoder_mlp_activation,
            decoder_mlp_structure=decoder_mlp_structure,
        )

        # Use his section to define any other module to be used for pooling or fusion


    def forward(self, last_features, last_features_rel):
        """
        :param last_features: The features of the final timestep of the observed sequence
            of the shape (batch, input_size)
        :param last_features_rel: The relative features of the final timestep of the observed sequence
            of the shape (batch, input_size)
        :return:
        """
        batch_size = last_features.shape[0]

        state_tuple = (
            torch.zeros(self._num_layers, batch_size, self._hidden_size),
            torch.zeros(self._num_layers, batch_size, self._hidden_size)
        )

        """
        Do what ever you want with the state_tuple[0] which stands for the hidden_state to be used
            in prediction.
        state_tuple[0] = ...
        """

        predicted_traj, _ = self.decoder(last_features, last_features_rel, state_tuple)

        return predicted_traj

##################################################################################
#                               GAN Discriminator
# ________________________________________________________________________________
class TrajectoryDiscriminator(nn.Module):
    def __init__(self,
                 embedder = None,
                 input_size: int = 13,
                 embedding_dim: int = 7,
                 num_layers: int = 1,
                 encoder_h_dim:int = 64,
                 mlp_structure: list = [64, 128, 1],
                 mlp_activation: str = "Relu",
                 batch_normalization: bool = True,
                 dropout: float = 0.0):
        """
        :param embedder:
        :param input_size:
        :param embedding_dim: The dimension that the input data will be converted to
        (if embedder is not None then it should be the same size as the cae's latent dimension)
        :param num_layers:
        :param encoder_h_dim: The hidden dimension of the encoder
        :param mlp_structure: A list defining the structure of the mlp in discriminator,
        """

        super(TrajectoryDiscriminator, self).__init__()

        self.classifier = make_mlp(layers=[encoder_h_dim] + mlp_structure,
                                   activation=mlp_activation,
                                   dropout=dropout,
                                   batch_normalization=batch_normalization)


        self.encoder = Encoder(
            embedder=embedder,
            input_size=input_size,
            embedding_dim=embedding_dim,
            encoder_h_dim=encoder_h_dim,
            dropout=dropout,
            num_layers=num_layers
        )

    def forward(self, traj):
        encoded_features = self.encoder(traj)
        scores = self.classifier(encoded_features[0])
        return scores

if __name__ == '__main__':
    pass
