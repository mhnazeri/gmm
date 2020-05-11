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
        # 3 modules that comprise the fusion
        self.fuse_traj = nn.LSTM(input_size=92, hidden_size=hidden_size)
        self.fuse_context = nn.LSTM(input_size=144, hidden_size=hidden_size)
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

    def forward(self, real_past, rel_past, pool, context_feature, agent_idx=-1):
        """receives the whole feature matrix as input (num_agents * 52)
        22 is for 2 seconds input (each second is 2 frame, each frame has 13
        features)
        args:
            real_past: a matrix containing unmodified past locations (num_agents,7)
            pool: modified past locations (num_agents, 64)
            context_feature: tensor of size=(1024, 144)
            i: desired agent number to forecast future, if -1, it will predict all the agents at the same time
        """
        batch = pool.size(1)
        sequence_length = pool.size(0)
        if agent_idx == -1:
            agent = pool
            noise = self.get_noise((agent.size(0), 5))
        else:
            agent = pool[agent_idx] # a vector of size 64
            rel_past = rel_past[agent_idx] # vector of size 7
            real_past = real_past[agent_idx]
            noise = self.get_noise((5,))

        # concat relative_past with encoded features (vector of 28 + 64 = 92)
        agent = torch.cat((rel_past, agent), 1)
        agent = agent.view(-1, batch, 92)
        # feed lidar stream to lstm for fusion
        _, traj_hidden, _ = self.fuse_traj(agent,
                                                self.initiate_hidden(batch, sequence_length))

        # feed camera stream to lstm for fusion (1024, batch, 144)
        context_feature = torch.transpose(context_feature, 1, 0)
        _, context_hidden, _ = self.fuse_context(context_feature,
                                                self.initiate_hidden(batch, 1024))

        # fusing the hidden state of two streams together with an mlp
        fused_features = torch.cat((traj_hidden, context_hidden), 1)
        fused_features = self.fuse(fused_features)
        # concat all features (noise, real_past, fused_features) together to feed to the generator
        # dim: 5 + 28 + 256 = 289
        fused_features_hidden = torch.cat(
            (noise, real_past, fused_features),
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
                 seq_len: int = 10,
                 embedder = None,
                 anti_embedder = None,
                 embedding_dim: int = 7,
                 input_size: int = 13,
                 hidden_dim: int = 64,
                 num_layers: int = 1,
                 dropout:float = 0.0,
                 decoder_mlp_structure: list = [128],
                 decoder_mlp_activation: str = "Relu"
                 ):
        """
        The Decoder is responsible to forecast the whole sequence length of the future trajectories
        :param embedder: the cae encoder module passed to be used as the encoder
        :param anti_embdder: if not None, the cae decoder will be used for converting hidden state to positions
        :param embedding_dim: the embedding dimension which should be the same as the cae latent dim
        :param input_size: the inout size of the model
        :param hidden_dim: the hidden dimension of the LSTM module
        :param decoder_mlp_structure: the structure of mlp used to convert hidden to predictions
        """
        super(Decoder, self).__init__()

        self._seq_len = seq_len
        self._embedding_dim = embedding_dim
        self._num_layers = num_layers
        self._use_cae_decoder = False

        if embedder is not None:
            # This is supposed to be the CAE encoder
            self.embedder = embedder
        else:
            self.embedder = nn.Linear(input_size, embedding_dim)

        if anti_embedder is None:
            hidden2pos_structure = [hidden_dim] + decoder_mlp_structure + [input_size]
            self.hidden2pos = make_mlp(
                layers=hidden2pos_structure,
                activation=decoder_mlp_activation,
                dropout=dropout,
                batch_normalization=False
            )

        else:
            self._use_cae_decoder = True
            self.hidden2latent = nn.Linear(hidden_dim, embedding_dim)
            self.hidden2pos = anti_embedder

        self.decoder = nn.LSTM(embedding_dim, hidden_dim,
                           num_layers, dropout=dropout)

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

            if not self._use_cae_decoder:
                curr_rel = self.hidden2pos(decoder_output[0])

            else:
                anti_embedder_input = self.hidden2latent(decoder_output[0])
                curr_rel = self.hidden2pos(anti_embedder_input)

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
                 anti_embedder = None,
                 embedding_dim: int = 7,
                 encoder_h_dim: int = 64,
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
            anti_embedder=anti_embedder,
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

        # this module is used at the begining to convert the input features
        self.encoder = Encoder(
            embedder=embedder,
            input_size=input_size,
            embedding_dim=embedding_dim,
            encoder_h_dim=encoder_h_dim,
            dropout=dropout,
            num_layers=num_layers)

        # Use his section to define any other module to be used for pooling or fusion
        # Todo: add the other required submodules

    def forward(self, obs_traj, obs_traj_rel):
        """

        :param obs_traj: shape (obs_length, batch, inputs_size)
        :param obs_traj_rel: shape (obs_length, batch, inputs_size)
        :return:
        """
        batch_size = obs_traj.shape[1]

        state_tuple = (
            torch.zeros(self._num_layers, batch_size, self._hidden_size),
            torch.zeros(self._num_layers, batch_size, self._hidden_size)
        )

        """
        Do what ever you want with the state_tuple[0] which stands for the hidden_state to be used
            in prediction.
        state_tuple[0] = ...
        """

        last_features = obs_traj[-1]
        last_features_rel = obs_traj_rel[-1]
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
        :param embedder: if not None, the cae_encoder will be used for embedding otherwise a linear_layer
        :param embedding_dim: The dimension that the input data will be converted to
        (if embedder is not None then it should be the same size as the cae's latent dimension)
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
