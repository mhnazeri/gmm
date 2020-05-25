import torch
import torch.nn as nn
import numpy as np
import sys
import os
from utils import *
import logging
from data.loader import CAEDataset
import copy

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

    def forward(self, inputs, state_tuple=None):
        """
        :param inputs: A tensor of the shape (sequence_length, batch, input_size)
        :return: A tensor of the shape (self.num_layers, batch, self.encoder_h_dim)
        """
        sequence_length = inputs.shape[0]
        batch_size = inputs.shape[1]


        # Check the integrity of the shapes
        logger.debug("The size of the inputs: " + str(inputs.size()))

        if state_tuple is None:
            states = self.initiate_hidden(batch_size)
        else:
            states = state_tuple

        # Embed the input data to the desired dimension using a cae encoder or a linear layer
        embedder_inputs = inputs.view(-1, self._input_size)
        embedded_features = self.embedder(embedder_inputs)

        # Return the shape of the inputs to the desired shapes for lstm layer to be encoded
        lstm_inputs = embedded_features.view(sequence_length, batch_size, self._embedding_dim)
        _, states = self.lstm(lstm_inputs, states)
        return states


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
            self.net = nn.Sequential(
                nn.Conv2d(in_channels=1, kernel_size=11, stride=4,
                          out_channels=96),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=96, out_channels=256,
                          kernel_size=5, stride=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=512,
                          kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=1024,
                          kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=1024, out_channels=1024,
                          kernel_size=3, stride=1, padding=1)
                                     )

        self.softmax = nn.Softmax2d()
            # self.layer_1 = nn.Sequential(
            #     nn.Conv2d(in_channels=1, kernel_size=11, stride=4,
            #                          out_channels=96),
            #     nn.MaxPool2d(kernel_size=2, stride=2)
            #     )
            # self.layer_1 = nn.Conv2d(in_channels=1, kernel_size=11, stride=4,
            #                          out_channels=96)
            # self.layer_1_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
            # self.layer_2 = nn.Sequential(
            #     nn.Conv2d(in_channels=96, out_channels=256,
            #                          kernel_size=5, stride=1),
            #     nn.MaxPool2d(kernel_size=2, stride=2)
            #     )
            # self.layer_3 = nn.Conv2d(in_channels=256, out_channels=512,
            #                          kernel_size=3, stride=1, padding=1)
            # self.layer_4 = nn.Conv2d(in_channels=512, out_channels=1024,
            #                          kernel_size=3, stride=1, padding=1)
            # self.layer_5 = nn.Conv2d(in_channels=1024, out_channels=1024,
            #                          kernel_size=3, stride=1, padding=1)
            # self.layer_5_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        # self-attention method proposed in self-attention gan Zhang et al.
        self.frame_fx = nn.Conv2d(in_channels=1024, out_channels=1024,
                             kernel_size=1, stride=1)
        self.frame_gx = nn.Conv2d(in_channels=1024, out_channels=1024,
                             kernel_size=1, stride=1)
        self.frame_hx = nn.Conv2d(in_channels=1024, out_channels=1024,
                             kernel_size=1, stride=1)
        self.frame_vx = nn.Conv2d(in_channels=1024, out_channels=1024,
                             kernel_size=1, stride=1)

    def forward(self, frame: torch.Tensor):
        # forward pass through overfeat
        frame = self.net(frame)
        # frame = self.layer_1(frame)
        # frame = self.layer_2(frame)
        # frame = self.layer_3(frame)
        # frame = self.layer_4(frame)
        # frame = self.layer_5(frame)
        # self-attention gan
        frame_fx = self.frame_fx(frame)
        frame_gx = self.frame_gx(frame)
        frame_hx = self.frame_hx(frame)
        frame = self.softmax(frame_fx.transpose_(3, 2).matmul(frame_gx))
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
                                    nn.MaxPool2d(kernel_size=2, stride=2),
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
#       1. implement the pooling every timestep mechanism ---> Done
#       2. Not sure about adding for all the features ----> Done

class Decoder(nn.Module):
    def __init__(self,
                 fusion_length,
                 output_size: int = 7,
                 hidden_dim: int = 64,
                 num_layers: int = 1,
                 dropout:float = 0.0,
                 decoder_mlp_structure: list = [128],
                 decoder_mlp_activation: str = "Relu"
                 ):
        """
        The Decoder is responsible to forecast the whole sequence length of the future trajectories
        :param output_size: the size of the prediction, the default stands for 3(translation) + 4(rotation)
        :param hidden_dim: the hidden dimension of the LSTM module
        :param decoder_mlp_structure: the structure of mlp used to convert hidden to predictions
        """
        super(Decoder, self).__init__()

        self._num_layers = num_layers
        self._output_size = output_size

        hidden2pos_structure = [hidden_dim] + decoder_mlp_structure + [output_size]
        self.hidden2pos = make_mlp(
            layers=hidden2pos_structure,
            activation=decoder_mlp_activation,
            dropout=dropout,
            batch_normalization=False
        )

        self.decoder = nn.LSTM(fusion_length, hidden_dim,
                           num_layers, dropout=dropout)

    def forward(self, last_features, last_features_rel , fused_features, state_tuple):
        """
        :param state_tuple: A tuple of two states as the initial state of the LSTM where the first item
            contains the required info from the past, both in the shape (num_layers, batch_size, embedding_dim)
        """
        decoder_input = self.embedder(fused_features).unsqueeze(0)
        decoder_output, state_tuple = self.decoder(decoder_input, state_tuple)
        curr_features_rel = self.hidden2pos(decoder_output[0])

        predicted_traj = last_features.clone()
        predicted_traj_rel = last_features_rel.clone()
        predicted_traj[:, :self._output_size] += curr_features_rel
        predicted_traj_rel[:, :self._output_size] = curr_features_rel

        return predicted_traj, predicted_traj_rel, state_tuple


##################################################################################
#                               Generator
# ________________________________________________________________________________
class GenerationUnit(nn.Module):
    """This class is responsible for generating just one frame"""
    def __init__(self, embedder, embedding_dim, encoder_h_dim, decoder_h_dim, input_size, output_size,
                 decoder_mlp_structure, decoder_mlp_activation, dropout, num_layers, fusion_pool_dim,
                 fusion_hidden_dim, fused_vector_length: int = 289):

        super(GenerationUnit, self).__init__()

        self.decoder = Decoder(
            fusion_length=fused_vector_length,
            output_size=output_size,
            hidden_dim=decoder_h_dim,
            num_layers=num_layers,
            dropout=dropout,
            decoder_mlp_activation=decoder_mlp_activation,
            decoder_mlp_structure=decoder_mlp_structure,
        )

        # This module is used at the beginning to convert the input features
        self.encoder = Encoder(
            embedder=embedder,
            input_size=input_size,
            embedding_dim=embedding_dim,
            encoder_h_dim=encoder_h_dim,
            dropout=dropout,
            num_layers=num_layers)

        self.fusion = Fusion(pool_dim=fusion_pool_dim,
                             hidden_size=fusion_hidden_dim)

        encoder_decoder_mlp_structure = [encoder_h_dim, decoder_h_dim]
        self.encoder_decoder_h = make_mlp(
            layers=encoder_decoder_mlp_structure,
            activation="Relu",
            batch_normalization=True,
        )

        self._num_layers = num_layers
        self._decoder_h_dim = decoder_h_dim
        self._encoder_h_dim =encoder_h_dim

    def forward(self, obs, obs_rel, context_features):
        """
        :param obs: should be of the shape (seq_length, batch, input_size)
        :param obs_rel: should be of the shape (seq_length, batch, input_size)
        :param context_features: should be of the shape (batch, 1024, 144)
        :return: two items:
            1. predicted_traj: the next absolute features for the observed trajectory: (batch, input_size)
            2. predicted_traj_rel: the next relative features for the observed trajectory: (batch, input_size)
        """
        batch_size = obs_rel.shape[1]

        states = self.encoder(obs_rel)
        fused_features = self.fusion(obs, obs_rel, states[0], context_features)

        decoder_h = self.encoder_decoder_h(states[0].view(-1, self._encoder_h_dim))
        decoder_h = decoder_h.view(self._num_layers,  batch_size, self._decoder_h_dim)
        decoder_c = torch.zeros(self._num_layers,  batch_size, self._decoder_h_dim)

        decoder_output = self.decoder(obs[-1], obs_rel[-1], fused_features, (decoder_h, decoder_c))
        return decoder_output[0], decoder_output[1]


class TrajectoryGenerator(nn.Module):
    """The GenerationUnit will be used to forecast for sequence_length"""
    def __init__(self,
                 embedder=None,
                 embedding_dim: int = 7,
                 encoder_h_dim: int = 64,
                 decoder_h_dim: int = 64,
                 seq_length: int = 10,
                 input_size: int = 13,
                 output_size:int = 7, 	# 3(transformation) + 4(rotation)
                 decoder_mlp_structure: list = [128],
                 decoder_mlp_activation: str = "Relu",
                 dropout: float = 0.0,
                 fusion_pool_dim:int = 64,
                 fusion_hidden_dim:int = 64,
                 context_feature_model_arch:str = "overfeat",
                 num_layers: int = 1):

        super(TrajectoryGenerator, self).__init__()

        self.context_features = ContextualFeatures(model_arch=context_feature_model_arch)

        self.gu = GenerationUnit(
            embedder=embedder,
            embedding_dim=embedding_dim,
            encoder_h_dim=encoder_h_dim,
            decoder_h_dim=decoder_h_dim,
            input_size=input_size,
            output_size=output_size,
            decoder_mlp_structure=decoder_mlp_structure,
            decoder_mlp_activation=decoder_mlp_activation,
            dropout=dropout,
            num_layers=num_layers,
            fusion_pool_dim=fusion_pool_dim,
            fusion_hidden_dim=fusion_hidden_dim
        )

        self._num_layers = num_layers
        self._seq_len = seq_length


    def forward(self, obs_traj, obs_traj_rel, frames):
        """
        :param obs_traj: shape (obs_length, batch, inputs_size)
        :param obs_traj_rel: shape (obs_length, batch, inputs_size)
        :param frames: Tensor of shape (4, 256, 256)
        :return: final_prediction: shape (seq_length, batch, input_size)
        """
        batch_size = len(obs_traj[0])
        obs_length = len(frames)
        final_prediction = [obs_traj]
        final_prediction_rel = [obs_traj_rel]
        gu_input = copy.deepcopy(obs_traj)
        gu_input_rel = copy.deepcopy(obs_traj_rel)

        context_features = []
        for i in range(obs_length):
            context_features.append(self.context_features(frames[i]))

        context_features_sum = torch.stack(context_features, dim=0).sum(dim=0)
        # Should be of the shape (batch_size, 1024, 12 * 12)

        for _ in range(self._seq_len):
            predicted_features, predicted_features_rel = self.gu(obs=gu_input,
                                                                 obs_rel=gu_input_rel,
                                                                 context_features=context_features_sum)

            # build the inputs for the next timestep
            final_prediction = torch.cat(final_prediction + predicted_features, dim=1)
            gu_input = final_prediction[:, -obs_length:, :]
            gu_input_rel = torch.cat(final_prediction_rel + predicted_features_rel, dim=1)[:, -obs_length:, :]

        return final_prediction[:, -self._seq_len:, :]

##################################################################################
#                               GAN Discriminator
# ________________________________________________________________________________
class TrajectoryDiscriminator(nn.Module):
    def __init__(self,
                 embedder = None,
                 input_size: int = 13,
                 embedding_dim: int = 7,
                 num_layers: int = 1,
                 encoder_h_dim: int = 64,
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

        self.encoder = Encoder(
            embedder=embedder,
            input_size=input_size,
            embedding_dim=embedding_dim,
            encoder_h_dim=encoder_h_dim,
            dropout=dropout,
            num_layers=num_layers
        )

        self.classifier = make_mlp(layers=[encoder_h_dim] + mlp_structure,
                                   activation=mlp_activation,
                                   dropout=dropout,
                                   batch_normalization=batch_normalization)

    def forward(self, traj):
        encoded_features = self.encoder(traj)
        scores = self.classifier(encoded_features[0])
        return scores

if __name__ == '__main__':
    pass
