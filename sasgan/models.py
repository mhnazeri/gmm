import torch
import torch.nn as nn
import torchvision
# import numpy as np
# import sys
import os
import logging
from utils import *
from cfex import Encoder as CFEX
# from data.loader import CAEDataset
# import copy

# The logger used for debugging
logger = logging.getLogger(__name__)
device = get_device(logger)
DIRECTORIES = config("Directories")

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

        # if embedder is not None:
        #     self.embedder = embedder
        # else:
        #     self.embedder = nn.Linear(input_size, embedding_dim)

        self._num_layers = num_layers
        self._input_size = input_size
        self._embedding_dim = embedding_dim
        self._hidden_size = encoder_h_dim
        self.encoder = nn.LSTM(embedding_dim, encoder_h_dim, num_layers, dropout=dropout)

    def initiate_hidden(self, batch_size):
        return (
            torch.zeros(self._num_layers, batch_size, self._hidden_size).to(device),
            torch.zeros(self._num_layers, batch_size, self._hidden_size).to(device)
        )

    def forward(self, inputs, state_tuple=None):
        """
        :param inputs: A tensor of the shape (sequence_length, batch, input_size)
        :return: A tensor of the shape (self.num_layers, batch, self.encoder_h_dim)
        """
        # sequence_length = inputs.shape[0]
        batch_size = inputs.shape[1]


        # Check the integrity of the shapes
        logger.debug("The size of the inputs: " + str(inputs.size()))

        if state_tuple is None:
            states = self.initiate_hidden(batch_size)
        else:
            states = state_tuple

        # Embed the input data to the desired dimension using a cae encoder or a linear layer
        # changed by mohammad
        # embedded_features = []
        # for i in range(inputs.size()[0]): # sequence length
        #     embedded_features.append(self.embedder(inputs[i]))
        # embedder_inputs = inputs.view(-1, self._input_size)
        # embedded_features = self.embedder(embedder_inputs)
        # embedded_features = torch.stack(embedded_features, dim=0)

        # Return the shape of the inputs to the desired shapes for lstm layer to be encoded
        # lstm_inputs = embedded_features.view(sequence_length, batch_size, self._embedding_dim)
        _, states = self.encoder(inputs, states)
        return states


##################################################################################
#                        Contextual Feature Extractor
# ________________________________________________________________________________
class ContextualFeatures(nn.Module):
    """Extract contextual features from the environment
        Networks that can be used for feature extraction are:
            overfeat: returned matrix is 1024*12*12
    """
    def __init__(self, model_arch: str="overfeat", pretrained: bool=True, refine: bool=True):
        super(ContextualFeatures, self).__init__()

        if model_arch == "overfeat":
            self.net = CFEX.net
            saving_dictionary = torch.load(os.path.join(DIRECTORIES["save_model"], "cfex/best.pt"))
            self.net.load_state_dict(saving_dictionary["encoder"])

        elif model_arch == "vgg":
            vgg = torchvision.models.vgg11(pretrained=pretrained).features
            for i, layer in enumerate(vgg.children()):
                if isinstance(layer, nn.MaxPool2d):
                    vgg[i] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0,
                                          ceil_mode=False)
                elif isinstance(layer, nn.ReLU):
                    vgg[i] = nn.LeakyReLU(inplace=True)

            if refine:
                for param in vgg.parameters():
                    param.requires_grad = True
            else:
                for param in vgg.parameters():
                    param.requires_grad = False

            self.net = vgg

        else:
            raise ValueError(f"Unrecognized model architecture {model_arch}")

        self.softmax = nn.Softmax2d()

        # self-attention method proposed in self-attention gan Zhang et al.
        self.frame_fx = nn.Conv2d(in_channels=512, out_channels=512,
                             kernel_size=1, stride=1)
        self.frame_gx = nn.Conv2d(in_channels=512, out_channels=512,
                             kernel_size=1, stride=1)
        self.frame_hx = nn.Conv2d(in_channels=512, out_channels=512,
                             kernel_size=1, stride=1)
        self.frame_vx = nn.Conv2d(in_channels=512, out_channels=512,
                             kernel_size=1, stride=1)

    def forward(self, frame: torch.Tensor):
        # forward pass through feature_extractor
        frame = self.net(frame)
        # self-attention gan
        frame_fx = self.frame_fx(frame)
        frame_gx = self.frame_gx(frame)
        frame_hx = self.frame_hx(frame)
        frame = self.softmax(frame_fx.transpose_(3, 2).matmul(frame_gx))
        frame = frame_hx.matmul(frame)
        return self.frame_vx(frame).view(-1, 512, 12 * 12)

##################################################################################
#                               Fusion modules
# ________________________________________________________________________________
class Fusion(nn.Module):
    """Feature Pool and Fusion module"""
    def __init__(self, pool_dim=256, hidden_size=128, noise_dim=16):
        super(Fusion, self).__init__()
        self.hidden_size = hidden_size
        self._noise_dim = noise_dim
        # 3 modules that comprise the fusion
        # self.fuse_traj = nn.LSTM(input_size=92, hidden_size=hidden_size)
        self.fuse_context = nn.LSTM(input_size=144, hidden_size=hidden_size)
        self.fuse = nn.Sequential(nn.Linear(hidden_size * 2, pool_dim),
                                    # nn.MaxPool1d(kernel_size=2, stride=2),
                                    nn.Tanh())

    def initiate_hidden(self, batch):
        return (
            torch.zeros(1, batch, self.hidden_size).to(device),
            torch.zeros(1, batch, self.hidden_size).to(device)
        )

    def get_noise(self, shape, noise_type="gaussian"):
        if noise_type == 'gaussian':
            return torch.randn(*shape).to(device)
        elif noise_type == 'uniform':
            return torch.rand(*shape).sub_(0.5).mul_(2.0)
        raise ValueError('Unrecognized noise type "%s"' % noise_type)

    def forward(self, real_past, rel_past, pool, context_feature, agent_idx=-1):
        """receives the whole feature matrix as input (num_agents * 52)
        22 is for 2 seconds input (each second is 2 frame, each frame has 13
        features)
        args:
            real_past: a matrix containing unmodified past locations (num_agents, 7)
            pool: modified past locations (num_agents, 64)
            context_feature: tensor of size=(512, 144)
            i: desired agent number to forecast future, if -1, it will predict all the agents at the same time
        """
        batch = pool.size(1)
        if agent_idx == -1:
            agent = pool
            noise = self.get_noise((agent.size(1), self._noise_dim))

        # else:
        #     agent = pool[agent_idx] # a vector of size 64
        #     rel_past = rel_past[agent_idx] # vector of size 7
        #     real_past = real_past[agent_idx]
        #     noise = self.get_noise((5,))

        # concat relative_past with encoded features (vector of 28 + 64 = 92)
        # agent = torch.cat((rel_past, agent), 1)
        # agent = agent.view(-1, batch, 92)
        # feed lidar stream to lstm for fusion
        # _, traj_hidden, _ = self.fuse_traj(agent,
        #                                         self.initiate_hidden(batch, sequence_length))

        context_feature = torch.transpose(context_feature, 1, 0)
        _, context_hidden = self.fuse_context(context_feature,
                                                self.initiate_hidden(batch))

        # fusing the hidden state of two streams together with an mlp
        fused_features = torch.cat((agent[0], context_hidden[0][0]), 1)
        fused_features = self.fuse(fused_features)
        # concat all features (noise, real_past, fused_features) together to feed to the generator
        # dim: 5 + 28 + 256 = 289
        # fused_features_hidden = torch.cat(
        #     (noise, real_past, fused_features),
        #     1)

        # fused_features_hidden = torch.cat(
        #     (noise, fused_features),
        #     1)
        fused_features = torch.randn_like(fused_features) * fused_features
        return fused_features

##################################################################################
#                                    Decoder
# ________________________________________________________________________________
class Decoder(nn.Module):
    def __init__(self,
                 fusion_length,
                 de_embedder=None,
                 output_size: int = 7,
                 hidden_dim: int = 64,
                 num_layers: int = 1,
                 dropout: float = 0.0,
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

        if de_embedder is None:
            self.hidden2pos = []
            # make_mlp(
            #     layers=hidden2pos_structure,
            #     activation=decoder_mlp_activation,
            #     dropout=dropout,
            #     batch_normalization=False
            # )
            # nn_layers = []
            self.hidden2pos.append(nn.Linear(hidden_dim, 128))
            self.hidden2pos.append(nn.Tanh())
            self.hidden2pos.append(nn.BatchNorm1d(128))
            # self.hidden2pos.append(nn.Linear(128, 128))
            # self.hidden2pos.append(nn.Tanh())
            # self.hidden2pos.append(nn.BatchNorm1d(128))
            self.hidden2pos.append(nn.Linear(128, output_size))
            self.hidden2pos = nn.Sequential(*self.hidden2pos)

        else:
            self.hidden2pos = nn.Sequential()
            self.hidden2pos.add_module("map_input", nn.Linear(hidden_dim, de_embedder.mlp[0].in_features))
            self.hidden2pos.add_module("cae_decoder", de_embedder)
            self.hidden2pos.add_module("map_outout", nn.Linear(de_embedder.mlp[-3].out_features, output_size))

        self.decoder = nn.LSTM(fusion_length, hidden_dim,
                           num_layers, dropout=dropout)

    def forward(self, last_features, last_features_rel , fused_features, state_tuple):
        """
        :param state_tuple: A tuple of two states as the initial state of the LSTM where the first item
            contains the required info from the past, both in the shape (num_layers, batch_size, embedding_dim)
        """
        decoder_output, state_tuple = self.decoder(fused_features.unsqueeze(0), state_tuple)
        curr_features_rel = self.hidden2pos(decoder_output[-1])

        # predicted_traj = last_features.clone()
        # predicted_traj_rel = last_features_rel
        # predicted_traj[:, :self._output_size] += curr_features_rel
        predicted_traj_rel = curr_features_rel

        return predicted_traj_rel


##################################################################################
#                               Generator
# ________________________________________________________________________________
class GenerationUnit(nn.Module):
    """This class is responsible for generating just one frame"""
    def __init__(self, embedder, de_embedder, embedding_dim, encoder_h_dim, decoder_h_dim, input_size, output_size,
                 decoder_mlp_structure, decoder_mlp_activation, dropout, num_layers, fusion_pool_dim,
                 fusion_hidden_dim, fused_vector_length: int = 64):

        super(GenerationUnit, self).__init__()

        self.decoder = Decoder(
            de_embedder=de_embedder,
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
            activation="Tanh",
            batch_normalization=True,
        )

        self._num_layers = num_layers
        self._decoder_h_dim = decoder_h_dim
        self._encoder_h_dim = encoder_h_dim
        self.__input_size = input_size
        self.__embedding_dim = embedding_dim

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

        fused_features = self.fusion(obs,
                                     obs_rel,
                                     states[0],
                                     context_features)

        # decoder_h = self.encoder_decoder_h(states[0].view(-1, self._encoder_h_dim))
        # decoder_h = decoder_h.view(self._num_layers,  batch_size, self._decoder_h_dim)
        decoder_h = states[0]
        decoder_c = states[1]
        # decoder_c = torch.zeros(self._num_layers,  batch_size, self._decoder_h_dim).to(device)

        decoder_output = self.decoder(obs[-1], obs_rel[-1], fused_features, (decoder_h, decoder_c))
        return decoder_output


class TrajectoryGenerator(nn.Module):
    """The GenerationUnit will be used to forecast for sequence_length"""
    def __init__(self,
                 embedder=None,
                 de_embedder=None,
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
                 num_layers: int = 1,
                 pretrain: bool=True,
                 refine:bool = True):

        super(TrajectoryGenerator, self).__init__()

        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = nn.Linear(input_size, embedding_dim)

        self.context_features = ContextualFeatures(model_arch=context_feature_model_arch, pretrained=pretrain, refine=refine)

        self.gu = GenerationUnit(
            embedder=embedder,
            de_embedder=de_embedder,
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
        :param frames: Tensor of shape (batch, 4, 1, 256, 256)
        :return: final_prediction: shape (seq_length, batch, input_size)
        """
        batch_size = obs_traj.shape[1]
        obs_length = obs_traj.shape[0]
        final_prediction_rel = []
        gu_input = obs_traj
        gu_input_rel = obs_traj_rel

        # gu_input_rel = self.embedder(gu_input_rel)
        embedded_features = []
        for i in range(gu_input_rel.size()[0]): # sequence length
            embedded_features.append(self.embedder(gu_input_rel[i]))
        # embedder_inputs = inputs.view(-1, self._input_size)
        # embedded_features = self.embedder(embedder_inputs)
        gu_input_rel = torch.stack(embedded_features, dim=0)

        # Return the shape of the inputs to the desired shapes for lstm layer to be encoded
        gu_input_rel = gu_input_rel.view(obs_length, batch_size, -1)

        context_features = []
        for i in range(obs_length):
            context_features.append(self.context_features(frames[:, i]))

        # Should be of the shape (batch_size, 1024, 12 * 12)
        context_features_sum = torch.stack(context_features, dim=0).sum(dim=0)
        # next_gu_rel = gu_input_rel
        for i in range(self._seq_len):
            predicted_features_rel = self.gu(obs=gu_input, obs_rel=gu_input_rel[i:], context_features=context_features_sum)

            # build the inputs for the next timestep
            gu_input_rel = torch.cat([gu_input_rel, predicted_features_rel.unsqueeze(0)], dim=0)
            # next_gu_rel = gu_input_rel[i:]
            final_prediction_rel.append(predicted_features_rel)
            # final_prediction = torch.cat([final_prediction] + [predicted_features.unsqueeze(0)], dim=0)
            # gu_input = final_prediction[i:]
            # final_prediction_rel = torch.cat([final_prediction_rel] + [predicted_features_rel.unsqueeze(0)], dim=0)
            # gu_input_rel = final_prediction_rel[i:]
            

        # final_prediction = gu_input_rel[-self._seq_len:]
        # return final_prediction[-self._seq_len:]
        output_rel = torch.stack(final_prediction_rel, dim=0)
        # output = relative_to_abs(output_rel, obs_traj[-1, :, :2])
        return output_rel

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
                 mlp_structure: list = [72, 512, 256, 128],
                 activation: str = "Relu",
                 batch_normalization: bool = True,
                 dropout: float = 0.0,
                 spectral_norm: bool = True):
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

        self.hidden_dim = encoder_h_dim                                        
        # because past and future has different dimensions and we want to keep their temporal relation
        # self.encoder_past = nn.LSTM(13, encoder_h_dim)
        self.encoder = nn.LSTM(2, encoder_h_dim)
        # self.encoder_fuse = nn.LSTM(encoder_h_dim, encoder_h_dim)

        nn_layers = []
        if mlp_structure[-1] != encoder_h_dim:
            mlp_structure.insert(0, encoder_h_dim)
            
        for dim_in, dim_out in zip(mlp_structure[:-1], mlp_structure[1:]):
            if spectral_norm:
                nn_layers.append(nn.utils.spectral_norm(nn.Linear(dim_in, dim_out)))
            else:
                nn_layers.append(nn.Linear(dim_in, dim_out))
            if activation == "Relu":
                nn_layers.append(nn.ReLU())
            elif activation == "LeakyRelu":
                nn_layers.append(nn.LeakyReLU())
            elif activation == "Sigmoid":
                nn_layers.append(nn.Sigmoid())
            elif activation == "Tanh":
                nn_layers.append(nn.Tanh())
            elif activation == "ELU":
                nn_layers.append(nn.ELU())
            if batch_normalization and dim_out != mlp_structure[-1]:
                nn_layers.append(nn.BatchNorm1d(dim_out))
            if dropout > 0:
                nn_layers.append(nn.Dropout(p=dropout))

        nn_layers.append(nn.Linear(dim_out, 1))

        self.classifier = nn.Sequential(*nn_layers)

        # add spectral normalization for linear layer of the discriminator
        # for i, layer in enumerate(self.encoder.children()):
        #     if isinstance(layer, nn.Linear):
        #         self.encoder[i] = nn.utils.spectral_norm(layer)

        # for i, layer in enumerate(self.classifier.children()):
        #     if isinstance(layer, nn.Linear):
        #         self.classifier[i] = nn.utils.spectral_norm(layer)

    def init_hidden(self, batch):                                              
        return (                                                               
            torch.zeros(1, batch, self.hidden_dim).to(device),                                                                                           
            torch.zeros(1, batch, self.hidden_dim).to(device)                  
        )

    def forward(self, traj):
        # encoded_features = self.encoder(traj)
        # scores = self.classifier(encoded_features[0][0])
        # scores = self.classifier(traj)
        # return scores
        _, (hidden, c) = self.encoder(traj, None)
        # _, (hidden_future, _) = self.encoder_future(traj_future, (hidden_past, c_past))
        # _, hidden = self.encoder_fuse(torch.cat([hidden_past, hidden_future], dim=0), self.init_hidden(traj_past.size()[1]))
        scores = self.classifier(hidden[0].view(-1, self.hidden_dim))          
        return scores

if __name__ == '__main__':
    pass
