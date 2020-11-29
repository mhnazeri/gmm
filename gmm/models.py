import torch
import torch.nn as nn
import torchvision
import os
import logging
from utils import *
from cfex import Encoder as CFEX

# The logger used for debugging
logger = logging.getLogger(__name__)
device = get_device(logger)
DIRECTORIES = config("Directories")


##################################################################################
#                                    Encoder
# ________________________________________________________________________________
class Encoder(nn.Module):
    def __init__(
        self,
        embedder=None,
        input_size: int = 13,
        embedding_dim: int = 7,
        encoder_h_dim: int = 64,
        dropout: float = 0.0,
        num_layers: int = 1,
    ):
        super(Encoder, self).__init__()

        self._num_layers = num_layers
        self._input_size = input_size
        self._embedding_dim = embedding_dim
        self._hidden_size = encoder_h_dim
        self.encoder = nn.LSTM(
            embedding_dim, encoder_h_dim, num_layers, dropout=dropout
        )

    def initiate_hidden(self, batch_size):
        return (
            torch.zeros(self._num_layers, batch_size, self._hidden_size).to(device),
            torch.zeros(self._num_layers, batch_size, self._hidden_size).to(device),
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

    def __init__(
        self, model_arch: str = "overfeat", pretrained: bool = True, refine: bool = True
    ):
        super(ContextualFeatures, self).__init__()

        if model_arch == "overfeat":
            self.net = CFEX()
            saving_dictionary = torch.load(
                os.path.join(DIRECTORIES["save_model"], "cfex/best.pt")
            )
            self.net.load_state_dict(saving_dictionary["encoder"])
            if refine:
                self.net.requires_grad_(True)
            else:
                self.net.requires_grad_(False)

        elif model_arch == "vgg":
            vgg = torchvision.models.vgg11(pretrained=pretrained).features
            for i, layer in enumerate(vgg.children()):
                if isinstance(layer, nn.MaxPool2d):
                    vgg[i] = nn.AvgPool2d(
                        kernel_size=2, stride=2, padding=0, ceil_mode=False
                    )
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
        self.frame_fx = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=1, stride=1
        )
        self.frame_gx = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=1, stride=1
        )
        self.frame_hx = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=1, stride=1
        )
        self.frame_vx = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=1, stride=1
        )

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
        self.fuse_context = nn.LSTM(input_size=144, hidden_size=hidden_size)
        self.fuse_h = nn.Sequential(nn.Linear(hidden_size * 2, pool_dim), nn.Tanh())
        self.fuse_c = nn.Sequential(nn.Linear(hidden_size * 2, pool_dim), nn.Tanh())

    def initiate_hidden(self, batch):
        return (
            torch.zeros(1, batch, self.hidden_size).to(device),
            torch.zeros(1, batch, self.hidden_size).to(device),
        )

    def get_noise(self, shape, noise_type="gaussian"):
        if noise_type == "gaussian":
            return torch.randn(*shape).to(device)
        elif noise_type == "uniform":
            return torch.rand(*shape).sub_(0.5).mul_(2.0)
        raise ValueError('Unrecognized noise type "%s"' % noise_type)

    def forward(self, pool, context_feature, agent_idx=-1):
        """receives the whole feature matrix as input (num_agents * 52)
        22 is for 2 seconds input (each second is 2 frame, each frame has 13
        features)
        args:
            real_past: a matrix containing unmodified past locations (num_agents, 7)
            pool: modified past locations (num_agents, 64)
            context_feature: tensor of size=(512, 144)
            i: desired agent number to forecast future, if -1, it will predict all the agents at the same time
        """
        batch = pool[0].size(1)
        if agent_idx == -1:
            agent = pool

        context_feature = torch.transpose(context_feature, 1, 0)
        _, context_hidden = self.fuse_context(
            context_feature, self.initiate_hidden(batch)
        )

        # fusing the hidden state of two streams together with an mlp
        fused_features_h = torch.cat(
            (
                agent[0].view(-1, self.hidden_size),
                context_hidden[0].view(-1, self.hidden_size),
            ),
            1,
        )
        fused_features_c = torch.cat(
            (
                agent[0].view(-1, self.hidden_size),
                context_hidden[0].view(-1, self.hidden_size),
            ),
            1,
        )
        fused_features_h = self.fuse_h(fused_features_h)
        fused_features_c = self.fuse_c(fused_features_c)

        noise = torch.randn_like(fused_features_c)
        fused_features_h = noise * fused_features_h
        fused_features_c = noise * fused_features_c
        return fused_features_h.unsqueeze(0), fused_features_c.unsqueeze(0)


##################################################################################
#                               Generator
# ________________________________________________________________________________
class TrajectoryGenerator(nn.Module):
    """The GenerationUnit will be used to forecast for sequence_length"""

    def __init__(
        self,
        embedder=None,
        de_embedder=None,
        embedding_dim: int = 7,
        encoder_h_dim: int = 64,
        decoder_h_dim: int = 64,
        seq_length: int = 10,
        input_size: int = 13,
        output_size: int = 7,  # 3(transformation) + 4(rotation)
        decoder_mlp_structure: list = [128],
        decoder_mlp_activation: str = "Relu",
        dropout: float = 0.0,
        fusion_pool_dim: int = 64,
        fusion_hidden_dim: int = 64,
        context_feature_model_arch: str = "overfeat",
        num_layers: int = 1,
        pretrain: bool = True,
        refine: bool = True,
    ):

        super(TrajectoryGenerator, self).__init__()

        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = nn.Linear(input_size, embedding_dim)

        self.context_features = ContextualFeatures(
            model_arch=context_feature_model_arch, pretrained=pretrain, refine=refine
        )
        self.fusion = Fusion(pool_dim=fusion_pool_dim, hidden_size=fusion_hidden_dim)
        if de_embedder is None:
            hidden2pos = []
            hidden2pos.append(nn.Linear(decoder_h_dim, 128))
            hidden2pos.append(nn.Tanh())
            hidden2pos.append(nn.BatchNorm1d(128))
            hidden2pos.append(nn.Linear(128, output_size))
            self.hidden2pos = nn.Sequential(*hidden2pos)

        else:
            self.hidden2pos = nn.Sequential()
            self.hidden2pos.add_module(
                "map_input", nn.Linear(decoder_h_dim, de_embedder.mlp[0].in_features)
            )
            self.hidden2pos.add_module("cae_decoder", de_embedder)
            self.hidden2pos.add_module(
                "map_outout", nn.Linear(de_embedder.mlp[-3].out_features, output_size)
            )

        # This module is used at the beginning to convert the input features
        self.encoder = Encoder(
            embedder=embedder,
            input_size=input_size,
            embedding_dim=embedding_dim,
            encoder_h_dim=encoder_h_dim,
            dropout=dropout,
            num_layers=num_layers,
        )

        self.decoder = nn.LSTM(
            output_size, fusion_hidden_dim, num_layers, dropout=dropout
        )

        self._num_layers = num_layers
        self._seq_len = seq_length
        self.output_size = output_size

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
        gu_input_rel = obs_traj_rel

        embedded_features = []
        for i in range(gu_input_rel.size()[0]):  # sequence length
            embedded_features.append(self.embedder(gu_input_rel[i]))

        gu_input_rel = torch.stack(embedded_features, dim=0)
        # Return the shape of the inputs to the desired shapes for lstm layer to be encoded
        gu_input_rel = gu_input_rel.view(obs_length, batch_size, -1)
        states = self.encoder(gu_input_rel)
        # extract frames features
        frames = frames.squeeze()
        context_features = self.context_features(frames)
        # fuse features
        fused_features = self.fusion(states, context_features)

        decoder_input = torch.zeros((batch_size, self.output_size), device=device)
        for i in range(self._seq_len):
            decoder_input = decoder_input.unsqueeze(0)
            output_decoder, fused_features = self.decoder(decoder_input, fused_features)
            output_decoder = self.hidden2pos(output_decoder[0])
            final_prediction_rel.append(output_decoder)
            decoder_input = output_decoder

        output_rel = torch.stack(final_prediction_rel, dim=0)
        return output_rel


##################################################################################
#                               GAN Discriminator
# ________________________________________________________________________________
class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self,
        embedder=None,
        input_size: int = 13,
        embedding_dim: int = 7,
        num_layers: int = 1,
        encoder_h_dim: int = 64,
        mlp_structure: list = [72, 512, 256, 128],
        activation: str = "Relu",
        batch_normalization: bool = True,
        dropout: float = 0.0,
        spectral_norm: bool = True,
    ):
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
            num_layers=num_layers,
        )

        self.hidden_dim = encoder_h_dim
        self.encoder = nn.LSTM(2, encoder_h_dim)

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

    def init_hidden(self, batch):
        return (
            torch.zeros(1, batch, self.hidden_dim).to(device),
            torch.zeros(1, batch, self.hidden_dim).to(device),
        )

    def forward(self, traj):
        _, (hidden, c) = self.encoder(traj, None)
        scores = self.classifier(hidden[0].view(-1, self.hidden_dim))
        return scores


if __name__ == "__main__":
    pass
