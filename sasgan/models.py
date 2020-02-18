""" This file contains the implementation of the main models including:
    1. Encoder(Boh)
    2.
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os
from sasgan.utils import *
import logging
from sasgan.data.loader import CAEDataset


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
        self.embedder = nn.Linear(input_size, embedding_dimension).to(device)
        self.embedding_dimension = embedding_dimension
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers).to(device)

    def initiate_hidden(self, batch_size):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
            torch.zeros(self.num_layers, batch_size, self.hidden_size)
        )

    def forward(self, input):
        """
        :param input: A tensor of the shape (sequence_length, batch, input_size)
        :return: A tensor of the shape (self.num_layers, batch, self.hidden_size)
        """

        # Check the integrity of the shapes
        logger.debug("The size of the inputs: " + str(input.size()))

        batch_size = input.size(1)
        embed = self.embedder(input.view(-1, 2))
        embed = embed.view(-1, batch_size, self.embedding_dimension)

        states = self.initiate_hidden(batch_size)
        _, hidden_state, _ = self.lstm(embed, states)

        return hidden_state


##################################################################################
#                                Other modules
# ________________________________________________________________________________









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