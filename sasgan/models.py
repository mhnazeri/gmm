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
    def __init__(self, input_size: int = 14, embedding_dimension: int = 64, hidden_size: int = 16, num_layers=32, ):
        super(Encoder, self).__init__()
        self.num_layers = 32
        self.hidden_size = hidden_size
        self.embedder = nn.Linear(input_size, embedding_dimension).to(device)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers).to(device)

    def initiate_hidden(self, batch_size):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
            torch.zeros(self.num_layers, batch_size, self.hidden_size)
        )

    def forward(self, input):
        """
        :param input: A tensor of the shape (sequence_length, batch, input_size)
        :return: A tensor of the shape (seqeunce_length, batch, hidden_size)
        """

        # Check the integrity of the shapes
        logger.debug("The size of the inputs: " + str(input.size()))

        batch_size = input.size(1)
        embed = self.embedder(input)
        states = self.initiate_hidden(batch_size)
        out, _, _ = self.lstm(embed, states)

        # The output is of the shape (batch_size, sequence_length, hidden_size)
        return out


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

# This the whole number of the timestamps on each sample ----> not sure yet
sequence_length = 80

# The Batch size ------> hyper
Batch_size = 128

# The number of the hidden_layers on each LSTM layer (This parameter acts like the width of the network) ------> hyper
hidden_layers = 40

if __name__ == '__main__':
    files_list = os.listdir(path_for_json_data)
    for i, file in enumerate(files_list):
        dataset = CAEDataset(path_for_json_data + file, dataset_dir)

        # In this part the data should be batched
        print(dataset.scene_frames.numpy().shape)
        print(len(dataset))
        encoder = Encoder()
        encoder()
