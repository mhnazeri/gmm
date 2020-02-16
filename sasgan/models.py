""" This the file to implement the main model including:
    1. Encoder(Boh)
    2.
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os
from utils import *

sys.path.append("data/")
from loader import CAEDataset

class Encoder(nn.Module):
    def __init__(self, num_layers=32):
        super(Encoder, self).__init__()
        self.num_layers = 32
        self.lstm = nn.LSTM()


    def forward(x):
        pass


num_layers = 10
path_for_json_data = "data/exported_json_data/"
dataset_dir = "data/nuScene-mini"
input_size = 14


# used for testing
if __name__ == '__main__':
    files_list = os.listdir(path_for_json_data)
    for i, file in enumerate(files_list):
        dataset = CAEDataset(path_for_json_data + file, dataset_dir)
        print(dataset.shape)
        print(len(dataset))
        encoder = Encoder()
        encoder()