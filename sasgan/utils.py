"""
This is the main file for utility functions
"""
import os
import time
import io
import configparser
import attrdict
from typing import Dict
from torch import nn
import numpy as np
import torch

def get_device(logger):
    """
    Check if the running host have any GPUs available, if yes print the number of the available GPUs
    :return: the number of available GPUs
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info(f"The number of GPUs available: {torch.cuda.device_count()}")
        logger.info("Using GPU...")

    else:
        device = torch.device("cpu")
        logger.info("No available GPU, running on CPU...")

    return device

def config(module_name: str=None) -> Dict[str, str]:
    """
    To be completed by mohammad
    :param module_name:
    :return:
    """
    config = configparser.ConfigParser()
    config.read("config.ini")
    try:
        return config[module_name]
    except KeyError as err:
        print(f"Module name should be one of the:\n "
              f"{config.sections()} not {err}")


def _get_loading_strategy(model_type:str = "main"):
    """
    returns the loading strategy based on what model queried
    :param model_type: cae or main
    :return: str showing the type
    """
    if model_type == "main":
        return config("Training")["loading_strategy"]

    elif model_type == "cae":
        return config("CAE")["loading_strategy"]


def checkpoint_path(path):
    """
    This is the function for checking if there is any saved model in the given path
    :return:(str) if found any paths, returns the path otherwise None
    """
    try:
        directories = os.listdir(path)
    except:
        os.makedirs(path)
        return None
    processed_path = dict()
    processed_path["checkpoint"] = []

    for dir in directories:
        if "best" in dir:
            processed_path["best"] = dir

        elif "checkpoint" in dir:
            processed_path["checkpoint"].append(tuple((dir.split(sep="-")[1].split(sep=".")[0], dir)))

    loading_strategy = _get_loading_strategy("cae") if "cae" in path else _get_loading_strategy("main")

    if len(processed_path["checkpoint"]) == 0 and "best" not in processed_path:
        return None

    elif loading_strategy == "best" and "best" in processed_path:
        return os.path.join(path, processed_path["best"])

    elif loading_strategy == "best" and "best" not in processed_path:
        return None

    elif loading_strategy == "last" and "checkpoint" in processed_path:

        # Finding the last saved_checkpoint
        directories_list = np.asarray(processed_path["checkpoint"], dtype=object)
        return os.path.join(path, directories_list[directories_list[:, 0].astype(np.int).argmax(), 1])

    else:
        return None

def get_tensor_type():
    """
    To convert the datatype of all the tensors in the models
    :return: Float tensor datatype
    """
    if torch.cuda.is_available():
        return torch.cuda.FloatTensor
    else:
        return torch.FloatTensor


def make_mlp(layers: list,
             activation: str = "Relu",
             dropout: float = 0.0,
             batch_normalization: bool = True):
    """
    Makes a mlp with the specified inputs
    :param layers: a list containing the dimensions of the linear layers
    :param activation: "Relu" or "LeakyRelu"
    :param dropout: a float between 0.0 and 1.0
    :return: the nn.module object constructed with nn.Sequential
    """
    nn_layers = []
    for dim_in, dim_out in zip(layers[:-1], layers[1:]):
        nn_layers.append(nn.Linear(dim_in, dim_out))
        if batch_normalization :
            nn_layers.append(nn.BatchNorm1d(dim_out))
        if activation == "Relu":
            nn_layers.append(nn.ReLU())
        elif activation == "LeakyRelu":
            nn_layers.append(nn.LeakyReLU())
        elif activation == "Sigmoid":
            nn_layers.append(nn.Sigmoid())
        elif activation == "Tanh":
            nn_layers.append(nn.Tanh())
        if dropout > 0:
            nn_layers.append(nn.Dropout(p=dropout))

    return nn.Sequential(*nn_layers)

def convert_str_to_list(string_list):
    return [int(item.strip()) for item in string_list.strip('][').split(",")]

if __name__ == '__main__':
    cae_config = config("CAE")
    print(cae_config["latent_dim"])
    # print(cae_config.latent_dim)

