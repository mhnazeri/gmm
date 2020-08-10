"""
This is the main file for utility functions
"""
import os
import time
import io
import functools
import configparser
from typing import Dict
from torch import nn
import numpy as np
import torch


def default_collate(batch):
    elem = batch[0]
    data = dict()
    for key in elem.keys():
        if key == "motion":
            stacked_list = []
            for item in batch:
                stacked_images = torch.stack(item[key], dim=0)
                stacked_images = stacked_images.expand(item["past"].shape[1], *stacked_images.shape)
                stacked_list.append(stacked_images)

            data[key] = torch.cat(stacked_list, dim=0)

        else:
            data[key] = torch.cat([item[key] for item in batch], dim=1)
    return data


def init_weights(m):
    if m.__class__.__name__ == "Linear":
        nn.init.kaiming_normal_(m.weight)


def get_device(logger, use_gpu=True):
    """
    Check if the running host have any GPUs available, if yes print the number of the available GPUs
    :return: the number of available GPUs
    """
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:0")
        if "main" in logger.name:
            logger.info(f"The number of GPUs available: {torch.cuda.device_count()}")
            logger.info("Using GPU...")

    else:
        device = torch.device("cpu")
        if "main" in logger.name:
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


def get_tensor_type(use_gpu=True):
    """
    To convert the datatype of all the tensors in the models
    :return: Float tensor datatype
    """
    if torch.cuda.is_available() and use_gpu:
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
    :param activation: "Relu", "LeakyRelu", "Sigmoid" or "Tanh"
    :param dropout: a float between 0.0 and 1.0
    :return: the nn.module object constructed with nn.Sequential
    """
    nn_layers = []
    for dim_in, dim_out in zip(layers[:-1], layers[1:]):
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
        if batch_normalization and dim_out != layers[-1]:
            nn_layers.append(nn.BatchNorm1d(dim_out))
        if dropout > 0:
            nn_layers.append(nn.Dropout(p=dropout))

    return nn.Sequential(*nn_layers)


def convert_str_to_list(string_list):
    return [int(item.strip()) for item in string_list.strip('][').split(",")]


def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)


def check_grad_norm(net):
    """Computes the grad norm of all parameters of the network"""
    total_norm = 0                                               
    for p in list(filter(lambda p: p.grad is not None, net.parameters())): 
        param_norm = p.grad.data.norm(2)                         
        total_norm += param_norm.item() ** 2                     
    total_norm = total_norm ** (1. / 2)                          
    return total_norm


def timeit_cuda(fn):
    """A function decorator to calculate the time a funcion needed for completion on GPU.
    returns: the function result and the time taken
    """
    @functools.wraps(func)
    def wrapper_fn(*args, **kwargs):
        torch.cuda.synchronize()
        t1 = time.time()
        result = fn(*args, **kwargs)
        torch.cuda.synchronize()
        t2 = time.time()
        take = t2 - t1
        return result, take

    return wrapper_fn


if __name__ == '__main__':
    cae_config = config("CAE")
    print(cae_config["latent_dim"])
    # print(cae_config.latent_dim)

