"""
This is the main file for utility functions
"""
import torch
import logging

logger = logging.getLogger(__name__)

def get_the_number_of_GPUs():
    """
    Check if the running host have any GPUs available, if yes print the number of the available GPUs
    :return: the number of available GPUs
    """
    if torch.cuda.device_count() > 1:
        logger.info("Number of the GPU: ", str(torch.cuda.device_count()))

    else:
        logger.info("No GPU devices available")

    return torch.cuda.device_count()


def save_model(path, dict):
    """
    Method used for saving. All the items should be given inside the dictionary and it will be saved in the specified path
    :param path: The path the model will be saved in
    :param dict: The parameters of interest
    :return: None
    """
    torch.save(dict, path + "RNN_checkpoint.pt")


def load_model(path):
    """
    Load the model and return the loaded dictionary to recover the model from the last steps
    :param path: The path the model will be loaded from
    :return: returns the loaded dictionary
    """
    return torch.load(path + "RNN_checkpoint.pt")


