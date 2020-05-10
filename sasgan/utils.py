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

# class Logger(object):
#     """
#     The following class handles all the logging operations related to the model to be later used for visualizing using
#         tensorboard.
#     Note that, this class differs from the logger imported from the python Logger package, since it is only used for
#         visualizing the tensors, or other  evaluation metrics related to the model.
#     """
#     def __init__(self, log_dir:str = "./logs"):
#         """
#         The constructor to build the writer
#         :param log_dir: the directory the log files will be saved in.
#         """
#         self.__writer = tf.summary.create_file_writer(log_dir)
#         self.__writer.set_as_default()
#
#     def scalar_summary(self, tag, value, step):
#         """
#         Logs a scalar value
#         :param tag: The tag of the metric
#         :param value: The value of the metric ----> y axis
#         :param step: The step of logging ----> x axis
#         :return: None
#         """
#
#         tf.summary.scalar(tag, value, step)
#
#     def image_summary(self, tag, images, step=0, max_outputs = 20):
#         """
#         Log a list of images
#         :param tag: The specified tag
#         :param images: The list of the images of the shape (batch_size, width, height, channels)
#         :param step: The step of Logging
#         :param max_outputs: The maximum number of images that will be showed
#         :return: None
#         """
#
#         tf.summary.image(tag, data=images, max_outputs=max_outputs, step=step)
#
#     @staticmethod
#     def plot_to_image(figure):
#         """
#         Function used for converting a figure to an image to be shown in tensorboard
#         :param figure: The source figure
#         :return: Image format of the figure
#         """
#         buf = io.BytesIO()
#         plt.savefig(buf, format='png')
#         plt.close(figure)
#         buf.seek(0)
#         image = tf.image.decode_png(buf.getvalue(), channels=4)
#         image = tf.expand_dims(image, 0)
#         return image
#
#     def figure_summary(self, tag, figure, step=0):
#         """
#         Used for plotting a figure of any type in the tensorboard dashboard
#         :param tag: The specified tag
#         :param figure: The source figure
#         :param step: The step of visualizing
#         :return: None
#         """
#         image = Logger.plot_to_image(figure)
#         self.image_summary(tag, image, step, max_outputs=1)

    # Still have to work on
    # def histo_summary(self, tag, values, step, bins=1000):
    #     """
    #     Log a histogram of tensor values
    #     :param tag: The specified tag
    #     :param values: The values of the metric which will be used for histograming
    #     :param step: The step of visualizing
    #     :param bins: Number of the total bins
    #     :return: None
    #     """
    #     # Create a histogram using numpy
    #     counts, bin_edges = np.histogram(values, bins=bins)
    #
    #     # Fill the fields of the histogram proto
    #     hist = tf.HistogramProto()
    #     hist.min = float(np.min(values))
    #     hist.max = float(np.max(values))
    #     hist.num = int(np.prod(values.shape))
    #     hist.sum = float(np.sum(values))
    #     hist.sum_squares = float(np.sum(values ** 2))
    #
    #     # Drop the start of the first bin
    #     bin_edges = bin_edges[1:]
    #
    #     # Add bin edges and counts
    #     for edge in bin_edges:
    #         hist.bucket_limit.append(edge)
    #     for c in counts:
    #         hist.bucket.append(c)
    #
    #     # Create and write Summary
    #     summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
    #     self.__writer.add_summary(summary, step)
    #     self.__writer.flush()


def get_the_number_of_GPUs(logger):
    """
    Check if the running host have any GPUs available, if yes print the number of the available GPUs
    :return: the number of available GPUs
    """
    if torch.cuda.device_count() > 1:
        logger.info("Number of the GPU: ", str(torch.cuda.device_count()))

    else:
        logger.info("No GPU devices available")

    return torch.cuda.device_count()

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

def get_tensor_type(args):
    """
    To convert the datatype of all the tensors in the models
    :return: Float tensor datatype
    """
    if args.use_gpu:
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
        if dim_out != layers[-1]:
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

