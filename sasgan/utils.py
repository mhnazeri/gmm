"""
This is the main file for utility functions
"""
import os
import time
import io
import configparser
import attrdict
from typing import Dict
import torch
import logging
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


logger = logging.getLogger(__name__)

class Logger(object):
    """
    The following class handles all the logging operations related to the model to be later used for visualizing using
        tensorboard.
    Note that, this class differs from the logger imported from the python Logger package, since it is only used for
        visualizing the tensors, or other  evaluation metrics related to the model.
    """
    def __init__(self, log_dir:str = "./logs"):
        """
        The constructor to build the writer
        :param log_dir: the directory the log files will be saved in.
        """
        self.__writer = tf.summary.create_file_writer(log_dir)
        self.__writer.set_as_default()

    def scalar_summary(self, tag, value, step):
        """
        Logs a scalar value
        :param tag: The tag of the metric
        :param value: The value of the metric ----> y axis
        :param step: The step of logging ----> x axis
        :return: None
        """

        tf.summary.scalar(tag, value, step)

    def image_summary(self, tag, images, step=0, max_outputs = 20):
        """
        Log a list of images
        :param tag: The specified tag
        :param images: The list of the images of the shape (batch_size, width, height, channels)
        :param step: The step of Logging
        :param max_outputs: The maximum number of images that will be showed
        :return: None
        """

        tf.summary.image(tag, data=images, max_outputs=max_outputs, step=step)

    @staticmethod
    def plot_to_image(figure):
        """
        Function used for converting a figure to an image to be shown in tensorboard
        :param figure: The source figure
        :return: Image format of the figure
        """
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image

    def figure_summary(self, tag, figure, step=0):
        """
        Used for plotting a figure of any type in the tensorboard dashboard
        :param tag: The specified tag
        :param figure: The source figure
        :param step: The step of visualizing
        :return: None
        """
        image = Logger.plot_to_image(figure)
        self.image_summary(tag, image, step, max_outputs=1)

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
    :param path: The path the model will be saved in, should contain / at the end
    :param dict: The parameters of interest
    :return: None
    """
    torch.save(dict, path + "checkpoint-" + str(time.time()) + ".pt")


def load_model(path):
    """
    Load the last saved model from the specified path
    :param path: The path the model will be loaded from, should contain / at the end
    :return: returns the loaded dictionary containing all the state_dicts
    """
    checkpoints_lists = os.listdir(path)
    last_times = 0
    index = 0
    for i, file_name in enumerate(checkpoints_lists):
        try:
            if float(file_name.replace("checkpoint-", "").replace(".pt", "")) > last_times:
                last_times = file_name
                index = i
        except:
                logger.warning("Unknown file occurred in the %s directory as checkpoint, ignoring ..." % path)

    return torch.load(path + checkpoints_lists[index])


def config(modeule_name: str=None) -> Dict[str, str]:
    config = configparser.ConfigParser()
    config.read("config.ini")
    try:
        # using attrdict
        # return attrdict.AttrDict(config[modeule_name])
        return config[modeule_name]
    except KeyError as err:
        print(f"Module name should be one of the:\n "
              f"{config.sections()} not {err}")


if __name__ == '__main__':
    cae_config = config("CAE")
    # print(cae_config["latent_dim"])
    # print(cae_config.latent_dim)