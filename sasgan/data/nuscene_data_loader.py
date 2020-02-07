"""This module is responsible for loading of NuScene dataset"""
# import required packages
import os
import math
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes


logger = logging.getLogger(__name__)


def load_dataset(_path, version='v1.0-mini', verbose=False):
    nusc = NuScenes(version=version, dataroot=_path, verbose=verbose)
    logger.log("DEBUG", str(nusc.list_scenes), exc_info=1)
    print(len(nusc.scene))


if __name__ == "__main__":
    load_dataset("nuScene-mini", verbose=True)
