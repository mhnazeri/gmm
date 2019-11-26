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
