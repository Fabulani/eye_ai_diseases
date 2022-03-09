# %%
from io import StringIO
from xmlrpc.client import Boolean
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset, DataLoader
from enum import Enum, IntEnum, unique, auto
from typing import Tuple, List, Union, Any, Optional, Callable
import pandas as pd
import numpy as np
import functools
import skimage.io as io
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.filters import threshold_mean, sobel
from skimage.color import rgb2gray
from skimage import feature
import os.path as ospath
from PIL import Image
import torchvision.transforms as transforms

from eye_classifier import *
from eye_dataset import *

#base_dir = "../../data"
base_dir = "/workspaces/VisaoComputacional/trabalho-final/data"
image_dir = f"{base_dir}/preprocessed_images"
csv_file = f'{base_dir}/ODIR-5K/data.csv'

#ds = read_images(base_dir, image_path=image_dir, data_info_csv_file=csv_file, limit_input_count=100)
ds = read_images(base_dir, image_path=image_dir, data_info_csv_file=csv_file)

print(ds.targets[0])
train_loader = DataLoader(ds, batch_size=4, shuffle=True)

nn = EyeClassifier(image_shape=[512,512,3], num_classes=len(ds.classes))
print(nn)

nn.train(ds)
nn.test(ds)

