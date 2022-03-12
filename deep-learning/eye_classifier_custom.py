from io import StringIO
from xmlrpc.client import Boolean
from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from enum import Enum, IntEnum, unique, auto
from typing import Tuple, List, Union

from eye_classifier import *


class CustomEyeClassifier(EyeClassifier):
    def __init__(self, num_classes: int) -> None:
        super(CustomEyeClassifier, self).__init__(model=[

            (nn.Conv2d(in_channels=3, out_channels=6,
                       kernel_size=(5, 5), stride=(2, 2), padding=(0, 0), dilation=(1, 1)),
             TransferFunction.NotApplicable),

            (nn.MaxPool2d(
                kernel_size=(5, 5), stride=(2, 2), padding=(0, 0), dilation=(1, 1)),
             TransferFunction.NotApplicable),

            (nn.Dropout(),
             TransferFunction.NotApplicable),

            (nn.Conv2d(in_channels=6, out_channels=16,
                       kernel_size=(5, 5), stride=(2, 2), padding=(0, 0), dilation=(1, 1)),
             TransferFunction.NotApplicable),

            (nn.MaxPool2d(
                kernel_size=(5, 5), stride=(2, 2), padding=(0, 0), dilation=(1, 1)),
             TransferFunction.NotApplicable),

            (nn.Dropout(),
             TransferFunction.NotApplicable),

            (nn.Linear(in_features=13456, out_features=84),
             TransferFunction.Relu),

            (nn.Linear(in_features=84, out_features=42),
             TransferFunction.Relu),

            (nn.Linear(in_features=42, out_features=num_classes),
             TransferFunction.NotApplicable),
        ])
