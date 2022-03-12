import torch.nn as nn
import torchvision.models as models
from eye_classifier import *


class ResnetEyeClassifier(EyeClassifier):
    def __init__(self, num_classes: int) -> None:
        super(ResnetEyeClassifier, self).__init__(model=[

            (models.resnet18(), TransferFunction.NotApplicable),

            (nn.Linear(in_features=1000, out_features=84),
             TransferFunction.Relu),

            (nn.Linear(in_features=84, out_features=42),
             TransferFunction.Relu),

            (nn.Linear(in_features=42, out_features=num_classes),
             TransferFunction.NotApplicable),
        ])
