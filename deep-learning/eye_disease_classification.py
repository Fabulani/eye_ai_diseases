#! /usr/bin/python3
import sys
sys.path.insert(0, '..')

import torch.nn as nn
from torch.utils.data import DataLoader

from eye_classifier import *
from utils.eye_dataset import *
import torchvision.transforms as transforms


<<<<<<< HEAD
base_dir = "/home/cristiano/Documents/Projects/Mestrado/VisaoComputacional/trabalho-final/data"
#base_dir = "/workspaces/VisaoComputacional/trabalho-final/data"
=======
#base_dir = "/home/cristiano/Documents/Projects/Mestrado/VisaoComputacional/trabalho-final/data"
base_dir = "/workspaces/VisaoComputacional/trabalho-final/data"
>>>>>>> rearranging the code for allowing the dataset reader to be accessed by the classical code
image_dir = f"{base_dir}/preprocessed_images"
csv_file = f'{base_dir}/ODIR-5K/data.csv'

print ('reading input dataset')
tranform_method = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

ds = EyeImageDataset.read_images(base_dir, image_path=image_dir, data_info_csv_file=csv_file, tranform=tranform_method)
train_loader = DataLoader(ds, batch_size=4, shuffle=True)

print ('building model')
nn = EyeClassifier(image_shape=[512, 512, 3],num_classes=len(ds.classes))
print(nn)

print ('training model')
nn.train(ds)

print ('testing model')
nn.test(ds)

