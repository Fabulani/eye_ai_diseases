#! /usr/bin/python3
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.eye_dataset import *
from eye_classifier_resnet import *


base_dir = "/shared/data"
#base_dir = "/home/cristiano/Documents/Projects/Mestrado/VisaoComputacional/trabalho-final/data"
#base_dir = "/workspaces/VisaoComputacional/trabalho-final/data"
image_dir = f"{base_dir}/preprocessed_images"
csv_file = f'{base_dir}/ODIR-5K/data.csv'

#ds = read_images(base_dir, image_path=image_dir, data_info_csv_file=csv_file, limit_input_count=100)
print ('reading input dataset')
ds = EyeImageDataset.read_images(base_dir, image_path=image_dir, data_info_csv_file=csv_file, limit_input_count=50)
train_loader = DataLoader(ds, batch_size=4, shuffle=True)

print ('building model')
#nn = EyeClassifierResnet(image_shape=[512,512,3], num_classes=len(ds.classes))
nn = EyeClassifierResnet(num_classes=len(ds.classes))
print(nn)

print ('training model')
nn.train(ds)

print ('testing model')
nn.test(ds)

