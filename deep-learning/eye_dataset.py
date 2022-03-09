from torchvision.datasets import VisionDataset
from enum import IntEnum, unique
from typing import Tuple, Any, Optional, Callable
import pandas as pd
import numpy as np
import skimage.io as io
import os.path as ospath
from PIL import Image
import torchvision.transforms as transforms

from eye_classifier import *

@unique
class TargetLabel(IntEnum):
    Normal = 0,
    Diabetes = 1,
    Glaucoma = 2,
    Cataract = 3,
    AgeRelatedMacularDegeneration = 4,
    Hypertension = 5,
    PathologicalMyopia = 6,
    Other = 7,

@unique
class TargetType(IntEnum):
    GoodImage = 0,
    BadImage = 1,
    NotUsefulDiagnostics = 2

class ImageDataset (VisionDataset):
    files = []
    data = None
    classes = []
    targets = []

    def __init__(self, root: str, transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None) -> None:

        super(ImageDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:           
            img, target = self.data[index], self.targets[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, torch.Tensor(target)

__ignore_image_str = [
    'no fundus image',
    'lens dust',
    'optic disk photographically invisible',
    'asteroid hyalosis',
    'image offset',
    'low image quality',
    'anterior segment image'
]
__myopia_str = [
    'myopic maculopathy',
    'myopic retinopathy'
]
__glaucoma_str = [
    'glaucoma',
    'suspected glaucoma',
    'intraretinal hemorrhage'
]
_other_diagnostics_str = [
    'piretinal membrane over the macula',
    'branch retinal vein occlusion',
    'peripapillary atrophy',
    'refractive media opacity',
    'depigmentation of the retinal pigment epithelium',
    'spotted membranous change',
    'optic nerve atrophy',
    'tessellated fundus',
    'old central retinal vein occlusion',
    'macular epiretinal membrane',
    'retinitis pigmentosa',
    'optic disk epiretinal membrane',
    'pigment epithelium proliferation',
    'epiretinal membrane',
    'laser spot',
    'old choroiditis',
    'chorioretinal atrophy with pigmentation proliferation',
    'drusen',
    'old branch retinal vein occlusion',
    'retinal pigment epithelium atrophy',
    'retina fold',
    'fundus laser photocoagulation spots',
    'idiopathic choroidal neovascularization',
    'suspected retinal vascular sheathing',
    'post retinal laser surgery',
    'vessel tortuosity',
    'vitreous degeneration',
    'retinal artery macroaneurysm',
    'silicone oil eye',
    'macular pigmentation disorder',
    'branch retinal artery occlusion',
    'punctate inner choroidopathy',
    'central retinal vein occlusion',
    'myelinated nerve fibers',
    'old chorioretinopathy',
    'wedge white line change',
    'post laser photocoagulation',
    'diffuse retinal atrophy',
    'atrophic change',
    'retinal pigmentation',
    'choroidal nevus',
    'optic disc edema',
    'arteriosclerosis',
    'macular hole',
    'rhegmatogenous retinal detachment',
    'diffuse chorioretinal atrophy',
    'maculopathy',
    'suspected microvascular anomalies',
    'morning glory syndrome',
    'oval yellow-white atrophy',
    'suspected abnormal color of  optic disc',
    'intraretinal microvascular abnormality',
    'wedge-shaped change',
    'vitreous opacity',
    'central serous chorioretinopathy',
    'vascular loops',
    'glial remnants anterior to the optic disc',
    'central retinal artery occlusion',
    'abnormal pigment',
    'retinal detachment',
    'suspected macular epimacular membrane',
    'suspected retinitis pigmentosa',
    'retinal pigment epithelial hypertrophy',
    'epiretinal membrane over the macula'
]

def __translate_diagonstics(diagnostics:str) -> Tuple[TargetLabel, TargetType]:
    
    diag = diagnostics.lower().strip()
    if diag == "normal fundus": return (TargetLabel.Normal, TargetType.GoodImage)
    elif diag == "white vessel": return (TargetLabel.Normal, TargetType.NotUsefulDiagnostics)
    elif diag in  __ignore_image_str: return (TargetLabel.Normal, TargetType.BadImage)
    elif 'proliferative retinopathy' in diag or \
         'nonproliferative' in diag or \
         'diabetic' in diag: return (TargetLabel.Diabetes, TargetType.GoodImage)
    elif 'myopia' in diag or diag in __myopia_str: return (TargetLabel.PathologicalMyopia, TargetType.GoodImage)
    elif diag == 'hypertensive retinopathy': return (TargetLabel.Hypertension, TargetType.GoodImage)
    elif diag in __glaucoma_str: return (TargetLabel.Glaucoma, TargetType.GoodImage)
    elif 'cataract' in diag: return (TargetLabel.Cataract, TargetType.GoodImage)
    elif 'age-related macular degeneration' in diag: return (TargetLabel.AgeRelatedMacularDegeneration, TargetType.GoodImage)
    elif 'coloboma' in diag or \
         'chorioretinal atrophy' in diag or \
         'chorioretinal atrophy' in diag or \
         diag in _other_diagnostics_str: return (TargetLabel.Other, TargetType.GoodImage)
    else:
        raise Exception(str.format(f'Diagnostics \"{diagnostics}\"not expected'))

def __build_diagostics_labels(diagnostics_str: str, num_classes: int) -> Tuple[dict, bool]:
    labels = [0 for _ in range(0, num_classes)]

    for d in diagnostics_str.split(','):
        label, type = __translate_diagonstics(d)
        if (type == TargetType.NotUsefulDiagnostics):
            continue
        elif (type == TargetType.BadImage):
            return [None, False]
        else:
            labels[int(label)] = 1
    
    return [labels, True]
            
def __build_empty_dict(classes: List[str]) -> dict:
    labels = dict()
    for c in classes:
        labels[c] = torch.Tensor([0])
    return labels



def read_images(base_dir: str, image_path: str, data_info_csv_file: str, limit_input_count = -1) -> ImageDataset:

    tranform_method = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    dataset = ImageDataset(base_dir, transform=tranform_method)
    dataset.classes = [e.name for e in TargetLabel]

    data = []
    limit = limit_input_count

    for idx, row in pd.read_csv(data_info_csv_file).iterrows():
        left_file = f"{image_path}/{row['Left-Fundus']}"
        right_file = f"{image_path}/{row['Right-Fundus']}"
        
        if limit >= 0:
            if limit == 0: break
            limit -= 1

        num_classes = len(dataset.classes)

        if ospath.exists(left_file):
            labels, valid_image = __build_diagostics_labels(row['Left-Diagnostic Keywords'], num_classes)
            if valid_image:
                dataset.targets.append(labels)   
                dataset.files.append(left_file)
                data.append(io.imread(left_file))
        
        if ospath.exists(right_file):
            labels, valid_image = __build_diagostics_labels(row['Right-Diagnostic Keywords'], num_classes)
            if valid_image:
                dataset.targets.append(labels)   
                dataset.files.append(right_file)
                data.append(io.imread(right_file))

    
    dataset.data = np.asarray(data, dtype=np.uint8)
    return dataset