# EYE diseases detection

This project is a prof-of-concept of using computer vision to detect several anomalies in fundoscopy eye exams

A classical approach is done on classic/ folder currently for cataract only, while a deep-learning approach is done for
diabetes, cataract, glaucoma, age-related macular degeneration, hypertension and pathological myopia.

The source dataset includes other kind of eye-problems which are classified as other


## Image Dataset for this project

You can find the dataset on https://www.kaggle.com/andrewmvd/ocular-disease-recognition-odir5k

It is comprised of fundoscopic images already split for training and testing on separate folders and an excel file containing 
the dataset metadata revised by physicians for supervised tranining

The metadata is comprised of flags for normal fundus, diabetes, cataract, glaucoma, age-related macular degeneration, hypertension, pathological myopia and other diseases.

Be aware, though, that those flags are related to the patient's final diagnostics, therefore if a person has cataract on one eye and a normal fundus on the other eye, the patient's diagnostics will be cataract.

The columns named Left-Diagnostic Keywords and Right-Diagnostic Keywords provide keywords related to the fundoscopic diagnostics for each eye


## Running from a docker container

If you wish to run this project from a docker container, you can use the Dockerfiles in the pytorch_dockerfile/ path

if you have CUDA enabled
```sh
docker build -f ./cuda_Dockerfile -t pytorch-cuda-img .
```

if not
```sh
docker build -f ./cpu_Dockerfile -t pytorch-img
```


## Running from a python virtualenv 

In the project root, run

```sh
python -m pip install -r requirements.txt
```

To open the Jupyter interface, run

```sh
jupyter notebook
```

## Code explained

First and foremost, we deal with the dataset. We need to convert it to a format which is easier to connect each eye-image to each diagnostics. 
Therefore, we use the columns Left-Diagnostic Keywords and Right-Diagnostic Keywords to provide the fundoscopic diagnostics and we translate it
into one of the labels for our dataset.

The labels are:

- Normal
- Diabetes
- Glaucoma
- Cataract
- AgeRelatedMacularDegeneration
- Hypertension
- PathologicalMyopia
- Other

All of this is done by the eye_dataset.py file, which also filters off some unwanted images, classified as bad, low quality, dust etc.

If you wish to take a look at the mapping conversion we used on our dataset, please refer to the diagnostics_classification.txt file or
take a look at the code on eye_dataset.py itself.

### Classical approach

The classical approach code is included in the classic/ folder.

cataract_detection.ipynb deals with cataract detection using Canny

classification_svm uses SVM and wavelet haar transformation to classify all images according to the expected label, following the work of [taspinar](
https://github.com/taspinar/siml/blob/master/notebooks/WV4%20-%20Classification%20of%20ECG%20signals%20using%20the%20Discrete%20Wavelet%20Transform.ipynb)


### Deep-learning approach

The convolutional approach code is included in the deep-learning/ folder.

We provide several classification examples using convolutional neural networks. 


classification_cnn_custom.ipynb carries a custom CNN approach
classification_resnet_18.ipynb uses resnet18 as the convolutional layers
classification_resnet_34.ipynb uses resnet34 as the convolutional layers

You can easily change the model and try different kinds of neural networks to deal with the problem.

All you have to do is build each layer of your model as a Tuple ( nn.Model, transfer function ) where
the transfer function is one of the TransferFunction enum available. TransferFunction.NoApplicable will simply ignore the transfer function.

#### utilities

Our eye_classifier util will provide you a train_model() method and a test_model() method to train and test the model using the dataset

You can also change the optimizer and the loss function by using set_optimizer() and set_loss_function()

You can also freeze/ unfreeze layers to prevent them from being trained, using freeze_layer() and unfreeze_layer().

Finally, you can save and load individual layer weights or the whole model weights by using save_layer_weights(), load_layer_weights(),
save_weights() and load_weights() respectivelly.

### Some results

#### Classic Cataract Detection with Canny

True positives: 198/286: 69.23 %
False positives: 133/286: 46.50 %
True negatives: 5863/5996: 97.78 %
False negatives: 88/5996: 1.47 %

#### Classic All-Diseases Detection with SVM and Wavelet Haar Transformations





#### RESNET-18

accuracy: 82.48% [43310/52512]
	 - Normal: 92.34% [6061/6564]
	 - Diabetes: 83.87% [5505/6564]
	 - Glaucoma: 74.56% [4894/6564]
	 - Cataract: 90.40% [5934/6564]
	 - AgeRelatedMacularDegeneration: 83.68% [5493/6564]
	 - Hypertension: 74.73% [4905/6564]
	 - PathologicalMyopia: 92.00% [6039/6564]
	 - Other: 68.24% [4479/6564]

#### RESNET-34



#### CUSTOM 

accuracy: 70.59% [37066/52512]
	 - Normal: 42.93% [2818/6564]
	 - Diabetes: 27.70% [1818/6564]
	 - Glaucoma: 94.03% [6172/6564]
	 - Cataract: 95.67% [6280/6564]
	 - AgeRelatedMacularDegeneration: 94.24% [6186/6564]
	 - Hypertension: 97.04% [6370/6564]
	 - PathologicalMyopia: 96.01% [6302/6564]
	 - Other: 17.06% [1120/6564]