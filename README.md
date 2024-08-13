#  Eyeglasses Detection In Photos
 


  <p align="center">
   Nofar Ben Porat</a>
  <br>
    Hanan Ben shitrit
  </p>
  
## Background
In this project, the goal is to explore the impact of unfreezing different numbers of layers in a ResNet50 model when training on the CIFAR-100 dataset.
The approach involves six distinct steps, where progressively more layers of ResNet50 are unfrozen during training. 
Initially, only one layer is unfrozen, allowing the rest of the network to remain fixed. In subsequent steps,three, six, twelve, fifteen, and finally, twenty-five layers are unfrozen. This strategy enables a gradual adaptation of the network to the CIFAR-100 dataset, starting from a heavily pre-trained using only feature extraction on the model and moving towards a more fine-tuned version.



## Prerequisites
|Library         | Version |
|----------------------|----|
|`Python`|  `3.10.12`|
|`torch`|  `2.3.1+cu121`|
|`gdown`|  `4.7.3`|
|`torchvision`|  `0.18.1+cu121`|
|`Pillow`|  `9.4.0`|
|`matplotlib`|  `3.7.1`|
|`numpy`|  `1.26.4`|

## Files in the repository

|File name         | Purpsoe |
|----------------------|------|
|``| images from the dataset|
|`split.py`| the Train/Val/Test split we used|
|`Data.py`| loading the data from MeGlass and splitting it accourding to the tags| 
|`train.py`| code for training the model| 
|`output`| the output of the train| 
|`Project_report.pdf`| project report of our network|

## pre train model DINOv2
We used the pretrained DINOv2 architecture which is designed to produce high-performance visual features that can be effectively employed with simple classifiers.
The DINOv2 models were pretrained on a dataset of 142 M images without using any labels or annotations, showcasing the model's ability to learn powerful representations in a self-supervised manner.

We used the dinov2_vitg14_lc model in our project.
```python
# Import the pretrained dinov2 architecture
import torch
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
```
## The dataset
for the data set we used MeGlass which is an eyeglass dataset originaly designed for eyeglass face recognition evaluation. 
For each person in the dataset at least two face images with eyeglass and two face images without eyeglass.
the dataset has	47,917 Images in total devided to 14,832 Images with glasses and 33,085 Images without glasses.

## results 

### results after unfreezing the last layer   

### results after unfreezing the last ?? layers 

### results with and without data augmentation

## How To Run
?????????????????????????


## Sources
* MeGlass - the dataset used: https://github.com/cleardusk/MeGlass
* DINOv2 - the network architecture: https://github.com/facebookresearch/dinov2
