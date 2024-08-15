<h1 align="center">Exploring Layer-Wise Fine-Tuning of a Pretrained ResNet-50 on CIFAR-100: Impact Analysis and Performance Evaluation</h1>
  <div align="center">
  <img src="TransferLearning.jpg" alt="Transfer Learning" width="500"/>
  </div>

  <p align="center">
  <a href="https://github.com/supernuper" target="_blank">Nofar Ben Porat</a>
  <br>
  <a href="https://github.com/hananbenshitrit" target="_blank">Hanan Ben Shitrit</a>
</p>

## Background
In this project, the goal is to explore the impact of unfreezing different numbers of layers in a ResNet50 model when training on the CIFAR-100 dataset.
The approach involves six distinct steps, where progressively more layers of ResNet50 are unfrozen during training. 
Initially, only one layer is unfrozen, allowing the rest of the network to remain fixed (Also known as feature extraction). In subsequent steps,three, six, twelve, fifteen, and finally, twenty-five layers are unfrozen. This strategy enables a gradual adaptation of the network to the CIFAR-100 dataset, starting from a heavily pre-trained using only feature extraction on the model and moving towards a more fine-tuned version.



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

|File name         | Purpose |
|----------------------|------|
|`code_files/Imports.ipynb`| Necessary Imports for the Project|
|`code_files/Dataset.ipynb`| Loading the CIFAR100 dataset and providing an example of the images| 
|`code_files/Split.ipynb`| Splitting the dataset to train-valid-test| 
|`code_files/HFuncs_HParam_DataLoaders.ipynb`| Helper functions for defining the layers parameters to learn, training and evaluating. Hyperparameters definition. Creating dataloaders| 
|`code_files/Model_Train_Eval.ipynb`| Training and evaluating the model on our data using all of the above|
|`output`| Contains an image and a text file for each run. The image is a graph of the training loss, and the text file includes the training epochs and the accuracy on the validation set (or the test set for the final test)|
|`CIFAR-100.png`| Example image from the CIFAR-100 dataset|
|`TransferLearning.jpg`| Example image illustrating the Transfer Learning theme|

## Pretrained model - ResNet-50
ResNet-50 is a deep convolutional neural network with 50 layers, known for its use of residual connections (skip connections) that help mitigate the vanishing gradient problem, making it easier to train. It’s part of the ResNet family, introduced in 2015, and is widely used for image classification and other computer vision tasks. The model is available pretrained on ImageNet via `torchvision`, making it ideal for transfer learning.

Key Features:
- 50 Layers: Deep architecture with multiple convolutional and fully connected layers.
- Residual Connections: Skip connections that allow the network to learn residuals, improving training stability and performance.
- Pretrained Weights: Available with pretrained weights on ImageNet, enabling quick transfer learning for new tasks.
- Versatile: Commonly used in image classification, object detection, and feature extraction tasks.

```python
import torch
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.DEFAULT)
```
## The dataset - CIFAR100
<div align="center">
  <img src="CIFAR-100.png" alt="CIFAR-100 Dataset" width="500"/>
</div>

CIFAR-100 is a popular image classification dataset that is widely used in the field of computer vision and deep learning. It consists of 100 different classes of images, with each class containing 600 images. The images are small, with a resolution of 32x32 pixels, and are in RGB format (three color channels: red, green, and blue).
The dataset is divided into 50,000 training images and 10,000 test images.

## Results 
### Feature Extraction 
```python
Epoch: 1 | Loss: 10.2540 | Training accuracy: 5.788% Epoch Time: 53.68 secs
Epoch: 2 | Loss: 4.7189 | Training accuracy: 28.540% Epoch Time: 52.66 secs
...
Epoch: 13 | Loss: 1.2130 | Training accuracy: 69.260% Epoch Time: 52.87 secs
Epoch: 14 | Loss: 1.1969 | Training accuracy: 69.582% Epoch Time: 53.17 secs
Epoch: 15 | Loss: 1.2051 | Training accuracy: 67.890% Epoch Time: 52.33 secs
```
<img src="output/feature_extraction.png" width="500"/>

```python
Validation Accuracy: 50.150%
```

### Fine-tuning the last 3 Layers
```python
Epoch: 1 | Loss: 7.6341 | Training accuracy: 23.985% Epoch Time: 55.46 secs
Epoch: 2 | Loss: 2.6999 | Training accuracy: 49.343% Epoch Time: 55.63 secs
...
Epoch: 13 | Loss: 0.5369 | Training accuracy: 91.465% Epoch Time: 55.11 secs
Epoch: 14 | Loss: 0.5210 | Training accuracy: 91.760% Epoch Time: 55.44 secs
Epoch: 15 | Loss: 0.5190 | Training accuracy: 91.748% Epoch Time: 54.90 secs
```
<img src="output/3layers.png" width="500"/>

```python
Validation Accuracy: 55.300%
```

### Fine-tuning the last 6 Layers
```python
Epoch: 1 | Loss: 5.9412 | Training accuracy: 22.975% Epoch Time: 54.01 secs
Epoch: 2 | Loss: 2.3124 | Training accuracy: 57.840% Epoch Time: 54.93 secs
...
Epoch: 13 | Loss: 0.0882 | Training accuracy: 99.918% Epoch Time: 54.40 secs
Epoch: 14 | Loss: 0.0759 | Training accuracy: 99.915% Epoch Time: 54.51 secs
Epoch: 15 | Loss: 0.0738 | Training accuracy: 99.925% Epoch Time: 53.50 secs
```
<img src="output/6layers.png" width="500"/>

```python
Validation Accuracy: 53.400%
```
### Fine-tuning the last 12 Layers
```python
Epoch: 1 | Loss: 5.0775 | Training accuracy: 40.832% Epoch Time: 57.40 secs
Epoch: 2 | Loss: 1.6795 | Training accuracy: 71.380% Epoch Time: 56.76 secs
...
Epoch: 13 | Loss: 0.0130 | Training accuracy: 99.983% Epoch Time: 56.99 secs
Epoch: 14 | Loss: 0.0098 | Training accuracy: 99.983% Epoch Time: 58.12 secs
Epoch: 15 | Loss: 0.0142 | Training accuracy: 99.972% Epoch Time: 56.49 secs
```
<img src="output/12layers.png" width="500"/>

```python
Validation Accuracy: 60.420%
```
### Fine-tuning the last 15 Layers
```python
Epoch: 1 | Loss: 4.6894 | Training accuracy: 53.417% Epoch Time: 56.68 secs
Epoch: 2 | Loss: 1.3141 | Training accuracy: 81.958% Epoch Time: 58.81 secs
...
Epoch: 13 | Loss: 0.0091 | Training accuracy: 99.983% Epoch Time: 56.24 secs
Epoch: 14 | Loss: 0.0055 | Training accuracy: 99.983% Epoch Time: 56.20 secs
Epoch: 15 | Loss: 0.0056 | Training accuracy: 99.983% Epoch Time: 55.34 secs
```
<img src="output/15layers.png" width="500"/>

```python
Validation Accuracy: 66.100%
```
### Fine-tuning the last 25 Layers
```python
Epoch: 1 | Loss: 4.2789 | Training accuracy: 60.365% Epoch Time: 62.52 secs
Epoch: 2 | Loss: 1.0720 | Training accuracy: 87.495% Epoch Time: 62.23 secs
...
Epoch: 13 | Loss: 0.0027 | Training accuracy: 99.983% Epoch Time: 61.55 secs
Epoch: 14 | Loss: 0.0024 | Training accuracy: 99.980% Epoch Time: 61.44 secs
Epoch: 15 | Loss: 0.0022 | Training accuracy: 99.983% Epoch Time: 61.41 secs
```
<img src="output/25layers.png" width="500"/>

```python
Validation Accuracy: 69.990%
```

### Training Times, Validation Accuracy, and Fine-Tuning Strategies

| Fine-Tuning Strategy          | Total Time (seconds) | Approx. Time (minutes:seconds) | Validation Accuracy (%) |
|-------------------------------|----------------------|--------------------------------|--------------|
| Feature Extraction            | 795.76               | 13:16                          | 50.150       |
| Fine-tuning the last 3 Layers  | 834.47               | 13:54                          | 55.300       |
| Fine-tuning the last 6 Layers  | 811.29               | 13:31                          | 53.400       |
| Fine-tuning the last 12 Layers | 856.11               | 14:16                          | 60.420       |
| Fine-tuning the last 15 Layers | 859.72               | 14:20                          | 66.100       |
| Fine-tuning the last 25 Layers | 928.52               | 15:29                          | 69.990       |

As we can see, Feature Extraction is the fastest approach but yields the lowest accuracy at 50.15%. Increasing the number of fine-tuned layers generally improves accuracy but also increases training time. For instance, fine-tuning the last 3 layers achieves 55.30% accuracy with a moderate training time, while fine-tuning the last 25 layers results in the highest accuracy of 69.99% but takes the longest. This demonstrates a trade-off between training time and model performance: more fine-tuning typically enhances accuracy but requires more time.

### Evaluation on the test set
Using the 25-layer setup since it achieved the best performance on the validation set.

```python
Epoch: 1 | Loss: 3.7327 | Training accuracy: 69.178% Epoch Time: 79.12 secs
Epoch: 2 | Loss: 0.8810 | Training accuracy: 90.884% Epoch Time: 79.13 secs
...
Epoch: 13 | Loss: 0.0009 | Training accuracy: 99.982% Epoch Time: 78.82 secs
Epoch: 14 | Loss: 0.0008 | Training accuracy: 99.982% Epoch Time: 78.91 secs
Epoch: 15 | Loss: 0.0008 | Training accuracy: 99.982% Epoch Time: 78.77 secs
```
<img src="output/finaltest.png" width="500"/>

```python
Test Accuracy: 72.180%
```
## How To Run
To run the project, you can download all the notebooks under the folder `code_files`, combine them, and execute them by their order:
`Imports.ipynb` -> `Dataset.ipynb` -> `Split.ipynb` -> `HFuncs_HParam_DataLoaders.ipynb` -> `Model_Train_Eval.ipynb`

A more efficient method (if you don't mind just viewing the outputs of the notebooks) is as follows - first, clone the repository:
```python
!git clone https://github.com/supernuper/DL_project.git
```
Then, start running the files by their order:
`Imports.ipynb` -> `Dataset.ipynb` -> `Split.ipynb` -> `HFuncs_HParam_DataLoaders.ipynb` -> `Model_Train_Eval.ipynb`
Using the following commands:
```python
%run /content/DL_project/code_files/Imports.ipynb
%run /content/DL_project/code_files/Dataset.ipynb
%run /content/DL_project/code_files/Split.ipynb
%run /content/DL_project/code_files/HFuncs_HParam_DataLoaders.ipynb
%run /content/DL_project/code_files/Model_Train_Eval.ipynb
```
(The path `/content/` assumes you're running the project in Google Colab. You can also change it to `./` or to any directory where you cloned the repository)

Notes: 
- We recommend running one line at a time to view proper execution and output.
- The command `%run` is a `magic command` and is specific to IPython environments, such as Jupyter Notebooks and Google Colab.
- About the runtime: The final file, `Model_Train_Eval.ipynb`, executes code that trains and evaluates the model 7 times - 6 times with different fine-tuning setups and once more for the final test evaluation. This process will take some time (approximately 65 seconds per epoch, totaling around 6825 seconds or 113.75 minutes). This estimate is based on running with a Google Colab T4 GPU.
- The output for the final file will include 7 evaluations: feature extraction, fine-tuning of 3, 6, 12, 15, and 25 layers, and finally the evaluation on the test set, in that order.

## Sources
* CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html
* Resnet50: https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
* images/TransferLearning.jpg: https://d3lkc3n5th01x7.cloudfront.net/wp-content/uploads/2023/07/14001536/Transfer-learning.png
