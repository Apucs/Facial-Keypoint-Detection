[//]: # (Image References)

[image1]: ./images/key_pts_example.png "Facial Keypoint Detection"

# Facial Keypoint Detection

## Project Overview

This is an project from udacity computer vision nanodegree. In this project, I have used computer vision techniques and deep learning architectures to build a facial keypoint detection system. Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition. My model is able to look at any image, detect faces, and predict the locations of facial keypoints on each face; examples of these keypoints are displayed below.

![Facial Keypoint Detection][image1]

__Notebook 1__ : Loading and Visualizing the Facial Keypoint Data

__Notebook 2__ : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

__Notebook 3__ : Facial Keypoint Detection Using Haar Cascades and your Trained CNN

__Notebook 4__ : Fun Filters and Keypoint Uses


### Local Environment Instruction## Project Instructions
1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
```
git clone https://github.com/Apucs/Facial-Keypoint-Detection.git
cd Facial-Keypoint-detection
```

2. Create (and activate) a new environment, named `cv-nd` with Python 3.6. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__: 
	```
	conda create -n cv-nd python=3.6
	source activate cv-nd
	```
	- __Windows__: 
	```
	conda create --name cv-nd python=3.6
	activate cv-nd
	```

3. Install PyTorch and torchvision; this should install the latest version of PyTorch.
	
	- __Linux__ or __Mac__: 
	```
	conda install pytorch torchvision -c pytorch 
	```
	- __Windows__: 
	```
	conda install pytorch-cpu -c pytorch
	pip install torchvisionhttps://github.com/Apucs/Facial-Keypoint-Detection.git
	```

4. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```
### Model Architecture
| Layer name | Layer shape |
| ------ | ------ |
| Input | 1,224,224 |
| Convolutional2d_1 | 32,221,221 |
| activation | 32,221,221 |
| Maxpooling2d_1 | 32,110,110 |
| Dropout_1 | 32,110,110 |
| Convolutional2d_2 | 64,108,108 |
| activation | 64,108,108 |
| Maxpooling2d_2 | 64,54,54 |
| Dropout_2 | 64,54,54 |
| Convolutional2d_3 | 128,53,53 |
| activation | 128,53,53 |
| Maxpooling2d_3 | 128,26,26 |
| Dropout_3 | 128,26,26 |
| Convolutional2d_4 | 256,26,26 |
| activation | 256,26,26 |
| Maxpooling2d_4 | 256,13,13 |
| Dropout_4 | 256,13,13 |
| Flatten | 43264 |
| Dense_1 | 1024 |
| Activation | 1024 |
| Dropout_5 | 1024 |
| Dense_2 | 512 |
| Activation | 512 |
| Dropout_6 | 512 |
| Dense_3 | 136 |




LICENSE: This project is licensed under the terms of the MIT license.
