# PCML Project2: Road Segmentation

## Axel de la Harpe, Clémentine Auguet, María Cervera - group_acm

This folder contains all the necessary information to run and see our methodology regarding the Road Segmentation for PCLM project 2.
The question addressed in this project is related to the development of system capable of automatically extracting roads from urban areas.
The dataset provided for this project consists of 100 satellite images acquired from GoogleMaps, as well as the corresponding ground-truth images in which each pixel is label as road or background. 


Among all the files present in this folder, two notebooks are included. They detail the two approaches implemented for the road segmentation task.
- random_forest.ipynb contains the random forest classifierbuilt with scikit-learn
- cnn_final_model.ipynb contains the final convolutional neural network built with keras and tensorflow


The fold named Data contains 3 subfolders: 
- The prediction folder: contains the images predicted by our best CNN model.
- The test folder: actually is empty (due to lack of space for the submission). Before running any file, the test set should be placed here.
- The training folder: is also empty and corresponds to the folder where the training set should be placed.


### Getting Started
The CNN model has been implemented using the library Keras. This is a neural network library that allows to run NN model on top of tensorflow. You need to have Tensorflow installed on your machine in order to use it.
To install keras use the following command in your terminal :
```
sudo pip install keras
```
see [Keras](https://keras.io/#installation) webpage for more details

If you have Tensorflow installed correctly, keras should run automatically on top of it. If there is a problem, it is possible that your version try to run on top of Theano. In this case have a look at see [Keras backend page](https://keras.io/backend/)  in order to change the default.


### Obtaining the final submission file
The submission file corresponding to our final and best model can be obtained by running the run.py file in the terminal 
```
pyhton run.py
```
This file will use the functions present in the *function.py* file and *mask_to_submission.py*.
As the training part of the model (explained below on the CNN notebook part) is time consuming, the weights of the corresponding model were saved and are simply loaded in the *run.py*.
Before running it, training and test datasets should be added to the good folder, as described above.

## CNN NOTEBOOK
The CNN notebook is constructed in subsequent cells as followed:

#### Library and function loading
All library and needed function are loaded in the notebook. 
Principally all the keras functions necessary for the construction of the neural network

#### Image loading
The 100 images of the data set are loaded, zero padded with 4 pixels at each sides

#### Processing
- Images are cropped into 24X24 pixels patches. Groundtruth images are cropped into corresponding 16x16 pixels patches.
- Patches are linearized 
- These value are renamed as the training samples: X_train, Y_train

#### Neural network model architecture
- Definition of the first CNN model 
- Definition of the second CNN model 
- Merging of the two CNN

#### Compilation of the model
- Definition of a stochastic gradient optimizer
- Compilation of the model with "mse" loss function and the sgd optimizer

#### Fitting the model to the training samples
- The model if fitted to the train data. This is the computation intensive and time consumming part of the notebook. It has been put in comment in the notebook to avoid accidental running it. The weights of our final model have already been saved and are loaded later. The cell can be runned but will take several hours to run (~12h ^^').
- A summary of the model is displayed, showing each layer constructing the model with the corresponding number of parameters involved.
- The weights were saved. (model_mergeCNN_final.h5)
- The weights are loaded from the previously saved version.

#### Testing the model on 1 image of the training set
- 1 image is selected and cropped as before.
- The model predict the road on this image.
- The size of the prediction output are displayed
- The data are binarized as they are outputed as the results of a sigmoid function. Value more than 0.5 are transformed into 1 and less are transformed into 2.
- The predicted value are reshaped back into corresponding patches.
- The predicted roads are displayed in red over the real image as red patch.
- The real groundTruth image is displayed for comparison

#### Build submission file on new images (test set)
- Test images are loaded and zero padded
- Each test images is cropped as before and the model predict the road from this inputs patches. The same binarization and reshaping as for the test on 1 image is done for each images.
- The files are saved as .png into the folder prediction

#### Create submit.csv
- A .csv file is created from this .png image results.

## RFC NOTEBOOK
The random forest notebook is constructed in subsequent cells as followed:

#### Loading of the libraries

#### Define helper functions
- Definition of helpers functions

#### Load and visualize images
- Loading of the images and groundtruth images
- Visualization of the loaded images

#### Build features
- Definition of the functions to extract the features from the images.
- Defininiton of the function to prepare the features so that they can be feed to the classifier

#### Classifiers training
- Definition of the function that compute the training error of the classifier
- Definition of the Logistic regression classifier
- Training of the Logistic regression classifier 
- Output of the results of the Logisitc regressio classifier

#### Random forest classifier
- Definition of the random forest classifier model
- Training of the random forest classifier model
- output of the results of the random forest classifier model

Then:
- Definition, training and output of the results of two other random forest classifier build with different parameters and number of features.

#### Prepare submission on test images:
- Loading of the test images 
- Building of the features
- Training of the model on all images
- Saving of the results in the folders prediction_RFC into the folder data

END:










