# Useful starting lines
#%matplotlib inline
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, LSTM, Merge
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Reshape
import scipy
from mask_to_submission import *
from function import *



#Definition of the model so that we can load the weights

# first part
model_cnn = Sequential()
model_cnn.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', input_shape=(24, 24, 3)))
model_cnn.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn.add(Dropout(0.25)) 

model_cnn.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model_cnn.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn.add(Dropout(0.25))

model_cnn.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
model_cnn.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn.add(Dropout(0.25))

model_cnn.add(Flatten())
#model_cnn.add(Dense(1024, activation='relu')) 
#model_cnn.add(Dropout(0.5))
model_cnn.add(Dense(1024, activation='relu'))
model_cnn.add(Dropout(0.5))

# second part
model_cnn2 = Sequential()
model_cnn2.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', input_shape=(24, 24, 3)))
model_cnn2.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
model_cnn2.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn2.add(Dropout(0.25)) 

model_cnn2.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model_cnn2.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model_cnn2.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn2.add(Dropout(0.25))

model_cnn2.add(Flatten())
model_cnn2.add(Dense(1024, activation='relu'))
model_cnn2.add(Dropout(0.5))

# merging part
merged = Merge([model_cnn, model_cnn2], mode='ave')

final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(256, activation='sigmoid'))


# The neural network was trained previously and the weights saved
# First thing to do is thus to load it.
final_model.load_weights("model_mergeCNN_final.h5")

# To get the submission file, the images on which the model will be tested have to be loaded.
image_filenames = []
for i in range(1, 51):
    image_filename = 'data/test/test_%d' % i + '/test_%d' % i + '.png'
    image_filenames.append(image_filename)

pad = 4
images_test = [np.pad(load_image(image_filename), ((pad,pad),(pad,pad),(0,0)), 'constant')
               for image_filename in image_filenames]

#------------------------------------------------------------------------------------------------
# Generate prediction
#------------------------------------------------------------------------------------------------

# parameters
w = images_test[0].shape[0] - 2*pad
h = images_test[0].shape[1] - 2*pad
patch_size = 16

# for each picture of the testset
for i in range(0, 50):
    
    # format the image for the neural net
    X_test = np.asarray(img_crop_padded(images_test[i], patch_size, patch_size, pad))
    
    # Make the prediction
    prediction = final_model.predict([X_test, X_test], batch_size=32)
    
    # binarization of the output
    # 1 for road and 0 otherwise
    predict_binary = np.where(prediction > 0.5, 1, 0)
    
    # Reshapping the linear output into patch of (16,16)
    im_test = deflatten(predict_binary, patch_size)
    # Transform it into numpy array
    im_non_flat = np.asarray(im_test)
    
    # Transform the patches into a complete black and white image
    predicted_im = label_to_img(w, h, patch_size, patch_size, im_non_flat)
    
    j = i + 1
    
    # Save each image into a folder prediction (already created)
    scipy.misc.imsave('data/prediction/resultImage_' + '%.3d' % j + '.png', predicted_im)

#------------------------------------------------------------------------------------------------
# Make submission file
#------------------------------------------------------------------------------------------------

submission_filename = 'submission.csv'
image_filenames = []
for i in range(1, 51):
    image_filename = 'data/prediction/resultImage_' + '%.3d' % i + '.png'
    image_filenames.append(image_filename)

masks_to_submission(submission_filename, *image_filenames)



