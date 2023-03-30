# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 13:12:44 2023

@author: Mehmet
"""

import os
from glob import glob
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras import optimizers, callbacks
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'}

#Lesion Dictionary
lesion_code_dict = {
    'nv': 0,
    'mel': 1,
    'bkl': 2,
    'bcc': 3,
    'akiec': 4,
    'vasc': 5,
    'df': 6}

image_path = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join('dataset/mdataset/', '*', '*.jpg'))}

skin_df = pd.read_csv('dataset/mdataset/HAM10000_metadata.csv')

skin_df['path'] = skin_df['image_id'].map(image_path.get)
skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get) 
skin_df['cell_type_idx'] = skin_df['dx'].map(lesion_code_dict.get)       
        
skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((64,64))))  

def df_prep(skin_df):
    features=skin_df.drop(columns=['cell_type_idx'],axis=1)
    target=skin_df['cell_type_idx']

    # Create First Train and Test sets
    x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.20,random_state=123)

    #The normalisation is done using the training set Mean and Std. Deviation as reference
    x_train = np.asarray(x_train_o['image'].tolist())
    x_test = np.asarray(x_test_o['image'].tolist())

    x_train_mean = np.mean(x_train)
    x_train_std = np.std(x_train)

    x_train = (x_train - x_train_mean)/x_train_std
    x_test = (x_test - x_train_mean)/x_train_std

    # Perform one-hot encoding on the labels
    y_train = to_categorical(y_train_o, num_classes = 7)
    y_test = to_categorical(y_test_o, num_classes = 7)

    #Splitting training into Train and Validatation sets
    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.1,random_state=123)

    #Reshaping the Images into 3 channels (RGB)
    x_train = x_train.reshape(x_train.shape[0], *(64,64, 3))
    x_test = x_test.reshape(x_test.shape[0], *(64,64, 3))
    x_validate = x_validate.reshape(x_validate.shape[0], *(64,64, 3))
    
    return x_train,x_validate,x_test,y_train,y_validate,y_test

x_train,x_validate,x_test,y_train,y_validate,y_test = df_prep(skin_df) 
   
input_shape = (64,64, 3)
num_classes = 7

optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

epochs = 2
batch_size = 64

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape= input_shape,padding="same",activation="relu"))
model.add(Conv2D(64, (3,3), padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(128, (3,3), padding="same",activation="relu"))
model.add(Conv2D(128, (3,3), padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(256, (3,3), padding="same",activation="relu"))
model.add(Conv2D(256, (3,3), padding="same",activation="relu"))
model.add(Conv2D(256, (3,3), padding="same", activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(512, (3,3), padding="same",activation="relu"))
model.add(Conv2D(512, (3,3), padding="same",activation="relu"))
model.add(Conv2D(512, (3,3), padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(512, (3,3), padding="same",activation="relu"))
model.add(Conv2D(512, (3,3), padding="same",activation="relu"))
model.add(Conv2D(512, (3,3), padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation="relu"))
model.add(Dense(4096, activation="relu"))
model.add(Dense(7, activation="softmax"))

model.summary()

# dataaugment_baseline = ImageDataGenerator(
#         featurewise_center=False,  # set input mean to 0 over the dataset
#         samplewise_center=False,  # set each sample mean to 0
#         featurewise_std_normalization=False,  # divide inputs by std of the dataset
#         samplewise_std_normalization=False,  # divide each input by its std
#         zca_whitening=False,  # apply ZCA whitening
#         rotation_range=90,  # randomly rotate images in the range (degrees, 0 to 180)
#         zoom_range = 0.1, # Randomly zoom image 
#         width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#         height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#         horizontal_flip=True,  # randomly flip images
#         vertical_flip=True,  # randomly flip images
#         shear_range = 10) 

# dataaugment_baseline.fit(x_train)


# accuracy = []
# for i in range(0,3):
#     model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
#     history_baseline = model.fit(dataaugment_baseline.flow(x_train,y_train, batch_size=batch_size),
#                             epochs = epochs, validation_data = (x_validate,y_validate),
#                             verbose = 0, steps_per_epoch=x_train.shape[0] // batch_size,shuffle = False)

#     #Predictions and Baseline Accuracy
#     baseline_predictions = model.predict(x_test)
#     _v, baseline_accuracy_v = model.evaluate(x_validate, y_validate, verbose=0)
#     _t, baseline_accuracy_t = model.evaluate(x_train, y_train, verbose=0)
#     _, baseline_accuracy = model.evaluate(x_test, y_test, verbose=0)
#     accuracy.append(baseline_accuracy)

accuracy = []
for i in range(0,3):
    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
    history_baseline = model.fit(x_train,y_train, batch_size=batch_size,
                            epochs = epochs, validation_data = (x_validate,y_validate),
                            verbose = 0, steps_per_epoch=x_train.shape[0] // batch_size,shuffle = False)

    #Predictions and Baseline Accuracy
    baseline_predictions = model.predict(x_test)
    _v, baseline_accuracy_v = model.evaluate(x_validate, y_validate, verbose=0)
    _t, baseline_accuracy_t = model.evaluate(x_train, y_train, verbose=0)
    _, baseline_accuracy = model.evaluate(x_test, y_test, verbose=0)
    accuracy.append(baseline_accuracy)


# model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# history = model.fit(dataaugment_baseline.flow(x_train,y_train, batch_size=batch_size),
#                     epochs = epochs,
#                     validation_data = (x_validate,y_validate), verbose=0,
#                     steps_per_epoch = x_train.shape[0] // batch_size,shuffle = False)

# score = model.evaluate(x_test, y_test)
# print("Test accuracy: %.4f" %score[1])
 

print("Baseline Results")
print("Training: accuracy = %f" % (baseline_accuracy_t))
print("Validation: accuracy = %f" % (baseline_accuracy_v))
print("Test: accuracy = %.2f +/-%.4f" % (np.mean(accuracy),np.std(accuracy)))


