# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:58:17 2023

@author: Mehmet
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SIZE = 100

skin_df = pd.read_csv('dataset/mdataset/HAM10000_metadata.csv')

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

lesion_ID_dict = {
    'nv': 0,
    'mel': 1,
    'bkl': 2,
    'bcc': 3,
    'akiec': 4,
    'vasc': 5,
    'df': 6
}

lesion_names = ['Melanocytic nevi','Melanoma','Benign keratosis-like lesions ',
               'Basal cell carcinoma','Actinic keratoses','Vascular lesions',
               'Dermatofibroma']

lesion_names_short = ['nv','mel','bkl','bcc','akiec','vasc','df']

skin_df['lesion_type']=skin_df['dx'].map(lesion_type_dict)
skin_df['lesion_ID'] = skin_df['dx'].map(lesion_ID_dict)

print('Total number of images',len(skin_df))

skin_df['lesion_type'].value_counts()

    
import cv2
from cv2 import imread, resize

def produce_new_img(img2):
    imga = cv2.rotate(img2,cv2.ROTATE_90_CLOCKWISE)
    imgb = cv2.rotate(img2,cv2.ROTATE_90_COUNTERCLOCKWISE)
    imgc = cv2.rotate(img2,cv2.ROTATE_180)
    imgd = cv2.flip(img2,0)
    imge = cv2.flip(img2,1)
    return imga,imgb,imgc,imgd,imge
    
import os
X = []
y = []

lista1 = os.listdir('dataset/mdataset/images1')
lista2 = os.listdir('dataset/mdataset/images2')


#import images from folder 1
for i in range(len(lista1)):
    fname_image = lista1[i]
    fname_ID = fname_image.replace('.jpg','')
    
    file_to_read ='dataset/mdataset/images1/'+str(fname_image)
    img = imread(file_to_read)
    img2 = resize(img,(SIZE,SIZE))
    X.append(img2)
    
    output = np.array(skin_df[skin_df['image_id'] == fname_ID].lesion_ID)
    y.append(output[0])
    
    if output != 0:
        new_img = produce_new_img(img2)
        for i in range(5):
            X.append(new_img[i])
            y.append(output[0])
       
    if i % int(100) == 0:
        print(i,'images loaded')

for i in range(len(lista2)):
    fname_image = lista2[i]
    fname_ID = fname_image.replace('.jpg','')
    
    file_to_read ='dataset/mdataset/images2/'+str(fname_image)
    img = imread(file_to_read)
    img2 = resize(img,(SIZE,SIZE))
    X.append(img2)
    
    output = np.array(skin_df[skin_df['image_id'] == fname_ID].lesion_ID)
    y.append(output[0])
    
    if output != 0:
        new_img = produce_new_img(img2)
        for i in range(5):
            X.append(new_img[i])
            y.append(output[0])
    
    if i % int(100) == 0:

        print(len(lista1)+i,'images loaded')

from keras.utils.np_utils import to_categorical

X = np.array(X)
y = np.array(y)

y_train = to_categorical(y, num_classes=7)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_train, test_size=0.30, random_state=50,stratify=y)

print('Train dataset shape',X_train.shape)
print('Test dataset shape',X_test.shape)

import keras
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers.core import Dropout, Activation
from keras.layers import Conv2D,BatchNormalization,MaxPool2D,Flatten,Dense

from sklearn.utils.class_weight import compute_class_weight
y_id = np.array(skin_df['lesion_ID'])

# compute weights for the loss function, because the problem is unbalanced
class_weights = np.around(compute_class_weight(class_weight='balanced',classes=np.unique(y_id),y=y),2)
class_weights = dict(zip(np.unique(y_id),class_weights))

import tensorflow as tf
from tensorflow.keras import layers
from keras.optimizers import Adam
#resnet152 - 564 layers
pre_trained_model = tf.keras.applications.ResNet152V2(include_top=False,
                             input_shape=(SIZE, SIZE, 3),
                             weights='imagenet')

for layer in pre_trained_model.layers[:1]:
    layer.trainable = False
for layer in pre_trained_model.layers[1:]:
    layer.trainable = True


model = tf.keras.models.Sequential([
    pre_trained_model,
    layers.Flatten(),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')
])


from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

early_stopping = EarlyStopping(monitor='val_accuracy',
                               mode='max',
                               patience=10)

checkpoint = ModelCheckpoint('ResNet_model.h5',
                             monitor='val_accuracy',
                             mode='max',
                             save_best_only=True,
                             verbose=1)

callback_list = [early_stopping, checkpoint]

batch_size = 32
epochs = 50
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-3)
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(zoom_range = 0.2, horizontal_flip=True, shear_range=0.2)

datagen.fit(X_train)

history=model.fit(datagen.flow(X_train,y_train), 
                  epochs=epochs, 
                  batch_size=batch_size, shuffle=True, 
                  callbacks=callback_list, 
                  validation_data=(X_test, y_test), 
                  class_weight=class_weights)


