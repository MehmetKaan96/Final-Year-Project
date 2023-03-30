import tensorflow as tf
from tensorflow.keras import layers
from keras.optimizers import Adam
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

from keras.models import Sequential, load_model

model = load_model('ResNet_model.h5')
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))



