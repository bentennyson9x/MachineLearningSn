import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.impute import SimpleImputer
from mpl_toolkits.mplot3d import axes3d
from sklearn import tree
from mpl_toolkits.mplot3d import axes3d
from tensorflow import nn ## de goi cac active function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split  ## dung de tach bo test ra
from sklearn.datasets import load_iris
from keras.utils import np_utils ## dung de categorical cac label
## ham train_test_split se tach bo train voi bo test ra rieng biet de xay dung model voi nhung thu vien dataset
## ham train_test_split se tra ve 4 doi so,  doi so 1 danh cho x_train dung de trainning voi doi so thu 3 y_train
## 2 doi so con lai tuong tu nhung dung de test
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_train=x_train.astype("float32")
x_train=x_train/255
print(y_train)
y_train=np_utils.to_categorical(y_train,10)
print(y_train)
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(28,28,1),activation=nn.relu))
model.add(MaxPooling2D(pool_size=(2,2),strides=1))
model.add(Conv2D(filters=32,kernel_size=(3,3),activation=nn.relu))
model.add(MaxPooling2D(pool_size=(2,2),strides=1))
model.add(Flatten())
model.add(Dense(300,activation=nn.relu))
model.add(Dense(10,activation=nn.softmax))
model.compile(optimizer='sgd',loss='mean_squared_error',metrics=['accuracy'])
model.summary()
model.fit(x_train,y_train,epochs=1)