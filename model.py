from utils import load_data
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

(feature , labels) = load_data()
X_train , X_test , y_train , y_test = train_test_split( feature, labels , test_size = 0.1)

categories = ['daisy','dandelion','rose','sunflower','tulip']

input_layer = tf.keras.layers.Input([128,128,3])

conv1 = tf.keras.layers.Conv2D(filters = 32 , kernel_size = (5,5) , padding ='Same',
        activation='relu')(input_layer)
pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))( conv1)

conv2 = tf.keras.layers.Conv2D(filters = 64 , kernel_size = (3,3) , padding ='Same',
        activation='relu')(pool1)
pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2) , strides=(2,2))(conv2)

conv3 = tf.keras.layers.Conv2D(filters = 96 , kernel_size = (3,3) , padding ='Same',
        activation='relu')(pool2)
pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2) , strides=(2,2))(conv3)

conv4 = tf.keras.layers.Conv2D(filters = 96 , kernel_size = (3,3) , padding ='Same',
        activation='relu')(pool3)
pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2,2) , strides=(2,2))(conv4)

flatten = tf.keras.layers.Flatten()(pool4)
dense = tf.keras.layers.Dense(512 , activation = 'relu')(flatten)
out = tf.keras.layers.Dense(5 , activation='softmax' )(dense)

model = tf.keras.Model(input_layer , out,)

model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy',
                metrics = ['accuracy'])

model.fit(X_train , y_train , batch_size=100 , epochs=10)

model.save('model.h5')