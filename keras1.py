#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 15:55:11 2020

@author: Govardhan

Classification with tensorflow-keras
"""
#import tensorflow package
import tensorflow as tf
import numpy as np
#load minst dataset
data = tf.keras.datasets.mnist
#split training and test data
(x_train,y_train),(x_test,y_test) = data.load_data()
#check the shape
print(x_train.shape)
print(y_train.shape)
#input shape is 28,28
#build a sequential model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                                    tf.keras.layers.Dense(128,activation='relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(10)])

#predictions
prediction = model(x_train[:1]).numpy()
prediction
tf.nn.softmax(prediction).numpy()

#loss function
loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
loss_fun(y_train[:1],prediction).numpy()
#add a compile function
model.compile(optimizer = 'adam',loss=loss_fun,metrics=['accuracy'])

#fit the model
model.fit(x_train,y_train,epochs=10)

#evaluate the accuracy of model
model.evaluate(x_test,y_test,verbose=2)



