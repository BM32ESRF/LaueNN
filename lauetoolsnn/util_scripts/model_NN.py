# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 23:53:21 2022

@author: PURUSHOT

Nueral network models for grid optimization

"""
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
import keras
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.constraints import maxnorm
import numpy as np


def model_arch_general(kernel_coeff = 0.0005, 
                       bias_coeff = 0.0005, 
                       init_mode='uniform',
                       learning_rate=0.0001, 
                       neurons_multiplier= [1200, 1200, 1200*7, 1200*15, 32], 
                       layers=3, 
                       batch_norm=False, 
                       optimizer="adam",
                       activation='relu',
                       dropout_rate=0.0, 
                       weight_constraint=0):
    model = Sequential()
    # Input layer
    model.add(keras.Input(shape=(neurons_multiplier[0],)))
    
    if layers > 0:
        for lay in range(layers):
            ## Hidden layer n
            if kernel_coeff == None and bias_coeff == None and\
                                weight_constraint == None and init_mode == None:
                model.add(Dense(neurons_multiplier[lay+1],))
                
            elif kernel_coeff == None and bias_coeff == None and\
                                weight_constraint == None:
                model.add(Dense(neurons_multiplier[lay+1], 
                                kernel_initializer=init_mode))
                
            elif kernel_coeff == None and bias_coeff == None:
                model.add(Dense(neurons_multiplier[lay+1], 
                                kernel_initializer=init_mode,
                                kernel_constraint=maxnorm(weight_constraint)))
                
            else:
                model.add(Dense(neurons_multiplier[lay+1], 
                                kernel_initializer=init_mode,
                                kernel_regularizer=l2(kernel_coeff), 
                                bias_regularizer=l2(bias_coeff), 
                                kernel_constraint=maxnorm(weight_constraint)))
            
            if batch_norm:
                model.add(BatchNormalization())
            model.add(Activation(activation))
            model.add(Dropout(dropout_rate))
    ## Output layer 
    model.add(Dense(neurons_multiplier[-1], activation='softmax'))
    if optimizer == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    else:
        opt = optimizer
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy", "loss"])
    return model


def model_arch_general_noarg():
    model = Sequential()
    model.add(keras.Input(shape=(900,)))
    ## Hidden layer 1
    model.add(Dense(900, ))
    model.add(Activation('relu'))
    model.add(Dropout(0.3)) ## Adding dropout as we introduce some uncertain data with noise
    ## Hidden layer 2
    model.add(Dense(((64)*10)//2,))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    ## Hidden layer 3
    model.add(Dense((64)*10,))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    ## Output layer 
    model.add(Dense(309, activation='softmax'))
    ## Compile model
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])
    return model


def model_arch_general_optimizer(optimizer='adam'):
    model = Sequential()
    model.add(keras.Input(shape=(900,)))
    ## Hidden layer 1
    model.add(Dense(900, ))
    model.add(Activation('relu'))
    model.add(Dropout(0.3)) ## Adding dropout as we introduce some uncertain data with noise
    ## Hidden layer 2
    model.add(Dense(((64)*10)//2,))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    ## Hidden layer 3
    model.add(Dense((64)*10,))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    ## Output layer 
    model.add(Dense(309, activation='softmax'))
    ## Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model


def model_arch_general_optimizer_lr(learning_rate=0.0001):
    model = Sequential()
    model.add(keras.Input(shape=(900,)))
    ## Hidden layer 1
    model.add(Dense(900, ))
    model.add(Activation('relu'))
    model.add(Dropout(0.3)) ## Adding dropout as we introduce some uncertain data with noise
    ## Hidden layer 2
    model.add(Dense(((64)*10)//2,))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    ## Hidden layer 3
    model.add(Dense((64)*10,))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    ## Output layer 
    model.add(Dense(309, activation='softmax'))
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    ## Compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    return model


def model_arch_general_weight(init_mode='uniform'):
    model = Sequential()
    model.add(keras.Input(shape=(900,)))
    ## Hidden layer 1
    model.add(Dense(900, kernel_initializer=init_mode,))
    model.add(Activation('relu'))
    model.add(Dropout(0.3)) ## Adding dropout as we introduce some uncertain data with noise
    ## Hidden layer 2
    model.add(Dense(((64)*10)//2, kernel_initializer=init_mode,))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    ## Hidden layer 3
    model.add(Dense((64)*10, kernel_initializer=init_mode,))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    ## Output layer 
    model.add(Dense(309, activation='softmax'))
    ## Compile model
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])
    return model


def model_arch_general_activation(activation='relu'):
    model = Sequential()
    model.add(keras.Input(shape=(900,)))
    ## Hidden layer 1
    model.add(Dense(900, ))
    model.add(Activation(activation))
    model.add(Dropout(0.3)) ## Adding dropout as we introduce some uncertain data with noise
    ## Hidden layer 2
    model.add(Dense(((64)*10)//2, ))
    model.add(Activation(activation))
    model.add(Dropout(0.3))
    ## Hidden layer 3
    model.add(Dense((64)*10, ))
    model.add(Activation(activation))
    model.add(Dropout(0.3))
    ## Output layer 
    model.add(Dense(309, activation='softmax'))
    ## Compile model
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])
    return model


def model_arch_general_dropout(dropout_rate=0.0):
    model = Sequential()
    model.add(keras.Input(shape=(900,)))
    ## Hidden layer 1
    model.add(Dense(900, ))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate)) ## Adding dropout as we introduce some uncertain data with noise
    ## Hidden layer 2
    model.add(Dense(((64)*10)//2, ))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    ## Hidden layer 3
    model.add(Dense((64)*10, ))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    ## Output layer 
    model.add(Dense(309, activation='softmax'))
    ## Compile model
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])
    return model


def model_arch_general_dropoutweight(dropout_rate=0.0, weight_constraint=0):
    model = Sequential()
    model.add(keras.Input(shape=(900,)))
    ## Hidden layer 1
    model.add(Dense(900, kernel_initializer='he_normal', kernel_constraint=maxnorm(weight_constraint)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate)) ## Adding dropout as we introduce some uncertain data with noise
    ## Hidden layer 2
    model.add(Dense(((64)*10)//2, kernel_initializer='he_normal', kernel_constraint=maxnorm(weight_constraint)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    ## Hidden layer 3
    model.add(Dense((64)*10, kernel_initializer='he_normal', kernel_constraint=maxnorm(weight_constraint)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    ## Output layer 
    model.add(Dense(309, activation='softmax'))
    ## Compile model
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])
    return model


def model_arch_general_neurons(neurons=1):
    model = Sequential()
    model.add(keras.Input(shape=(900,)))
    ## Hidden layer 1
    model.add(Dense(900, ))
    model.add(Activation('relu'))
    model.add(Dropout(0.3)) ## Adding dropout as we introduce some uncertain data with noise
    ## Hidden layer 2
    model.add(Dense(neurons, ))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    ## Hidden layer 3
    model.add(Dense(neurons*2, ))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    ## Output layer 
    model.add(Dense(309, activation='softmax'))
    ## Compile model
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])
    return model


def model_arch_general_layers(layers=3):
    model = Sequential()
    model.add(keras.Input(shape=(900,)))
    ## Hidden layer 1
    model.add(Dense(900, ))
    model.add(Activation('relu'))
    model.add(Dropout(0.3)) ## Adding dropout as we introduce some uncertain data with noise
    divi = np.arange(layers)[::-1]+1
    for i in range(layers):
        ## Hidden layer 2
        model.add(Dense(640//divi[i], ))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
    ## Output layer 
    model.add(Dense(309, activation='softmax'))
    ## Compile model
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])
    return model


