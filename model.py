from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf
from keras.callbacks import TensorBoard
from collections import deque
import random
import json

class model:
	def buildmodel():
	    print("Now we build the model")
	    model = Sequential()
	    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(img_cols,img_rows,img_channels)))  #80*80*4
	    model.add(Activation('relu'))
	    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
	    model.add(Activation('relu'))
	    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
	    model.add(Activation('relu'))
	    model.add(Flatten())
	    model.add(Dense(512))
	    model.add(Activation('relu'))
	    model.add(Dense(ACTIONS))
	    adam = Adam(lr=LEARNING_RATE)
	    model.compile(loss='mse',optimizer=adam)
	    print("We finish building the model")
	    return model

