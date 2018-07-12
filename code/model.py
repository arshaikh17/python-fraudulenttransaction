
"""
Created on Tue Jul 9 8:16:06 2018
@author: Ali Rasheed

"""

from keras.layers import Dense
from keras.models import Sequential


# return neural network model
def get_model():
	print("\nLoaded training NN model")

	# build neural network using Keras python machine learning library
	# input layer  :	9
	# hidden layer :	18
	# output layer :	1
	model = Sequential()
	model.add(Dense(units=18,
		input_dim=9,
		activation='relu'))
	model.add(Dense(units=9,
		activation='relu'))
	model.add(Dense(units=1,
		activation='sigmoid'))

	# load model weights
	# model.load_weights('weight_model.h5')

	# compile the neural network model
	model.compile(loss='binary_crossentropy',
		optimizer='adam',
		metrics=['accuracy'])

	# return neural network model
	return model