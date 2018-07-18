
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
	# input layer  :	[7]
	# hidden layer :	[14, 14, 7]
	# output layer :	[1]
	model = Sequential()
	model.add(Dense(units=14,
		input_dim=7,
		activation='relu'))
	model.add(Dense(units=14,
		activation='relu'))
	model.add(Dense(units=7,
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