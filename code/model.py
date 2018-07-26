
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
	# hidden layer :	[14, 28, 14, 7, 1]
	# output layer :	[1]
	model = Sequential()

	# hidden layer
	# we can use many activation functions here.
	# for non-linear function, we often use ReLU.
	model.add(Dense(units=14,
		input_dim=7,
		activation='relu'))
	model.add(Dense(units=28,
		activation='relu'))
	model.add(Dense(units=14,
		activation='relu'))
	model.add(Dense(units=7,
		activation='relu'))

	# output layer
	model.add(Dense(units=1,
		activation='sigmoid'))

	# load model weights
	# model.load_weights('weight_model.hdf5')

	# compile the neural network model
	# there are only 2 classes - real and fraud
	# we can simulate this as 0 and 1 (binary representation)
	model.compile(loss='binary_crossentropy',
		optimizer='adam',
		metrics=['accuracy'])

	# return neural network model
	return model