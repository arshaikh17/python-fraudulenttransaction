
"""
Created on Tue Jul 9 9:12:28 2018
@author: Paul Raita

"""

from keras.layers import Dense
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.models import Sequential


# return neural network model
def get_model():
	print("\nLoaded training NN model")

	# build neural network using Keras python machine learning library
	# input layer  :	[6]
	# hidden layer :	[12, 30, 24, 6]
	# output layer :	[1]
	model = Sequential()

	# hidden layer
	# we can use many activation functions here.
	# for non-linear function, we often use ReLU.
	model.add(Dense(units=12, input_dim=6, use_bias=False))
	model.add(BatchNormalization())
	model.add(Activation("relu"))

	model.add(Dense(units=30, use_bias=False))
	model.add(BatchNormalization())
	model.add(Activation("relu"))

	model.add(Dense(units=24, use_bias=False))
	model.add(BatchNormalization())
	model.add(Activation("relu"))

	model.add(Dense(units=6, use_bias=False))
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	
	# output layer
	model.add(Dense(units=1, use_bias=False))
	model.add(BatchNormalization())
	model.add(Activation("sigmoid"))

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