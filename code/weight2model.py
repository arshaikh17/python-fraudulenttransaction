
"""
Created on Tue Jul 15 10:19:57 2018
@author: Paul Raita

"""

from keras.layers import Dense
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.models import Sequential


# covnert from weight model to common model (h5 format)
def convert_model(weight_h5):
	print("\nConverting weight information to model ...")

	# build neural network using Keras python machine learning library
	# input layer  :	[6]
	# hidden layer :	[12, 30, 15, 6]
	# output layer :	[1]
	model = Sequential()
	# model.add(Dense(units=12,
	# 	input_dim=6,
	# 	activation='relu'))
	# model.add(Dense(units=30,
	# 	activation='relu'))
	# model.add(Dense(units=15,
	# 	activation='relu'))
	# model.add(Dense(units=6,
	# 	activation='sigmoid'))

	# 	# output layer
	# model.add(Dense(units=1,
	# 	activation='sigmoid'))

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
	model.load_weights(weight_h5)

	# compile the neural network model
	model.compile(loss='binary_crossentropy',
		optimizer='adam',
		metrics=['accuracy'])

	# save neural network model for compatibility
	model.save('nn_model.h5')
	return model
	

# main method
if __name__ == "__main__":
	convert_model()