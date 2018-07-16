
"""
Created on Tue Jul 15 10:19:57 2018
@author: Ali Rasheed

"""

from keras.layers import Dense
from keras.models import Sequential


# covnert from weight model to common model (h5 format)
def convert_model(weight_h5):
	print("\nConverting weight information to model ...")

	# build neural network using Keras python machine learning library
	# input layer  :	[9]
	# hidden layer :	[18, 24, 9]
	# output layer :	[1]
	model = Sequential()
	model.add(Dense(units=18,
		input_dim=9,
		activation='relu'))
	model.add(Dense(units=24,
		activation='relu'))
	model.add(Dense(units=9,
		activation='relu'))
	model.add(Dense(units=1,
		activation='sigmoid'))

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