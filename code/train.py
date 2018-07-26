
"""
Created on Tue Jul 9 8:16:06 2018
@author: Ali Rasheed

"""

import os
from util import load_dataset
from preprocess import preprocess_dataset
from model import get_model


def train():
	# main procedure start
	os.system("cls")
	print("\nTraining procedure...")

	# read dataset
	# file = "../data/creditcard.csv"
	file = "../data/financial_log.csv"
	df = load_dataset(file)

	# preprocess the dataset
	# it may be not essential
	X, Y = preprocess_dataset(df)

	# load model and see the model architecture
	# transfer learning
	model = get_model()

	# model summary
	model.summary()

	# fit and save the model for later usage
	model.fit(X, Y,
		validation_split=0.20,	# we use 20% dataset for validation
		epochs=20,				# simply set the maximum log as 20
		batch_size=32)			# batch training for saving training time
	model.save_weights('weight_model.hdf5')

	# evaluate the model
	scores = model.evaluate(X, Y)
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


if __name == '__main__':
	train()