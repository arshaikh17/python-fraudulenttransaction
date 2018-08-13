
"""
Created on Tue Jul 9 8:16:06 2018
@author: Ali Rasheed

"""

import os
import platform
from util import load_dataset
from model import get_model
from preprocess import preprocess_dataset
import matplotlib.pyplot as plt


def train():
	if platform.system() == 'Windows':
		os.system("cls")		# for windows
	else:
		os.system("clear")	# for linux, macos

	# main procedure start
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
	history = model.fit(X, Y,
		validation_split=0.20,	# we use 20% dataset for validation
		epochs=10,				# simply set the maximum log as 20
		batch_size=32)			# batch training for saving training time
	model.save_weights('weight_model.hdf5')

	# evaluate the model
	scores = model.evaluate(X, Y)
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

	# show training graph
	# list all data in history
	print(history.history.keys())
	
	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()


if __name__ == '__main__':
	train()