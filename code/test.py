
"""
Created on Tue Jul 16 21:25:24 2018
@author: Paul Raita

"""

import os
import platform
from util import load_dataset
from weight2model import convert_model
from model import get_model
from preprocess import get_test_dataset
from sklearn import preprocessing


def test():
	# main procedure start
	if platform.system() == 'Windows':
		os.system("cls")	# for windows
	else:
		os.system("clear")	# for linux, macos

	print("\nTesting ...")

	# read test dataset
	# file = "../data/paysim_normal.csv"
	file = "../data/paysim_fraud.csv"
	df = load_dataset(file)
	Xnew = get_test_dataset(df)

	# load model
	weight = 'weight_model.hdf5'
	model = convert_model(weight)
	model.summary()

	# make a prediction
	ynew = model.predict_classes(Xnew)
	probs = model.predict_proba(Xnew)

	# show the inputs and predicted outputs
	for i in range(len(Xnew)):
		if ynew[i] == 0:
			class_label = 'Normal'
			prob = 50 + 100 * (0.5 - probs[i])
		else:
			class_label = 'Fradulent'
			prob = 50 + 100 * (probs[i] - 0.5)

		print("%d --> class=%s, confidence=%.2f%%" % (i, class_label, prob))


# call main function
if __name__ == '__main__':
	test()