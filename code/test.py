
"""
Created on Tue Jul 16 21:25:24 2018
@author: Ali Rasheed

"""

import os
import platform
from util import load_dataset
from weight2model import convert_model
from model import get_model
from preprocess import analyze_dataset
from sklearn import preprocessing


def test():
	# main procedure start
	if platform.system() == 'Windows':
		os.system("cls")	# for windows
	else:
		os.system("clear")	# for linux, macos

	print("\nTesting ...")

	# read test dataset
	file = "../data/financial_test.csv"
	df = load_dataset(file)
	Xnew = analyze_dataset(df)

	# load model
	weight = 'weight_model.hdf5'
	model = convert_model(weight)
	model.summary()

	# make a prediction
	ynew = model.predict_classes(Xnew)

	# show the inputs and predicted outputs
	for i in range(len(Xnew)):
		if ynew[i] == 0:
			class_label = 'Normal'
		else:
			class_label = 'Fradulent'

		print("%d --> class=%s" % (i, class_label))


# call main function
if __name__ == '__main__':
	test()