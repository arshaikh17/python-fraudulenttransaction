
"""
Created on Tue Jul 9 8:19:45 2018
@author: Ali Paul Raita

"""
import time
import numpy as np
from keras.utils import np_utils
from sklearn import preprocessing


"""
header = [
	'step',
	'type',
	'amount',
	'nameOrig',
	'oldbalanceOrg',
	'newbalanceOrg',
	'nameDest',
	'oldbalanceDest',
	'newbalanceDest',
	'isFraud',
	'isFlaggedFraud'
]

"""

# preprocess the dataset
def preprocess_dataset(df):
	# need to check if this is useful
	df = df.loc[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]
	# df.loc[df.type == 'PAYMENT', 'type'] = 0
	df.loc[df.type == 'TRANSFER', 'type'] = 0
	# df.loc[df.type == 'CASH_IN', 'type'] = 2
	df.loc[df.type == 'CASH_OUT', 'type'] = 1
	# df.loc[df.type == 'DEBIT', 'type'] = 4

	# convert dtype('O') to dtype(int)
	df.type = df.type.astype(int)
	
	# return categorical result
	return df


# get the training dataset X, Y
def get_training_dataset(df):
	# start time tracker
	print("\nPreprocessing train dataset")
	start = time.time()

	# convert to numeric data
	# show first 20 rows dataset for poc
	# df = convert(df)
	print("DATASET OVERVIEW")
	print(df.head(20))

	# preprocessing
	df = preprocess_dataset(df)

	# show first 20 rows dataset for poc
	# df = convert(df)
	print("EXTRACT OVERVIEW")
	print(df.head(20))

	dataset = df.values

	# training dataset
	X = dataset[:, 0:6]
	Y = dataset[:, 6]

	# calculate the elapsed time
	end = time.time()
	print("Done. Elapsed time : %f seconds" % (end - start))

	# return
	return X, Y


# get the test dataset X
def get_test_dataset(df):
	# preprocessing
	df = preprocess_dataset(df)
	dataset = df.values

	# test dataset
	X = dataset[:, 0:6]

	# return
	return X


# convert non-numeric data to numeric data format
def convert(df):
	columns = df.columns.values

	# investigate the feature elements
	for column in columns:
		text_digit_vals = {}

		# change the non-numeric unit to numeric one
		def convert_to_int(val):
			return text_digit_vals[val]

		# check the datatype (int64 or float64)
		# convert the non-numeric value to numeric value
		if df[column].dtype != np.int64 and df[column].dtype != np.float64:
			column_contents = df[column].values.tolist()
			unique_elements = set(column_contents)

			# change the element to corresponding value
			x = 0
			for unique in unique_elements:
				if unique not in text_digit_vals:
					text_digit_vals[unique] = x
					x += 1

			df[column] = list(map(convert_to_int, df[column]))

	print(text_digit_vals)
	return df