
"""
Created on Tue Jul 9 8:16:06 2018
@author: Ali Rasheed

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
	# start time tracker
	print("\nEncoding train dataset")
	start = time.time()

	# convert to numeric data
	df = convert(df)
	print("DATASET OVERVIEW")
	print(df.head())

	# variables
	dataset = df.values
	X = dataset[:, 0:9]
	Y = dataset[:, 9]

	X = preprocessing.scale(X)

	# calculate the elapsed time
	end = time.time()
	print("Done. Elapsed time : %f seconds" % (end - start))

	# return categorical result
	return X, Y


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

	return df