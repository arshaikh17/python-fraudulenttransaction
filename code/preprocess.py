
"""
Created on Tue Jul 9 8:16:06 2018
@author: Paul

"""
import time
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder


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
def preprocess_dataset(X, Y):
	# start time tracker
	print("\nEncoding train dataset by 'One hot encoding'")
	start = time.time()

	# encode the dataset field using "One hot encoding"
	# encoder = LabelEncoder()
	# encoder.fit(X)
	# encoded = encoder.transform(X)
	# result = np_utils.to_categorical(encoded)

	# calculate the elapsed time
	end = time.time()
	print("Done. Elapsed time : %f seconds" % (end - start))

	# return categorical result
	return X, Y
