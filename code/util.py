
"""
Created on Tue Jul 10 13:24:41 2018
@author: Ali Rasheed

"""

import time
import pandas


# return training input and output data
def load_dataset(file):
	# DATASET HEADER FORMAT
	#
	# step - maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation).
	# type - CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.
	# amount - amount of the transaction in local currency.
	# nameOrig - customer who started the transaction
	# oldbalanceOrg - initial balance before the transaction
	# newbalanceOrig - new balance after the transaction
	# nameDest - customer who is the recipient of the transaction
	# oldbalanceDest - initial balance recipient before the transaction. Note that there is not information for customers that start with M (Merchants).
	# newbalanceDest - new balance recipient after the transaction. Note that there is not information for customers that start with M (Merchants).
	# isFraud - This is the transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system.
	# isFlaggedFraud - The business model aims to control massive transfers from one account to another and flags illegal attempts. An illegal attempt in this dataset is an attempt to transfer more than 200.000 in a single transaction.


	# start time tracker
	print("\nLoading train dataset from .CSV file")
	start = time.time()

	# read dataset from *.CSV file
	# the dataset file is CSV file.
	dataFrame = pandas.read_csv(file, header=0)

	# remove nameOrig and nameDest column
	# for simplicity, remove these two columns since it is useless
	dataFrame = dataFrame.drop(['nameOrig', 'nameDest'], axis=1)

	# calculate the elapsed time
	end = time.time()
	print("Done. Elapsed time : %f seconds" % (end - start))

	# return training dataset with input X and output Y
	return dataFrame