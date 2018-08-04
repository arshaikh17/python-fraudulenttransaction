
"""
Created on Tue Jul 16 21:25:24 2018
@author: Ali Rasheed

"""

import os
from util import load_dataset
from weight2model import convert_model
from model import get_model
from preprocess import convert
from sklearn import preprocessing


# main procedure start
os.system("cls")
print("\nTesting ...")

# read test dataset
file = "../data/financial_test.csv"
df = load_dataset(file)
df = convert(df)
Xnew = preprocessing.scale(df.values)

# load model
weight = 'weight_model.hdf5'
model = convert_model(weight)
model.summary()

# make a prediction
ynew = model.predict_classes(Xnew)
prob_new = model.predict_proba(Xnew)

# show the inputs and predicted outputs
for i in range(len(Xnew)):
	print("Predicted=%s, Confidence level=%s" % (ynew[i], prob_new[i]))
