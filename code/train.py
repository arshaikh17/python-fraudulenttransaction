
"""
Created on Tue Jul 9 8:16:06 2018
@author: Ali Rasheed

"""

import os
from util import load_dataset
from preprocess import preprocess_dataset
from model import get_model


# main procedure start
os.system("cls")
print("\nTraining procedure...")

# read dataset
file = "../data/financial_log.csv"
df = load_dataset(file)

# print(df.columns)

# preprocess the dataset
X, Y = preprocess_dataset(df)

# load model
model = get_model()
model.summary()

# fit and save the model for later usage
model.fit(X, Y,
	validation_split=0.20,
	epochs=10,
	batch_size=20)
model.save_weights('weight_model.h5')

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))