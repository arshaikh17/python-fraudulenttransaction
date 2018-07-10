
"""
Created on Tue Jul 9 8:16:06 2018
@author: Paul

"""

import os
from util import load_dataset
from preprocess import preprocess_dataset
from model import get_model


# main procedure start
os.system("cls")
print("\nTraining procedure...")

# read dataset
X, Y = load_dataset()

# # preprocess the dataset
X, Y = preprocess_dataset(X, Y)

# load model
model = get_model()
model.summary()

# # fit and save the model for later usage
# model.fit(X, Y,
# 	validation_split=0.20,
# 	epochs=100,
# 	batch_size=20)
# model.save_weights('weight_model.h5')

# # evaluate the model
# scores = model.evaluate(X, Y)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))