# -*- coding: utf-8 -*-

"""
Created on Sun Jul  8 16:08:54 2018
@author: Paul

"""

import numpy as np
import random


class NeuralNetwork:

	# Artificial neural network initialization
	def __init__(self, numInput, numHidden, numOutput, seed):
		self.ni = numInput		# number of neurons in input layer
		self.nh = numHidden		# number of neurons in hidden layer
		self.no = numOutput		# number of neurons in output layer

		# Initialize the individual neurons in each layer
		self.iNodes = np.zeros(shape=[self.ni], dtype=np.float32)
		self.hNodes = np.zeros(shape=[self.nh], dtype=np.float32)
		self.oNodes = np.zeros(shape=[self.no], dtype=np.float32)

		# We have 3 layers in this simple implementation
		# Initialize the weight vector between input and hidden layers
		self.ihWeights = np.zeros(shape=[self.ni, self.nh], dtype=np.float32)
		# Initialize the weight vector between hidden and output layers
		self.hoWeights = np.zeros(shape=[self.nh, self.no], dtype=np.float32)

		# Initialize the bias values.
		self.hBiases = np.zeros(shape=[self.nh], dtype=np.float32)
		self.oBiases = np.zeros(shape=[self.no], dtype=np.float32)

		# Weight initialization, we initialize weight vector with random values
		self.rnd = random.Random(seed) # allows multiple instances
		self.initializeWeights()


	# Set weight vector with given weights parameters
	def setWeights(self, weights):
		# Check if we can use this input parameter to initialize the entire neural network
		# If error, print the warning message
		if len(weights) != self.totalWeights(self.ni, self.nh, self.no):
			print("Warning: len(weights) error in setWeights()")    

		# Initialize the weight vector
		idx = 0
		for i in range(self.ni):
			for j in range(self.nh):
				self.ihWeights[i,j] = weights[idx]
				idx += 1

		for j in range(self.nh):
			self.hBiases[j] = weights[idx]
			idx += 1

		for j in range(self.nh):
			for k in range(self.no):
				self.hoWeights[j,k] = weights[idx]
				idx += 1

		for k in range(self.no):
			self.oBiases[k] = weights[idx]
			idx += 1


	# Return the weight vector of this neural network
	def getWeights(self):
		# Get the weight vector size for all layers
		# This will be used to generate the return value
		tw = self.totalWeights(self.ni, self.nh, self.no)
		result = np.zeros(shape=[tw], dtype=np.float32)
		
		# Build the return value from NN structure
		# Points into result
		idx = 0
		for i in range(self.ni):
			for j in range(self.nh):
				result[idx] = self.ihWeights[i,j]
				idx += 1

		for j in range(self.nh):
			result[idx] = self.hBiases[j]
			idx += 1

		for j in range(self.nh):
			for k in range(self.no):
				result[idx] = self.hoWeights[j,k]
				idx += 1

		for k in range(self.no):
			result[idx] = self.oBiases[k]
			idx += 1

		return result
     

	# Weight intialization by random
	def initializeWeights(self):
		# Calculate total size of weight vectors
		numWts = self.totalWeights(self.ni, self.nh, self.no)
		wts = np.zeros(shape=[numWts], dtype=np.float32)
		lo = -0.01; hi = 0.01

		# The weights boundary (lo, hi)
		for idx in range(len(wts)):
			wts[idx] = (hi - lo) * self.rnd.random() + lo

		self.setWeights(wts)


    # Calculate the size of total weight vectors
    # This neural network is a fully-connected network
	def totalWeights(nInput, nHidden, nOutput):
		tw = (nInput * nHidden) + (nHidden * nOutput) + nHidden + nOutput
		return tw


