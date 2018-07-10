
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


	def computeOutputs(self, xValues):
		hSums = np.zeros(shape=[self.nh], dtype=np.float32)
		oSums = np.zeros(shape=[self.no], dtype=np.float32)

		for i in range(self.ni):
			self.iNodes[i] = xValues[i]

		for j in range(self.nh):
			for i in range(self.ni):
				hSums[j] += self.iNodes[i] * self.ihWeights[i, j]

		for j in range(self.nh):
			hSums[j] += self.hBiases[j]

		for j in range(self.nh):
			self.hNodes[j] = self.hypertan(hSums[j])

		for k in range(self.no):
			for j in range(self.nh):
				oSums[k] += self.hNodes[j] * self.hoWeights[j,k]

		for k in range(self.no):
			oSums[k] += self.oBiases[k]
 
		softOut = self.softmax(oSums)
		for k in range(self.no):
			self.oNodes[k] = softOut[k]

		result = np.zeros(shape=self.no, dtype=np.float32)
		for k in range(self.no):
			result[k] = self.oNodes[k]
	  
		return result
	

	def train(self, trainData, maxEpochs, learnRate):
		hoGrads = np.zeros(shape=[self.nh, self.no], dtype=np.float32)  # hidden-to-output weights gradients
		obGrads = np.zeros(shape=[self.no], dtype=np.float32)  # output node biases gradients
		ihGrads = np.zeros(shape=[self.ni, self.nh], dtype=np.float32)  # input-to-hidden weights gradients
		hbGrads = np.zeros(shape=[self.nh], dtype=np.float32)  # hidden biases gradients

		oSignals = np.zeros(shape=[self.no], dtype=np.float32)  # output signals: gradients w/o assoc. input terms
		hSignals = np.zeros(shape=[self.nh], dtype=np.float32)  # hidden signals: gradients w/o assoc. input terms

		epoch = 0
		x_values = np.zeros(shape=[self.ni], dtype=np.float32)
		t_values = np.zeros(shape=[self.no], dtype=np.float32)
		numTrainItems = len(trainData)
		indices = np.arange(numTrainItems)  # [0, 1, 2, . . n-1]  # rnd.shuffle(v)

		while epoch < maxEpochs:
			self.rnd.shuffle(indices)  # scramble order of training items
			for ii in range(numTrainItems):
				idx = indices[ii]

				for j in range(self.ni):
					x_values[j] = trainData[idx, j]  # get the input values	

				for j in range(self.no):
					t_values[j] = trainData[idx, j+self.ni]  # get the target values
				
				self.computeOutputs(x_values)  # results stored internally

				# 1. compute output node signals
				for k in range(self.no):
					derivative = (1 - self.oNodes[k]) * self.oNodes[k]  # softmax
					oSignals[k] = derivative * (self.oNodes[k] - t_values[k])  # E=(t-o)^2 do E'=(o-t)

				# 2. compute hidden-to-output weight gradients using output signals
				for j in range(self.nh):
					for k in range(self.no):
						hoGrads[j, k] = oSignals[k] * self.hNodes[j]

				# 3. compute output node bias gradients using output signals
				for k in range(self.no):
					obGrads[k] = oSignals[k] * 1.0  # 1.0 dummy input can be dropped

				# 4. compute hidden node signals
				for j in range(self.nh):
					sum = 0.0
					for k in range(self.no):
						sum += oSignals[k] * self.hoWeights[j,k]
						
					derivative = (1 - self.hNodes[j]) * (1 + self.hNodes[j])  # tanh activation
					hSignals[j] = derivative * sum
		 
				# 5 compute input-to-hidden weight gradients using hidden signals
				for i in range(self.ni):
					for j in range(self.nh):
						ihGrads[i, j] = hSignals[j] * self.iNodes[i]

				# 6. compute hidden node bias gradients using hidden signals
				for j in range(self.nh):
					hbGrads[j] = hSignals[j] * 1.0  # 1.0 dummy input can be dropped

				# update weights and biases using the gradients

				# 1. update input-to-hidden weights
				for i in range(self.ni):
					for j in range(self.nh):
						delta = -1.0 * learnRate * ihGrads[i,j]
						self.ihWeights[i, j] += delta

				# 2. update hidden node biases
				for j in range(self.nh):
					delta = -1.0 * learnRate * hbGrads[j]
					self.hBiases[j] += delta      

				# 3. update hidden-to-output weights
				for j in range(self.nh):
					for k in range(self.no):
						delta = -1.0 * learnRate * hoGrads[j,k]
						self.hoWeights[j, k] += delta

				# 4. update output node biases
				for k in range(self.no):
					delta = -1.0 * learnRate * obGrads[k]
					self.oBiases[k] += delta
 		  
			epoch += 1
			if epoch % 10 == 0:
				mse = self.meanSquaredError(trainData)
				print("epoch = " + str(epoch) + " ms error = %0.4f " % mse)

		# end while

		result = self.getWeights()
		return result


	# train or test data matrix
	def accuracy(self, tdata):
		num_correct = 0; num_wrong = 0
		x_values = np.zeros(shape=[self.ni], dtype=np.float32)
		t_values = np.zeros(shape=[self.no], dtype=np.float32)

		for i in range(len(tdata)):  # walk thru each data item
			for j in range(self.ni):  # peel off input values from curr data row 
				x_values[j] = tdata[i,j]
			
			for j in range(self.no):  # peel off tareget values from curr data row
				t_values[j] = tdata[i, j+self.ni]

			y_values = self.computeOutputs(x_values)  # computed output values)
			max_index = np.argmax(y_values)  # index of largest output value 

			if abs(t_values[max_index] - 1.0) < 1.0e-5:
				num_correct += 1
			else:
				num_wrong += 1

		return (num_correct * 1.0) / (num_correct + num_wrong)


	# on train or test data matrix
	def meanSquaredError(self, tdata):
		sumSquaredError = 0.0
		x_values = np.zeros(shape=[self.ni], dtype=np.float32)
		t_values = np.zeros(shape=[self.no], dtype=np.float32)

		for ii in range(len(tdata)):  # walk thru each data item
			for jj in range(self.ni):  # peel off input values from curr data row 
				x_values[jj] = tdata[ii, jj]

			for jj in range(self.no):  # peel off tareget values from curr data row
				t_values[jj] = tdata[ii, jj+self.ni]

			y_values = self.computeOutputs(x_values)  # computed output values
	  
			for j in range(self.no):
				err = t_values[j] - y_values[j]
				sumSquaredError += err * err  # (t-o)^2
		
		return sumSquaredError / len(tdata)


	@staticmethod
	def hypertan(x):
		if x < -20.0:
			return -1.0
		elif x > 20.0:
			return 1.0
		else:
			return math.tanh(x)


	@staticmethod	  
	def softmax(oSums):
		result = np.zeros(shape=[len(oSums)], dtype=np.float32)
		m = max(oSums)
		divisor = 0.0
		for k in range(len(oSums)):
			divisor += math.exp(oSums[k] - m)
		
		for k in range(len(result)):
			result[k] =  math.exp(oSums[k] - m) / divisor
		
		return result


	@staticmethod
	def totalWeights(nInput, nHidden, nOutput):
		tw = (nInput * nHidden) + (nHidden * nOutput) + nHidden + nOutput
		return tw

# end class NeuralNetwork


