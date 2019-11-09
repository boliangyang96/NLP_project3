import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os

import time
from tqdm import tqdm
from data_loader import fetch_data

unk = '<UNK>'


class RNN(nn.Module):
	def __init__(self): # Add relevant parameters
		super(RNN, self).__init__()
		# Fill in relevant parameters
		# Ensure parameters are initialized to small values, see PyTorch documentation for guidance
		self.softmax = nn.LogSoftmax()
		self.loss = nn.NLLLoss()

	def compute_Loss(self, predicted_vector, gold_label):
		return self.loss(predicted_vector, gold_label)	

	def forward(self, inputs): 
		#begin code


		predicted_vector = self.softmax() # Remember to include the predicted unnormalized scores which should be normalized into a (log) probability distribution
		#end code
		return predicted_vector

# You may find the functions make_vocab() and make_indices from ffnn.py useful; you are free to copy them directly (or call those functions from this file)



def main(): # Add relevant parameters
	train_data, valid_data = fetch_data() # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

	# Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
	# Further, think about where the vectors will come from. There are 3 reasonable choices:
	# 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
	# 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
	# 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further. 
	# Option 3 will be the most time consuming, so we do not recommend starting with this

	model = RNN() # Fill in parameters
	optimizer = optim.SGD(model.parameters()) 

	while not stopping_condition: # How will you decide to stop training and why
		optimizer.zero_grad()
		# You will need further code to operationalize training, ffnn.py may be helpful

		predicted_vector = model(input_vector)
		predicted_label = torch.argmax(predicted_vector)
		# You may find it beneficial to keep track of training accuracy or training loss; 

		# Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance

		# You will need to validate your model. All results for Part 3 should be reported on the validation set. 
		# Consider ffnn.py; making changes to validation if you find them necessary

