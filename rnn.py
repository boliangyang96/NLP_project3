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
from ffnn1fix import make_vocab, make_indices
from gensim.models import Word2Vec
 
unk = '<UNK>'


class RNN(nn.Module):
	def __init__(self, input_dim, h): # Add relevant parameters
		super(RNN, self).__init__()
		# Fill in relevant parameters
		# Ensure parameters are initialized to small values, see PyTorch documentation for guidance
		self.h = h
		self.rnn = nn.RNN(
			input_size = input_dim,
			hidden_size = h,
			num_layers = 2,
			batch_first = True,
		)
		self.out = nn.Linear(h, 5)
		self.softmax = nn.LogSoftmax()
		self.loss = nn.NLLLoss()

	def compute_Loss(self, predicted_vector, gold_label):
		return self.loss(predicted_vector, gold_label)	

	def forward(self, inputs): 
		#begin code
		# inputs : (batch_size, time_step, input_size)
		# h_state : (num_layers, batch_size, hidden_size)
		# output: (batch_size, time_step, output_size)

		# or: initial_h_state = torch.randn(1, inputs.size(0), self.h)
		initial_h_state = torch.zeros(2, inputs.size(0), self.h)
		output, h_state = self.rnn(inputs, initial_h_state)
		output_score = self.out(output[:, -1, :])
		predicted_vector = self.softmax(output_score) # Remember to include the predicted unnormalized scores which should be normalized into a (log) probability distribution
		#end code
		return predicted_vector

# You may find the functions make_vocab() and make_indices from ffnn.py useful; you are free to copy them directly (or call those functions from this file)

# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2vec_model):
	vectorized_data = []
	for temp in data:
		temp_vectorized = []
		for word in temp[0]:
			temp_vectorized.append(word2vec_model.wv[word])
		vectorized_data.append((torch.from_numpy(np.array(temp_vectorized)), temp[1]))
	return vectorized_data

# choose embedding_dim = 128, hidden_dim = 32, number_of_epochs = 10
def main(embedding_dim, hidden_dim, number_of_epochs): # Add relevant parameters
	print("Fetching data")
	train_data, valid_data = fetch_data() # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

	temp_list = []
	for data in train_data:
		temp_list.append(data[0])
	for data in valid_data:
		temp_list.append(data[0])
	word2vec_model = Word2Vec(temp_list, size = embedding_dim, window = 5, min_count = 1)
	vectorized_train = convert_to_vector_representation(train_data, word2vec_model)
	# or change to handle unk in validation data
	vectorized_valid = convert_to_vector_representation(valid_data, word2vec_model)

	print("Fetched and Vectorized data")


	# Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
	# Further, think about where the vectors will come from. There are 3 reasonable choices:
	# 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
	# 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
	# 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further. 
	# Option 3 will be the most time consuming, so we do not recommend starting with this

	# similar to ffnn1fix.py, make some changes in validation part, also check the early stopping condition
	model = RNN(embedding_dim, hidden_dim) # Fill in parameters
	optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)
	#optimizer = optim.Adam(model.parameters(), lr=0.01)

	# early stopping condition
	min_valid_loss = 1e10 # keep track of the minimum validation loss
	number_to_stop = 5 # when reach this number, do the early stopping
	counter = 0 # keep track of the number of epoches that do not decrease from minimum loss
	stop_flag = False # early stopping flag

	print("Training for {} epochs".format(number_of_epochs))
	for epoch in range(number_of_epochs):
		model.train()
		optimizer.zero_grad()
		loss = None
		correct = 0
		total = 0
		start_time = time.time()
		print("Training started for epoch {}".format(epoch + 1))
		random.shuffle(vectorized_train) # Good practice to shuffle order of training data
		minibatch_size = 16 
		N = len(vectorized_train) 
		for minibatch_index in tqdm(range(N // minibatch_size)):
			optimizer.zero_grad()
			loss = None
			for example_index in range(minibatch_size):
				input_vector, gold_label = vectorized_train[minibatch_index * minibatch_size + example_index]
				predicted_vector = model(input_vector.unsqueeze(0))
				predicted_label = torch.argmax(predicted_vector)
				correct += int(predicted_label == gold_label)
				total += 1
				example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
				if loss is None:
					loss = example_loss
				else:
					loss += example_loss
			loss = loss / minibatch_size
			loss.backward()
			optimizer.step()
		print("Training completed for epoch {}".format(epoch + 1))
		print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
		print("Training time for this epoch: {}".format(time.time() - start_time))
		loss = None
		correct = 0
		total = 0
		start_time = time.time()
		print("Validation started for epoch {}".format(epoch + 1))
		random.shuffle(vectorized_valid) # Good practice to shuffle order of validation data
		minibatch_size = len(vectorized_valid) 
		N = len(vectorized_valid) 
		for minibatch_index in tqdm(range(N // minibatch_size)):
			loss = None
			for example_index in range(minibatch_size):
				input_vector, gold_label = vectorized_valid[minibatch_index * minibatch_size + example_index]
				predicted_vector = model(input_vector.unsqueeze(0))
				predicted_label = torch.argmax(predicted_vector)
				correct += int(predicted_label == gold_label)
				total += 1
				example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
				if loss is None:
					loss = example_loss
				else:
					loss += example_loss
			loss = loss / minibatch_size
			
			# check for early stopping condition
			if loss < min_valid_loss:
				min_valid_loss = loss
				counter = 0
			else:
				counter += 1
			print("Counter: {}".format(counter))
			if counter == number_to_stop:
				stop_flag = True
				break

		print("Validation completed for epoch {}".format(epoch + 1))
		print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
		print("Validation time for this epoch: {}".format(time.time() - start_time))

		if stop_flag:
			print("Early stopping, with minimum validation loss {}".format(min_valid_loss))
			break



	#while not stopping_condition: # How will you decide to stop training and why
		#optimizer.zero_grad()
		# You will need further code to operationalize training, ffnn.py may be helpful

		#predicted_vector = model(input_vector)
		#predicted_label = torch.argmax(predicted_vector)
		# You may find it beneficial to keep track of training accuracy or training loss; 

		# Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance

		# You will need to validate your model. All results for Part 3 should be reported on the validation set. 
		# Consider ffnn.py; making changes to validation if you find them necessary

