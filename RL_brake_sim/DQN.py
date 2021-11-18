import math
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

Transition = namedtuple('Transition',
				('state', 'action', 'next_state', 'reward')) 

class ReplayMemory(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)
	
class TraumaMemory(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)      
		
# define the NN architecture
class DQN(nn.Module):
	def __init__(self):
		super(DQN, self).__init__()
		input_size = 5
		hidden_1 = 200
		hidden_2 = 140
		hidden_3 = 80
		hidden_4 = 140
		hidden_5 = 200
		output_size = 4

		self.fc1 = nn.Linear(input_size, hidden_1)
		self.fc2 = nn.Linear(hidden_1, hidden_2)
		self.fc3 = nn.Linear(hidden_2, hidden_3)
		self.fc4 = nn.Linear(hidden_3, hidden_4)
		self.fc5 = nn.Linear(hidden_4, hidden_5)
		self.fc6 = nn.Linear(hidden_5, output_size)

	def forward(self, x):
		x = x.view(-1, 5)
		# add hidden layer, with relu activation function
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))        
		x = F.relu(self.fc5(x))        
		x = self.fc6(x.view(x.size(0), -1))
		return x
