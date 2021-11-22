import gym
import gym_rl_brake

import math
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from .DQN import DQN, ReplayMemory, TraumaMemory

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from .conversion import Converter

	
Transition = namedtuple('Transition',
				('state', 'action', 'next_state', 'reward')) 

class DQN_training:
	"""
	Process for performing the DQN training, including model parameter and optimizer settings
	"""
	def __init__(self, sg_models, env, BATCH_SIZE_REPLAY = 80, GAMMA = 0.999, EPS_START = 0.8, EPS_END = 0.2, EPS_DECAY = 50, TARGET_UPDATE = 20,
		ReplayMemory_Size = 10000, TraumaMemory_Size = 1000, lr = 0.00005): #lr = 0.00002
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
		self.BATCH_SIZE_REPLAY = BATCH_SIZE_REPLAY
		self.GAMMA = GAMMA
		self.EPS_START = EPS_START
		self.EPS_END = EPS_END
		self.EPS_DECAY = EPS_DECAY
		self.TARGET_UPDATE = TARGET_UPDATE
		self.env = env
		self.lr = lr
		self.steps_done = 0	
		self.policy_net = DQN().to(self.device)
		self.target_net = DQN().to(self.device)
		self.policy_net.load_state_dict(sg_models.state_dict())
		self.target_net.load_state_dict(self.policy_net.state_dict())
		self.target_net.eval()
		self.policy_net.train()
		self.ReplayMemory_Size = ReplayMemory_Size
		self.TraumaMemory_Size = TraumaMemory_Size
		self.memory = ReplayMemory(self.ReplayMemory_Size)
		self.memory_trauma = TraumaMemory(self.TraumaMemory_Size)
		self.optimizer = optim.SGD(self.policy_net.parameters(), lr = lr, momentum=0.9)

		
		# storing training results
		self.episode_durations = []
		self.episode_reward = []
		self.decel_reward = []
		self.bump_reward = []
		self.stop_reward = []
		self.passing_reward = []
		self.dist = []
		self.n_steps = []
		self.veh_vel_ini = []
		self.end_condition = []
		self.ttc = []
		self.trigger_dist = []
		self.ped_vel = []
		self.episode_durations = []
		self.episode_reward = []
		self.loss_fn = []
		self.end_condition_perc_1 = []
		self.end_condition_perc_2 = []
		self.end_condition_perc_3 = []
		self.final_dist_ave = []
		self.best_action_buffer = []
		self.best_state_buffer = []
		self.result = []

	def select_action(self):
		"""
		Selecting an action based on Epsilon-Greedy algorithm

		:return: tensor - selected action
		"""
		#global steps_done
		n_actions = self.env.action_space.n
		sample = random.random()
		eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
			math.exp(-1. * self.steps_done / self.EPS_DECAY)
		self.steps_done += 1
		if sample > eps_threshold:
			with torch.no_grad():
				return self.policy_net(state).max(1)[1].view(1, 1)
		else:
			return torch.tensor([[random.randrange(n_actions)]], device=self.device, dtype=torch.long)
 
	
	def optimize_model(self):
		"""
		Compute the loss for the batched state-action values

		:return: tensor - loss
		"""	
		if len(self.memory) < self.BATCH_SIZE_REPLAY:
			return
		transitions = self.memory.sample(self.BATCH_SIZE_REPLAY)
		batch = Transition(*zip(*transitions))      

		# Compute a mask of non-final states and concatenate the batch elements
		# (a final state would've been the one after which simulation ended)
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
										  batch.next_state)), device=self.device, dtype=torch.bool)
		non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
	
		state_batch = torch.cat(batch.state)
		action_batch = torch.cat(batch.action)
		reward_batch = torch.cat(batch.reward)

		# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
		# columns of actions taken. 
		state_action_values = self.policy_net(state_batch).gather(1, action_batch).float()

		# Compute V(s_{t+1}) for all next states.
		# Expected values of actions for non_final_next_states are computed based
		# on the "older" target_net; selecting their best reward with max(1)[0].
		next_state_values = torch.zeros(self.BATCH_SIZE_REPLAY, device=self.device).float()
		next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
		# Compute the expected Q values
		expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
		expected_state_action_values = expected_state_action_values.type(torch.float32)

		# Compute Huber loss
		loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

		# Optimize the model
		self.optimizer.zero_grad()
		loss.backward()
		for param in self.policy_net.parameters():
			param.grad.data.clamp_(-1, 1)
		self.optimizer.step()
		return loss  

	def execute_training(self, n_iter):
		"""
		Training routine
		:param n_iter: int, number of training iteration
		:return: tensor -> trained policy net
				 Dict[str,np.array] -> storing performance information
		"""
		highest_reward = -50000.0 # some random low number

		best_final_dist = 0.0    

		for i_episode in range(n_iter):
			# Initialize the environment and state
			state = torch.from_numpy(self.env.reset()).type(torch.float32)
			action_buffer = []
			state_buffer = []
			for t in count():
				# Select and perform an action
				action_ten = self.select_action(state)
				action = action_ten.item()
				action_buffer.append(action)
				state_buffer.append(state)
				next_state, reward, done, info = self.env.step(action)
					
				action = torch.from_numpy(np.array([action])).type(torch.long)
				reward = torch.tensor([reward], device=self.device)
				next_state = torch.from_numpy(next_state).type(torch.float32)
					
				# Store the transition in memory
				self.memory.push(state, action_ten, next_state, reward)
					
				# Move to the next state
				state = next_state
					
				# Perform one step of the optimization (on the target network)
				loss = self.optimize_model()
				if done:
					self.episode_reward.append(reward.numpy())
					self.decel_reward.append(info['decel_reward'])
					self.bump_reward.append(info['bump_reward'])
					self.stop_reward.append(info['stop_reward'])
					self.passing_reward.append(info['passing_reward'])
					self.episode_durations.append(t + 1)
					self.dist.append(info['dist'])
					self.end_condition.append(info['end_condition'])
					self.ttc.append(info['ttc'])
					self.trigger_dist.append(info['trigger_dist'])
					self.ped_vel.append(info['ped_vel'])
					if(loss == None):
						loss = torch.tensor(500.0)
					self.loss_fn.append(loss.detach().numpy())
						
					break
			# Update the target network, copying all weights and biases in DQN
			if i_episode % self.TARGET_UPDATE == 0:
				self.target_net.load_state_dict(self.policy_net.state_dict())
					
			self.result.append(reward.numpy())

			if (reward.numpy() >= highest_reward):
				highest_reward = reward.numpy()
				self.best_action_buffer = action_buffer
				best_state_buffer = state_buffer
				best_final_dist = info['dist']
		
		# Storing and compute training results		
		ec_100 = self.end_condition
		print('Average ending condition :',np.sum([ec == 0 for ec in ec_100])/len(self.end_condition),'%, ',np.sum([ec == 1 for ec in ec_100])/len(self.end_condition),'%, ',np.sum([ec == 2 for ec in ec_100])/len(self.end_condition),'%')
		print('Average final dist: {:0.2f}'.format(np.mean([abs(cd - 1)*d for d, cd in zip(self.dist,self.end_condition)])))
		print(' ')
		print('Best final dist:', best_final_dist)
		if i_episode > 1:
			self.end_condition_perc_1.append(np.sum([ec == 0 for ec in ec_100])/len(self.end_condition))
			self.end_condition_perc_2.append(np.sum([ec == 1 for ec in ec_100])/len(self.end_condition))
			self.end_condition_perc_3.append(np.sum([ec == 2 for ec in ec_100])/len(self.end_condition))
			self.final_dist_ave.append(np.mean([abs(cd - 1)*d for d, cd in zip(self.dist,self.end_condition)]))
			
		data_epo = {}
		data_epo['loss_fn'] = self.loss_fn
		data_epo['reward'] = self.episode_reward
		data_epo['decel_reward'] = self.decel_reward
		data_epo['bump_reward'] = self.bump_reward
		data_epo['stop_reward'] = self.stop_reward
		data_epo['passing_reward'] = self.passing_reward
		data_epo['dist'] = self.dist
		data_epo['end_condition'] = self.end_condition
		
		data_epo_ave = {}
		data_epo_ave['mean_reward'] = np.mean(np.array(self.episode_reward))
		data_epo_ave['dist_ave'] = self.final_dist_ave
		data_epo_ave['end_condition_perc_1'] = self.end_condition_perc_1	
		data_epo_ave['end_condition_perc_2'] = self.end_condition_perc_2	
		data_epo_ave['end_condition_perc_3'] = self.end_condition_perc_3	

		return self.policy_net, data_epo, data_epo_ave
