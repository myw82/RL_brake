import pathlib
import numpy as np
import time
import logging
from typing import Dict
import sys
import os
import random
from collections import namedtuple

from .DQN import DQN, ReplayMemory, TraumaMemory
from .conversion import Converter
from .DQN_training import DQN_training
import gym
import gym_rl_brake
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocess as mp
from multiprocess import Pool

    
def init_models():
    """
    Return the templates of models (in a dict) to tell the structure
    The models need not to be trained
    :return: Dict[str,np.array]
    """
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = DQN()#.to(device)
    #model = Converter.cvtr().convert_nn_to_dict_nparray(net)
    #model = {}
    #model['models'] = dict
    return model


def training(sg_models: Dict[str,np.array], n_iter: int = 50, init_flag: bool = False):
    """
    A place holder function for each ML application
    Return the trained models
    Note that each models should be decomposed into numpy arrays
    Logic should be in the form: sg_models -- training --> new local models
    :param sg_models: Dict[str,np.array]
    :param init_flag: bool - True if it's at the init step.
    False if it's an actual training step
    :return: Dict[str,np.array] - trained models
    """
    # return templates of models to tell the structure
    # This model is not necessarily actually trained
    if not sg_models :
        sg_models = init_models()

    # Do ML Training
    logging.info(f'--- Training ---')
    

    # Create a DQN based on SG models
    #print('sg_models.keys()=',sg_models.keys())
    #fl_model = Converter.cvtr().convert_dict_nparray_to_nn(sg_models)  

    env = gym.make('rl_brake-v0')

    # sg_models -- training --> new local models
    training = DQN_training(sg_models, env)
    trained_net, data_epo, data_epo_ave = training.execute_training(n_iter)
    return trained_net, data_epo, data_epo_ave

    
    
def output_performace(data_epo, data_epo_ave, model_path,OUTPUT_HEADER = True):
	"""
	Output the performance matrix values to local csv files.
	:param data_epo: dict - a dictionary that store RL training results
	:param data_epo_ave: dict - a dictionary that store averaged RL training results
	:param model_path: str - output file directory 
	:param OUTPUT_HEADER: bool - output just file header or content   
	"""
	
	if OUTPUT_HEADER:
		file = open(model_path+'/performace.csv',"a")
		file.write('loss_fn,reward,dist,end_condition,decel_reward,bump_reward,stop_reward,passing_reward\n')
		file.close
		
		file = open(model_path+'/performace_ave.csv',"a")
		file.write('mean_reward,dist_ave,end_condition_1,end_condition_2,end_condition_3\n')
		file.close
	else:
		file = open(model_path+'/performace.csv',"a")		
		for i in range(len(data_epo['loss_fn'])):
			file.write('{:.2f}, {:.2f}, {:.2f}, {:.0f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(data_epo['loss_fn'][i],
				data_epo['reward'][i][0],data_epo['dist'][i],data_epo['end_condition'][i],
				data_epo['decel_reward'][i],data_epo['bump_reward'][i],data_epo['stop_reward'][i],
				data_epo['passing_reward'][i]))
			file.write('\n')
		file.close
		
		file_ave = open(model_path+'/performace_ave.csv',"a")
		file_ave.write('{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(data_epo_ave['mean_reward'],
			data_epo_ave['dist_ave'][0],data_epo_ave['end_condition_perc_1'][0],data_epo_ave['end_condition_perc_2'][0],
			data_epo_ave['end_condition_perc_3'][0]))
		file_ave.write('\n')
		file_ave.close 	
		 		


def run_agent_training_loop(epoch, model_path, agg_save_dir):

    if epoch != 0:
    	sg_models = torch.load(agg_save_dir+'/agg_model_'+str(10000+epoch-1)[1:]+'.pt')
    else:
    	sg_models = init_models()
    	
    file = open(model_path+'/state',"w")
    file.write('1')
    file.close()
    #logging.info(f'--- Training Starts ---')
    models, data_epo, data_epo_ave = training(sg_models)
    torch.save(models, model_path+'/latest.pt')

    logging.info(f'--- Training is Done ---')
    
    file = open(model_path+'/state',"w")
    file.write('0')
    file.close()    
        
    output_performace(data_epo, data_epo_ave, model_path, bool(epoch == 0))
    #cl.save_model(models)
    #cl.send_model()
    logging.info(f'--- Normal transition: The trained local models saved ---')
    #return models

    #time.sleep(1)


def aggregation(buffer):
	cluster_model = {}
	for key in buffer[0].keys():
		cluster_model[key] = buffer[0][key]/len(buffer)
		for i in range(1,len(buffer)):
			cluster_model[key] += buffer[i][key]/len(buffer)
	
	return cluster_model
	     	

if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO)
	logging.info('--- This is a FL simulation of RL brake model ---')
	
	if len(sys.argv) > 1:
		num_of_agents = int(sys.argv[1])
		print(f'--- num_of_agents is {num_of_agents} ---')
		n_epochs = int(sys.argv[2])
		print(f'--- num of epochs is {n_epochs} ---')
	else:
		logging.error("--- input parameters needed ---")
		sys.exit()
	
	if not os.path.exists(f'./data/'):
		os.mkdir(f'./data/')
	agg_save_dir = f'./data/agg_files/'
	if not os.path.exists(agg_save_dir):
		os.mkdir(agg_save_dir)
	
	model_buffer = []
	args_list = []
	for i in range(n_epochs):
		for agent_id in range(num_of_agents):
			model_path = f'./data/agent{agent_id}'
			if not os.path.exists(model_path):
				os.mkdir(model_path)
			args_list.append((i, model_path, agg_save_dir))
		
		with Pool(processes=num_of_agents) as p:
			p.starmap(run_agent_training_loop, args_list)
			p.close()
			p.join()
		
		for arg in args_list:
			agent_id, model_path, agg_dir = arg
			model = Converter.cvtr().convert_nn_to_dict_nparray(torch.load(model_path+'/latest.pt'))
			model_buffer.append(model)
		
		cluster_model = aggregation(model_buffer)
		torch.save(Converter.cvtr().convert_dict_nparray_to_nn(cluster_model), agg_save_dir+'/agg_model_'+str(10000+i)[1:]+'.pt')
		print('Aggregated model is formed at ',i,' epoch.')
		model_buffer.clear()
		args_list.clear()
    		
    	    