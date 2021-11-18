# Reinforcement Learning Simulation of Autonomous Braking System

This application is created following the idea proposed by [Chae et al. 2017](https://arxiv.org/abs/1702.02302) with some modification.
The RL simulation enviroment is created using a customed OpenAI gym enviroment. We also add simulations of the federated learning process.

## Installation

First run the following commands to set up enviroment:
```sh
conda env create -n rlsimenv -f ./setup/rlsimenv.yaml

```

Be sure to activate the virtual environment.  
```sh
conda activate rlsimenv
```

Please also install the customed OpenAI gym enviroment, which is created specifically for this simulation:  
```sh
pip install -e ./OpenAI_gym_brake
```

To run the simulation, please use the following command:
```sh
python -m RL_brake_sim.RL_brake_sim [number of agents] [number of training epochs]
```

## Performance Tracking Dashboard

This application, which is writtem with plotly dash, provides performance visualization for the Reinforcement Learning Braking Simulator 

## How to Run

Please making sure the simulator (```RL_brake/RL_brak_sim```) is alredy running or has been run, so that the data is generated (```RL_brake/data```). 
If the simulator is still running, the results will be updated real-time.

Then type the following command to start the application:

```python
python -m sim-dashboard.app
```

Go to the following web page to display the dashboard:
```
http://127.0.0.1:8050/
```

