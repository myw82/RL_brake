# Braking simulation environment 

## Overview
Based on a modified version of the [car_racing environment](https://gym.openai.com/envs/CarRacing-v0/).
This is a framework for setting up envirorment for reinforcement learning (RL) training. It can be adopted to different RL algorithm, especially the [Stable Baseline](https://stable-baselines.readthedocs.io/en/master/)

## Installation
First, please make sure the OpenAI gym is installed. To do so, please type
```python
pip install gym
```
For other installation options, please refer to their [website](https://gym.openai.com/docs/)

Then, to install this enviroment, please download the pagckage the run
```python
pip install -e ./OpenAI_gym_ADAS_AZAPA
```

## State
This environment consists of 6 state items: 
 - vehicle position y-axis (veh_pos_y)
 - vehicle velocity y-axis (veh_vel_y)
 - pedestrian posistion x-axis (ped_pos_x)
 - pedestrian position y-axis (ped_pos_y)
 - pedestrian velocity x-axis (ped_vel_x)
 - pedestrian velocity y-axis (ped_vel_y)

The vehicle is moving along the y direction only, while the pedestrian can moving along both x (crossing the road) and y-axis.
The track is a fixed as a straight track going along the y direction, and the vehical starts from (0, 0) in every episode.
Please note that, in the paper ([Chae et al. 2017](https://arxiv.org/pdf/1702.02302v2)) where the inspiration of the enviroment seeting is from, the car is moving along the x direction. So the definition of x, y axis is flipped in this module comparing to their setting. 

## Action

The action space consist of two actions: 
- Constant deceleration of -1.5 m/s
- Moving in constant speed of 4 km/s (~ 1.11 m/s)

## Reward Function
The reward function is a modified version of the one described in [Chae et al. 2017](https://arxiv.org/pdf/1702.02302v2). It consists of two terms:

- The first term: 
<img src="https://render.githubusercontent.com/render/math?math=r_1 = -(\alpha \times d^2 - g(d)) \times decel">

, where <img src="https://render.githubusercontent.com/render/math?math=d"> is the distance between the vehicle and the pedestrian, and <img src="https://render.githubusercontent.com/render/math?math=decel"> is the deceleration magnitude. This term aims to apply less penalty higher reward when brake is hitted at closer distance.

-  The second term: 
<img src="https://render.githubusercontent.com/render/math?math=r_2 = - (\eta \times v_{veh}^2 %2B \lambda) \times \textbf{1}(S = bump)">

, where <img src="https://render.githubusercontent.com/render/math?math=v_{veh}"> is the velocity of the vehicle, <img src="https://render.githubusercontent.com/render/math?math=\textbf{1}(S = bump)"> is 1 when the collision happend otherwise 0. This term aims to apply a heavy penalty when the vehicle bumps the pedestrian.

The total reward function (r_t) at time step t is the summation of these two terms <img src="https://render.githubusercontent.com/render/math?math=r_t= r_1 %2B r_2">

## Termination condition

Once each round of similations starts, the simulation will be done when the following conditions are reached:

1. The vehicle passes the pedestrian.
2. The vehicle bumps the pedestrian (distance < 2.5 meter)
3. The vehicle is completely stopped. (speed = 0)

## Manually controlling the simulation

To manually control the senior car, please run 
```python
python sc_adas_env.py
```
To apply brakes, press LEFT key. Once the LEFT key is released, the speed will go back to the original speed. 
