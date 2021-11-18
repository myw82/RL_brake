# Reinforcement Learning Braking Simulator Dashboard
This application provides performance visualization for the ADAS Reinforcement Learning Simulator (```RL_brak/RL_brake_sim```)

## How to Run
To configurate the enviroment, run:
```python
pip install -r requirements.txt
```

Please making sure the simulator (```RL_brake/RL_brake_sim```) is alredy running or has been run, so that the data is generated (```RL_brake/data```). 
If the simulator is still running, the results will be updated real-time.

Then type the following command to start the application:

```python
python -m sim-dashboard.app
```

Go to the following web page to display the dashboard:
```
http://127.0.0.1:8050/
```
