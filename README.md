
### Install

```sh
# macOS
conda env create -n rlsimenv -f ./setup/rlsimenv.yaml

```

Be sure to activate the virtual environment.  
```sh
# macOS
conda activate rlsimenv
```

## Usage
# Reinforcement Learning Braking Simulator Dashboard
This application provides performance visualization for the Reinforcement Learning Braking Simulator 

## How to Run
To configurate the enviroment, run:
```python
pip install -r requirements.txt
```

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

Here is an example of how the dashboard should look like:

