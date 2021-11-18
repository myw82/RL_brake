import os
import pandas as pd
import pathlib
import numpy as np
import datetime as dt
import dash
import dash_daq as daq
import dash_core_components as dcc
import dash_html_components as html
import dash_player as player

from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from scipy.stats import rayleigh, norm


GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 5000)

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server

app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"} 

# Get current directory
curdir = os.getcwd()
file_list = os.listdir(curdir+'/data')

# Determine the number of agents using the agent folder number in the ./data directory
global agent_number
agent_number = 0
for file in file_list:
	if "agent" in file:
		agent_number += 1

# Generate applications (dropdown, agent status) based on number of agents
agent_dropdown_list = []
agent_state_list = []
agent_output_list = []

for i in range(agent_number):
	agent_output_list.append(Output("Agent "+str(i)+" status", "children"))
	agent_dropdown_list.append({"label": "Agent "+str(i),"value": str(i)})
	agent_state_list.append(html.Div([
								html.H5(
                                    id = "Agent "+str(i)+" status",
                                    children = ['Agent '+str(i)+' : training ; training round : 0'],
                            	style={'color': '#c2c0ba', 'fontSize': 16,'marginLeft': 30}
                                ),                                      
                            ]),)


# Main app layout
app.layout = html.Div(
    [
        # header
        html.Div(
            [
                html.Div(
                    [
                        html.H4("RL Braking Model Performace Dashboard", className="app__header__title"),
                    ],
                    className="app__header__desc",
                ),
                html.Div(
                    [
                        #html.Img(
                        #    src=app.get_asset_url("Tieset-logo-dark.png"),
                        #    className="app__menu__img",
                        #)
                    ],
                    className="app__header__logo",
                ),
            ],
            className="app__header",
        ),
        html.Div(
            [
                # Performance graphs 
                html.Div(
                    [
                        html.Div(
                            [html.H6("Reward Function", className="graph__title")]
                        ),
                        dcc.Graph(
                            id="reward",
                            figure=dict(
                                layout=dict(
                                    plot_bgcolor=app_color["graph_bg"],
                                    paper_bgcolor=app_color["graph_bg"],
                                )
                            ),
                        ),
                        html.Div(
                            [html.H6("Loss Function", className="graph__title")],
                        ),                        
                        dcc.Graph(
                            id="loss",
                            figure=dict(
                                layout=dict(
                                    plot_bgcolor=app_color["graph_bg"],
                                    paper_bgcolor=app_color["graph_bg"],
                                )
                            ),
                        ),   
                        html.Div(
                            [html.H6("Ending Condition", className="graph__title")],
                        ),                        
                        dcc.Graph(
                            id="ending_condition",
                            figure=dict(
                                layout=dict(
                                    plot_bgcolor=app_color["graph_bg"],
                                    paper_bgcolor=app_color["graph_bg"],
                                )
                            ),
                        ), 
                        html.Div(
                            [html.H6("Final Stopping Distance", className="graph__title")],
                        ),                        
                        dcc.Graph(
                            id="distance",
                            figure=dict(
                                layout=dict(
                                    plot_bgcolor=app_color["graph_bg"],
                                    paper_bgcolor=app_color["graph_bg"],
                                )
                            ),
                        ),       
                        html.Div(
                            [html.H6("Reward v.s. Final Distance", className="graph__title")],
                        ),                        
                        dcc.Graph(
                            id="reward_distance",
                            figure=dict(
                                layout=dict(
                                    plot_bgcolor=app_color["graph_bg"],
                                    paper_bgcolor=app_color["graph_bg"],
                                )
                            ),
                        ),  
                                                                                                                  
                        dcc.Interval(
                            id="interval-update",
                            interval=int(GRAPH_INTERVAL),
                            n_intervals=0,
                        ),
                    ],
                    className="two-thirds column container",
                ),
                html.Div(
                    [
                        # Agent information
                        html.Div(
                            [
                                html.Div(
                                    [
                                    	html.Div([
                                        	html.H5(
                                            	"Number of Agents",
                                            	className="graph__title",
                                        )]),
                                        daq.LEDDisplay(
                            				id='my-LED-display',
                            				label="",
                            				value=agent_number,
                            				size = 25,
                            				#color = "#2e98d1",#"#d46224",
                            				backgroundColor="#082255"
                            			),
                                    ],style={'textAlign': 'center'}
                                ),

                            ],
                            className="graph__container first",
                        ),
                        # Agent status
                        html.Div(
                            [html.H6("Agent Status", className="graph__title")]
                        ),                        
                        html.Div(
                        	agent_state_list,
                            className="graph__container second",
                            style={"maxHeight": "300px", "overflow": "scroll"},
                        ),
                        html.Div([
                        	html.Div(
                            	[html.H6("Agent Data Selection:", className="graph__title")]
                        	),
                            html.Div(children=["Agent Data Selection"]),
                                dcc.Dropdown(
                                    id="dropdown-agent-selection",
                                    options = agent_dropdown_list,
                                    value="0",
                                    clearable=False,
                            ),
                        ]),
                    ],
                    className="one-third column histogram__direction",
                ),
            ],
            className="app__content",
        ),
    ],
    className="app__container",
)


@app.callback(
    [Output("reward", "figure"),
    Output("loss", "figure"),
    Output("ending_condition", "figure"),
    Output("distance", "figure"),
    Output("reward_distance", "figure")] + 
    [Output("Agent "+str(i)+" status", "children") for i in range(agent_number)],
    [Input("interval-update", "n_intervals"),Input("dropdown-agent-selection", "value"),]
)

def generate_graphs(interval,agent_sel):
    """
    Generate the performance graph. 

    :params interval: update the graph based on an interval
    :params value: agent data selection
    :return reward : figure of reward function
    :return loss : figure of loss function
    :return ending_condition : figure of ending condition percentage
    :return distance : figure of stopping distance
    :return agent status: text of updated agent status
    """
    
    # Reading in performance data
    filesize = os.path.getsize(curdir+"/data/agent"+str(agent_sel)+"/performace.csv")
    if filesize != 0:
    	df_epo = pd.read_csv(curdir+"/data/agent"+str(agent_sel)+"/performace.csv")
    else:
    	df_epo = {}
    	key_list = ['loss_fn', 'reward', 'dist', 'end_condition','decel_reward', 'bump_reward', 'stop_reward', 'passing_reward']
    	for key_s in key_list:
    		df_epo[key_s] = []
    	df_epo = pd.DataFrame(df_epo)
    
    filesize = os.path.getsize(curdir+"/data/agent"+str(agent_sel)+"/performace_ave.csv")
    if filesize != 0:
    	df_epo_ave = pd.read_csv(curdir+"/data/agent"+str(agent_sel)+"/performace_ave.csv")
    else:
    	df_epo_ave = {}
    	key_list = ['mean_reward', 'dist_ave', 'end_condition_1', 'end_condition_2', 'end_condition_3']
    	for key_s in key_list:
    		df_epo_ave[key_s] = [] 
    	df_epo_ave = pd.DataFrame(df_epo_ave)
    
    # Update the agent status
    stat_str_list = []
    for i in range(agent_number):
    	file_status = open(curdir+"/data/agent"+str(i)+"/state",'r')
    	filesize = os.path.getsize(curdir+"/data/agent"+str(i)+"/performace_ave.csv")
    	if filesize != 0:
    		df_t = pd.read_csv(curdir+"/data/agent"+str(i)+"/performace_ave.csv")
    		round = len(df_t)-1
    	else:
    		round = 0
    	status = file_status.read()
    	if status == '0':
    		stat_str = "Agent "+str(i)+" : waiting FL model ; training round : "+str(round)
    	if status == '1':
    		stat_str = "Agent "+str(i)+" : training ; training round : "+str(round)
    	if status == '2':
    		stat_str = "Agent "+str(i)+" : sending ; training round : "+str(round) 
    	if status == '3':
    		stat_str = "Agent "+str(i)+" : sg ready ; training round : "+str(round) 
    	stat_str_list.append(stat_str)  	   	    	
    file_status.close()
	
	# Generate reward function plot
    trace = dict(
        type="scatter",
        y=df_epo["reward"],
        line={"color": "#f5c65f"}, #42C4F7
        hoverinfo="skip",
		#name = 'total reward',
        mode="lines",
    )
                   

    layout = dict(
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
        font={"color": "#fff"},
        height=200,
        margin=dict(l=100, r=50, t=20, b=40),
        xaxis={
            "range": [0, len(df_epo["reward"])],
            "showline": True,
            "zeroline": False,
            "fixedrange": True,
            "title": "Training Episodes",
        },
        yaxis={
            "range": [
                -10000,
                12000,
            ],
            "showgrid": True,
            "showline": True,
            "fixedrange": True,
            "zeroline": False,
            "gridcolor": app_color["graph_line"],
        },
    )
    
    # Generate loss function plot
    trace_2 = dict(
        type="scatter",
        y=df_epo["loss_fn"],
        marker=dict(color = "#429ed4", size =5),
        #line={"color": "#f5c65f"},
        hoverinfo="skip",
        mode="markers",
    )
    if len(df_epo["loss_fn"]) > 0:
    	y_max = max(df_epo["loss_fn"])
    else:
    	y_max = 1000
    layout_2 = dict(
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
        font={"color": "#fff"},
        height=150,
        margin=dict(l=100, r=50, t=20, b=40),
        xaxis={
            "range": [0, len(df_epo["loss_fn"])],
            "showline": True,
            "zeroline": False,
            "fixedrange": True,
            "title": "Training Episodes",
        },
        yaxis={
            "range": [
                0,
                y_max,
            ],
            "showgrid": True,
            "showline": True,
            "fixedrange": True,
            "zeroline": False,
            "gridcolor": app_color["graph_line"],
        },
    )
    
    # Generate end-condition plot
    trace_3 = dict(
        type="scatter",
        y=df_epo_ave["end_condition_1"].values*100,
        line={"color": "#f0bc1f"}, #42C4F7
        hoverinfo="skip",
		name = 'Passing the Pedestrian',
        mode="lines+markers",
    )
    
    trace_4 = dict(
        type="scatter",
        y=df_epo_ave["end_condition_2"].values*100,
        line={"color": "#66dec4"}, #42C4F7
        hoverinfo="skip",
		name = 'Bumping the Pedestrian',
        mode="lines+markers",
    )  
    
    trace_5 = dict(
        type="scatter",
        y=df_epo_ave["end_condition_3"].values*100,
        line={"color": "#23a616"}, #42C4F7
        hoverinfo="skip",
		name = 'Vehicle stopped',
        mode="lines+markers",
    )     

    layout_3 = dict(
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
        font={"color": "#fff"},
        height=150,
        margin=dict(l=100, r=50, t=20, b=40),
        xaxis={
            "range": [0, len(df_epo_ave["end_condition_2"])],
            "showline": True,
            "zeroline": False,
            "fixedrange": True,
            "title": "Training Rounds",
        },
        yaxis={
            "range": [
                0,
                100,
            ],
            "showgrid": True,
            "showline": True,
            "fixedrange": True,
            "zeroline": False,
            "gridcolor": app_color["graph_line"],
            "title" : "Percentage [%]"
        },
    )    
	
	# Generate distance plot
    dist_x = np.arange(len(df_epo["dist"]))
        
    n_dist = len(df_epo["dist"])//100 + int(len(df_epo["dist"]) % 100 != 0)
    dist_median_x = np.zeros(n_dist)
    dist_median_y = np.zeros(n_dist)
    for i in range(n_dist-1):
    	dist_median_x[i] = 50 + 100* (i)
    	ind_t = [i == 2 for i in np.array(df_epo["end_condition"])[i*100:(i+1)*100]]
    	dist_median_y[i] = np.median(np.array(df_epo["dist"].values)[i*100:(i+1)*100][ind_t])

    if n_dist > 1:
    	dist_median_x[-1] = (100* (n_dist-1) + len(df_epo["dist"]))/2.0
    	ind_t = [i == 2 for i in np.array(df_epo["end_condition"])[(n_dist-1)*100:]]
    	dist_median_y[-1] = np.median(np.array(df_epo["dist"].values)[(n_dist-1)*100:][ind_t])
    	
    ind = [i == 2 for i in df_epo["end_condition"]] 	    	
    trace_6 = dict(
        type="scatter",
        y=np.array(df_epo["dist"].values)[ind],
        x = dist_x[ind],
        opacity=0.7,
        marker=dict(color = "#9dbd4f", size =5),
        hoverinfo="skip",
        name = 'Distance when stopped',
        mode="markers",
    )
    
    trace_7 = dict(
        type="scatter",
        y=dist_median_y,
        x=dist_median_x,
        line={"color": "#66dec4"}, #42C4F7
        hoverinfo="skip",
		name = 'media distance',
        mode="lines+markers",
    )
    if len(df_epo["dist"]) > 0:
    	max_dist_y = max(df_epo["dist"])
    else:
    	max_dist_y = 30
    layout_6 = dict(
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
        font={"color": "#fff"},
        height=150,
        margin=dict(l=100, r=50, t=20, b=40),
        xaxis={
            "range": [0, len(df_epo["dist"])],
            "showline": True,
            "zeroline": False,
            "fixedrange": True,
            "title": "Training Episodes",
        },
        yaxis={
            "range": [
                0,
                40,
            ],
            "showgrid": True,
            "showline": True,
            "fixedrange": True,
            "zeroline": False,
            "gridcolor": app_color["graph_line"],
        },
    )
    
    # Generate reward v.s. distance plot
    if len(df_epo["reward"]) > 0:
    	max_reward_y = max(df_epo["reward"])
    else:
    	max_reward_y = 500    
    
    ind_1 = [i == 1 for i in df_epo["end_condition"]] 
    ind_0 = [i == 0 for i in df_epo["end_condition"]] 
    trace_8 = dict(
        type="scatter",
        y=np.array(df_epo["reward"].values),
        x=np.array(df_epo["dist"].values),
        marker=dict(color = "#9dbd4f", size =3), #42C4F7
        hoverinfo="skip",
        mode="markers",
    )
        

    layout_8 = dict(
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
        font={"color": "#fff"},
        height=250,
        width=650,
        margin=dict(l=100, r=5, t=20, b=40),
        xaxis={
            "range": [-5, 55.0],
            "showline": True,
            "zeroline": False,
            "fixedrange": True,
            "title": "Distance [m]",
        },
        yaxis={
            "range": [
                -5000,
                max(700, max_reward_y),
            ],
            "showgrid": True,
            "showline": True,
            "fixedrange": True,
            "zeroline": False,
            "gridcolor": app_color["graph_line"],
            "title":"Reward"
        },
    )    
    
          
    

    return [dict(data=[trace], layout=layout), dict(data=[trace_2], layout=layout_2), dict(data=[trace_3, trace_4, trace_5], layout=layout_3), dict(data=[trace_6, trace_7], layout=layout_6), dict(data=[trace_8], layout=layout_8)]+ stat_str_list


if __name__ == "__main__":
    app.run_server(debug=True)
