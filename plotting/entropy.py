import numpy as np
import pandas as pd
import plotly as py

import plotly.graph_objs as go
import plotly.express as px

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors, path


def add_opacity(color, opacity):
    return f'rgba{color[3:-1]}, {opacity})'

my_colours = []

for colour in reversed(px.colors.qualitative.Bold):
    my_colours.append(colour)

for colour in reversed(px.colors.qualitative.Vivid):
    my_colours.append(colour)

translucent_colours = [add_opacity(c, 0.15) for c in my_colours]



def active_information():

    path = f"../data/simplex/stats/entropy_k20.csv"

    df = pd.read_csv(path)

    dimensionality = df["neurons"]

    sink_active = df["sink_active_info"]
    sink_active_std = df["sink_active_std"]

    fig = go.Figure([
        go.Scatter(
            name = "Transfer entropy",
            x = dimensionality, 
            y=sink_active, 
            line=dict(width=3, color='rgba(0, 131, 143, 1)'),
            showlegend=False),
                
        go.Scatter(
            name = "Upper bound",
            x = dimensionality,
            y = sink_active + sink_active_std,
            mode = 'lines',
            marker = dict(color = 'rgba(0, 131, 143)'),
            line = dict(width = 0),
            showlegend=False
        ),

        go.Scatter(
            name = "Lower bound",
            x = dimensionality,
            y = sink_active - sink_active_std,
            marker = dict(color = 'rgba(0, 131, 143)'),
            line = dict(width = 0),
            mode = 'lines',
            fillcolor = 'rgba(0, 131, 143, 0.3)', 
            fill = 'tonexty',
            showlegend=False
        )
    ])
    
    fig.update_layout(title='Active information of sink neurons in simplices of different sizes', 
                   xaxis_title='Neurons',
                   yaxis_title='Active information',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image("figures/entropy_active_k20.pdf")


#active_information()



def transfer_entropy():

    path = f"../data/simplex/stats/transfer_entropy.csv"

    df = pd.read_csv(path)

    dimensionality = df["neurons"].to_numpy()

    transfer_entropy = df["transfer_entropy"].to_numpy()
    transfer_entropy_std = df["std"].to_numpy()

    std_upper, std_lower = error_bounds(transfer_entropy, transfer_entropy_std)

    fig = go.Figure([
        
        go.Scatter(
            name = "Transfer entropy",
            x = dimensionality, 
            y=transfer_entropy, 
            line=dict(width=3, color='rgba(0, 131, 143, 1)'),
            showlegend=False),
                  
        go.Scatter(
            name = "Upper bound",
            x = dimensionality,
            y = transfer_entropy + transfer_entropy_std,
            mode = 'lines',
            marker = dict(color = 'rgba(0, 131, 143)'),
            line = dict(width = 0),
            showlegend=False
        ),

        go.Scatter(
            name = "Lower bound",
            x = dimensionality,
            y = transfer_entropy - transfer_entropy_std,
            marker = dict(color = 'rgba(0, 131, 143)'),
            line = dict(width = 0),
            mode = 'lines',
            fillcolor = 'rgba(0, 131, 143, 0.3)', 
            fill = 'tonexty',
            showlegend=False
        )
        ])
        
    
    fig.update_layout(title='Transfer entropy between source and sink <br> neurons as function of simplex size', 
                   xaxis_title='Neurons',
                   yaxis_title='Transfer entropy',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image("figures/transfer_entropy.pdf")


#transfer_entropy()






def transfer_entropy_time():

    path = f"../data/simplex/stats/transfer_entropy_10_neurons.csv"

    df = pd.read_csv(path)

    dimensionality = df["time_shift"].to_numpy()

    transfer_entropy = df["transfer_entropy"].to_numpy()
    transfer_entropy_std = df["std"].to_numpy()

    std_upper, std_lower = error_bounds(transfer_entropy, transfer_entropy_std)

    fig = go.Figure([
        
        go.Scatter(
            name = "Transfer entropy",
            x = dimensionality, 
            y=transfer_entropy, 
            line=dict(width=3, color='rgba(0, 131, 143, 1)'),
            showlegend=False),
                  
        go.Scatter(
            name = "Upper bound",
            x = dimensionality,
            y = transfer_entropy + transfer_entropy_std,
            mode = 'lines',
            marker = dict(color = 'rgba(0, 131, 143)'),
            line = dict(width = 0),
            showlegend=False
        ),

        go.Scatter(
            name = "Lower bound",
            x = dimensionality,
            y = transfer_entropy - transfer_entropy_std,
            marker = dict(color = 'rgba(0, 131, 143)'),
            line = dict(width = 0),
            mode = 'lines',
            fillcolor = 'rgba(0, 131, 143, 0.3)', 
            fill = 'tonexty',
            showlegend=False
        )
        ])
        
    
    fig.update_layout(title='Transfer entropy between source and sink neurons<br>in a 9-simplex as function of time-shift', 
                   xaxis_title='Time shift',
                   yaxis_title='Transfer entropy',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image("figures/transfer_entropy_time.pdf")

#transfer_entropy_time()




def conditional_entropy(t_shift, min, max):

    all_conditional_entropy = np.zeros(max-min)
    all_conditional_entropy_std = np.zeros(max-min)
    dimensionality = np.arange(min, max)

    for neurons in range(min, max):
        path = f"../data/simplex/stats/conditional_entropy/conditional_entropy_{neurons}.csv"

        df = pd.read_csv(path)

        conditional_entropy = np.mean(df["conditional_entropy"][:t_shift].to_numpy())
        conditional_entropy_std = np.mean(df["std"][:t_shift].to_numpy())

        all_conditional_entropy[neurons-min] = conditional_entropy
        all_conditional_entropy_std[neurons-min] = conditional_entropy_std


    fig = go.Figure([
        
        go.Scatter(
            name = "Transfer entropy",
            x = dimensionality, 
            y=all_conditional_entropy, 
            line=dict(width=3, color='rgba(0, 131, 143, 1)'),
            showlegend=False),
                  
        go.Scatter(
            name = "Upper bound",
            x = dimensionality,
            y = all_conditional_entropy + all_conditional_entropy_std,
            mode = 'lines',
            marker = dict(color = 'rgba(0, 131, 143)'),
            line = dict(width = 0),
            showlegend=False
        ),

        go.Scatter(
            name = "Lower bound",
            x = dimensionality,
            y = all_conditional_entropy - all_conditional_entropy_std,
            marker = dict(color = 'rgba(0, 131, 143)'),
            line = dict(width = 0),
            mode = 'lines',
            fillcolor = 'rgba(0, 131, 143, 0.3)', 
            fill = 'tonexty',
            showlegend=False
        )
        ])


    fig.update_layout(title='Conditional entropy between source and sink neurons<br> as function of simplex size', 
                xaxis_title='Neurons',
                yaxis_title='Transfer entropy',
                font_family = "Garamond",
                font_size = 15)

    fig.write_image("figures/conditional_entropy_dimension.pdf")


conditional_entropy(15, 3, 15)