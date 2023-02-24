import pickle
import numpy as np
import pandas as pd

import plotly as py
import plotly.graph_objs as go
import plotly.express as px

def simplex_dim(neurons: list):

    sm_simplices = np.zeros((4, 7))
    rm_simplices = np.zeros((4, 7))

    fig = go.Figure()

    for idx, neuron in enumerate(neurons):
        path = f"../data/stats/sm_rm_count_{neuron}_neurons.csv"

        df = pd.read_csv(path)

        sm_avg_global_count = df['small_world_avg_global_count'].to_numpy()
        rm_avg_global_count = df['random_avg_global_count'].to_numpy()

        sm_avg_connectivity = df['small_world_avg_connectivity'].to_numpy()
        rm_avg_connectivity = df['random_avg_connectivity'].to_numpy()

        for i in range(4):
            sm_simplices[i, idx] = sm_avg_global_count[2+i]
            rm_simplices[i, idx] = rm_avg_global_count[2+i]


    for idx, neuron in enumerate(neurons):
        fig.add_trace(go.Scatter(y=sm_simplices[:, idx], x = np.arange(2, 6), name=f"Small world: {neuron} neurons", line=dict(width=3, color=px.colors.qualitative.Bold[-idx])))
        fig.add_trace(go.Scatter(y=rm_simplices[:, idx], x = np.arange(2, 6), name=f"Random: {neuron} neurons", line=dict(width=3, color=px.colors.qualitative.Bold[-idx], dash='dash')))

    fig.update_layout(title='Number of simplices of different dimensions', 
                   xaxis_title='Simplex dimension',
                   yaxis_title='Number of simplices',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image("figures/simplex_count.pdf")



neurons = [10, 15, 20, 25, 30, 40, 50]

simplex_dim(neurons)