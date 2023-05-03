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



def max_corr_dim(min_size, max_size, error, error_type="std"):

    corr_15 = np.zeros((max_size - min_size + 1))
    corr_15_std = np.zeros((max_size - min_size + 1))

    corr_20 = np.zeros((max_size - min_size + 1))
    corr_20_std = np.zeros((max_size - min_size + 1))

    size_list = np.arange(min_size, max_size + 1)

    for count, size in enumerate(range(min_size, max_size + 1)):
        df_15 = pd.read_csv(f"../data/simplex/stats/autocorrelation/autocorrelation_{size}.csv")
        df_20 = pd.read_csv(f"../data/simplex/stats/autocorrelation/autocorrelation_{size}.csv")

        corr_15[count]=df_15["stim_15"][14]
        corr_15_std[count]=df_15["stim_15_std"][14]

        corr_20[count]=df_20["stim_20"][19]
        corr_20_std[count]=df_20["stim_20_std"][19]

    fig = go.Figure()

    corr = [corr_15, corr_20]
    corr_std = [corr_15_std, corr_20_std]

    if error_type == 'se':
        corr_std = [corr_15_std/np.sqrt(200), corr_20_std/np.sqrt(200)]

    iteration = [15, 20]
    if error:
        iteration = [20, 15]

    for i, (P, c, c_std) in enumerate(zip(iteration, corr, corr_std)):

        fig.add_trace(
            go.Scatter(
                x = size_list,
                y=c, 
                name=P,
                line=dict(width=3, color=my_colours[i+3])))

        if error:
            fig.add_trace(
                go.Scatter(
                name = "Upper bound",
                x = size_list, 
                y = c + c_std,
                mode = 'lines',
                marker = dict(color=my_colours[i+3]),
                line = dict(width = 0),
                showlegend=False
            ))

            fig.add_trace(
                go.Scatter(
                    name = "Lower bound",
                    x = size_list,
                    y = c - c_std,
                    marker = dict(color=my_colours[i+3]),
                    line = dict(width = 0),
                    mode = 'lines',
                    fillcolor = translucent_colours[i+3], 
                    fill = 'tonexty',
                    showlegend=False
                ))
            

    fig.update_layout(title=f'Autocorrelation at <i>P</i> as function of simplex size', 
                   legend_title='<i>P</i> [ms]',
                   xaxis_title='Neurons',
                   yaxis_title='Autocorrelation at <i>P</i>',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image(f"figures/auto_corr_dim_error_{error}_{error_type}.pdf")


max_corr_dim(3, 10, True, 'se')
#max_corr_dim(3, 10, False)



        

def auto_corr_added_removed(max_change, error, error_type="std"):

    corr_15 = np.zeros((2*max_change + 1))
    corr_15_std = np.zeros((2*max_change + 1))
    corr_15_se = np.zeros((2*max_change + 1))

    corr_20 = np.zeros((2*max_change + 1))
    corr_20_std = np.zeros((2*max_change + 1))
    corr_20_se = np.zeros((2*max_change + 1))

    complete_path = f"../data/simplex/stats/autocorrelation/autocorrelation_8.csv"

    df_15_complete = pd.read_csv(complete_path)
    df_20_complete = pd.read_csv(complete_path)

    corr_15[max_change]=df_15_complete["stim_15"][14]
    corr_15_std[max_change]=df_15_complete["stim_15_std"][14]
    corr_15_se[max_change]=df_15_complete["stim_15_se"][14]
    
    corr_20[max_change]=df_20_complete["stim_20"][19]
    corr_20_std[max_change]=df_20_complete["stim_20_std"][19]
    corr_20_se[max_change]=df_20_complete["stim_20_se"][19]

    x = np.arange(-max_change, max_change + 1)

    for change in range(1, max_change + 1):

        removed_path = f"../data/simplex/stats/autocorrelation/autocorrelation_8_removed_{change}.csv"
        added_path = f"../data/simplex/stats/autocorrelation/autocorrelation_8_added_{change}.csv"

        r_df = pd.read_csv(removed_path)
        a_df = pd.read_csv(added_path)

        corr_15[max_change - change]=r_df["stim_15"][14]
        corr_15_std[max_change - change]=r_df["stim_15_std"][14]
        corr_15_se[max_change - change]=r_df["stim_15_se"][14]

        corr_15[max_change + change]=a_df["stim_15"][14]
        corr_15_std[max_change + change]=a_df["stim_15_std"][14]
        corr_15_se[max_change + change]=a_df["stim_15_se"][14]

        corr_20[max_change - change]=r_df["stim_20"][19]
        corr_20_std[max_change - change]=r_df["stim_20_std"][19]
        corr_20_se[max_change - change]=r_df["stim_20_se"][19]

        corr_20[max_change + change]=a_df["stim_20"][19]
        corr_20_std[max_change + change]=a_df["stim_20_std"][19]
        corr_20_se[max_change + change]=a_df["stim_20_se"][19]

    fig = go.Figure()

    corr = [corr_15, corr_20]
    corr_std = [corr_15_std, corr_20_std]

    if error_type == "se":
        corr_std = [corr_15_se, corr_20_se]

    iteration = [15, 20]
    if error:
        iteration = [20, 15]

    for i, (P, c, c_std) in enumerate(zip(iteration, corr, corr_std)):

        fig.add_trace(
            go.Scatter(
                x = x,
                y=c, 
                name=P,
                line=dict(width=3, color=my_colours[i+3])))

        if error:
            fig.add_trace(
                go.Scatter(
                name = "Upper bound",
                x = x, 
                y = c + c_std,
                mode = 'lines',
                marker = dict(color=my_colours[i+3]),
                line = dict(width = 0),
                showlegend=False
            ))

            fig.add_trace(
                go.Scatter(
                    name = "Lower bound",
                    x = x,
                    y = c - c_std,
                    marker = dict(color=my_colours[i+3]),
                    line = dict(width = 0),
                    mode = 'lines',
                    fillcolor = translucent_colours[i+3], 
                    fill = 'tonexty',
                    showlegend=False
                ))

            

    fig.update_layout(title=f'Autocorrelation at <i>P</i> as function of<br>edges added and removed in a 7-simplex',
                   legend_title='<i>P</i> [ms]',
                   xaxis_title='Change in edges',
                   yaxis_title='Autocorrelation at <i>P</i>',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image(f"figures/auto_corr_added_removed_error_{error}_{error_type}.pdf")


#auto_corr_added_removed(5, False)
auto_corr_added_removed(5, True, 'se')




def auto_corr_time_dim(min_time, max_time, min_size, max_size, p, error, error_type = 'std'):
    fig =  go.Figure()

    iteration = range(min_size, max_size + 1)

    if error:
        iteration = reversed(iteration)

    for i, size in enumerate(iteration):
        path = f"../data/simplex/stats/autocorrelation/autocorrelation_{size}.csv"
        df = pd.read_csv(path)

        time = np.arange(min_time, max_time + 1)
        corr = df[f"stim_{p}"][min_time -1:max_time]
        corr_std = df[f"stim_{p}_std"][min_time -1:max_time]
        corr_se = df[f"stim_{p}_se"][min_time -1:max_time]

        if error_type == "se":
            corr_std = corr_se

        fig.add_trace(
            go.Scatter(
                x = time,
                y=corr, 
                name=size,
                line=dict(width=3, color=my_colours[i+3])))

        if error:
            fig.add_trace(
                go.Scatter(
                name = "Upper bound",
                x = time, 
                y = corr + corr_std,
                mode = 'lines',
                marker = dict(color=my_colours[i+3]),
                line = dict(width = 0),
                showlegend=False
            ))

            fig.add_trace(
                go.Scatter(
                    name = "Lower bound",
                    x = time,
                    y = corr - corr_std,
                    marker = dict(color=my_colours[i+3]),
                    line = dict(width = 0),
                    mode = 'lines',
                    fillcolor = translucent_colours[i+3], 
                    fill = 'tonexty',
                    showlegend=False
                ))


    fig.update_layout(
                    title=f'Autocorrelation for <i>P</i> = {p} ms as function<br>of time-shift for different simplex sizes',
                    legend_title='Neurons',
                    xaxis_title='Time-shift [ms]',
                    yaxis_title='Autocorrelation',
                    font_family = "Garamond",
                    font_size = 15)

    fig.write_image(f"figures/auto_corr_time_dim_p_{p}_error_{error}_{error_type}.pdf")



# auto_corr_time_dim(10, 50, 3, 10, 15, False)
auto_corr_time_dim(10, 50, 3, 10, 15, True, 'se')


# auto_corr_time_dim(10, 50, 3, 10, 20, False)
auto_corr_time_dim(10, 50, 3, 10, 20, True, 'se')






def auto_corr_time_change(min_time, max_time, change_type, max_change, p, error, error_type = 'std'):
    fig =  go.Figure()

    iteration = range(0, max_change + 1)

    if error:
        iteration = reversed(iteration)

    for i, change in enumerate(iteration):
        if change == 0:
            path = f"../data/simplex/stats/autocorrelation/autocorrelation_8.csv"

        else: 
            path = f"../data/simplex/stats/autocorrelation/autocorrelation_8_{change_type}_{change}.csv"

        df = pd.read_csv(path)

        time = np.arange(min_time, max_time + 1)
        corr = df[f"stim_{p}"][min_time -1:max_time]
        corr_std = df[f"stim_{p}_std"][min_time -1:max_time]
        corr_se = df[f"stim_{p}_se"][min_time -1:max_time]

        if error_type == "se":
            corr_std = corr_se

        fig.add_trace(
            go.Scatter(
                x = time,
                y=corr, 
                name=change,
                line=dict(width=3, color=my_colours[i+3])))

        if error:
            fig.add_trace(
                go.Scatter(
                name = "Upper bound",
                x = time, 
                y = corr + corr_std,
                mode = 'lines',
                marker = dict(color=my_colours[i+3]),
                line = dict(width = 0),
                showlegend=False
            ))

            fig.add_trace(
                go.Scatter(
                    name = "Lower bound",
                    x = time,
                    y = corr - corr_std,
                    marker = dict(color=my_colours[i+3]),
                    line = dict(width = 0),
                    mode = 'lines',
                    fillcolor = translucent_colours[i+3], 
                    fill = 'tonexty',
                    showlegend=False
                ))


    fig.update_layout(
                    title=f'Autocorrelation for <i>P</i> = {p} ms as function of<br>time-shift for 7-simplex with edges {change_type}',
                    legend_title=f'Edges {change_type}',
                    xaxis_title='Time-shift [ms]',
                    yaxis_title='Autocorrelation',
                    font_family = "Garamond",
                    font_size = 15)

    fig.write_image(f"figures/auto_corr_time_{change_type}_p_{p}_error_{error}_{error_type}.pdf")


#auto_corr_time_change(10, 50, "added", 5, 15, False)
auto_corr_time_change(10, 50, "added", 5, 15, True, 'se')

#auto_corr_time_change(10, 50, "removed", 5, 15, False)
auto_corr_time_change(10, 50, "removed", 5, 15, True, 'se')

#auto_corr_time_change(10, 50, "added", 5, 20, False)
auto_corr_time_change(10, 50, "added", 5, 20, True, 'se')

#auto_corr_time_change(10, 50, "removed", 5, 20, False)
auto_corr_time_change(10, 50, "removed", 5, 20, True, 'se')