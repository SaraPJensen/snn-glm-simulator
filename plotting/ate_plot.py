import pickle 

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

#my_colours = px.colors.qualitative.Bold + px.colors.qualitative.Vivid
translucent_colours = [add_opacity(c, 0.15) for c in my_colours]






def ate_dimension(time, max_dim, error, error_type = 'std'):
    fig = go.Figure()

    iteration = range(3, max_dim)

    if error:
        iteration = reversed(iteration)

    for dim in iteration:
        path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}].csv"

        df = pd.read_csv(path)

        rel_ate = df["relative_ATE"]
        rel_ate_std = df["relative_ATE_std"]

        if error_type == 'se':
            rel_ate_std = rel_ate_std/np.sqrt(200)

        fig.add_trace(go.Scatter(y=rel_ate[0:time], name=dim, line=dict(width=3, color=my_colours[dim])))

        if error: 
            fig.add_trace(
                go.Scatter(
                name = "Upper bound",
                y = rel_ate[0:time] + rel_ate_std[0:time],
                mode = 'lines',
                marker = dict(color=my_colours[dim]),
                line = dict(width = 0),
                showlegend=False
            ))

            fig.add_trace(
                go.Scatter(
                    name = "Lower bound",
                    y = rel_ate[0:time] - rel_ate_std[0:time],
                    marker = dict(color=my_colours[dim]),
                    line = dict(width = 0),
                    mode = 'lines',
                    fillcolor = translucent_colours[dim], 
                    fill = 'tonexty',
                    showlegend=False
                )
            )

    fig.update_layout(
                title='Relative ATE for different simplex dimensions over time', 
                legend_title='Neurons',
                xaxis_title='Time-shift [ms]',
                yaxis_title='Relative ATE',
                font_family = "Garamond",
                font_size = 15)

    fig.write_image(f"figures/ate_dimension_error_{error}_{error_type}.pdf")

# ate_dimension(15, 10, True)
# ate_dimension(15, 10, False)
#ate_dimension(15, 10, True, 'se')



def ate_dimension_filter(time, max_dim, filter: str, error, error_type = "std"):
    fig = go.Figure()

    iteration = range(3, max_dim)
    if error:
        iteration = reversed(iteration)

    for dim in iteration:
        if filter == 'inhib':
            path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}]_inhib.csv"
            title = 'Relative ATE for different simplex dimensions<br>over time for inhibitory source neurons'
        
        else: 
            path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}]_excit.csv"
            title = 'Relative ATE for different simplex dimensions<br>over time for excitatory source neurons'

        df = pd.read_csv(path)

        rel_ate = df["relative_ATE"]
        rel_ate_std = df["relative_ATE_std"]

        if error_type == 'se':
            rel_ate_std = df["relative_ATE_se"]

        fig.add_trace(go.Scatter(y=rel_ate[0:time], name=dim, line=dict(width=3, color=my_colours[dim])))

        if error: 
            fig.add_trace(
                go.Scatter(
                name = "Upper bound",
                y = rel_ate[0:time] + rel_ate_std[0:time],
                mode = 'lines',
                marker = dict(color=my_colours[dim]),
                line = dict(width = 0),
                showlegend=False
            ))

            fig.add_trace(
                go.Scatter(
                    name = "Lower bound",
                    y = rel_ate[0:time] - rel_ate_std[0:time],
                    marker = dict(color=my_colours[dim]),
                    line = dict(width = 0),
                    mode = 'lines',
                    fillcolor = translucent_colours[dim], 
                    fill = 'tonexty',
                    showlegend=False
                )
            )

    
    fig.update_layout(
                title=title, 
                legend_title='Neurons',
                xaxis_title='Time-shift [ms]',
                yaxis_title='Relative ATE',
                font_family = "Garamond",
                font_size = 15)

    fig.write_image(f"figures/ate_dimension_{filter}_error_{error}_{error_type}.pdf")


# ate_dimension_filter(15, 10, 'inhib', True, 'se')
# ate_dimension_filter(15, 10, 'excit', True, 'se')

# ate_dimension_filter(15, 10, 'inhib', False)
# ate_dimension_filter(15, 10, 'excit', False)






def ate_change(time, min_dim, max_dim, p_change, change, error):
    fig = go.Figure()

    iteration = range(min_dim, max_dim)
    if error:
        iteration = reversed(iteration)

    for dim in iteration:
        complete_path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}].csv"
        removed_path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}]_removed_{p_change}.csv"
        added_path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}]_added_{p_change}.csv"

        c_df = pd.read_csv(complete_path)
        r_df = pd.read_csv(removed_path)
        a_df = pd.read_csv(added_path)

        c_rel_ate = c_df["relative_ATE"]
        r_rel_ate = r_df["relative_ATE"]
        a_rel_ate = a_df["relative_ATE"]

        fig.add_trace(go.Scatter(y=c_rel_ate[0:time], name=f"{dim}, complete", line=dict(width=3, color=my_colours[dim])))

        if change == "removed":
            fig.add_trace(go.Scatter(y=r_rel_ate[0:time], name=f"{dim}, {p_change} % removed", line=dict(width=3, dash='dash', color=my_colours[dim])))
        
        elif change == "added":
            fig.add_trace(go.Scatter(y=a_rel_ate[0:time], name=f"{dim}, {p_change} % added", line=dict(width=3, dash='dot', color=my_colours[dim])))

    
    fig.update_layout(title=f'Relative ATE for different simplex dimensions, both complete<br> and with {p_change} % of the connections {change}', 
                   legend_title='Neurons',
                   xaxis_title='Time-shift [ms]',
                   yaxis_title='Relative ATE',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image(f"figures/ate_dimension_{change}_{p_change}_error_{error}.pdf")


# ate_change(11, 4, 10, 10, "removed", False)
# ate_change(11, 4, 10, 15, "removed", False)

# ate_change(11, 4, 10, 10, "added", False)
# ate_change(11, 4, 10, 15, "added", False)




def ate_diff(time, min_dim, max_dim, p_change, change, error, error_type):
    fig = go.Figure()

    iteration = range(min_dim, max_dim)
    if error:
        iteration = reversed(iteration)

    for dim in iteration:
        complete_path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}].csv"
        change_path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}]_{change}_{p_change}.csv"

        c_df = pd.read_csv(complete_path)
        r_df = pd.read_csv(change_path)

        c_rel_ate = c_df["relative_ATE"]
        c_rel_ate_std = c_df["relative_ATE_std"]

        r_rel_ate = r_df["relative_ATE"]
        r_rel_ate_std = r_df["relative_ATE_std"]

        if error_type == "se":
            c_rel_ate_std = c_rel_ate_std / np.sqrt(200)
            r_rel_ate_std = r_rel_ate_std / np.sqrt(200)


        diff = r_rel_ate - c_rel_ate
        diff_std = np.sqrt(c_rel_ate_std**2 + r_rel_ate_std**2)

        fig.add_trace(go.Scatter(y=diff[0:time], name=dim, line=dict(width=3, color=my_colours[dim])))
        
        if error: 
            fig.add_trace(
                go.Scatter(
                name = "Upper bound",
                y = diff[0:time] + diff_std[0:time],
                mode = 'lines',
                marker = dict(color=my_colours[dim]),
                line = dict(width = 0),
                showlegend=False
            ))

            fig.add_trace(
                go.Scatter(
                    name = "Lower bound",
                    y = diff[0:time] - diff_std[0:time],
                    marker = dict(color=my_colours[dim]),
                    line = dict(width = 0),
                    mode = 'lines',
                    fillcolor = translucent_colours[dim], 
                    fill = 'tonexty',
                    showlegend=False
                )
            )
    
    fig.update_layout(title=f'Difference in relative ATE for different simplex dimensions,<br>with {p_change} % of the connections {change}', 
                   legend_title='Neurons',
                   xaxis_title='Time-shift [ms]',
                   yaxis_title='Difference in relative ATE',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image(f"figures/ate_dimension_{change}_{p_change}_diff_error_{error}_{error_type}.pdf")



# ate_diff(11, 4, 11, 10, "added", True, 'se')
# ate_diff(11, 4, 11, 10, "removed", True, 'se')

# ate_diff(11, 4, 11, 15, "added", True, 'se')
# ate_diff(11, 4, 11, 15, "removed", True, 'se')

# ate_diff(11, 4, 11, 10, "added", True, 'std')
# ate_diff(11, 4, 11, 10, "removed", True, 'std')

# ate_diff(11, 4, 11, 15, "added", True, 'std')
# ate_diff(11, 4, 11, 15, "removed", True, 'std')



def ate_diff_added_removed(time, neurons, removed, error, error_type):
    fig = go.Figure()

    complete_path = f"../data/simplex/stats/ATE/cluster_sizes_[{neurons}].csv"
    removed_path = f"../data/simplex/stats/ATE/cluster_sizes_[{neurons}]_removed_{removed}.csv"
    added_path = f"../data/simplex/stats/ATE/cluster_sizes_[{neurons}]_added_{removed}.csv"

    c_df = pd.read_csv(complete_path)
    r_df = pd.read_csv(removed_path)
    a_df = pd.read_csv(added_path)

    c_rel_ate = c_df["relative_ATE"]
    c_rel_ate_std = c_df["relative_ATE_std"]

    r_rel_ate = r_df["relative_ATE"]
    r_rel_ate_std = r_df["relative_ATE_std"]

    a_rel_ate = a_df["relative_ATE"]
    a_rel_ate_std = a_df["relative_ATE_std"]

    if error_type == "se":
        c_rel_ate_std = c_rel_ate_std / np.sqrt(200)
        r_rel_ate_std = r_rel_ate_std / np.sqrt(200)
        a_rel_ate_std = a_rel_ate_std / np.sqrt(200)

    diff_added = a_rel_ate - c_rel_ate
    diff_added_std = np.sqrt(c_rel_ate_std**2 + a_rel_ate_std**2)

    diff_removed = r_rel_ate - c_rel_ate
    diff_removed_std = np.sqrt(c_rel_ate_std**2 + r_rel_ate_std**2)

    fig.add_trace(go.Scatter(y=diff_added[0:time], name=f"Edges added", line=dict(width=3, color=my_colours[neurons])))
    fig.add_trace(go.Scatter(y=diff_removed[0:time], name=f"Edges removed", line=dict(width=3, color=my_colours[-neurons])))

    if error: 
        fig.add_trace(
            go.Scatter(
            name = "Upper bound",
            y = diff_added[0:time] + diff_added_std[0:time],
            mode = 'lines',
            marker = dict(color=my_colours[neurons]),
            line = dict(width = 0),
            showlegend=False
        ))

        fig.add_trace(
            go.Scatter(
                name = "Lower bound",
                y = diff_added[0:time] - diff_added_std[0:time],
                marker = dict(color=my_colours[neurons]),
                line = dict(width = 0),
                mode = 'lines',
                fillcolor = translucent_colours[neurons], 
                fill = 'tonexty',
                showlegend=False
            )
        )

        fig.add_trace(
            go.Scatter(
            name = "Upper bound",
            y = diff_removed[0:time] + diff_removed_std[0:time],
            mode = 'lines',
            marker = dict(color=my_colours[-neurons]),
            line = dict(width = 0),
            showlegend=False
        ))

        fig.add_trace(
            go.Scatter(
                name = "Lower bound",
                y = diff_removed[0:time] - diff_removed_std[0:time],
                marker = dict(color=my_colours[-neurons]),
                line = dict(width = 0),
                mode = 'lines',
                fillcolor = translucent_colours[-neurons], 
                fill = 'tonexty',
                showlegend=False
            )
        )
    
    fig.update_layout(title=f'Difference in relative ATE for simplex with {neurons} neurons,<br>with {removed} % of the connections added or removed', 
                   xaxis_title='Time-shift [ms]',
                   yaxis_title='Difference in relative ATE',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image(f"figures/ate_dimension_added_removed_{removed}_diff_error_{error}_{error_type}.pdf")


# ate_diff_added_removed(15, 10, 10, True, 'se')

# ate_diff_added_removed(15, 10, 15, True, 'se')





def ate_diff_added_removed_continuous(time, neurons, max_change, error, error_type):
    fig_diff = go.Figure()
    fig_abs = go.Figure()

    complete_path = f"../data/simplex/stats/ATE/cluster_sizes_[{neurons}].csv"

    x = np.arange(-max_change, max_change + 1)

    y_abs = np.zeros((time, 2*max_change + 1))
    y_abs_std = np.zeros((time, 2*max_change + 1))

    y_diff = np.zeros((time, 2*max_change + 1))
    y_diff_std = np.zeros((time, 2*max_change + 1))

    c_df = pd.read_csv(complete_path)
    c_rel_ate = c_df["relative_ATE"]
    c_rel_ate_std = c_df["relative_ATE_std"]

    if error_type == "se":
        c_rel_ate_std = c_rel_ate_std / np.sqrt(200)

    y_abs[0:time, max_change] = c_rel_ate[0:time]
    y_abs_std[0:time, max_change] = c_rel_ate_std[0:time]

    y_diff[0:time, max_change] = np.zeros(time)
    y_diff_std[0:time, max_change] = c_rel_ate_std[0:time]


    for change in range(1, max_change + 1):

        removed_path = f"../data/simplex/stats/ATE/cluster_sizes_[{neurons}]_removed_{change}.csv"
        added_path = f"../data/simplex/stats/ATE/cluster_sizes_[{neurons}]_added_{change}.csv"

        r_df = pd.read_csv(removed_path)
        a_df = pd.read_csv(added_path)

        r_rel_ate = r_df["relative_ATE"]
        r_rel_ate_std = r_df["relative_ATE_std"]

        a_rel_ate = a_df["relative_ATE"]
        a_rel_ate_std = a_df["relative_ATE_std"]

        if error_type == "se":
            r_rel_ate_std = r_rel_ate_std / np.sqrt(200)
            a_rel_ate_std = a_rel_ate_std / np.sqrt(200)

        diff_added = a_rel_ate - c_rel_ate
        diff_added_std = np.sqrt(c_rel_ate_std**2 + a_rel_ate_std**2)

        diff_removed = r_rel_ate - c_rel_ate
        diff_removed_std = np.sqrt(c_rel_ate_std**2 + r_rel_ate_std**2)

        y_diff[0:time, max_change - change] = diff_removed[0:time]
        y_diff_std[0:time, max_change - change] = diff_removed_std[0:time]

        y_diff[0:time, max_change + change] = diff_added[0:time]
        y_diff_std[0:time, max_change + change] = diff_added_std[0:time]


        y_abs[0:time, max_change - change] = r_rel_ate[0:time]
        y_abs_std[0:time, max_change - change] = r_rel_ate_std[0:time]

        y_abs[0:time, max_change + change] = a_rel_ate[0:time]
        y_abs_std[0:time, max_change + change] = a_rel_ate_std[0:time]

    for t in range(0, time):
        fig_diff.add_trace(go.Scatter(x = x, y=y_diff[t], name=f"{t} ms", line=dict(width=3, color=my_colours[t])))

        fig_abs.add_trace(go.Scatter(x = x, y=y_abs[t], name=f"{t} ms", line=dict(width=3, color=my_colours[t])))
        
        if error: 
            fig_diff.add_trace(
                go.Scatter(
                name = "Upper bound",
                y = y_diff[t] + y_diff_std[t],
                x = x,
                mode = 'lines',
                marker = dict(color=my_colours[t]),
                line = dict(width = 0),
                showlegend=False
            ))

            fig_diff.add_trace(
                go.Scatter(
                    name = "Lower bound",
                    y = y_diff[t] - y_diff_std[t],
                    x = x,
                    marker = dict(color=my_colours[t]),
                    line = dict(width = 0),
                    mode = 'lines',
                    fillcolor = translucent_colours[t], 
                    fill = 'tonexty',
                    showlegend=False
                )
            )

            fig_abs.add_trace(
                go.Scatter(
                name = "Upper bound",
                y = y_abs[t] + y_abs_std[t],
                x = x,
                mode = 'lines',
                marker = dict(color=my_colours[t]),
                line = dict(width = 0),
                showlegend=False
            ))

            fig_abs.add_trace(
                go.Scatter(
                    name = "Lower bound",
                    y = y_abs[t] - y_abs_std[t],
                    x = x,
                    marker = dict(color=my_colours[time]),
                    line = dict(width = 0),
                    mode = 'lines',
                    fillcolor = translucent_colours[t], 
                    fill = 'tonexty',
                    showlegend=False
                )
            )

    fig_diff.update_layout(
                title="Difference in relative ATE for different time-shifts for simplex<br>of 8 neurons with edges removed or added for time-shifts", 
                legend_title='Time-shift',
                xaxis_title='Change in edges',
                yaxis_title='Difference in relative ATE',
                font_family = "Garamond",
                font_size = 15)

    fig_diff.write_image(f"figures/ate_{neurons}_neurons_added_removed_diff_error_{error}_{error_type}.pdf")


    fig_abs.update_layout(
                title="Relative ATE for different time-shifts for simplex of 8<br>neurons with edges removed or added for time-shifts", 
                legend_title='Time-shift',
                xaxis_title='Change in edges',
                yaxis_title='Relative ATE',
                font_family = "Garamond",
                font_size = 15)

    fig_abs.write_image(f"figures/ate_{neurons}_neurons_added_removed_abs_error_{error}_{error_type}.pdf")


# ate_diff_added_removed_continuous(5, 8, 5, True, 'std')
# ate_diff_added_removed_continuous(5, 8, 5, True, 'se')






def ate_diff_added_removed_filter(time, neurons, removed, filter, error, error_type):
    fig = go.Figure()

    complete_path = f"../data/simplex/stats/ATE/cluster_sizes_[{neurons}]_{filter}.csv"
    removed_path = f"../data/simplex/stats/ATE/cluster_sizes_[{neurons}]_removed_{removed}_{filter}.csv"
    added_path = f"../data/simplex/stats/ATE/cluster_sizes_[{neurons}]_added_{removed}_{filter}.csv"

    c_df = pd.read_csv(complete_path)
    r_df = pd.read_csv(removed_path)
    a_df = pd.read_csv(added_path)

    c_rel_ate = c_df["relative_ATE"]
    c_rel_ate_std = c_df["relative_ATE_std"]

    r_rel_ate = r_df["relative_ATE"]
    r_rel_ate_std = r_df["relative_ATE_std"]

    a_rel_ate = a_df["relative_ATE"]
    a_rel_ate_std = a_df["relative_ATE_std"]

    if error_type == "se":
        c_rel_ate_std = c_rel_ate_std / np.sqrt(200)
        r_rel_ate_std = r_df["relative_ATE_se"] 
        a_rel_ate_std =  a_df["relative_ATE_se"] 

    diff_added = a_rel_ate - c_rel_ate
    diff_added_std = np.sqrt(c_rel_ate_std**2 + a_rel_ate_std**2)

    diff_removed = r_rel_ate - c_rel_ate
    diff_removed_std = np.sqrt(c_rel_ate_std**2 + r_rel_ate_std**2)

    fig.add_trace(go.Scatter(y=diff_added[0:time], name=f"Edges added", line=dict(width=3, color=my_colours[8])))
    fig.add_trace(go.Scatter(y=diff_removed[0:time], name=f"Edges removed", line=dict(width=3, color=my_colours[5])))

    if error: 
        fig.add_trace(
            go.Scatter(
            name = "Upper bound",
            y = diff_added[0:time] + diff_added_std[0:time],
            mode = 'lines',
            marker = dict(color=my_colours[8]),
            line = dict(width = 0),
            showlegend=False
        ))

        fig.add_trace(
            go.Scatter(
                name = "Lower bound",
                y = diff_added[0:time] - diff_added_std[0:time],
                marker = dict(color=my_colours[8]),
                line = dict(width = 0),
                mode = 'lines',
                fillcolor = translucent_colours[8], 
                fill = 'tonexty',
                showlegend=False
            )
        )

        fig.add_trace(
            go.Scatter(
            name = "Upper bound",
            y = diff_removed[0:time] + diff_removed_std[0:time],
            mode = 'lines',
            marker = dict(color=my_colours[5]),
            line = dict(width = 0),
            showlegend=False
        ))

        fig.add_trace(
            go.Scatter(
                name = "Lower bound",
                y = diff_removed[0:time] - diff_removed_std[0:time],
                marker = dict(color=my_colours[5]),
                line = dict(width = 0),
                mode = 'lines',
                fillcolor = translucent_colours[5], 
                fill = 'tonexty',
                showlegend=False
            )
        )

    if filter == 'inhib':
        title = f'Difference in relative ATE for simplex with {neurons} neurons, with<br>{removed} % of the connections added or removed, inhibitory source'

    elif filter == 'excit':
        title = f'Difference in relative ATE for simplex with {neurons} neurons, with<br>{removed} % of the connections added or removed, excitatory source'
    
    fig.update_layout(title=title, 
                   xaxis_title='Time-shift [ms]',
                   yaxis_title='Difference in relative ATE',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image(f"figures/ate_{neurons}_neurons_added_removed_{removed}_diff_{filter}_error_{error}_{error_type}.pdf")


for size in range(4, 10):
    ate_diff_added_removed_filter(15, size, 15, 'inhib', True, 'se')
    ate_diff_added_removed_filter(15, size, 15, 'excit', True, 'se')

    ate_diff_added_removed_filter(15, size, 10, 'inhib', True, 'se')
    ate_diff_added_removed_filter(15, size, 10, 'excit', True, 'se')



def p_sink_given_source(time, min_dim, max_dim, error, error_type):
    fig = go.Figure()

    iteration = range(min_dim, max_dim)

    if error:
        iteration = reversed(iteration)

    for dim in iteration:
        path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}].csv"

        df = pd.read_csv(path)

        p_sink_given_source= df["p_sink_given_source"]
        p_sink_given_source_std = df["p_sink_given_source_std"]

        if error_type == 'se':
            p_sink_given_source_std = p_sink_given_source_std / np.sqrt(200)

        fig.add_trace(go.Scatter(y=p_sink_given_source[0:time], name=dim, line=dict(width=3, color=my_colours[dim])))

        fig.add_trace(
        go.Scatter(
        name = "Upper bound",
        y = p_sink_given_source[0:time] + p_sink_given_source_std[0:time],
        mode = 'lines',
        marker = dict(color=my_colours[dim]),
        line = dict(width = 0),
        showlegend=False
        ))

        if error: 
            fig.add_trace(
                go.Scatter(
                name = "Upper bound",
                y = p_sink_given_source[0:time] + p_sink_given_source_std[0:time],
                mode = 'lines',
                marker = dict(color=my_colours[dim]),
                line = dict(width = 0),
                showlegend=False
                ))

            fig.add_trace(
                go.Scatter(
                    name = "Lower bound",
                    y = p_sink_given_source[0:time] - p_sink_given_source_std[0:time],
                    marker = dict(color=my_colours[dim]),
                    line = dict(width = 0),
                    mode = 'lines',
                    fillcolor = translucent_colours[dim], 
                    fill = 'tonexty',
                    showlegend=False
                )
            )

    
    fig.update_layout(title='P(sink|source) for different simplex sizes over time', 
                    legend_title='Neurons',
                    xaxis_title='Time-shift [ms]',
                    yaxis_title='Firing probability',
                    font_family = "Garamond",
                    font_size = 15)

    fig.update_yaxes(range=[0.01, 0.08])

    fig.write_image(f"figures/p_sink_given_source_error_{error}_{error_type}.pdf")


#p_sink_given_source(15, 3, 10, True, 'se')


def p_sink_given_not_source(time, min_dim, max_dim, error = True, error_type = 'std'):
    fig = go.Figure()

    iteration = range(min_dim, max_dim)

    if error:
        iteration = reversed(iteration)

    for dim in iteration:
        path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}].csv"

        df = pd.read_csv(path)

        p_sink_given_not_source= df["p_sink_given_not_source"]
        p_sink_given_not_source_std = df["p_sink_given_not_source_std"]

        if error_type == 'se':
            p_sink_given_not_source_std = p_sink_given_not_source_std / np.sqrt(200)


        fig.add_trace(
            go.Scatter(
                y=p_sink_given_not_source[0:time], 
                name=dim, 
                line=dict(width=3, 
                color=my_colours[dim])))

        if error: 
            fig.add_trace(
                go.Scatter(
                name = "Upper bound",
                y = p_sink_given_not_source[0:time] + p_sink_given_not_source_std[0:time],
                mode = 'lines',
                marker = dict(color=my_colours[dim]),
                line = dict(width = 0),
                showlegend=False
                ))

            fig.add_trace(
                go.Scatter(
                    name = "Lower bound",
                    y = p_sink_given_not_source[0:time] - p_sink_given_not_source_std[0:time],
                    marker = dict(color=my_colours[dim]),
                    line = dict(width = 0),
                    mode = 'lines',
                    fillcolor = translucent_colours[dim], 
                    fill = 'tonexty',
                    showlegend=False
                )
            )

    
    fig.update_layout(
                    title='P(sink|¬source) for different simplex sizes over time', 
                    legend_title='Neurons',
                    xaxis_title='Time-shift [ms]',
                    yaxis_title='Firing probability',
                    font_family = "Garamond",
                    font_size = 15)

    fig.write_image(f"figures/p_sink_given_not_source_error_{error}_{error_type}.pdf")


#p_sink_given_not_source(15, 3, 10, True, 'se')



def p_fire(min_dim, max_dim, error_type):

    f_rates = np.zeros(max_dim - min_dim)
    f_rates_std = np.zeros(max_dim - min_dim)
    dims = np.zeros(max_dim - min_dim)

    for idx, dim in enumerate(range(min_dim, max_dim)):
        dims[idx] = dim

        path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}].csv"
        p_fire = pd.read_csv(path)["p_sink"][0]
        p_fire_std = pd.read_csv(path)["p_sink_std"][0]

        if error_type == 'se':
            p_fire_std = p_fire_std / np.sqrt(200)

        f_rates[idx] = p_fire
        f_rates_std[idx] = p_fire_std

    fig = go.Figure([
        go.Scatter(
            x=dims, 
            y=f_rates, 
            line=dict(width=3, color='rgba(0, 131, 143, 1)'),
            showlegend=False),

        go.Scatter(
            name = "Upper bound",
            x = dims,
            y = f_rates + f_rates_std,
            mode = 'lines',
            marker = dict(color = 'rgba(0, 131, 143)'),
            line = dict(width = 0),
            showlegend=False
        ),

        go.Scatter(
            name = "Lower bound",
            x = dims,
            y = f_rates - f_rates_std,
            marker = dict(color = 'rgba(0, 131, 143)'),
            line = dict(width = 0),
            mode = 'lines',
            fillcolor = 'rgba(0, 131, 143, 0.3)', 
            fill = 'tonexty',
            showlegend=False
        )
        ])
        

    fig.update_layout(title='Average overall firing probability of sink<br>neuron for simplices of different sizes',
                        xaxis_title='Neurons',
                        yaxis_title='Firing probability',
                        font_family = "Garamond",
                        font_size = 15)

    fig.write_image(f"figures/p_sink_fire_{error_type}.pdf")


#p_fire(3, 10, 'se')






# time = 10
# min_dim = 4
# max_dim = 10


# ate_dimension(time, max_dim)
# ate_removed(time, min_dim, max_dim, 20)

# ate_diff(time, min_dim, max_dim, 20)
# p_sink_given_source(time, min_dim, max_dim)
# p_sink_given_not_source(time, min_dim, max_dim)
# p_fire(min_dim, max_dim)





