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






def ate_dimension(time, max_dim):
    fig = go.Figure()

    for dim in range(3, max_dim):
        path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}].csv"

        df = pd.read_csv(path)

        rel_ate = df["relative_ATE"]
        rel_ate_std = df["relative_ATE_std"]

        fig.add_trace(go.Scatter(y=rel_ate[0:time], name=f"Neurons: {dim}", line=dict(width=3, color=my_colours[dim])))

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

    fig.update_layout(title='Relative ATE for different simplex dimensions over time', 
                   xaxis_title='<i>t</i> [ms]',
                   yaxis_title='Relative ATE',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image("figures/ate_dimension.pdf")

#ate_dimension(15, 15)


def ate_dimension_filter(time, max_dim, filter: str):
    fig = go.Figure()

    for dim in reversed(range(3, max_dim)):
        if filter == 'inhib':
            path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}]_inhib.csv"
            title = 'Relative ATE for different simplex dimensions<br>over time for inhibitory source neurons'
        
        else: 
            path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}]_excit.csv"
            title = 'Relative ATE for different simplex dimensions<br>over time for excitatory source neurons'

        df = pd.read_csv(path)

        rel_ate = df["relative_ATE"]
        rel_ate_std = df["relative_ATE_std"]

        fig.add_trace(go.Scatter(y=rel_ate[0:time], name=f"Neurons: {dim}", line=dict(width=3, color=my_colours[dim])))

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

    
    fig.update_layout(title=title, 
                   xaxis_title='<i>t</i> [ms]',
                   yaxis_title='Relative ATE',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image(f"figures/ate_dimension_{filter}.pdf")


#ate_dimension_filter(15, 15, 'inhib')



def correlation(time, max_dim):
    fig = go.Figure()

    for dim in reversed(range(3, max_dim)):
        path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}].csv"

        df = pd.read_csv(path)

        corr = df["correlation"][0:time]
        corr_std = df["correlation_std"][0:time]

        fig.add_trace(go.Scatter(y=corr, name=f"Neurons: {dim}", line=dict(width=3, color=my_colours[dim])))

        
        fig.add_trace(
            go.Scatter(
            name = "Upper bound",
            y = corr + corr_std,
            mode = 'lines',
            marker = dict(color=my_colours[dim]),
            line = dict(width = 0),
            showlegend=False
        ))

        fig.add_trace(
            go.Scatter(
                name = "Lower bound",
                y = corr - corr_std,
                marker = dict(color=my_colours[dim]),
                line = dict(width = 0),
                mode = 'lines',
                fillcolor = translucent_colours[dim], 
                fill = 'tonexty',
                showlegend=False
            )
        )

    fig.update_layout(title='Correlation between source and sink for<br>different simplex dimensions over time', 
                   xaxis_title='<i>t</i> [ms]',
                   yaxis_title='Correlation',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image("figures/correlation_dimension.pdf")


correlation(15, 15)


def correlation_filter(time, max_dim, filter):
    fig = go.Figure()

    for dim in reversed(range(3, max_dim)):
        if filter == 'inhib':
            path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}]_inhib.csv"
            title = 'Correlation between source and sink for different simplex<br>dimensions over time for inhibitory source neurons'
        
        else: 
            path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}]_excit.csv"
            title = 'Correlation between source and sink for different simplex<br>dimensions over time for excitatory source neurons'

        df = pd.read_csv(path)

        corr = df["correlation"][0:time]
        corr_std = df["correlation_std"][0:time]

        fig.add_trace(go.Scatter(y=corr, name=f"Neurons: {dim}", line=dict(width=3, color=my_colours[dim])))

        
        fig.add_trace(
            go.Scatter(
            name = "Upper bound",
            y = corr + corr_std,
            mode = 'lines',
            marker = dict(color=my_colours[dim]),
            line = dict(width = 0),
            showlegend=False
        ))

        fig.add_trace(
            go.Scatter(
                name = "Lower bound",
                y = corr - corr_std,
                marker = dict(color=my_colours[dim]),
                line = dict(width = 0),
                mode = 'lines',
                fillcolor = translucent_colours[dim], 
                fill = 'tonexty',
                showlegend=False
            )
        )

    fig.update_layout(title=title, 
                   xaxis_title='<i>t</i> [ms]',
                   yaxis_title='Correlation',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image(f"figures/correlation_dimension_{filter}.pdf")


correlation_filter(15, 15, 'inhib')
correlation_filter(15, 15, 'excit')




def ate_removed(time, min_dim, max_dim, removed):
    fig = go.Figure()

    for dim in range(min_dim, max_dim):
        complete_path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}].csv"
        removed_path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}]_removed_{removed}.csv"
        #added_path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}]_added_{removed}.csv"

        c_df = pd.read_csv(complete_path)
        r_df = pd.read_csv(removed_path)
        #a_df = pd.read_csv(added_path)

        c_rel_ate = c_df["relative_ATE"]
        r_rel_ate = r_df["relative_ATE"]
        #a_rel_ate = a_df["relative_ATE"]

        fig.add_trace(go.Scatter(y=c_rel_ate[0:time], name=f"Neurons: {dim}, complete", line=dict(width=3, color=my_colours[dim])))
        fig.add_trace(go.Scatter(y=r_rel_ate[0:time], name=f"Neurons: {dim}, removed {removed}", line=dict(width=3, dash='dash', color=my_colours[dim])))
        #fig.add_trace(go.Scatter(y=a_rel_ate[0:time], name=f"Neurons: {dim}, added {removed}", line=dict(width=3, dash='dot', color=my_colours[dim])))

    
    fig.update_layout(title=f'Relative ATE for different simplex dimensions, both complete<br> and with {removed} % of the connections removed', 
                   xaxis_title='<i>t</i> [ms]',
                   yaxis_title='Relative ATE',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image(f"figures/ate_dimension_removed_{removed}.pdf")


ate_removed(10, 4, 10, 15)


def ate_diff(time, min_dim, max_dim, removed):
    fig = go.Figure()

    for dim in range(min_dim, max_dim):
        complete_path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}].csv"
        removed_path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}]_removed_{removed}.csv"

        c_df = pd.read_csv(complete_path)
        r_df = pd.read_csv(removed_path)

        c_rel_ate = c_df["relative_ATE"]
        c_rel_ate_std = c_df["relative_ATE_std"]

        r_rel_ate = r_df["relative_ATE"]
        r_rel_ate_std = r_df["relative_ATE_std"]

        diff = c_rel_ate - r_rel_ate
        diff_std = np.sqrt(c_rel_ate_std**2 + r_rel_ate_std**2)

        fig.add_trace(go.Scatter(y=diff[0:time], name=f"Neurons: {dim}", line=dict(width=3, color=my_colours[dim])))

        fig.add_trace(
            go.Scatter(
            name = "Upper bound",
            y = diff + diff_std,
            mode = 'lines',
            marker = dict(color=my_colours[dim]),
            line = dict(width = 0),
            showlegend=False
        ))

        fig.add_trace(
            go.Scatter(
                name = "Lower bound",
                y = diff - diff_std,
                marker = dict(color=my_colours[dim]),
                line = dict(width = 0),
                mode = 'lines',
                fillcolor = translucent_colours[dim], 
                fill = 'tonexty',
                showlegend=False
            )
        )
    
    fig.update_layout(title=f'Difference in relative ATE for different simplex dimensions,<br>with {removed} % of the connections removed', 
                   xaxis_title='<i>t</i> [ms]',
                   yaxis_title='Relative ATE',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image(f"figures/ate_dimension_removed_{removed}_diff.pdf")

#ate_diff(15, 4, 11, 15)



def ate_diff_added_removed(time, neurons, removed):
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

    diff_added = c_rel_ate - a_rel_ate
    diff_added_std = np.sqrt(c_rel_ate_std**2 + a_rel_ate_std**2)

    diff_removed = c_rel_ate - r_rel_ate
    diff_removed_std = np.sqrt(c_rel_ate_std**2 + r_rel_ate_std**2)

    fig.add_trace(go.Scatter(y=diff_added[0:time], name=f"Edges added", line=dict(width=3, color=my_colours[neurons])))
    fig.add_trace(go.Scatter(y=diff_removed[0:time], name=f"Edges removed", line=dict(width=3, color=my_colours[-neurons])))

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
                   xaxis_title='<i>t</i> [ms]',
                   yaxis_title='Relative ATE',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image(f"figures/ate_dimension_added_removed_{removed}_diff.pdf")


#ate_diff_added_removed(15, 10, 15)




def ate_diff_added_removed_filter(time, neurons, removed, filter):
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

    diff_added = c_rel_ate - a_rel_ate
    diff_added_std = np.sqrt(c_rel_ate_std**2 + a_rel_ate_std**2)

    diff_removed = c_rel_ate - r_rel_ate
    diff_removed_std = np.sqrt(c_rel_ate_std**2 + r_rel_ate_std**2)

    fig.add_trace(go.Scatter(y=diff_added[0:time], name=f"Edges added", line=dict(width=3, color=my_colours[neurons])))
    fig.add_trace(go.Scatter(y=diff_removed[0:time], name=f"Edges removed", line=dict(width=3, color=my_colours[-neurons])))

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

    if filter == 'inhib':
        title = f'Difference in relative ATE for simplex with {neurons} neurons, with<br>{removed} % of the connections added or removed, inhibitory source'

    elif filter == 'excit':
        title = f'Difference in relative ATE for simplex with {neurons} neurons, with<br>{removed} % of the connections added or removed, excitatory source'
    
    fig.update_layout(title=title, 
                   xaxis_title='<i>t</i> [ms]',
                   yaxis_title='Relative ATE',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image(f"figures/ate_{neurons}_neurons_added_removed_{removed}_diff_{filter}.pdf")



# ate_diff_added_removed_filter(15, 10, 10, 'inhib')
# ate_diff_added_removed_filter(15, 10, 10, 'excit')



def p_sink_given_source(time, min_dim, max_dim):
    fig = go.Figure()

    for dim in range(min_dim, max_dim):
        path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}].csv"

        df = pd.read_csv(path)

        p_sink_given_source= df["p_sink_given_source"]

        fig.add_trace(go.Scatter(y=p_sink_given_source[0:time], name=f"Neurons: {dim}", line=dict(width=3, color=my_colours[dim])))

    
    fig.update_layout(title='P(sink|source) for different simplex dimensions over time', 
                   xaxis_title='<i>t</i> [ms]',
                   yaxis_title='Firing probability',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image("figures/p_sink_given_source.pdf")



def p_sink_given_not_source(time, min_dim, max_dim):
    fig = go.Figure()

    for dim in range(min_dim, max_dim):
        path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}].csv"

        df = pd.read_csv(path)

        p_sink_given_not_source= df["p_sink_given_not_source"]

        fig.add_trace(go.Scatter(y=p_sink_given_not_source[0:time], name=f"Neurons: {dim}", line=dict(width=3, color=my_colours[dim])))

    
    fig.update_layout(title='P(sink|¬source) for different simplex dimensions over time', 
                   xaxis_title='<i>t</i> [ms]',
                   yaxis_title='Firing probability',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image("figures/p_sink_given_not_source.pdf")



def p_fire(min_dim, max_dim):

    f_rates = []
    dims = []

    for dim in range(min_dim, max_dim):
        dims.append(dim)

        path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}].csv"
        p_fire = pd.read_csv(path)["p_sink"][0]
        f_rates.append(p_fire)

    fig = go.Figure(data=go.Scatter(x=dims, y=f_rates, mode='lines+markers'))

    fig.update_layout(title='Sink neuron average overall firing probability<br>for different simplex dimensions',
                        xaxis_title='Dimension',
                        yaxis_title='Firing probability',
                        font_family = "Garamond",
                        font_size = 15)

    fig.write_image("figures/p_sink_fire.pdf")




def correlation_second_last(time, max_dim):
    fig = go.Figure()

    for dim in range(3, max_dim):
        path = f"../data/simplex/stats/ATE/second_last_cluster_sizes_[{dim}].csv"

        df = pd.read_csv(path)

        corr = df["correlation"]

        fig.add_trace(go.Scatter(y=corr[0:time], name=f"Neurons: {dim}", line=dict(width=3, color=my_colours[dim])))

    
    fig.update_layout(title='Correlation between second last and sink neurons<br>for different simplex dimensions over time', 
                   xaxis_title='<i>t</i> [ms]',
                   yaxis_title='Correlation',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image("figures/correlation_dimension_second_last.pdf")





# time = 10
# min_dim = 4
# max_dim = 10

# correlation(time, max_dim)
# ate_dimension(time, max_dim)
# ate_removed(time, min_dim, max_dim, 20)

# ate_diff(time, min_dim, max_dim, 20)
# p_sink_given_source(time, min_dim, max_dim)
# p_sink_given_not_source(time, min_dim, max_dim)
# p_fire(min_dim, max_dim)

# correlation_second_last(time, max_dim)


