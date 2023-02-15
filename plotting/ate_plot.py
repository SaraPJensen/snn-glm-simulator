import pickle 

import numpy as np
import pandas as pd
import plotly as py

import plotly.graph_objs as go
import plotly.express as px


def ate_dimension(time, max_dim):
    fig = go.Figure()

    for dim in range(3, max_dim):
        path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}].csv"

        df = pd.read_csv(path)

        rel_ate = df["relative_ATE"]
        abs_ate = df["absolute_ATE"]

        fig.add_trace(go.Scatter(y=rel_ate[0:time], name=f"Dim.: {dim}", line=dict(width=3)))

    
    fig.update_layout(title='Relative ATE for different simplex dimensions over time', 
                   xaxis_title='<i>t</i> [ms]',
                   yaxis_title='Relative ATE',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image("figures/ate_dimension.pdf")



def correlation(time, max_dim):
    fig = go.Figure()

    for dim in range(3, max_dim):
        path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}].csv"

        df = pd.read_csv(path)

        corr = df["correlation"]

        fig.add_trace(go.Scatter(y=corr[0:time], name=f"Dim.: {dim}", line=dict(width=3)))

    
    fig.update_layout(title='Correlation between source and sink for<br>different simplex dimensions over time', 
                   xaxis_title='<i>t</i> [ms]',
                   yaxis_title='Correlation',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image("figures/correlation_dimension.pdf")





def ate_removed(time, min_dim, max_dim, removed):
    fig = go.Figure()

    for dim in range(min_dim, max_dim):
        complete_path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}].csv"
        removed_path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}]_removed_{removed}.csv"

        c_df = pd.read_csv(complete_path)
        r_df = pd.read_csv(removed_path)

        c_rel_ate = c_df["relative_ATE"]
        r_rel_ate = r_df["relative_ATE"]

        fig.add_trace(go.Scatter(y=c_rel_ate[0:time], name=f"Dim.: {dim}, complete", line=dict(width=3)))
        fig.add_trace(go.Scatter(y=r_rel_ate[0:time], name=f"Dim.: {dim}, removed {removed}", line=dict(width=3, dash='dash')))

    
    fig.update_layout(title=f'Relative ATE for different simplex dimensions, both complete<br> and with {removed} % of the connections removed', 
                   xaxis_title='<i>t</i> [ms]',
                   yaxis_title='Relative ATE',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image(f"figures/ate_dimension_removed_{removed}.pdf")



def ate_diff(time, min_dim, max_dim, removed):
    fig = go.Figure()

    for dim in range(min_dim, max_dim):
        complete_path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}].csv"
        removed_path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}]_removed_{removed}.csv"

        c_df = pd.read_csv(complete_path)
        r_df = pd.read_csv(removed_path)

        c_rel_ate = c_df["relative_ATE"]
        r_rel_ate = r_df["relative_ATE"]
        diff = c_rel_ate - r_rel_ate

        fig.add_trace(go.Scatter(y=diff[0:time], name=f"Dim.: {dim}", line=dict(width=3)))
    
    fig.update_layout(title=f'Difference in relative ATE for different simplex dimensions,<br>with {removed} % of the connections removed', 
                   xaxis_title='<i>t</i> [ms]',
                   yaxis_title='Relative ATE',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image(f"figures/ate_dimension_removed_{removed}_diff.pdf")



def p_sink_given_source(time, min_dim, max_dim):
    fig = go.Figure()

    for dim in range(min_dim, max_dim):
        path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}].csv"

        df = pd.read_csv(path)

        p_sink_given_source= df["p_sink_given_source"]

        fig.add_trace(go.Scatter(y=p_sink_given_source[0:time], name=f"Dim.: {dim}", line=dict(width=3)))

    
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

        fig.add_trace(go.Scatter(y=p_sink_given_not_source[0:time], name=f"Dim.: {dim}", line=dict(width=3)))

    
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


time = 20
min_dim = 3
max_dim = 10

#correlation(time, max_dim)
#ate_dimension(time, max_dim)
#ate_removed(time, min_dim, max_dim, 15)

#ate_diff(time, min_dim, max_dim, 15)
p_sink_given_source(time, min_dim, max_dim)
p_sink_given_not_source(time, min_dim, max_dim)
p_fire(min_dim, max_dim)


