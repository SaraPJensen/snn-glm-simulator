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

    sink_block = df["sink_bloch_entropy"]
    sink_block_std = df["sink_bloch_std"]


    fig = go.Figure([
        go.Scatter(
            name = "Active Information",
            x = dimensionality, 
            y=sink_active, 
            line=dict(width=3, color='rgba(0, 131, 143, 1)'),
            showlegend=False),
        
        # go.Scatter(
        #     name = "Upper bound",
        #     x = dimensionality,
        #     y = sink_active + sink_active_std,
        #     mode = 'lines',
        #     marker = dict(color = 'rgba(0, 131, 143)'),
        #     line = dict(width = 0),
        #     showlegend=False
        # ),

        # go.Scatter(
        #     name = "Lower bound",
        #     x = dimensionality,
        #     y = sink_active - sink_active_std,
        #     marker = dict(color = 'rgba(0, 131, 143)'),
        #     line = dict(width = 0),
        #     mode = 'lines',
        #     fillcolor = 'rgba(0, 131, 143, 0.3)', 
        #     fill = 'tonexty',
        #     showlegend=False
        # )
    ])
    
    fig.update_layout(title='Active information of sink neurons in<br>simplices of different sizes', 
                   xaxis_title='Neurons',
                   yaxis_title='Active information',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image("figures/entropy_active_k20_error_False.pdf")

    fig_block = go.Figure([
        go.Scatter(
            name = "Block entropy",
            x = dimensionality,
            y = sink_block,
            line=dict(width=3, color='rgba(0, 131, 143, 1)'),
            showlegend=False),

        # go.Scatter(
        #     name = "Upper bound",
        #     x = dimensionality,
        #     y = sink_block + sink_block_std,
        #     mode = 'lines',
        #     marker = dict(color = 'rgba(0, 131, 143)'),
        #     line = dict(width = 0),
        #     showlegend=False
        # ),

        # go.Scatter(
        #     name = "Lower bound",
        #     x = dimensionality,
        #     y = sink_block - sink_block_std,
        #     marker = dict(color = 'rgba(0, 131, 143)'),
        #     line = dict(width = 0),
        #     mode = 'lines',
        #     fillcolor = 'rgba(0, 131, 143, 0.3)',
        #     fill = 'tonexty',
        #     showlegend=False
        # )

    ])

    fig_block.update_layout(
                    title='Block entropy of sink neurons in<br>simplices of different sizes',
                    xaxis_title='Neurons',
                    yaxis_title='Block entropy',
                    font_family="Garamond",
                    font_size=15)

    fig_block.write_image("figures/entropy_block_k20_error_False.pdf")



#active_information()




def active_information_added_removed(neurons, max_change, error):
    fig_active = go.Figure()
    fig_block = go.Figure()

    x = np.arange(-max_change, max_change + 1)
    y_active = np.zeros(2 * max_change + 1)
    y_active_std = np.zeros(2 * max_change + 1)
    y_block = np.zeros(2 * max_change + 1)
    y_block_std = np.zeros(2 * max_change + 1)

    complete_path = f"../data/simplex/stats/entropy_k20.csv"
    complete_df = pd.read_csv(complete_path)

    idx = complete_df[complete_df["neurons"] == neurons].index[0]

    y_active[max_change] = complete_df.loc[idx]["sink_active_info"]
    y_active_std[max_change] = complete_df.loc[idx]["sink_active_std"]
    y_block[max_change] = complete_df.loc[idx]["sink_bloch_entropy"]
    y_block_std[max_change] = complete_df.loc[idx]["sink_bloch_std"]

    removed_path = f"../data/simplex/stats/entropy_k20_removed_cluster_size_{neurons}.csv"
    added_path = f"../data/simplex/stats/entropy_k20_added_cluster_size_{neurons}.csv"
    r_df = pd.read_csv(removed_path)
    a_df = pd.read_csv(added_path)


    for change in range(1, max_change + 1):

        y_active[max_change - change] = r_df["sink_active_info"][change - 1]
        y_active[max_change + change] = a_df["sink_active_info"][change - 1]
        y_active_std[max_change - change] = r_df["sink_active_std"][change - 1]
        y_active_std[max_change + change] = a_df["sink_active_std"][change - 1]

        y_block[max_change - change] = r_df["sink_block_entropy"][change - 1]
        y_block[max_change + change] = a_df["sink_block_entropy"][change - 1]
        y_block_std[max_change - change] = r_df["sink_block_std"][change - 1]
        y_block_std[max_change + change] = a_df["sink_block_std"][change - 1]


    fig = go.Figure([
        go.Scatter(
            name = "Active Information",
            x = x, 
            y=y_active, 
            line=dict(width=3, color='rgba(0, 131, 143, 1)'),
            showlegend=False),
        
        # go.Scatter(
        #     name = "Upper bound",
        #     x = x,
        #     y = y_active + y_active_std,
        #     mode = 'lines',
        #     marker = dict(color = 'rgba(0, 131, 143)'),
        #     line = dict(width = 0),
        #     showlegend=False
        # ),

        # go.Scatter(
        #     name = "Lower bound",
        #     x = x,
        #     y = y_active - y_active_std,
        #     marker = dict(color = 'rgba(0, 131, 143)'),
        #     line = dict(width = 0),
        #     mode = 'lines',
        #     fillcolor = 'rgba(0, 131, 143, 0.3)', 
        #     fill = 'tonexty',
        #     showlegend=False
        # )
    ])
    
    fig.update_layout(title='Active information of sink neurons in a simplex of<br>8 neurons with edges added and removed', 
                   xaxis_title='Change in edges',
                   yaxis_title='Active information',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image("figures/entropy_active_added_removed_k20_error_False.pdf")



    fig_block = go.Figure([
        go.Scatter(
            name = "Block entropy",
            x = x,
            y = y_block,
            line=dict(width=3, color='rgba(0, 131, 143, 1)'),
            showlegend=False),

        # go.Scatter(
        #     name = "Upper bound",
        #     x = x,
        #     y = y_block + y_block_std,
        #     mode = 'lines',
        #     marker = dict(color = 'rgba(0, 131, 143)'),
        #     line = dict(width = 0),
        #     showlegend=False
        # ),

        # go.Scatter(
        #     name = "Lower bound",
        #     x = x,
        #     y = y_block - y_block_std,
        #     marker = dict(color = 'rgba(0, 131, 143)'),
        #     line = dict(width = 0),
        #     mode = 'lines',
        #     fillcolor = 'rgba(0, 131, 143, 0.3)',
        #     fill = 'tonexty',
        #     showlegend=False
        # )

    ])

    fig_block.update_layout(
                    title='Block entropy of sink neurons in a simplex of<br>8 neurons with edges added and removed',
                    xaxis_title='Change in edges',
                    yaxis_title='Block entropy',
                    font_family="Garamond",
                    font_size=15)

    fig_block.write_image("figures/entropy_block_added_removed_k20_error_False.pdf")



#active_information_added_removed(8, 5, True)




def transfer_entropy():

    path = f"../data/simplex/stats/transfer_entropy.csv"

    df = pd.read_csv(path)

    dimensionality = df["neurons"].to_numpy()

    transfer_entropy = df["transfer_entropy"].to_numpy()
    transfer_entropy_std = df["std"].to_numpy()


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
        
    
    fig.update_layout(title='Transfer entropy between source and sink neurons<br>for time-shift 1ms as function of simplex size', 
                   xaxis_title='Neurons',
                   yaxis_title='Transfer entropy',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image("figures/transfer_entropy.pdf")


#transfer_entropy()



def transfer_entropy_type(source_type):

    path = f"../data/simplex/stats/transfer_entropy_{source_type}.csv"

    df = pd.read_csv(path)

    dimensionality = df["neurons"].to_numpy()

    transfer_entropy = df["transfer_entropy"].to_numpy()
    transfer_entropy_std = df["std"].to_numpy()

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
        
    if source_type == "inhib":
        title = f'Transfer entropy between source and sink neurons as<br>function of simplex size with inhibitory source neurons'
    else: 
        title = f'Transfer entropy between source and sink neurons as<br>function of simplex size with excitatory source neurons'

    fig.update_layout(title=title, 
                   xaxis_title='Neurons',
                   yaxis_title='Transfer entropy',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image(f"figures/transfer_entropy_{source_type}.pdf")



# transfer_entropy_type("inhib")
# transfer_entropy_type("excit")




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


#conditional_entropy(15, 3, 15)




def transfer_entropy_dim_time(time, min_dim, max_dim, error):
    fig = go.Figure()

    iteration = range(min_dim, max_dim)
    if error:
        iteration = reversed(iteration)

    for dim in iteration:
        path = f"../data/simplex/stats/transfer_entropy/{dim}_neurons.csv"

        df = pd.read_csv(path)

        transfer_entropy = df["transfer_entropy"][0:time]
        transfer_entropy_std = df["std"][0:time]

        fig.add_trace(
            go.Scatter(
                y=transfer_entropy, 
                name=dim,
                line=dict(width=3, color=my_colours[dim])))

        if error:
            fig.add_trace(
                go.Scatter(
                name = "Upper bound",
                y = transfer_entropy + transfer_entropy_std,
                mode = 'lines',
                marker = dict(color=my_colours[dim]),
                line = dict(width = 0),
                showlegend=False
            ))

            fig.add_trace(
                go.Scatter(
                    name = "Lower bound",
                    y = transfer_entropy - transfer_entropy_std,
                    marker = dict(color=my_colours[dim]),
                    line = dict(width = 0),
                    mode = 'lines',
                    fillcolor = translucent_colours[dim], 
                    fill = 'tonexty',
                    showlegend=False
                )
            )


    fig.update_layout(title=f'Transfer entropy between source and sink neurons in<br>simplices of different dimensionality as function of time-shift', 
                   legend_title='Neurons',
                   xaxis_title='Time-shift [ms]',
                   yaxis_title='Transfer entropy',
                   font_family = "Garamond",
                   font_size = 15)


    fig.write_image(f"figures/transfer_entropy_dim_time_error_{error}.pdf")


# transfer_entropy_dim_time(11, 3, 10, True)
# transfer_entropy_dim_time(11, 3, 10, False)




def transfer_entropy_dim_time_sum(time, min_dim, max_dim, error):
    fig = go.Figure()

    entropy_sum = np.zeros(max_dim - min_dim)
    entropy_sum_std = np.zeros(max_dim - min_dim)

    dimensionality = np.arange(min_dim, max_dim)

    iteration = range(min_dim, max_dim)
    if error:
        iteration = reversed(iteration)

    for dim in iteration:
        path = f"../data/simplex/stats/transfer_entropy/{dim}_neurons.csv"

        df = pd.read_csv(path)

        transfer_entropy = np.sum(df["transfer_entropy"].to_numpy())
        transfer_entropy_std = np.sqrt(np.sum(df["std"].to_numpy()**2))

        entropy_sum[dim-min_dim] = transfer_entropy
        entropy_sum_std[dim-min_dim] = transfer_entropy_std

    fig = go.Figure([
        go.Scatter(
            name = "Entropy sum",
            x = dimensionality,
            y = entropy_sum,
            line=dict(width=3, color='rgba(0, 131, 143, 1)'),
            showlegend=False),

        go.Scatter(
            name = "Upper bound",
            x = dimensionality,
            y = entropy_sum + entropy_sum_std,
            mode = 'lines',
            marker = dict(color = 'rgba(0, 131, 143)'),
            line = dict(width = 0),
            showlegend=False
        ),

        go.Scatter(
            name = "Lower bound",
            x = dimensionality,
            y = entropy_sum - entropy_sum_std,
            marker = dict(color = 'rgba(0, 131, 143)'),
            line = dict(width = 0),
            mode = 'lines',
            fillcolor = 'rgba(0, 131, 143, 0.3)',
            fill = 'tonexty',
            showlegend=False
        )
    ])


    fig.update_layout(title=f'Sum of transfer entropy between source and sink neurons in <br>simplices of different dimensionality for {time} time-shifts', 
                   xaxis_title='Neurons',
                   yaxis_title='Sum of transfer entropy',
                   font_family = "Garamond",
                   font_size = 15)


    fig.write_image(f"figures/transfer_entropy_dim_time_sum_error_{error}.pdf")


#transfer_entropy_dim_time_sum(10, 3, 10, True)









def transfer_entropy_added_removed(neurons, max_change, error):
    fig = go.Figure()

    x = np.arange(-max_change, max_change + 1)

    y = np.zeros(2 * max_change + 1)
    y_std = np.zeros(2 * max_change + 1)

    complete_path = f"../data/simplex/stats/transfer_entropy/{neurons}_neurons.csv"
    complete_df = pd.read_csv(complete_path)

    y[max_change] = complete_df["transfer_entropy"][0]
    y_std[max_change] = complete_df["std"][0]

    removed_path = f"../data/simplex/stats/transfer_entropy/removed_cluster_size_{neurons}.csv"
    added_path = f"../data/simplex/stats/transfer_entropy/added_cluster_size_{neurons}.csv"
    r_df = pd.read_csv(removed_path)
    a_df = pd.read_csv(added_path)


    for change in range(1, max_change + 1):

        y[max_change - change] = r_df["transfer_entropy"][change - 1]
        y[max_change + change] = a_df["transfer_entropy"][change - 1]
        y_std[max_change - change] = r_df["std"][change - 1]
        y_std[max_change + change] = a_df["std"][change - 1]

    fig = go.Figure([
        go.Scatter(
            name = "Transfer entropy",
            x = x, 
            y=y, 
            line=dict(width=3, color='rgba(0, 131, 143, 1)'),
            showlegend=False),
    ])
    #     go.Scatter(
    #         name = "Upper bound",
    #         x = x,
    #         y = y + y_std,
    #         mode = 'lines',
    #         marker = dict(color = 'rgba(0, 131, 143)'),
    #         line = dict(width = 0),
    #         showlegend=False
    #     ),

    #     go.Scatter(
    #         name = "Lower bound",
    #         x = x,
    #         y = y - y_std,
    #         marker = dict(color = 'rgba(0, 131, 143)'),
    #         line = dict(width = 0),
    #         mode = 'lines',
    #         fillcolor = 'rgba(0, 131, 143, 0.3)', 
    #         fill = 'tonexty',
    #         showlegend=False
    #     )
    # ])
    
    fig.update_layout(title='Transfer entropy between sink and source in a<br>simplex of 8 neurons with edges added and removed', 
                   xaxis_title='Change in edges',
                   yaxis_title='Transfer entropy',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image("figures/transfer_entropy_added_removed_no_error.pdf")



#transfer_entropy_added_removed(8, 5, True)