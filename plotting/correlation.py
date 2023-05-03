



def correlation(time, max_dim, error):
    fig = go.Figure()

    iteration = range(3, max_dim)
    if error:
        iteration = reversed(iteration)

    for dim in iteration:
        path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}].csv"

        df = pd.read_csv(path)

        corr = df["correlation"][0:time]
        corr_std = df["correlation_std"][0:time]

        fig.add_trace(go.Scatter(y=corr, name=dim, line=dict(width=3, color=my_colours[dim])))

        if error: 
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

    fig.update_layout(
                title='Correlation between source and sink for<br>different simplex dimensions over time', 
                legend_title='Neurons',
                xaxis_title='Time-shift [ms]',
                yaxis_title='Correlation',
                font_family = "Garamond",
                font_size = 15)

    fig.write_image(f"figures/correlation_dimension_error_{error}.pdf")


# correlation(11, 10, True)
# correlation(11, 10, False)


def correlation_filter(time, max_dim, filter, error):

    fig = go.Figure()

    iteration = range(3, max_dim)
    if error:
        iteration = reversed(iteration)

    for dim in iteration:
        if filter == 'inhib':
            path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}]_inhib.csv"
            title = 'Correlation between source and sink for different simplex<br>dimensions over time for inhibitory source neurons'
        
        else: 
            path = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}]_excit.csv"
            title = 'Correlation between source and sink for different simplex<br>dimensions over time for excitatory source neurons'

        df = pd.read_csv(path)

        corr = df["correlation"][0:time]
        corr_std = df["correlation_std"][0:time]

        fig.add_trace(go.Scatter(y=corr, name=dim, line=dict(width=3, color=my_colours[dim])))

        if error:
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
                    legend_title='Neurons',
                    xaxis_title='Time-shift [ms]',
                    yaxis_title='Correlation',
                    font_family = "Garamond",
                    font_size = 15)

    fig.write_image(f"figures/correlation_dimension_{filter}_error_{error}.pdf")


# correlation_filter(11, 10, 'inhib', True)
# correlation_filter(11, 10, 'excit', True)

# correlation_filter(11, 10, 'inhib', False)
# correlation_filter(11, 10, 'excit', False)






def correlation_second_last(time, max_dim):
    fig = go.Figure()

    for dim in range(3, max_dim):
        path = f"../data/simplex/stats/ATE/second_last_cluster_sizes_[{dim}].csv"

        df = pd.read_csv(path)

        corr = df["correlation"]

        fig.add_trace(go.Scatter(y=corr[0:time], name=dim, line=dict(width=3, color=my_colours[dim])))

    
    fig.update_layout(title='Correlation between second last and sink neurons<br>for different simplex dimensions over time', 
                   legend_title='Neurons',
                   xaxis_title='Time-shift [ms]',
                   yaxis_title='Correlation',
                   font_family = "Garamond",
                   font_size = 15)

    fig.write_image("figures/correlation_dimension_second_last.pdf")

