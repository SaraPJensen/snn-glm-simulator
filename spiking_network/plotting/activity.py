import plotly.figure_factory as ff
import numpy as np
import pickle
import plotly.graph_objects as go
import torch
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd



def more_activity(neuron_id: list, path: str):
        
    with open(path, "rb") as f:
        stuff = pickle.load(f)   #stuff = dictionary

    X_sparse = stuff["X_sparse"]

    #X = X_sparse.to_dense()

    coo = coo_matrix(X_sparse)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    X = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense() #X has shape (n_neurons, n_timesteps), with 1 indicating that the neuron fired at that time step

    # print(X.shape)
    # print(type(X))

    tot_secs = 10000/1000
    frequency = torch.sum(X)/tot_secs/X.shape[0]


    print(frequency.item())

    data = [X[id, :].numpy() for id in neuron_id]
    labels = [f"Neuron {id}" for id in neuron_id]

    print(sum(data[0][0:1000]))
    print(sum(data[0][1000:2000]))
    print(sum(data[0][0:3000]))

    fig = px.histogram(data, x="day")

    savepath = "spiking_network/plotting/figures/activity.png"
    fig.write_image(savepath)




def plot_activity(neuron_id: list, path: str):

    with open(path, "rb") as f:
        stuff = pickle.load(f)   #stuff = dictionary

    X_sparse = stuff["X_sparse"]

    #X = X_sparse.to_dense()

    coo = coo_matrix(X_sparse)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    X = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense() #X has shape (n_neurons, n_timesteps), with 1 indicating that the neuron fired at that time step

    # print(X.shape)
    # print(type(X))

    tot_secs = 10000/1000
    frequency = torch.sum(X)/tot_secs/X.shape[0]


    #print(frequency.item())

    n_timesteps = X.shape[1]

    data = [X[id, :].numpy() for id in neuron_id]
    labels = [f"Neuron {id}" for id in neuron_id]

    bin_size = 500

    time = {"Time": np.linspace(0, n_timesteps, n_timesteps//bin_size, endpoint = False)}

    data = [[np.sum(data[idx][i:i+bin_size]) for i in range(0, n_timesteps, bin_size)] for idx in range(len(neuron_id))] 

    df = pd.DataFrame(time)

    for i in range(len(neuron_id)):
        df.insert(i+1, column = labels[i], value = data[i])


    print(df)


    #exit()

    #fig = px.histogram(df, x = "Time")

    colours = ["firebrick", "Teal"]

    fig = go.Figure()

    for i in range(len(neuron_id)):
        fig.add_trace(go.Bar(x=df["Time"], y = df[labels[i]], name = labels[i], marker_color = colours[i], opacity = 0.3))

        #fig.update_traces(c)
        fig.update_xaxes(title_font_family='Times New Roman')

        fig.add_trace(go.Scatter(x = df["Time"], y = df[labels[i]], name = labels[i], 
        line_shape = 'spline', line=dict(color=colours[i])))

    fig.update_layout(  barmode='group', 
                        xaxis_title="Time (ms)", 
                        yaxis_title="Number of firings", 
                        title = "Activity of neurons over time", 
                        font_family = "Times New Roman",
                        #font_size = 20,
                        title_font_family = "Times New Roman")

    
    savepath = "spiking_network/plotting/figures/activity.png"
    fig.write_image(savepath)






def visualize_spikes(path):
    """
    Plots the number of firings per neuron and per timestep.

    Parameters:
    ----------
    X: torch.Tensor
    """

    with open(path, "rb") as f:
        stuff = pickle.load(f)   #stuff = dictionary

    X_sparse = stuff["X_sparse"]
    coo = coo_matrix(X_sparse)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    X = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense() #X has shape (n_neurons, n_timesteps), with 1 indicating that the neuron fired at that time step

    n_neurons = X.shape[0]
    n_timesteps = X.shape[1]
    n_bins = 100
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))

    fig.set_figheight(4.5)
    fig.set_figwidth(12)
    axes[0].set_title("Firings per neuron")
    axes[0].set_ylabel("Firings")
    axes[0].set_xlabel("Neuron")
    axes[0].bar(range(1, n_neurons + 1), torch.sum(X, axis=1), lw=0)

    axes[1].set_title("Firings per timestep")
    axes[1].set_ylabel("Firings")
    axes[1].set_xlabel(f"Timebin ({n_timesteps // n_bins} steps per bin)")

    firings_per_bin = torch.sum(X, axis=0).reshape(n_timesteps // n_bins, -1).sum(axis=0)
    axes[1].plot(
        range(1, n_bins + 1),
        firings_per_bin,
    )

    savepath = "spiking_network/plotting/figures/activity_avg.png"
    plt.savefig(savepath)
