import pyinform as pi
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import coo_matrix
import torch
from tqdm import tqdm
import networkx as nx
import pathlib
from pathlib import Path
import pingouin
from tqdm.contrib import tzip
import statsmodels.tsa.stattools 
from scipy.stats import sem



def get_data(path):
    with open(path, "rb") as f:
        stuff = pickle.load(f)   #stuff = dictionary

    X_sparse = stuff["X_sparse"]

    coo = coo_matrix(X_sparse)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    X = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense().numpy() #X has shape (n_neurons, n_timesteps), with 1 indicating that the neuron fired at that time step

    return X



def auto_corr_size(min_neurons, max_neurons, max_shift):

    for size in tqdm(range(min_neurons, max_neurons)):

        filename = f"../data/simplex/stats/autocorrelation/autocorrelation_{size}.csv"
        with open(filename, "w") as f:
            f.write("t_shift,stim_15,stim_15_std,stim_15_se,stim_20,stim_20_std,stim_20_se\n")

        all_corr_15 = np.zeros((200, max_shift))
        all_corr_20 = np.zeros((200, max_shift))
        
        for network in tqdm(range(200)):
            filename_15 = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/stimulus_rate_15/{network}.pkl"

            filename_20 = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/stimulus_rate_20/{network}.pkl"

            X_15 = get_data(filename_15)
            X_20 = get_data(filename_20)

            sink_15 = X_15[-1, :]
            sink_20 = X_20[-1, :]

            corr_15 = statsmodels.tsa.stattools.acf(sink_15, nlags=max_shift)
            corr_15 = corr_15[1:]


            corr_20 = statsmodels.tsa.stattools.acf(sink_20, nlags=max_shift)
            corr_20 = corr_20[1:]

            all_corr_15[network, :] = corr_15
            all_corr_20[network, :] = corr_20

        all_corr_15_mean = np.mean(all_corr_15, axis=0)
        all_corr_15_std = np.std(all_corr_15, axis=0)
        all_corr_15_se = sem(all_corr_15, axis=0)

        all_corr_20_mean = np.mean(all_corr_20, axis=0)
        all_corr_20_std = np.std(all_corr_20, axis=0)
        all_corr_20_se = sem(all_corr_20, axis=0)

        for t_shift in range(max_shift):
            with open(filename, "a") as f:
                f.write(f"{t_shift+1},{all_corr_15_mean[t_shift]},{all_corr_15_std[t_shift]},{all_corr_15_se[t_shift]},{all_corr_20_mean[t_shift]},{all_corr_20_std[t_shift]},{all_corr_20_se[t_shift]}\n")
                
            f.close()
            

auto_corr_size(3, 11, 50)



def auto_corr_added_removed(neurons, change_type, changed_neurons,  max_shift):

    for change in tqdm(range(1, changed_neurons)):

        filename = f"../data/simplex/stats/autocorrelation/autocorrelation_{neurons}_{change_type}_{change}.csv"

        with open(filename, "w") as f:
            f.write("t_shift,stim_15,stim_15_std,stim_15_se,stim_20,stim_20_std,stim_20_se\n")

        all_corr_15 = np.zeros((200, max_shift))
        all_corr_20 = np.zeros((200, max_shift))

        for network in tqdm(range(200)):
            filename_15 = f"../data/simplex/cluster_sizes_[{neurons}]_n_steps_200000/{change_type}_{change}/stimulus_rate_15/{network}.pkl"

            filename_20 = f"../data/simplex/cluster_sizes_[{neurons}]_n_steps_200000/{change_type}_{change}/stimulus_rate_20/{network}.pkl"

            X_15 = get_data(filename_15)
            X_20 = get_data(filename_20)

            sink_15 = X_15[-1, :]
            sink_20 = X_20[-1, :]

            corr_15 = statsmodels.tsa.stattools.acf(sink_15, nlags=max_shift)
            corr_15 = corr_15[1:]

            corr_20 = statsmodels.tsa.stattools.acf(sink_20, nlags=max_shift)
            corr_20 = corr_20[1:]

            all_corr_15[network, :] = corr_15
            all_corr_20[network, :] = corr_20

        all_corr_15_mean = np.mean(all_corr_15, axis=0)
        all_corr_15_std = np.std(all_corr_15, axis=0)
        all_corr_15_se = sem(all_corr_15, axis=0)

        all_corr_20_mean = np.mean(all_corr_20, axis=0)
        all_corr_20_std = np.std(all_corr_20, axis=0)
        all_corr_20_se = sem(all_corr_20, axis=0)

        for t_shift in range(max_shift):
            with open(filename, "a") as f:
                f.write(f"{t_shift+1},{all_corr_15_mean[t_shift]},{all_corr_15_std[t_shift]},{all_corr_15_se[t_shift]},{all_corr_20_mean[t_shift]},{all_corr_20_std[t_shift]},{all_corr_20_se[t_shift]}\n")
                
            f.close()
            

auto_corr_added_removed(8, "added", 6, 50)
auto_corr_added_removed(8, "removed", 6, 50)
