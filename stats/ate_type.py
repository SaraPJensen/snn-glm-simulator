import numpy as np
import pingouin as pg
import pandas as pd
import pickle
from scipy.sparse import coo_matrix
import torch
import sys
sys.path.append("./..")
from tqdm import tqdm
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

    W0 = stuff["W0"]

    return X, W0



def count_type(dim, network_name):
    inhib = 0
    excit = 0

    for network in range(200):
        path = f"../data/{network_name}/cluster_sizes_[{dim}]_n_steps_200000/{network}.pkl"
        X, W0 = get_data(path)
        summing = torch.sum(W0[0, :])

        if summing < 0:
            inhib += 1
        else:
            excit += 1

    print(f"Number of inhibitory networks: {inhib}")
    print(f"Number of excitatory networks: {excit}")

    return inhib, excit



def ate_type(dim, t_shift, network_name):

    inhib, excit = count_type(dim, network_name)   #Differentiate between the networks where the source neuron is excitatory and inhibitory

    filename_inhib = f"../data/{network_name}/stats/ATE/cluster_sizes_[{dim}]_inhib.csv"
    with open(filename_inhib, "w") as f:
        f.write("t_shift,absolute_ATE,absolute_ATE_std,absolute_ATE_se,relative_ATE,relative_ATE_std,relative_ATE_se,p_source,p_source_std,p_sink,p_sink_std,p_sink_given_source,p_sink_given_source_std,p_sink_given_not_source,p_sink_given_not_source_std,correlation,correlation_std\n")
    
    filename_excit = f"../data/{network_name}/stats/ATE/cluster_sizes_[{dim}]_excit.csv"
    with open(filename_excit, "w") as f:
        f.write("t_shift,absolute_ATE,absolute_ATE_std,absolute_ATE_se,relative_ATE,relative_ATE_std,relative_ATE_se,p_source,p_source_std,p_sink,p_sink_std,p_sink_given_source,p_sink_given_source_std,p_sink_given_not_source,p_sink_given_not_source_std,correlation,correlation_std\n")

    in_all_rel_ate = np.zeros((t_shift, inhib))
    in_all_abs_ate = np.zeros((t_shift, inhib))
    in_all_p_source = np.zeros((t_shift, inhib))
    in_all_p_sink = np.zeros((t_shift, inhib))
    in_all_p_sink_given_source = np.zeros((t_shift, inhib))
    in_all_sink_given_not_source = np.zeros((t_shift, inhib))
    in_correlation = np.zeros((t_shift, inhib))

    ex_all_rel_ate = np.zeros((t_shift, excit))
    ex_all_abs_ate = np.zeros((t_shift, excit))
    ex_all_p_source = np.zeros((t_shift, excit))
    ex_all_p_sink = np.zeros((t_shift, excit))
    ex_all_p_sink_given_source = np.zeros((t_shift, excit))
    ex_all_sink_given_not_source = np.zeros((t_shift, excit))
    ex_correlation = np.zeros((t_shift, excit))

    inhib = 0
    excit = 0

    for network in tqdm(range(0, 200)):
        path = f"../data/{network_name}/cluster_sizes_[{dim}]_n_steps_200000/{network}.pkl"
        X, W0 = get_data(path)

        source = X[0, :]
        sink = X[-1, :]

        summing = torch.sum(W0[0, :])

        # print(f"Number of inhibitory networks: {inhib}")
        # print(f"Number of excitatory networks: {excit}")

        for t in range(0, t_shift):
            p_source = np.sum(source)/len(source)
            p_sink = np.sum(np.roll(sink, -t))/len(sink)
            p_source_and_sink = (np.sum(source*np.roll(sink, -t))/len(sink))
            p_sink_given_source = p_source_and_sink/p_source
            p_sink_given_not_source = (p_sink - p_source_and_sink)/(1 - p_source)
            ATE_abs = p_sink_given_source - p_sink_given_not_source
            ATE_rel = ATE_abs/p_sink_given_not_source   #The relative percentage-wise increase in the probability of the sink neuron firing given that the source neuron fired

            if summing < 0:
                in_all_rel_ate[t, inhib] = ATE_rel
                in_all_abs_ate[t, inhib] = ATE_abs
                in_all_p_source[t, inhib] = p_source
                in_all_p_sink[t, inhib] = p_sink
                in_all_p_sink_given_source[t, inhib] = p_sink_given_source
                in_all_sink_given_not_source[t, inhib] = p_sink_given_not_source
                in_correlation[t, inhib] = pg.corr(source, np.roll(sink, -t))['r'].values[0]
            
            else:
                ex_all_rel_ate[t, excit] = ATE_rel
                ex_all_abs_ate[t, excit] = ATE_abs
                ex_all_p_source[t, excit] = p_source
                ex_all_p_sink[t, excit] = p_sink
                ex_all_p_sink_given_source[t, excit] = p_sink_given_source
                ex_all_sink_given_not_source[t, excit] = p_sink_given_not_source
                ex_correlation[t, excit] = pg.corr(source, np.roll(sink, -t))['r'].values[0]

        if summing < 0:
            inhib += 1
        else:
            excit += 1

    with open(filename_inhib, "a") as f:
        for t in range(0, t_shift):
            f.write(f"{t},{np.mean(in_all_abs_ate[t, :])},{np.std(in_all_abs_ate[t, :])},{sem(in_all_abs_ate[t, :])},{np.mean(in_all_rel_ate[t, :])},{np.std(in_all_rel_ate[t, :])},{sem(in_all_rel_ate[t,:])},{np.mean(in_all_p_source[t, :])},{np.std(in_all_p_source[t, :])},{np.mean(in_all_p_sink[t, :])},{np.std(in_all_p_sink[t, :])},{np.mean(in_all_p_sink_given_source[t, :])},{np.std(in_all_p_sink_given_source[t, :])},{np.mean(in_all_sink_given_not_source[t, :])},{np.std(in_all_sink_given_not_source[t, :])},{np.mean(in_correlation[t, :])},{np.std(in_correlation[t, :])}\n")
        f.close()

    with open(filename_excit, "a") as f:
        for t in range(0, t_shift):
            f.write(f"{t},{np.mean(ex_all_abs_ate[t, :])},{np.std(ex_all_abs_ate[t, :])},{sem(ex_all_abs_ate[t, :])},{np.mean(ex_all_rel_ate[t, :])},{np.std(ex_all_rel_ate[t, :])},{sem(ex_all_rel_ate[t,:])},{np.mean(ex_all_p_source[t, :])},{np.std(ex_all_p_source[t, :])},{np.mean(ex_all_p_sink[t, :])},{np.std(ex_all_p_sink[t, :])},{np.mean(ex_all_p_sink_given_source[t, :])},{np.std(ex_all_p_sink_given_source[t, :])},{np.mean(ex_all_sink_given_not_source[t, :])},{np.std(ex_all_sink_given_not_source[t, :])},{np.mean(ex_correlation[t, :])},{np.std(ex_correlation[t, :])}\n")
        f.close()


# for i in tqdm(range(3, 15)):
#     ate_type(i, 30, "simplex")






def ate_type_change(dim, t_shift, change_dict, change: str):

    inhib, excit = count_type(dim, "simplex")   #Differentiate between the networks where the source neuron is excitatory and inhibitory

    filename_inhib = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}]_{change}_{change_dict['percentage']}_inhib.csv"
    with open(filename_inhib, "w") as f:
        f.write("t_shift,absolute_ATE,absolute_ATE_std,absolute_ATE_se,relative_ATE,relative_ATE_std,relative_ATE_se,p_source,p_source_std,p_sink,p_sink_std,p_sink_given_source,p_sink_given_source_std,p_sink_given_not_source,p_sink_given_not_source_std,correlation,correlation_std\n")
    
    filename_excit = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}]_{change}_{change_dict['percentage']}_excit.csv"
    with open(filename_excit, "w") as f:
        f.write("t_shift,absolute_ATE,absolute_ATE_std,absolute_ATE_se,relative_ATE,relative_ATE_std,relative_ATE_se,p_source,p_source_std,p_sink,p_sink_std,p_sink_given_source,p_sink_given_source_std,p_sink_given_not_source,p_sink_given_not_source_std,correlation,correlation_std\n")

    in_all_rel_ate = np.zeros((t_shift, inhib))
    in_all_abs_ate = np.zeros((t_shift, inhib))
    in_all_p_source = np.zeros((t_shift, inhib))
    in_all_p_sink = np.zeros((t_shift, inhib))
    in_all_p_sink_given_source = np.zeros((t_shift, inhib))
    in_all_sink_given_not_source = np.zeros((t_shift, inhib))
    in_correlation = np.zeros((t_shift, inhib))

    ex_all_rel_ate = np.zeros((t_shift, excit))
    ex_all_abs_ate = np.zeros((t_shift, excit))
    ex_all_p_source = np.zeros((t_shift, excit))
    ex_all_p_sink = np.zeros((t_shift, excit))
    ex_all_p_sink_given_source = np.zeros((t_shift, excit))
    ex_all_sink_given_not_source = np.zeros((t_shift, excit))
    ex_correlation = np.zeros((t_shift, excit))

    inhib = 0
    excit = 0

    for network in tqdm(range(0, 200)):
        path = f"../data/simplex/cluster_sizes_[{dim}]_n_steps_200000/{change}_{change_dict[str(dim)]}/{network}.pkl"
        X, W0 = get_data(path)

        source = X[0, :]
        sink = X[-1, :]

        summing = torch.sum(W0[0, :])

        # print(f"Number of inhibitory networks: {inhib}")
        # print(f"Number of excitatory networks: {excit}")

        for t in range(0, t_shift):
            p_source = np.sum(source)/len(source)
            p_sink = np.sum(np.roll(sink, -t))/len(sink)
            p_source_and_sink = (np.sum(source*np.roll(sink, -t))/len(sink))
            p_sink_given_source = p_source_and_sink/p_source
            p_sink_given_not_source = (p_sink - p_source_and_sink)/(1 - p_source)
            ATE_abs = p_sink_given_source - p_sink_given_not_source
            ATE_rel = ATE_abs/p_sink_given_not_source   #The relative percentage-wise increase in the probability of the sink neuron firing given that the source neuron fired

            if summing < 0:
                in_all_rel_ate[t, inhib] = ATE_rel
                in_all_abs_ate[t, inhib] = ATE_abs
                in_all_p_source[t, inhib] = p_source
                in_all_p_sink[t, inhib] = p_sink
                in_all_p_sink_given_source[t, inhib] = p_sink_given_source
                in_all_sink_given_not_source[t, inhib] = p_sink_given_not_source
                in_correlation[t, inhib] = pg.corr(source, np.roll(sink, -t))['r'].values[0]
            
            else:
                ex_all_rel_ate[t, excit] = ATE_rel
                ex_all_abs_ate[t, excit] = ATE_abs
                ex_all_p_source[t, excit] = p_source
                ex_all_p_sink[t, excit] = p_sink
                ex_all_p_sink_given_source[t, excit] = p_sink_given_source
                ex_all_sink_given_not_source[t, excit] = p_sink_given_not_source
                ex_correlation[t, excit] = pg.corr(source, np.roll(sink, -t))['r'].values[0]

        if summing < 0:
            inhib += 1
        else:
            excit += 1

    with open(filename_inhib, "a") as f:
        for t in range(0, t_shift):
            f.write(f"{t},{np.mean(in_all_abs_ate[t, :])},{np.std(in_all_abs_ate[t, :])},{sem(in_all_abs_ate[t, :])},{np.mean(in_all_rel_ate[t, :])},{np.std(in_all_rel_ate[t, :])},{sem(in_all_rel_ate[t, :])},{np.mean(in_all_p_source[t, :])},{np.std(in_all_p_source[t, :])},{np.mean(in_all_p_sink[t, :])},{np.std(in_all_p_sink[t, :])},{np.mean(in_all_p_sink_given_source[t, :])},{np.std(in_all_p_sink_given_source[t, :])},{np.mean(in_all_sink_given_not_source[t, :])},{np.std(in_all_sink_given_not_source[t, :])},{np.mean(in_correlation[t, :])},{np.std(in_correlation[t, :])}\n")
        f.close()

    with open(filename_excit, "a") as f:
        for t in range(0, t_shift):
            f.write(f"{t},{np.mean(ex_all_abs_ate[t, :])},{np.std(ex_all_abs_ate[t, :])},{sem(ex_all_abs_ate[t, :])},{np.mean(ex_all_rel_ate[t, :])},{np.std(ex_all_rel_ate[t, :])},{sem(ex_all_rel_ate[t, :])},{np.mean(ex_all_p_source[t, :])},{np.std(ex_all_p_source[t, :])},{np.mean(ex_all_p_sink[t, :])},{np.std(ex_all_p_sink[t, :])},{np.mean(ex_all_p_sink_given_source[t, :])},{np.std(ex_all_p_sink_given_source[t, :])},{np.mean(ex_all_sink_given_not_source[t, :])},{np.std(ex_all_sink_given_not_source[t, :])},{np.mean(ex_correlation[t, :])},{np.std(ex_correlation[t, :])}\n")
        f.close()




change_10 = {'4': '1',
                '5': '1',
                '6': '2',
                '7': '2',
                '8': '3',
                '9': '4',
                '10': '5',
                'percentage': '10'}


change_15 = {'4': '1',
                '5': '2',
                '6': '2',
                '7': '3',
                '8': '4',
                '9': '5',
                '10': '7',
                'percentage': '15'}



for i in range(4, 11):
    ate_type_change(i, 30, change_10, "removed")    

for i in range(4, 11):
    ate_type_change(i, 30, change_15, "removed")    

for i in range(4, 11):
    ate_type_change(i, 30, change_10, "added")    

for i in range(4, 11):
    ate_type_change(i, 30, change_15, "added")    