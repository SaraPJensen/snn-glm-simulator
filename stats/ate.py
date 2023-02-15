import numpy as np
import pingouin as pg
import pandas as pd
import pickle
from scipy.sparse import coo_matrix
import torch
import sys
sys.path.append("./..")
from tqdm import tqdm


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


def ate(dim, t_shift):

    filename = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}].csv"
    with open(filename, "w") as f:
        f.write("t_shift,absolute_ATE,relative_ATE,p_sink,p_sink_given_source,p_sink_given_not_source,correlation\n")

    all_rel_ate = np.zeros((t_shift, 200))
    all_abs_ate = np.zeros((t_shift, 200))

    all_p_sink = np.zeros((t_shift, 200))
    all_p_sink_given_source = np.zeros((t_shift, 200))
    all_sink_given_not_source = np.zeros((t_shift, 200))

    correlation = np.zeros((t_shift, 200))

    for network in tqdm(range(0, 200)):
        path = f"../data/simplex/cluster_sizes_[{dim}]_n_steps_200000/{network}.pkl"
        X, W0 = get_data(path)

        source = X[0, :]
        sink = X[-1, :]

        for t in range(0, t_shift):
            p_source = np.sum(source)/len(source)
            p_sink = np.sum(np.roll(sink, -t))/len(sink)
            p_source_and_sink = (np.sum(source*np.roll(sink, -t))/len(sink))
            p_sink_given_source = p_source_and_sink/p_source
            p_sink_given_not_source = (p_sink - p_source_and_sink)/(1 - p_source)

            correlation[t, network] = pg.corr(source, np.roll(sink, -t))['r'].values[0]
            
            ATE_abs = p_sink_given_source - p_sink_given_not_source
            ATE_rel = ATE_abs/p_sink_given_not_source   #The relative percentage-wise increase in the probability of the sink neuron firing given that the source neuron fired

            all_rel_ate[t, network] = ATE_rel
            all_abs_ate[t, network] = ATE_abs

            all_p_sink[t, network] = p_sink
            all_p_sink_given_source[t, network] = p_sink_given_source
            all_sink_given_not_source[t, network] = p_sink_given_not_source

    with open(filename, "a") as f:
        for t in range(0, t_shift):
            f.write(f"{t},{np.mean(all_abs_ate[t, :])},{np.mean(all_rel_ate[t, :])},{np.mean(all_p_sink[t, :])},{np.mean(all_p_sink_given_source[t, :])},{np.mean(all_sink_given_not_source[t, :])},{np.mean(correlation[t, :])}\n")

        f.close()




        
def ate_removed(dim, t_shift, removed_dict):

    filename = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}]_removed_{removed_dict['percentage']}.csv"
    with open(filename, "w") as f:
        f.write("t_shift,absolute_ATE,relative_ATE,p_sink,p_sink_given_source,p_sink_given_not_source,correlation\n")

    all_rel_ate = np.zeros((t_shift, 200))
    all_abs_ate = np.zeros((t_shift, 200))

    all_p_sink = np.zeros((t_shift, 200))
    all_p_sink_given_source = np.zeros((t_shift, 200))
    all_sink_given_not_source = np.zeros((t_shift, 200))

    correlation = np.zeros((t_shift, 200))

    for network in tqdm(range(0, 200)):
        path = f"../data/simplex/cluster_sizes_[{dim}]_n_steps_200000/removed_{removed_dict[str(dim)]}/{network}.pkl"
        X, W0 = get_data(path)

        source = X[0, :]
        sink = X[-1, :]

        for t in range(0, t_shift):
            p_source = np.sum(source)/len(source)
            p_sink = np.sum(np.roll(sink, -t))/len(sink)
            p_source_and_sink = (np.sum(source*np.roll(sink, -t))/len(sink))
            p_sink_given_source = p_source_and_sink/p_source
            p_sink_given_not_source = (p_sink - p_source_and_sink)/(1 - p_source)

            correlation[t, network] = pg.corr(source, np.roll(sink, -t))['r'].values[0]
            
            ATE_abs = p_sink_given_source - p_sink_given_not_source
            ATE_rel = ATE_abs/p_sink_given_not_source   #The relative percentage-wise increase in the probability of the sink neuron firing given that the source neuron fired

            all_rel_ate[t, network] = ATE_rel
            all_abs_ate[t, network] = ATE_abs

            all_p_sink[t, network] = p_sink
            all_p_sink_given_source[t, network] = p_sink_given_source
            all_sink_given_not_source[t, network] = p_sink_given_not_source

    with open(filename, "a") as f:
        for t in range(0, t_shift):
            f.write(f"{t},{np.mean(all_abs_ate[t, :])},{np.mean(all_rel_ate[t, :])},{np.mean(all_p_sink[t, :])},{np.mean(all_p_sink_given_source[t, :])},{np.mean(all_sink_given_not_source[t, :])},{np.mean(correlation[t, :])}\n")

        f.close()




# for i in range(3, 15):
#     ate(i, 50)


removed_15 = {'4': '1',
                '5': '2',
                '6': '2',
                '7': '3',
                '8': '4',
                '9': '5',
                '10': '7',
                'percentage': '15'}

removed_20 = {'4': '1',
                '5': '2',
                '6': '3',
                '7': '4',
                '8': '6',
                '9': '7',
                '10': '9',
                'percentage': '20'}



for i in range(4, 11):
    ate_removed(i, 30, removed_20)    