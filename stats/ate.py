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
        f.write("t_shift,absolute_ATE,absolute_ATE_std,relative_ATE,relative_ATE_std,p_source,p_source_std,p_sink,p_sink_std,p_sink_given_source,p_sink_given_source_std,p_sink_given_not_source,p_sink_given_not_source_std,correlation,correlation_std\n")

    all_rel_ate = np.zeros((t_shift, 200))
    all_abs_ate = np.zeros((t_shift, 200))

    all_p_source = np.zeros((t_shift, 200))
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

            all_p_source[t, network] = p_source
            all_p_sink[t, network] = p_sink
            all_p_sink_given_source[t, network] = p_sink_given_source
            all_sink_given_not_source[t, network] = p_sink_given_not_source

    with open(filename, "a") as f:
        for t in range(0, t_shift):
            f.write(f"{t},{np.mean(all_abs_ate[t, :])},{np.std(all_abs_ate[t, :])},{np.mean(all_rel_ate[t, :])},{np.std(all_rel_ate[t, :])},{np.mean(all_p_source[t, :])},{np.std(all_p_source[t, :])},{np.mean(all_p_sink[t, :])},{np.std(all_p_sink[t, :])},{np.mean(all_p_sink_given_source[t, :])},{np.std(all_p_sink_given_source[t, :])},{np.mean(all_sink_given_not_source[t, :])},{np.std(all_sink_given_not_source[t, :])},{np.mean(correlation[t, :])},{np.std(correlation[t, :])}\n")

        f.close()




        
def ate_change(dim, t_shift, change_dict, change: str):

    # change = "removed" or "added"

    filename = f"../data/simplex/stats/ATE/cluster_sizes_[{dim}]_{change}_{change_dict['percentage']}.csv"
    with open(filename, "w") as f:
        f.write("t_shift,absolute_ATE,absolute_ATE_std,relative_ATE,relative_ATE_std,p_source,p_source_std,p_sink,p_sink_std,p_sink_given_source,p_sink_given_source_std,p_sink_given_not_source,p_sink_given_not_source_std,correlation,correlation_std\n")

    all_rel_ate = np.zeros((t_shift, 200))
    all_abs_ate = np.zeros((t_shift, 200))

    all_p_source = np.zeros((t_shift, 200))
    all_p_sink = np.zeros((t_shift, 200))
    all_p_sink_given_source = np.zeros((t_shift, 200))
    all_sink_given_not_source = np.zeros((t_shift, 200))

    correlation = np.zeros((t_shift, 200))

    for network in tqdm(range(0, 200)):
        path = f"../data/simplex/cluster_sizes_[{dim}]_n_steps_200000/{change}_{change_dict[str(dim)]}/{network}.pkl"
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

            all_p_source[t, network] = p_source
            all_p_sink[t, network] = p_sink
            all_p_sink_given_source[t, network] = p_sink_given_source
            all_sink_given_not_source[t, network] = p_sink_given_not_source

    with open(filename, "a") as f:
        for t in range(0, t_shift):
            f.write(f"{t},{np.mean(all_abs_ate[t, :])},{np.std(all_abs_ate[t, :])},{np.mean(all_rel_ate[t, :])},{np.std(all_rel_ate[t, :])},{np.mean(all_p_source[t, :])},{np.std(all_p_source[t, :])},{np.mean(all_p_sink[t, :])},{np.std(all_p_sink[t, :])},{np.mean(all_p_sink_given_source[t, :])},{np.std(all_p_sink_given_source[t, :])},{np.mean(all_sink_given_not_source[t, :])},{np.std(all_sink_given_not_source[t, :])},{np.mean(correlation[t, :])},{np.std(correlation[t, :])}\n")  

        f.close()




def ate_second_last(dim, t_shift):
    filename = f"../data/simplex/stats/ATE/second_last_cluster_sizes_[{dim}].csv"
    with open(filename, "w") as f:
        f.write("t_shift,absolute_ATE,absolute_ATE_std,relative_ATE,relative_ATE_std,p_second_last,p_second_last_std,p_sink,p_sink_std,p_sink_given_second_last,p_sink_given_second_last_std,p_sink_given_not_second_last,p_sink_given_not_second_last_std,correlation,correlation_std\n")

    all_rel_ate = np.zeros((t_shift, 200))
    all_abs_ate = np.zeros((t_shift, 200))

    all_p_second_last = np.zeros((t_shift, 200))
    all_p_sink = np.zeros((t_shift, 200))
    all_p_sink_given_second_last = np.zeros((t_shift, 200))
    all_sink_given_not_second_last = np.zeros((t_shift, 200))

    correlation = np.zeros((t_shift, 200))

    for network in tqdm(range(0, 200)):
        path = f"../data/simplex/cluster_sizes_[{dim}]_n_steps_200000/{network}.pkl"
        X, W0 = get_data(path)

        second_last = X[-2, :]
        sink = X[-1, :]

        for t in range(0, t_shift):
            p_second_last = np.sum(second_last)/len(second_last)
            p_sink = np.sum(np.roll(sink, -t))/len(sink)
            p_second_last_and_sink = (np.sum(second_last*np.roll(sink, -t))/len(sink))
            p_sink_given_second_last = p_second_last_and_sink/p_second_last
            p_sink_given_not_second_last = (p_sink - p_second_last_and_sink)/(1 - p_second_last)

            correlation[t, network] = pg.corr(second_last, np.roll(sink, -t))['r'].values[0]
            
            ATE_abs = p_sink_given_second_last - p_sink_given_not_second_last
            ATE_rel = ATE_abs/p_sink_given_not_second_last   #The relative percentage-wise increase in the probability of the sink neuron firing given that the second_last neuron fired

            all_rel_ate[t, network] = ATE_rel
            all_abs_ate[t, network] = ATE_abs

            all_p_second_last[t, network] = p_second_last
            all_p_sink[t, network] = p_sink
            all_p_sink_given_second_last[t, network] = p_sink_given_second_last
            all_sink_given_not_second_last[t, network] = p_sink_given_not_second_last

    with open(filename, "a") as f:
        for t in range(0, t_shift):
            f.write(f"{t},{np.mean(all_abs_ate[t, :])},{np.std(all_abs_ate[t, :])},{np.mean(all_rel_ate[t, :])},{np.std(all_rel_ate[t, :])},{np.mean(all_p_second_last[t, :])},{np.std(all_p_second_last[t, :])},{np.mean(all_p_sink[t, :])},{np.std(all_p_sink[t, :])},{np.mean(all_p_sink_given_second_last[t, :])},{np.std(all_p_sink_given_second_last[t, :])},{np.mean(all_sink_given_not_second_last[t, :])},{np.std(all_sink_given_not_second_last[t, :])},{np.mean(correlation[t, :])},{np.std(correlation[t, :])}\n")

        f.close()




# for i in tqdm(range(3, 15)):
#      ate(i, 30)


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

change_20 = {'4': '1',
                '5': '2',
                '6': '3',
                '7': '4',
                '8': '6',
                '9': '7',
                '10': '9',
                'percentage': '20'}



for i in range(4, 11):
    ate_change(i, 30, change_10, "removed")    

for i in range(4, 11):
    ate_change(i, 30, change_15, "removed")    

for i in range(4, 11):
    ate_change(i, 30, change_10, "added")    

for i in range(4, 11):
    ate_change(i, 30, change_10, "added")    