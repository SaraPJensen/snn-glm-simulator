import pyinform as pi
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import coo_matrix
import torch
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




def entropy(min_neurons, max_neurons, k = 20):

    neurons = max_neurons - min_neurons 
    filename = f"../data/simplex/stats/entropy_k{k}.csv"

    with open(filename, "w") as f:
        f.write("neurons,source_active_info,source_active_std,source_bloch_entropy,source_bloch_std,sink_active_info,sink_active_std,sink_bloch_entropy,sink_bloch_std\n")

    for neurons in tqdm(range(min_neurons, max_neurons)):

        source_active_per_neurons = np.zeros((200))
        source_bloch_per_neurons = np.zeros((200))

        sink_active_per_neurons = np.zeros((200))
        sink_bloch_per_neurons = np.zeros((200))

        for network in tqdm(range(0, 200)):
            path = f"../data/simplex/cluster_sizes_[{neurons}]_n_steps_200000/{network}.pkl"
            X, W0 = get_data(path)

            source_active_per_neurons[network] = pi.activeinfo.active_info(X[0, :], k = k)   #k is the history length
            source_bloch_per_neurons[network] = pi.blockentropy.block_entropy(X[0, :], k = k)
            
            sink_active_per_neurons[network] = pi.activeinfo.active_info(X[-1, :], k = k)   #k is the history length
            sink_bloch_per_neurons[network] = pi.blockentropy.block_entropy(X[-1, :], k = k)

        source_mean_active_info = np.mean(source_active_per_neurons)
        source_std_active_info = np.std(source_active_per_neurons)
        source_mean_bloch = np.mean(source_bloch_per_neurons)
        source_std_bloch = np.std(source_bloch_per_neurons)

        sink_mean_active_info = np.mean(sink_active_per_neurons)
        sink_std_active_info = np.std(sink_active_per_neurons)
        sink_mean_bloch = np.mean(sink_bloch_per_neurons)
        sink_std_bloch = np.std(sink_bloch_per_neurons)

        with open(filename, "a") as f:
            f.write(f"{neurons},{source_mean_active_info},{source_std_active_info},{source_mean_bloch},{source_std_bloch},{sink_mean_active_info},{sink_std_active_info},{sink_mean_bloch},{sink_std_bloch}\n")



entropy(3, 15, 20)



def conditional_entropy(neurons, t_shift):
    filename = f"../data/simplex/stats/conditional_entropy_{neurons}.csv"

    with open(filename, "w") as f:
        f.write("t_shift,conditional_entropy,std\n")

    entropy = np.zeros((t_shift, 200))

    for network in range(0, 200):
        path = f"../data/simplex/cluster_sizes_[{neurons}]_n_steps_200000/{network}.pkl"
        X, W0 = get_data(path)

        source = X[0, :]
        sink = X[-1, :]

        for t in range(t_shift):
            conditional_entropy = pi.conditionalentropy.conditional_entropy(source, np.roll(sink, -t))
            entropy[t, network] = conditional_entropy

    
    for t in range(t_shift):
        mean_entropy = np.mean(entropy[t, :])
        std_entropy = np.std(entropy[t, :])

        with open(filename, "a") as f:
            f.write(f"{t},{mean_entropy},{std_entropy}\n")


#for neurons in range(3, 15):
    #conditional_entropy(neurons, 15)



def cond_entropy_dim(min_neurons, max_neurons):
    filename = f"../data/simplex/stats/conditional_entropy.csv"

    with open(filename, "w") as f:
        f.write("neurons,conditional_entropy,std\n")

    neuronss = max_neurons - min_neurons

    entropy = np.zeros((neuronss, 200))

    for idx, neurons in enumerate(range(min_neurons, max_neurons)):

        for network in range(0, 200):
            path = f"../data/simplex/cluster_sizes_[{neurons}]_n_steps_200000/{network}.pkl"
            X, W0 = get_data(path)

            source = X[0, :]
            sink = X[-1, :]

            conditional_entropy = pi.conditionalentropy.conditional_entropy(source, sink)
            entropy[idx, network] = conditional_entropy

    for idx, neurons in enumerate(range(min_neurons, max_neurons)):
        mean_entropy = np.mean(entropy[idx, :])
        std_entropy = np.std(entropy[idx, :])

        with open(filename, "a") as f:
            f.write(f"{neurons},{mean_entropy},{std_entropy}\n")
        
        idx += 1

#cond_entropy_dim(3, 15)



ordered = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

ordered = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0])

ordered = np.random.randint(0, 1, (1000))

ordered = np.ones((1000))

ordered = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0])

#ordered = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])


# for k in range(1, 10):
#     print(k, pi.activeinfo.active_info(ordered, k))

# print()

# for k in range(1, 10):
#     print(k, pi.blockentropy.block_entropy(ordered, k))



def transfer_entropy_dim(min_neurons, max_neurons):
    filename = f"../data/simplex/stats/transfer_entropy.csv"

    with open(filename, "w") as f:
        f.write("neurons,transfer_entropy,std\n")

    neuronss = max_neurons - min_neurons

    entropy = np.zeros((neuronss, 200))

    for idx, neurons in enumerate(range(min_neurons, max_neurons)):

        for network in range(0, 200):
            path = f"../data/simplex/cluster_sizes_[{neurons}]_n_steps_200000/{network}.pkl"
            X, W0 = get_data(path)

            source = X[0, :]
            sink = X[-1, :]

            transfer_entropy = pi.transferentropy.transfer_entropy(source, sink, k = 20)
            entropy[idx, network] = transfer_entropy

    for idx, neurons in enumerate(range(min_neurons, max_neurons)):
        mean_entropy = np.mean(entropy[idx, :])
        std_entropy = np.std(entropy[idx, :])

        with open(filename, "a") as f:
            f.write(f"{neurons},{mean_entropy},{std_entropy}\n")
        




def transfer_entropy_time(neurons, time_shift):
    filename = f"../data/simplex/stats/transfer_entropy_{neurons}_neurons.csv"

    with open(filename, "w") as f:
        f.write("time_shift,transfer_entropy,std\n")

    entropy = np.zeros((time_shift, 200))

    for network in tqdm(range(0, 200)):
        path = f"../data/simplex/cluster_sizes_[{neurons}]_n_steps_200000/{network}.pkl"
        X, W0 = get_data(path)

        source = X[0, :]
        sink = X[-1, :]

        for t in range(time_shift):
            transfer_entropy = pi.transferentropy.transfer_entropy(source, np.roll(sink, -t), k = 20)
            entropy[t, network] = transfer_entropy

    for t in range(time_shift):
        mean_entropy = np.mean(entropy[t, :])
        std_entropy = np.std(entropy[t, :])

        with open(filename, "a") as f:
            f.write(f"{t},{mean_entropy},{std_entropy}\n")




#transfer_entropy_dim(3, 15)


# neurons = 10
# time_shift = 20

# transfer_entropy_time(neurons, time_shift)

