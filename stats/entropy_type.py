import pyinform as pi
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import coo_matrix
import torch
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





def transfer_entropy_dim(min_neurons, max_neurons, network_type):


    filename_inhib = f"../data/{network_type}/stats/transfer_entropy_inhib.csv"
    with open(filename_inhib, "w") as f:
        f.write("neurons,transfer_entropy,std,se\n")    
    
    filename_excit = f"../data/{network_type}/stats/transfer_entropy_excit.csv"
    with open(filename_excit, "w") as f:
        f.write("neurons,transfer_entropy,std,se\n")

    neurons = max_neurons - min_neurons

    avg_entropy_inhib = np.zeros((neurons))
    avg_entropy_excit = np.zeros((neurons))

    std_entropy_inhib = np.zeros((neurons))
    std_entropy_excit = np.zeros((neurons))

    se_entropy_inhib = np.zeros((neurons))
    se_entropy_excit = np.zeros((neurons))

    for idx, neurons in tqdm(enumerate(range(min_neurons, max_neurons))):

        inhib, excit = count_type(neurons, network_type)   #Differentiate between the networks where the source neuron is excitatory and inhibitory

        entropy_inhib = np.zeros((inhib))
        entropy_excit = np.zeros((excit))

        inhib_idx = 0
        excit_idx = 0

        for network in range(0, 200):
            path = f"../data/{network_type}/cluster_sizes_[{neurons}]_n_steps_200000/{network}.pkl"
            X, W0 = get_data(path)

            source = X[0, :]
            sink = X[-1, :]

            summing = torch.sum(W0[0, :])

            transfer_entropy = pi.transferentropy.transfer_entropy(source, sink, k = 20)

            if summing < 0:
                entropy_inhib[inhib_idx] = transfer_entropy
            else:
                entropy_excit[excit_idx] = transfer_entropy

            if summing < 0:
                inhib_idx += 1
            else:
                excit_idx += 1
        
        avg_entropy_inhib[idx] = np.mean(entropy_inhib)
        std_entropy_inhib[idx] = np.std(entropy_inhib)
        se_entropy_inhib[idx] = sem(entropy_inhib.ravel()) #, None, ddof=0)

        avg_entropy_excit[idx] = np.mean(entropy_excit)
        std_entropy_excit[idx] = np.std(entropy_excit)
        se_entropy_excit[idx] = sem(entropy_excit.ravel())

    for idx, neurons in enumerate(range(min_neurons, max_neurons)):
        with open(filename_inhib, "a") as f:
            f.write(f"{neurons},{avg_entropy_inhib[idx]},{std_entropy_inhib[idx]},{se_entropy_inhib[idx]}\n")

        with open(filename_excit, "a") as f:
            f.write(f"{neurons},{avg_entropy_excit[idx]},{std_entropy_excit[idx]},{se_entropy_excit[idx]}\n")


transfer_entropy_dim(3, 10, "simplex")

