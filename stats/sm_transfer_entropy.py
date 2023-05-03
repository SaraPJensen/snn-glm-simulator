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

    W0 = stuff["W0"].numpy()

    neuron_simplex_count = stuff["neuron_simplex_count"]

    # print(neuron_simplex_count[0])
    # print(neuron_simplex_count[0].shape)
    # print(len(neuron_simplex_count[0]))


    return X, W0, neuron_simplex_count  




def small_world_tranfer_entropy(neurons):

    for network in tqdm(range(0, 200)):
        path = f"../data/small_world/cluster_sizes_[{neurons}]_n_steps_200000/{network}.pkl"
        X, W0, neuron_simplex_count = get_data(path)

        network_graph = nx.from_numpy_matrix(W0, create_using=nx.DiGraph)
        

        for neuron in range(0, neurons):
            path = Path(f"../data/small_world/stats/transfer_entropy/{neurons}_neurons/network_{network}")
            path.mkdir(parents= True, exist_ok=True)
            summary_filename = f"../data/small_world/stats/transfer_entropy/{neurons}_neurons/network_{network}/neuron_{neuron}_node_degree.csv"

            in_degree = network_graph.in_degree(neuron)
            out_degree = network_graph.out_degree(neuron)
            simplex_count = neuron_simplex_count[neuron]

            with open(summary_filename, "w") as f:
                f.write("dimension,source,mediator,sink,in_degree,out_degree\n")

                for dim in range(len(simplex_count)):
                    #print(dim)
                    f.write(f"{dim+1},{simplex_count[dim][0]},{simplex_count[dim][1]},{simplex_count[dim][2]},{in_degree},{out_degree}\n")

        for time in tqdm(range(0, 20)):

            transfer_filename = f"../data/small_world/stats/transfer_entropy/{neurons}_neurons/network_{network}/transfer_entropy_timeshift_{time}.csv"
            transfer = np.zeros((neurons, neurons))

            for source in range(0, neurons):
                for sink in range(0, neurons):
                    if source != sink:
                        transfer[source, sink] = pi.transfer_entropy(X[source, :], np.roll(X[sink, :], -time), k = 20)

            df = pd.DataFrame(transfer, columns = np.arange(0, neurons), index = np.arange(0, neurons))
            df.to_csv(transfer_filename, index = True, header = True)


#sm_sizes = [10, 15, 20, 25, 30, 40, 50, 60, 70]

sm_sizes = [25, 30, 40, 50, 60, 70]

# for size in sm_sizes:
#     small_world_tranfer_entropy(size)



def te_sm_analyse(neurons, timeshift, max_simplex_dim):
    data_file = f"../data/small_world/stats/transfer_entropy/{neurons}_neurons/summary.csv"

    df = pd.read_csv(data_file)

    te_out = df[f"te_out_shift_{timeshift}"].values
    te_in = df[f"te_in_shift_{timeshift}"].values

    in_degree = df["in_degree"].values
    out_degree = df["out_degree"].values

    sink_2 = df["2_sink"]
    source_2 = df["2_source"]

    sink_3 = df["3_sink"]
    source_3 = df["3_source"]

    sink_4 = df["4_sink"]
    source_4 = df["4_source"]

    te_in_4_sink_corr = pingouin.corr(te_in, sink_4)
    te_out_4_source_corr = pingouin.corr(te_out, source_4)

    te_in_3_sink_corr = pingouin.corr(te_in, sink_3)
    te_out_3_source_corr = pingouin.corr(te_out, source_3)

    te_in_2_sink_corr = pingouin.corr(te_in, sink_2)
    te_out_2_source_corr = pingouin.corr(te_out, source_2)

    te_in_degree = pingouin.corr(te_in, in_degree)
    te_out_degree = pingouin.corr(te_out, out_degree)

    print()
    print("Time shift: ", timeshift)
    print("Max simplex dim: ", max_simplex_dim)
    print("Neurons: ", neurons)
    print()

    print("Transfer entropy in vs. in-degree")
    print(te_in_degree)

    print()
    print()

    print("Transfer entropy out vs. out-degree")
    print(te_out_degree)

    print()
    print()

    print("Transfer entropy in vs. 2-sink")
    print(te_in_2_sink_corr)

    print()
    print()

    print("Transfer entropy out vs. 2-source")
    print(te_out_2_source_corr)

    print()
    print()

    print("Transfer entropy in vs. 3-sink")
    print(te_in_3_sink_corr)

    print()
    print()

    print("Transfer entropy out vs. 3-source")
    print(te_out_3_source_corr)

    print()
    print()

    print("Transfer entropy in vs. 4-sink")
    print(te_in_4_sink_corr)

    print()
    print()

    print("Transfer entropy out vs. 4-source")
    print(te_out_4_source_corr)

    print()
    print()

    

te_sm_analyse(50, 0, 4)
