import pyinform as pi
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import coo_matrix
import torch
from tqdm import tqdm
from tqdm.contrib import tzip
import pingouin



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



def entropy(min_neurons, max_neurons, network_type, k = 20):

    neurons = max_neurons - min_neurons 
    filename = f"../data/{network_type}/stats/entropy_k{k}.csv"

    with open(filename, "w") as f:
        f.write("neurons,source_active_info,source_active_std,source_block_entropy,source_block_std,sink_active_info,sink_active_std,sink_block_entropy,sink_block_std\n")

    for neurons in tqdm(range(min_neurons, max_neurons)):

        source_active_per_neurons = np.zeros((200))
        source_block_per_neurons = np.zeros((200))

        sink_active_per_neurons = np.zeros((200))
        sink_block_per_neurons = np.zeros((200))

        for network in tqdm(range(0, 200)):
            path = f"../data/{network_type}/cluster_sizes_[{neurons}]_n_steps_200000/{network}.pkl"
            X, W0 = get_data(path)

            source_active_per_neurons[network] = pi.activeinfo.active_info(X[0, :], k = k)   #k is the history length
            source_block_per_neurons[network] = pi.blockentropy.block_entropy(X[0, :], k = k)
            
            sink_active_per_neurons[network] = pi.activeinfo.active_info(X[-1, :], k = k)   #k is the history length
            sink_block_per_neurons[network] = pi.blockentropy.block_entropy(X[-1, :], k = k)

        source_mean_active_info = np.mean(source_active_per_neurons)
        source_std_active_info = np.std(source_active_per_neurons)
        source_mean_block = np.mean(source_block_per_neurons)
        source_std_block = np.std(source_block_per_neurons)

        sink_mean_active_info = np.mean(sink_active_per_neurons)
        sink_std_active_info = np.std(sink_active_per_neurons)
        sink_mean_block = np.mean(sink_block_per_neurons)
        sink_std_block = np.std(sink_block_per_neurons)

        with open(filename, "a") as f:
            f.write(f"{neurons},{source_mean_active_info},{source_std_active_info},{source_mean_block},{source_std_block},{sink_mean_active_info},{sink_std_active_info},{sink_mean_block},{sink_std_block}\n")

        f.close()


#entropy(2, 15, "line", 20)


def entropy_added_removed(size, changed_neurons, change_type, k = 20):

    filename = f"../data/simplex/stats/entropy_k{k}_{change_type}_cluster_size_{size}.csv"

    with open(filename, "w") as f:
        f.write("change,source_active_info,source_active_std,source_block_entropy,source_block_std,sink_active_info,sink_active_std,sink_block_entropy,sink_block_std\n")

    for change_count in range(1, changed_neurons + 1):
        source_active_per_neurons = np.zeros((200))
        source_block_per_neurons = np.zeros((200))

        sink_active_per_neurons = np.zeros((200))
        sink_block_per_neurons = np.zeros((200))

        for network in tqdm(range(0, 200)):
            path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/{change_type}_{change_count}/{network}.pkl"
            X, W0 = get_data(path)

            source_active_per_neurons[network] = pi.activeinfo.active_info(X[0, :], k = k)   #k is the history length
            source_block_per_neurons[network] = pi.blockentropy.block_entropy(X[0, :], k = k)
            
            sink_active_per_neurons[network] = pi.activeinfo.active_info(X[-1, :], k = k)   #k is the history length
            sink_block_per_neurons[network] = pi.blockentropy.block_entropy(X[-1, :], k = k)

        source_mean_active_info = np.mean(source_active_per_neurons)
        source_std_active_info = np.std(source_active_per_neurons)
        source_mean_block = np.mean(source_block_per_neurons)
        source_std_block = np.std(source_block_per_neurons)

        sink_mean_active_info = np.mean(sink_active_per_neurons)
        sink_std_active_info = np.std(sink_active_per_neurons)
        sink_mean_block = np.mean(sink_block_per_neurons)
        sink_std_block = np.std(sink_block_per_neurons)

        with open(filename, "a") as f:
            f.write(f"{change_count},{source_mean_active_info},{source_std_active_info},{source_mean_block},{source_std_block},{sink_mean_active_info},{sink_std_active_info},{sink_mean_block},{sink_std_block}\n")
        f.close()



# entropy_added_removed(8, 5, "added", 20)
# entropy_added_removed(8, 5, "removed", 20)


def transfer_entropy_added_removed(size, changed_neurons, change_type, k = 20):

    filename = f"../data/simplex/stats/transfer_entropy/{change_type}_cluster_size_{size}.csv"

    with open(filename, "w") as f:
        f.write("change,transfer_entropy,std\n")


    for change_count in range(1, changed_neurons + 1):
        transfer_entropy = np.zeros((200))

        for network in tqdm(range(0, 200)):
            path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/{change_type}_{change_count}/{network}.pkl"
            X, W0 = get_data(path)

            source = X[0, :]
            sink = X[-1, :]

            transfer_entropy[network] = pi.transferentropy.transfer_entropy(source, sink, k = k)

        mean_entropy = np.mean(transfer_entropy)
        std_entropy = np.std(transfer_entropy)

        with open(filename, "a") as f:
            f.write(f"{change_count},{mean_entropy},{std_entropy}\n")
        f.close()


# transfer_entropy_added_removed(8, 5, "added", 20)
# transfer_entropy_added_removed(8, 5, "removed", 20)



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



# ordered = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

# ordered = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0])

# ordered = np.random.randint(0, 1, (1000))

# ordered = np.ones((1000))

# ordered = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0])

#ordered = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])


# for k in range(1, 10):
#     print(k, pi.activeinfo.active_info(ordered, k))

# print()

# for k in range(1, 10):
#     print(k, pi.blockentropy.block_entropy(ordered, k))



def transfer_entropy_dim(min_neurons, max_neurons, network_type):
    filename = f"../data/{network_type}/stats/transfer_entropy.csv"

    with open(filename, "w") as f:
        f.write("neurons,transfer_entropy,std\n")

    neurons = max_neurons - min_neurons

    entropy = np.zeros((neurons, 200))

    for idx, neurons in enumerate(range(min_neurons, max_neurons)):

        for network in range(0, 200):
            path = f"../data/{network_type}/cluster_sizes_[{neurons}]_n_steps_200000/{network}.pkl"
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
        




def transfer_entropy_time(neurons, time_shift, network_type):
    filename = f"../data/{network_type}/stats/transfer_entropy/{neurons}_neurons.csv"

    with open(filename, "w") as f:
        f.write("time_shift,transfer_entropy,std\n")

    entropy = np.zeros((time_shift, 200))

    for network in tqdm(range(0, 200)):
        path = f"../data/{network_type}/cluster_sizes_[{neurons}]_n_steps_200000/{network}.pkl"
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




#transfer_entropy_dim(2, 15, "line")


#time_shift = 15

# for neurons in tqdm(range(2, 10)):
#     transfer_entropy_time(neurons, time_shift, "line")

# for neurons in tqdm(range(3, 10)):
#     transfer_entropy_time(neurons, time_shift, "simplex")

# transfer_entropy_time(neurons, time_shift, "line")





def transfer_entropy_small_world(neurons, max_timeshift, max_simplex_dim = 5):
    base_path = f"../data/small_world/stats/transfer_entropy/{neurons}_neurons/"

    save_file = f"../data/small_world/stats/transfer_entropy/{neurons}_neurons/summary.csv"

    simplex_str = ""
    for size in range(1, max_simplex_dim + 1):
        simplex_str += f",{size}_source,{size}_mediator,{size}_sink"

    in_timeshift_str = ""
    out_timeshift_str = ""
    for timeshift in range(max_timeshift):
        in_timeshift_str += f",te_in_shift_{timeshift}"
        out_timeshift_str += f",te_out_shift_{timeshift}"

    with open(save_file, "w") as f:
        f.write(f"in_degree,out_degree{out_timeshift_str}{in_timeshift_str}{simplex_str}\n")

    for network in range(200):

        in_te = np.zeros((neurons, max_timeshift))
        out_te = np.zeros((neurons, max_timeshift))

        test_size =  f"{base_path}network_{network}/neuron_0_node_degree.csv"
        test_df = pd.read_csv(test_size)
        network_max_simplex = len(test_df["dimension"])   #In some networks, the max simplex dimension is smaller than the max simplex dimension of the network with the most neurons

        # print(network_max_simplex)
        # print()

        simplex_count = np.zeros((neurons, max_simplex_dim*3))

        for timeshift in range(max_timeshift):
            entropy_path = f"{base_path}network_{network}/transfer_entropy_timeshift_{timeshift}.csv"
            entropy_df = pd.read_csv(entropy_path).to_numpy()[:,1:]   #Ignore first column (neuron index)

            in_te[:, timeshift] = np.sum(entropy_df, axis = 0)
            out_te[:, timeshift] = np.sum(entropy_df, axis = 1)

        for neuron_idx in range(neurons):
            degree_path = f"{base_path}network_{network}/neuron_{neuron_idx}_node_degree.csv"
            degree_df = pd.read_csv(degree_path)

            in_degree = degree_df["in_degree"].values[0]
            out_degree = degree_df["out_degree"].values[0]

            # print("Network: ", network)
            # print("Neuron: ", neuron_idx)
            # print()

            
            for count in range(network_max_simplex):    
                simplex_count[neuron_idx][3*count] = degree_df["source"][count]
                simplex_count[neuron_idx][3*count+1] = degree_df["mediator"][count]
                simplex_count[neuron_idx][3*count+2] = degree_df["sink"][count]
            

            with open(save_file, "a") as f:
                f.write(f"{in_degree},{out_degree},{','.join(out_te[neuron_idx, :].astype(str))},{','.join(in_te[neuron_idx, :].astype(str))},{','.join(simplex_count[neuron_idx, :].astype(str))}\n")
                f.close()
        

# cluster_sizes = [10, 15, 20, 25, 30, 40, 50, 60, 70]
# max_simplex_size = [3, 4, 4, 4, 4, 4, 5, 5, 5]

# for size, max_size in tzip(cluster_sizes, max_simplex_size):
#     transfer_entropy_small_world(size, 10, max_size)




    
        



