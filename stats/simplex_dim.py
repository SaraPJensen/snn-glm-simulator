import pickle
import networkx as nx
import numpy as np
import pingouin as pg

def simplex_dim(neurons):

    filename = f"../data/stats/sm_rm_count_{neurons}_neurons.csv"

    with open(filename, "w") as f:
        f.write("small_world_avg_global_count,random_avg_global_count,small_world_avg_connectivity,random_avg_connectivity\n")

    sm_avg_global_count = np.zeros((10, 200))
    rm_avg_global_count = np.zeros((10, 200))

    sm_avg_connectivity = np.zeros((1, 200))
    rm_avg_connectivity = np.zeros((1, 200))

    for n in range(200):
        sm_file = f"../data/small_world/cluster_sizes_[{neurons}]_n_steps_200000/{n}.pkl"
        rm_file = f"../data/random/cluster_sizes_[{neurons}]_n_steps_200000/{n}.pkl"

        with open(sm_file, "rb") as f:
            sm_data = pickle.load(f)
        
        with open(rm_file, "rb") as f:
            rm_data = pickle.load(f)

        sm_global = sm_data['global_simplex_count']
        rm_global = rm_data['global_simplex_count']

        for i in range(len(sm_global)):
            sm_avg_global_count[i, n] = sm_global[i]
            
        for i in range(len(rm_global)):
            rm_avg_global_count[i, n] = rm_global[i]

        sm_w0 = sm_data['W0'].numpy()
        rm_w0 = rm_data['W0'].numpy()

        sm_graph = nx.from_numpy_matrix(sm_w0)
        rm_graph = nx.from_numpy_matrix(rm_w0)

        sm_avg_connectivity[0, n] = nx.average_clustering(sm_graph)
        rm_avg_connectivity[0, n] = nx.average_clustering(rm_graph)

    sm_avg_global_count = np.mean(sm_avg_global_count, axis=1)
    rm_avg_global_count = np.mean(rm_avg_global_count, axis=1)

    sm_avg_connectivity = np.mean(sm_avg_connectivity)
    rm_avg_connectivity = np.mean(rm_avg_connectivity)

    with open(filename, "a") as f:
        for i in range(sm_avg_global_count.shape[0]):
            f.write(f"{sm_avg_global_count[i]},{rm_avg_global_count[i]},{sm_avg_connectivity},{rm_avg_connectivity}\n")

dim = [10, 15, 20, 25, 30, 40, 50]


for neurons in dim:
    simplex_dim(neurons)


