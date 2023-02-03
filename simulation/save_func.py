from pathlib import Path
from scipy.sparse import coo_matrix
import numpy as np
import torch
import pickle



def save(x, model, w0_data, seed, previous_batches, data_path, w0_generator, stimulation=[]):
    """Saves the spikes and the connectivity filter to a file"""
    x = x.cpu()
    xs = torch.split(x, [network.num_nodes for network in w0_data], dim=0)

    count = previous_batches

    

    for (x, network) in (zip(xs, w0_data)):
        sparse_x = coo_matrix(x)

        # print(network.Hasse_diagram)
        # print(network.W0_mat)
        # print(network.edge_index)
        # print(network.global_simplex_count)
        # exit()

        PATH = data_path/Path(f"{count}.pkl")

        dictionary = {
            "X_sparse" : sparse_x,
            "W0" : network.W0_mat,
            "edge_index" : network.edge_index,
            "W0_hubs" : network.W0_hubs, 
            "edge_index_hubs" : network.edge_index_hubs,
            "global_simplex_count" : network.global_simplex_count, 
            "neuron_simplex_count" : network.neuron_simplex_count, 
            "Hasse_diagram" : network.Hasse_diagram,
            "W0_parameters" : w0_generator.parameters,
            "stimulation" : stimulation,
            "parameters" : model.save_params(),
            "seed" : seed}

        with open(PATH, "wb") as f:
            pickle.dump(dictionary, f)

        # with open(PATH, "rb") as f:
        #     stuff = pickle.load(f)   #stuff = dictionary

        # np.savez_compressed(
        #     data_path/Path(f"{count}.npz"),
        #     X_sparse=sparse_x,
        #     W0 = network.W0_mat,
        #     edge_index = network.edge_index,
        #     W0_hubs = network.W0_hubs, 
        #     edge_index_hubs = network.edge_index_hubs,
        #     global_simplex_count = network.global_simplex_count, 
        #     #neuron_simplex_count = network.neuron_simplex_count, 
        #     Hasse_diagram = network.Hasse_diagram,
        #     W0_parameters = w0_generator.parameters,
        #     stimulation=stimulation,
        #     parameters=model.save_params(),
        #     seed=seed,
        # )

        count+=1

        