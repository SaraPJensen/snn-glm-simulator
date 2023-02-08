from pathlib import Path
from scipy.sparse import coo_matrix
import numpy as np
import torch
import pickle

from my_flagser.torch_flagser import flagser_count_unweighted




def save(x, model, w0_data, seed, previous_batches, data_path, w0_generator, stimulation=[]):
    """Saves the spikes and the connectivity filter to a file"""
    x = x.cpu()

    # for network in w0_data:
    #     print(network.num_nodes)
    #     print(network.W0_mat.shape)

    xs = torch.split(x, [network.num_nodes for network in w0_data], dim=0)    #This is where it crashes XXX 

    count = previous_batches

    

    for (x, network) in (zip(xs, w0_data)):
        sparse_x = coo_matrix(x)

        # print(network.Hasse_diagram)
        # print(network.W0_mat)
        # print(network.edge_index)
        # print(network.global_simplex_count)
        # exit()

        W0_mat = network.W0_mat

        Hasse_simplex = flagser_count_unweighted(W0_mat)   #The Hasse_simplex object contains the Hasse diagram and the number of simplices in each level

        global_simplex_count = Hasse_simplex.simplex_counter()  #The global simplex count is the number of simplices in the entire graph
        Hasse_diagram = Hasse_simplex.levels_id   #nested list where the outer list contains all the simplices of that dimension. The inner list contains all the simplex-objects at that level, which in turn contains the information about the constituent nodes
        neuron_simplex_count = []  #The neuron simplex count is a list of all the neurons, where each entry is the number of simplices of each dimension each neuron forms a part of and its role in those simplices (source, mediator, sink)

        for neuron in Hasse_simplex.level_0:
            neuron_simplex_count.append(neuron.simplex_count)

        PATH = data_path/Path(f"{count}.pkl")

        dictionary = {
            "X_sparse" : sparse_x,
            "W0" : W0_mat,
            "edge_index" : network.edge_index,
            "W0_hubs" : network.W0_hubs, 
            "edge_index_hubs" : network.edge_index_hubs,
            "global_simplex_count" : global_simplex_count, 
            "neuron_simplex_count" : neuron_simplex_count, 
            "Hasse_diagram" : Hasse_diagram,
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

        