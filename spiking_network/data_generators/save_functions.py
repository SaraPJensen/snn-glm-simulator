from pathlib import Path
from scipy.sparse import coo_matrix
import numpy as np
import torch



def save(x, model, w0_data, seed, previous_batches, data_path, w0_generator, stimulation=[]):
    """Saves the spikes and the connectivity filter to a file"""
    x = x.cpu()
    xs = torch.split(x, [network.num_nodes for network in w0_data], dim=0)

    count = previous_batches

    for (x, network) in (zip(xs, w0_data)):
        sparse_x = coo_matrix(x)

        np.savez_compressed(
            data_path/Path(f"{count}.npz"),
            X_sparse=sparse_x,
            W0 = network.W0_mat,
            edge_index = network.edge_index,
            W0_hubs = network.W0_hubs, 
            edge_index_hubs = network.edge_index_hubs,
            W0_parameters = w0_generator.parameters,
            stimulation=stimulation,
            parameters=model.save_parameters(),
            seed=seed,
        )

        count+=1


