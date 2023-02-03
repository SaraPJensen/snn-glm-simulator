import torch
import numpy as np
from torch_geometric.data import Data
from spiking_network.datasets.w0_generator import DistributionParams

from my_flagser.torch_flagser import flagser_count_unweighted

class ConnectivityDataset:
    @classmethod
    def from_list(cls, w0_list):   
        #w0_list is now a list of tuples, where the tuples are (W0_mat, W0_hubs, edge_index_hubs)
        dataset = cls()
        dataset.data = []
        for tuple in w0_list:
            W0_mat = tuple[0]

            Hasse_simplex = flagser_count_unweighted(W0_mat)   #The Hasse_simplex object contains the Hasse diagram and the number of simplices in each level

            global_simplex_count = Hasse_simplex.simplex_counter()  #The global simplex count is the number of simplices in the entire graph
            Hasse_diagram = Hasse_simplex.levels_id   #nested list where the outer list contains all the simplices of that dimension. The inner list contains all the simplex-objects at that level, which in turn contains the information about the constituent nodes
            neuron_simplex_count = []  #The neuron simplex count is a list of all the neurons, where each entry is the number of simplices of each dimension each neuron forms a part of and its role in those simplices (source, mediator, sink)

            for neuron in Hasse_simplex.level_0:
                neuron_simplex_count.append(neuron.simplex_count)

            n_neurons = W0_mat.shape[0]
            edge_index = W0_mat.nonzero().t()
            W0_sparse = W0_mat[edge_index[0], edge_index[1]]
            W0_hubs = tuple[1]
            edge_index_hubs = tuple[2]
            dataset.data.append(Data(W0_sparse = W0_sparse, edge_index=edge_index, W0_mat = W0_mat, W0_hubs = W0_hubs, edge_index_hubs = edge_index_hubs, num_nodes=n_neurons, global_simplex_count = global_simplex_count, neuron_simplex_count = neuron_simplex_count, Hasse_diagram = Hasse_diagram))
        
        return dataset


    @property
    def w0_list(self):
        return [data.W0 for data in self.data]

    @property
    def edge_index_list(self):
        return [data.edge_index for data in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]




