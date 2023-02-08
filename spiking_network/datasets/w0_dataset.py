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
            n_neurons = W0_mat.shape[0]
            edge_index = W0_mat.nonzero().t()
            W0_sparse = W0_mat[edge_index[0], edge_index[1]]
            W0_hubs = tuple[1]
            edge_index_hubs = tuple[2]
            
            dataset.data.append(Data(W0_sparse = W0_sparse, edge_index=edge_index, W0_mat = W0_mat, W0_hubs = W0_hubs, edge_index_hubs = edge_index_hubs, num_nodes=n_neurons))
        

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




