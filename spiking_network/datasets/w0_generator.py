import torch
from torch_geometric.data import Data, InMemoryDataset
import numpy as np
from dataclasses import dataclass
import networkx as nx
import numpy as np
import seaborn
import matplotlib.pyplot as plt

@dataclass
class DistributionParams:
    """Class for storing distribution parameters."""
    def _to_dict(self):
        return self.__dict__

@dataclass
class SmallWorldParams(DistributionParams):
    min: float = 0.0
    max: float = 20.0
    name: str = "small_world"


@dataclass
class SimplexParams(DistributionParams):
    min: float = 0.0
    max: float = 20.0
    name: str = "simplex"

@dataclass
class RandomParams(DistributionParams):
    min: float = 0.0
    max: float = 20.0
    name: str = "random"

@dataclass
class SM_RemoveParams(DistributionParams):
    min: float = 0.0
    max: float = 20.0
    name: str = "sm_remove"



class W0Generator:
    def __init__(self, cluster_sizes: list, random_cluster_connections: bool, dist_params: DistributionParams, remove_connections: int):
        self.n_clusters = len(cluster_sizes)
        self.cluster_sizes = cluster_sizes
        self.random_cluster_connections = random_cluster_connections
        self.dist_params = dist_params
        self.remove_connections = remove_connections

    @property
    def parameters(self):
        return {
                "n_clusters": self.n_clusters,
                "cluster_sizes": self.cluster_sizes,
                "random_cluster": self.random_cluster_connections,
                "remove_connections": self.remove_connections,
                "dist_params": {"min": self.dist_params.min, 
                                "max": self.dist_params.max, 
                                "name": self.dist_params.name
                }
            }

    def generate(self, seed):   #make_dataset calls on this 
        W0_hubs, edge_index_hubs, low_dim_edges = [], [], []

        if self.n_clusters == 1:
            W0_graph, _ = self.generate_w0(self.cluster_sizes[0], self.dist_params, seed)
            W0_mat = nx.to_numpy_array(W0_graph)
            

        elif self.dist_params.name == "simplex":
            print("Generating simplex graph")
            W0_mat, edge_index_hubs, low_dim_edges = self.simplex_generator(self.cluster_sizes, self.dist_params, seed)
            W0_mat = W0_mat.numpy()
            W0_hubs = np.zeros((self.n_clusters, self.n_clusters))
            W0_hubs[low_dim_edges[0], low_dim_edges[1]] = W0_mat[edge_index_hubs[0], edge_index_hubs[1]]   #This seems to be correct 
                

        #Do this for random, small world and sm_remove
        else:    
            W0_mat, edge_index_hubs, low_dim_edges = self.build_connected_clusters(self.cluster_sizes, self.random_cluster_connections, self.dist_params, seed)
            W0_hubs = np.zeros((self.n_clusters, self.n_clusters))
            W0_hubs[low_dim_edges[0], low_dim_edges[1]] = W0_mat[edge_index_hubs[0], edge_index_hubs[1]]   #This seems to be correct 


        W0_mat = self.insert_values(self.cluster_sizes, W0_mat, self.dist_params.min, self.dist_params.max, seed)
        W0_mat = torch.from_numpy(W0_mat)
        W0_mat = W0_mat.fill_diagonal_(0)

        #edge_index_hubs gives the precise identity of the hub nodes
        return W0_mat, W0_hubs, edge_index_hubs 


    def generate_list(self, n_sims, seed):   #This just returns a list of tuples of the form (W0_mat, W0_hubs, edge_index_hubs)

        #Special case where the weight matrix is the same for all simulations
        if self.dist_params.name == 'sm_remove':  

            return_list = []
            W0_hubs, edge_index_hubs, low_dim_edges = [], [], []

            if self.n_clusters == 1:
                W0_graph, _ = self.generate_w0(self.cluster_sizes[0], self.dist_params, seed)
                W0_mat = nx.to_numpy_array(W0_graph)

            else: 
                W0_mat, edge_index_hubs, low_dim_edges = self.build_connected_clusters(self.cluster_sizes, self.random_cluster_connections, self.dist_params, seed)
                W0_hubs = np.zeros((self.n_clusters, self.n_clusters))
                W0_hubs[low_dim_edges[0], low_dim_edges[1]] = W0_mat[edge_index_hubs[0], edge_index_hubs[1]]  
                
            W0_mat = self.insert_values(self.cluster_sizes, W0_mat, self.dist_params.min, self.dist_params.max, seed)  #Use the same W0 with the same values for all of the simulations
            W0_mat = torch.from_numpy(W0_mat)
            W0_mat = W0_mat.fill_diagonal_(0)

            for i in range(0, n_sims-1):
                current_w0 = W0_mat.clone()
                current_w0[i,:] = 0
                current_w0[:,i] = 0
                return_list.append((current_w0, W0_hubs, edge_index_hubs))

            return_list.append((W0_mat, W0_hubs, edge_index_hubs))
            
            return return_list

        else:
            return [self.generate(seed + i) for i in range(n_sims)]



    
    def build_connected_clusters(self, cluster_sizes: list, random_cluster_connections: bool, dist_params, seed):
        W0_mat, hub_neurons = self.build_clusters(cluster_sizes, dist_params, seed)   #W0 is now the connectivity matrix for all the unconnected clusters
        W0_mat, edge_index_hub, low_dim_edges = self.connect_hub_neurons(W0_mat, hub_neurons, random_cluster_connections, dist_params)

        return W0_mat, edge_index_hub, low_dim_edges


    
    def build_clusters(self, cluster_sizes: list, dist_params: DistributionParams, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Builds the connectivity matrices for each cluster W and the edge_index matrix, also returns the hub neurons"""    
        W0s_mat = []
        neuron_count = 0
        hub_neurons = np.empty((0, 2), int)   #Create an empty array to store the hub neurons))

        for i in range(len(cluster_sizes)): # Builds the internal structure of each cluster
            W0_graph, ranking = self.generate_w0(cluster_sizes[i], dist_params, seed)   #W0 is a graph object
            W0s_mat.append(nx.to_numpy_array(W0_graph))
            hub_neurons = np.append(hub_neurons, np.array([[i, ranking[0] + neuron_count]]), axis=0)  #need to take into account that there can be several hub neurons per cluster XXX
            neuron_count += cluster_sizes[i]

        W0_mat = np.zeros((sum(cluster_sizes), sum(cluster_sizes)))   #This is the connectivity matrix for all the unconnected clusters

        for row in range(len(cluster_sizes)): 
            start = sum(cluster_sizes[:row])
            end = sum(cluster_sizes[:row+1])
            W0_mat[start:end, start:end] = W0s_mat[row]
        
        return W0_mat, hub_neurons 
    

    
    def connect_hub_neurons(self, W0_mat: int, hub_neurons: list[int], random_connections: bool, dist_params) -> tuple[torch.Tensor, torch.Tensor]:
        """For each hubneuron, connects it to a randomly selected hubneuron in another cluster"""
        W0_graph = nx.from_numpy_array(W0_mat, create_using=nx.DiGraph)
        edge_index_hubs = torch.tensor([], dtype=torch.long)
        low_dim_edges = torch.tensor([], dtype=torch.long)

        for sender_cluster, sender_node in hub_neurons:
            available_neurons = []
            for i in range(len(hub_neurons)):
                if hub_neurons[i][0] != sender_cluster:
                    available_neurons.append(hub_neurons[i])

            if random_connections == True and len(available_neurons) > 1:
                connections = np.random.randint(1, len(available_neurons))
            else: 
                connections = 1

            for i in range(connections):
                choice = torch.randint(len(available_neurons), (1,))[0]
                receiver_cluster, receiver_node = available_neurons[choice]
                del available_neurons[choice]
                W0_graph.add_edge(sender_node, receiver_node)
                new_edge = torch.tensor([sender_node, receiver_node], dtype=torch.long).unsqueeze(1)  #This is just for storing in edge_index_hubs
                new_edge_low = torch.tensor([sender_cluster, receiver_cluster], dtype=torch.long).unsqueeze(1)
                edge_index_hubs = torch.cat((edge_index_hubs, new_edge), dim=1)   #this contains the information about the location of the inter-cluster connections
                low_dim_edges = torch.cat((low_dim_edges, new_edge_low), dim=1)   #this contains the information about the location of the inter-cluster connections

        W0_mat = nx.to_numpy_array(W0_graph)

        return W0_mat, edge_index_hubs, low_dim_edges


    
    def generate_w0(self, cluster_size: int, dist_params: DistributionParams, seed: int) -> torch.Tensor:
        """Generates a normally-drawn connectivity matrix W0 that follows Dale's law and has zeros on the diagonal"""
        ranking = []

        if dist_params.name == 'small_world' or dist_params.name == 'sm_remove':   
            k = int(cluster_size/4)

            if k < 2:
                k = 2

            upper = nx.to_numpy_array(nx.watts_strogatz_graph(cluster_size, k = k, p = 0.3, seed = seed))  #This is the upper triangular part of the matrix
            lower = nx.to_numpy_array(nx.watts_strogatz_graph(cluster_size, k = k, p = 0.3, seed = seed))  #This is the lower triangular part of the matrix
            # upper = nx.to_numpy_array(nx.barabasi_albert_graph(cluster_size, m = k, seed = seed))  #This is the upper triangular part of the matrix
            # lower = nx.to_numpy_array(nx.barabasi_albert_graph(cluster_size, m = k, seed = seed))  #This is the lower triangular part of the matrix
            
            out = np.zeros((cluster_size, cluster_size))
            out[np.triu_indices(cluster_size)] = upper[np.triu_indices(cluster_size)]
            out[np.tril_indices(cluster_size)] = lower[np.tril_indices(cluster_size)]
            
        elif dist_params.name == 'random':
            upper = nx.to_numpy_array(nx.erdos_renyi_graph(cluster_size, p = 0.21, seed = seed, directed = True))  #This is the upper triangular part of the matrix
            lower = nx.to_numpy_array(nx.erdos_renyi_graph(cluster_size, p = 0.21, seed = seed, directed = True))  #This is the lower triangular part of the matrix
            out = np.zeros((cluster_size, cluster_size))
            out[np.triu_indices(cluster_size)] = upper[np.triu_indices(cluster_size)]
            out[np.tril_indices(cluster_size)] = lower[np.tril_indices(cluster_size)]

        elif dist_params.name == 'simplex':
            out = torch.zeros((cluster_size, cluster_size))
            out[0:cluster_size, 0:cluster_size] = torch.triu(torch.ones((cluster_size, cluster_size)), diagonal = 1)
            out = out.numpy()

            #Remove connections
            if self.remove_connections > 0:
                indices = torch.triu_indices(cluster_size, cluster_size, offset=1)
                remove = np.random.choice(np.arange(0, indices.shape[1]), self.remove_connections, replace=False)
                
                for i in range(len(remove)):
                    out[indices[0, remove[i]], indices[1, remove[i]]] = 0
                    

        W0_graph = nx.from_numpy_array(out, create_using=nx.DiGraph)
        ranking = nx.voterank(W0_graph)

        return W0_graph, ranking



    
    def simplex_generator(self, cluster_sizes: list, dist_params: DistributionParams, seed: int) -> torch.Tensor:
        """Generates a simplex connectivity matrix W0"""
        edge_index_hubs = torch.tensor([], dtype=torch.long)
        low_dim_edges = torch.tensor([], dtype=torch.long)

        count = 0
        W0 = torch.zeros((sum(cluster_sizes), sum(cluster_sizes)))

        for size in cluster_sizes:
            W0[count:count+size, count:count+size] = torch.triu(torch.ones((size, size)), diagonal = 1)
            count += size

            if count < sum(cluster_sizes):   #Connect the sink to the next source
                W0[count - 1, count] = 1.0
                edge_index_hubs = torch.cat((edge_index_hubs, torch.tensor([count -1, count], dtype=torch.long).unsqueeze(1)), dim=1)   #this contains the information about the location of the inter-cluster connections
            
            else:    #Connect the last sink to the first source
                W0[count-1, 0] = 1.0
                edge_index_hubs = torch.cat((edge_index_hubs, torch.tensor([count-1, 0], dtype=torch.long).unsqueeze(1)), dim=1)   #this contains the information about the location of the inter-cluster connections
            

        if len(cluster_sizes) > 1:   #Always cyclic for simplicity
            for i in range(len(cluster_sizes) - 1):
                low_dim_edges = torch.cat((low_dim_edges, torch.tensor([i, i + 1], dtype=torch.long).unsqueeze(1)), dim=1)   #this contains the information about the location of the inter-cluster connections

            low_dim_edges = torch.cat((low_dim_edges, torch.tensor([len(cluster_sizes) -1, 0], dtype=torch.long).unsqueeze(1)), dim=1)

        return W0, edge_index_hubs, low_dim_edges
    

    
    
    def insert_values(self, cluster_sizes: list, W0: torch.Tensor, min: int, max: int, seed: int)  -> torch.Tensor:
        """Inserts values from a normal distribution into to the binary connectivity matrix"""
        """Make it random what rows are positive and negative. Also, the ratio should be 80/20, with more excitatory neurons"""
        count = 0
        np.random.seed(seed)

        for size in cluster_sizes: 
            inhib = int(0.5 * size)  #number of inhibitory rows
            inhib_rows = np.random.randint(count, count+size, inhib)  #randomly select inhibitory rows

            #Use uniformly distributed values to prevent network from exploding, but scale by dividing by the square root of the cluster size
            pos_tensor = np.abs(np.random.uniform(0, 6, size = (size, sum(cluster_sizes)))/(np.sqrt(size)))   
            #pos_tensor = np.abs(np.random.normal(0, 5, size = (size, sum(cluster_sizes)))/(0.5*np.sqrt(size)))  

            W0[count:count+size, :] = W0[count:count+size, :] * pos_tensor    #Insert positive values in the whole tensor
            W0[inhib_rows, :] = -1*W0[inhib_rows, :]    #Insert negative values in the inhibitory rows
            #W0[inhib_rows, :] = W0[inhib_rows, :] * neg_tensor    #Insert negative values in the inhibitory rows

            count += size

        return W0



