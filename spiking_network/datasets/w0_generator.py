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
    max: float = 10.0
    name: str = "small_world"


@dataclass
class SimplexParams(DistributionParams):
    min: float = 0.0
    max: float = 10.0
    name: str = "simplex"




class W0Generator:
    def __init__(self, cluster_sizes: list, random_cluster_connections: bool, dist_params: DistributionParams):
        self.n_clusters = len(cluster_sizes)
        self.cluster_sizes = cluster_sizes
        self.random_cluster_connections = random_cluster_connections
        self.dist_params = dist_params

    @property
    def parameters(self):
        return {
                "n_clusters": self.n_clusters,
                "cluster_sizes": self.cluster_sizes,
                "random_cluster": self.random_cluster_connections,
                "dist_params": self.dist_params,
            }

    def generate(self, seed):   #make_dataset calls on this 
        rng = torch.Generator().manual_seed(seed)
        W0_hubs, edge_index_hubs, low_dim_edges = [], [], []

        if self.n_clusters == 1:
            W0_graph, _ = W0Generator._generate_w0(self.cluster_sizes[0], self.dist_params, rng)
            W0_mat = nx.to_numpy_array(W0_graph)
            

        if self.dist_params.name == "simplex":
            W0_mat, edge_index_hubs, low_dim_edges = W0Generator._simplex_generator(self.cluster_sizes, self.dist_params, rng)
            W0_mat = W0_mat.numpy()
            W0_hubs = np.zeros((self.n_clusters, self.n_clusters))
            W0_hubs[low_dim_edges[0], low_dim_edges[1]] = W0_mat[edge_index_hubs[0], edge_index_hubs[1]]   #This seems to be correct 


        else: 
            W0_mat, edge_index_hubs, low_dim_edges = W0Generator._build_connected_clusters(self.cluster_sizes, self.random_cluster_connections, self.dist_params, rng)
            W0_hubs = np.zeros((self.n_clusters, self.n_clusters))
            W0_hubs[low_dim_edges[0], low_dim_edges[1]] = W0_mat[edge_index_hubs[0], edge_index_hubs[1]]   #This seems to be correct 


        W0_mat = W0Generator._insert_values(self.cluster_sizes, W0_mat, self.dist_params.min, self.dist_params.max, seed)
        W0_mat = torch.from_numpy(W0_mat)
        W0_mat = W0_mat.fill_diagonal_(0)

        #edge_index_hubs gives the precise identity of the hub nodes
        #return W0Generator._to_tensor(W0, edge_index)   #Decide whether the full or only the sparse W0 should be returned
        return W0_mat, W0_hubs, edge_index_hubs 


    def generate_list(self, n_sims, seed):   #This just returns a list of tuples of the form (W0_mat, W0_hubs, edge_index_hubs)
        #return [self.generate(seed + i)[0] for i in range(n_sims)]
        return [self.generate(seed + i) for i in range(n_sims)]



    @staticmethod
    def _build_connected_clusters(cluster_sizes: list, random_cluster_connections: bool, dist_params, rng):
        W0_mat, hub_neurons = W0Generator._build_clusters(cluster_sizes, dist_params, rng)   #W0 is now the connectivity matrix for all the unconnected clusters
        W0_mat, edge_index_hub, low_dim_edges = W0Generator._connect_hub_neurons(W0_mat, hub_neurons, random_cluster_connections, dist_params)

        return W0_mat, edge_index_hub, low_dim_edges


    @staticmethod
    def _build_clusters(cluster_sizes: list, dist_params: DistributionParams, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Builds the connectivity matrices for each cluster W and the edge_index matrix, also returns the hub neurons"""    
        W0s_mat = []
        neuron_count = 0
        hub_neurons = np.empty((0, 2), int)   #Create an empty array to store the hub neurons))

        for i in range(len(cluster_sizes)): # Builds the internal structure of each cluster
            rng.manual_seed(rng.seed()+1)
            W0_graph, ranking = W0Generator._generate_w0(cluster_sizes[i], dist_params, rng)   #W0 is a graph object
            W0s_mat.append(nx.to_numpy_array(W0_graph))
            hub_neurons = np.append(hub_neurons, np.array([[i, ranking[0] + neuron_count]]), axis=0)  #need to take into account that there can be several hub neurons per cluster XXX
            neuron_count += cluster_sizes[i]

        W0_mat = np.zeros((sum(cluster_sizes), sum(cluster_sizes)))   #This is the connectivity matrix for all the unconnected clusters

        for row in range(len(cluster_sizes)): 
            start = sum(cluster_sizes[:row])
            end = sum(cluster_sizes[:row+1])
            W0_mat[start:end, start:end] = W0s_mat[row]
        
        return W0_mat, hub_neurons 
    

    @staticmethod
    def _connect_hub_neurons(W0_mat: int, hub_neurons: list[int], random_connections: bool, dist_params) -> tuple[torch.Tensor, torch.Tensor]:
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


    @staticmethod
    def _generate_w0(cluster_size: int, dist_params: DistributionParams, rng: torch.Generator) -> torch.Tensor:
        """Generates a normally-drawn connectivity matrix W0 that follows Dale's law and has zeros on the diagonal"""
        ranking = []

        if dist_params.name == 'small_world':   
            upper = nx.to_numpy_array(nx.watts_strogatz_graph(cluster_size, k = cluster_size//3, p = 0.3, seed = rng.seed()))  #This is the upper triangular part of the matrix
            lower = nx.to_numpy_array(nx.watts_strogatz_graph(cluster_size, k = cluster_size//3, p = 0.3, seed = rng.seed()))  #This is the lower triangular part of the matrix


        out = np.zeros((cluster_size, cluster_size))
        out[np.triu_indices(cluster_size)] = upper[np.triu_indices(cluster_size)]
        out[np.tril_indices(cluster_size)] = lower[np.tril_indices(cluster_size)]
        W0_graph = nx.from_numpy_array(out, create_using=nx.DiGraph)
        ranking = nx.voterank(W0_graph)

        return W0_graph, ranking



    @staticmethod
    def _simplex_generator(cluster_sizes: list, dist_params: DistributionParams, rng: torch.Generator) -> torch.Tensor:
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
            
            # print(W0)
            # print()
            # print("Count ", count)

        if len(cluster_sizes) > 1:   #Always cyclic for simplicity
            for i in range(len(cluster_sizes) - 1):
                low_dim_edges = torch.cat((low_dim_edges, torch.tensor([i, i + 1], dtype=torch.long).unsqueeze(1)), dim=1)   #this contains the information about the location of the inter-cluster connections

            low_dim_edges = torch.cat((low_dim_edges, torch.tensor([len(cluster_sizes) -1, 0], dtype=torch.long).unsqueeze(1)), dim=1)

        # print(edge_index_hubs)
        # print(low_dim_edges)
        # exit()
        return W0, edge_index_hubs, low_dim_edges
    

    @staticmethod
    def _dales_law(W0: torch.Tensor) -> torch.Tensor:
        """Applies Dale's law to the connectivity matrix W0"""
        W0 = torch.concat((W0 * (W0 > 0), W0 * (W0 < 0)), 0)
        return W0

    
    @staticmethod
    def _insert_values(cluster_sizes: list, W0: torch.Tensor, min: int, max: int, rng: torch.Generator)  -> torch.Tensor:
        """Inserts values from a normal distribution into to the binary connectivity matrix"""
        """Make it random what rows are positive and negative. Also, the ratio should be 80/20, with more excitatory neurons"""
        count = 0
        #np.random.seed(rng.seed())

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



