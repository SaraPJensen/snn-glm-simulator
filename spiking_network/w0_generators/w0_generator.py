import torch
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
class NormalParams(DistributionParams):
    mean: float = 0.0
    std: float = 5.0
    name: str = "normal"

@dataclass
class GlorotParams(DistributionParams):
    mean: float = 0.0
    std: float = 5.0
    name: str = "glorot"

@dataclass
class SmallWorldParams(DistributionParams):
    mean: float = 0.0
    std: float = 5.0
    name: str = "small_world"

@dataclass
class BarabasiParams(DistributionParams):
    mean: float = 0.0
    std: float = 5.0
    name: str = "barabasi"

@dataclass
class CavemanParams(DistributionParams):
    name: str = "caveman"


class W0Generator:
    def __init__(self, cluster_sizes, random_cluster_connections, dist_params: DistributionParams):
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

        else: 
            W0_mat, edge_index_hubs, low_dim_edges = W0Generator._build_connected_clusters(self.cluster_sizes, self.random_cluster_connections, self.dist_params, rng)
            W0_hubs = np.zeros((self.n_clusters, self.n_clusters))
            W0_hubs[low_dim_edges[0], low_dim_edges[1]] = W0_mat[edge_index_hubs[0], edge_index_hubs[1]]   #This seems to be correct 

        W0_mat = W0Generator._insert_values(self.cluster_sizes, W0_mat, self.dist_params.mean, self.dist_params.std)
        W0_mat = torch.from_numpy(W0_mat)
        W0_mat = W0_mat.fill_diagonal_(0)

        #Is it really necessary to return in low_dim_edges, the information about it is to be found in W0_hubs
        #edge_index_hubs gives the precise identity of the hub nodes
        #return W0Generator._to_tensor(W0, edge_index)   #Decide whether the full or only the sparse W0 should be returned
        return W0_mat, W0_hubs, edge_index_hubs #, low_dim_edges


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

            if random_connections == True:
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
            upper = nx.to_numpy_array(nx.watts_strogatz_graph(cluster_size, k = cluster_size//5, p = 0.3))  #This is the upper triangular part of the matrix
            lower = nx.to_numpy_array(nx.watts_strogatz_graph(cluster_size, k = cluster_size//5, p = 0.3))  #This is the lower triangular part of the matrix

        elif dist_params.name == 'barabasi':  #Binary connectivity
            upper = nx.to_numpy_array(nx.barabasi_albert_graph(cluster_size, m = cluster_size//5))  #This is the upper triangular part of the matrix
            lower = nx.to_numpy_array(nx.barabasi_albert_graph(cluster_size, m = cluster_size//5))  #This is the lower triangular part of the matrix

        elif dist_params.name == "caveman":   #As of now, not very useful...
            W0_graph = nx.connected_caveman_graph(3, 10)
            W0_graph = W0_graph.to_directed()
            ranking = nx.voterank(W0_graph)

        out = np.zeros((cluster_size, cluster_size))
        out[np.triu_indices(cluster_size)] = upper[np.triu_indices(cluster_size)]
        out[np.tril_indices(cluster_size)] = lower[np.tril_indices(cluster_size)]
        W0_graph = nx.from_numpy_array(out, create_using=nx.DiGraph)
        ranking = nx.voterank(W0_graph)

        return W0_graph, ranking
    
    @staticmethod
    def _dales_law(W0: torch.Tensor) -> torch.Tensor:
        """Applies Dale's law to the connectivity matrix W0"""
        W0 = torch.concat((W0 * (W0 > 0), W0 * (W0 < 0)), 0)
        return W0

    @staticmethod
    def _insert_values(cluster_sizes: list, W0: torch.Tensor, mean: int, std: int)  -> torch.Tensor:
        """Inserts values from a normal distribution into to the binary connectivity matrix"""
        """Make it random what rows are positive and negative. Also, the ratio should be 80/20, with more excitatory neurons"""
        count = 0

        for size in cluster_sizes: 
            inhib = int(0.2 * size)  #number of inhibitory rows
            inhib_rows = np.random.randint(count, count+size, inhib)  #randomly select inhibitory rows

            pos_tensor = 0.2*np.abs(np.random.normal(mean, std, size = (size, sum(cluster_sizes))))    #Multiply by 0.2 to get correct scaling
            neg_tensor = -np.abs(np.random.normal(mean, std, size = (inhib, sum(cluster_sizes))))

            W0[count:count+size, :] = W0[count:count+size, :] * pos_tensor    #Insert positive values in the whole tensor
            W0[inhib_rows, :] = W0[inhib_rows, :] * neg_tensor    #Insert negative values in the inhibitory rows

            count += size

        return W0


    @staticmethod
    def _generate_normal_w0(cluster_size: int, mean: float, std: float, rng: torch.Generator) -> torch.Tensor:
        """Generates a normal n_neurons/2*n_neurons/2 matrix from a normal distribution"""
        half_cluster_size = cluster_size // 2
        W0 = torch.normal(mean, std, (half_cluster_size, cluster_size), generator=rng)
        return W0
    
    @staticmethod
    def _generate_glorot_w0(cluster_size: int, mean: float, std: float, rng: torch.Generator) -> torch.Tensor:
        """Generates a normal n_neurons/2*n_neurons/2 matrix from a normal distribution"""
        normal_W0 = W0Generator._generate_normal_w0(cluster_size, mean, std, rng)
        glorot_W0 = normal_W0 / torch.sqrt(torch.tensor(cluster_size, dtype=torch.float32))
        return glorot_W0

    @staticmethod
    def _to_tensor(W0, edge_index):
        """Converts the W0 and edge_index to a tensor"""
        W0 = torch.sparse_coo_tensor(edge_index, W0)
        return W0.to_dense()

