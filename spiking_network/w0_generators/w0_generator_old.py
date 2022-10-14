import torch
from dataclasses import dataclass
import networkx as nx

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
    def __init__(self, n_clusters, cluster_size, n_cluster_connections, dist_params: DistributionParams):
        self.n_clusters = n_clusters
        self.cluster_size = cluster_size
        self.n_cluster_connections = n_cluster_connections
        self.dist_params = dist_params

    def generate(self, seed):   #make_dataset calls on this 
        rng = torch.Generator().manual_seed(seed)
        W0, edge_index, n_neurons_list, n_edges_list, hub_neurons = W0Generator._build_connected_clusters(self.n_clusters, self.cluster_size, self.n_cluster_connections, self.dist_params, rng)
        return W0Generator._to_tensor(W0, edge_index), n_neurons_list, n_edges_list, hub_neurons

    @staticmethod
    def generate_parallel(n_sims, n_neurons, dist_params, seed=None):
        rng = torch.Generator().manual_seed(seed) if seed else torch.Generator()
        W0, edge_index, n_neurons_list, n_edges_list = W0Generator._build_clusters_parallel(n_sims, n_neurons, dist_params, rng)
        return W0Generator._to_tensor(W0, edge_index), n_neurons_list, n_edges_list

    @staticmethod
    def generate_simple(n_neurons, dist_params, seed=None):
        rng = torch.Generator().manual_seed(seed) if seed else torch.Generator()
        W0, edge_index, n_neurons_list, n_edges_list = W0Generator._build_clusters_simple(1, n_neurons, dist_params, rng)
        return W0Generator._to_tensor(W0, edge_index)

    @staticmethod
    def _build_clusters_parallel(n_sims, n_neurons, dist_params, rng):
        return W0Generator._build_clusters(n_sims, n_neurons, dist_params, rng)

    @staticmethod
    def _build_connected_clusters(n_clusters, cluster_size, n_cluster_connections, dist_params, rng):
        W0, edge_index, n_neurons_list, n_edges_list, hub_neurons = W0Generator._build_clusters(n_clusters, cluster_size, dist_params, rng)
        
        if n_cluster_connections > 1:
            raise NotImplementedError("n_cluster_connections != 1 not implemented")
        elif n_cluster_connections == 1:
            # Identifies the hub neurons
            #hub_neurons = W0Generator._select_hub_neurons(n_clusters, cluster_size, rng)   #Change this function

            # Connects the hub neurons and adds them to the graph
            W0_hub, edge_index_hub = W0Generator._connect_hub_neurons(hub_neurons, cluster_size, dist_params)
            W0 = torch.cat((W0, W0_hub), dim=0)
            edge_index = torch.cat((edge_index, edge_index_hub), dim=1)
        else:
            hub_neurons = []

        print(W0_hub)

        return W0, edge_index, n_neurons_list, n_edges_list, hub_neurons

    @staticmethod
    def _build_clusters(n_clusters: int, cluster_size: int, dist_params: DistributionParams, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Builds the connectivity matrices for each cluster W and the edge_index matrix, also returns the hub neurons"""    
        if n_clusters == 1:
            W0, edge_index = W0Generator._build_cluster(cluster_size, dist_params, rng)
            return W0, edge_index, [], [], []

        W0s = []
        hub_neurons = []
        edge_indices = []
        n_neurons_list = []
        n_edges_list = []
        for i in range(n_clusters): # Builds the internal structure of each cluster, cluster_size could be a list of sizes to iterate over
            rng.manual_seed(rng.seed()+1)
            W0, edge_index, hub_node = W0Generator._build_cluster(cluster_size, dist_params, rng)
            W0s.append(W0)
            hub_neurons.append(hub_node + i*cluster_size)
            edge_indices.append(edge_index)
            n_neurons_list.append(cluster_size)   #Include this to facilitate extending to clusters of different sizes
            n_edges_list.append(edge_index.shape[1])

        W0 = torch.cat(W0s, dim=0) # Concatenates the W matrices of the clusters
        edge_index = torch.cat([edge_index + i*cluster_size for i, edge_index in enumerate(edge_indices)], dim=1) # Concatenates the edge_index matrices of the clusters
        
        return W0, edge_index, n_neurons_list, n_edges_list, hub_neurons


    # @staticmethod
    # def _select_hub_neurons(n_clusters: int, cluster_size: int, rng: torch.Generator) -> list[int]:
    #     """Chooses the hubneurons for the network"""
    #     hub_neurons = []
    #     for i in range(n_clusters):
    #         hub_neuron = torch.randint(0, cluster_size, (1,), dtype=torch.long, generator=rng) + i * cluster_size
    #         hub_neurons.append(hub_neuron)
    #     return hub_neurons
    

    @staticmethod
    def _connect_hub_neurons(hub_neurons: list[int], cluster_size, dist_params) -> tuple[torch.Tensor, torch.Tensor]:
        """For each hubneuron, connects it to a randomly selected hubneuron in another cluster"""
        edge_index = torch.tensor([], dtype=torch.long)
        W = torch.tensor([])
        for i in hub_neurons:
            available_neurons = [hub_neuron for hub_neuron in hub_neurons if hub_neuron != i]
            # an = []
            # for hub_neuron in hub_neurons:
            #     if hub_neuron != i:
            #         an.append(hub_neuron)
            j = available_neurons[torch.randint(len(available_neurons), (1,))[0]]
            new_edge = torch.tensor([i, j], dtype=torch.long).unsqueeze(1)
            chm_action = "excitatory" if (i % cluster_size) < (cluster_size // 2) else "inhibitory"  # Determines the type of connection based on which half of the cluster the hubneuron is in
            w = W0Generator._new_single_weight(chemical_action=chm_action, cluster_size=cluster_size, dist_params=dist_params)
            edge_index = torch.cat((edge_index, new_edge), dim=1)
            W = torch.cat((W, w), dim=0)
        return W, edge_index

    @staticmethod
    def _build_cluster(cluster_size: int, dist_params: DistributionParams, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """Builds the internal structure of a cluster"""
        W0, hub_node = W0Generator._generate_w0(cluster_size, dist_params, rng)

        # Get on sparse form
        edge_index = W0.nonzero().t()
        W0 = W0[edge_index[0], edge_index[1]]

        return W0, edge_index, hub_node

    @staticmethod
    def _generate_w0(cluster_size: int, dist_params: DistributionParams, rng: torch.Generator) -> torch.Tensor:
        """Generates a normally-drawn connectivity matrix W0 that follows Dale's law and has zeros on the diagonal"""
        if dist_params.name == 'glorot':
            W0 = W0Generator._generate_glorot_w0(cluster_size, dist_params.mean, dist_params.std, rng)
            W0 = W0Generator._dales_law(W0)
        if dist_params.name == 'normal':
            W0 = W0Generator._generate_normal_w0(cluster_size, dist_params.mean, dist_params.std, rng)
            W0 = W0Generator._dales_law(W0)
        elif dist_params.name == 'uniform':
            W0 = W0Generator._generate_uniform_w0((cluster_size, cluster_size), rng)
            W0 = W0Generator._dales_law(W0)
        elif dist_params.name == 'mexican_hat':
            W0 = W0Generator._generate_mexican_hat_w0((cluster_size, cluster_size), rng)
            W0 = W0Generator._dales_law(W0)

        elif dist_params.name == 'small_world':   #Binary connectivity, need to assign values to the weights (range -20, 20?)
            #W0_graph = W0Generator._generate_small_world_w0(cluster_size, rng)
            W0_graph = nx.watts_strogatz_graph(cluster_size, k = cluster_size//3, p = 0.2)
            ranking = nx.voterank(W0_graph)
            hub_node = ranking[0]
            W0 = nx.to_numpy_array(W0_graph)
            W0 = torch.from_numpy(W0)
            W0 = W0Generator._insert_values(cluster_size, W0, dist_params.mean, dist_params.std)
            W0 = W0.fill_diagonal_(1)

            return W0, hub_node


        elif dist_params.name == 'barabasi':  #Binary connectivity
            W0_graph = nx.barabasi_albert_graph(cluster_size, m = cluster_size//2) 
            ranking = nx.voterank(W0_graph)
            hub_node = ranking[0]
            W0 = nx.to_numpy_array(W0_graph)
            W0 = torch.from_numpy(W0)
            W0 = W0Generator._insert_values(cluster_size, W0, dist_params.mean, dist_params.std)
            W0 = W0.fill_diagonal_(1)

            return W0, hub_node

        elif dist_params.name == "caveman":   #As of now, not very useful...
            W0_graph = nx.connected_caveman_graph(3, 10)
            W0_graph = W0_graph.to_directed()
            W0 = nx.to_numpy_array(W0_graph)
            W0 = torch.from_numpy(W0)
            

        W0 = W0.fill_diagonal_(1)

        return W0, 0
    
    @staticmethod
    def _dales_law(W0: torch.Tensor) -> torch.Tensor:
        """Applies Dale's law to the connectivity matrix W0"""
        W0 = torch.concat((W0 * (W0 > 0), W0 * (W0 < 0)), 0)
        return W0


    @staticmethod
    def _insert_values(cluster_size: int, W0: torch.Tensor, mean: int, std: int)  -> torch.Tensor:
        """Inserts values from a normal distribution into to the binary connectivity matrix"""
        vals = torch.normal(mean, std, size = (cluster_size, cluster_size))
        pos_tensor = torch.abs(vals[:cluster_size//2, :])
        neg_tensor = -torch.abs(vals[cluster_size//2:, :])
        W0_vals = torch.cat((pos_tensor, neg_tensor), 0)
        W0 = W0 * W0_vals
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
    def _new_single_weight(chemical_action: str, cluster_size: int, dist_params: DistributionParams) -> torch.Tensor:
        """Generates a new weight for one of the hub neurons"""
        if dist_params.name == 'glorot':
            w0 = torch.normal(dist_params.mean, dist_params.std, (1,)) / torch.sqrt(torch.tensor(cluster_size))
        elif dist_params.name == 'normal':
            w0 = torch.normal(dist_params.mean, dist_params.std, (1,))
        elif dist_params.name == 'barabasi':
            w0 = torch.normal(dist_params.mean, dist_params.std, (1,))
        elif dist_params.name == 'small_world':
            w0 = torch.normal(dist_params.mean, dist_params.std, (1,))
        else:
            raise NotImplementedError
        if chemical_action == 'excitatory':
            w0  = w0 * torch.sign(w0)
        elif chemical_action == 'inhibitory':
            w0  = w0 * -torch.sign(w0)
        return w0

    @staticmethod
    def _to_tensor(W0, edge_index):
        """Converts the W0 and edge_index to a tensor"""
        W0 = torch.sparse_coo_tensor(edge_index, W0)
        return W0.to_dense()

