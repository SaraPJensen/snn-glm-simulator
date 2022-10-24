from spiking_network.models.spiking_model import SpikingModel
from spiking_network.w0_generators.w0_dataset import ConnectivityDataset
from spiking_network.w0_generators.w0_generator import W0Generator, GlorotParams, SmallWorldParams, BarabasiParams
from spiking_network.data_generators.save_functions import save
from pathlib import Path
from spiking_network.plotting.visualize_sim import visualize_spikes, load_data
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader


def make_sara_dataset(network_type: str, cluster_sizes: list[int], random_cluster_connections: bool, n_sims: int, n_steps: int, data_path: str, max_parallel: int, threshold: float):

    #(n_neurons, n_sims, n_steps, data_path, max_parallel):

    """Generates a dataset"""
    # Set data path
    data_path = Path(data_path)/network_type/f"cluster_sizes_{cluster_sizes}_n_steps_{n_steps}"
    data_path.mkdir(parents=True, exist_ok=True)

    # Parameters for the simulation
    batch_size = min(n_sims, max_parallel)
    total_neurons = sum(cluster_sizes) * batch_size

    # Set parameters for W0
    if network_type == "small_world":
        dist_params = SmallWorldParams()
    elif network_type == "barabasi":
        dist_params = BarabasiParams()
 

    # You can generate a list of W0s here in your own way
    w0_generator = W0Generator(cluster_sizes, random_cluster_connections, dist_params) 
    w0_list = w0_generator.generate_list(n_sims, seed=0) 

    # Now you can load the W0s into a dataset in this way. This makes it easy to parallelize.
    # Note that the square w0s will be split into a sparse representation with w0.shape = [n_edges] and edge_index.shape = [2, n_edges]
    w0_data = ConnectivityDataset.from_list(w0_list)
    data_loader = DataLoader(w0_data, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for i, batch in enumerate(data_loader):
        batch = batch.to(device)

        # Initalize model
        model = SpikingModel(
            batch.W0_sparse, batch.edge_index, total_neurons, threshold, seed=i, device=device   
        )   
        
        # Simulate the model for n_steps
        spikes = model.simulate(n_steps)

        save(spikes, model, w0_data, i, data_path, w0_generator) # Insert your own way of saving the data here (see save_functions.py)

        # xs = torch.split(spikes, [network.num_nodes for network in w0_data], dim=0)
        # for X in xs:
        #     tot_secs = n_steps/1000
        #     frequency = torch.sum(X)/tot_secs/sum(cluster_sizes)
        #     print(f"Average frequency: {frequency}")
        #     visualize_spikes(X)


if __name__ == "__main__":
    make_dataset(10, 20, 1, 1000, 1)
