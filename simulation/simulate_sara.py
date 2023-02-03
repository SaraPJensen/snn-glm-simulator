from spiking_network.models import GLMModel

from spiking_network.datasets.w0_generator import W0Generator, SmallWorldParams, SimplexParams 
from spiking_network.datasets.w0_dataset import ConnectivityDataset
from simulation.save_func import save
from spiking_network.utils import simulate

from pathlib import Path
#from spiking_network.plotting.visualize_sim import visualize_spikes, load_data
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader


def sara_simulate(network_type: str, cluster_sizes: list[int], random_cluster_connections: bool, n_sims: int, n_steps: int, data_path: str, max_parallel: int, threshold: float, seed: int):

    # Reproducibility
    rng = torch.Generator().manual_seed(seed)
    seeds = {
            "w0": torch.randint(0, 100000, (n_sims,), generator=rng).tolist(),
            "model": torch.randint(0, 100000, (1,), generator=rng).item(),
         }

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
    elif network_type == "simplex":
        dist_params = SimplexParams()
 

    # Generate a list of W0s 
    w0_generator = W0Generator(cluster_sizes, random_cluster_connections, dist_params) 
    w0_list = w0_generator.generate_list(n_sims, seed=0) 

    # Load the W0s into a dataset. This makes it easy to parallelize.
    # Sparse representation with w0.shape = [n_edges] and edge_index.shape = [2, n_edges]
    w0_data = ConnectivityDataset.from_list(w0_list)
    data_loader = DataLoader(w0_data, batch_size=batch_size, shuffle=False)

    previous_batches = 0

    params = {
            "threshold": threshold             
        }

    device = "cuda" if torch.cuda.is_available() else "cpu" 
    model = GLMModel(params=params, seed=seeds["model"], device = device)

    for batch_idx, batch in enumerate(data_loader):
        batch = batch.to(device)
        batch_size = len(batch)


        # Simulate the model for n_steps
        spikes = simulate(model, batch, n_steps)       
        w0_data_subset = w0_data[previous_batches:previous_batches + batch_size]   #Pick out the corresponding subset of W0s for each batch
        
        save(spikes, model, w0_data_subset, seeds, previous_batches, data_path, w0_generator) 

        previous_batches += batch_size

        xs = torch.split(spikes, [network.num_nodes for network in w0_data], dim=0)
        for X in xs:
            tot_secs = n_steps/1000
            frequency = torch.sum(X)/tot_secs/sum(cluster_sizes)
            print(f"Average frequency: {frequency}")
            #visualize_spikes(X)


        


if __name__ == "__main__":
    make_dataset(10, 20, 1, 1000, 1)