from spiking_network.models import GLMModel

from spiking_network.datasets.w0_generator import W0Generator, SmallWorldParams, SimplexParams, RandomParams, SM_RemoveParams, LineParams 
from spiking_network.datasets.w0_dataset import ConnectivityDataset
from simulation.save_func import save
from spiking_network.utils import simulate
from spiking_network.stimulation import RegularStimulation, PoissonStimulation, SinStimulation

#from spiking_network.plotting.activity import visualize_direct 

from pathlib import Path
#from spiking_network.plotting.visualize_sim import visualize_spikes, load_data
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader


def sara_simulate(
        network_type: str, 
        cluster_sizes: list[int], 
        random_cluster_connections: bool, 
        n_sims: int, 
        n_steps: int, 
        data_path: str, 
        max_parallel: int, 
        threshold: float, 
        remove_connections: int, 
        add_connections: int, 
        stimulus: bool,
        stim_rate: float,
        seed: int):

    # Reproducibility
    rng = torch.Generator().manual_seed(seed)

    """Generates a dataset"""
    # Set data path
    data_path = Path(data_path)/network_type/f"cluster_sizes_{cluster_sizes}_n_steps_{n_steps}"

    if remove_connections > 0:
        data_path = data_path/f"removed_{remove_connections}"

    elif add_connections > 0:
        data_path = data_path/f"added_{add_connections}"

    if stimulus:
        data_path = data_path/"stimulus"
        
    data_path.mkdir(parents=True, exist_ok=True)

    if network_type == "sm_remove":
        n_sims = sum(cluster_sizes) + 1

    # Parameters for the simulation
    batch_size = min(n_sims, max_parallel)
    total_neurons = sum(cluster_sizes) * batch_size

    # Set parameters for W0
    if network_type == "small_world":
        dist_params = SmallWorldParams()
    elif network_type == "simplex":
        dist_params = SimplexParams()
    elif network_type == "random":
        dist_params = RandomParams()
    elif network_type == "sm_remove":
        dist_params = SM_RemoveParams()
    elif network_type == "line":
        dist_params = LineParams()
 

    # Generate a list of W0s 
    w0_generator = W0Generator(cluster_sizes, random_cluster_connections, dist_params, remove_connections, add_connections) 
    w0_list = w0_generator.generate_list(n_sims, seed=seed) 

    # Load the W0s into a dataset. This makes it easy to parallelize.
    # Sparse representation with w0.shape = [n_edges] and edge_index.shape = [2, n_edges]
    w0_data = ConnectivityDataset.from_list(w0_list)
    data_loader = DataLoader(w0_data, batch_size=batch_size, shuffle=False)
    previous_batches = 0

    params = {"threshold": threshold}

    device = "cuda" if torch.cuda.is_available() else "cpu" 
    model = GLMModel(params=params, seed=seed, device = device)

    
    if stimulus:
        stimuli = RegularStimulation(
            targets = [0],
            rates = [stim_rate],   #The period of the stimulus
            strengths = [1],
            temporal_scales = [1],   #How long the stimulus lasts
            duration = n_steps,
            n_neurons = sum(cluster_sizes),
            device = device
        )

    else: 
        stimuli = None
        
        
        # stimulus_mask = torch.isin(torch.arange(sum(cluster_sizes)), torch.tensor([0])) #Always stimulate the source neurons
        # data.stimulus_mask = stimulus_mask


    for batch_idx, batch in enumerate(data_loader):

        batch = batch.to(device)
        batch_size = int(batch.num_nodes/sum(cluster_sizes))   #The number of W0s in the batch

        # Simulate the model for n_steps
        spikes = simulate(model, batch, n_steps, stimuli)

        w0_data_subset = w0_data[previous_batches:previous_batches + batch_size]   #Pick out the corresponding subset of W0s for each batch

        save(spikes, model, w0_data_subset, seed, previous_batches, data_path, w0_generator) 

        previous_batches += batch_size

        xs = torch.split(spikes, [network.num_nodes for network in w0_data_subset], dim=0)
        print(xs[0][0])
        exit()
        count = 0
        for X in xs:
            tot_secs = n_steps/1000
            frequency = torch.sum(X)/tot_secs/sum(cluster_sizes)
            print(f"Average frequency: {frequency}")
            count += 1
            #visualize_direct(X.cpu())
            #exit()


