from spiking_network.models.spiking_model import SpikingModel
#from spiking_network.w0_generators.w0_dataset import W0Dataset
from spiking_network.stimulation.regular_stimulation import RegularStimulation
from spiking_network.stimulation.poisson_stimulation import PoissonStimulation
from spiking_network.stimulation.sin_stimulation import SinStimulation
from spiking_network.w0_generators.w0_generator import GlorotParams, NormalParams
from spiking_network.data_generators.save_functions import save
from pathlib import Path
from tqdm import tqdm
from spiking_network.plotting.visualize_sim import visualize_spikes, load_data
import torch
from torch_geometric.loader import DataLoader



def initial_condition(n_neurons, time_scale, seed):
    """Initializes the network with a random number of spikes"""
    rng = torch.Generator()
    rng.manual_seed(seed)
    #init_cond = torch.ones((n_neurons,), dtype=torch.bool)   #Didn't really help 
    init_cond = torch.randint(0, 2, (n_neurons,), dtype=torch.bool, generator=rng)
    x_initial = torch.zeros((n_neurons, time_scale), dtype=torch.bool)
    x_initial[:, -1] = init_cond
    return x_initial

def save_parallel(spikes, connectivity_filter, n_steps, n_neurons_list, n_edges_list, seed, data_path: str) -> None:
    """Saves the spikes to a file"""
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    n_clusters = len(n_neurons_list)
    cluster_size = connectivity_filter.n_neurons  // n_clusters

    x = torch.sparse_coo_tensor(spikes, torch.ones_like(spikes[0]), size=(connectivity_filter.n_neurons, n_steps)).to_dense()

    x_sims = torch.split(x, n_neurons_list, dim=0)
    Ws = torch.split(connectivity_filter.W, n_edges_list, dim=0)
    edge_indices = torch.split(connectivity_filter.edge_index, n_edges_list, dim=1)
    for i, (x_sim, W_sim, edge_index_sim) in enumerate(zip(x_sims, Ws, edge_indices)):
        sparse_x = coo_matrix(x_sim)
        np.savez(
                data_path / Path(f"{seed}_{i}.npz"),
                X_sparse = sparse_x,
                W=W_sim,
                edge_index=edge_index_sim,
                seed=seed,
                filter_params = connectivity_filter.parameters
)


def save(sparse_x, w0_generator, connectivity_filter, n_steps, edge_index_hubs, seed, data_path):
    """Saves the spikes and the connectivity filter to a file"""
    # x = spikes[0]
    # t = spikes[1]
    # data = torch.ones_like(t)
    # sparse_x = coo_matrix((data, (x, t)), shape=(connectivity_filter.W0.shape[0], n_steps))
    
    np.savez_compressed(
            data_path,
            X_sparse = sparse_x,
            W0 = connectivity_filter.W0,
            W = connectivity_filter.W,
            edge_index=connectivity_filter.edge_index,
            W0_hubs = connectivity_filter.W0_hubs, 
            edge_index_hubs = edge_index_hubs,
            W0_parameters = w0_generator.parameters,
            time_parameters = connectivity_filter.parameters,
            seed=seed,
        )


def make_dataset(network_type, cluster_sizes, random_cluster_connections, n_steps, n_datasets, data_path):
    """Generates a dataset"""
    # Set data path

    n_clusters = len(cluster_sizes)
    data_path = Path(data_path)/network_type/f"cluster_sizes_{cluster_sizes}_n_steps_{n_steps}"
    data_path.mkdir(parents=True, exist_ok=True)

    # Set parameters for W0
    if network_type == "small_world":
        dist_params = SmallWorldParams()
    elif network_type == "barabasi":
        dist_params = BarabasiParams()

    #As of now, it is not possible to vary these parameters between datasets in the same folder
    #Future: make the cluster_sizes etc properties of the connectivity filter, not the generator
    #Let the connectivity filter class inherit the functions from W0Generator
    w0_generator = W0Generator(cluster_sizes, random_cluster_connections, dist_params)   

    device = "cuda" if torch.cuda.is_available() else "cpu" 
    with torch.no_grad(): # Disables gradient computation (the models are built on top of torch)
        for i in tqdm(range(n_datasets)):

            frequency = 100

            while frequency > 50:   #If something exploded, try again... 
                W0, W0_hubs, edge_index_hubs = w0_generator.generate(i) # Generates a random W0
                connectivity_filter = ConnectivityFilter(W0, W0_hubs) # Creates a connectivity filter from W0, with the time dependency    Note: this also creates the edge_index, so no need to return that from anywhere else
                W, edge_index = connectivity_filter.W, connectivity_filter.edge_index

                #connectivity_filter.plot_graph()
                #connectivity_filter.plot_connectivity()

                model = SpikingModel(W, edge_index, n_steps, seed=i, device=device) # Initializes the model
                x_initial = initial_condition(connectivity_filter.n_neurons, connectivity_filter.time_scale, seed=i) # Initializes the network with a random number of spikes
                x_initial = x_initial.to(device)
                spikes = model(x_initial) # Simulates the network

                tot_secs = n_steps/1000

                x = spikes[0]
                t = spikes[1]
                data = torch.ones_like(t)
                sparse_x = coo_matrix((data, (x, t)), shape=(connectivity_filter.W0.shape[0], n_steps))

                num_spikes = sparse_x.sum()
                frequency = num_spikes/tot_secs/connectivity_filter.n_neurons

                print(f"Frequency: {frequency}")
            
            save(sparse_x, w0_generator, connectivity_filter, n_steps, edge_index_hubs, i, data_path/Path(f"{i}.npz")) # Saves the spikes and the connectivity filter to a file
            
            # X, _, _ = load_data(data_path/Path(f"{i}.npz"))
            # visualize_spikes(X)
            
            # x = np.load(data_path / Path(f"{i}.npz"), allow_pickle= True)
            # for k in x.files:
            #     print(k) 





#This is Jakob's current version
def make_dataset(n_neurons, n_sims, n_steps, data_path, max_parallel, p=0.1):
    """Generates a dataset"""
    # Set data path
    data_path = (
        "spiking_network" / Path(data_path) / f"{n_neurons}_neurons_{n_sims}_datasets_{n_steps}_steps"
    )
    data_path.mkdir(parents=True, exist_ok=True)

    # Parameters for the simulation
    batch_size = min(n_sims, max_parallel)
    w0_params = GlorotParams(0, 5)
    total_neurons = batch_size*n_neurons

    # Generate W0s to simulate and put them in a data loader
    w0_data = W0Dataset(n_neurons, n_sims, w0_params, seed=0)
    data_loader = DataLoader(w0_data, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for i, batch in enumerate(data_loader):
        batch = batch.to(device)

        # Initalize model
        model = SpikingModel(
            batch.W0, batch.edge_index, total_neurons, seed=i, device=device
        )

        # If we already have a tuned model for this initial distribution, 
        # number of neurons and probability of firing, we can load it. Otherwise we need to tune it.
        # Note that p is the probability that each neuron will fire per timestep.
        model_path = (
                Path("spiking_network/models/saved_models") / f"{w0_params.name}_{n_neurons}_neurons_{p}_probability.pt"
        )
        if model_path.exists():
            model.load(model_path)
        else:
            model.tune(p=p, epsilon=1e-6)

        # Generate different types of stimulation
        regular_stimulation = RegularStimulation([0, 1, 2], rates=0.1, strengths=3, temporal_scales=2, duration=n_steps, n_neurons=total_neurons, device=device)
        poisson_stimulation = PoissonStimulation([5, 3, 9], periods=5, strengths=6, temporal_scales=4, duration=100, n_neurons=total_neurons, device=device)
        sin_stimulation = SinStimulation([4, 6, 8], amplitudes=10, frequencies=0.001, duration=n_steps, n_neurons=total_neurons, device=device)
        stimulation = [regular_stimulation, poisson_stimulation, sin_stimulation]

        # Simulate the model for n_steps
        spikes = model.simulate(n_steps, stimulation)

        # Save the data and the model
        save(spikes, model, w0_data, i, data_path, stimulation=stimulation)
        model.save(model_path)

