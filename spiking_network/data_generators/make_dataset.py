from spiking_network.models.spiking_model import SpikingModel
from spiking_network.w0_generators.w0_dataset import W0Dataset, SparseW0Dataset
from spiking_network.stimulation.regular_stimulation import RegularStimulation
from spiking_network.stimulation.poisson_stimulation import PoissonStimulation
from spiking_network.stimulation.sin_stimulation import SinStimulation
from spiking_network.stimulation.mixed_stimulation import MixedStimulation
from spiking_network.w0_generators.w0_generator import GlorotParams, NormalParams
from spiking_network.data_generators.save_functions import save
from spiking_network.models.spiking_model import SpikingModel
from spiking_network.connectivity_filters.connectivity_filter import ConnectivityFilter
from pathlib import Path
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader

def calculate_isi(spikes, N, n_steps, dt=0.001) -> float:
    return N * n_steps * dt / spikes.sum()

def make_dataset(n_neurons, n_sims, n_steps, data_path, max_parallel=100, firing_rate=0.1):
    """Generates a dataset"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility
    rng = torch.Generator().manual_seed(0)
    seeds = {
            "w0": torch.randint(0, 100000, (n_sims,), generator=rng).tolist(),
            "model": torch.randint(0, 100000, (1,), generator=rng).item(),
         }

    # Set path for saving data
    data_path = (
        "spiking_network" / Path(data_path) / f"{n_neurons}_neurons_{n_sims}_datasets_{n_steps}_steps"
    )
    data_path.mkdir(parents=True, exist_ok=True)

    # Prepare dataset
    w0_params = GlorotParams(0, 5)
    w0_data = W0Dataset(n_neurons, n_sims, w0_params, seeds=seeds["w0"])
    data_loader = DataLoader(w0_data, batch_size=min(n_sims, max_parallel), shuffle=False)

    model_path = Path("spiking_network/models/saved_models") / f"{w0_params.name}_{n_neurons}_neurons_{firing_rate}_firing_rate.pt"
    model = SpikingModel(seed=seeds["model"], device=device).load(model_path) if model_path.exists() else SpikingModel(seed=seeds["model"], device=device)

    results = []
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        stim0 = RegularStimulation(targets=0, strengths=1, duration=n_steps, rates=0.2, temporal_scales=2, n_neurons=data.num_nodes, device=device)
        stim1 = PoissonStimulation(targets=0, strengths=1, duration=n_steps, periods=5, temporal_scales=4, n_neurons=data.num_nodes, device=device)
        mixed_stim = MixedStimulation([stim0, stim1])
        spikes = model.simulate(data, n_steps, stimulation=mixed_stim)
        #  print(f"ISI: {calculate_isi(spikes, data.num_nodes, n_steps)}")
        print(f"Firing rate: {spikes.mean()}")
        results.append(spikes)

    save(results, model, w0_data, seeds, data_path)
