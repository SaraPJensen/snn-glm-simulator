import numpy as np
from spiking_network.w0_generators.w0_generator import W0Generator
from spiking_network.w0_generators.w0_dataset import HermanDataset
from spiking_network.models.herman_model import HermanModel
from pathlib import Path
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader
from spiking_network.data_generators.save_functions import save
from scipy.sparse import coo_matrix

#  def save(x, model, seed, data_path):
#      """Saves the spikes and the connectivity filter to a file"""
#      # To numpy and cpu
#      x = x.cpu().numpy()
#      W0 = model.W0.cpu().numpy()
#      sparse_x = coo_matrix(x)
#      sparse_W0 = coo_matrix(W0)
#      np.savez_compressed(
#          data_path,
#          X_sparse=sparse_x,
#          w_0=sparse_W0,
#          parameters=model.save_parameters(),
#          seed=seed,
#      )

def herman_save(x, model, w0_data, seed, data_path, n_neurons, stimulation=None):
    """Saves the spikes and the connectivity filter to a file"""
    x = x.cpu()
    w0_data = w0_data.cpu()
    xs = torch.split(x, n_neurons, dim=0)
    w0s = torch.split(w0_data, n_neurons*n_neurons, dim=0)
    for i, (x, w_0) in enumerate(zip(xs, w0s)):
        sparse_x = coo_matrix(x)
        sparse_W0 = coo_matrix(
            w_0.view(n_neurons, n_neurons),
        )
        np.savez_compressed(
            data_path / Path(f"seed_{seed}_simulation_{i}.npz"),
            X_sparse=sparse_x,
            w_0=sparse_W0,
            stimulation=stimulation.__dict__() if stimulation is not None else None,
            parameters=model.save_parameters(),
            seed=seed,
        )

def calculate_isi(spikes: np.ndarray, N, n_steps, dt=0.0001) -> float:
    return N * n_steps * dt / spikes.sum()

def make_herman_dataset(n_neurons, n_sims, n_steps, data_path, max_parallel, p=0.016):
    # Path to save results
    data_path = Path(data_path) / Path(f"herman_{n_neurons}_neurons_{n_sims}_sims_{n_steps}_steps")
    data_path.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = min(n_sims, max_parallel)
    herman_dataset = HermanDataset(n_neurons, n_sims, emptiness=0, seed=0)
    data_loader = DataLoader(herman_dataset, batch_size=batch_size, shuffle=False)
    for i, batch in enumerate(data_loader):
        batch.to(device)

        model = HermanModel(
                batch.W0,
                batch.edge_index,
                n_neurons*batch_size,
                seed=i,
                device=device
            )

        #  model_path = (
        #          Path("models/saved_models") / f"herman_{n_neurons}_neurons_{p}_probability.pt"
        #  )
        #  if model_path.exists():
        #      model.load(model_path)
        #  else:
        #      model.tune(p=p, epsilon=1e-6)

        spikes = model.simulate(n_steps)

        #  print(f"ISI: {calculate_isi(spikes, n_neurons, n_steps)}")

        #  save(spikes, model, i, data_path / Path(f"herman_{i}.npz"))

        herman_save(spikes, model, batch.W0, i, data_path, n_neurons)
