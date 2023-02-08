from simulation.simulate import run_simulation
from simulation.simulate_sara import sara_simulate

import argparse
import torch

import sys
from IPython.core import ultratb

import pickle
from pathlib import Path

sys.excepthook = ultratb.FormattedTB(
    mode="Context", color_scheme="Linux", call_pdb=False
)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--network_type",                 type=str,       default="sm_remove",               help="Type of network")
    parser.add_argument("-cs", "--cluster_sizes",               type=list,      default=[6, 15],                    help="Size of each cluster")
    parser.add_argument("-c", "--n_cluster_connections",        type=int,       default=1,                      help="Number of cluster connections")   #Is this still in use? 
    parser.add_argument("-rm", "--random_cluster_connections",  type = bool,    default=True,                   help="Whether to use a random number of cluster connections")
    parser.add_argument("-th", "--threshold",                   type=float,     default=4.3,                    help="The threshold to use for small world or barabasi networks")
    parser.add_argument("-t", "--n_steps",                      type=int,       default=50000,                   help="Number of steps in simulation")
    parser.add_argument("-sims", "--n_sims",                    type=int,       default=24,                      help="Number of simulations to run")
    parser.add_argument("-p", "--max_parallel",                 type=int,       default=100,                    help="The max number of simulations to run in parallel")
    parser.add_argument("-path", "--data_path",                 type=str,       default="spiking_network/data", help="The path where the data should be saved")
    parser.add_argument("-f", "--firing_rate",                  type=float,     default=0.1,                    help="The average firing fate of the neurons")
    parser.add_argument("-r", "--r",                            type=float,     default=0.025,                  help="The r to use for the herman case")
    parser.add_argument("-s", "--seed",                         type=int,       default=0,                      help="The seed to use for the simulation")

    args = parser.parse_args()

    print("Generating datasets...")
    print(f"Network type:                    {args.network_type}")
    print(f"Number of clusters:              {len(args.cluster_sizes)}")
    print(f"Cluster sizes:                   {args.cluster_sizes}")
    print(f"Random cluster connections:      {args.random_cluster_connections}")
    print(f"Number of steps:                 {args.n_steps}")
    print(f"Number of simulations:           {args.n_sims}")
    print(f"Simulations running in parallel: {args.max_parallel}")
    print(f"Path to store data:              {args.data_path}")


    sara_simulate(args.network_type, args.cluster_sizes, args.random_cluster_connections, args.n_sims, args.n_steps, args.data_path, args.max_parallel, args.threshold, args.seed)
    

if __name__ == "__main__":
    main()



#Den klarer ikke å lagre grafer for simplex med flere enn 1 cluster