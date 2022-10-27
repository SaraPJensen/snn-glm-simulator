from spiking_network.data_generators.make_dataset import make_dataset
#from spiking_network.data_generators.make_herman_dataset import make_herman_dataset
from spiking_network.data_generators.make_sara_dataset import make_sara_dataset
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--network_type", type=str, default="small_world", help="Type of network")
    parser.add_argument("-s", "--cluster_sizes", type=list, default=[20, 30, 50], help="Size of each cluster")
    parser.add_argument("-c", "--n_cluster_connections", type=int, default=1, help="Number of cluster connections")   #Is this still in use? 
    parser.add_argument("-r", "--random_cluster_connections", type = bool, default=True, help="Whether to use a random number of cluster connections")
    parser.add_argument("-th", "--threshold",   type=float, default=4.3, help="The threshold to use for small world or barabasi networks")
    parser.add_argument("-t", "--n_steps", type=int, default=200000, help="Number of steps in simulation")
    parser.add_argument("-sims", "--n_sims",       type=int,   default=200,        help="Number of simulations to run")
    parser.add_argument("-p", "--max_parallel", type=int,   default=50,      help="The max number of simulations to run in parallel")
    parser.add_argument("-path", "--data_path", type=str, default="spiking_network/data", help="The path where the data should be saved")
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

    make_sara_dataset(args.network_type, args.cluster_sizes, args.random_cluster_connections, args.n_sims, args.n_steps, args.data_path, args.max_parallel, args.threshold)


if __name__ == "__main__":
    main()
