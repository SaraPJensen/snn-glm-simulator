from spiking_network.data_generators.make_dataset import make_dataset
#from spiking_network.data_generators.make_herman_dataset import make_herman_dataset
from spiking_network.data_generators.make_sara_dataset import make_sara_dataset
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--network_type", type=str, default="small_world", help="Type of network")
    parser.add_argument("-s", "--cluster_sizes", type=list, default=[10, 20], help="Size of each cluster")
    parser.add_argument("-c", "--n_cluster_connections", type=int, default=1, help="Number of cluster connections")   #Is this still in use? 
    parser.add_argument("-r", "--random_cluster_connections", type = bool, default=True, help="Whether to use a random number of cluster connections")
    parser.add_argument("-th", "--threshold",   type=float, default=4.3, help="The threshold to use for small world or barabasi networks")
    parser.add_argument("-t", "--n_steps", type=int, default=10000, help="Number of steps in simulation")
    parser.add_argument("-sims", "--n_sims",       type=int,   default=5,        help="Number of simulations to run")
    parser.add_argument("-p", "--max_parallel", type=int,   default=100,      help="The max number of simulations to run in parallel")
    parser.add_argument("-path", "--data_path", type=str, default="spiking_network/data", help="The path where the data should be saved")
    args = parser.parse_args()

    print("Generating datasets...")
    print(f"Network type:                    {args.network_type}")
    print(f"Number of clusters:              {len(args.cluster_sizes)}")
    print(f"Cluster sizes:                   {args.cluster_sizes}")
    print(f"Random cluster connections:      {args.random_cluster_connections}")
    print(f"Number of steps:                 {args.n_steps}")
    print(f"Number of simulations:           {args.n_sims}")
    print(f"Path to store data:              {args.data_path}")

 
    #make_dataset(args.network_type, args.cluster_sizes, args.random_cluster_connections, args.n_steps, args.n_datasets, args.data_path)
    
    
    make_sara_dataset(args.network_type, args.cluster_sizes, args.random_cluster_connections, args.n_sims, args.n_steps, args.data_path, args.max_parallel, args.threshold)

    
    # parser.add_argument("-n", "--n_neurons",    type=int,   default=20,       help="Number of neurons in the network")
    # parser.add_argument("-t", "--n_steps",      type=int,   default=10_000,   help="Number of steps in simulation")
    # parser.add_argument("-s", "--n_sims",       type=int,   default=1,        help="Number of simulations to run")
    # parser.add_argument("-r", "--r",            type=float, default=0.025,    help="The r to use for the herman case")
    # parser.add_argument("-th", "--threshold",   type=float, default=1.378e-3, help="The threshold to use for the herman case")
    # parser.add_argument("--data_path",          type=str,   default="data",   help="The path where the data should be saved")
    # parser.add_argument("-p", "--max_parallel", type=int,   default=100,      help="The max number of simulations to run in parallel")
    # parser.add_argument("--herman",                                           help="Run hermans simulation", action="store_true")
    # args = parser.parse_args()

    # print("Generating datasets...")
    # print(f"Number of neurons:                            {args.n_neurons}")
    # print(f"Number of simulations:                        {args.n_sims}")
    # print(f"Number of steps:                              {args.n_steps}")
    # print(f"Path to store data:                           {args.data_path}")
    # print(f"Running Hermans simulation?:                  {args.herman}")
    # print(f"Max number of simulation to run in parallel:  {args.max_parallel}")

    # if args.herman:
    #     make_herman_dataset(args.n_neurons, args.n_sims, args.n_steps, args.data_path, args.max_parallel)
    # else:
    #     make_dataset(args.n_neurons, args.n_sims, args.n_steps, args.data_path, args.max_parallel)


if __name__ == "__main__":
    main()
