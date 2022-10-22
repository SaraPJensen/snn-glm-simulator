from spiking_network.data_generators.make_dataset import make_dataset
import argparse

# Next steps
# 2. Tuning with torch
# 4. Find better way of aggregating in numpy
# 4. Compare speed gains from parallelization
# 5. Compare speed gains from sparsification
# 7. Get torch_geometric on the cluster
# 8. Make network unaware of simulator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--network_type", type=str, default="small_world", help="Type of network")
    parser.add_argument("-s", "--cluster_sizes", type=list, default=[24, 10, 12], help="Size of each cluster")
    parser.add_argument("-c", "--n_cluster_connections", type=int, default=1, help="Number of cluster connections")
    parser.add_argument("-r", "--random_cluster_connections", type = bool, default=True, help="Whether to use a random number of cluster connections")
    parser.add_argument("-t", "--n_steps", type=int, default=100000, help="Number of steps in simulation")
    parser.add_argument("-d", "--n_datasets", type=int, default=150, help="Number of datasets to generate")
    parser.add_argument("-p", "--data_path", type=str, default="spiking_network/data", help="The path where the data should be saved")
    args = parser.parse_args()

    print("Generating datasets...")
    print(f"Network type: {args.network_type}")
    print(f"n_clusters: {len(args.cluster_sizes)}")
    print(f"cluster_sizes: {args.cluster_sizes}")
    print(f"random_cluster_connections: {args.random_cluster_connections}")
    print(f"n_steps: {args.n_steps}")
    print(f"n_datasets: {args.n_datasets}")
    print(f"path: {args.data_path}")

    make_dataset(args.network_type, args.cluster_sizes, args.random_cluster_connections, args.n_steps, args.n_datasets, args.data_path)


if __name__ == "__main__":
    main()
