from spiking_network.plotting.activity import plot_activity, visualize_spikes, more_activity

#path = "/home/users/sarapje/snn-glm-simulator/spiking_network/data/simplex/cluster_sizes_[5, 6, 7]_n_steps_10000/0.pkl"

path = "spiking_network/data/simplex/cluster_sizes_[5, 6, 7]_n_steps_10000/0.pkl"

neurons = [5, 17]

plot_activity(neurons, path)

#visualize_spikes(path)

#more_activity(neurons, path)

