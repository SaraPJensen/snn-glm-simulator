from spiking_network.plotting.activity import plot_activity, visualize_spikes, more_activity
from spiking_network.plotting.graph import plot_graph

#path = "/home/users/sarapje/snn-glm-simulator/spiking_network/data/simplex/cluster_sizes_[5, 6, 7]_n_steps_10000/0.pkl"

path = "/home/users/sarapje/snn-glm-simulator/spiking_network/data/simplex/cluster_sizes_[6, 7]_n_steps_50000/3.pkl"

neurons = [5, 11]

plot_activity(neurons, path)

#visualize_spikes(path)

#more_activity(neurons, path)


#plot_graph(path)

