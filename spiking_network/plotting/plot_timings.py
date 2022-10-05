import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Make plots with seaborn
sns.set_theme()
def load_data(filename):
    data = np.load("benchmarking/benchmark_data/" + filename + ".npz")
    timings = data["timings"]
    neuron_list = data["neuron_list"]
    return timings, neuron_list

def plot_timings(filename):
    timings, neuron_list = load_data(filename)
    plt.figure()
    plt.plot(neuron_list, timings, label=["Old simulation", "New simulation", "Rolling x"])
    plt.xlabel("Number of neurons")
    plt.ylabel("Avg time (s)")
    plt.legend()
    plt.savefig("plots/" + filename + ".png")
    plt.show()

def plot_parallization(filename):
    data = np.load("benchmarking/benchmark_data/" + filename + ".npz")
    timings = data["timings"]
    p_sims = data["p_sims"]
    plt.figure()
    plt.plot(p_sims, timings, label=[str(10*i) for i in range(1, 11)])
    plt.legend()
    plt.title("Parallelization of 512 simulations")
    plt.xlabel("Number of parallel simulations")
    plt.ylabel("Avg time (s) per simulation")
    plt.savefig("plots/" + filename + ".png")
    plt.show()

# plot_parallization("parallelization_compare_across_neurons")
plot_timings("comp_with_rolling_for_memory")
