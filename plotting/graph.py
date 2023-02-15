import networkx as nx
import matplotlib.pyplot as plt

def plot_graph(path, graph):

    with open(path, "rb") as f:
        stuff = pickle.load(f)   #stuff = dictionary

    W = stuff["W0"]

    graph = nx.from_numpy_matrix(W)

    fig = nx.draw(graph)

    fig.show()

