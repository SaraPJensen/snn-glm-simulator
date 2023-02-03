import pickle
import numpy as np


#PATH = "spiking_network/data/simplex/cluster_sizes_[3,4,5]_n_steps_10000/0.pkl"

PATH = "0.pkl"

with open(PATH, "rb") as f:
    stuff = pickle.load(f)   #stuff = dictionary


#print(stuff["W0"])

#print(stuff['Hasse_diagram'])

Hasse = stuff['Hasse_diagram']

print(Hasse[-1])