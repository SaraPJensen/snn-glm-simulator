import numpy as np
import pingouin as pg
import pandas as pd
import pickle
from scipy.sparse import coo_matrix
import torch
import sys
sys.path.append("./..")
#import spiking_network
#import data 


def get_data(path):
    with open(path, "rb") as f:
        print(f)
        stuff = pickle.load(f)   #stuff = dictionary

    X_sparse = stuff["X_sparse"]

    coo = coo_matrix(X_sparse)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    X = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense().numpy() #X has shape (n_neurons, n_timesteps), with 1 indicating that the neuron fired at that time step

    W0 = stuff["W0"]

    return X, W0



def test(path):

    X, W0 = get_data(path)

    source = X[0, :]

    sink = X[-1, :]
    #sink = X[0, :]

    corr = pg.corr(source, sink)

    print()
    print('W0')
    print(W0)
    print()
    print("Correlation between source and sink")
    print('-----------------------------------')

    for i in range(0, 11):
        corr = pg.corr(source, np.roll(sink, -i))
        print(corr['r'].values[0].round(5))
    

    print()
    print("Probability for sink: ", np.sum(sink)/len(sink))
    print()
    print("Conditonal probability for sink given source")
    print('-----------------------------------')
    for i in range(0, 15):
        p_source = np.sum(source)/len(source)
        p_sink = np.sum(np.roll(sink, -i))/len(sink)
        p_source_and_sink = (np.sum(source*np.roll(sink, -i))/len(sink))

        p_sink_given_source = p_source_and_sink/p_source

        relative_change = (p_sink_given_source - p_sink)/p_sink

        print("Conditional prob: ", p_sink_given_source.round(2), "  Relative change: ", relative_change.round(5), "  Absolute change: ", (p_sink_given_source - p_sink).round(5))



path = "../data/simplex/cluster_sizes_[5]_n_steps_200000/3.pkl"
#path = "/home/users/sarapje/snn-glm-simulator/data/simplex/cluster_sizes_[6]_n_steps_50000/3.pkl"


test(path)



def ATE(path):
    X, W0 = get_data(path)

    source = X[0, :]
    sink = X[-1, :]
    print()

    for i in range(0, 20):
        p_source = np.sum(source)/len(source)
        p_not_source = 1 - p_source

        p_sink = np.sum(sink)/len(sink)
        p_source_and_sink = (np.sum(source*np.roll(sink, -i))/len(sink))
        p_not_sink = 1 - p_sink

        p_sink_and_not_source = p_sink - p_source_and_sink

        correlation = pg.corr(source, np.roll(sink, -i))['r'].values[0]

        print("Time: ", i)

        p_sink_given_source = p_source_and_sink/p_source
        print("P sink given source: ", p_sink_given_source.round(5))

        p_sink_given_not_source = (p_sink - p_source_and_sink)/(1 - p_source)
        print("P sink given not source: ", p_sink_given_not_source.round(5))

        print("Sum of probabilities: ", (p_sink_given_source*p_source + p_sink_given_not_source*p_not_source).round(5)) 
        print("P sink: ", p_sink.round(5))

        ATE = (p_sink_given_source - p_sink_given_not_source)/p_sink_given_not_source   #The relative percentage-wise increase in the probability of the sink neuron firing given that the source neuron fired

        effective_change = (p_sink_given_source - p_sink)/p_sink

        print("ATE: ", ATE.round(5))
        print("Effective change: ", effective_change.round(5))
        print("Correlation: ", correlation.round(5))
        print()



ATE(path)

