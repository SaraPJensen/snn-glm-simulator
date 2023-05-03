import scipy.stats as stats
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import coo_matrix
import torch
import pingouin as pg
import sys
from tqdm import tqdm
import pyinform as pi
sys.path.append("./..")
import seaborn as sns
import statsmodels.tsa.stattools 
from scipy.stats import sem

import matplotlib.pyplot as plt


def get_data(path):
    with open(path, "rb") as f:
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






def anova_ac(min_dim, max_dim, t_shift):

    dims = max_dim - min_dim +1

    all_autocorr = np.zeros((200, dims))

    for dim_idx, dim in tqdm(enumerate(range(min_dim, max_dim+1))):

        for network in tqdm(range(0, 200)):
            path = f"../data/simplex/cluster_sizes_[{dim}]_n_steps_200000/stimulus_rate_15/{network}.pkl"
            X, W0 = get_data(path)

            sink = X[-1, :]

            corr_15 = statsmodels.tsa.stattools.acf(sink, nlags=t_shift)[1:]
            
            #print(np.argmax(corr_15[10:]))

            all_autocorr[network, dim_idx] = corr_15[-1]

        #print("Dim: ", dim, np.mean(all_autocorr[:, dim_idx]))


    #print("Time shift: ", t)
    df = pd.DataFrame(all_autocorr)

    mwu = np.zeros((dims, dims))

    for i in range(0, dims):
        for j in range(0, dims):
            U1, p = stats.mannwhitneyu(df[i], df[j])
            mwu[i, j] = p

    sns_plot = sns.heatmap(mwu, annot=True, fmt=".2f", cmap="crest")
    sns_plot.set_xticklabels(range(min_dim, max_dim+1))
    sns_plot.set_yticklabels(range(min_dim, max_dim+1))

    plt.xlabel("Neurons", fontsize=14)
    plt.ylabel("Neurons", fontsize=14)
    plt.title(r'$\mathit{p}$') 
    title = f"AC differences for \n time-shift " + r"$\mathit{P}$" +  " (" + r"$\mathit{p}$" + "-values)"
    if t_shift == 22:
        title = f"AC differences for \n time-shift 1.5" + r"$\mathit{P}$" +  " (" + r"$\mathit{p}$" + "-values)"
    plt.title(title, fontsize=16)

    sns_plot.figure.savefig(f"figures/autocorr/mwu_pvals_ac_dim_tshift_{t_shift}.pdf")

    plt.close(sns_plot.figure)


    binary = (mwu < 0.05).astype(np.int_)
    sns_binary = sns.heatmap(binary, annot=True, fmt=".0f", cmap="crest_r")
    sns_binary.set_xticklabels(range(min_dim, max_dim+1))
    sns_binary.set_yticklabels(range(min_dim, max_dim+1))

    plt.xlabel("Neurons", fontsize=14)
    plt.ylabel("Neurons", fontsize=14)
    title = f"Significant AC differences for \n time-shift " + r"$\mathit{P}$" +  " (" + r"$\mathit{p}$" + "-values)"
    if t_shift == 22:
        title = f"Significant AC differences for \n time-shift 1.5" + r"$\mathit{P}$" +  " (" + r"$\mathit{p}$" + "-values)"
    plt.title(title, fontsize=16)

    sns_binary.figure.savefig(f"figures/autocorr/mwu_binary_ac_dim_tshift_{t_shift}.pdf")
    plt.close(sns_binary.figure)

    

# anova_ac(3, 10, 15)
# anova_ac(3, 10, 15+7)





def anova_ac(min_dim, max_dim, t_shift):

    dims = max_dim - min_dim +1

    all_autocorr = np.zeros((200, dims))

    for dim_idx, dim in tqdm(enumerate(range(min_dim, max_dim+1))):

        for network in tqdm(range(0, 200)):
            path = f"../data/simplex/cluster_sizes_[{dim}]_n_steps_200000/stimulus_rate_15/{network}.pkl"
            X, W0 = get_data(path)

            sink = X[-1, :]

            corr_15 = statsmodels.tsa.stattools.acf(sink, nlags=t_shift)[1:]
            
            #print(np.argmax(corr_15[10:]))

            all_autocorr[network, dim_idx] = corr_15[-1]

        #print("Dim: ", dim, np.mean(all_autocorr[:, dim_idx]))


    #print("Time shift: ", t)
    df = pd.DataFrame(all_autocorr)

    mwu = np.zeros((dims, dims))

    for i in range(0, dims):
        for j in range(0, dims):
            U1, p = stats.mannwhitneyu(df[i], df[j])
            mwu[i, j] = p

    sns_plot = sns.heatmap(mwu, annot=True, fmt=".2f", cmap="crest")
    sns_plot.set_xticklabels(range(min_dim, max_dim+1))
    sns_plot.set_yticklabels(range(min_dim, max_dim+1))

    plt.xlabel("Neurons", fontsize=14)
    plt.ylabel("Neurons", fontsize=14)
    plt.title(r'$\mathit{p}$') 
    title = f"AC differences for \n time-shift " + r"$\mathit{P}$" +  " (" + r"$\mathit{p}$" + "-values)"
    if t_shift == 22:
        title = f"AC differences for \n time-shift 1.5" + r"$\mathit{P}$" +  " (" + r"$\mathit{p}$" + "-values)"
    plt.title(title, fontsize=16)

    sns_plot.figure.savefig(f"figures/autocorr/mwu_pvals_ac_dim_tshift_{t_shift}.pdf")

    plt.close(sns_plot.figure)


    binary = (mwu < 0.05).astype(np.int_)
    sns_binary = sns.heatmap(binary, annot=True, fmt=".0f", cmap="crest_r")
    sns_binary.set_xticklabels(range(min_dim, max_dim+1))
    sns_binary.set_yticklabels(range(min_dim, max_dim+1))

    plt.xlabel("Neurons", fontsize=14)
    plt.ylabel("Neurons", fontsize=14)
    title = f"Significant AC differences for \n time-shift " + r"$\mathit{P}$" +  " (" + r"$\mathit{p}$" + "-values)"
    if t_shift == 22:
        title = f"Significant AC differences for \n time-shift 1.5" + r"$\mathit{P}$" +  " (" + r"$\mathit{p}$" + "-values)"
    plt.title(title, fontsize=16)

    sns_binary.figure.savefig(f"figures/autocorr/mwu_binary_ac_dim_tshift_{t_shift}.pdf")
    plt.close(sns_binary.figure)

    

# anova_ac(3, 10, 15)
# anova_ac(3, 10, 15+7)








def ac_mann_whitney_added_removed(size, t_shift, max_change):

    all_added = []
    all_removed = []
    complete = []

    for network in tqdm(range(0, 200)):
        complete_path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/stimulus_rate_15/{network}.pkl"

        X_complete, _ = get_data(complete_path)
        sink_complete = X_complete[-1, :]

        corr_15 = statsmodels.tsa.stattools.acf(sink_complete, nlags=t_shift)[1:]

        complete.append(corr_15[-1])
                        
        for change in range(1, max_change+1):   
            added_path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/added_{change}/stimulus_rate_15/{network}.pkl"
            removed_path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/removed_{change}/stimulus_rate_15/{network}.pkl"

            X_added, _ = get_data(added_path)
            X_removed, _ = get_data(removed_path)

            sink_added = X_added[-1, :]
            sink_removed = X_removed[-1, :]

            corr_15_added = statsmodels.tsa.stattools.acf(sink_added, nlags=t_shift)[1:]
            corr_15_removed = statsmodels.tsa.stattools.acf(sink_removed, nlags=t_shift)[1:]

            all_added.append(corr_15_added[-1])
            all_removed.append(corr_15_removed[-1])
        
            
    print("Time shift: ", t_shift)

    all = [all_removed, complete, all_added]

    print(np.mean(all[0]), np.std(all[0]))
    print(np.mean(all[1]), np.std(all[1]))
    print(np.mean(all[2]), np.std(all[2]))
    
    mwu = np.zeros((3, 3))

    print("Removed vs complete: ", stats.mannwhitneyu(all_removed, complete)[1])#stats.mannwhitneyu(df[0], df[1])[1])
    print("Removed vs added: ", stats.mannwhitneyu(all_removed, all_added)[1]) #stats.mannwhitneyu(df[0], df[2])[1])
    print("Complete vs added: ", stats.mannwhitneyu(complete, all_added)[1])  #stats.mannwhitneyu(df[1], df[2])[1])

    for i in range(0, 3):
        for j in range(0, 3):
            U1, p = stats.mannwhitneyu(all[i], all[j])
            mwu[i, j] = p
    
    print()

    sns_plot = sns.heatmap(mwu, annot=True, fmt=".2f", cmap="crest")
    xtick_loc = sns_plot.get_xticks() 
    sns_plot.set_xticks(ticks = xtick_loc, labels = ["Removed", "Complete", "Added"])
    sns_plot.set_yticks(ticks = xtick_loc, labels = ["Removed", "Complete", "Added"])
    plt.xlabel("Edges", fontsize=14)
    plt.ylabel("Edges", fontsize=14)

    title = f"AC differences for time-shift " + r"$\mathit{P}$" +  f",\nsimplex size {size}, change in edges, (" + r"$\mathit{p}$" + "-values)"
    if t_shift == 22:
        f"AC differences for time-shift 1.5" + r"$\mathit{P}$" +  f",\nsimplex size {size}, change in edges, (" + r"$\mathit{p}$" + "-values)"

    title = f"TE differences for time-shift {t_shift} ms, simplex size {size},\n change in edges (" + r"$\mathit{p}$" + "-values)"
    plt.title(title, fontsize=15)
    sns_plot.figure.savefig(f"figures/autocorr/mwu_pvals_size_{size}_ac_change_tshift_{t_shift}.pdf")
    plt.close(sns_plot.figure)


    binary = (mwu < 0.05).astype(np.int_)
    sns_binary = sns.heatmap(binary, annot=True, fmt=".0f", cmap="crest_r")
    xtick_loc = sns_binary.get_xticks() 
    sns_binary.set_xticks(ticks = xtick_loc, labels = ["Removed", "Complete", "Added"])
    sns_binary.set_yticks(ticks = xtick_loc, labels = ["Removed", "Complete", "Added"])
    plt.xlabel("Edges", fontsize=14)
    plt.ylabel("Edges", fontsize=14)
    title = f"Significant AC differences for time-shift " + r"$\mathit{P}$" +  f",\nsimplex size {size}, change in edges, (" + r"$\mathit{p}$" + "-values)"
    if t_shift == 22:
        f"Significant AC differences for time-shift 1.5" + r"$\mathit{P}$" +  f",\nsimplex size {size}, change in edges, (" + r"$\mathit{p}$" + "-values)"
    plt.title(title, fontsize=15)
    sns_binary.figure.savefig(f"figures/autocorr/mwu_binary_size_{size}_ac_change_tshift_{t_shift}.pdf")
    plt.close(sns_binary.figure)



# ac_mann_whitney_added_removed(8, 15, 5)
# ac_mann_whitney_added_removed(8, 22, 5)





def ac_mann_whitney_added_removed_exact(size, t_shift, change):

    all_added = []
    all_removed = []
    complete = []


    for network in tqdm(range(0, 200)):
        complete_path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/stimulus_rate_15/{network}.pkl"

        X_complete, _ = get_data(complete_path)
        sink_complete = X_complete[-1, :]

        corr_15 = statsmodels.tsa.stattools.acf(sink_complete, nlags=t_shift)[1:]

        complete.append(corr_15[-1])

        added_path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/added_{change}/stimulus_rate_15/{network}.pkl"
        removed_path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/removed_{change}/stimulus_rate_15/{network}.pkl"

        X_added, _ = get_data(added_path)
        X_removed, _ = get_data(removed_path)

        sink_added = X_added[-1, :]
        sink_removed = X_removed[-1, :]

        corr_15_added = statsmodels.tsa.stattools.acf(sink_added, nlags=t_shift)[1:]
        corr_15_removed = statsmodels.tsa.stattools.acf(sink_removed, nlags=t_shift)[1:]

        all_added.append(corr_15_added[-1])
        all_removed.append(corr_15_removed[-1])

            
    print("Time shift: ", t_shift)

    all = [np.asarray(all_removed), np.asarray(complete), np.asarray(all_added)]

    print(np.mean(all[0]), np.std(all[0]))
    print(np.mean(all[1]), np.std(all[1]))
    print(np.mean(all[2]), np.std(all[2]))
    print()
    
    mwu = np.zeros((3, 3))

    print("Removed vs complete: ", stats.mannwhitneyu(all_removed, complete)[1])#stats.mannwhitneyu(df[0], df[1])[1])
    print("Removed vs added: ", stats.mannwhitneyu(all_removed, all_added)[1]) #stats.mannwhitneyu(df[0], df[2])[1])
    print("Complete vs added: ", stats.mannwhitneyu(complete, all_added)[1])  #stats.mannwhitneyu(df[1], df[2])[1])

    for i in range(0, 3):
        for j in range(0, 3):
            U1, p = stats.mannwhitneyu(all[i].ravel(), all[j].ravel())
            mwu[i, j] = p
    
    print()

    # sns_plot = sns.heatmap(mwu, annot=True, fmt=".2f", cmap="crest")
    # xtick_loc = sns_plot.get_xticks() 
    # sns_plot.set_xticks(ticks = xtick_loc, labels = ["Removed", "Complete", "Added"])
    # sns_plot.set_yticks(ticks = xtick_loc, labels = ["Removed", "Complete", "Added"])
    # plt.xlabel("Edges", fontsize=14)
    # plt.ylabel("Edges", fontsize=14)
    # title = f"TE differences for time-shift {t_shift} ms, simplex size {size},\n change in edges: {change} (" + r"$\mathit{p}$" + "-values)"
    # plt.title(title, fontsize=15)
    # sns_plot.figure.savefig(f"figures/te_added_removed/mwu_pvals_size_{size}_te_change_{change}_tshift_{t_shift}.pdf")
    # plt.close(sns_plot.figure)


    # binary = (mwu < 0.05).astype(np.int_)
    # sns_binary = sns.heatmap(binary, annot=True, fmt=".0f", cmap="crest_r")
    # xtick_loc = sns_binary.get_xticks() 
    # sns_binary.set_xticks(ticks = xtick_loc, labels = ["Removed", "Complete", "Added"])
    # sns_binary.set_yticks(ticks = xtick_loc, labels = ["Removed", "Complete", "Added"])
    # plt.xlabel("Edges", fontsize=14)
    # plt.ylabel("Edges", fontsize=14)
    # title = f"TE differences for time-shift {t_shift} ms,\n simplex size {size}, change in edges: {change} (" + r"$\mathit{p}$" + " < 0.05)"
    # plt.title(title, fontsize=15)
    # sns_binary.figure.savefig(f"figures/te_added_removed/mwu_binary_size_{size}_te_change_{change}_tshift_{t_shift}.pdf")
    # plt.close(sns_binary.figure)


change = 5
neurons = 8

# print()
# print("Change: ", change)
# print("Neurons: ", neurons)
# print()

for change in range(1, 5):
    print("Change: ", change)
    ac_mann_whitney_added_removed_exact(neurons, 15, change)