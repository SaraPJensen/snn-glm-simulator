import scipy.stats as stats
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import coo_matrix
import torch
import pingouin as pg
import sys
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append("./..")


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







def ate_mann_whitney(size, t_shift, change_dict, percentage):

    all_rel_ate = np.zeros((t_shift, 200, 3))

    for network in tqdm(range(0, 200)):
        complete_path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/{network}.pkl"
        added_path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/added_{change_dict[str(size)]}/{network}.pkl"
        removed_path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/removed_{change_dict[str(size)]}/{network}.pkl"

        X_complete, _ = get_data(complete_path)
        X_added, _ = get_data(added_path)
        X_removed, _ = get_data(removed_path)

        source_complete = X_complete[0, :]
        sink_complete = X_complete[-1, :]

        source_added = X_added[0, :]
        sink_added = X_added[-1, :]

        source_removed = X_removed[0, :]
        sink_removed = X_removed[-1, :]

        for t in range(0, t_shift):
            p_source = np.sum(source_complete)/len(source_complete)
            p_sink = np.sum(np.roll(sink_complete, -t))/len(sink_complete)
            p_source_and_sink = (np.sum(source_complete*np.roll(sink_complete, -t))/len(sink_complete))
            p_sink_given_source = p_source_and_sink/p_source
            p_sink_given_not_source = (p_sink - p_source_and_sink)/(1 - p_source)
            ATE_abs = p_sink_given_source - p_sink_given_not_source
            ATE_rel_complete = ATE_abs/p_sink_given_not_source   #The relative percentage-wise increase in the probability of the sink neuron firing given that the source neuron fired

            all_rel_ate[t, network, 1] = ATE_rel_complete

            p_source = np.sum(source_added)/len(source_added)
            p_sink = np.sum(np.roll(sink_added, -t))/len(sink_added)
            p_source_and_sink = (np.sum(source_added*np.roll(sink_added, -t))/len(sink_added))
            p_sink_given_source = p_source_and_sink/p_source
            p_sink_given_not_source = (p_sink - p_source_and_sink)/(1 - p_source)
            ATE_abs = p_sink_given_source - p_sink_given_not_source
            ATE_rel_added = ATE_abs/p_sink_given_not_source   #The relative percentage-wise increase in the probability of the sink neuron firing given that the source neuron fired

            all_rel_ate[t, network, 2] = ATE_rel_added

            p_source = np.sum(source_removed)/len(source_removed)
            p_sink = np.sum(np.roll(sink_removed, -t))/len(sink_removed)
            p_source_and_sink = (np.sum(source_removed*np.roll(sink_removed, -t))/len(sink_removed))
            p_sink_given_source = p_source_and_sink/p_source
            p_sink_given_not_source = (p_sink - p_source_and_sink)/(1 - p_source)
            ATE_abs = p_sink_given_source - p_sink_given_not_source
            ATE_rel_removed = ATE_abs/p_sink_given_not_source   #The relative percentage-wise increase in the probability of the sink neuron firing given that the source neuron fired

            all_rel_ate[t, network, 0] = ATE_rel_removed

            
    for t in range(0, t_shift):
        print("Time shift: ", t)
        df = pd.DataFrame(all_rel_ate[t, :, :])

        #print(df)

        print(np.mean(df[0]))
        print(np.mean(df[1]))
        print(np.mean(df[2]))
        
        mwu = np.zeros((3, 3))

        print("Removed vs complete: ", stats.mannwhitneyu(df[0], df[1])[1], stats.mannwhitneyu(all_rel_ate[t, :, 0], all_rel_ate[t, :, 1])[1])
        print("Removed vs added: ", stats.mannwhitneyu(df[0], df[2])[1], stats.mannwhitneyu(all_rel_ate[t, :, 0], all_rel_ate[t, :, 2])[1])
        print("Complete vs added: ", stats.mannwhitneyu(df[1], df[2])[1], stats.mannwhitneyu(all_rel_ate[t, :, 1], all_rel_ate[t, :, 2])[1])

        for i in range(0, 3):
            for j in range(0, 3):
                U1, p = stats.mannwhitneyu(df[i], df[j])
                #print("p: ", p)
                #exit()
                mwu[i, j] = p
        
        print()

        sns_plot = sns.heatmap(mwu, annot=True, fmt=".2f", cmap="crest")
        xtick_loc = sns_plot.get_xticks() 
        sns_plot.set_xticks(ticks = xtick_loc, labels = ["Removed", "Complete", "Added"])
        sns_plot.set_yticks(ticks = xtick_loc, labels = ["Removed", "Complete", "Added"])
        plt.xlabel("Edges", fontsize=14)
        plt.ylabel("Edges", fontsize=14)
        title = f"ATE differences for time-shift {t} ms, simplex size {size},\n {percentage} % change in edges (" + r"$\mathit{p}$" + "-values)"
        plt.title(title, fontsize=15)
        sns_plot.figure.savefig(f"figures/ate_added_removed/mwu_pvals_size_{size}_ate_{percentage}_change_tshift_{t}.pdf")
        plt.close(sns_plot.figure)


        binary = (mwu < 0.05).astype(np.int_)
        sns_binary = sns.heatmap(binary, annot=True, fmt=".0f", cmap="crest_r")
        xtick_loc = sns_binary.get_xticks() 
        sns_binary.set_xticks(ticks = xtick_loc, labels = ["Removed", "Complete", "Added"])
        sns_binary.set_yticks(ticks = xtick_loc, labels = ["Removed", "Complete", "Added"])
        plt.xlabel("Edges", fontsize=14)
        plt.ylabel("Edges", fontsize=14)
        title = f"Significant ATE differences for time-shift {t} ms,\n simplex size {size}, {percentage} % change in edges (" + r"$\mathit{p}$" + " < 0.05)"
        plt.title(title, fontsize=15)
        sns_binary.figure.savefig(f"figures/ate_added_removed/mwu_binary_size_{size}_ate_{percentage}_change_tshift_{t}.pdf")
        plt.close(sns_binary.figure)





change_10 = {'4': '1',
                '5': '1',
                '6': '2',
                '7': '2',
                '8': '3',
                '9': '4',
                '10': '5',
                'percentage': '10'}


change_15 = {'4': '1',
                '5': '2',
                '6': '2',
                '7': '3',
                '8': '4',
                '9': '5',
                '10': '7',
                'percentage': '15'}

# print("10 % change")
# for size in range(4, 10):
#     print("Size: ", size)
#     ate_mann_whitney(size, 6, change_10, 10)

#     print()

    

#     #ate_mann_whitney(size, 9, change_15, 15)

# print("15 % change")
# for size in range(4, 10):
#     print("Size: ", size)
#     ate_mann_whitney(size, 6, change_15, 15)

#     print()




def ate_mann_whitney_added_removed_cont(size, t_shift, max_change):

    all_added = []
    all_removed = []
    complete = []

    for network in tqdm(range(0, 200)):
        complete_path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/{network}.pkl"

        X_complete, _ = get_data(complete_path)
        source_complete = X_complete[0, :]
        sink_complete = X_complete[-1, :]

        p_source = np.sum(source_complete)/len(source_complete)
        p_sink = np.sum(np.roll(sink_complete, -t_shift))/len(sink_complete)
        p_source_and_sink = (np.sum(source_complete*np.roll(sink_complete, -t_shift))/len(sink_complete))
        p_sink_given_source = p_source_and_sink/p_source
        p_sink_given_not_source = (p_sink - p_source_and_sink)/(1 - p_source)
        ATE_abs = p_sink_given_source - p_sink_given_not_source
        ATE_rel = ATE_abs/p_sink_given_not_source   #The relative percentage-wise increase in the probability of the sink neuron firing given that the source neuron fired

        complete.append(ATE_rel)
                        
        for change in range(1, max_change+1):   
            added_path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/added_{change}/{network}.pkl"
            removed_path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/removed_{change}/{network}.pkl"

            X_added, _ = get_data(added_path)
            X_removed, _ = get_data(removed_path)

            source_added = X_added[0, :]
            sink_added = X_added[-1, :]

            source_removed = X_removed[0, :]
            sink_removed = X_removed[-1, :]

            p_source = np.sum(source_added)/len(source_added)
            p_sink = np.sum(np.roll(sink_added, -t_shift))/len(sink_added)
            p_source_and_sink = (np.sum(source_added*np.roll(sink_added, -t_shift))/len(sink_added))
            p_sink_given_source = p_source_and_sink/p_source
            p_sink_given_not_source = (p_sink - p_source_and_sink)/(1 - p_source)
            ATE_abs = p_sink_given_source - p_sink_given_not_source
            ATE_rel_added = ATE_abs/p_sink_given_not_source   #The relative percentage-wise increase in the probability of the sink neuron firing given that the source neuron fired

            p_source = np.sum(source_removed)/len(source_removed)
            p_sink = np.sum(np.roll(sink_removed, -t_shift))/len(sink_removed)
            p_source_and_sink = (np.sum(source_removed*np.roll(sink_removed, -t_shift))/len(sink_removed))
            p_sink_given_source = p_source_and_sink/p_source
            p_sink_given_not_source = (p_sink - p_source_and_sink)/(1 - p_source)
            ATE_abs = p_sink_given_source - p_sink_given_not_source
            ATE_rel_removed = ATE_abs/p_sink_given_not_source   #The relative percentage-wise increase in the probability of the sink neuron firing given that the source neuron fired


            all_added.append(ATE_rel_added)
            all_removed.append(ATE_rel_removed)
        
    print()
    print("Time shift: ", t_shift)

    all = [all_removed, complete, all_added]

    print(np.mean(all[0]), np.std(all[0]))
    print(np.mean(all[1]), np.std(all[1]))
    print(np.mean(all[2]), np.std(all[2]))
    
    mwu = np.zeros((3, 3))

    #print("Removed vs complete: ", stats.mannwhitneyu(all_removed, complete)[1])#stats.mannwhitneyu(df[0], df[1])[1])
    print("Removed vs complete: ", stats.mannwhitneyu(all[0], all[1])[1])
    #print("Removed vs added: ", stats.mannwhitneyu(all_removed, all_added)[1]) #stats.mannwhitneyu(all[0], all[2])[1])
    print("Removed vs added: ", stats.mannwhitneyu(all[0], all[2])[1])
    #print("Complete vs added: ", stats.mannwhitneyu(complete, all_added)[1])  #stats.mannwhitneyu(all[1], all[2])[1])
    print("Complete vs added: ", stats.mannwhitneyu(all[1], all[2])[1])

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
    title = f"ATE differences for time-shift {t_shift} ms, simplex size {size},\n change in edges (" + r"$\mathit{p}$" + "-values)"
    plt.title(title, fontsize=15)
    sns_plot.figure.savefig(f"figures/ate_added_removed/mwu_pvals_size_{size}_ate_change_tshift_{t_shift}.pdf")
    plt.close(sns_plot.figure)


    binary = (mwu < 0.05).astype(np.int_)
    sns_binary = sns.heatmap(binary, annot=True, fmt=".0f", cmap="crest_r")
    xtick_loc = sns_binary.get_xticks() 
    sns_binary.set_xticks(ticks = xtick_loc, labels = ["Removed", "Complete", "Added"])
    sns_binary.set_yticks(ticks = xtick_loc, labels = ["Removed", "Complete", "Added"])
    plt.xlabel("Edges", fontsize=14)
    plt.ylabel("Edges", fontsize=14)
    title = f"ATE differences for time-shift {t_shift} ms,\n simplex size {size}, change in edges (" + r"$\mathit{p}$" + " < 0.05)"
    plt.title(title, fontsize=15)
    sns_binary.figure.savefig(f"figures/ate_added_removed/mwu_binary_size_{size}_ate_change_tshift_{t_shift}.pdf")
    plt.close(sns_binary.figure)



# for t_shift in range(5):
#     ate_mann_whitney_added_removed_cont(8, t_shift, 5)






def ate_mann_whitney_added_removed_exact(size, t_shift, change):

    all_added = []
    all_removed = []
    complete = []

    for network in tqdm(range(0, 200)):
        complete_path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/{network}.pkl"

        X_complete, _ = get_data(complete_path)
        source_complete = X_complete[0, :]
        sink_complete = X_complete[-1, :]

        p_source = np.sum(source_complete)/len(source_complete)
        p_sink = np.sum(np.roll(sink_complete, -t_shift))/len(sink_complete)
        p_source_and_sink = (np.sum(source_complete*np.roll(sink_complete, -t_shift))/len(sink_complete))
        p_sink_given_source = p_source_and_sink/p_source
        p_sink_given_not_source = (p_sink - p_source_and_sink)/(1 - p_source)
        ATE_abs = p_sink_given_source - p_sink_given_not_source
        ATE_rel = ATE_abs/p_sink_given_not_source   #The relative percentage-wise increase in the probability of the sink neuron firing given that the source neuron fired

        complete.append(ATE_rel)
                        
  
        added_path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/added_{change}/{network}.pkl"
        removed_path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/removed_{change}/{network}.pkl"

        X_added, _ = get_data(added_path)
        X_removed, _ = get_data(removed_path)

        source_added = X_added[0, :]
        sink_added = X_added[-1, :]

        source_removed = X_removed[0, :]
        sink_removed = X_removed[-1, :]

        p_source = np.sum(source_added)/len(source_added)
        p_sink = np.sum(np.roll(sink_added, -t_shift))/len(sink_added)
        p_source_and_sink = (np.sum(source_added*np.roll(sink_added, -t_shift))/len(sink_added))
        p_sink_given_source = p_source_and_sink/p_source
        p_sink_given_not_source = (p_sink - p_source_and_sink)/(1 - p_source)
        ATE_abs = p_sink_given_source - p_sink_given_not_source
        ATE_rel_added = ATE_abs/p_sink_given_not_source   #The relative percentage-wise increase in the probability of the sink neuron firing given that the source neuron fired

        p_source = np.sum(source_removed)/len(source_removed)
        p_sink = np.sum(np.roll(sink_removed, -t_shift))/len(sink_removed)
        p_source_and_sink = (np.sum(source_removed*np.roll(sink_removed, -t_shift))/len(sink_removed))
        p_sink_given_source = p_source_and_sink/p_source
        p_sink_given_not_source = (p_sink - p_source_and_sink)/(1 - p_source)
        ATE_abs = p_sink_given_source - p_sink_given_not_source
        ATE_rel_removed = ATE_abs/p_sink_given_not_source   #The relative percentage-wise increase in the probability of the sink neuron firing given that the source neuron fired


        all_added.append(ATE_rel_added)
        all_removed.append(ATE_rel_removed)
        
    print()
    print("Time shift: ", t_shift)
    print()

    all = [all_removed, complete, all_added]

    print(np.mean(all[0]), np.std(all[0]))
    print(np.mean(all[1]), np.std(all[1]))
    print(np.mean(all[2]), np.std(all[2]))
    
    mwu = np.zeros((3, 3))

    #print("Removed vs complete: ", stats.mannwhitneyu(all_removed, complete)[1])#stats.mannwhitneyu(df[0], df[1])[1])
    print("Removed vs complete: ", stats.mannwhitneyu(all[0], all[1])[1])
    #print("Removed vs added: ", stats.mannwhitneyu(all_removed, all_added)[1]) #stats.mannwhitneyu(all[0], all[2])[1])
    print("Removed vs added: ", stats.mannwhitneyu(all[0], all[2])[1])
    #print("Complete vs added: ", stats.mannwhitneyu(complete, all_added)[1])  #stats.mannwhitneyu(all[1], all[2])[1])
    print("Complete vs added: ", stats.mannwhitneyu(all[1], all[2])[1])

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
    title = f"ATE differences for time-shift {t_shift} ms, simplex size {size},\n change in edges: {change} (" + r"$\mathit{p}$" + "-values)"
    plt.title(title, fontsize=15)
    sns_plot.figure.savefig(f"figures/ate_added_removed/mwu_pvals_size_{size}_ate_change_{change}_tshift_{t_shift}.pdf")
    plt.close(sns_plot.figure)

    binary = (mwu < 0.05).astype(np.int_)
    sns_binary = sns.heatmap(binary, annot=True, fmt=".0f", cmap="crest_r")
    xtick_loc = sns_binary.get_xticks() 
    sns_binary.set_xticks(ticks = xtick_loc, labels = ["Removed", "Complete", "Added"])
    sns_binary.set_yticks(ticks = xtick_loc, labels = ["Removed", "Complete", "Added"])
    plt.xlabel("Edges", fontsize=14)
    plt.ylabel("Edges", fontsize=14)
    title = f"ATE differences for time-shift {t_shift} ms,\n simplex size {size}, change in edges: {change} (" + r"$\mathit{p}$" + " < 0.05)"
    plt.title(title, fontsize=15)
    sns_binary.figure.savefig(f"figures/ate_added_removed/mwu_binary_size_{size}_ate_change_{change}_tshift_{t_shift}.pdf")
    plt.close(sns_binary.figure)


change = 4
print("Change in edges: ", change)

for t_shift in range(5):
    ate_mann_whitney_added_removed_exact(8, t_shift, change)





