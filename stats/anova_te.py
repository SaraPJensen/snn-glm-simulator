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





def te_normality(dim, t_shift, network_name):
    
    entropy = np.zeros((t_shift, 200))

    for network in tqdm(range(0, 200)):
        path = f"../data/{network_name}/cluster_sizes_[{dim}]_n_steps_200000/{network}.pkl"
        X, W0 = get_data(path)

        source = X[0, :]
        sink = X[-1, :]

        for t in range(t_shift):
            transfer_entropy = pi.transferentropy.transfer_entropy(source, np.roll(sink, -t), k = 20)
            entropy[t, network] = transfer_entropy

    print("Testing normality")

    for t in range(0, t_shift):
        print("Time shift: ", t)
        stat, p_normal = stats.normaltest(entropy[t, :])
        if p_normal < 0.05:
            print("Not normal")
        else:
            print("Normal")
        #print("Normality test: ", p_normal)

        stat, p_shapiro = stats.shapiro(entropy[t, :])
        #print("Shapiro test: ", p_shapiro)

        print()




# for dim in range(3, 15):
#     print("Size: ", dim)
#     te_normality(dim, 20, "simplex")






def te_variance(min_dim, max_dim, t_shift, network_name):
    dims = max_dim - min_dim +1

    entropy = np.zeros((t_shift, 200, dims))

    for dim_idx, dim in tqdm(enumerate(range(min_dim, max_dim+1))):

        for network in tqdm(range(0, 200)):
            path = f"../data/{network_name}/cluster_sizes_[{dim}]_n_steps_200000/{network}.pkl"
            X, W0 = get_data(path)

            source = X[0, :]
            sink = X[-1, :]

            for t in range(0, t_shift):
                transfer_entropy = pi.transferentropy.transfer_entropy(source, np.roll(sink, -t), k = 20)
                entropy[t, network, dim_idx] = transfer_entropy


    print("Testing homogeneity of variance")
    for t in range(0, t_shift):
        print("Time shift: ", t)
        df = pd.DataFrame(entropy[t, :, :])

        print(pg.homoscedasticity(df, method="levene"))

        print()

    

#te_variance(3, 10, 20, "simplex")






def te_normality_added_removed(size, changed_neurons, change_type, k = 20):
    print("Change type: ", change_type)
    
    for change_count in range(1, changed_neurons + 1):

        transfer_entropy = np.zeros((200))

        for network in tqdm(range(0, 200)):
            path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/{change_type}_{change_count}/{network}.pkl"
            X, W0 = get_data(path)

            source = X[0, :]
            sink = X[-1, :]

            transfer_entropy[network] = pi.transferentropy.transfer_entropy(source, sink, k = k)

        print("TE: ", transfer_entropy)

        #plt.scatter(np.arange(0, 200), transfer_entropy)

        plt.hist(transfer_entropy, 30)

        plt.savefig(f"../te_{change_type}_{change_count}.png")
        plt.close()
        #exit()


        print("Change count: ", change_count)
        stat, p_normal = stats.normaltest(transfer_entropy)
    
        if p_normal < 0.05:
            print("Not normal")
        else:
            print("Normal")
        stat, p_shapiro = stats.shapiro(transfer_entropy)
        print()


# te_normality_added_removed(8, 5, "added")
# te_normality_added_removed(8, 5, "removed")




def te_variance_added_removed(size, max_change, change_type, k = 20):
    print("Change type: ", change_type)

    transfer_entropy = np.zeros((200, max_change + 1))
    
    for change_count in range(0, max_change + 1):

        for network in tqdm(range(0, 200)):

            if change_count == 0:
                path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/{network}.pkl"
            else:
                path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/{change_type}_{change_count}/{network}.pkl"
            X, W0 = get_data(path)

            source = X[0, :]
            sink = X[-1, :]

            transfer_entropy[network, change_count] = pi.transferentropy.transfer_entropy(source, sink, k = k)

        #plt.scatter(np.arange(0, 200), transfer_entropy)

        # plt.hist(transfer_entropy, 30)

        # plt.savefig(f"../te_{change_type}_{change_count}.png")
        # plt.close()
        #exit()

    #print("Testing homogeneity of variance")

    df = pd.DataFrame(transfer_entropy)

    print(pg.homoscedasticity(df, method="levene"))

    print()
    print("Transposed")
    print(pg.homoscedasticity(pd.DataFrame(transfer_entropy.T), method="levene"))


# te_variance_added_removed(8, 5, "added")
# te_variance_added_removed(8, 5, "removed")



def te_variance_added_removed_combined(size, max_change, k = 20):

    transfer_entropy = np.zeros((200, (max_change*2) + 1))
    
    for change_count in range(0, max_change + 1):

        for network in tqdm(range(0, 200)):

            if change_count == 0:
                path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/{network}.pkl"
                X, W0 = get_data(path)
                source = X[0, :]
                sink = X[-1, :]

                transfer_entropy[network, max_change] = pi.transferentropy.transfer_entropy(source, sink, k = k)

            else:
                add_path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/added_{change_count}/{network}.pkl"
                remove_path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/removed_{change_count}/{network}.pkl"

                X_add, W0 = get_data(add_path)
                X_remove, W0 = get_data(remove_path)

                add_source = X_add[0, :]
                add_sink = X_add[-1, :]

                remove_source = X_remove[0, :]
                remove_sink = X[-1, :]

                transfer_entropy[network, max_change - change_count] = pi.transferentropy.transfer_entropy(remove_source, remove_sink, k = k)
                transfer_entropy[network, max_change + change_count] = pi.transferentropy.transfer_entropy(add_source, add_sink, k = k)

    df = pd.DataFrame(transfer_entropy)

    print(pg.homoscedasticity(df, method="levene"))


#te_variance_added_removed_combined(8, 5)






def te_kruskal_added_removed(size, max_change, change_type, k = 20):
    print("Change type: ", change_type)

    transfer_entropy = np.zeros((200, max_change + 1))
    
    for change_count in range(0, max_change + 1):

        for network in tqdm(range(0, 200)):

            if change_count == 0:
                path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/{network}.pkl"
            else:
                path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/{change_type}_{change_count}/{network}.pkl"
            X, W0 = get_data(path)

            source = X[0, :]
            sink = X[-1, :]

            transfer_entropy[network, change_count] = pi.transferentropy.transfer_entropy(source, sink, k = k)

    df = pd.DataFrame(transfer_entropy)

    print(stats.kruskal(df[0], df[1], df[2], df[3], df[4], df[5]))
    print()
        


# te_kruskal_added_removed(8, 5, "added")
# te_kruskal_added_removed(8, 5, "removed")







def count_type(dim, network_name):
    inhib = 0
    excit = 0

    for network in range(200):
        path = f"../data/{network_name}/cluster_sizes_[{dim}]_n_steps_200000/{network}.pkl"
        X, W0 = get_data(path)
        summing = torch.sum(W0[0, :])

        if summing < 0:
            inhib += 1
        else:
            excit += 1


    return inhib, excit




def ate_normality_inhib_excit(dim, t_shift, network_name):
    inhib, excit = count_type(dim, network_name)   #Differentiate between the networks where the source neuron is excitatory and inhibitory

    in_entropy = np.zeros((t_shift, inhib))
    ex_entropy = np.zeros((t_shift, excit))

    inhib = 0
    excit = 0

    for network in tqdm(range(0, 200)):
        path = f"../data/{network_name}/cluster_sizes_[{dim}]_n_steps_200000/{network}.pkl"
        X, W0 = get_data(path)

        source = X[0, :]
        sink = X[-1, :]

        summing = torch.sum(W0[0, :])

        for t in range(t_shift):
            transfer_entropy = pi.transferentropy.transfer_entropy(source, np.roll(sink, -t), k = 20)

            if summing < 0:
                in_entropy[t, inhib] = transfer_entropy

            else:
                ex_entropy[t, excit] = transfer_entropy

        if summing < 0:
            inhib += 1
        else:
            excit += 1
            

    print("Testing normality")
    for t in range(0, t_shift):
        print("Time shift: ", t)
        stat, ex_p_normal = stats.normaltest(ex_entropy[t, :])
        if ex_p_normal < 0.05:
            print("Excitatory not normal")
        else:
            print("Excitatory normal")

        stat, in_p_normal = stats.normaltest(in_entropy[t, :])
        if in_p_normal < 0.05:
            print("Inhibitory not normal")
        else:
            print("Inhibitory normal")

        print()




# for dim in range(3, 15):
#     print("Size: ", dim)
#     ate_normality_inhib_excit(dim, 20, "simplex")


def te_variance_inhib_excit(min_dim, max_dim, t_shift, network_name):
    dims = max_dim - min_dim +1

    inhib, excit = count_type(5, network_name)   #Differentiate between the networks where the source neuron is excitatory and inhibitory

    in_entropy = np.zeros((t_shift, 100, dims))
    ex_entropy = np.zeros((t_shift, 150, dims))


    for dim_idx, dim in tqdm(enumerate(range(min_dim, max_dim+1))):

        inhib = 0
        excit = 0

        for network in tqdm(range(0, 200)):
            path = f"../data/{network_name}/cluster_sizes_[{dim}]_n_steps_200000/{network}.pkl"
            X, W0 = get_data(path)

            source = X[0, :]
            sink = X[-1, :]

            summing = torch.sum(W0[0, :])

            for t in range(0, t_shift):
                transfer_entropy = pi.transferentropy.transfer_entropy(source, np.roll(sink, -t), k = 20)

                if summing < 0:
                    in_entropy[t, inhib, dim_idx] = transfer_entropy
            
                else:
                    ex_entropy[t, excit, dim_idx] = transfer_entropy


            if summing < 0:
                inhib += 1
            else:
                excit += 1



    print("Testing homogeneity of variance")
    for t in range(0, t_shift):
        print("Time shift: ", t)
        df_in = pd.DataFrame(in_entropy[t, :, :])
        df_ex = pd.DataFrame(ex_entropy[t, :, :])

        print("Excitatory")
        print(pg.homoscedasticity(df_ex, method="levene"))

        print()

        print("Inhibitory")
        print(pg.homoscedasticity(df_in, method="levene"))

        print()


#te_variance_inhib_excit(3, 10, 10, "simplex")





def kruskal_inhib_excit(min_dim, max_dim, t_shift, network_name):
    dims = max_dim - min_dim +1

    inhib, excit = count_type(5, network_name)   #Differentiate between the networks where the source neuron is excitatory and inhibitory

    in_all_rel_ate = np.zeros((t_shift, inhib, dims))
    ex_all_rel_ate = np.zeros((t_shift, excit, dims))


    for dim_idx, dim in tqdm(enumerate(range(min_dim, max_dim+1))):


        inhib = 0
        excit = 0

        for network in tqdm(range(0, 200)):
            path = f"../data/{network_name}/cluster_sizes_[{dim}]_n_steps_200000/{network}.pkl"
            X, W0 = get_data(path)

            source = X[0, :]
            sink = X[-1, :]

            summing = torch.sum(W0[0, :])

            for t in range(0, t_shift):
                p_source = np.sum(source)/len(source)
                p_sink = np.sum(np.roll(sink, -t))/len(sink)
                p_source_and_sink = (np.sum(source*np.roll(sink, -t))/len(sink))
                p_sink_given_source = p_source_and_sink/p_source
                p_sink_given_not_source = (p_sink - p_source_and_sink)/(1 - p_source)
                ATE_abs = p_sink_given_source - p_sink_given_not_source
                ATE_rel = ATE_abs/p_sink_given_not_source   #The relative percentage-wise increase in the probability of the sink neuron firing given that the source neuron fired

                if summing < 0:
                    in_all_rel_ate[t, inhib, dim_idx] = ATE_rel
            
                else:
                    ex_all_rel_ate[t, excit, dim_idx] = ATE_rel


    print("Testing Kruskal-Wallis")
    for t in range(0, t_shift):
        print("Time shift: ", t)
        df_in = pd.DataFrame(in_all_rel_ate[t, :, :])
        df_ex = pd.DataFrame(ex_all_rel_ate[t, :, :])

        print("Excitatory")
        print(stats.kruskal(df_ex[0], df_ex[1], df_ex[2], df_ex[3], df_ex[4], df_ex[5], df_ex[6], df_ex[7], df_ex[8], df_ex[9], df_ex[10], df_ex[11]))
        #print(pg.kruskal(df_ex))

        print()

        print("Inhibitory")
        print(stats.kruskal(df_in[0], df_in[1], df_in[2], df_in[3], df_in[4], df_in[5], df_in[6], df_in[7], df_in[8], df_in[9], df_in[10], df_in[11]))
        #print(pg.kruskal(df_in))

        print()


#kruskal_inhib_excit(3, 14, 20, "simplex")



def anova_te(min_dim, max_dim, t_shift):

    dims = max_dim - min_dim +1

    all_te = np.zeros((t_shift, 200, dims))

    for dim_idx, dim in tqdm(enumerate(range(min_dim, max_dim+1))):

        for network in tqdm(range(0, 200)):
            path = f"../data/simplex/cluster_sizes_[{dim}]_n_steps_200000/{network}.pkl"
            X, W0 = get_data(path)

            source = X[0, :]
            sink = X[-1, :]

            for t in range(0, t_shift):
                all_te[t, network, dim_idx] = pi.transferentropy.transfer_entropy(source, np.roll(sink, -t), k = 20)


    for t in range(0, t_shift):
        print("Time shift: ", t)
        df = pd.DataFrame(all_te[t, :, :])

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
        title = f"TE differences for \n time-shift {t} ms (" + r"$\mathit{p}$" + "-values)"
        plt.title(title, fontsize=16)

        sns_plot.figure.savefig(f"figures/transfer_entropy/mwu_pvals_te_dim_tshift_{t}.pdf")

        plt.close(sns_plot.figure)


        binary = (mwu < 0.05).astype(np.int_)
        sns_binary = sns.heatmap(binary, annot=True, fmt=".0f", cmap="crest_r")
        sns_binary.set_xticklabels(range(min_dim, max_dim+1))
        sns_binary.set_yticklabels(range(min_dim, max_dim+1))

        plt.xlabel("Neurons", fontsize=14)
        plt.ylabel("Neurons", fontsize=14)
        title = f"Significant TE differences for \n time-shift {t} ms (" + r"$\mathit{p}$" + " < 0.05)"
        plt.title(title, fontsize=16)

        sns_binary.figure.savefig(f"figures/transfer_entropy/mwu_binary_te_dim_tshift_{t}.pdf")
        plt.close(sns_binary.figure)

    

#anova_te(3, 9, 10)







def mann_whitney_inhib_excit(min_dim, max_dim, t_shift, network_name):
    dims = max_dim - min_dim +1

    in_all_te = []
    ex_all_te = []

    for dim_idx, dim in tqdm(enumerate(range(min_dim, max_dim+1))):

        inhib, excit = count_type(dim, network_name)

        in_te = np.zeros((t_shift, inhib))
        ex_te = np.zeros((t_shift, excit))

        inhib = 0
        excit = 0

        for network in tqdm(range(0, 200)):
            path = f"../data/simplex/cluster_sizes_[{dim}]_n_steps_200000/{network}.pkl"
            X, W0 = get_data(path)

            source = X[0, :]
            sink = X[-1, :]

            summing = torch.sum(W0[0, :])

            for t in range(0, t_shift):
                te = pi.transferentropy.transfer_entropy(source, np.roll(sink, -t), k = 20)

                if summing < 0:
                    in_te[t, inhib] = te
                else:
                    ex_te[t, excit] = te

            if summing < 0:
                inhib += 1
            else:
                excit += 1


        in_all_te.append(in_te)
        ex_all_te.append(ex_te)

    for t in range(0, t_shift):
        print("Time shift: ", t)

        ex_mwu = np.zeros((dims, dims))
        in_mwu = np.zeros((dims, dims))

        for i in range(0, dims):
            for j in range(0, dims):
                
                U1, in_p = stats.mannwhitneyu(in_all_te[i][t], in_all_te[j][t])
                in_mwu[i, j] = in_p

                U1, ex_p = stats.mannwhitneyu(ex_all_te[i][t], ex_all_te[j][t])
                ex_mwu[i, j] = ex_p


        in_sns_plot = sns.heatmap(in_mwu, annot=True, fmt=".2f", cmap="crest")
        in_sns_plot.set_xticklabels(range(min_dim, max_dim+1))
        in_sns_plot.set_yticklabels(range(min_dim, max_dim+1))
        plt.xlabel("Neurons", fontsize=14)
        plt.ylabel("Neurons", fontsize=14)
        title = f"TE differences for time-shift {t} ms \n (" + r"$\mathit{p}$" + "-values), inhibitory source neuron"
        plt.title(title, fontsize=16)
        in_sns_plot.figure.savefig(f"figures/transfer_entropy/mwu_pvals_te_inhib_dim_tshift_{t}.pdf")
        plt.close(in_sns_plot.figure)

        ex_sns_plot = sns.heatmap(ex_mwu, annot=True, fmt=".2f", cmap="crest")
        ex_sns_plot.set_xticklabels(range(min_dim, max_dim+1))
        ex_sns_plot.set_yticklabels(range(min_dim, max_dim+1))
        plt.xlabel("Neurons", fontsize=14)
        plt.ylabel("Neurons", fontsize=14)
        title = f"TE differences for time-shift {t} ms \n (" + r"$\mathit{p}$" + "-values), excitatory source neuron"
        plt.title(title, fontsize=16)
        ex_sns_plot.figure.savefig(f"figures/transfer_entropy/mwu_pvals_te_excit_dim_tshift_{t}.pdf")
        plt.close(ex_sns_plot.figure)

        in_binary = (in_mwu < 0.05).astype(np.int_)
        in_sns_binary = sns.heatmap(in_binary, annot=True, fmt=".0f", cmap="crest_r")
        in_sns_binary.set_xticklabels(range(min_dim, max_dim+1))
        in_sns_binary.set_yticklabels(range(min_dim, max_dim+1))
        plt.xlabel("Neurons", fontsize=14)
        plt.ylabel("Neurons", fontsize=14)
        title = f"Significant TE differences for time-shift {t} ms \n (" + r"$\mathit{p}$" + " < 0.05), inhibitory source neuron"
        plt.title(title, fontsize=16)
        in_sns_binary.figure.savefig(f"figures/transfer_entropy/mwu_binary_te_inhib_dim_tshift_{t}.pdf")
        plt.close(in_sns_binary.figure)


        ex_binary = (ex_mwu < 0.05).astype(np.int_)
        ex_sns_binary = sns.heatmap(ex_binary, annot=True, fmt=".0f", cmap="crest_r")
        ex_sns_binary.set_xticklabels(range(min_dim, max_dim+1))
        ex_sns_binary.set_yticklabels(range(min_dim, max_dim+1))
        plt.xlabel("Neurons", fontsize=14)
        plt.ylabel("Neurons", fontsize=14)
        title = f"Significant TE differences for time-shift {t} ms \n (" + r"$\mathit{p}$" + " < 0.05), excitatory source neuron"
        plt.title(title, fontsize=16)
        ex_sns_binary.figure.savefig(f"figures/transfer_entropy/mwu_binary_te_excit_dim_tshift_{t}.pdf")
        plt.close(ex_sns_binary.figure)



#mann_whitney_inhib_excit(3, 9, 10, "simplex")







def te_mann_whitney_added_removed(size, t_shift, max_change):

    all_added = []
    all_removed = []
    complete = []

    for network in tqdm(range(0, 200)):
        complete_path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/{network}.pkl"

        X_complete, _ = get_data(complete_path)
        source_complete = X_complete[0, :]
        sink_complete = X_complete[-1, :]

        complete.append(pi.transferentropy.transfer_entropy(source_complete, np.roll(sink_complete, -t_shift), k = 20))
                        
        for change in range(1, max_change+1):   
            added_path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/added_{change}/{network}.pkl"
            removed_path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/removed_{change}/{network}.pkl"

            X_added, _ = get_data(added_path)
            X_removed, _ = get_data(removed_path)

            source_added = X_added[0, :]
            sink_added = X_added[-1, :]

            source_removed = X_removed[0, :]
            sink_removed = X_removed[-1, :]

            all_added.append(pi.transferentropy.transfer_entropy(source_added, np.roll(sink_added, -t_shift), k = 20))
            all_removed.append(pi.transferentropy.transfer_entropy(source_removed, np.roll(sink_removed, -t_shift), k = 20))
        
            
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
    title = f"TE differences for time-shift {t_shift} ms, simplex size {size},\n change in edges (" + r"$\mathit{p}$" + "-values)"
    plt.title(title, fontsize=15)
    sns_plot.figure.savefig(f"figures/te_added_removed/mwu_pvals_size_{size}_te_change_tshift_{t_shift}.pdf")
    plt.close(sns_plot.figure)


    binary = (mwu < 0.05).astype(np.int_)
    sns_binary = sns.heatmap(binary, annot=True, fmt=".0f", cmap="crest_r")
    xtick_loc = sns_binary.get_xticks() 
    sns_binary.set_xticks(ticks = xtick_loc, labels = ["Removed", "Complete", "Added"])
    sns_binary.set_yticks(ticks = xtick_loc, labels = ["Removed", "Complete", "Added"])
    plt.xlabel("Edges", fontsize=14)
    plt.ylabel("Edges", fontsize=14)
    title = f"TE differences for time-shift {t_shift} ms,\n simplex size {size}, change in edges (" + r"$\mathit{p}$" + " < 0.05)"
    plt.title(title, fontsize=15)
    sns_binary.figure.savefig(f"figures/te_added_removed/mwu_binary_size_{size}_te_change_tshift_{t_shift}.pdf")
    plt.close(sns_binary.figure)


# for t_shift in range(10):
#     te_mann_whitney_added_removed(8, t_shift, 5)





def te_mann_whitney_added_removed_exact(size, t_shift, change):

    all_added = []
    all_removed = []
    complete = []

    for network in tqdm(range(0, 200)):
        complete_path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/{network}.pkl"

        X_complete, _ = get_data(complete_path)
        source_complete = X_complete[0, :]
        sink_complete = X_complete[-1, :]

        complete.append(pi.transferentropy.transfer_entropy(source_complete, np.roll(sink_complete, -t_shift), k = 20))
                        
        added_path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/added_{change}/{network}.pkl"
        removed_path = f"../data/simplex/cluster_sizes_[{size}]_n_steps_200000/removed_{change}/{network}.pkl"

        X_added, _ = get_data(added_path)
        X_removed, _ = get_data(removed_path)

        source_added = X_added[0, :]
        sink_added = X_added[-1, :]

        source_removed = X_removed[0, :]
        sink_removed = X_removed[-1, :]

        all_added.append(pi.transferentropy.transfer_entropy(source_added, np.roll(sink_added, -t_shift), k = 20))
        all_removed.append(pi.transferentropy.transfer_entropy(source_removed, np.roll(sink_removed, -t_shift), k = 20))
    
            
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

    print("Kruskal")
    print("Removed vs complete: ", stats.kruskal(all_removed, complete)[1])#stats.mannwhitneyu(df[0], df[1])[1])
    print("Removed vs added: ", stats.kruskal(all_removed, all_added)[1]) #stats.mannwhitneyu(df[0], df[2])[1])
    print("Complete vs added: ", stats.kruskal(complete, all_added)[1])  #stats.mannwhitneyu(df[1], df[2])[1])

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

print()
print("Change: ", change)
print("Neurons: ", neurons)
print()

for t_shift in range(5):
    te_mann_whitney_added_removed_exact(neurons, t_shift, change)