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





def ate_normality(dim, t_shift, network_name):
    all_rel_ate = np.zeros((t_shift, 200))
    all_abs_ate = np.zeros((t_shift, 200))

    all_p_source = np.zeros((t_shift, 200))
    all_p_sink = np.zeros((t_shift, 200))
    all_p_sink_given_source = np.zeros((t_shift, 200))
    all_sink_given_not_source = np.zeros((t_shift, 200))


    for network in tqdm(range(0, 200)):
        path = f"../data/{network_name}/cluster_sizes_[{dim}]_n_steps_200000/{network}.pkl"
        X, W0 = get_data(path)

        source = X[0, :]
        sink = X[-1, :]

        for t in range(0, t_shift):
            p_source = np.sum(source)/len(source)
            p_sink = np.sum(np.roll(sink, -t))/len(sink)
            p_source_and_sink = (np.sum(source*np.roll(sink, -t))/len(sink))
            p_sink_given_source = p_source_and_sink/p_source
            p_sink_given_not_source = (p_sink - p_source_and_sink)/(1 - p_source)
            
            ATE_abs = p_sink_given_source - p_sink_given_not_source
            ATE_rel = ATE_abs/p_sink_given_not_source   #The relative percentage-wise increase in the probability of the sink neuron firing given that the source neuron fired

            all_rel_ate[t, network] = ATE_rel
            all_abs_ate[t, network] = ATE_abs

            all_p_source[t, network] = p_source
            all_p_sink[t, network] = p_sink
            all_p_sink_given_source[t, network] = p_sink_given_source
            all_sink_given_not_source[t, network] = p_sink_given_not_source

    print("Testing normality")
    for t in range(0, t_shift):
        print("Time shift: ", t)
        stat, p_normal = stats.normaltest(all_rel_ate[t, :])
        if p_normal < 0.05:
            print("Not normal")
        else:
            print("Normal")
        #print("Normality test: ", p_normal)

        stat, p_shapiro = stats.shapiro(all_rel_ate[t, :])
        #print("Shapiro test: ", p_shapiro)

        print()




# for dim in range(3, 15):
#     print("Size: ", dim)
#     ate_normality(dim, 20, "simplex")




def ate_variance(min_dim, max_dim, t_shift, network_name):
    dims = max_dim - min_dim +1

    all_rel_ate = np.zeros((t_shift, 200, dims))

    for dim_idx, dim in tqdm(enumerate(range(min_dim, max_dim+1))):

        for network in tqdm(range(0, 200)):
            path = f"../data/{network_name}/cluster_sizes_[{dim}]_n_steps_200000/{network}.pkl"
            X, W0 = get_data(path)

            source = X[0, :]
            sink = X[-1, :]

            for t in range(0, t_shift):
                p_source = np.sum(source)/len(source)
                p_sink = np.sum(np.roll(sink, -t))/len(sink)
                p_source_and_sink = (np.sum(source*np.roll(sink, -t))/len(sink))
                p_sink_given_source = p_source_and_sink/p_source
                p_sink_given_not_source = (p_sink - p_source_and_sink)/(1 - p_source)
                ATE_abs = p_sink_given_source - p_sink_given_not_source
                ATE_rel = ATE_abs/p_sink_given_not_source   #The relative percentage-wise increase in the probability of the sink neuron firing given that the source neuron fired

                all_rel_ate[t, network, dim_idx] = ATE_rel


    print("Testing homogeneity of variance")
    for t in range(0, t_shift):
        print("Time shift: ", t)
        df = pd.DataFrame(all_rel_ate[t, :, :])

        print(pg.homoscedasticity(df, method="levene"))

        print()

    

#ate_variance(3, 10, 10, "simplex")







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

    in_all_rel_ate = np.zeros((t_shift, inhib))
    ex_all_rel_ate = np.zeros((t_shift, excit))

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
                in_all_rel_ate[t, inhib] = ATE_rel
            
            else:
                ex_all_rel_ate[t, excit] = ATE_rel

        if summing < 0:
            inhib += 1
        else:
            excit += 1


    print("Testing normality")
    for t in range(0, t_shift):
        print("Time shift: ", t)
        stat, ex_p_normal = stats.normaltest(ex_all_rel_ate[t, :])
        if ex_p_normal < 0.05:
            print("Excitatory not normal")
        else:
            print("Excitatory normal")

        stat, in_p_normal = stats.normaltest(in_all_rel_ate[t, :])
        if in_p_normal < 0.05:
            print("Inhibitory not normal")
        else:
            print("Inhibitory normal")

        print()



# for dim in range(3, 15):
#     print("Size: ", dim)
#     ate_normality_inhib_excit(dim, 20, "simplex")


def ate_variance_inhib_excit(min_dim, max_dim, t_shift, network_name):
    dims = max_dim - min_dim +1

    inhib, excit = count_type(5, network_name)   #Differentiate between the networks where the source neuron is excitatory and inhibitory

    in_all_rel_ate = np.zeros((t_shift, 100, dims))
    ex_all_rel_ate = np.zeros((t_shift, 150, dims))


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

            
            if summing < 0:
                inhib += 1
            else:
                excit += 1

        print("Excitatory: ", excit)


    print("Testing homogeneity of variance")
    for t in range(0, t_shift):
        print("Time shift: ", t)
        df_in = pd.DataFrame(in_all_rel_ate[t, :, :])
        df_ex = pd.DataFrame(ex_all_rel_ate[t, :, :])

        print("Excitatory")
        print(pg.homoscedasticity(df_ex, method="levene"))

        print()

        print("Inhibitory")
        print(pg.homoscedasticity(df_in, method="levene"))

        print()


#ate_variance_inhib_excit(3, 14, 10, "simplex")



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



def anova_ate(min_dim, max_dim, t_shift, network_name):

    dims = max_dim - min_dim +1

    all_rel_ate = np.zeros((t_shift, 200, dims))

    for dim_idx, dim in tqdm(enumerate(range(min_dim, max_dim+1))):

        for network in tqdm(range(0, 200)):
            path = f"../data/{network_name}/cluster_sizes_[{dim}]_n_steps_200000/{network}.pkl"
            X, W0 = get_data(path)

            source = X[0, :]
            sink = X[-1, :]

            for t in range(0, t_shift):
                p_source = np.sum(source)/len(source)
                p_sink = np.sum(np.roll(sink, -t))/len(sink)
                p_source_and_sink = (np.sum(source*np.roll(sink, -t))/len(sink))
                p_sink_given_source = p_source_and_sink/p_source
                p_sink_given_not_source = (p_sink - p_source_and_sink)/(1 - p_source)
                ATE_abs = p_sink_given_source - p_sink_given_not_source
                ATE_rel = ATE_abs/p_sink_given_not_source   #The relative percentage-wise increase in the probability of the sink neuron firing given that the source neuron fired

                all_rel_ate[t, network, dim_idx] = ATE_rel


    for t in range(0, t_shift):
        print("Time shift: ", t)
        df = pd.DataFrame(all_rel_ate[t, :, :])

        print(stats.f_oneway(df[0], df[1], df[2], df[3], df[4], df[5], df[6], df[7], df[8], df[9], df[10], df[11]))
        print()

    

#anova_ate(3, 14, 10, "simplex")



def ate_mann_whitney(min_dim, max_dim, t_shift, network_name):

    dims = max_dim - min_dim +1

    all_rel_ate = np.zeros((t_shift, 200, dims))

    for dim_idx, dim in tqdm(enumerate(range(min_dim, max_dim+1))):

        for network in tqdm(range(0, 200)):
            path = f"../data/{network_name}/cluster_sizes_[{dim}]_n_steps_200000/{network}.pkl"
            X, W0 = get_data(path)

            source = X[0, :]
            sink = X[-1, :]

            for t in range(0, t_shift):
                p_source = np.sum(source)/len(source)
                p_sink = np.sum(np.roll(sink, -t))/len(sink)
                p_source_and_sink = (np.sum(source*np.roll(sink, -t))/len(sink))
                p_sink_given_source = p_source_and_sink/p_source
                p_sink_given_not_source = (p_sink - p_source_and_sink)/(1 - p_source)
                ATE_abs = p_sink_given_source - p_sink_given_not_source
                ATE_rel = ATE_abs/p_sink_given_not_source   #The relative percentage-wise increase in the probability of the sink neuron firing given that the source neuron fired

                all_rel_ate[t, network, dim_idx] = ATE_rel

            
    for t in range(0, t_shift):
        print("Time shift: ", t)
        df = pd.DataFrame(all_rel_ate[t, :, :])

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
        title = f"ATE differences for \n time-shift {t} ms (" + r"$\mathit{p}$" + "-values)"
        plt.title(title, fontsize=16)

        sns_plot.figure.savefig(f"figures/ate/mwu_pvals_ate_dim_tshift_{t}.pdf")

        
        # sns_plot.set(xlabel='Neurons', 
        #              ylabel='Neurons',
        #              title=f"ATE differences at time shift {t} ms (p-values)")
        
        plt.close(sns_plot.figure)


        binary = (mwu < 0.05).astype(np.int_)
        sns_binary = sns.heatmap(binary, annot=True, fmt=".0f", cmap="crest_r")
        sns_binary.set_xticklabels(range(min_dim, max_dim+1))
        sns_binary.set_yticklabels(range(min_dim, max_dim+1))

        plt.xlabel("Neurons", fontsize=14)
        plt.ylabel("Neurons", fontsize=14)
        title = f"Significant ATE differences for \n time-shift {t} ms (" + r"$\mathit{p}$" + " < 0.05)"
        plt.title(title, fontsize=16)

        # sns_binary.set(xlabel='Neurons', 
        #                ylabel='Neurons',
        #                title=f"Significant ATE differences at time shift {t} ms (p < 0.05)")
        
        sns_binary.figure.savefig(f"figures/ate/mwu_binary_ate_dim_tshift_{t}.pdf")
        plt.close(sns_binary.figure)
        # print(mwu)
        # print()

        # print(binary)

        # print()




#ate_mann_whitney(3, 9, 10, "simplex")




def mann_whitney_inhib_excit(min_dim, max_dim, t_shift, network_name):
    dims = max_dim - min_dim +1

    in_all_rel_ate = []
    ex_all_rel_ate = []

    for dim_idx, dim in tqdm(enumerate(range(min_dim, max_dim+1))):

        inhib, excit = count_type(dim, network_name)

        # print("Inhib: ", inhib)
        # print("Excit: ", excit)

        in_rel_ate = np.zeros((t_shift, inhib))
        ex_rel_ate = np.zeros((t_shift, excit))

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
                    in_rel_ate[t, inhib] = ATE_rel
            
                else:
                    ex_rel_ate[t, excit] = ATE_rel

            if summing < 0:
                inhib += 1
            else:
                excit += 1


        in_all_rel_ate.append(in_rel_ate)
        ex_all_rel_ate.append(ex_rel_ate)
            #List of numpy arrays, each array is a matrix of relative ATEs for a given time shift
    
    # for element in in_all_rel_ate:
    #     print(element.shape)
    
    # exit()

    for t in range(0, t_shift):
        print("Time shift: ", t)

        ex_mwu = np.zeros((dims, dims))
        in_mwu = np.zeros((dims, dims))

        for i in range(0, dims):
            for j in range(0, dims):
                
                U1, in_p = stats.mannwhitneyu(in_all_rel_ate[i][t], in_all_rel_ate[j][t])
                in_mwu[i, j] = in_p

                #print(in_all_rel_ate[i][t].shape, in_all_rel_ate[j][t].shape)
                U1, ex_p = stats.mannwhitneyu(ex_all_rel_ate[i][t], ex_all_rel_ate[j][t])
                ex_mwu[i, j] = ex_p

                #print(ex_all_rel_ate[i][t].shape, ex_all_rel_ate[j][t].shape)

        in_sns_plot = sns.heatmap(in_mwu, annot=True, fmt=".2f", cmap="crest")
        in_sns_plot.set_xticklabels(range(min_dim, max_dim+1))
        in_sns_plot.set_yticklabels(range(min_dim, max_dim+1))
        plt.xlabel("Neurons", fontsize=14)
        plt.ylabel("Neurons", fontsize=14)
        title = f"ATE differences for time-shift {t} ms \n (" + r"$\mathit{p}$" + "-values), inhibitory source neuron"
        plt.title(title, fontsize=16)
        in_sns_plot.figure.savefig(f"figures/ate/mwu_pvals_ate_inhib_dim_tshift_{t}.pdf")
        plt.close(in_sns_plot.figure)

        ex_sns_plot = sns.heatmap(ex_mwu, annot=True, fmt=".2f", cmap="crest")
        ex_sns_plot.set_xticklabels(range(min_dim, max_dim+1))
        ex_sns_plot.set_yticklabels(range(min_dim, max_dim+1))
        plt.xlabel("Neurons", fontsize=14)
        plt.ylabel("Neurons", fontsize=14)
        title = f"ATE differences for time-shift {t} ms \n (" + r"$\mathit{p}$" + "-values), excitatory source neuron"
        plt.title(title, fontsize=16)
        ex_sns_plot.figure.savefig(f"figures/ate/mwu_pvals_ate_excit_dim_tshift_{t}.pdf")
        plt.close(ex_sns_plot.figure)

        in_binary = (in_mwu < 0.05).astype(np.int_)
        in_sns_binary = sns.heatmap(in_binary, annot=True, fmt=".0f", cmap="crest_r")
        in_sns_binary.set_xticklabels(range(min_dim, max_dim+1))
        in_sns_binary.set_yticklabels(range(min_dim, max_dim+1))
        plt.xlabel("Neurons", fontsize=14)
        plt.ylabel("Neurons", fontsize=14)
        title = f"Significant ATE differences for time-shift {t} ms \n (" + r"$\mathit{p}$" + " < 0.05), inhibitory source neuron"
        plt.title(title, fontsize=16)
        in_sns_binary.figure.savefig(f"figures/ate/mwu_binary_ate_inhib_dim_tshift_{t}.pdf")
        plt.close(in_sns_binary.figure)


        ex_binary = (ex_mwu < 0.05).astype(np.int_)
        ex_sns_binary = sns.heatmap(ex_binary, annot=True, fmt=".0f", cmap="crest_r")
        ex_sns_binary.set_xticklabels(range(min_dim, max_dim+1))
        ex_sns_binary.set_yticklabels(range(min_dim, max_dim+1))
        plt.xlabel("Neurons", fontsize=14)
        plt.ylabel("Neurons", fontsize=14)
        title = f"Significant ATE differences for time-shift {t} ms \n (" + r"$\mathit{p}$" + " < 0.05), excitatory source neuron"
        plt.title(title, fontsize=16)
        ex_sns_binary.figure.savefig(f"figures/ate/mwu_binary_ate_excit_dim_tshift_{t}.pdf")
        plt.close(ex_sns_binary.figure)



#mann_whitney_inhib_excit(3, 9, 10, "simplex")




