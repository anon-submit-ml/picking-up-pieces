import numpy as np
import pickle
import contextlib
import matplotlib
import bisect
import math
import matplotlib.pyplot as plt
from nas_201_api import NASBench201API as API
from scipy.stats import spearmanr, rankdata
from copy import deepcopy
import seaborn as sns
import sys
sys.path.insert(0, '../')

from baselines.exp_lists import *
import genotypes

ALGOS = ['darts', 'darts-','snas', 'gdas', 'dirichlet']
ALGOS2 = ['valid', 'synflow', 'jacob_cov', 'perturb']
cols = {'darts': 'tab:blue', 'snas': 'tab:pink',  'darts-': 'tab:purple', 'dirichlet': 'tab:olive', 'gdas': 'tab:brown'}

def plot_s1_eval(runs_list, dset, epochs=100, n_trials=3):
    seed2idx = {10:0, 11:1, 12:2} # redefine for seeds used
    api = API('../../nasbench201/NAS-Bench-201-v1_0-e61699.pth')
    accuracies = []
    for i in range(len(api)):
        with contextlib.redirect_stdout(None):
            info = api.query_by_index(i) #, hp='200') # add in commented portion if using latest version of NAS-Bench-201
        if dset == 'imagenet16-120':
            metrics = info.get_metrics('ImageNet16-120', 'ori-test', is_random=False)
        else:
            metrics = info.get_metrics(dset, 'ori-test', is_random=False)
        accuracies.append(metrics['accuracy'])
    s_terr = sorted([100-a for a in accuracies])
    threshs = [s_terr[i*625] for i in range(1, 25)]
    thresh_groups = [bisect.bisect(threshs, 100-a) for a in accuracies]
    sprob_mass = {alg : np.zeros((n_trials, 25)) for alg in ALGOS}
    perc_recall = {alg : np.zeros((n_trials, 25)) for alg in ALGOS}

    thresh_size = [sum(np.array(thresh_groups) == i) for i in range(25)]
    for algo, seed, stime in runs_list:
        save_dir = f"../experiments/nasbench201/{algo}-search-exp-{stime}-{seed}"
        if dset != 'cifar10':
            save_dir += "-"+dset
        if algo != 'gdas':
            save_dir += "-pc-1"
        aprob = np.load(save_dir+"/archprob.npy")
        aperf = np.load(save_dir+"/perfpred.npy")
        s_verr = sorted([100-a for a in aperf])
        v_threshs = [s_verr[i*625] for i in range(1, 25)]
        v_threshgr = [bisect.bisect(v_threshs, 100-a) for a in aperf]
        vthresh_size = [sum(np.array(v_threshgr) == i) for i in range(25)]
        perc_hit = np.zeros((25,), dtype=int)
        sprob = np.zeros((25,), dtype=float)
        for i in range(len(accuracies)):
            sprob[thresh_groups[i]] += aprob[i]
            for j in range(thresh_groups[i], 25):
                if j >= v_threshgr[i]:
                    perc_hit[j] += 1
        for i in range(25):
            sprob_mass[algo][seed2idx[seed], i] = sum(sprob[:i+1]) if i<24 else sum(sprob)
            perc_recall[algo][seed2idx[seed], i] = perc_hit[i]/sum(thresh_size[:i+1]) if i<24 else perc_hit[i]/sum(thresh_size)

    rand_base = [sum(thresh_size[:i])/15625 for i in range(1,25)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))
    for algo in ALGOS:
        y = np.mean(perc_recall[algo], axis=0)[:-1]
        ax1.plot(threshs, y, '-', c=cols[algo], label=algo.upper())
        ax1.fill_between(threshs, y-np.std(perc_recall[algo], axis=0)[:-1], y+np.std(perc_recall[algo], axis=0)[:-1], color=cols[algo], alpha=0.2)
    ax1.plot(threshs, rand_base, ':', c='black', label='random baseline')
    ax1.set_xlabel('Test error thresh')
    ax1.set_ylabel('Top arch ID rate')
    ax1.set_xscale('log')
    ax1.legend()

    for algo in ALGOS:
        y = np.mean(sprob_mass[algo], axis=0)[:-1]
        ax2.plot(threshs, y, '-', c=cols[algo], label=algo.upper())
        ax2.fill_between(threshs, y-np.std(sprob_mass[algo], axis=0)[:-1], y+np.std(sprob_mass[algo], axis=0)[:-1], color=cols[algo], alpha=0.2)
    ax2.plot(threshs, rand_base, ':', c='black', label='random baseline')
    ax2.set_xlabel('Test error thresh ')
    ax2.set_xscale('log')
    ax2.set_ylabel('Sampling probability mass')
    ax2.legend()

    fig.tight_layout()

    plt.savefig(f"plots/s1eval{epochs}_{dset}.pdf")


def plot_s2_eval(runs_list, dset, n_trials=3):
    sresults = {alg: np.zeros((n_trials, 4)) for alg in ALGOS2}

    base_id = {'none': 0, 'rsps': 1, 'rsps+': 2, 'rsps2': 1, 'none+':3}
    seed2idx = {10 : 0, 11 : 1, 12 : 2} # redefine for seeds used

    for algo, seed, stime in runs_list:
        if algo == 'rsps2':
            save_dir = f"../baselines/nasbench201/rsps+-search-exp-{stime}-{seed}"
            pfname = "/search22.pickle"
        elif algo == 'none+':
            save_dir = f"../baselines/nasbench201/none-search-exp-{stime}-{seed}"
            pfname = "/search22.pickle"
        else:
            save_dir = f"../baselines/nasbench201/{algo}-search-exp-{stime}-{seed}"
            pfname = "/search2.pickle"
        if dset != 'cifar10':
            save_dir += "-"+dset
        if algo != 'gdas':
            save_dir += "-pc-1"
        with open(save_dir+pfname, 'rb') as f:
            syn_res = []
            jac_res = []
            res = pickle.load(f)
            for x in res['measures']:
                syn_res.append(x['synflow'])
                jac_res.append(x['jacob_cov'])
            sresults['perturb'][seed2idx[seed], base_id[algo]] = res['perturb'][0]
            sresults['valid'][seed2idx[seed], base_id[algo]] = res['sampled_test'][np.argmax(res['valid'])]
            sresults['jacob_cov'][seed2idx[seed], base_id[algo]] = res['sampled_test'][np.nanargmax(jac_res)]
            sresults['synflow'][seed2idx[seed], base_id[algo]] = res['sampled_test'][np.nanargmax(syn_res)]
    x = np.arange(len(ALGOS2))
    width = 0.2

    sno_means = [np.mean(sresults[alg][:,3]-sresults[alg][:, 0])  for alg in ALGOS2]
    srs_means = [np.mean(sresults[alg][:,2]-sresults[alg][:, 1])  for alg in ALGOS2]
    wno_means = [np.mean(sresults[alg][:,1]-sresults[alg][:, 0])  for alg in ALGOS2]
    wrs_means = [np.mean(sresults[alg][:,2]-sresults[alg][:, 3])  for alg in ALGOS2]
    sno_std = [np.std(sresults[alg][:,3]-sresults[alg][:, 0])  for alg in ALGOS2]
    srs_std = [np.std(sresults[alg][:,2]-sresults[alg][:, 1])  for alg in ALGOS2]
    wno_std = [np.std(sresults[alg][:,1]-sresults[alg][:, 0])  for alg in ALGOS2]
    wrs_std = [np.std(sresults[alg][:,2]-sresults[alg][:, 3])  for alg in ALGOS2]


    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 1.5*width, sno_means, width, label='N_I - N_U', yerr=sno_std)
    rects2 = ax.bar(x - 0.5*width, srs_means, width, label='R_I - R_U', yerr=srs_std)
    rects3 = ax.bar(x + 0.5*width, wno_means, width, label='R_U - N_U', yerr=wno_std)
    rects4 = ax.bar(x + 1.5*width, wrs_means, width, label='R_I - N_I', yerr=wrs_std)

    ax.set_ylabel('Test Accuracy Gain')
    ax.set_xticks(x)
    ax.set_xticklabels(ALGOS2)
    ax.legend()

    plt.savefig(f"plots/s2eval_{dset}.pdf")


def pretty_print(runs_list, dset, epochs=100, n_trials=3):
    alg2idx = {'prune': 0, 'valid': 1, 'synflow': 2, 'jacob_cov': 3, 'perturb': 4}
    seed2idx = {10 : 0, 11 : 1, 12 : 2} # redefine for seeds used
    combo_algo_res = {alg : np.zeros((n_trials, 5)) for alg in ALGOS}
    for algo, seed, stime in runs_list:
        save_dir = f"../experiments/nasbench201/{algo}-search-exp-{stime}-{seed}"
        if dset != 'cifar10':
            save_dir += "-"+dset
        if algo != 'gdas':
            save_dir += "-pc-1"
        with open(save_dir+"/log.txt", "r") as f:
            done_train = False
            for line in f:
                lwords = line.strip().split(' ')
                if len(lwords) > 3 and (lwords[2] == "epoch" and int(lwords[3]) == epochs-1):
                    done_train = True
                if len(lwords) > 3 and (done_train and lwords[2] == dset and lwords[-2] == "test"):
                    prune_acc = float(lwords[-1])
                    combo_algo_res[algo][seed2idx[seed], 0] = 100-prune_acc
                elif len(lwords) > 3 and (done_train and dset == "imagenet16-120" and lwords[2] == "imagenet16" and lwords[-2] == "test"):
                    prune_acc = float(lwords[-1])
                    combo_algo_res[algo][seed2idx[seed], 0] = 100-prune_acc

        with open(save_dir+"/search2.pickle", 'rb') as f:
            syn_res = []
            jac_res = []
            res = pickle.load(f)
            for x in res['measures']:
                syn_res.append(x['synflow'])
                jac_res.append(x['jacob_cov'])
            combo_algo_res[algo][seed2idx[seed], 4] = 100-res['perturb'][0]
            combo_algo_res[algo][seed2idx[seed], 1] = 100-res['sampled_test'][np.argmax(res['valid'])]
            combo_algo_res[algo][seed2idx[seed], 3] = 100-res['sampled_test'][np.nanargmax(jac_res)]
            combo_algo_res[algo][seed2idx[seed], 2] = 100-res['sampled_test'][np.nanargmax(syn_res)]

    print("          |     prune     |    valid     |    synflow    |    jac_cov    |    perturb    |")
    for alg1 in ALGOS:
        print("{0:9s} | {1:.2f} +/- {2:.2f} |{3:.2f} +/- {4:.2f} | {5:.2f} +/- {6:.2f} | {7:.2f} +/- {8:.2f} | {9:.2f} +/- {10:.2f} | ".format(alg1, *[val for pair in zip(np.mean(combo_algo_res[alg1], axis=0), np.std(combo_algo_res_res[alg1], axis=0)) for val in pair]))

if __name__=="__main__":
    plot_s2_eval(BASELINE_RUNS_CIFAR10, 'cifar10')

