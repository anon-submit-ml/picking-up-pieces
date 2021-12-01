import os
import numpy as np
import pickle
import contextlib
import matplotlib
import math
import matplotlib.pyplot as plt
from copy import deepcopy


ALGOS = ['darts', 'darts-','snas', 'gdas', 'dirichlet']
ALGOS2 = ['valid', 'synflow', 'jacob_cov', 'perturb']

def pretty_print(dset, n_trials=3):
    # NOTE : This function assumes all evaluated models for the given data set of a particular algo
    # represent trials with different random seeds but otherwised identical hyperparameters
    alg2idx = {'prune': 0, 'valid': 1, 'synflow': 2, 'jacob_cov': 3, 'perturb': 4}
    seed2idx = {10: 0, 11: 1, 12: 2} # redefine for seeds used
    full_res = {alg : np.zeros((n_trials,5)) for alg in ALGOS}
    dir_list = os.listdir(f"../experiments/{dset}/")
    eval_list = [x for x in dir_list if x[:4] == "eval"]
    for eval_dir in eval_list:
        eval_info = eval_dir.strip().split("-")[4]
        eval_info = eval_info.split("_")
        if eval_info[0] == "dartsm":
            algo1 = "darts-"
        else:
            algo1 = eval_info[0]
        seed = int(eval_info[1])
        if eval_info[4] == "jacob":
            algo2 = "jacob_cov"
        else:
            algo2 = eval_info[4]
        with open(f"../experiments/{dset}/"+eval_dir+"/log.txt", "r") as f:
            best_acc = 0
            for line in f:
                lwords = line.strip().split(' ')
                if len(lwords) > 3 and lwords[2] == "valid_acc":
                    cbest = float(lwords[-1])
                    if cbest > best_acc:
                        best_acc = cbest
        full_res[algo1][seed2idx[seed], alg2idx[algo2]] = 100-best_acc
    print("          |     prune     |    valid     |    synflow    |    jac_cov    |    perturb    |")
    for alg1 in ALGOS:
        print("{0:9s} | {1:.2f} +/- {2:.2f} |{3:.2f} +/- {4:.2f} | {5:.2f} +/- {6:.2f} | {7:.2f} +/- {8:.2f} | {9:.2f} +/- {10:.2f} | ".format(alg1, *[val for pair in zip(np.mean(full_res[alg1], axis=0), np.std(full_res[alg1], axis=0)) for val in pair]))
if __name__=="__main__":
    pretty_print("cifar10")

