import os
from exp_lists import *
import pickle
import sys
sys.path.append('DARTS-space/')
import genotypes

def make_genotypes(exp_list, dset, epochs):
  outLines = []
  for algo, seed, stime in exp_list:
    save_dir = f"./experiments/{dset}/search-exp-{stime}-{seed}-init_channels-36-{algo}--init_pc-2"
    with open(save_dir+"/log.txt", "r") as f:
      done_train = False
      prune_geno = ""
      for line in f:
        lwords = line.strip().split(' ')
        if len(lwords) > 3 and (lwords[2] == "epoch" and int(lwords[3]) == epochs-1):
          done_train = True
        if len(lwords) > 3 and (done_train and lwords[2] == "genotype"):
          prune_geno = " ".join(lwords[4:])
    with open(save_dir+"/search2.pickle", "rb") as f:
      syn_res = []
      jac_res = []
      res = pickle.load(f)
      for x, gen in res['measures']:
        syn_res.append((x['synflow'], gen))
        jac_res.append((x['jacob_cov'], gen))
    perturb_geno = res['perturb'][0]
    val, valid_geno = sorted(res['valid'], key = lambda x: x[0])[-1]
    syn, synflow_geno = sorted(syn_res, key = lambda x: x[0])[-1]
    jac, jaccov_geno = sorted(jac_res, key = lambda x: x[0])[-1]

    if algo == "darts-":
      algnm = "dartsm"
    else:
      algnm = algo
    outLines.append(f"{algnm}_{seed}_{dset}_{epochs}_prune = "+str(prune_geno)+"\n")
    outLines.append(f"{algnm}_{seed}_{dset}_{epochs}_valid = "+str(valid_geno)+"\n")
    outLines.append(f"{algnm}_{seed}_{dset}_{epochs}_synflow = "+str(synflow_geno)+"\n")
    outLines.append(f"{algnm}_{seed}_{dset}_{epochs}_jacob_cov = "+str(jaccov_geno)+"\n")
    outLines.append(f"{algnm}_{seed}_{dset}_{epochs}_perturb = "+str(perturb_geno)+"\n\n")
  return outLines


if __name__=="__main__":
    exp_list = [
        ('algo1', seed1, 'stime1'),
        ('algo1', seed2, 'stime2'),
        ('algo2', seed1, 'stime3'),
        ('algo2', seed2, 'stime4')
    ]
    genotypes = make_genotypes(exp_list, 'cifar10', 50)

    with open('./DARTS-space/genotypes.py', 'a') as f:
        f.writelines(genotypes)
