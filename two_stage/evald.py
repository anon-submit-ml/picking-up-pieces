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

def finish_stopped_runs(dset):
  dir_list = os.listdir(f"./experiments/{dset}/")
  eval_list = [x for x in dir_list if x[:4] == "eval"]
  to_finish = []
  for eval_dir in eval_list:
    eval_info = eval_dir.strip().split("-")
    arch_str = eval_info[4]
    s_time = "-".join(eval_info[2:4])
    with open(f"./experiments/{dset}/"+eval_dir+"/log.txt", "r") as f:
      last_ep = 0
      last_lr = 0
      for line in f:
        lwords = line.strip().split(' ')
        if len(lwords) > 3 and lwords[2] == "epoch":
          last_ep = int(lwords[3])
          last_lr = float(lwords[4])
    if last_ep < 599:
      to_finish.append((arch_str, s_time, last_ep, last_lr))
  return to_finish

if __name__=="__main__":
  #to_finish = finish_stopped_runs('cifar10')
  to_finish = [
    ('darts_10_cifar10_50_prune', '20211118-235850', 544, 5.150841e-04),
    ('darts_11_cifar10_50_prune', '20211118-225856', 576, 9.084881e-05),
    ('darts_11_cifar10_50_valid', '20211118-225856', 589, 1.741760e-05),
    ('darts_12_cifar10_50_prune', '20211119-113244', 569, 1.542172e-04),
    ('darts_12_cifar10_50_valid', '20211119-113244', 530, 8.072828e-04),
    ('darts_12_cifar10_50_perturb', '20211119-113244', 568, 1.646257e-04),
    ('snas_10_cifar10_50_prune', '20211119-113653', 571, 1.344159e-04),
    ('gdas_11_cifar10_50_prune', '20211118-225856', 548, 4.433622e-04),
    ('dirichlet_11_cifar10_50_valid', '20211118-225900', 593, 6.427702e-06)
  ]
  for a, t, ep, lr in to_finish:
    core_cmd = f"sbatch --export=ALL,arch='{a}',dset='cifar10',t='{t}',rep={ep},rlr={lr} evald2.sh"
    os.system(core_cmd)
