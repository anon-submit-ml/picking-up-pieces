import os
from exp_lists import *

#for m, s, t in EXP_RUNS_CIFAR100:
#  core_cmd = f"sbatch --export=ALL,method='{m}',seed={s},dset='cifar100',t='{t}' recval.sh"
#  os.system(core_cmd)

#for m, s, t in EXP_RUNS_IMAGENET:
#  core_cmd = f"sbatch --export=ALL,method='{m}',seed={s},dset='imagenet16-120',t='{t}' recval.sh"
#  os.system(core_cmd)

#for s, t, ds, beta_decay in DARTSM_CIFAR:
#  core_cmd = f"sbatch --export=ALL,method='darts-',seed={s},dset='{ds}',t='{t}' recval.sh"
#  os.system(core_cmd)

#for s, t, beta_decay in DARTSM_IMAGENET:
#  core_cmd = f"sbatch --export=ALL,method='darts-',seed={s},dset='imagenet16-120',t='{t}' recval.sh"
#  os.system(core_cmd)

for m, s, t in EXP_RUNS_50_CIFAR100:
  core_cmd = f"sbatch --export=ALL,method='{m}',seed={s},dset='cifar100',t='{t}' recval.sh"
  os.system(core_cmd)

