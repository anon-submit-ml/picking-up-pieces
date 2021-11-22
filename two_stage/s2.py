import os
from exp_lists import *

for m, s, t in BASELINE_RUNS_CIFAR10:
  core_cmd = f"sbatch --export=ALL,method='{m}',seed={s},dset='cifar10',t='{t}' s2.sh"
  os.system(core_cmd)

for m, s, t in BASELINE_RUNS_CIFAR100:
  core_cmd = f"sbatch --export=ALL,method='{m}',seed={s},dset='cifar100',t='{t}' s2.sh"
  os.system(core_cmd)

for m, s, t in BASELINE_RUNS_IMAGENET:
  core_cmd = f"sbatch --export=ALL,method='{m}',seed={s},dset='imagenet16-120',t='{t}' s2.sh"
  os.system(core_cmd)

#for s, t, ds, beta_decay in DARTSM_CIFAR:
#  core_cmd = f"sbatch --export=ALL,method='darts-',seed={s},dset='{ds}',t='{t}' s2.sh"
#  os.system(core_cmd)

#for s, t, beta_decay in DARTSM_IMAGENET:
#  core_cmd = f"sbatch --export=ALL,method='darts-',seed={s},dset='imagenet16-120',t='{t}' s2.sh"
#  os.system(core_cmd)

#for m, s, t, dset in BASELINE_2:
#  core_cmd = f"sbatch --export=ALL,method='{m}',seed={s},dset='{dset}',t='{t}' s2.sh"
#  os.system(core_cmd)

#for m, s, t, dset in RSPS2:
#  core_cmd = f"sbatch --export=ALL,method='{m}',seed={s},dset='{dset}',t='{t}' s2.sh"
#  os.system(core_cmd)

#for m, s, t in EXP_RUNS_100_DS_CIFAR10:
#  core_cmd = f"sbatch --export=ALL,method='{m}',seed={s},dset='cifar10',t='{t}' s2ds.sh"
#  os.system(core_cmd)

#for d, s, t in DARTSM_DS_NO_DECAY:
#  core_cmd = f"sbatch --export=ALL,method='darts-',seed={s},dset='{d}',t='{t}' s2ds.sh"
#  os.system(core_cmd)

#for m, s, t in EXP_RUNS_50_CIFAR100:
#  core_cmd = f"sbatch --export=ALL,method='{m}',seed={s},dset='cifar100',t='{t}' s2.sh"
#  os.system(core_cmd)

