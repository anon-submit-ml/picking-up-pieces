# DrNAS



## About
Code originally accompanying the paper  
[ICLR'2021](https://iclr.cc/): ***DrNAS: Dirichlet Neural Architecture Search*** [paper](https://openreview.net/forum?id=9FWas6YbmB3)  
Xiangning Chen*, Ruochen Wang*, Minhao Cheng*, Xiaocheng Tang, Cho-Jui Hsieh

This code is based on the implementation of [NAS-Bench-201](https://github.com/D-X-Y/AutoDL-Projects) and [PC-DARTS](https://github.com/yuhuixu1993/PC-DARTS).


## Usage


### Architecture Search

**Stage-1 Search on NAS-Bench-201 Space: (3 datasets to choose from)**

* Data preparation: Please first download the 201 benchmark file and prepare the api follow [this repository](https://github.com/D-X-Y/NAS-Bench-201).

* ```cd 201-space && python train_search.py``` (default syntax)

* ```cd 201-space && python train_search.py --method {algo} --seed {seed} --dataset {dset} --epochs {epochs}``` (set args which varied in our experiments)
  - algo is in [rsps/rsps+/gdas/snas/darts/darts-/blank/dirichlet], seed and epochs are integers, dset is in [cifar10/cifar100/imagenet16-120]

* ```With Progressively Pruning: cd 201-space && python train_search_progressive.py``` (not implemented for RSPS, GDAS, DARTS-)

**Stage-2 Search on NAS-Bench-201 Space:**

* Running for any Stage-1 methods other than the baseline requires having first run the Stage-1 algorithm as shown above and input args must match the output dir of that run
 - input args for baselines can be found in baselines/exp_lists.py listed in ({algo}, {seed}, {t}) format.

* ```cd 201-space && python search2.py --method {algo} --seed {seed} --dataset {dset} --s_time {t}```

 - Note that algo includes options listed above plus [rsps2/none/none+] and t is time of Stage-1 run as YYYYMMDD-HHMMSS.

**Record Validation Accuracy across NAS-Bench-201 architectures:**

* This is a necessary step for our "top arch ID rate" statistic and also computes the arch sampling probability in the process

* ```python record_val.py --method {algo} --seed {seed} --dataset {dset} --s_time {t}```
  - implemented options for algo and dset are the same as in Stage-1 search, t is time of Stage-1 run as YYYYMMDD-HHMMSS.

**Compute Stage-1 and Stage-2 search statistics**

* We have provided code used to compute and plot the Stage-1 and Stage-2 search statistics demonstrated in our paper in 201-space/stats.py

* ```record_val.py``` must be run for each Stage-1 search method before its stats can be plotted and ```search2.py``` must be run on the baselines before Stage-2 search stats can be plotted

* stats.py includes two functions ```plot_s1_eval(exp_list, dset)``` and ```plot_s2_eval(exp_list, dset)``` which take a list of Stage-1 experiments, formatted as ({algo}, {seed}, {t}) as in baselines/exp_lists.py, and a dset in ['cifar10'/'cifar100'/'imagenet16-120']

**Stage-1 Search on DARTS Space:**

* Data preparation: For a direct search on ImageNet, we follow [PC-DARTS](https://github.com/yuhuixu1993/PC-DARTS) to sample 10% and 2.5% images for earch class as train and validation.

* ```CIFAR-10: cd DARTS-space && python train_search.py``` (default syntax)

* ```CIFAR-10/CIFAR-100: cd DARTS-space && python train_search.py --method {algo} --seed {seed} --dataset {dset} --k 2``` (set args for our experiments)
 - algo is in [gdas/snas/darts/darts-/dirichlet], dset is in [cifar10/cifar100], seed is an integer

* ```ImageNet: cd DARTS-space && python train_search_imagenet.py``` (only implemented for DrNAS)

**Stage-2 Search on DARTS Space:**

* ```cd DARTS-space && python search2.py --method {algo} --seed {seed} --dataset {dset} --s_time {t}```
 - options for args are same as in Stage-1 Search on DARTS space and must match the output dir of the completed Stage-1 run, with t as the time of that run
* As search2.py performs all Stage-2 search algorithms in succesion on the results of a given Stage-1 search algorithm and the perturbation Stage-2 search method can take considerably longer than the other methods, we find it is sometimes necessary to resume Stage-2 Search mid-perturbation and provide a simple method to do so:
 - substitute "search3.py" for "search2.py" and add ```--dev_resume_epoch {ep}``` where ep is the last fine tuning epoch of the prior perturbation process

### Architecture Evaluation (DARTS space)

* ```CIFAR-10: cd DARTS-space && python train.py --cutout --auxiliary``` (DrNAS architecture)

* ```CIFAR-10/CIFAR-100: cd DARTS-space && python train.py --dataset {dset} --arch {arch} --cutout --auxiliary```
 - dset is in [cifar10/cifar100] and arch must be defined in DARTS-space/genotypes.py

* ```ImageNet: cd DARTS-space && python train_imagenet.py --auxiliary```

### View combined Stage-1 + Stage-2 results

* For NAS-Bench-201 we include a ```pretty_print(exp_list, dset)``` function which prints a table of results where rows are Stage-1 search algorithms and columns are Stage-2 search algorithms

* For the DARTS space, we also include a corresponding ```pretty_print(dset)``` function which prints the same table for DARTS space results


## Reference
If you find this code useful in your research please cite
```
@inproceedings{chen2021drnas,
    title={Dr{\{}NAS{\}}: Dirichlet Neural Architecture Search},
    author={Xiangning Chen and Ruochen Wang and Minhao Cheng and Xiaocheng Tang and Cho-Jui Hsieh},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=9FWas6YbmB3}
}
```



## Related Publications

* Xiangning Chen, Cho-Jui Hsieh. [Stabilizing Differentiable Architecture Search via Perturbation-based Regularization](http://proceedings.mlr.press/v119/chen20f.html). ICML 2020. [Code](https://github.com/xiangning-chen/SmoothDARTS).
* Ruochen Wang, Minhao Cheng, Xiangning Chen, Xiaocheng Tang, Cho-Jui Hsieh. [Rethinking Architecture Selection in Differentiable NAS](https://openreview.net/forum?id=PKubaeJkw3). ICLR 2021 (oral). [Code](https://github.com/ruocwang/darts-pt).
