import os
import sys
sys.path.insert(0, '../')
import time
import glob
import pickle
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from net2wider import configure_optimizer, configure_scheduler
from model_search import Network
from model_search_darts_proj import DartsNetworkProj
from architect import Architect
from projection import pt_project
from pruners import *
import utils as utils

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='datapath', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--epochs', type=int, default=50, help='epochs to train supernet')
parser.add_argument('--method', type=str, default='dirichlet', help='choose nas method')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--tau_max', type=float, default=10, help='Max temperature (tau) for the gumbel softmax.')
parser.add_argument('--tau_min', type=float, default=0.1, help='Min temperature (tau) for the gumbel softmax.')
parser.add_argument('--k', type=int, default=2, help='init partial channel parameter')
parser.add_argument('--s_time', type=str, help='save dir time in %Y%m%d-%H%M%S')
parser.add_argument('--n_sample', type=int, default=100, help='number of architectures to sample')

#### regularization
parser.add_argument('--reg_type', type=str, default='l2', choices=['l2', 'kl'], help='regularization type')
parser.add_argument('--reg_scale', type=float, default=1e-3, help='scaling factor of the regularization term, default value is proper for l2, for kl you might adjust reg_scale to match l2')

parser.add_argument('--dev_resume_epoch', type=int, default=-1, help="resume epoch for arch selection phase, starting from 0")
parser.add_argument('--dev_resume_log', type=str, default='search2', help="resume log name for arch selection phase")
## projection
parser.add_argument('--edge_decision', type=str, default='random', choices=['random'], help='used for both proj_op and proj_edge')
parser.add_argument('--proj_crit_normal', type=str, default='acc', choices=['loss', 'acc'])
parser.add_argument('--proj_crit_reduce', type=str, default='acc', choices=['loss', 'acc'])
parser.add_argument('--proj_crit_edge',   type=str, default='acc', choices=['loss', 'acc'])
parser.add_argument('--proj_intv', type=int, default=5, help='interval between two projections')
parser.add_argument('--proj_mode_edge', type=str, default='reg', choices=['reg'],
                    help='edge projection evaluation mode, reg: one edge at a time')
### zero cost measures
parser.add_argument('--dataload', type=str, default='random', help='random or grasp supported')
parser.add_argument('--dataload_info', type=int, default=1, help='number of batches to use for random dataload or number of samples per class for grasp dataload')


args = parser.parse_args()

args.save = '../experiments/{}/search-{}-{}-{}'.format(
    args.dataset, args.save, args.s_time, args.seed)
args.save += '-init_channels-' + str(args.init_channels)
#args.save += '-layers-' + str(args.layers) 
args.save += '-'+args.method+'-'
args.save += '-init_pc-' + str(args.k)
#utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
log_file = os.path.join(args.save, 'search3_log.txt')
fh = logging.FileHandler(log_file)
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

args.dev_resume_checkpoint_dir = os.path.join(args.save, args.dev_resume_log)
if not os.path.exists(args.dev_resume_checkpoint_dir):
    os.mkdir(args.dev_resume_checkpoint_dir)
args.dev_save_checkpoint_dir = os.path.join(args.save, 'search2')
if not os.path.exists(args.dev_save_checkpoint_dir):
    os.mkdir(args.dev_save_checkpoint_dir)


CIFAR_CLASSES = 10
if args.dataset == 'cifar100':
    CIFAR_CLASSES = 100

def sample(model, n):
  def _parse(weights, steps):
    n = 2
    start = 0
    ncol = weights.shape[1]
    sampleW = torch.zeros_like(weights)
    for i in range(steps):
      end = start + n
      W = weights[start:end].detach().clone()
      probs = W.view(-1).softmax(0)
      id1 = torch.multinomial(probs, 1)
      row = id1.item() // ncol
      probs[row*ncol:(row+1)*ncol] = 0.0
      probs /= sum(probs)
      id2 = torch.multinomial(probs, 1)
      idx = torch.cat((id1, id2))
      one_h = torch.zeros_like(probs).scatter_(-1, idx, 1.0).view(*W.shape)
      sampleW[start:end].copy_(one_h)
      start = end
      n += 1
    return sampleW

  sampled_weights = []
  for i in range(n):
    weights_normal = _parse(model.alphas_normal, model._steps)
    weights_reduce = _parse(model.alphas_reduce, model._steps)
    sampled_weights.append((weights_normal.bool(), weights_reduce.bool()))
  return sampled_weights

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  '''
  if args.method == 'gdas' or args.method == 'snas':
    # Create the decrease step for the gumbel softmax temperature
    tau_step = (args.tau_min - args.tau_max) / args.epochs
    tau_epoch = args.tau_max
    if args.method == 'gdas':
      model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, k=args.k,
                      species='gdas', reg_type=args.reg_type, reg_scale=args.reg_scale)
    else:
      model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, k=args.k,
                      species='gumbel', reg_type=args.reg_type, reg_scale=args.reg_scale)
  elif args.method == 'dirichlet':
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, k=args.k,
                    species='dirichlet', reg_type=args.reg_type, reg_scale=args.reg_scale)
  elif args.method == 'darts':
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, k=args.k,
                    species='softmax', reg_type=args.reg_type, reg_scale=args.reg_scale)
  elif args.method == 'darts-':
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, k=args.k,
                    species='softmax', reg_type=args.reg_type, reg_scale=args.reg_scale, aux_skip=True)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
    model.parameters(),
    args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay)
  '''

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  if args.dataset=='cifar100':
    train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
  else:
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))


  with open(os.path.join(args.save, 'search2.pickle'), 'rb') as f:
    res = pickle.load(f)

  ########################################################################################################
  '''
  logging.info("Begin sampling based stage 2 search")

  train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
    pin_memory=True)

  valid_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
    pin_memory=True)
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, k=args.k)

  filename = os.path.join(args.save, 'checkpoint.pth.tar')
  if os.path.isfile(filename):
    logging.info("=> loading checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename, map_location='cpu')
    model_state_dict = checkpoint['state_dict']
    model.load_state_dict(model_state_dict, strict=False) # model
    saved_arch_parameters = checkpoint['alpha'] # arch
    model.set_arch_parameters(saved_arch_parameters)
    logging.info("=> loaded checkpoint '{}' ".format(filename))
  else:
    print("=> no checkpoint found at '{}'".format(filename))

  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  sampled_archs = sample(model, args.n_sample)
  device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

  for arch_n, arch_r in sampled_archs:
    model.set_mask(arch_n, arch_r)
    measures = predictive.find_measures(model,
                                        train_queue,
                                        (args.dataload, args.dataload_info, CIFAR_CLASSES),
                                        device)
    res["measures"].append((measures, model.genotype()))
    val_acc, val_loss = infer(valid_queue, model, criterion)
    res["valid"].append((val_acc.item(), model.genotype()))

  logging.info(res)
  with open(os.path.join(args.save, 'search3.pickle'), 'wb') as f:
    pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)

  '''
  #######################################################################################################
  
  logging.info("Begin perturbation based stage 2 search")

  train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
    pin_memory=True)

  valid_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
    pin_memory=True)


  model = DartsNetworkProj(args.init_channels, CIFAR_CLASSES, args.layers, criterion, k=args.k)

  filename = os.path.join(args.save, 'checkpoint.pth.tar')
  if os.path.isfile(filename):
    logging.info("=> loading checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename, map_location='cpu')
    model_state_dict = checkpoint['state_dict']
    model.load_state_dict(model_state_dict, strict=False) # model
    saved_arch_parameters = checkpoint['alpha'] # arch
    model.set_arch_parameters(saved_arch_parameters)
    logging.info("=> loaded checkpoint '{}' ".format(filename))
  else:
    print("=> no checkpoint found at '{}'".format(filename))

  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  architect = Architect(model, args)

  p_geno = pt_project(train_queue, valid_queue, model, architect, args, infer)

  res["perturb"].append(p_geno)

  logging.info(res)
  with open(os.path.join(args.save, 'search2.pickle'), 'wb') as f:
    pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
  


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)
    input = input.cuda()
    target = target.cuda(non_blocking=True)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = input_search.cuda()
    target_search = target_search.cuda(non_blocking=True)

    #if epoch >= 10:
    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
    optimizer.zero_grad()
    architect.optimizer.zero_grad()

    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    optimizer.zero_grad()
    architect.optimizer.zero_grad()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data, n)
    top1.update(prec1.data, n)
    top5.update(prec5.data, n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    if 'debug' in args.save:
      break

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion, log=True, _eval=True, weights_dict=None):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval() if _eval else model.train() # disable running stats for projection

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = input.cuda()
      target = target.cuda(non_blocking=True)

      if weights_dict is None:
        logits = model(input)
      else:
        logits = model(input, weights_dict=weights_dict)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.data, n)
      top1.update(prec1.data, n)
      top5.update(prec5.data, n)

      if step % args.report_freq == 0 and log:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 
