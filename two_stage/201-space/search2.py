import os
import sys
sys.path.insert(0, '../')
import time
import glob
import pickle
import numpy as np
import random
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from search_model_gdas import TinyNetworkGDAS
from search_model import TinyNetwork
from search_model_minus import TinyNetworkMinus
from search_model_random import TinyNetworkRandom
from search_model_darts_proj import TinyNetworkDartsProj
from cell_operations import NAS_BENCH_201
from architect import Architect
from projection import pt_project
from pruners import *

from copy import deepcopy
from numpy import linalg as LA

from torch.utils.tensorboard import SummaryWriter
from nas_201_api import NASBench201API as API


parser = argparse.ArgumentParser("sota")
parser.add_argument('--data', type=str, default='datapath', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
parser.add_argument('--method', type=str, default='dirichlet', help='choose nas method')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--tau_max', type=float, default=10, help='Max temperature (tau) for the gumbel softmax.')
parser.add_argument('--tau_min', type=float, default=0.1, help='Min temperature (tau) for the gumbel softmax.')
parser.add_argument('--k', type=int, default=1, help='partial channel parameter')
parser.add_argument('--s_time', type=str, help='save dir time in %Y%m%d-%H%M%S')
parser.add_argument('--n_sample', type=int, default=100, help='number of architectures to sample')
#### regularization
parser.add_argument('--reg_type', type=str, default='l2', choices=[
                    'l2', 'kl'], help='regularization type, kl is implemented for dirichlet only')
parser.add_argument('--reg_scale', type=float, default=1e-3,
                    help='scaling factor of the regularization term, default value is proper for l2, for kl you might adjust reg_scale to match l2')
#### projection
parser.add_argument('--edge_decision', type=str, default='random', choices=['random'], help='which edge to be projected next')
parser.add_argument('--proj_crit', type=str, default='acc', choices=['loss', 'acc'], help='criteria for projection')
parser.add_argument('--proj_intv', type=int, default=5, help='fine tune epochs between two projections')
### zero cost measures
parser.add_argument('--dataload', type=str, default='random', help='random or grasp supported')
parser.add_argument('--dataload_info', type=int, default=1, help='number of batches to use for random dataload or number of samples per class for grasp dataload')

args = parser.parse_args()
if args.method == 'none':
    args.s_time = time.strftime("%Y%m%d-%H%M%S")

if args.method == 'rsps2':
    args.save = '../experiments/nasbench201/{}-search-{}-{}-{}'.format(
        'rsps+', args.save, args.s_time, args.seed)
elif args.method == 'none+':
    args.save = '../experiments/nasbench201/{}-search-{}-{}-{}'.format(
        'none', args.save, args.s_time, args.seed)
else:
    args.save = '../experiments/nasbench201/{}-search-{}-{}-{}'.format(
        args.method, args.save, args.s_time, args.seed)

if not args.dataset == 'cifar10':
    args.save += '-' + args.dataset
if args.unrolled:
    args.save += '-unrolled'
if not args.weight_decay == 3e-4:
    args.save += '-weight_l2-' + str(args.weight_decay)
if not args.arch_weight_decay == 1e-3:
    args.save += '-alpha_l2-' + str(args.arch_weight_decay)
if not args.method == 'gdas':
    args.save += '-pc-' + str(args.k)

if args.method == 'none':
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
if args.method in ['rsps2', 'none+']:
    fh = logging.FileHandler(os.path.join(args.save, 'search22_log.txt'))
else:
    fh = logging.FileHandler(os.path.join(args.save, 'search2_log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter(args.save + '/search2')


if args.dataset == 'cifar100':
    n_classes = 100
elif args.dataset == 'imagenet16-120':
    n_classes = 120
else:
    n_classes = 10


def distill(result):
    result = result.split('\n')
    cifar10 = result[5].replace(' ', '').split(':')
    cifar100 = result[7].replace(' ', '').split(':')
    imagenet16 = result[9].replace(' ', '').split(':')

    cifar10_train = float(cifar10[1].strip(',test')[-7:-2].strip('='))
    cifar10_test = float(cifar10[2][-7:-2].strip('='))
    cifar100_train = float(cifar100[1].strip(',valid')[-7:-2].strip('='))
    cifar100_valid = float(cifar100[2].strip(',test')[-7:-2].strip('='))
    cifar100_test = float(cifar100[3][-7:-2].strip('='))
    imagenet16_train = float(imagenet16[1].strip(',valid')[-7:-2].strip('='))
    imagenet16_valid = float(imagenet16[2].strip(',test')[-7:-2].strip('='))
    imagenet16_test = float(imagenet16[3][-7:-2].strip('='))

    return cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
        cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test

def str2w(model, xstr, default_name='none'):
    assert isinstance(xstr, str), 'must take string (not {:}) as input'.format(type(xstr))
    nodestrs = xstr.split('+')
    with torch.no_grad():
      weights = torch.zeros_like(model._arch_parameters, dtype=torch.bool)
      for i, node_str in enumerate(nodestrs):
        inputs = list(filter(lambda x: x != '', node_str.split('|')))
        for xinput in inputs: assert len(xinput.split('~')) == 2, 'invalid input length : {:}'.format(xinput)
        inputs = ( xi.split('~') for xi in inputs )
        input_infos = list( (op, int(IDX)) for (op, IDX) in inputs)
        all_in_nodes= list(x[1] for x in input_infos)
        for j in range(i+1):
          if j not in all_in_nodes: input_infos.append((default_name, j))
        node_info = sorted(input_infos, key=lambda x: (x[1], x[0]))
        for op_name, j in node_info:
          n_str = '{:}<-{:}'.format(i+1,j)
          op_idx = model.op_names.index(op_name)
          weights.data[ model.edge2index[n_str]][ op_idx] = 1
    return weights

class ArchSampler:

    def __init__(self, model, method, api, dataset=None):
        self.model = model
        self.method = method
        self.api = api
        if method in ['rsps+', 'none+']:
            if dataset == 'imagenet16-120':
                self.build_sampler('ImageNet16-120')
            else:
                self.build_sampler(dataset)

    def sample(self, n):
        if 'rsps' in self.method:
            if self.method == 'rsps+':
                arch_ids = random.choices(list(range(15625)), weights=self.a_prob, k=n)
            else:
                arch_ids = random.choices(list(range(15625)), k=n)
            sampled_weights = [(self.api[i], str2w(self.model, self.api[i])) for i in arch_ids]
        elif self.method == 'none+':
            arch_ids = random.choices(list(range(15625)), weights=self.a_prob, k=n)
            sampled_weights = [(self.api[i], str2w(self.model, self.api[i])) for i in arch_ids]
        else:
            sampled_weights = []
            for i in range(n):
              with torch.no_grad():
                while True:
                  probs   = nn.functional.softmax(self.model._arch_parameters, dim=1)
                  index   = torch.multinomial(probs, 1)
                  one_h   = torch.zeros_like(probs).scatter_(1, index, 1.0)
                  if ((torch.isinf(probs).any()) or (torch.isnan(probs).any())):
                    continue
                  else:
                    arch_str = self.model.get_geno(one_h)
                    sampled_weights.append((arch_str, one_h.bool()))
                    break
        return sampled_weights


    def build_sampler(self, dataset):
        accuracies = []
        for i in range(len(self.api)):
          info = self.api.query_by_index(i)
          metrics = info.get_metrics(dataset, 'ori-test', is_random=False)
          accuracies.append(metrics['accuracy'])
        s_ind = np.argsort(accuracies).astype(int)
        probs = np.zeros_like(s_ind)
        probs[s_ind] = np.arange(len(accuracies))
        self.a_prob = probs

def main():
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    if not 'debug' in args.save:
        api = API('../../nasbench201/NAS-Bench-201-v1_0-e61699.pth')
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    res = {"valid" : [],
           "measures" : [],
           "sampled_test" : [],
           "perturb" : []}

    logging.info("Begin sampling based stage 2 search")

    model = TinyNetwork(C=args.init_channels, N=5, max_nodes=4, num_classes=n_classes,
                                 criterion=criterion, search_space=NAS_BENCH_201, k=args.k, species='softmax')
    if not args.method == 'none':
        logging.info('loading checkpoint from {}'.format(args.save))
        filename = os.path.join(args.save, 'checkpoint.pth.tar')

        if os.path.isfile(filename):
          logging.info("=> loading checkpoint '{}'".format(filename))
          checkpoint = torch.load(filename, map_location='cpu')
          model_state_dict = checkpoint['state_dict']
          model.load_state_dict(model_state_dict, strict=False) # model
        else:
          print("=> no checkpoint found at '{}'".format(filename))

    sampler = ArchSampler(model, args.method, api, args.dataset)

    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    if args.dataset == 'cifar10':
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'cifar100':
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'svhn':
        train_transform, valid_transform = utils._data_transforms_svhn(args)
        train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
    elif args.dataset == 'imagenet16-120':
        import torchvision.transforms as transforms
        from DownsampledImageNet import ImageNet16
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22,  61.26, 65.09]]
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(16, padding=2), transforms.ToTensor(), transforms.Normalize(mean, std)]
        train_transform = transforms.Compose(lists)
        train_data = ImageNet16(root=os.path.join(args.data,'imagenet16'), train=True, transform=train_transform, use_num_of_class_only=120)
        assert len(train_data) == 151700

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True)

    sampled_archs = sampler.sample(args.n_sample)
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

    for arch_s, arch_w in sampled_archs:
        result = api.query_by_arch(arch_s)
        if args.dataset == 'cifar10':
            train_acc, test_acc, _, _, _, _, _, _ = distill(result)
        elif args.dataset == 'cifar100':
            _, _, train_acc, valid_acc, test_acc, _, _, _ = distill(result)
        elif args.dataset == 'imagenet16-120':
            _, _, _, _, _, train_acc, valid_acc, test_acc = distill(result)
        res['sampled_test'].append(test_acc)
        model.set_mask(arch_w)
        measures = predictive.find_measures(model,
                                            train_queue,
                                            (args.dataload, args.dataload_info, n_classes),
                                            device)
        res["measures"].append(measures)
        val_acc, val_loss = infer(valid_queue, model, criterion)
        res["valid"].append(val_acc.item())

    #######################################################################################################
    '''
    
    logging.info("Begin perturbation based stage 2 search")
    model = TinyNetworkDartsProj(C=args.init_channels, N=5, max_nodes=4, num_classes=n_classes,
                                 criterion=criterion, search_space=NAS_BENCH_201, k=args.k)
    optimizer = torch.optim.SGD(
        model.get_weights(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    #checkpoint  = torch.load(args.save+"/checkpoint.pth.tar")
    #model.load_state_dict( checkpoint['state_dict'] )
    logging.info('loading checkpoint from {}'.format(args.save))
    filename = os.path.join(args.save, 'checkpoint.pth.tar')

    start_epoch = 0

    if os.path.isfile(filename):
      logging.info("=> loading checkpoint '{}'".format(filename))
      checkpoint = torch.load(filename, map_location='cpu')
      start_epoch = checkpoint['epoch'] # epoch
      model_state_dict = checkpoint['state_dict']
      if '_arch_parameters' in model_state_dict: del model_state_dict['_arch_parameters']
      model.load_state_dict(model_state_dict, strict=False) # model
      if 'rsps' not in args.method:
        saved_arch_parameters = checkpoint['alpha'] # arch
        model.set_arch_parameters(saved_arch_parameters)
      optimizer.load_state_dict(checkpoint['optimizer']) # optimizer
      logging.info("=> loaded checkpoint '{}' (epoch {})".format(filename, start_epoch - 1))
    else:
      print("=> no checkpoint found at '{}'".format(filename))

    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    arch_str = pt_project(train_queue, valid_queue, model, architect, criterion,
                          start_epoch, args, infer)

    result = api.query_by_arch(arch_str)
    logging.info('{:}'.format(result))
    if args.dataset == 'cifar10':
        train_acc, test_acc, _, _, _, _, _, _ = distill(result)
    elif args.dataset == 'cifar100':
        _, _, train_acc, valid_acc, test_acc, _, _, _ = distill(result)
    elif args.dataset == 'imagenet16-120':
        _, _, _, _, _, train_acc, valid_acc, test_acc = distill(result)
    res['perturb'].append(test_acc)
    res['perturb'].append(arch_str)
    '''
     
    logging.info(res)
    if args.method in ['rsps2', 'none+']:
        with open(os.path.join(args.save, 'search22.pickle'), 'wb') as f:
            pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(os.path.join(args.save, 'search2.pickle'), 'wb') as f:
            pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)

    writer.close()

'''
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
        
        # if epoch >= 15:
        if not (architect is None):
            architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
            optimizer.zero_grad()
            architect.optimizer.zero_grad()

        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        if not (architect is None):
            architect.optimizer.zero_grad()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        if 'debug' in args.save:
            break

    return  top1.avg, objs.avg
'''

def infer(valid_queue, model, criterion, weights=None, eval=True, log=True):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval() if eval else model.train()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = model(input, weights)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if log and step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            if 'debug' in args.save:
                break
    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
