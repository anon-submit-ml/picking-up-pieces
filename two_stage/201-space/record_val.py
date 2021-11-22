import os
import sys
sys.path.insert(0, '../')
import time
import glob
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
from cell_operations import NAS_BENCH_201
from architect import Architect

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
#### regularization
parser.add_argument('--reg_type', type=str, default='l2', choices=[
                    'l2', 'kl'], help='regularization type, kl is implemented for dirichlet only')
parser.add_argument('--reg_scale', type=float, default=1e-3,
                    help='scaling factor of the regularization term, default value is proper for l2, for kl you might adjust reg_scale to match l2')
args = parser.parse_args()

resume = False

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

#utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
sv_filename = os.path.join(args.save, 'val_rec_log.txt')
if os.path.isfile(sv_filename):
    fh = logging.FileHandler(os.path.join(args.save, 'val_rec_log2.txt'))
    resume = True
else:
    fh = logging.FileHandler(os.path.join(sv_filename))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter(args.save + '/valrec')


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

def str2w_p(model, xstr, default_name='none'):
    assert isinstance(xstr, str), 'must take string (not {:}) as input'.format(type(xstr))
    nodestrs = xstr.split('+')
    arch_prob = 1.0
    with torch.no_grad():
      weights = torch.zeros_like(model._arch_parameters)
      probs = nn.functional.softmax(model._arch_parameters, dim=-1).cpu()
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
          arch_prob *= probs[ model.edge2index[n_str]][ op_idx].item()
          weights.data[ model.edge2index[n_str]][ op_idx] = 1.0
    return weights, arch_prob


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

    if resume:
        prev_vals = []
        with open(sv_filename, 'r') as f:
            for line in f:
                lwords = line.strip().split(' ')
                if lwords[-3] == 'accuracy':
                    prev_vals.append(float(lwords[-1]))
    if args.method == 'gdas' or args.method == 'snas':
        # Create the decrease step for the gumbel softmax temperature
        tau_step = (args.tau_min - args.tau_max) / args.epochs
        tau_epoch = args.tau_max
        if args.method == 'gdas':
            model = TinyNetworkGDAS(C=args.init_channels, N=5, max_nodes=4, num_classes=n_classes, criterion=criterion, search_space=NAS_BENCH_201)
        else:
            model = TinyNetwork(C=args.init_channels, N=5, max_nodes=4, num_classes=n_classes,
                                criterion=criterion, search_space=NAS_BENCH_201, k=args.k, species='gumbel')
    elif args.method == 'dirichlet':
        model = TinyNetwork(C=args.init_channels, N=5, max_nodes=4, num_classes=n_classes,
                            criterion=criterion, search_space=NAS_BENCH_201, k=args.k, species='dirichlet',
                            reg_type=args.reg_type, reg_scale=args.reg_scale)
    elif args.method == 'darts':
        model = TinyNetwork(C=args.init_channels, N=5, max_nodes=4, num_classes=n_classes,
                            criterion=criterion, search_space=NAS_BENCH_201, k=args.k, species='softmax')
    elif args.method == 'darts-':
        model = TinyNetworkMinus(C=args.init_channels, N=5, max_nodes=4, num_classes=n_classes,
                            criterion=criterion, search_space=NAS_BENCH_201, k=args.k, beta=0.0)
    elif args.method == 'rsps':
        model = TinyNetworkRandom(C=args.init_channels, N=5, max_nodes=4, num_classes=n_classes,
                                  criterion=criterion, search_space=NAS_BENCH_201, species='uniform')
    elif args.method == 'rsps+':
        model = TinyNetworkRandom(C=args.init_channels, N=5, max_nodes=4, num_classes=n_classes,
                                  criterion=criterion, search_space=NAS_BENCH_201, species='biased')
        if args.dataset == 'imagenet16-120':
            model.build_sampler(api, 'ImageNet16-120')
        else:
            model.build_sampler(api, args.dataset)

    optimizer = torch.optim.SGD(
        model.get_weights(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    #checkpoint  = torch.load(args.save+"/checkpoint.pth.tar")
    #model.load_state_dict( checkpoint['state_dict'] )
    logging.info('loading checkpoint from {}'.format(args.save))
    filename = os.path.join(args.save, 'checkpoint.pth.tar')

    if os.path.isfile(filename):
      logging.info("=> loading checkpoint '{}'".format(filename))
      checkpoint = torch.load(filename, map_location='cpu')
      #start_epoch = checkpoint['epoch'] # epoch
      model_state_dict = checkpoint['state_dict']
      #if '_arch_parameters' in model_state_dict: del model_state_dict['_arch_parameters']
      model.load_state_dict(model_state_dict) # model
      #saved_arch_parameters = checkpoint['alpha'] # arch
      #model._arch_parameters = saved_arch_parameters
      optimizer.load_state_dict(checkpoint['optimizer']) # optimizer
      logging.info("=> loaded checkpoint '{}' ".format(filename))
    else:
      print("=> no checkpoint found at '{}'".format(filename))

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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    if args.method not in ['rsps', 'rsps+']:
      architect = Architect(model, args)
    else:
      architect = None

    perf = np.zeros((len(api),))
    prob = np.zeros((len(api),))

    # validate with every arch
    for i, arch_str in enumerate(api):
      w, pr = str2w_p(model, arch_str)
      prob[i] = pr
      if resume and i < len(prev_vals):
        acc = prev_vals[i]
      else:
        acc, loss = infer(valid_queue, model, criterion, w)
      perf[i] = acc
      logging.info('[%d] valid : accuracy = %f', i, acc)
    np.save(args.save+'/perfpred.npy', perf)
    np.save(args.save+'/archprob.npy', prob)


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

def infer(valid_queue, model, criterion, weights=None):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

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

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            if 'debug' in args.save:
                break
            if step >= 25:
                break
    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
