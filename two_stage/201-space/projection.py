import os
import sys
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, '../')
import utils as ig_utils
import logging
import torch.utils

from nas_201_api import NASBench201API as API

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)


def pt_project(train_queue, valid_queue, model, architect, criterion,
               epoch, args, infer): #, query):
    def project(model, args):
        ## macros
        num_edge, num_op = model.num_edge, model.num_op

        ## select an edge
        remain_eids = torch.nonzero(model.candidate_flags).cpu().numpy().T[0]
        if args.edge_decision == "random":
            selected_eid = np.random.choice(remain_eids, size=1)[0]

        ## select the best operation
        if args.proj_crit == 'loss':
            crit_idx = 1
            compare = lambda x, y: x > y
        if args.proj_crit == 'acc':
            crit_idx = 0
            compare = lambda x, y: x < y
        
        best_opid = 0
        crit_extrema = None
        for opid in range(num_op):
            ## projection
            weights = model.get_projected_weights()
            proj_mask = torch.ones_like(weights[selected_eid])
            proj_mask[opid] = 0
            weights[selected_eid] = weights[selected_eid] * proj_mask

            ## proj evaluation
            valid_stats = infer(valid_queue, model, criterion, log=False, eval=False, weights=weights)
            crit = valid_stats[crit_idx]
            
            if crit_extrema is None or compare(crit, crit_extrema):
                crit_extrema = crit
                best_opid = opid
            logging.info('valid_acc %f', valid_stats[0])
            logging.info('valid_loss %f', valid_stats[1])

        logging.info('best opid %d', best_opid)
        return selected_eid, best_opid

    ## query
    # api = API('../../nasbench201/NAS-Bench-201-v1_0-e61699.pth')

    model.train()
    model.printing(logging)

    train_acc, train_obj = infer(train_queue, model, criterion, log=False)
    logging.info('train_acc  %f', train_acc)
    logging.info('train_loss %f', train_obj)

    valid_acc, valid_obj = infer(valid_queue, model, criterion, log=False)
    logging.info('valid_acc  %f', valid_acc)
    logging.info('valid_loss %f', valid_obj)

    objs = ig_utils.AvgrageMeter()
    top1 = ig_utils.AvgrageMeter()
    top5 = ig_utils.AvgrageMeter()

    num_edges = model.arch_parameters()[0].shape[0]
    proj_intv = args.proj_intv
    tune_epochs = proj_intv * (num_edges - 1)
    optimizer = torch.optim.SGD(
      model.get_weights(),
      args.learning_rate / 10,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

    for epoch in range(tune_epochs):
        logging.info('epoch %d', epoch)

        if epoch % proj_intv == 0 or epoch == tune_epochs - 1:
            logging.info('project')
            selected_eid, best_opid = project(model, args)
            model.project_op(selected_eid, best_opid)
            model.printing(logging)

        for step, (input, target) in enumerate(train_queue):
            model.train()
            n = input.size(0)

            ## fetch data
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            input_search, target_search = next(iter(valid_queue))
            input_search = input_search.cuda()
            target_search = target_search.cuda(non_blocking=True)

            ## train alpha
            optimizer.zero_grad(); architect.optimizer.zero_grad()
            architect.step(input, target, input_search, target_search, args.learning_rate/10, optimizer, unrolled=args.unrolled)

            ## train weight
            optimizer.zero_grad(); architect.optimizer.zero_grad()
            logits = model(input)
            loss = criterion(logits, target)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            ## logging
            prec1, prec5 = ig_utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if step % args.report_freq == 0:
                logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        ## one epoch end
        model.printing(logging)

        train_acc, train_obj = infer(train_queue, model, criterion, log=False)
        logging.info('train_acc  %f', train_acc)
        logging.info('train_loss %f', train_obj)

        valid_acc, valid_obj = infer(valid_queue, model, criterion, log=False)
        logging.info('valid_acc  %f', valid_acc)
        logging.info('valid_loss %f', valid_obj)

    # nasbench201
    # query(api, model.genotype(), logging)

    return model.genotype()
