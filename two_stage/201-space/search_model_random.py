import sys
import random
sys.path.insert(0, '../')
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from cell_operations import ResNetBasicblock
from search_cells     import NAS201SearchCell as SearchCell
from genotypes        import Structure
from utils import process_step_matrix
import logging


class TinyNetworkRandom(nn.Module):

  def __init__(self, C, N, max_nodes, num_classes, criterion, search_space, affine=False, track_running_stats=True, species='uniform'):
    super(TinyNetworkRandom, self).__init__()
    self._C        = C
    self._layerN   = N
    self.species   = species
    self.max_nodes = max_nodes
    self._criterion = criterion
    self.stem = nn.Sequential(
                    nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(C))
 
    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N    
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

    C_prev, num_edge, edge2index = C, None, None
    self.cells = nn.ModuleList()
    for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
      if reduction:
        cell = ResNetBasicblock(C_prev, C_curr, 2)
      else:
        cell = SearchCell(C_prev, C_curr, 1, max_nodes, search_space, affine, track_running_stats)
        if num_edge is None: num_edge, edge2index = cell.num_edges, cell.edge2index
        else: assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
      self.cells.append( cell )
      C_prev = cell.out_dim
    self.op_names   = deepcopy( search_space )
    self._Layer     = len(self.cells)
    self.edge2index = edge2index
    self.lastact    = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
  
  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target)

  def get_weights(self):
    xlist = list( self.stem.parameters() ) + list( self.cells.parameters() )
    xlist+= list( self.lastact.parameters() ) + list( self.global_pooling.parameters() )
    xlist+= list( self.classifier.parameters() )
    return xlist

  def arch_parameters(self):
    return []

  def show_arch_parameters(self):
    pass

  def build_sampler(self, api, dataset):
    accuracies = []
    for i in range(len(api)):
      info = api.query_by_index(i)
      metrics = info.get_metrics(dataset, 'ori-test', is_random=False)
      accuracies.append(metrics['accuracy'])
    #probs = [0] * len(api)
    s_ind = np.argsort(accuracies).astype(int)
    probs = np.zeros_like(s_ind)
    probs[s_ind] = np.arange(len(accuracies)) 
    #topacc = max(accuracies)
    #for i, acc in enumerate(accuracies):
    #  if (topacc - acc) < 0.1:
    #    probs[i] = 0.01
    #  elif (topacc - acc) < 0.5:
    #    probs[i] = 0.002
    #  elif (topacc - acc) < 1.0:
    #    probs[i] = 0.001
    #  elif (topacc - acc) < 2.0:
    #    probs[i] = 0.0001
    self.a_prob = probs
    self.api = api

  def get_message(self):
    string = self.extra_repr()
    for i, cell in enumerate(self.cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
    return string

  def extra_repr(self):
    return ('{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

  def genotype(self):
    genotypes = []
    for i in range(1, self.max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        op_name = random.choice( self.op_names )
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return Structure( genotypes )

  def forward(self, inputs, weights=None):
    if weights is None:
      if self.species == "uniform":
        struct = self.genotype()
      else:
        arch_id = random.choices(list(range(15625)), weights=self.a_prob)
        arch_str = self.api[arch_id[0]]
        struct = Structure.str2fullstructure(arch_str)
    feature = self.stem(inputs)
    for i, cell in enumerate(self.cells):
      if isinstance(cell, SearchCell) and weights is None:
        feature = cell.forward_dynamic(feature, struct)
      elif isinstance(cell, SearchCell):
        feature = cell(feature, weights)
      else:
        feature = cell(feature)
    out = self.lastact(feature)
    out = self.global_pooling( out )
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)

    return logits

