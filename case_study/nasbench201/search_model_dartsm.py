import torch
import torch.nn as nn
from .cell_operations import ResNetBasicblock
from .search_cells import NAS201SearchCell_Minus as SearchCell
from .search_model import TinyNetwork as TinyNetwork


class TinyNetworkDartsMinus(TinyNetwork):
  # adds auxiliary skip connection to each op choice in DARTS
  def __init__(self, C, N, max_nodes, num_classes, criterion, search_space, args,
               affine=False, track_running_stats=True, beta=1.0):
    super(TinyNetworkDartsMinus, self).__init__(C, N, max_nodes, num_classes, criterion, search_space, args,
          affine=affine, track_running_stats=track_running_stats)

    self.theta_map = lambda x: torch.softmax(x, dim=-1)
    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

    C_prev, num_edge, edge2index = C, None, None
    self.cells = nn.ModuleList()
    for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
      if reduction:
        cell = ResNetBasicblock(C_prev, C_curr, 2)
      else:
        cell = SearchCell(C_prev, C_curr, 1, max_nodes, search_space, affine, track_running_stats, beta)
        if num_edge is None: num_edge, edge2index = cell.num_edges, cell.edge2index
        else: assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
      self.cells.append( cell )
      C_prev = cell.out_dim

  def get_theta(self):
    return self.theta_map(self._arch_parameters).cpu()

  def set_beta(self, beta):
    for cell in self.cells:
        cell.beta = beta

  def forward(self, inputs):
    weights = self.theta_map(self._arch_parameters)
    feature = self.stem(inputs)

    for i, cell in enumerate(self.cells):
      if isinstance(cell, SearchCell):
        feature = cell(feature, weights)
      else:
        feature = cell(feature)

    out = self.lastact(feature)
    out = self.global_pooling( out )
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)

    return logits
