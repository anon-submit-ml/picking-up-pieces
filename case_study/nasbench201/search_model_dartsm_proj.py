import torch
import torch.nn as nn
from .cell_operations import ResNetBasicblock
from .search_cells import NAS201SearchCell_Minus as SearchCell
from .search_model import TinyNetwork as TinyNetwork
from .genotypes        import Structure


class TinyNetworkDartsMinusProj(TinyNetwork):
  def __init__(self, C, N, max_nodes, num_classes, criterion, search_space, args,
               affine=False, track_running_stats=True, beta=1.0):
    super(TinyNetworkDartsMinusProj, self).__init__(C, N, max_nodes, num_classes, criterion, search_space, args,
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

    #### for edgewise projection
    self.candidate_flags = torch.tensor(len(self._arch_parameters) * [True], requires_grad=False, dtype=torch.bool).cuda()
    self.proj_weights = torch.zeros_like(self._arch_parameters)

  def project_op(self, eid, opid):
      self.proj_weights[eid][opid] = 1 ## hard by default
      self.candidate_flags[eid] = False

  def get_projected_weights(self):
      weights = self.theta_map(self._arch_parameters)

      ## proj op
      for eid in range(len(self._arch_parameters)):
        if not self.candidate_flags[eid]:
          weights[eid].data.copy_(self.proj_weights[eid])

      return weights

  def forward(self, inputs, weights=None):
    if weights is None:
      weights = self.get_projected_weights()

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

  #### utils
  def get_theta(self):
    return self.get_projected_weights()

  def set_beta(self, beta):
    for cell in self.cells:
        cell.beta = beta

  def arch_parameters(self):
    return [self._arch_parameters]

  def set_arch_parameters(self, new_alphas):
    for eid, alpha in enumerate(self.arch_parameters()):
      alpha.data.copy_(new_alphas[eid])
  
  def genotype(self):
    proj_weights = self.get_projected_weights()

    genotypes = []
    for i in range(1, self.max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        with torch.no_grad():
          weights = proj_weights[ self.edge2index[node_str] ]
          op_name = self.op_names[ weights.argmax().item() ]
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return Structure( genotypes )
