import torch
from torch import nn
from torch_geometric.utils import softmax
import torch_sparse
import torch.nn.functional as F
from torch_geometric.utils.loop import add_remaining_self_loops

from .base_classes import ODEFunc
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType, OptTensor)
from torch import Tensor
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing,GATConv
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.norm import PairNorm
from torch_scatter import scatter

class ODEFuncAttNorm(ODEFunc):

  def __init__(self, in_features, out_features, opt,  device):
    super(ODEFuncAttNorm, self).__init__(opt,  device)

    # if opt['self_loop_weight'] > 0:
    #   self.edge_index, self.edge_weight = add_remaining_self_loops(data.edge_index, data.edge_attr,
    #                                                                fill_value=opt['self_loop_weight'])
    # else:
    #   self.edge_index, self.edge_weight = data.edge_index, data.edge_attr

    self.multihead_att_layer = SpGraphAttentionLayer(in_features, out_features, opt,
                                                     device).to(device)
    try:
      self.attention_dim = opt['attention_dim']
    except KeyError:
      self.attention_dim = out_features

    assert self.attention_dim % opt['heads'] == 0, "Number of heads must be a factor of the dimension size"
    self.d_k = self.attention_dim // opt['heads']

    self.multihead_att_layer_norm =DeepGATConv(in_features, out_features, heads=opt['heads'],concat=False,dropout=0.3).to(device)

  def multiply_attention(self, x, attention, wx):
    if self.opt['mix_features']:
      wx = torch.mean(torch.stack(
        [torch_sparse.spmm(self.edge_index, attention[:, idx], wx.shape[0], wx.shape[0], wx) for idx in
         range(self.opt['heads'])], dim=0),
        dim=0)
      ax = torch.mm(wx, self.multihead_att_layer.Wout)
    else:
      ax = torch.mean(torch.stack(
        [torch_sparse.spmm(self.edge_index, attention[:, idx], x.shape[0], x.shape[0], x) for idx in
         range(self.opt['heads'])], dim=0),
        dim=0)
    return ax

  def forward(self, t, x_full):  # t is needed when called by the integrator
    x = x_full[:, :self.opt['hidden_dim']]
    y = x_full[:, self.opt['hidden_dim']:]

    # attention, wy = self.multihead_att_layer(y, self.edge_index)
    # ay = self.multiply_attention(y, attention, wy)
    # # todo would be nice if this was more efficient


    ay = self.multihead_att_layer_norm(y, self.edge_index)

    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train
    f = (ay - y - x)
    if self.opt['add_source']:
      f = (1. - torch.sigmoid(self.beta_train)) * f + torch.sigmoid(self.beta_train) * self.x0[:, self.opt['hidden_dim']:]
    f = torch.cat([f, (1. - torch.sigmoid(self.beta_train2)) * alpha * x + torch.sigmoid(self.beta_train2) * self.x0[:,:self.opt['hidden_dim']]],dim=1)
    return f

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpGraphAttentionLayer(nn.Module):
  """
  Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
  """

  def __init__(self, in_features, out_features, opt, device, concat=True):
    super(SpGraphAttentionLayer, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.alpha = opt['leaky_relu_slope']
    self.concat = concat
    self.device = device
    self.opt = opt
    self.h = opt['heads']

    try:
      self.attention_dim = opt['attention_dim']
    except KeyError:
      self.attention_dim = out_features

    assert self.attention_dim % opt['heads'] == 0, "Number of heads must be a factor of the dimension size"
    self.d_k = self.attention_dim // opt['heads']

    self.W = nn.Parameter(torch.zeros(size=(in_features, self.attention_dim))).to(device)
    nn.init.xavier_normal_(self.W.data, gain=1.414)

    self.Wout = nn.Parameter(torch.zeros(size=(self.attention_dim, self.in_features))).to(device)
    nn.init.xavier_normal_(self.Wout.data, gain=1.414)

    self.a = nn.Parameter(torch.zeros(size=(2 * self.d_k, 1, 1))).to(device)
    nn.init.xavier_normal_(self.a.data, gain=1.414)

    self.leakyrelu = nn.LeakyReLU(self.alpha)

  def forward(self, x, edge):
    wx = torch.mm(x, self.W)  # h: N x out
    h = wx.view(-1, self.h, self.d_k)
    h = h.transpose(1, 2)

    # Self-attention on the nodes - Shared attention mechanism
    edge_h = torch.cat((h[edge[0, :], :, :], h[edge[1, :], :, :]), dim=1).transpose(0, 1).to(
      self.device)  # edge: 2*D x E
    edge_e = self.leakyrelu(torch.sum(self.a * edge_h, dim=0)).to(self.device)
    attention = softmax(edge_e, edge[self.opt['attention_norm_idx']])
    return attention, wx

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class DeepGATConv(MessagePassing):

  _alpha: OptTensor

  def __init__(self, in_channels: Union[int, Tuple[int, int]],
               out_channels: int, heads: int = 1, concat: bool = True,
               negative_slope: float = 0.2, dropout: float = 0.,
               add_self_loops: bool = True, bias: bool = True, norm="LipschitzNorm", **kwargs):
    super(DeepGATConv, self).__init__(aggr='add', node_dim=0, **kwargs)

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.heads = heads
    self.concat = concat
    self.negative_slope = negative_slope
    self.dropout = dropout
    self.add_self_loops = add_self_loops
    self.num_nodes, self.num_features, self.degrees = None, None, None
    self.norm = LipschitzNorm(scale_individually=False)  # normalization method: {lipschitznorm, neighbornorm, pairnorm, None}

    if isinstance(in_channels, int):
      self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
      self.lin_r = self.lin_l
    else:
      self.lin_l = Linear(in_channels[0], heads * out_channels, False)
      self.lin_r = Linear(in_channels[1], heads * out_channels, False)
    self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
    self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

    if bias and concat:
      self.bias = Parameter(torch.Tensor(heads * out_channels))
    elif bias and not concat:
      self.bias = Parameter(torch.Tensor(out_channels))
    else:
      self.register_parameter('bias', None)

    self._alpha = None

    self.reset_parameters()

  def reset_parameters(self):
    glorot(self.lin_l.weight)
    glorot(self.lin_r.weight)
    glorot(self.att_l)
    glorot(self.att_r)
    zeros(self.bias)

  def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
              size: Size = None, return_attention_weights=None):
    r"""
    Args:
        return_attention_weights (bool, optional): If set to :obj:`True`,
            will additionally return the tuple
            :obj:`(edge_index, attention_weights)`, holding the computed
            attention weights for each edge. (default: :obj:`None`)
    """
    self.num_nodes, self.num_features = x.shape[0], x.shape[1]
    self.edge_index = edge_index

    H, C = self.heads, self.out_channels

    x_l: OptTensor = None
    x_r: OptTensor = None
    alpha_l: OptTensor = None
    alpha_r: OptTensor = None

    if isinstance(x, Tensor):
      assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
      x_l = x_r = self.lin_l(x).view(-1, H, C)  # Theta parameter: lin_l, lin_r

      alpha_l = (x_l * self.att_l).sum(dim=-1)
      alpha_r = (x_r * self.att_r).sum(dim=-1)


    else:
      x_l, x_r = x[0], x[1]
      assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
      x_l = self.lin_l(x_l).view(-1, H, C)  # Theta parameter: lin_l, lin_r
      alpha_l = (x_l * self.att_l).sum(dim=-1)
      if x_r is not None:
        x_r = self.lin_r(x_r).view(-1, H, C)
        alpha_r = (x_r * self.att_r).sum(dim=-1)

    assert x_l is not None
    assert alpha_l is not None

    if self.add_self_loops:
      if isinstance(edge_index, Tensor):
        num_nodes = x_l.size(0)
        if x_r is not None:
          num_nodes = min(num_nodes, x_r.size(0))
        if size is not None:
          num_nodes = min(size[0], size[1])
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
      elif isinstance(edge_index, SparseTensor):
        edge_index = set_diag(edge_index)

    # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)

    out = self.propagate(edge_index, x=(x_l, x_r),
                         alpha=(alpha_l, alpha_r), size=size)

    alpha = self._alpha
    self._alpha = None
    self.edge_index = None

    if self.concat:
      out = out.view(-1, self.heads * self.out_channels)
    else:
      out = out.mean(dim=1)

    if self.bias is not None:
      out += self.bias

    if isinstance(return_attention_weights, bool):
      assert alpha is not None
      if isinstance(edge_index, Tensor):
        return out, (edge_index, alpha)
      elif isinstance(edge_index, SparseTensor):
        return out, edge_index.set_value(alpha, layout='coo')
    else:
      return out

  def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
              index: Tensor, ptr: OptTensor,
              size_i: Optional[int]) -> Tensor:

    alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

    if self.norm is not None:
      alpha = self.norm(x_j, att=(self.att_l, self.att_r), alpha=alpha, index=index)

    alpha = F.leaky_relu(alpha, self.negative_slope)
    alpha = softmax(alpha, index, ptr, size_i)

    self._alpha = alpha
    alpha = F.dropout(alpha, p=self.dropout, training=self.training)

    return x_j * alpha.unsqueeze(-1)

  def __repr__(self):
    return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                         self.in_channels,
                                         self.out_channels, self.heads)

class LipschitzNorm(nn.Module):
  def __init__(self, att_norm=4, recenter=False, scale_individually=True, eps=1e-12):
    super(LipschitzNorm, self).__init__()
    self.att_norm = att_norm
    self.eps = eps
    self.recenter = recenter
    self.scale_individually = scale_individually

  def forward(self, x, att, alpha, index):
    att_l, att_r = att

    if self.recenter:
      mean = scatter(src=x, index=index, dim=0, reduce='mean')
      x = x - mean

    norm_x = torch.norm(x, dim=-1) ** 2
    max_norm = scatter(src=norm_x, index=index, dim=0, reduce='max').view(-1, 1)
    max_norm = torch.sqrt(max_norm[index] + norm_x)  # simulation of max_j ||x_j||^2 + ||x_i||^2

    # scaling_factor =  4 * norm_att , where att = [ att_l | att_r ]
    if self.scale_individually == False:
      norm_att = self.att_norm * torch.norm(torch.cat((att_l, att_r), dim=-1))
    else:
      norm_att = self.att_norm * torch.norm(torch.cat((att_l, att_r), dim=-1), dim=-1)

    alpha = alpha / (norm_att * max_norm + self.eps)
    return alpha

if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  opt = {'dataset': 'Cora', 'self_loop_weight': 1, 'leaky_relu_slope': 0.2, 'beta_dim': 'vc', 'heads': 2, 'K': 10, 'attention_norm_idx': 0,
         'add_source':False, 'alpha_dim': 'sc', 'beta_dim': 'vc', 'max_nfe':1000, 'mix_features': False}
  dataset = get_dataset(opt, '../data', False)
  t = 1
  func = ODEFuncAtt(dataset.data.num_features, 6, opt, dataset.data, device)
  out = func(t, dataset.data.x)
