import torch
from torch import nn
from torch_geometric.utils import softmax
import torch_sparse
from torch_geometric.utils.loop import add_remaining_self_loops
import numpy as np
# from data import get_dataset

from .function_transformer_attention import SpGraphTransAttentionLayer
from torch.nn.utils import spectral_norm

from torch_geometric.nn.conv import MessagePassing

class ODEFunc(MessagePassing):

  # currently requires in_features = out_features
  def __init__(self, opt, device):
    super(ODEFunc, self).__init__()
    self.opt = opt
    self.device = device
    self.edge_index = None
    self.edge_weight = None
    self.attention_weights = None
    self.alpha_train = nn.Parameter(torch.tensor(0.0))
    self.beta_train = nn.Parameter(torch.tensor(0.0))
    self.x0 = None
    self.nfe = 0
    self.alpha_sc = nn.Parameter(torch.ones(1))
    self.beta_sc = nn.Parameter(torch.ones(1))


class ODEFuncTransformerAtt_GRAND(ODEFunc):

  def __init__(self, in_features, out_features, opt,  device):
    super(ODEFuncTransformerAtt_GRAND, self).__init__(opt,  device)

    # if opt['self_loop_weight'] > 0:
    #   self.edge_index, self.edge_weight = add_remaining_self_loops(data.edge_index, data.edge_attr,
    #                                                                fill_value=opt['self_loop_weight'])
    # else:
    #   self.edge_index, self.edge_weight = data.edge_index, data.edge_attr
    self.multihead_att_layer = SpGraphTransAttentionLayer(in_features, out_features, opt,
                                                          device, edge_weights=self.edge_weight).to(device)

  def multiply_attention(self, x, attention, v=None):
    # todo would be nice if this was more efficient
    if self.opt['mix_features']:
      vx = torch.mean(torch.stack(
        [torch_sparse.spmm(self.edge_index, attention[:, idx], v.shape[0], v.shape[0], v[:, :, idx]) for idx in
         range(self.opt['heads'])], dim=0),
        dim=0)
      ax = self.multihead_att_layer.Wout(vx)
    else:
      mean_attention = attention.mean(dim=1)
      ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
    return ax

  def forward(self, t, x):  # t is needed when called by the integrator
    # if self.nfe > self.opt["max_nfe"]:
    #   raise MaxNFEException

    # self.nfe += 1
    # if t!=0:
    #   #add gaussian noise to the input
    #     x = x + torch.randn_like(x) * 0.01

    # x = x + torch.randn_like(x) * 0.01
    attention, values = self.multihead_att_layer(x, self.edge_index)
    ax = self.multiply_attention(x, attention, values)

    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train
    f = alpha * (ax - x)
    if self.opt['add_source']:
      f = f + self.beta_train * self.x0
    return f

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



#
# if __name__ == '__main__':
#   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#   opt = {'dataset': 'Cora', 'self_loop_weight': 1, 'leaky_relu_slope': 0.2, 'heads': 2, 'K': 10,
#          'attention_norm_idx': 0, 'add_source': False,
#          'alpha_dim': 'sc', 'beta_dim': 'sc', 'max_nfe': 1000, 'mix_features': False
#          }
#   dataset = get_dataset(opt, '../data', False)
#   t = 1
#   func = ODEFuncTransformerAtt(dataset.data.num_features, 6, opt, dataset.data, device)
#   out = func(t, dataset.data.x)
