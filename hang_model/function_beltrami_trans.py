import torch
from torch import nn
from torch_geometric.utils import softmax
import torch_sparse
from torch_geometric.utils.loop import add_remaining_self_loops
import numpy as np
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

  def __repr__(self):
    return self.__class__.__name__
class ODEFuncBeltramiAtt(ODEFunc):

  def __init__(self, in_features, out_features, opt,  device):
    super(ODEFuncBeltramiAtt, self).__init__(opt,  device)

    # if opt['self_loop_weight'] > 0:
    #   self.edge_index, self.edge_weight = add_remaining_self_loops(data.edge_index, data.edge_attr,
    #                                                                fill_value=opt['self_loop_weight'])
    # else:
    #   self.edge_index, self.edge_weight = data.edge_index, data.edge_attr
    # print("self.edge_index: ", self.edge_index.shape)
    self.in_features = in_features
    self.out_features = out_features
    self.multihead_att_layer = SpGraphTransAttentionLayer(in_features, out_features,  opt,device,edge_weights=self.edge_weight).to(
      device)
    self.device = device

  def multiply_attention(self, x, attention, v=None):
    num_heads = 4
    mix_features = 0
    if mix_features:
      vx = torch.mean(torch.stack(
        [torch_sparse.spmm(self.edge_index, attention[:, idx], v.shape[0], v.shape[0], v[:, :, idx]) for idx in
         range(num_heads)], dim=0),
        dim=0)
      ax = self.multihead_att_layer.Wout(vx)
    else:
      mean_attention = attention.mean(dim=1)
      # mean_attention = self.edge_weight
      grad_x = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x) - x
      grad_x_abs = torch.abs(grad_x)
      grad_x_norm = torch.sqrt(torch.sum(torch.clamp(grad_x_abs * grad_x_abs, min=1e-1), 1))
      grad_x_norm_inv = 1 / grad_x_norm
      gu = grad_x_norm_inv[self.edge_index[0, :]]
      gv = grad_x_norm_inv[self.edge_index[1, :]]
      attention2 = gu * gu + gu * gv
      new_attn = mean_attention * softmax(attention2, self.edge_index[0])
      # Da = torch.diag(grad_x_norm_inv)
      W = torch.sparse.FloatTensor(self.edge_index, new_attn, (x.shape[0], x.shape[0])).coalesce()
      rowsum = torch.sparse.mm(W, torch.ones((W.shape[0], 1), device=self.device)).flatten()
      diag_index = torch.stack((torch.arange(x.shape[0]), torch.arange(x.shape[0]))).to(self.device)
      dx = torch_sparse.spmm(diag_index, rowsum, x.shape[0], x.shape[0], x)
      ax = torch_sparse.spmm(self.edge_index, new_attn, x.shape[0], x.shape[0], x)
    return ax - dx

  def forward(self, t, x):  # t is needed when called by the integrator
    # print("x.shape: ", x.shape)
    attention, values = self.multihead_att_layer(x, self.edge_index)
    ax = self.multiply_attention(x, attention, values)

    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train
    f = alpha * (ax - x)
    if self.opt['add_source']:
      f = f + self.beta_train * self.x0

    # f = ax - x
    return f

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpGraphTransAttentionLayer(nn.Module):
  """
  Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
  """

  def __init__(self, in_features, out_features, opt, device, concat=True, edge_weights=None):
    super(SpGraphTransAttentionLayer, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.alpha = opt['leaky_relu_slope']
    self.concat = concat
    self.device = device
    self.opt = opt
    self.h = int(opt['heads'])
    self.edge_weights = edge_weights

    try:
      self.attention_dim = opt['attention_dim']
    except KeyError:
      self.attention_dim = out_features

    assert self.attention_dim % self.h == 0, "Number of heads ({}) must be a factor of the dimension size ({})".format(
      self.h, self.attention_dim)
    self.d_k = self.attention_dim // self.h


    # print("in_features: ", in_features)
    # print("out_features: ", out_features)
    # print("self.attention_dim: ", self.attention_dim)

    self.Q = spectral_norm(nn.Linear(self.in_features, self.attention_dim))
    self.init_weights(self.Q)
    self.V = spectral_norm(nn.Linear(self.in_features, self.attention_dim))
    self.init_weights(self.V)
    self.K = spectral_norm(nn.Linear(self.in_features, self.attention_dim))
    self.init_weights(self.K)

    # self.Q = nn.Linear(in_features, self.attention_dim)
    # self.init_weights(self.Q)
    #
    # self.V = nn.Linear(in_features, self.attention_dim)
    # self.init_weights(self.V)
    #
    # self.K = nn.Linear(in_features, self.attention_dim)
    # self.init_weights(self.K)

    self.activation = nn.Sigmoid()  # nn.LeakyReLU(self.alpha)

    # self.Wout = nn.Linear(self.d_k, in_features)
    self.Wout = spectral_norm(nn.Linear(self.d_k, in_features))
    self.init_weights(self.Wout)

  def init_weights(self, m):
    if type(m) == nn.Linear:
      # nn.init.xavier_uniform_(m.weight, gain=1.414)
      # m.bias.data.fill_(0.01)
      nn.init.constant_(m.weight, 1e-5)

  def forward(self, x, edge):
    """
    x might be [features, augmentation, positional encoding, labels]
    """

    q = self.Q(x)
    k = self.K(x)
    v = self.V(x)

    # perform linear operation and split into h heads

    k = k.view(-1, self.h, self.d_k)
    q = q.view(-1, self.h, self.d_k)
    v = v.view(-1, self.h, self.d_k)

    # transpose to get dimensions [n_nodes, attention_dim, n_heads]

    k = k.transpose(1, 2)
    q = q.transpose(1, 2)
    v = v.transpose(1, 2)

    src = q[edge[0, :], :, :]
    dst_k = k[edge[1, :], :, :]



    prods = torch.sum(src * dst_k, dim=1) / np.sqrt(self.d_k)


    if self.opt['reweight_attention'] and self.edge_weights is not None:
      prods = prods * self.edge_weights.unsqueeze(dim=1)

    attention = softmax(prods, edge[self.opt['attention_norm_idx']])
    return attention, (v, prods)

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  opt = {'dataset': 'Cora', 'self_loop_weight': 1, 'leaky_relu_slope': 0.2, 'heads': 2, 'K': 10,
         'attention_norm_idx': 0, 'add_source': False,
         'alpha_dim': 'sc', 'beta_dim': 'sc', 'max_nfe': 1000, 'mix_features': False
         }
  dataset = get_dataset(opt, '../data', False)
  t = 1
  func = ODEFuncTransformerAtt(dataset.data.num_features, 6, opt, dataset.data, device)
  out = func(t, dataset.data.x)
