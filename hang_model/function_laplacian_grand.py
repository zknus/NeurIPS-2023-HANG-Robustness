import torch
from torch import nn
import torch_sparse


from torch_geometric.utils.loop import add_remaining_self_loops,remove_self_loops
from torch_geometric.utils import get_laplacian
from torch_geometric.nn.conv import MessagePassing
# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.

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

class LaplacianODEFuncGRAND(ODEFunc):

  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, device):
    super(LaplacianODEFuncGRAND, self).__init__(opt, device)

    self.in_features = in_features
    self.out_features = out_features
    self.alpha_sc = nn.Parameter(torch.ones(1))
    self.beta_sc = nn.Parameter(torch.ones(1))


  def sparse_multiply(self, x):
    if self.opt['block'] in ['attention']:  # adj is a multihead attention
      mean_attention = self.attention_weights.mean(dim=1)
      ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
    elif self.opt['block'] in ['mixed', 'hard_attention']:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.attention_weights, x.shape[0], x.shape[0], x)
    else:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
    return ax

  def forward(self, t, x):  # the t param is needed by the ODE solver.
    # if t!=0:
    #   #add gaussian noise to the input
    #     x = x + torch.randn_like(x) * 0.01
    ax = self.sparse_multiply(x)

    # ax = torch.cat([x, ax], axis=1)
    # ax = self.lin2(ax)

    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train

    f = alpha * (ax - x)
    if self.opt['add_source']:
      f = f + self.beta_train * self.x0
    return f
