import torch
from torch import nn
import torch_sparse
import torch.nn.functional as F
from .base_classes import ODEFunc
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils.loop import add_remaining_self_loops,remove_self_loops
import numpy as np


def batch_jacobian(func, x, create_graph=False):

  return torch.autograd.functional.jacobian(func, x, create_graph=create_graph).permute(1, 2, 0)





# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.

class H_gcn(nn.Module):
  """"replace this module by a aggregation function """

  def __init__(self, size_in):
    super().__init__()
    self.dim = size_in

    self.layer1 =GCNConv(size_in*2, size_in, normalize=True)

    self.layer2 =GCNConv(size_in,1, normalize=True)
    self.dropout = nn.Dropout(p=0.4)
    self.linear = nn.Linear(size_in, 1)
  def forward(self, x,edge_index):
    #
    out = self.layer1(x,edge_index)
    out = torch.tanh(out)
    out = self.layer2(out,edge_index)

    out = torch.norm(out, dim=0)

    return out

class HAMGCNFunc_VAN(ODEFunc):

  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, device):
    super(HAMGCNFunc_VAN, self).__init__(opt, device)
    self.in_features = in_features
    self.out_features = out_features
    self.H = H_gcn(in_features ).to(device)


  def forward(self, t, x_full):  # the t param is needed by the ODE solver.
    f_full = batch_jacobian(lambda xx: self.H(xx, self.edge_index), x_full, create_graph=True).squeeze()
    dx = f_full[..., self.in_features:]
    dv = -1 * f_full[..., 0:self.in_features]
    if self.opt['add_source']:
      dx = (1. - torch.sigmoid(self.beta_train)) * dx + torch.sigmoid(self.beta_train) * self.x0[:, self.opt['hidden_dim']:]
    f = torch.hstack([dx, dv])

    return f
