import torch
from torch import nn
import torch_sparse
import torch.nn.functional as F
from .base_classes import ODEFunc
# from utils import MaxNFEException
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils.loop import add_remaining_self_loops,remove_self_loops
# from utils import get_rw_adj
import numpy as np
from torch_geometric.utils import softmax, degree
from torch.nn.utils import spectral_norm
from torch_geometric.nn.conv import GCNConv,GATConv
from torch_geometric.utils import to_dense_adj,get_laplacian

def batch_jacobian(func, x, create_graph=False):

  return torch.autograd.functional.jacobian(func, x, create_graph=create_graph)












# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.

class H_gcn(nn.Module):
  """"replace this module by a aggregation function """

  def __init__(self, size_in,device):
    super().__init__()
    self.dim = size_in


    self.dropout = nn.Dropout(p=0.4)
    self.gat = GATConv(self.dim, self.dim, heads=1, dropout=0.4, concat=False)
    self.device =device

    self.in_features = size_in




  def forward(self, x_full,edge_index,edge_weight):
    # x refers to q, y refers to p in the paper
    x = x_full[:, :self.in_features]
    y = x_full[:, self.in_features:]

    out = torch_sparse.spmm(edge_index, edge_weight, y.shape[0], y.shape[0], y)
    out = torch_sparse.spmm(edge_index, edge_weight, y.shape[0], y.shape[0], out)

    out = torch.matmul(y.T, out)
    # out = torch.matmul(torch.matmul(y.T, T_pq), y)

    # normalize out matrix by dividing by the number of nodes
    out = out / y.shape[0]

    out = torch.trace(out)

    U = self.gat(x, edge_index)
    out = out + torch.norm(U)
    return out

class HAMGCNFunc_QUAD(ODEFunc):

  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, device):
    super(HAMGCNFunc_QUAD, self).__init__(opt, device)
    self.in_features = in_features
    self.out_features = out_features
    self.H = H_gcn(in_features,device ).to(device)

  def forward(self, t, x_full):  # the t param is needed by the ODE solver.
    x = x_full[:, :self.opt['hidden_dim']]
    y = x_full[:, self.opt['hidden_dim']:]
    f_full = batch_jacobian(lambda xx: self.H(xx, self.edge_index, self.edge_weight), x_full,
                            create_graph=True).squeeze()
    dx = f_full[..., self.in_features:]
    dv = -1 * f_full[..., 0:self.in_features]
    if self.opt['add_source']:
      dx = (1. - torch.sigmoid(self.beta_train)) * dx + torch.sigmoid(self.beta_train) * self.x0[:, self.opt['hidden_dim']:]
    f = torch.hstack([dv, dx])

    return f



