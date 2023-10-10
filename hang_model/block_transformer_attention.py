import torch
from .function_transformer_attention import SpGraphTransAttentionLayer
from .base_classes import ODEblock
from .utils import get_rw_adj, gcn_norm_fill_val
import numpy as np
from torch_geometric.utils.loop import add_remaining_self_loops

class AttODEblock(ODEblock):
  def __init__(self, odefunc, opt,  device, t=torch.tensor([0, 1]), gamma=0.5):
    super(AttODEblock, self).__init__(odefunc,  opt,  device, t)

    self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt, device)
    # self.odefunc.edge_index, self.odefunc.edge_weight = data.edge_index, edge_weight=data.edge_attr
    # edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
    #                                      fill_value=opt['self_loop_weight'],
    #                                      num_nodes=data.num_nodes,
    #                                      dtype=data.x.dtype)

    # self.odefunc.edge_index = edge_index.to(device)
    # self.odefunc.edge_weight = edge_weight.to(device)
    # self.reg_odefunc.odefunc.edge_index, self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_index, self.odefunc.edge_weight

    if(opt['method']=='symplectic_euler' or opt['method']=='leapfrog'):
      from odeint_geometric import odeint
    elif opt['adjoint']:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint
    self.train_integrator = odeint
    self.test_integrator = odeint
    self.set_tol()
    # parameter trading off between attention and the Laplacian
    self.multihead_att_layer = SpGraphTransAttentionLayer(opt['hidden_dim'], opt['hidden_dim'], opt,
                                                          device, edge_weights=self.odefunc.edge_weight).to(device)
    self.device = device
    self.opt = opt

  def get_attention_weights(self, x):
    attention, values = self.multihead_att_layer(x, self.odefunc.edge_index)
    return attention

  def forward(self, x_all, adj):
    x = x_all[:,:self.opt['hidden_dim']]
    y = x_all[:, self.opt['hidden_dim']:]
    t = self.t.type_as(x)


    # reg_states = tuple(torch.zeros(x.size(0)).to(x) for i in range(self.nreg))
    #
    # func = self.reg_odefunc if self.training and self.nreg > 0 else self.odefunc
    # state = (x_all,) + reg_states if self.training and self.nreg > 0 else x_all

    func = self.odefunc
    state = x_all

    self.edge_index = adj[0]
    self.edge_attr = adj[1]

    if self.opt['data_norm'] == 'rw':
      edge_index, edge_weight = get_rw_adj(self.edge_index, edge_weight=self.edge_attr, norm_dim=1,
                                           fill_value=self.opt['self_loop_weight'],
                                           num_nodes=self.opt['num_nodes'],
                                           dtype=x.dtype)
    else:
      edge_index, edge_weight = gcn_norm_fill_val(self.edge_index, edge_weight=self.edge_attr,
                                                  fill_value=self.opt['self_loop_weight'],
                                                  num_nodes=self.opt['num_nodes'],
                                                  dtype=x.dtype)

    if self.opt['self_loop_weight'] > 0:
      edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight,
                                                                   fill_value=self.opt['self_loop_weight'])
    else:
      edge_index, edge_weight = edge_index, edge_weight

    self.odefunc.edge_index = edge_index.to(self.device)
    self.odefunc.edge_weight = edge_weight.to(self.device)

    self.odefunc.attention_weights = self.get_attention_weights(y)
    # self.reg_odefunc.odefunc.attention_weights = self.odefunc.attention_weights
    integrator = self.train_integrator if self.training else self.test_integrator

    # print("ode solver: ", self.opt['method'])
    # print("ode step_size: ", self.opt['step_size'])
    # print("ode time: ", t)
    # print("ode adjoint: ", self.opt["adjoint"] )
    if self.opt["adjoint"] and self.training:
      state_dt = integrator(
        func, state, t,
        method=self.opt['method'],
        options={'step_size': self.opt['step_size']},
        adjoint_method=self.opt['adjoint_method'],
        adjoint_options={'step_size': self.opt['adjoint_step_size']},
        atol=self.atol,
        rtol=self.rtol,
        adjoint_atol=self.atol_adjoint,
        adjoint_rtol=self.rtol_adjoint)
    else:
      state_dt = integrator(
        func, state, t,
        method=self.opt['method'],
        options={'step_size': self.opt['step_size']},
        atol=self.atol,
        rtol=self.rtol)
    #
    # if self.training and self.nreg > 0:
    #   z = state_dt[0][1]
    #   reg_states = tuple(st[1] for st in state_dt[1:])
    #   return z, reg_states
    # else:
    #   z = state_dt[1]
    #   return z
    z = state_dt[1]
    return z

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"
