import torch
from torch import nn
import torch.nn.functional as F
from .base_classes import BaseGNN
from .model_configurations import set_block, set_function


# Define the GNN model.
class GNN_graphcon_time(BaseGNN):
  def __init__(self, opt, dataset, device=torch.device('cpu')):
    super(GNN_graphcon_time, self).__init__(opt, dataset, device)
    if opt['block'] != 'constanttime':
        raise ValueError('block must be constanttime')
    self.f = set_function(opt)
    block = set_block(opt)
    time_tensor = torch.tensor([0, self.T]).to(device)
    self.odeblock = block(self.f, opt, device, t=time_tensor).to(device)
    # self.prelu = nn.PReLU()
    self.bn = nn.BatchNorm1d(opt['hidden_dim'])
  def forward(self, x,adj,tend=None):
    # Encode each node based on its feature.
    # if self.opt['use_labels']:
    #   y = x[:, self.num_features:]
    #   x = x[:, :self.num_features]
    # x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    # x = self.m1(x)
    if self.opt['use_mlp']:
      x = F.dropout(x, self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)
    # todo investigate if some input non-linearity solves the problem with smooth deformations identified in the ANODE paper

    # if self.opt['use_labels']:
    #   x = torch.cat([x, y], dim=-1)

    # if self.opt['batch_norm']:
    #   x = self.bn_in(x)

    # x = self.bn(x)
    # Solve the initial value problem of the ODE.
    if self.opt['augment']:
      c_aux = torch.zeros(x.shape).to(self.device)
      x = torch.cat([x, c_aux], dim=1)
    # if self.training:
    #   print('x before ode: ',x.shape)
    #   print('x before ode: ',x)
    vt = x.clone()
    x = torch.cat([x, vt], dim=-1)
    self.odeblock.set_x0(x)

    # if self.training and self.odeblock.nreg > 0:
    #   z, self.reg_states = self.odeblock(x)
    # else:
    #   z = self.odeblock(x)
    z = self.odeblock(x,adj,tend)
    # print('z after ode: ', z.shape)
    if self.opt['augment']:
      z = torch.split(z, x.shape[1] // 2, dim=1)[0]

    z = z[:, self.opt['hidden_dim']:]
    # if self.training:
    #   print('z after ode: ', z)

    # Activation.
    z = F.relu(z)
    # z = self.prelu(z)
    if self.opt['fc_out']:
      z = self.fc(z)
      z = F.relu(z)

    # Dropout.
    z = F.dropout(z, self.opt['dropout'], training=self.training)

    # Decode each node embedding to get node label.
    # z = self.m2(z)
    return z
