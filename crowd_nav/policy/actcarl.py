import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable
import torch

from crowd_nav.policy.cadrl import mlp, mlp2, conv_mlp, conv_mlp2
from crowd_nav.policy.multi_human_rl import MultiHumanRL

def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            # layers.append(nn.BatchNorm1d(mlp_dims[i + 1]))
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRU, self).__init__()
        self.convz = nn.Linear(hidden_dim+input_dim, hidden_dim)
        self.convr = nn.Linear(hidden_dim+input_dim, hidden_dim)
        self.convq = nn.Linear(hidden_dim+input_dim, hidden_dim)


    def forward(self, x, h):
        hx = torch.cat([h,x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h


class ATCBasic(nn.Module):
    def __init__(self, input_dim, rnn_hidden_dim, max_ponder=3, epsilon=0.05):
        super(ATCBasic, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.epsilon = epsilon
        self.rnn_cell = GRU(input_dim, rnn_hidden_dim)
        self.max_ponder = max_ponder
        self.ponder_linear = nn.Linear(self.rnn_hidden_dim, 1)
    
    def forward(self, input_, hx=None):
        # Pre-allocate variables
        # time_size, batch_size, input_dim_size = input_.size()
        selector = input_.data.new(input_.shape[0]).byte()
        # ponder_times = []
        accum_p = 0
        accum_hx = torch.zeros([input_.shape[0], self.rnn_hidden_dim]).cuda()
        step_count = 0
        # For each t
        for act_step in range(self.max_ponder):
            hx = self.rnn_cell(input_, hx)
            halten_p = torch.sigmoid(self.ponder_linear(hx)) # halten state
            accum_p += halten_p
            accum_hx += halten_p * hx
            step_count += 1
            selector = (accum_p < 1 - self.epsilon).data
            if not selector.any():
                break

        # ponder_times.append(step_count.data.cpu().numpy())
        hx = torch.mean(accum_hx, 1, True) / step_count
        return hx


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, joint_state_dim, in_mlp_dims, action_dims, with_dynamic_net=True, with_global_state=False):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.self_state_dim = self_state_dim
        self.global_state_dim = in_mlp_dims[-1]
        self.joint_state_dim = joint_state_dim
        self.in_mlp_dims = in_mlp_dims
        self.input_dim = input_dim
        self.gru_hidden_dim = in_mlp_dims[-1]*2 + self.joint_state_dim
        self.with_dynamic_net = with_dynamic_net
        self.with_global_state = with_global_state

        self.in_mlp = mlp2(self.input_dim, in_mlp_dims, last_relu=True) 
        if self.with_dynamic_net:
            # self.gru = GRU(self.in_mlp_dims[-1]*2, self.gru_hidden_dim)
            if self.with_global_state:
                self.sort_mlp = ATCBasic(self.joint_state_dim, self.gru_hidden_dim, epsilon=0.05)
                action_input_dim = self.gru_hidden_dim + self.self_state_dim # 64 + 6
            else:
                self.sort_mlp = ATCBasic(self.joint_state_dim, in_mlp_dims[-1], epsilon=0.05)
                action_input_dim = in_mlp_dims[-1] + self.self_state_dim # 64 + 6

        
        self.action_mlp = mlp(action_input_dim, action_dims) #56,150,100,100,1
        self.attention_weights = None

    def forward(self, state):
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        agents_state = state[:, :, self.self_state_dim:].view(size[0]*size[1],-1) #batch_sz*num_agents*num_features
        state_att = state.view(-1,size[2])

        in_mlp_output = self.in_mlp(state_att) #batch_sz*num_agents*in_mlp_dims[-1]

        # compute attention scores
        global_state = torch.mean(in_mlp_output.view(size[0], size[1], -1), 1, keepdim=True) #batch_sz*1*in_mlp_dims[-1]
        global_state = global_state.expand((size[0], size[1], self.global_state_dim)).contiguous().view(-1, self.global_state_dim)  ##batch_sz*num_agents*in_mlp_dims[-1]

        if self.with_global_state:
            sort_mlp_input = torch.cat([in_mlp_output, global_state, state_att], dim=1) #batch_sz*num_agents*(in_mlp_dims[-1]*2 + self_state_size)
        else:
            sort_mlp_input = in_mlp_output #batch_sz*num_agents*(in_mlp_dims[-1]*2)
        # sort_mlp_input = torch.cat([in_mlp_output, global_state], dim=1)
        scores = self.sort_mlp(state_att, sort_mlp_input)

        # masked softmax
        # weights = softmax(scores, dim=1).unsqueeze(2)
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).view(size[0], size[1], -1)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

         # output feature is a linear combination of input features
        features = sort_mlp_input.view(size[0], size[1], -1) #100,5,50
        # weighted_feature = torch.mul(weights, features).view(size[0]*size[1], -1)#(100,5,1),(100,5,50)
    
        # concatenate agent's state with global weighted humans' state
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        value = self.action_mlp(joint_state)
        value = value.view(size[0],-1)
        return value


class ACTCARL(MultiHumanRL):
    """
    Simple Adaptive Computation Time Model, similar to https://arxiv.org/pdf/1603.08983.pdf
    """
    def __init__(self):
        super().__init__()
        self.name = 'ACTCARL'

    def configure(self, config):
        self.set_common_parameters(config)
        in_mlp_dims = [int(x) for x in config.get('actcarl', 'in_mlp_dims').split(', ')]
        action_dims = [int(x) for x in config.get('actcarl', 'action_dims').split(', ')]
        self.with_om = config.getboolean('actcarl', 'with_om')
        with_dynamic_net = config.getboolean('actcarl', 'with_dynamic_net')
        with_global_state = config.getboolean('actcarl', 'with_global_state')

        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, self.joint_state_dim, in_mlp_dims, action_dims, with_dynamic_net, with_global_state)

        self.multiagent_training = config.getboolean('actcarl', 'multiagent_training')
        # logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))
        # logging.info('Policy: {} {} interaction state'.format(self.name, 'w/' if with_interaction else 'w/o'))

    def get_attention_weights(self):
        return self.model.attention_weights