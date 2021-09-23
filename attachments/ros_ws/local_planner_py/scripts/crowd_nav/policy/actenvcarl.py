import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable
import torch

from crowd_nav.policy.cadrl import mlp, mlp2
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


class GRUEXND(nn.Module):
    def __init__(self, input_dim, hidden_dims, last_relu=True):
        super(GRUEXND, self).__init__()
        self.convz = mlp(input_dim+hidden_dims[-1], hidden_dims, last_relu=last_relu)
        self.convr = mlp(input_dim+hidden_dims[-1], hidden_dims, last_relu=last_relu)
        self.convq = nn.Linear(hidden_dims[-1]+input_dim, hidden_dims[-1])


    def forward(self, x, h):
        # print(x.device)
        # print(h.device)
        hx = torch.cat([h,x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h


class ATCBasic(nn.Module):
    def __init__(self, input_dim, rnn_hidden_dims, max_ponder=3, epsilon=0.05, last_relu=True):
        super(ATCBasic, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dims[-1]
        self.epsilon = epsilon
        self.rnn_cell = GRUEXND(input_dim, rnn_hidden_dims, last_relu)
        self.max_ponder = max_ponder
        self.ponder_linear = nn.Linear(rnn_hidden_dims[-1], 1)
        self.step_cnt = 0
        self.step2_cnt = 0
        self.step3_cnt = 0

    
    
    def forward(self, input, hx=None):
        # Pre-allocate variables
        # time_size, batch_size, input_dim_size = input_.size()
        size = input.shape
        input_ = input.view(-1, size[2])
        device = input_.device
        # print("Input on :", device)
        selector = input_.data.new(input_.shape[0]).byte()
        ponder_times = []
        accum_p = 0
        accum_hx = torch.zeros([input_.shape[0], self.rnn_hidden_dim]).to(device)
        step_count = 0
        self.step_cnt = 0
        self.step2_cnt = 0
        self.step3_cnt = 0
        # For each t
        for act_step in range(self.max_ponder):
            hx = self.rnn_cell(input_, hx)
            halten_p = torch.sigmoid(self.ponder_linear(hx)) # halten state
            accum_p += halten_p
            accum_hx += halten_p * hx
            step_count += 1
            selector = (accum_p < 1 - self.epsilon).data
            # if not selector.any():
            #     # print("step %d" % (step_count))
            #     break

        if step_count == 1:
            self.step_cnt = step_count
        elif step_count == 2:
            self.step2_cnt = step_count
        else:
            self.step3_cnt = step_count
        # ponder_times.append(step_count.data.cpu().numpy())
        # accum_hx = accum_hx.view(size[0], size[1], -1)
        # hx = torch.mean(accum_hx, 1, True) #[B, C]
        hx = accum_hx / step_count
        return hx


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, joint_state_dim, in_mlp_dims, sort_mlp_dims, sort_mlp_attention, action_dims, with_dynamic_net=True, with_global_state=True):
        super(ValueNetwork, self).__init__()
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.self_state_dim = self_state_dim
        self.global_state_dim = in_mlp_dims[-1]
        self.joint_state_dim = joint_state_dim
        self.in_mlp_dims = in_mlp_dims
        self.input_dim = input_dim
        self.lstm_hidden_dim = sort_mlp_attention[0]*2
        self.with_dynamic_net = with_dynamic_net
        self.with_global_state = with_global_state
        self.sort_mlp_attention = sort_mlp_attention
        self.sort_mlp_global_state_dim = sort_mlp_dims[-1]

        if self.with_dynamic_net:
            self.in_mlp = ATCBasic(self.input_dim, in_mlp_dims, epsilon=0.05, last_relu=True)
        else:
            self.in_mlp = mlp(self.input_dim, in_mlp_dims, last_relu=True)

        # avg+mlp1
        if self.with_global_state:
            self.sort_mlp = mlp(self.in_mlp_dims[-1]*2+self.joint_state_dim, sort_mlp_dims)     
        else:
            self.sort_mlp = mlp(self.in_mlp_dims[-1]+self.joint_state_dim, sort_mlp_dims)
        # avg+mlp2

        # (avg+mlp2)
        self.sort_mlp_attention = mlp(sort_mlp_dims[-1]*2, sort_mlp_attention)
        # add a soft_max layer after soft_mlp_attentions
        self.lstm = nn.LSTM(sort_mlp_dims[-1]*2,  self.lstm_hidden_dim, batch_first=True)

        action_input_dim = self.lstm_hidden_dim + self.self_state_dim # 50 + 6
        self.action_mlp = mlp(action_input_dim, action_dims) #56,150,100,100,1
        self.attention_weights = None
        self.step_cnt = 0
        self.step2_cnt = 0
        self.step3_cnt = 0


    def forward(self, state):
        # print("VN on:", self.device)
        # state = state.to(self.device)
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim].to(self.device)
        agents_state = state[:, :, self.self_state_dim:].view(size[0]*size[1],-1).to(self.device) #batch_sz*num_agents*num_features
        state_att = state.view(-1,size[2]).to(self.device)

        act_h0 = torch.zeros([size[0]*size[1], self.in_mlp_dims[-1]]).to(self.device)
        # print("act_h0 on:", act_h0.device)
        # print("state on:", state.device)
        in_mlp_output = self.in_mlp(state, act_h0) #batch_sz*num_agents*in_mlp_dims[-1]
        self.step_cnt = 0
        self.step2_cnt = 0
        self.step3_cnt = 0
        self.step_cnt = self.in_mlp.step_cnt
        self.step2_cnt = self.in_mlp.step2_cnt
        self.step3_cnt = self.in_mlp.step3_cnt
        global_state = torch.mean(in_mlp_output.view(size[0], size[1], -1), 1, keepdim=True) #batch_sz*1*in_mlp_dims[-1]
        global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
            contiguous().view(-1, self.global_state_dim)  ##batch_sz*num_agents*in_mlp_dims[-1]

        if self.with_global_state:
            sort_mlp_input = torch.cat([in_mlp_output, global_state, state_att], dim=1) #batch_sz*num_agents*(in_mlp_dims[-1]*2 + self_state_size)
        else:
            sort_mlp_input = torch.cat([in_mlp_output, state_att], dim=1)  #batch_sz*num_agents*(in_mlp_dims[-1]*2)
        sort_mlp_output = self.sort_mlp(sort_mlp_input)
        sort_mlp_global_state = torch.mean(sort_mlp_output.view(size[0], size[1], -1), 1, keepdim=True) #100,1,100
        sort_mlp_global_state = sort_mlp_global_state.expand((size[0], size[1], self.sort_mlp_global_state_dim)).\
            contiguous().view(-1, self.sort_mlp_global_state_dim)  #500,100
        sort_mlp_input = torch.cat([sort_mlp_output, sort_mlp_global_state], dim=1)
        sort_mlp_attention_output = self.sort_mlp_attention(sort_mlp_input)
        # sort_attention_score = selt.attention(sort_mlp_attention_output)
        scores = sort_mlp_attention_output.view(size[0],size[1],1).squeeze(dim=2) #100,5

        # masked softmax
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

         # output feature is a linear combination of input features
        features = sort_mlp_input.view(size[0], size[1], -1) #100,5,50
        weighted_feature = torch.mul(weights, features)#(100,5,1),(100,5,50)

        lstm_input = weighted_feature.view(size[0],size[1],-1)
        h0 = torch.zeros( 1, size[0], self.lstm_hidden_dim, device=self.device)
        c0 = torch.zeros( 1, size[0], self.lstm_hidden_dim, device=self.device)
        self.lstm.flatten_parameters() 
        output, (hn, cn) = self.lstm(lstm_input, (h0, c0))
        hn = hn.squeeze(0)
    
        # concatenate agent's state with global weighted humans' state
        
        joint_state = torch.cat([self_state, hn], dim=1)
        value = self.action_mlp(joint_state)
        value = value.view(size[0],-1)
        return value


class ACTENVCARL(MultiHumanRL):
    """
    Simple Adaptive Computation Time Model, similar to https://arxiv.org/pdf/1603.08983.pdf
    """
    def __init__(self):
        super(ACTENVCARL, self).__init__()
        self.name = 'ACTENVCARL'

    def configure(self, config):
        self.set_common_parameters(config)
        in_mlp_dims = [int(x) for x in config.get('actenvcarl', 'in_mlp_dims').split(', ')]
        # gru_hidden_dim = [int(x) for x in config.get('actenvcarl', 'gru_hidden_dim').split(', ')]
        sort_mlp_dims = [int(x) for x in config.get('actenvcarl', 'sort_mlp_dims').split(', ')]
        sort_mlp_attention = [int(x) for x in config.get('actenvcarl', 'sort_attention_dims').split(', ')]
        # aggregation_dims = [int(x) for x in config.get('actenvcarl', 'aggregation_dims').split(', ')]
        action_dims = [int(x) for x in config.get('actenvcarl', 'action_dims').split(', ')]
        self.with_om = config.getboolean('actenvcarl', 'with_om')
        with_dynamic_net = config.getboolean('actenvcarl', 'with_dynamic_net')
        with_global_state = config.getboolean('actenvcarl', 'with_global_state')

        # def __init__(self, input_dim, self_state_dim, joint_state_dim, in_mlp_dims, sort_mlp_dims, action_dims, with_dynamic_net=True):

        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, self.joint_state_dim, in_mlp_dims, sort_mlp_dims, sort_mlp_attention, action_dims, with_dynamic_net, with_global_state)

        self.multiagent_training = config.getboolean('actcarl', 'multiagent_training')
        # logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))
        # logging.info('Policy: {} {} interaction state'.format(self.name, 'w/' if with_interaction else 'w/o'))

    # def get_attention_weights(self):
    #     return self.model.attention_weights
    def get_step_count(self):
        return self.model.step_cnt
    def get_step2_count(self):
        return self.model.step2_cnt
    def get_step3_count(self):
        return self.model.step3_cnt