import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable
import torch

from crowd_nav.policy.cadrl import mlp, mlp2
from crowd_nav.policy.multi_human_rl import MultiHumanRL

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, joint_state_dim, in_mlp_dims, sort_mlp_dims, action_dims, sort_attention_dims, nheads, dropout, with_global_state):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.self_state_dim = self_state_dim
        self.global_state_dim = in_mlp_dims[-1]
        self.sort_mlp_global_state_dim = sort_mlp_dims[-1]
        self.joint_state_dim = joint_state_dim
        self.in_mlp_dims = in_mlp_dims
        self.heads = nheads
        self.dropout = dropout
        self.input_dim = input_dim
        self.lstm_hidden_dim = in_mlp_dims[-1]*2
        self.with_global_state = with_global_state

        self.in_mlp = mlp(self.input_dim, in_mlp_dims, last_relu=True)
        # avg+mlp1
        if self.with_global_state:
            self.sort_mlp = mlp(self.in_mlp_dims[-1]*2+self.joint_state_dim, sort_mlp_dims)     
        else:
            self.sort_mlp = mlp(self.in_mlp_dims[-1]*2, sort_mlp_dims)
        # avg+mlp2

        # (avg+mlp2)
        self.sort_mlp_attention = mlp(sort_mlp_dims[-1]*2, sort_attention_dims)    
        # self.attention = mlp(sort_attention_dims[-1]*2, attention_dims)
        # add a soft_max layer after soft_mlp_attentions
        self.lstm = nn.LSTM(sort_mlp_dims[-1]*2,  self.lstm_hidden_dim, batch_first=True)

        action_input_dim =  self.lstm_hidden_dim + self.self_state_dim # 50 + 6
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
        global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
            contiguous().view(-1, self.global_state_dim)  ##batch_sz*num_agents*in_mlp_dims[-1]

        if self.with_global_state:
            sort_mlp_input = torch.cat([in_mlp_output, global_state, state_att], dim=1) #batch_sz*num_agents*(in_mlp_dims[-1]*2 + self_state_size)
        else:
            sort_mlp_input = torch.cat([in_mlp_output, global_state], dim=1)  #batch_sz*num_agents*(in_mlp_dims[-1]*2)

        sort_mlp_output = self.sort_mlp(sort_mlp_input)
        sort_mlp_global_state = torch.mean(sort_mlp_output.view(size[0], size[1], -1), 1, keepdim=True) #100,1,100
        sort_mlp_global_state = sort_mlp_global_state.expand((size[0], size[1], self.sort_mlp_global_state_dim)).\
            contiguous().view(-1, self.sort_mlp_global_state_dim)  #500,100
        sort_mlp_input = torch.cat([sort_mlp_output, sort_mlp_global_state], dim=1)
        sort_mlp_attention_output = self.sort_mlp_attention(sort_mlp_input)
        # sort_attention_score = selt.attention(sort_mlp_attention_output)
        scores = sort_mlp_attention_output.view(size[0],size[1],1).squeeze(dim=2) #100,5

        # masked softmax
        # weights = softmax(scores, dim=1).unsqueeze(2)
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

         # output feature is a linear combination of input features
        features = sort_mlp_input.view(size[0], size[1], -1) #100,5,50
        weighted_feature = torch.mul(weights, features)#(100,5,1),(100,5,50)

        lstm_input = weighted_feature.view(size[0],size[1],-1)
        h0 = torch.zeros( 1, size[0], self.lstm_hidden_dim).cuda()
        c0 = torch.zeros( 1, size[0], self.lstm_hidden_dim).cuda()
        output, (hn, cn) = self.lstm(lstm_input, (h0, c0))
        hn = hn.squeeze(0)
    
        # concatenate agent's state with global weighted humans' state
        
        joint_state = torch.cat([self_state, hn], dim=1)
        value = self.action_mlp(joint_state)
        value = value.view(size[0],-1)
        return value

    ##Version:0##
    # def forward(self, state):
    #     size = state.shape
    #     self_state = state[:, :, :self.self_state_dim]
    #     mlp1_output = self.in_mlp(state.view((-1, size[2]))) #500,13; 500*100

    #     # compute attention scores
    #     global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True) #100,1,100
    #     global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
    #         contiguous().view(-1, self.global_state_dim)  #500,100
    #     attention_input = torch.cat([mlp1_output, global_state], dim=1) #500,200
    #     atn_output = torch.cat([att(F.dropout(attention_input, self.dropout, training=self.training)) for att in self.mh_attention], dim=1)
        
    #     weights = torch.mean(atn_output,dim=1,keepdim=True).view(size[0], size[1], -1)
    #     # weights = atn_output.view(size[0], size[1], -1) #100,10,1
        
    #     # output feature is a linear combination of input features
    #     # features = self_state
    #     weighted_feature = torch.mul(weights, self_state).view(size[0]*size[1],1,-1)
        
    #     h0 = torch.zeros( 1, size[0]*size[1], self.lstm_hidden_dim, device=self.device)
    #     c0 = torch.zeros( 1, size[0]*size[1], self.lstm_hidden_dim, device=self.device)
    #     output, (hn, cn) = self.lstm(weighted_feature, (h0, c0))
    #     hn = hn.squeeze(0)

    #     # concatenate agent's state with global weighted humans' state
    #     joint_state = torch.cat([self_state.view(size[0]*size[1],-1), hn], dim=1)
    #     joint_state = torch.cat([self_state.view(size[0]*size[1],-1), hn], dim=1)
    #     value = self.mlp3(joint_state)
    #     value = value.view(size[0],-1)[:,:1]
    #     return value


class GSCARL(MultiHumanRL):
    """
    Simple CommNet layer, similar to https://arxiv.org/pdf/1605.07736.pdf
    """
    def __init__(self):
        super().__init__()
        self.name = 'GSCARL'

    def configure(self, config):
        self.set_common_parameters(config)
        global_state_dim = config.getint('gscarl', 'global_state_dim')
        dropout = config.getfloat('gscarl','dropout')
        alpha = config.getfloat('gscarl','alpha')
        nheads = config.getint('gscarl','nheads')
        in_mlp_dims = [int(x) for x in config.get('gscarl', 'in_mlp_dims').split(', ')]
        sort_mlp_dims = [int(x) for x in config.get('gscarl', 'sort_mlp_dims').split(', ')]
        sort_attention_dims = [int(x) for x in config.get('gscarl', 'sort_attention_dims').split(', ')]
        action_dims = [int(x) for x in config.get('gscarl', 'action_dims').split(', ')]
        # attention_dims = [int(x) for x in config.get('comcarl', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('gscarl', 'with_om')
        with_global_state = config.getboolean('gscarl', 'with_global_state')
        self.multiagent_training = config.getboolean('gatcarl', 'multiagent_training')
        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, self.joint_state_dim, in_mlp_dims, sort_mlp_dims, action_dims, 
                                sort_attention_dims, nheads, dropout, with_global_state)

    def get_attention_weights(self):
        return self.model.attention_weights
        # pass

                
                                  





