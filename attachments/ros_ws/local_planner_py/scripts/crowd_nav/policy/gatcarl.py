# coding=utf-8
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import logging
from crowd_nav.policy.cadrl import mlp, mlp2
from crowd_nav.policy.multi_human_rl import MultiHumanRL

import torch.nn.functional as F
from .layers import GraphAttentionLayer
import numpy as np

class ValueNetwork(nn.Module): 
    def __init__(self, nfeat, nhid, nclass, nagents, nbatchsize, dropout, alpha, nheads, self_state_dim, mlp3_dims, lstm_hidden_dim):
        """
        Dense version of GATCARL.
        nfeat： number of features
        nhid: Number of hidden units
        nclass: Number of classes(设计为用于预测agents运动的参数)
        """
        super().__init__()
        self.dropout = dropout
        self.attention_weights = None
        self.nagents = nagents
        self.self_state_dim = self_state_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.nhid = nhid
        self.nbatchsize = nbatchsize
        adj = np.kron(np.diag(np.ones((self.nbatchsize))), np.ones((self.nagents,self.nagents),dtype='int'))
        self.adj = torch.FloatTensor(adj - np.diag(np.diag(adj))).cuda()
        self.attentions = [GraphAttentionLayer(nfeat, nhid, nagents, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, self.nagents, dropout=dropout, alpha=alpha, concat=False)

        self.lstm = nn.LSTM(nclass, lstm_hidden_dim, batch_first=True)
        # mlp3_input_dim = nclass + self.self_state_dim
        mlp3_input_dim = lstm_hidden_dim + self.self_state_dim
        self.mlp3 = mlp2(mlp3_input_dim, mlp3_dims)

    def forward(self, x):
        size = x.shape
        
        adj = torch.FloatTensor(np.kron(np.diag(np.ones((size[0]))), np.ones((self.nagents,self.nagents),dtype='int'))).cuda() if size[0] != self.nbatchsize else self.adj
        self_state = x[:, 0, :self.self_state_dim]

        x = F.dropout(x.view((-1, size[2])), self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        weights = F.log_softmax(x, dim=1).view(size[0], size[1], -1) #100,5,1
        # self.attention_weights = weights[0, :, 0].data.cpu().numpy() 

        # output feature is a linear combination of input features
        features = x.view(size[0], size[1], -1)
        # for converting to onnx
        # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        # weighted_feature = torch.sum(torch.mul(weights, features), dim=1) 
        weighted_feature = torch.mul(weights, features)
        
        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim).cuda()
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim).cuda()
        output, (hn, cn) = self.lstm(weighted_feature, (h0, c0))
        hn = hn.squeeze(0)

        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, hn], dim=1)
        mean_flag = False
        original_len = joint_state.shape[0]
        if (original_len % 2) == 1:
            joint_state_cat = torch.cat([joint_state,joint_state], dim=0)
            mean_flag = True
        else:
            joint_state_cat = joint_state
        value = self.mlp3(joint_state_cat)
        if mean_flag:
            value_trunc = value[0:original_len,:]
        else:
            value_trunc = value 
        return value_trunc

class GATCARL(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'GATCARL'
    
    def configure(self, config):
        # parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
        # parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
        # parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
        # parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
        # parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
        # parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
        self.set_common_parameters(config)
        nfeat = config.getint('gatcarl','nfeat')
        nhid = config.getint('gatcarl','nhid')
        nclass = config.getint('gatcarl','nclass')
        dropout = config.getfloat('gatcarl','dropout')
        alpha = config.getfloat('gatcarl','alpha')
        nheads = config.getint('gatcarl','nheads')
        global_state_dim = config.getint('gatcarl', 'global_state_dim')
        # self.with_om = config.getboolean('gatcarl', 'with_om')
        nbatchsize = config.getint('gatcarl','nbatchsize')
        nagents = nheads
        mlp3_dims = [int(x) for x in config.get('sarl', 'mlp3_dims').split(', ')]
        # if self.with_om:
        #     self.model = SpValueNetwork(nfeat,nhid,nclass,nagents, dropout,alpha,nheads, self.self_state_dim, mlp3_dims, global_state_dim)
        # else:
        self.model = ValueNetwork(nfeat,nhid,nclass,nagents, nbatchsize, dropout,alpha,nheads, self.self_state_dim, mlp3_dims, global_state_dim)
        
        
    def get_attention_weights(self):
        return self.model.attention_weights
