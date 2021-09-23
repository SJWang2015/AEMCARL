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

class AFAModule(nn.Module):
    def __init__(self, mlp, use_softmax=False):
        r"""
        :param mlp: mlp for learning weight
               mode: transformation or aggregation
        """
        super().__init__()
        self.mlp = mlp
        self.use_softmax = use_softmax

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        r"""
        Parameters
        ----------
        features : torch.Tensor
            (B, C, N, M) or (B, C, N)
        Returns
        -------
        new_features : torch.Tensor
            transformation: (B, C, N, M) or (B, C, N)
            aggregation: (B, C, N) or (B, C)
        """
        B, C, N = feature.size()
        feature = feature.view(B, C, N, 1).repeat(1, 1, 1, N)  # (BN, C, M, M)
        if feature.device.type == "cpu":
            feature = feature - feature.transpose(2, 3).contiguous() + torch.mul(feature, torch.eye(N).view(1, 1, N, N))  # (BN, C, M, M)
        else:
            feature = feature - feature.transpose(2, 3).contiguous() + torch.mul(feature, torch.eye(N).view(1, 1, N, N).cuda())  # (BN, C, M, M)
        weight = self.mlp(feature)
        if self.use_softmax:
            weight = F.softmax(weight, -1)
        # feature = (feature * weight).sum(-1).view(B, N, C).transpose(1, 2).contiguous()  # (B, C, N)
        feature = (feature * weight).sum(-1).view(B, N, C).contiguous()  # (B, N, C)
        return feature


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRU, self).__init__()
        self.convz = nn.Conv1d(hidden_dim+input_dim, hidden_dim, 1, bias=False)
        self.convr = nn.Conv1d(hidden_dim+input_dim, hidden_dim, 1, bias=False)
        self.convq = nn.Conv1d(hidden_dim+input_dim, hidden_dim, 1, bias=False)


    def forward(self, h, x):
        hx = torch.cat([h,x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, joint_state_dim, in_mlp_dims, ia_mlp_dims, sort_mlp_dims, sort_attention_dims, aggregation_dims, action_dims, with_global_state=True, with_interaction=True, with_om=False):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.self_state_dim = self_state_dim
        self.joint_state_dim = joint_state_dim
        self.gru_hidden_dim = ia_mlp_dims[-1]*2
        self.with_global_state = with_global_state
        self.with_om = with_om
        self.with_interaction = with_interaction

        self.input_mlp = mlp(self.input_dim - self.self_state_dim, in_mlp_dims, last_relu=True) # [B,C,N]

        if self.with_interaction:
            self.mlp = conv_mlp2(ia_mlp_dims[-1], ia_mlp_dims, (3,3))
            self.afa_mlp = AFAModule(self.mlp, use_softmax=True) #[B,C,N]
            self.ia_mlp = mlp(ia_mlp_dims[-1], ia_mlp_dims)
            self.sort_mlp = mlp(in_mlp_dims[-1]+self_state_dim, sort_mlp_dims) #[B,C*2,N]
        else:
            self.sort_mlp = mlp(in_mlp_dims[-1]*2+self_state_dim, sort_mlp_dims)    #[B,C*2+13,N]    

        # self.h0 = None
        # self.hn = None
        self.gru = GRU(ia_mlp_dims[-1]*2, self.gru_hidden_dim)
        self.sort_mlp_attention = mlp(sort_mlp_dims[-1]*2, sort_attention_dims)  
        # self.lstm = nn.LSTM(sort_mlp_dims[-1]*2,  self.lstm_hidden_dim, batch_first=True)
        action_input_dim = self.gru_hidden_dim + self.self_state_dim # 64 + 6
        # self.action_mlp = conv_mlp(action_input_dim, action_dims) #56,128,64,32,1
        self.action_mlp = mlp(action_input_dim, action_dims) #56,128,64,64,1
        self.attention_weights = None
        # self.aggration_mlp = conv_mlp2(self.gru_hidden_dim, aggregation_dims)


    def forward(self, state):
        # self.device = state.device
        in_size = state.shape
        # state_t = state.transpose(2,1)
        self_state = state[:, :, :self.self_state_dim]
        
        agents_state = state[:, :, self.self_state_dim:] #（batch_sz*num_agents）*num_features
        # state_att = state.view(-1,size[2])
        
        in_mlp_output = self.input_mlp(agents_state)

        if self.with_interaction:
            in_mlp_output = in_mlp_output + self.afa_mlp(in_mlp_output.transpose(2,1).contiguous())
            in_mlp_output = self.ia_mlp(in_mlp_output)
            # new_features = F.avg_pool1d(new_features, kernel_size=[1, new_features.size(3)]).squeeze(-1)  # (B, mlp[-1], npoint)
            sort_mlp_input = torch.cat([in_mlp_output, self_state], dim=-1)  #batch_sz*num_agents*(in_mlp_dims[-1]*2)
        else:
            in_mlp_output = in_mlp_output
            in_mlp_output = self.ia_mlp(in_mlp_output)
            # compute attention scores
            global_state = torch.mean(in_mlp_output, 1, keepdim=True) #[B,C,N]
            global_state = global_state.repeat((1, in_size[1], 1)).contiguous()
            sort_mlp_input = torch.cat([in_mlp_output, global_state, self_state], dim=-1) #batch_sz*num_agents*(in_mlp_dims[-1]*2 + self_state_size)
            

        sort_mlp_output = self.sort_mlp(sort_mlp_input)
        # sort_mlp_output = self.sort_mlp(in_mlp_output)
        sort_mlp_global_state = torch.mean(sort_mlp_output, 1, keepdim=True) #100,1,100
        sort_mlp_global_state = sort_mlp_global_state.repeat((1, in_size[1], 1)).contiguous()
        sort_mlp_input = torch.cat([sort_mlp_output, sort_mlp_global_state], dim=-1)

        sort_mlp_attention_output = self.sort_mlp_attention(sort_mlp_input)
        scores = sort_mlp_attention_output.squeeze(dim=-1) #100,5

        # masked softmax
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(-1)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        # features = sort_mlp_input.view(in_size[0], -1, in_size[1],) #100,5,50、
        weighted_feature = torch.mul(weights, sort_mlp_input)#(100,5,1),(100,5,50)
        
        gru_input = weighted_feature.view(in_size[0], -1, in_size[1])
        
        # if self.h0 is None:
        h0 = sort_mlp_input.transpose(2,1).contiguous()
            # self.h0 = torch.zeros_like(h0).cuda()
        hn = self.gru(gru_input, h0)
        # self.h0 = self.hn
        hn = torch.mean(hn, dim=2, keepdim=False) 
        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state[:,0,:], hn], dim=1)
        value = self.action_mlp(joint_state) #[B,C]
        value = value.view(in_size[0],-1)
        return value


class GIPCARL(MultiHumanRL):
    """
    Simple CommNet layer, similar to https://arxiv.org/pdf/1605.07736.pdf
    """
    def __init__(self):
        super().__init__()
        self.name = 'GIPCARL'

    def configure(self, config):
        self.set_common_parameters(config)
        in_mlp_dims = [int(x) for x in config.get('gipcarl', 'in_mlp_dims').split(', ')]
        ia_mlp_dims = [int(x) for x in config.get('gipcarl', 'ia_mlp_dims').split(', ')]
        sort_mlp_dims = [int(x) for x in config.get('gipcarl', 'sort_mlp_dims').split(', ')]
        sort_attention_dims = [int(x) for x in config.get('gipcarl', 'sort_attention_dims').split(', ')]
        aggregation_dims = [int(x) for x in config.get('gipcarl', 'aggregation_dims').split(', ')]
        action_dims = [int(x) for x in config.get('gipcarl', 'action_dims').split(', ')]
        self.with_om = config.getboolean('gipcarl', 'with_om')
        with_global_state = config.getboolean('gipcarl', 'with_global_state')
        with_interaction = config.getboolean('gipcarl', 'with_interaction')

        # def __init__(self, input_dim, self_state_dim, joint_state_dim, in_mlp_dims, ia_mlp_dims, sort_attention_dims, action_dims, with_global_state=True, with_om=False, with_interaction=True)

        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, self.joint_state_dim, in_mlp_dims, ia_mlp_dims, sort_mlp_dims, sort_attention_dims, aggregation_dims, action_dims, with_global_state, with_interaction)

        self.multiagent_training = config.getboolean('gipcarl', 'multiagent_training')
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))
        logging.info('Policy: {} {} interaction state'.format(self.name, 'w/' if with_interaction else 'w/o'))

    def get_attention_weights(self):
        return self.model.attention_weights

    
    def predict(self, state):
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)

        occupancy_maps = None
        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            self.action_values = list()
            max_value = float('-inf')
            max_action = None
            for action in self.action_space:
                next_self_state = self.propagate(state.self_state, action)
                if self.query_env:
                    next_human_states, reward, done, info = self.env.onestep_lookahead(action)
                else:
                    next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                                       for human_state in state.human_states]
                    reward = self.compute_reward(next_self_state, next_human_states)
                batch_next_states = torch.cat([torch.Tensor([next_self_state + next_human_state]).to(self.device)
                                              for next_human_state in next_human_states], dim=0)
                rotated_batch_input = self.rotate(batch_next_states).unsqueeze(0)
                if self.with_om:
                    if occupancy_maps is None:
                        occupancy_maps = self.build_occupancy_maps(next_human_states).unsqueeze(0)
                    rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps], dim=2)
                # VALUE UPDATE
                next_state_value = self.model(rotated_batch_input).data.item()
                # value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value
                if self.kinematics == "holonomic":
                    v = np.linalg.norm(np.array(action))
                    value = reward + pow(self.gamma, self.time_step * v) * next_state_value
                else:
                    value = reward + pow(self.gamma, self.time_step * action[0]) * next_state_value
                self.action_values.append(value)
                if value > max_value:
                    max_value = value
                    max_action = action
                    
            if max_action is None:
                raise ValueError('Value network is not well trained. ')

        if self.phase == 'train':
            self.last_state = self.transform(state)
        # print("Action:V:%f,\tR:%f\t"%(max_action.v, max_action.r))
        return max_action


    
    