from re import I
from utils.explorer import average
from common.naive_transformer import TransformerEncoder
import numpy as np
import logging
import torch
import torch.nn as nn
from crowd_nav.common.components import ATC_TFencoder, mlp
from crowd_nav.common.components import ATCBasic
from crowd_nav.common.components import ATCBasicTfencoder
from enum import IntEnum

# from torch.nn import TransformerEncoder
# from torch.nn import TransformerDecoder
# from torch.nn import Transformer


class Flag(IntEnum):
    UseNaiveNetWork = 0
    UseTfasLSTM = 1
    UseActLSTM = 2
    UseOnlyTf = 3
    UseOnlyTf_2 = 6
    UseActTfencoder = 4
    UseOnlyTfAndGRU = 5


class ValueNetworkBase(nn.Module):
    def __init__(self,
                 input_dim,
                 self_state_dim,
                 joint_state_dim,
                 in_mlp_dims,
                 sort_mlp_dims,
                 sort_mlp_attention,
                 action_dims,
                 with_dynamic_net=True,
                 with_global_state=True,
                 test_policy_flag=0,
                 multi_process_type="average",
                 act_steps=3,
                 act_fixed=False):
        super().__init__()
        self.input_dim = input_dim
        self.self_state_dim = self_state_dim
        self.joint_state_dim = joint_state_dim
        self.in_mlp_dims = in_mlp_dims
        self.sort_mlp_dims = sort_mlp_dims
        self.sort_mlp_attention = sort_mlp_attention
        self.action_dims = action_dims
        self.with_dynamic_net = with_dynamic_net
        self.with_global_state = with_global_state
        self.test_policy_flag = test_policy_flag
        self.multi_process_type = multi_process_type

        self.act_steps = act_steps
        self.act_fixed = act_fixed

    def product(self):
        print(type(self.test_policy_flag))
        print(self.test_policy_flag)
        if self.test_policy_flag == Flag.UseNaiveNetWork:
            return ValueNetworkNaive(self.input_dim, self.self_state_dim, self.joint_state_dim, self.in_mlp_dims,
                                     self.sort_mlp_dims, self.sort_mlp_attention, self.action_dims,
                                     self.with_dynamic_net, self.with_global_state)
        elif self.test_policy_flag == Flag.UseTfasLSTM:
            return ValueNetworkUseTransformerAsLSTM(self.input_dim, self.self_state_dim, self.joint_state_dim,
                                                    self.in_mlp_dims, self.sort_mlp_dims, self.sort_mlp_attention,
                                                    self.action_dims, self.with_dynamic_net, self.with_global_state,
                                                    self.multi_process_type)
            pass
        elif self.test_policy_flag == Flag.UseActTfencoder:
            return ValueNetworkActTfencoder(self.input_dim, self.self_state_dim, self.joint_state_dim, self.in_mlp_dims,
                                            self.sort_mlp_dims, self.sort_mlp_attention, self.action_dims,
                                            self.with_dynamic_net, self.with_global_state, self.multi_process_type)
            pass
        elif self.test_policy_flag == Flag.UseOnlyTf_2:
            return ValueNetworkUseOnlyTransformer(self.input_dim,
                                                  self.self_state_dim,
                                                  self.joint_state_dim,
                                                  self.in_mlp_dims,
                                                  self.sort_mlp_dims,
                                                  self.sort_mlp_attention,
                                                  self.action_dims,
                                                  self.with_dynamic_net,
                                                  self.with_global_state,
                                                  self.multi_process_type,
                                                  act_layer_type=2)
            pass
        elif self.test_policy_flag == Flag.UseOnlyTf:
            return ValueNetworkUseOnlyTransformer(self.input_dim,
                                                  self.self_state_dim,
                                                  self.joint_state_dim,
                                                  self.in_mlp_dims,
                                                  self.sort_mlp_dims,
                                                  self.sort_mlp_attention,
                                                  self.action_dims,
                                                  self.with_dynamic_net,
                                                  self.with_global_state,
                                                  self.multi_process_type,
                                                  self.act_steps,
                                                  self.act_fixed,
                                                  act_layer_type=1)
            pass
        elif self.test_policy_flag == Flag.UseOnlyTfAndGRU:
            return ValueNetworkTransformerAndGRU(self.input_dim, self.self_state_dim, self.joint_state_dim,
                                                 self.in_mlp_dims, self.sort_mlp_dims, self.sort_mlp_attention,
                                                 self.action_dims, self.with_dynamic_net, self.with_global_state,
                                                 self.multi_process_type, self.act_steps, self.act_fixed)
        pass


class ValueNetworkNaive(nn.Module):
    def __init__(
        self,
        input_dim,
        self_state_dim,
        joint_state_dim,
        in_mlp_dims,
        sort_mlp_dims,
        sort_mlp_attention,
        action_dims,
        with_dynamic_net=True,
        with_global_state=True,
    ):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.self_state_dim = self_state_dim
        self.global_state_dim = in_mlp_dims[-1]
        self.joint_state_dim = joint_state_dim
        self.in_mlp_dims = in_mlp_dims
        self.input_dim = input_dim
        self.lstm_hidden_dim = sort_mlp_attention[0] * 2
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
            self.sort_mlp = mlp(self.in_mlp_dims[-1] * 2 + self.joint_state_dim, sort_mlp_dims)
        else:
            self.sort_mlp = mlp(self.in_mlp_dims[-1] + self.joint_state_dim, sort_mlp_dims)
        # avg+mlp2

        # (avg+mlp2)
        self.sort_mlp_attention = mlp(sort_mlp_dims[-1] * 2, sort_mlp_attention)
        # add a soft_max layer after soft_mlp_attentions
        self.lstm = nn.LSTM(sort_mlp_dims[-1] * 2, self.lstm_hidden_dim, batch_first=True)

        action_input_dim = self.lstm_hidden_dim + self.self_state_dim  # 50 + 6
        self.action_mlp = mlp(action_input_dim, action_dims)  # 56,150,100,100,1
        self.attention_weights = None

    def forward(self, state):
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        agents_state = state[:, :, self.self_state_dim:].view(size[0] * size[1], -1)  # batch_sz*num_agents*num_features
        state_att = state.view(-1, size[2])

        act_h0 = torch.zeros([size[0] * size[1], self.in_mlp_dims[-1]]).cuda()
        in_mlp_output, env_score = self.in_mlp(state, act_h0)  # batch_sz*num_agents*in_mlp_dims[-1]
        env_score = 0.5

        global_state = torch.mean(in_mlp_output.view(size[0], size[1], -1), 1,
                                  keepdim=True)  # batch_sz*1*in_mlp_dims[-1]
        global_state = (global_state.expand(
            (size[0], size[1], self.global_state_dim)).contiguous().view(-1, self.global_state_dim)
                        )  ##batch_sz*num_agents*in_mlp_dims[-1]

        if self.with_global_state:
            sort_mlp_input = torch.cat([in_mlp_output, global_state, state_att],
                                       dim=1)  # batch_sz*num_agents*(in_mlp_dims[-1]*2 + self_state_size)
        else:
            sort_mlp_input = torch.cat([in_mlp_output, state_att], dim=1)  # batch_sz*num_agents*(in_mlp_dims[-1]*2)
        sort_mlp_output = self.sort_mlp(sort_mlp_input)
        sort_mlp_global_state = torch.mean(sort_mlp_output.view(size[0], size[1], -1), 1, keepdim=True)  # 100,1,100
        sort_mlp_global_state = (sort_mlp_global_state.expand(
            (size[0], size[1], self.sort_mlp_global_state_dim)).contiguous().view(-1, self.sort_mlp_global_state_dim)
                                 )  # 500,100
        sort_mlp_input = torch.cat([sort_mlp_output, sort_mlp_global_state], dim=1)
        sort_mlp_attention_output = self.sort_mlp_attention(sort_mlp_input)
        # sort_attention_score = selt.attention(sort_mlp_attention_output)
        scores = sort_mlp_attention_output.view(size[0], size[1], 1).squeeze(dim=2)  # 100,5

        # masked softmax
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = sort_mlp_input.view(size[0], size[1], -1)  # 100,5,50
        weighted_feature = torch.mul(weights, features)  # (100,5,1),(100,5,50)

        lstm_input = weighted_feature.view(size[0], size[1], -1)
        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim, device=self.device)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim, device=self.device)
        self.lstm.flatten_parameters()
        output, (hn, cn) = self.lstm(lstm_input, (h0, c0))
        hn = hn.squeeze(0)

        # concatenate agent's state with global weighted humans' state

        joint_state = torch.cat([self_state, hn], dim=1)
        value = self.action_mlp(joint_state)
        value = value.view(size[0], -1)
        return value, env_score


class ValueNetworkActTfencoder(nn.Module):
    def __init__(self,
                 input_dim,
                 self_state_dim,
                 joint_state_dim,
                 in_mlp_dims,
                 sort_mlp_dims,
                 sort_mlp_attention,
                 action_dims,
                 with_dynamic_net=True,
                 with_global_state=True,
                 multi_process_type="average"):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        self.self_state_dim = self_state_dim
        self.global_state_dim = in_mlp_dims[-1]
        self.joint_state_dim = joint_state_dim
        self.in_mlp_dims = in_mlp_dims
        self.input_dim = input_dim
        self.lstm_hidden_dim = sort_mlp_attention[0] * 2
        self.with_dynamic_net = with_dynamic_net
        self.with_global_state = with_global_state
        self.sort_mlp_attention = sort_mlp_attention
        self.sort_mlp_global_state_dim = sort_mlp_dims[-1]

        self.multi_process_type = multi_process_type

        if self.with_dynamic_net:
            self.in_mlp = ATCBasicTfencoder(self.input_dim, in_mlp_dims, epsilon=0.05, last_relu=True)
        else:
            self.in_mlp = mlp(self.input_dim, in_mlp_dims, last_relu=True)

        # avg+mlp1
        if self.with_global_state:
            self.sort_mlp = mlp(self.in_mlp_dims[-1] * 2 + self.joint_state_dim, sort_mlp_dims)
        else:
            self.sort_mlp = mlp(self.in_mlp_dims[-1] + self.joint_state_dim, sort_mlp_dims)

        # (avg+mlp2)
        # linear(100 50) Relu linear(50 50) Relu linear(50 1)
        self.sort_mlp_attention = mlp(sort_mlp_dims[-1] * 2, sort_mlp_attention)

        # add a soft_max layer after soft_mlp_attentions
        self.lstm = nn.LSTM(sort_mlp_dims[-1] * 2, self.lstm_hidden_dim, batch_first=True)

        action_input_dim = self.lstm_hidden_dim + self.self_state_dim  # 50 + 6
        self.action_mlp = mlp(action_input_dim, action_dims)  # 56,150,100,100,1
        self.attention_weights = None

    def forward(self, state):
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]  # 500 x 6 (100 x 5 x 6)

        # batch_sz*num_agents*num_features (500*13)
        agents_state = state[:, :, self.self_state_dim:].view(size[0] * size[1], -1)

        # torch.cat agent states with self states
        state_att = state.view(-1, size[2])

        act_h0 = torch.zeros([size[0] * size[1], self.in_mlp_dims[-1]]).cuda()  # 500 x 50
        in_mlp_output, env_score = self.in_mlp(state, act_h0)  # batch_sz*num_agents*in_mlp_dims[-1]
        env_score = 0.5

        # batch_sz*1*in_mlp_dims[-1]
        global_state = torch.mean(in_mlp_output.view(size[0], size[1], -1), 1, keepdim=True)

        ##batch_sz*num_agents*in_mlp_dims[-1]
        global_state = (global_state.expand(
            (size[0], size[1], self.global_state_dim)).contiguous().view(-1, self.global_state_dim))

        if self.with_global_state:
            # batch_sz*num_agents*(in_mlp_dims[-1]*2 + self_state_size)
            sort_mlp_input = torch.cat([in_mlp_output, global_state, state_att], dim=1)
        else:
            sort_mlp_input = torch.cat([in_mlp_output, state_att], dim=1)  # batch_sz*num_agents*(in_mlp_dims[-1]*2)
        sort_mlp_output = self.sort_mlp(sort_mlp_input)
        sort_mlp_global_state = torch.mean(sort_mlp_output.view(size[0], size[1], -1), 1, keepdim=True)  # 100,1,100
        sort_mlp_global_state = (sort_mlp_global_state.expand(
            (size[0], size[1], self.sort_mlp_global_state_dim)).contiguous().view(-1, self.sort_mlp_global_state_dim)
                                 )  # 500,100
        sort_mlp_input = torch.cat([sort_mlp_output, sort_mlp_global_state], dim=1)
        sort_mlp_attention_output = self.sort_mlp_attention(sort_mlp_input)
        # sort_attention_score = selt.attention(sort_mlp_attention_output)
        scores = sort_mlp_attention_output.view(size[0], size[1], 1).squeeze(dim=2)  # 100,5

        # masked softmax
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = sort_mlp_input.view(size[0], size[1], -1)  # 100,5,50
        weighted_feature = torch.mul(weights, features)  # (100,5,1),(100,5,50)

        lstm_input = weighted_feature.view(size[0], size[1], -1)
        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim, device=self.device)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim, device=self.device)
        self.lstm.flatten_parameters()
        output, (hn, cn) = self.lstm(lstm_input, (h0, c0))
        hn = hn.squeeze(0)

        # concatenate agent's state with global weighted humans' state

        joint_state = torch.cat([self_state, hn], dim=1)
        value = self.action_mlp(joint_state)
        value = value.view(size[0], -1)
        return value, env_score


class ValueNetworkUseTransformerAsLSTM(nn.Module):
    def __init__(self,
                 input_dim,
                 self_state_dim,
                 joint_state_dim,
                 in_mlp_dims,
                 sort_mlp_dims,
                 sort_mlp_attention,
                 action_dims,
                 with_dynamic_net=True,
                 with_global_state=True,
                 multi_process_type="average"):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.self_state_dim = self_state_dim
        self.global_state_dim = in_mlp_dims[-1]
        self.joint_state_dim = joint_state_dim
        self.in_mlp_dims = in_mlp_dims
        self.input_dim = input_dim
        self.lstm_hidden_dim = sort_mlp_attention[0] * 2
        self.with_dynamic_net = with_dynamic_net
        self.with_global_state = with_global_state
        self.sort_mlp_attention = sort_mlp_attention
        self.sort_mlp_global_state_dim = sort_mlp_dims[-1]

        self.multi_process_type = multi_process_type

        if self.with_dynamic_net:
            self.in_mlp = ATCBasicTfencoder(self.input_dim, in_mlp_dims, epsilon=0.05, last_relu=True)
        else:
            self.in_mlp = mlp(self.input_dim, in_mlp_dims, last_relu=True)

        # avg+mlp1
        if self.with_global_state:
            self.sort_mlp = mlp(self.in_mlp_dims[-1] * 2 + self.joint_state_dim, sort_mlp_dims)
        else:
            self.sort_mlp = mlp(self.in_mlp_dims[-1] + self.joint_state_dim, sort_mlp_dims)

        # (avg+mlp2)
        # linear(100 50) Relu linear(50 50) Relu linear(50 1)
        self.sort_mlp_attention = mlp(sort_mlp_dims[-1] * 2, sort_mlp_attention)

        # add a soft_max layer after soft_mlp_attentions
        # self.lstm = nn.LSTM(sort_mlp_dims[-1] * 2, self.lstm_hidden_dim, batch_first=True)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=100, nhead=2, dim_feedforward=300)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        action_input_dim = self.lstm_hidden_dim + self.self_state_dim  # 50 + 6
        self.action_mlp = mlp(action_input_dim, action_dims)  # 56,150,100,100,1
        self.transmlp = mlp(input_dim=500, mlp_dims=[300, 100])
        self.attention_weights = None

    def forward(self, state: torch.Tensor):
        size = state.shape
        batch_size = size[0]
        seq_len = size[1]
        original_dim = size[2]

        self_state = state[:, 0, :self.self_state_dim]  # 500 x 6 (100 x 5 x 6)
        self_agent_state = state[:, 0, :].clone().detach()
        self_agent_state = self_agent_state.view(self_agent_state.shape[0], -1, self_agent_state.shape[1])
        self_agent_state[:, 0, 6] = 0.0
        self_agent_state[:, 0, 7] = 0.0
        self_agent_state[:, 0, 8] = self_agent_state[:, 0, 4]
        self_agent_state[:, 0, 9] = self_agent_state[:, 0, 5]
        self_agent_state[:, 0, 10] = self_agent_state[:, 0, 3]
        self_agent_state[:, 0, 11] = 0.0
        self_agent_state[:, 0, 12] = self_agent_state[:, 0, 3]


        if self.multi_process_type == "self_attention":
            state = torch.cat([self_agent_state, state], dim=1)
            seq_len += 1

        # torch.cat agent states with self states
        state_att = state.view(-1, size[2])
 
        act_h0 = torch.zeros([size[0] * seq_len, self.in_mlp_dims[-1]]).cuda()  # 500 x 50
        in_mlp_output, env_score = self.in_mlp(state, act_h0)  # batch_sz*num_agents*in_mlp_dims[-1]
        env_score = 0.5
        
        global_state = torch.mean(in_mlp_output.view(size[0], seq_len, -1), 1, keepdim=True)
        global_state = (global_state.expand(
            (size[0], seq_len, self.global_state_dim)).contiguous().view(-1, self.global_state_dim))
        
        if self.with_global_state:
            sort_mlp_input = torch.cat([in_mlp_output, global_state, state_att], dim=1)
        else:
            sort_mlp_input = torch.cat([in_mlp_output, state_att], dim=1)  # batch_sz*num_agents*(in_mlp_dims[-1]*2)
        
        sort_mlp_output = self.sort_mlp(sort_mlp_input)
        sort_mlp_global_state = torch.mean(sort_mlp_output.view(size[0], seq_len, -1), 1, keepdim=True)  # 100,1,100
        sort_mlp_global_state = (sort_mlp_global_state.expand(
            (size[0], seq_len, self.sort_mlp_global_state_dim)).contiguous().view(-1, self.sort_mlp_global_state_dim))
        
        sort_mlp_input = torch.cat([sort_mlp_output, sort_mlp_global_state], dim=1)
        
        sort_mlp_attention_output = self.sort_mlp_attention(sort_mlp_input)

        # sort_attention_score = selt.attention(sort_mlp_attention_output)
        scores = sort_mlp_attention_output.view(size[0], seq_len, 1).squeeze(dim=2)  # 100,5

        # masked softmax
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()
        
        # output feature is a linear combination of input features
        features = sort_mlp_input.view(size[0], seq_len, -1)  # 100,5,50
        weighted_feature = torch.mul(weights, features)  # (100,5,1),(100,5,50)

        lstm_input = weighted_feature.view(size[0], seq_len, -1)
        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim, device=self.device)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim, device=self.device)
        # self.lstm.flatten_parameters()

        # 100 * 5 * 100
        tfencoder_input = lstm_input.transpose(0, 1).contiguous()
        tfencoder_output = self.transformer_encoder(tfencoder_input)
        tfencoder_output = tfencoder_output.transpose(0, 1).contiguous()  # batch_size seq_len feature_dims

        if self.multi_process_type == "average":
            env_info = torch.mean(tfencoder_output, 1, keepdim=True)
            env_info = env_info.view(env_info.shape[0], env_info.shape[2])
        elif self.multi_process_type == "self_attention":
            env_info = tfencoder_output[:, 0, :]
        else:
            env_info = self.transmlp(tfencoder_output.view(tfencoder_output.shape[0], -1))
        # tfencoder_input
        # env_info = self.transmlp(tfencoder_output.view(tfencoder_output.shape[0], -1))
        # output, (hn, cn) = self.lstm(lstm_input, (h0, c0))
        # hn = hn.squeeze(0)

        # concatenate agent's state with global weighted humans' state
        
        # joint_state = torch.cat([self_state, hn], dim=1)
        joint_state = torch.cat([self_state, env_info], dim=1)
        value = self.action_mlp(joint_state)
        value = value.view(size[0], -1)
        return value, env_score


class ValueNetworkUseOnlyTransformer(nn.Module):
    def __init__(self,
                 input_dim,
                 self_state_dim,
                 joint_state_dim,
                 in_mlp_dims,
                 sort_mlp_dims,
                 sort_mlp_attention,
                 action_dims,
                 with_dynamic_net=True,
                 with_global_state=True,
                 multi_process_type="average",
                 act_steps=3,
                 act_fixed=False,
                 act_layer_type=1):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.self_state_dim = self_state_dim
        self.global_state_dim = in_mlp_dims[-1]
        self.joint_state_dim = joint_state_dim
        self.in_mlp_dims = in_mlp_dims
        self.input_dim = input_dim
        self.lstm_hidden_dim = sort_mlp_attention[0] * 2
        self.with_dynamic_net = with_dynamic_net
        self.with_global_state = with_global_state
        self.sort_mlp_attention = sort_mlp_attention
        self.sort_mlp_global_state_dim = sort_mlp_dims[-1]

        self.act_steps = act_steps
        self.act_fixed = act_fixed

        self.multi_process_type = multi_process_type

        self.act_layer_type = act_layer_type

        if self.with_dynamic_net:
            if act_layer_type == 1:
                self.in_mlp = ATCBasicTfencoder(self.input_dim,
                                                in_mlp_dims,
                                                epsilon=0.05,
                                                last_relu=True,
                                                act_steps=self.act_steps,
                                                act_fixed=self.act_fixed)
            elif act_layer_type == 2:
                self.in_mlp = ATC_TFencoder(self.input_dim,
                                            in_mlp_dims,
                                            epsilon=0.05,
                                            last_relu=True,
                                            act_steps=self.act_steps,
                                            act_fixed=self.act_fixed)
        else:
            self.in_mlp = mlp(self.input_dim, in_mlp_dims, last_relu=True)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=50, nhead=2, dim_feedforward=150)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        action_input_dim = 50 + self.self_state_dim  # 50 + 6
        self.action_mlp = mlp(action_input_dim, action_dims)  # 56,150,100,100,1
        # self.transmlp = mlp(input_dim=250, mlp_dims=[250, 100])
        self.attention_weights = None

        self.step_cnt = 0

    def get_step_cnt(self):
        return self.step_cnt
        pass

    def forward(self, state: torch.Tensor):
        '''
        batch_size * seq_len * feature_size
        '''
        #   0   1       2       3       4   5   6   7     8    9    10      11      12
        # [dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum]
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim].clone().detach()  # 500 x 6 (100 x 5 x 6)

        
        self_agent_state = state[:, 0, :].clone().detach()
        self_agent_state = self_agent_state.view(self_agent_state.shape[0], -1, self_agent_state.shape[1])
        self_agent_state[:, 0, 6] = 0.0
        self_agent_state[:, 0, 7] = 0.0
        self_agent_state[:, 0, 8] = self_agent_state[:, 0, 4]
        self_agent_state[:, 0, 9] = self_agent_state[:, 0, 5]
        self_agent_state[:, 0, 10] = self_agent_state[:, 0, 3]
        self_agent_state[:, 0, 11] = 0.0
        self_agent_state[:, 0, 12] = self_agent_state[:, 0, 3]

        
        if self.multi_process_type == "self_attention":
            state = torch.cat([self_agent_state, state], dim=1)
        
        act_h0 = torch.zeros([size[0] * size[1], self.in_mlp_dims[-1]]).cuda()  # 500 x 50
        in_mlp_output, self.step_cnt = self.in_mlp(state, act_h0)  # batch_sz*num_agents*in_mlp_dims[-1]

        env_score = 0.5  
        
        # 100 * 5 * 100

        tfencoder_input = in_mlp_output.view(size[0], -1, 50)
        tfencoder_input = tfencoder_input.transpose(0, 1).contiguous()
        tfencoder_output = self.transformer_encoder(tfencoder_input)
        tfencoder_output = tfencoder_output.transpose(0, 1).contiguous()
       
        if self.multi_process_type == "self_attention":
            env_info = tfencoder_output[:, 0, :]
            pass
        elif self.multi_process_type == "average":
            env_info = torch.mean(tfencoder_output, dim=1, keepdim=True)
            env_info = env_info.view(env_info.shape[0], env_info.shape[2])
            pass
        
        joint_state = torch.cat([self_state, env_info], dim=1)
        value = self.action_mlp(joint_state)
        value = value.view(size[0], -1)
        return value, env_score


class ValueNetworkTransformerAndGRU(nn.Module):
    def __init__(self,
                 input_dim,
                 self_state_dim,
                 joint_state_dim,
                 in_mlp_dims,
                 sort_mlp_dims,
                 sort_mlp_attention,
                 action_dims,
                 with_dynamic_net=True,
                 with_global_state=True,
                 multi_process_type="average",
                 act_steps=3,
                 act_fixed=False):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.self_state_dim = self_state_dim
        self.global_state_dim = in_mlp_dims[-1]
        self.joint_state_dim = joint_state_dim
        self.in_mlp_dims = in_mlp_dims
        self.input_dim = input_dim
        self.lstm_hidden_dim = sort_mlp_attention[0] * 2
        self.with_dynamic_net = with_dynamic_net
        self.with_global_state = with_global_state
        self.sort_mlp_attention = sort_mlp_attention
        self.sort_mlp_global_state_dim = sort_mlp_dims[-1]

        self.act_fixed = act_fixed
        self.act_steps = act_steps

        self.multi_process_type = multi_process_type

        if self.with_dynamic_net:
            self.in_mlp = ATCBasic(self.input_dim,
                                   in_mlp_dims,
                                   epsilon=0.05,
                                   last_relu=True,
                                   act_steps=self.act_steps,
                                   act_fixed=self.act_fixed)
        else:
            self.in_mlp = mlp(self.input_dim, in_mlp_dims, last_relu=True)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=50, nhead=2, dim_feedforward=150)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        action_input_dim = 50 + self.self_state_dim  # 50 + 6
        self.action_mlp = mlp(action_input_dim, action_dims)  # 56,150,100,100,1
        # self.transmlp = mlp(input_dim=250, mlp_dims=[250, 100])
        self.attention_weights = None
        self.step_cnt = 0

    def get_step_cnt(self):
        return self.step_cnt
        pass

    def forward(self, state: torch.Tensor):
        '''
        batch_size * seq_len * feature_size
        '''
        #   0   1       2       3       4   5   6   7     8    9    10      11      12
        # [dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum]
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim].clone().detach()  # 500 x 6 (100 x 5 x 6)

        
        self_agent_state = state[:, 0, :].clone().detach()
        self_agent_state = self_agent_state.view(self_agent_state.shape[0], -1, self_agent_state.shape[1])
        self_agent_state[:, 0, 6] = 0.0
        self_agent_state[:, 0, 7] = 0.0
        self_agent_state[:, 0, 8] = self_agent_state[:, 0, 4]
        self_agent_state[:, 0, 9] = self_agent_state[:, 0, 5]
        self_agent_state[:, 0, 10] = self_agent_state[:, 0, 3]
        self_agent_state[:, 0, 11] = 0.0
        self_agent_state[:, 0, 12] = self_agent_state[:, 0, 3]

        
        if self.multi_process_type == "self_attention":
            state = torch.cat([self_agent_state, state], dim=1)
        
        act_h0 = torch.zeros([size[0] * (size[1] + 1), self.in_mlp_dims[-1]]).cuda()  # 500 x 50
        in_mlp_output, self.step_cnt = self.in_mlp(state, act_h0)  # batch_sz*num_agents*in_mlp_dims[-1]
        env_score = 0.5  
        
        # 100 * 5 * 100

        tfencoder_input = in_mlp_output.view(size[0], -1, 50)
        tfencoder_input = tfencoder_input.transpose(0, 1).contiguous()
        tfencoder_output = self.transformer_encoder(tfencoder_input)
        tfencoder_output = tfencoder_output.transpose(0, 1).contiguous()
        
        if self.multi_process_type == "self_attention":
            env_info = tfencoder_output[:, 0, :]
            pass
        elif self.multi_process_type == "average":
            env_info = torch.mean(tfencoder_output, dim=1, keepdim=True)
            env_info = env_info.view(env_info.shape[0], env_info.shape[2])
            pass
        
        joint_state = torch.cat([self_state, env_info], dim=1)
        value = self.action_mlp(joint_state)
        value = value.view(size[0], -1)
        return value, env_score
