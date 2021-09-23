import numpy as np
import logging
import torch
import torch.nn as nn
from crowd_nav.common.components import mlp

from crowd_nav.policy.multi_human_rl import MultiHumanRL
from crowd_nav.common.qnetwork_factory import ValueNetworkBase
import math
from crowd_nav.common.components import ATCBasic


class ACTFCARL(MultiHumanRL):
    """
    Simple Adaptive Computation Time Model, similar to https://arxiv.org/pdf/1603.08983.pdf
    """
    def __init__(self):
        super().__init__()
        self.name = "ACTFCARL"
        self.preprocess_type = "Normal"

    def configure(self, config):
        self.set_common_parameters(config)
        in_mlp_dims = [int(x) for x in config.get("actenvcarl", "in_mlp_dims").split(", ")]
        # gru_hidden_dim = [int(x) for x in config.get('actenvcarl', 'gru_hidden_dim').split(', ')]
        sort_mlp_dims = [int(x) for x in config.get("actenvcarl", "sort_mlp_dims").split(", ")]
        sort_mlp_attention = [int(x) for x in config.get("actenvcarl", "sort_attention_dims").split(", ")]
        # aggregation_dims = [int(x) for x in config.get('actenvcarl', 'aggregation_dims').split(', ')]
        action_dims = [int(x) for x in config.get("actenvcarl", "action_dims").split(", ")]
        self.with_om = config.getboolean("actenvcarl", "with_om")
        with_dynamic_net = config.getboolean("actenvcarl", "with_dynamic_net")
        with_global_state = config.getboolean("actenvcarl", "with_global_state")
        test_policy_flag = [int(x) for x in config.get("actenvcarl", "test_policy_flag").split(", ")]
        multi_process_type = config.get("actenvcarl", "multi_process")

        print("process type ", type(multi_process_type))
        print(type(test_policy_flag[0]))
        # def __init__(self, input_dim, self_state_dim, joint_state_dim, in_mlp_dims, sort_mlp_dims, action_dims, with_dynamic_net=True):

        print(
            self.self_state_dim,
            self.joint_state_dim,
            in_mlp_dims,
            sort_mlp_dims,
            sort_mlp_attention,
            action_dims,
        )
        self.model = ValueNetworkActf(self.input_dim(),
                                      self.self_state_dim,
                                      self.joint_state_dim,
                                      in_mlp_dims,
                                      sort_mlp_dims,
                                      sort_mlp_attention,
                                      action_dims,
                                      with_dynamic_net,
                                      with_global_state,
                                      multi_process_type=multi_process_type)

        self.multiagent_training = config.getboolean("actcarl", "multiagent_training")

    def get_attention_weights(self):
        return self.model.attention_weights


class ValueNetworkActf(nn.Module):  # act with transformer
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

        action_input_dim = 50 + self.self_state_dim  # 50 + 6
        self.action_mlp = mlp(action_input_dim, action_dims)  # 56,150,100,100,1
        self.attention_weights = None

        # acttf
        self.actf_enncoder = ActEncoder(hidden_size=50,
                                        num_layers=9,
                                        num_heads=2,
                                        filter_size=150,
                                        source_dims=self.input_dim)

    def get_step_cnt(self):
        return self.actf_enncoder.get_act_step_cnt()

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

        tfencoder_output, _ = self.actf_enncoder(state)
        env_score = 0.5 

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


def _gen_bias_mask(max_length):
    """
    Generates bias values (-Inf) to mask future timesteps during attention
    """
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    torch_mask = torch.from_numpy(np_mask).type(torch.FloatTensor)

    return torch_mask.unsqueeze(0).unsqueeze(1)


def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], 'constant', constant_values=[0.0, 0.0])
    signal = signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)


class ActEncoder(nn.Module):
    """
    A Transformer Encoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(
            self,
            # embedding_size, 
            hidden_size,  
            num_layers,  
            num_heads, 
            filter_size,  
            max_length=100,
            source_dims=13,
            target_dims=50):

        super(ActEncoder, self).__init__()

        self.preprocess_mlp = nn.Linear(source_dims, hidden_size)

        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        ## for t
        self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.num_layers = num_layers
        self.act = True

        self.proj_flag = False

        self.tf_encoder = nn.TransformerEncoderLayer(hidden_size, nhead=num_heads, dim_feedforward=filter_size)

        self.layer_norm = LayerNorm(hidden_size)

        self.act_fn = ACT_basic(hidden_size)

    def get_act_step_cnt(self):
        self.act_fn.get_act_step_cnt()
        pass
    def forward(self, inputs):

        x = self.preprocess_mlp(inputs)

        x, (remainders, n_updates) = self.act_fn(x, x, self.tf_encoder, self.timing_signal, self.position_signal,
                                                 self.num_layers)
        return x, (remainders, n_updates)


class LayerNorm(nn.Module):
    # Borrowed from jekbradbury
    # https://github.com/pytorch/pytorch/issues/1959
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


### CONVERTED FROM https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/universal_transformer_util.py#L1062
class ACT_basic(nn.Module):
    def __init__(self, hidden_size):
        super(ACT_basic, self).__init__()
        self.sigma = nn.Sigmoid()
        self.p = nn.Linear(hidden_size, 1)
        self.p.bias.data.fill_(1)
        self.threshold = 1 - 0.1
        self.act_step_cnt = 0

    def get_act_step_cnt(self):
        return self.act_step_cnt
        pass
    def forward(self, state, inputs, fn, time_enc, pos_enc, max_hop, encoder_output=None):
        # init_hdd
        ## [B, S]
        halting_probability = torch.zeros(inputs.shape[0], inputs.shape[1]).cuda()
        ## [B, S
        remainders = torch.zeros(inputs.shape[0], inputs.shape[1]).cuda()
        ## [B, S]
        n_updates = torch.zeros(inputs.shape[0], inputs.shape[1]).cuda()
        ## [B, S, HDD]
        previous_state = torch.zeros_like(inputs).cuda()
        step = 0
        # for l in range(self.num_layers):
        while (((halting_probability < self.threshold) & (n_updates < max_hop)).byte().any()):
            # Add timing signal
            # state = state + time_enc[:, :inputs.shape[1], :].type_as(inputs.data)
            # state = state + pos_enc[:, step, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)

            p = self.sigma(self.p(state)).squeeze(-1)
            # Mask for inputs which have not halted yet
            still_running = (halting_probability < 1.0).float()

            # Mask of inputs which halted at this step
            new_halted = (halting_probability + p * still_running > self.threshold).float() * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (halting_probability + p * still_running <= self.threshold).float() * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability = halting_probability + p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders = remainders + new_halted * (1 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability = halting_probability + new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates = n_updates + still_running + new_halted

            # Compute the weight to be applied to the new state and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = p * still_running + new_halted * remainders

            # just used by decoder
            if (encoder_output):
                state, _ = fn((state, encoder_output))
            else:
                # apply transformation on the state
                # size = state.shape()
                # tfencoder_input = state.view(size[0], -1, 50)
                state = state.transpose(0, 1).contiguous()
                state = fn(state)
                state = state.transpose(0, 1).contiguous()
                # tfencoder_output = tfencoder_output.transpose(0, 1).contiguous()

            # update running part in the weighted state and keep the rest
            previous_state = ((state * update_weights.unsqueeze(-1)) + (previous_state *
                                                                        (1 - update_weights.unsqueeze(-1))))
            ## previous_state is actually the new_state at end of hte loop
            ## to save a line I assigned to previous_state so in the next
            ## iteration is correct. Notice that indeed we return previous_state
            step += 1
        self.act_step_cnt = step
        return previous_state, (remainders, n_updates)
