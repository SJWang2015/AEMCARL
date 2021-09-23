import numpy as np
import logging
import torch
import torch.nn as nn
from crowd_nav.common.components import mlp

from crowd_nav.policy.multi_human_rl import MultiHumanRL
from crowd_nav.common.qnetwork_factory import ValueNetworkBase


class ACTENVCARL(MultiHumanRL):
    """
    Simple Adaptive Computation Time Model, similar to https://arxiv.org/pdf/1603.08983.pdf
    """
    def __init__(self):
        super().__init__()
        self.name = "ACTENVCARL"

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

        act_fixed = config.get("actenvcarl", "act_fixed")
        act_steps = config.get("actenvcarl", "act_steps")

        print("process type ", type(multi_process_type))
        print(type(test_policy_flag[0]))
        print("act steps: ", act_steps," act_fixed: ",act_fixed)
        # def __init__(self, input_dim, self_state_dim, joint_state_dim, in_mlp_dims, sort_mlp_dims, action_dims, with_dynamic_net=True):

        print(
            self.self_state_dim,
            self.joint_state_dim,
            in_mlp_dims,
            sort_mlp_dims,
            sort_mlp_attention,
            action_dims
        )
        self.model = ValueNetworkBase(self.input_dim(),
                                      self.self_state_dim,
                                      self.joint_state_dim,
                                      in_mlp_dims,
                                      sort_mlp_dims,
                                      sort_mlp_attention,
                                      action_dims,
                                      with_dynamic_net,
                                      with_global_state,
                                      test_policy_flag[0],
                                      multi_process_type=multi_process_type,
                                      act_steps=act_steps,
                                      act_fixed=act_fixed).product()

        self.multiagent_training = config.getboolean("actcarl", "multiagent_training")
        # logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))
        # logging.info('Policy: {} {} interaction state'.format(self.name, 'w/' if with_interaction else 'w/o'))

    def get_attention_weights(self):
        return self.model.attention_weights