import logging
import argparse
import configparser
import os
import torch
import numpy as np
import gym
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA

import glob


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=True, action='store_true')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    args = parser.parse_args()
    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
    # episode= 10
    # rl_weight_file_name = 'rl_model_{:d}.pth'.format(episode)
    # rl_weight_file = os.path.join(args.model_dir, rl_weight_file_name)
    model_path = glob.glob(os.path.join(args.model_dir, '*.pth'))
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    # device = torch.device("cpu")
    logging.info('Using device: %s', device)
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    model = policy.get_model()
    if args.model_dir is not None:
        for fn in model_path:
            with open(fn, 'rb') as fp:
                base_name = os.path.basename(fn)
                rl_weight_file_name = os.path.join(args.output_dir, base_name)
                print(rl_weight_file_name)
                model.load_state_dict(torch.load(fp))
                torch.save(model.state_dict(), rl_weight_file_name, _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    main()
