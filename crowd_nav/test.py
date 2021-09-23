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


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=True, action='store_true')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')

    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')

    parser.add_argument("--test_policy_flag", type=str, default="1")
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--multi_process", type=str, default="average")
    parser.add_argument("--human_num", type=int, default=5)

    # 环境reward相关的参数
    parser.add_argument("--agent_timestep", type=float, default=0.4)
    parser.add_argument("--human_timestep", type=float, default=0.5)
    parser.add_argument("--reward_increment", type=float, default=4.0)
    parser.add_argument("--position_variance", type=float, default=4.0)
    parser.add_argument("--direction_variance", type=float, default=4.0)

    # visable or not
    parser.add_argument("--visible", default=False, action="store_true")

    # act step cnt
    parser.add_argument("--act_steps", type=int, default=1)
    parser.add_argument("--act_fixed", default=False, action="store_true")

    args = parser.parse_args()

    human_num = args.human_num
    agent_timestep = args.agent_timestep
    human_timestep = args.human_timestep
    reward_increment = args.reward_increment
    position_variance = args.position_variance
    direction_variance = args.direction_variance

    agent_visible = args.visible
    print(agent_timestep, " ", human_timestep, " ", reward_increment, " ", position_variance, " ", direction_variance,
          " ", agent_visible)

    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
        if args.il:
            print("model: il_model.pth")
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
        else:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                print("model: resumed_rl_model.pth")
                model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            else:
                model_path_ = "rl_model_5400.pth"
                print("model: ", model_path_)
                model_weights = os.path.join(args.model_dir, model_path_)
    else:
        env_config_file = args.env_config
        policy_config_file = args.env_config

    # configure logging and device
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    # device = torch.device("cpu")
    logging.info('Using device: %s', device)

    # configure policy
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)

    policy_config.set("actenvcarl", "test_policy_flag", args.test_policy_flag)
    policy_config.set("actenvcarl", "multi_process", args.multi_process)
    policy_config.set("actenvcarl", "act_steps", args.act_steps)
    policy_config.set("actenvcarl", "act_fixed", args.act_fixed)


    policy.configure(policy_config)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        # policy.get_model().load_state_dict(torch.load(model_weights, map_location={'cuda:2':'cuda:0'}))
        policy.get_model().load_state_dict(torch.load(model_weights))
        # policy.get_model().load_state_dict(torch.jit.load(model_weights))

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)

    env_config.set("sim", "human_num", human_num)
    env_config.set("reward", "agent_timestep", agent_timestep)
    env_config.set("reward", "human_timestep", human_timestep)
    env_config.set("reward", "reward_increment", reward_increment)
    env_config.set("reward", "position_variance", position_variance)
    env_config.set("reward", "direction_variance", direction_variance)
    # env_config.set("robot", "visible", agent_visible)

    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'
    robot = Robot(env_config, 'robot')
    robot.visible = agent_visible

    print("robot visable: ", robot.visible)

    robot.set_policy(policy)
    env.set_robot(robot)
    explorer = Explorer(env, robot, device, gamma=0.9)

    policy.set_phase(args.phase)
    policy.set_device(device)
    # set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = 0
        else:
            robot.policy.safety_space = 0
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

    policy.set_env(env)
    robot.print_info()
    if args.visualize:
        ob = env.reset(args.phase, args.test_case)
        done = False
        last_pos = np.array(robot.get_position())
        while not done:
            action = robot.act(ob)
            ob, _, done, info = env.step(action)
            current_pos = np.array(robot.get_position())
            logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
            last_pos = current_pos
        if args.traj:
            env.render('traj', args.video_file)
        else:
            env.render('video', args.video_file)

        logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
        if robot.visible and info == 'reach goal':
            human_times = env.get_human_times()
            logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))
    else:
        logging.info("run k episodes")

        # explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)
        explorer.run_k_episodes(50, args.phase, print_failure=True)


if __name__ == '__main__':
    main()
