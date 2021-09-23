import sys
import logging
import argparse
import configparser
import os
import shutil
import torch
import gym

# import git
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.trainer import Trainer
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory

dir_prefix = "crowd_nav/"


def main():
    parser = argparse.ArgumentParser("Parse configuration file")
    parser.add_argument("--env_config", type=str, default="configs/env.config")
    parser.add_argument("--policy", type=str, default="cadrl")
    parser.add_argument("--policy_config", type=str, default="configs/policy.config")
    # parser.add_argument('--train_config', type=str, default='configs/train-test.config')
    parser.add_argument("--train_config", type=str, default="configs/train.config")
    parser.add_argument("--output_dir", type=str, default="data2/actenvcarl_alltf")
    parser.add_argument("--weights", type=str)
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--gpu", default=True, action="store_true")
    parser.add_argument("--debug", default=True, action="store_true")
    parser.add_argument("--phase", type=str, default="train")
    parser.add_argument("--test_policy_flag", type=str, default="1")
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--multi_process", type=str, default="average")

    # reward parameters
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

    agent_timestep = args.agent_timestep
    human_timestep = args.human_timestep
    reward_increment = args.reward_increment
    position_variance = args.position_variance
    direction_variance = args.direction_variance

    # agent_visable = args.visable

    optimizer_type = args.optimizer

    # configure paths
    make_new_dir = True
    if os.path.exists(args.output_dir):
        key = input("Output directory already exists! Overwrite the folder? (y/n)")
        if key == "y" and not args.resume:
            shutil.rmtree(args.output_dir)
        else:
            make_new_dir = False
            args.env_config = os.path.join(args.output_dir, os.path.basename(args.env_config))
            args.policy_config = os.path.join(args.output_dir, os.path.basename(args.policy_config))
            args.train_config = os.path.join(args.output_dir, os.path.basename(args.train_config))
            shutil.copy("utils/trainer.py", args.output_dir)
            shutil.copy("policy/" + args.policy + ".py", args.output_dir)

    if make_new_dir:
        os.makedirs(args.output_dir)
        shutil.copy(args.env_config, args.output_dir)
        shutil.copy(args.policy_config, args.output_dir)
        shutil.copy(args.train_config, args.output_dir)
        shutil.copy("utils/trainer.py", args.output_dir)
        shutil.copy("policy/" + args.policy + ".py", args.output_dir)
    log_file = os.path.join(args.output_dir, "output.log")
    il_weight_file = os.path.join(args.output_dir, "il_model.pth")
    rl_weight_file = os.path.join(args.output_dir, "rl_model.pth")

    # configure logging
    mode = "a" if args.resume else "w"
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(
        level=level,
        handlers=[stdout_handler, file_handler],
        format="%(asctime)s, %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # repo = git.Repo(search_parent_directories=True)
    # logging.info('Current git head hash code: %s'.format(repo.head.object.hexsha))
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info("Using device: %s", device)

    # configure policy
    policy = policy_factory[args.policy]()
    # policy.set_phase(args.phase)
    print("policy type: ", type(policy))

    if not policy.trainable:
        parser.error("Policy has to be trainable")
    if args.policy_config is None:
        parser.error("Policy config has to be specified for a trainable network")
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)

    policy_config.set("actenvcarl", "test_policy_flag", args.test_policy_flag)
    policy_config.set("actenvcarl", "multi_process", args.multi_process)
    policy_config.set("actenvcarl", "act_steps", args.act_steps)
    policy_config.set("actenvcarl", "act_fixed", args.act_fixed)

    policy.configure(policy_config)
    policy.set_device(device)

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env_config.set("reward", "agent_timestep", agent_timestep)
    env_config.set("reward", "human_timestep", human_timestep)
    env_config.set("reward", "reward_increment", reward_increment)
    env_config.set("reward", "position_variance", position_variance)
    env_config.set("reward", "direction_variance", direction_variance)
    env = gym.make("CrowdSim-v0")
    env.configure(env_config)

    # 只是为了获取部分的环境配置信息,如半径速度啥的
    robot = Robot(env_config, "robot")
    env.set_robot(robot)

    # read training parameters
    if args.train_config is None:
        parser.error("Train config has to be specified for a trainable network")

    train_config = configparser.RawConfigParser()
    train_config.read(args.train_config)
    rl_learning_rate = train_config.getfloat("train", "rl_learning_rate")
    train_batches = train_config.getint("train", "train_batches")
    train_episodes = train_config.getint("train", "train_episodes")
    sample_episodes = train_config.getint("train", "sample_episodes")
    target_update_interval = train_config.getint("train", "target_update_interval")
    evaluation_interval = train_config.getint("train", "evaluation_interval")
    capacity = train_config.getint("train", "capacity")
    epsilon_start = train_config.getfloat("train", "epsilon_start")
    epsilon_end = train_config.getfloat("train", "epsilon_end")
    epsilon_decay = train_config.getfloat("train", "epsilon_decay")
    checkpoint_interval = train_config.getint("train", "checkpoint_interval")

    # configure trainer and explorer
    memory = ReplayMemory(capacity)

    model = policy.get_model()

    batch_size = train_config.getint("trainer", "batch_size")

    trainer = Trainer(model, memory, device, batch_size)

    explorer = Explorer(env, robot, device, memory, policy.gamma, target_policy=policy)

    # imitation learning
    if args.resume:
        if not os.path.exists(rl_weight_file):
            logging.error("RL weights does not exist")
        model.load_state_dict(torch.load(rl_weight_file))
        rl_weight_file = os.path.join(args.output_dir, "resumed_rl_model.pth")
        logging.info("Load reinforcement learning trained weights. Resume training")
    elif os.path.exists(il_weight_file):
        model.load_state_dict(torch.load(il_weight_file))
        logging.info("Load imitation learning trained weights.")
    else:
        il_episodes = train_config.getint("imitation_learning", "il_episodes")
        il_policy = train_config.get("imitation_learning", "il_policy")
        il_epochs = train_config.getint("imitation_learning", "il_epochs")
        il_learning_rate = train_config.getfloat("imitation_learning", "il_learning_rate")
        trainer.set_learning_rate(il_learning_rate, optimizer_type)
        if robot.visible:
            safety_space = 0
        else:
            safety_space = train_config.getfloat("imitation_learning", "safety_space")
        il_policy = policy_factory[il_policy]()
        il_policy.multiagent_training = policy.multiagent_training
        il_policy.safety_space = safety_space

        print("il_policy: ", type(il_policy))
        robot.set_policy(il_policy)
        explorer.run_k_episodes(200, "train", update_memory=True, imitation_learning=True)

        trainer.optimize_epoch(il_epochs)

        torch.save(model.state_dict(), il_weight_file)
        logging.info("Finish imitation learning. Weights saved.")
        logging.info("Experience set size: %d/%d", len(memory), memory.capacity)

    explorer.update_target_model(model)

    # reinforcement learning
    policy.set_env(env)
    robot.set_policy(policy)
    robot.print_info()
    trainer.set_learning_rate(rl_learning_rate, optimizer_type)
    # fill the memory pool with some RL experienceo
    if args.resume:
        robot.policy.set_epsilon(epsilon_end)
        explorer.run_k_episodes(100, "train", update_memory=True, episode=0)
        logging.info("Experience set size: %d/%d", len(memory), memory.capacity)
    episode = 0
    while episode < train_episodes:
        if args.resume:
            epsilon = epsilon_end
        else:
            if episode < epsilon_decay:
                epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
            else:
                epsilon = epsilon_end
        robot.policy.set_epsilon(epsilon)

        # evaluate the model
        if episode % evaluation_interval == 0:
            explorer.run_k_episodes(env.case_size["val"], "val", episode=episode)

        # sample k episodes into memory and optimize over the generated memory
        explorer.run_k_episodes(sample_episodes, "train", update_memory=True, episode=episode)
        trainer.optimize_batch(train_batches)
        episode += 1

        if episode % target_update_interval == 0:
            explorer.update_target_model(model)

        if episode != 0 and episode % checkpoint_interval == 0:
            # rl_weight_file_name = 'rl_model_' + str(episode)  + '.pth'
            rl_weight_file_name = "rl_model_{:d}.pth".format(episode)
            rl_weight_file = os.path.join(args.output_dir, rl_weight_file_name)
            torch.save(model.state_dict(), rl_weight_file, _use_new_zipfile_serialization=False)

    # final test
    explorer.run_k_episodes(env.case_size["test"], "test", episode=episode)


if __name__ == "__main__":
    main()
