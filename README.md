# AEMCARL
This project is based on the [CrowdNav](https://github.com/vita-epfl/CrowdNav).


## Abstract
<img src="https://github.com/SJWang2015/AEMCARL/blob/main/attachments/imgs/overview.png" width="1000" />
The major challenges of collision avoidance for robot navigation in crowded scenes lie in accurate environment modeling,
fast perceptions, and trustworthy motion planning policies. This paper presents a novel adaptive environment model based
collision avoidance reinforcement learning (i.e., AEMCARL) framework for an unmanned robot to achieve collision-free motions
in challenging navigation scenarios. The novelty of this work is threefold: (1) developing a hierarchical network of gated-recurrent-unit (GRU) for environment modeling; (2) developing an adaptive perception mechanism with an attention module; (3)
developing an adaptive reward function for the reinforcement learning (RL) framework to jointly train the environment model, perception function and motion planning policy. The proposed method is tested with the Gym-Gazebo simulator and a group of robots (Husky and Turtlebot) under various crowded scenes. Both simulation and experimental results have demonstrated the superior performance of the proposed method over baseline methods.

## Citation
If you use rllab for academic research, you are highly encouraged to cite the following paper:

```latex

@INPROCEEDINGS{9982107,
  author={Wang, Shuaijun and Gao, Rui and Han, Ruihua and Chen, Shengduo and Li, Chengyang and Hao, Qi},
  booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Adaptive Environment Modeling Based Reinforcement Learning for Collision Avoidance in Complex Scenes}, 
  year={2022},
  volume={},
  number={},
  pages={9011-9018},
  doi={10.1109/IROS47612.2022.9982107}}
```

## Method Overview
<img src="https://github.com/SJWang2015/AEMCARL/blob/main/attachments/imgs/framework.png" width="1000" />

## Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install crowd_sim and crowd_nav into pip
```
pip install -e .
```
3. Create environment
```
conda env create -f env.yaml
```

## Getting started
This repository is organized in two parts: gym_crowd/ folder contains the simulation environment and
crowd_nav/ folder contains codes for training and testing the policies. Details of the simulation framework can be found
[here](crowd_sim/README.md). Below are the instructions for training and testing policies, and they should be executed
inside the crowd_nav/ folder.


1. Train a policy.
```
python train.py --policy actenvcarl --test_policy_flag 5 --multi_process self_attention --optimizer Adam  --agent_timestep 0.4 --human_timestep 0.5 --reward_increment 2.0 --position_variance 2.0  --direction_variance 2.0
```
2. Test policies with 500 test cases.
```
python test.py --policy actenvcarl --test_policy_flag 5 --multi_process self_attention --agent_timestep 0.4 --human_timestep 0.5 --reward_increment 2.0 --position_variance 2.0  --direction_variance 2.0 --model_dir data/output
```
3. Run policy for one episode and visualize the result.
```
python test.py --policy actenvcarl --test_policy_flag 5 --multi_process self_attention --agent_timestep 0.4 --human_timestep 0.5 --reward_increment 2.0 --position_variance 2.0  --direction_variance 2.0 --model_dir data/output --phase test --visualize --test_case 0
```
4. Visualize a test case.
```
python test.py --policy actenvcarl --test_policy_flag 5 --multi_process self_attention --agent_timestep 0.4 --human_timestep 0.5 --reward_increment 2.0 --position_variance 2.0  --direction_variance 2.0 --model_dir data/output --phase test --visualize --test_case 0
```
5. Plot training curve.
```
python utils/plot.py data/output/output.log
```


## Simulation Videos
AEMCARL           
<img src="https://github.com/SJWang2015/AEMCARL/blob/main/attachments/imgs/18-circle-aemcarl.gif" width="600" />

Gazebo(4X)         
<img src="https://github.com/SJWang2015/AEMCARL/blob/main/attachments/imgs/gazebo.gif" width="1000" />



