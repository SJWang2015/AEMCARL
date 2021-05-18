# AEMCARL
This repository contains the codes for our paper. For more details, please refer to the paper
[Adaptive Environment Modeling Based Reinforcement Learning(AEMCARL) for Collision Avoidance in Crowded Scenes](https://github.com/SJWang2015/AEMCARL/blob/main/attachments/IROS21_0246_MS.pdf). Our RL model, which is named AEMCARL, can be find in the "crowd_nav/policy/actenvcarl".

This project is based on the [CrowdNav](https://github.com/vita-epfl/CrowdNav).


## Abstract
The major challenges of collision avoidance forrobot navigation in crowded scenes lie in accurate environmentmodeling, fast perceptions, and trustworthy motion planningpolicies. This paper presents a novel adaptive environmentmodel based collision avoidance reinforcement learning (i.e.,AEMCARL) framework for an unmanned robot to achievecollision-free motions in challenging navigation scenarios. Thenovelty of this work is threefold: (1) developing a hierarchicalnetwork of extended gated recurrent units (EGRUs) for environ-ment modeling; (2) developing an adaptive perception mecha-nism with an attention module; (3) developing a comprehensivereinforcement learning (RL) framework to jointly train theenvironment model, perception function and motion planningpolicy. The proposed method is tested with the Gym-Gazebosimulator and a group of robots (Husky and Turtlebot ) undervarious crowded scenes. Both simulation and experimentalresults have demonstrated the superior performance of theproposed method over baseline methods.


## Method Overview
<img src="https://github.com/SJWang2015/AEMCARL/blob/main/attachments/imgs/overview.png" width="1000" />

## Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install crowd_sim and crowd_nav into pip
```
pip install -e .
```

## Getting started
This repository is organized in two parts: gym_crowd/ folder contains the simulation environment and
crowd_nav/ folder contains codes for training and testing the policies. Details of the simulation framework can be found
[here](crowd_sim/README.md). Below are the instructions for training and testing policies, and they should be executed
inside the crowd_nav/ folder.


1. Train a policy.
```
python train.py --policy actenvcarl
```
2. Test policies with 500 test cases.
```
python test.py --policy orca --phase test
python test.py --policy actenvcarl --model_dir data/output --phase test
```
3. Run policy for one episode and visualize the result.
```
python test.py --policy orca --phase test --visualize --test_case 0
python test.py --policy actenvcarl --model_dir data/output --phase test --visualize --test_case 0
```
4. Visualize a test case.
```
python test.py --policy actenvcarl --model_dir data/output --phase test --visualize --test_case 0
```
5. Plot training curve.
```
python utils/plot.py data/output/output.log
```


## Simulation Videos
AEMCARL           
<img src="https://github.com/SJWang2015/AEMCARL/blob/main/attachments/imgs/18-circle-aemcarl.gif" width="600" />

Gazebo          
<img src="https://github.com/SJWang2015/AEMCARL/blob/main/attachments/imgs/gazebo.gif" width="1000" />

[Full video](https://github.com/SJWang2015/AEMCARL/blob/main/attachments/imgs/v2_original_compressed.mp4)



