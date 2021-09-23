# coding=utf-8
import logging
# import gym
# import matplotlib.lines as mlines
import numpy as np
# import rvo2
# from matplotlib import patches
from numpy.linalg import norm
from envs.utils.human import Human
from envs.utils.info import *
from envs.utils.utils import point_to_segment_dist, dist


class CrowdRL():
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.human_times = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        self.stationary_penalty = None
        self.stationary_penalty_dist = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_sim = None
        self.test_sim = None
        self.square_width = None
        self.circle_radius = None # the negtive value of y-axis is the start point of robot 
        self.human_num = None
        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
        self.agent_prev_vx = None
        self.agent_prev_vy = None
        self.start_position_x = None
        self.start_position_y = None
        self.goal_position_x = None
        self.goal_position_y = None


    def configure(self, config):
        self.config = config
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        self.stationary_penalty = config.getfloat('reward','stationary_penalty')
        self.stationary_penalty_dist = config.getfloat('reward','stationary_penalty_dist')
        if self.config.get('humans', 'policy') == 'orca':
            self.circle_radius = config.getfloat('sim', 'circle_radius')
        #     self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        #     self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
        #                       'test': config.getint('env', 'test_size')}
        #     self.train_val_sim = config.get('sim', 'train_val_sim')
        #     self.test_sim = config.get('sim', 'test_sim')
        #     # self.square_width = config.getfloat('sim', 'square_width')
        #     # self.human_num = config.getint('sim', 'human_num')
        else:
            raise NotImplementedError
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def set_robot(self, robot):
        self.robot = robot

    def Set_Agent_State(agent_num):
        self.human_num = 3
        self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
        self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
        self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
        self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)


    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        # if test_case is not None:
        #     self.case_counter[phase] = test_case
        self.global_time = 0

        # counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
        #                     'val': 0, 'test': self.case_capacity['val']}
        self.robot.set(self.start_position_x, self.start_position_y, self.goal_position_x, self.goal_position_y, 0, 0, np.pi / 2)
        self.agent_prev_vx = None
        self.agent_prev_vy = None
      
        self.human_num = 3
        self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
        self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
        self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
        self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)

        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step


        self.states = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()

        # get current observation
        if self.robot.sensor == 'coordinates':
            ob = [human.get_observable_state() for human in self.humans]
            # ob = None #返回当前环境中的目标的状态信息
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError

        return ob

    def onestep_lookahead(self, action):
        return self.step(action, update=False)

    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)

        """
        # print("***********************")
        # print(self.robot.policy.device)
        # print(self.robot.policy.phase)
        # print("***********************")
        human_actions = []
        for human in self.humans:
            # observation for humans is always coordinates 从感知模块传进来消息
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
            # print(ob)
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
            human_actions.append(human.act(ob))

        # collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            if self.robot.kinematics == 'holonomic':
                vx = human.vx - action.vx
                vy = human.vy - action.vy
            else:
                vx = human.vx - action.v * np.cos(action.r + self.robot.theta)
                vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                break
            elif closest_dist < dmin:
                dmin = closest_dist
        # print(self.circle_radius)
        stationary_dist = ((self.robot.py + self.circle_radius) ** 2 + (self.robot.px ** 2 )) ** (1 / 2)

        # collision detection between humans
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        # check if reaching the goal
        end_position = np.array(self.robot.compute_position(action, self.time_step))
        # print(end_position)
        # print(self.robot.get_goal_position())
        reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < self.robot.radius
        # print(reaching_goal)

        stationary_state = False
        if self.robot.kinematics == 'holonomic':
            if abs(action.vx) <= 0.0001 and abs(action.vy) <= 0.0001:
                stationary_state = True
        else:
            if abs(action.v) <= 0.0001:
                stationary_state = True

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = False
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            # only penalize agent for getting too close if it's visible
            # adjust the reward based on FPS
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Danger(dmin)
        elif stationary_dist < self.stationary_penalty_dist and (self.agent_prev_vx is not None and self.agent_prev_vy is not None) and stationary_state and not(reaching_goal):
            reward = (self.stationary_penalty_dist - stationary_dist) / self.stationary_penalty_dist * self.stationary_penalty
            # reward = self.stationary_penalty
            done = False
            info = Stationary()
        else:
            reward = 0
            done = False
            info = Nothing()

        if self.robot.sensor == 'coordinates':
            ob = [human.get_observable_state() for human in self.humans]  # 从感知模块获取数据 self.humans 
            #[human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError

        if self.robot.kinematics == 'holonomic':
            self.agent_prev_vx = action.vx
            self.agent_prev_vy = action.vy
        else:
            self.agent_prev_vx = action.v * np.cos(action.r + self.robot.theta)
            self.agent_prev_vy = action.v * np.sin(action.r + self.robot.theta)

        return ob, reward, done, info

        # return ob, reward, done, info


