from math import atan2, pi
from operator import index
from time import time
import math
from crowd_sim.envs.utils.state import FullState
import torch
import numpy as np
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_nav.policy.cadrl import CADRL
import cv2
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
from torch._C import dtype
from numba import jit, cuda

table_size = 10000
table_offset = 5000
scale = 1000
norm_lookuptable = np.zeros(table_size, dtype=np.float)
for index in range(table_size):
    int_x = index - table_offset
    float_x = int_x / scale
    norm_lookuptable[index] = norm.pdf(float_x, 0.0, 1)
    pass
print("generate lookuptable")


@jit(nopython=True)
def lookup(raw_index):
    index = int(raw_index * scale + table_offset)
    if index > (table_size - 1) or index < 0:
        return 0.0
    else:
        return norm_lookuptable[index]


@jit(nopython=True)
def build_single_human_map(human_state, width, height, x_offset, y_offset, resolution, time_step):
    gmap = np.zeros((width, height), dtype=np.float)
    px = human_state[0]
    py = human_state[1]
    vx = human_state[2]
    vy = human_state[3]
    radius = human_state[4]
    p_x = px + x_offset
    p_y = py + y_offset
    center_x = round(p_x / resolution)
    center_y = round(p_y / resolution)
    radius_pixel_len = round(radius / resolution)

    if (vx == 0) and (vy == 0):
        return

    # movement in a time step
    delta_x = vx * time_step
    delta_y = vy * time_step
    move_distance = math.hypot(delta_x, delta_y)

    # calculate boundary of ROI
    roi_x_low_pos = p_x - move_distance
    roi_x_high_pos = p_x + move_distance

    roi_x_low_idx = round(roi_x_low_pos / resolution)
    roi_x_high_idx = round(roi_x_high_pos / resolution)

    roi_y_low_pos = p_y - move_distance
    roi_y_high_pos = p_y + move_distance

    roi_y_low_idx = round(roi_y_low_pos / resolution)
    roi_y_high_idx = round(roi_y_high_pos / resolution)

    roi_width = roi_x_high_idx - roi_x_low_idx
    roi_height = roi_y_high_idx - roi_y_low_idx

    heading_angle = atan2(delta_y, delta_x)  # rad
    # mean = [center_y, center_x, heading_angle]
    mean = [0.0, 0.0, 0.0]
    cov = np.eye(3)
    sum_probability = 0.0

    loop_cnt = 0
    for y_index in range(roi_y_low_idx, roi_y_high_idx):
        for x_index in range(roi_x_low_idx, roi_x_high_idx):
            if math.hypot((x_index - center_x), (y_index - center_y)) * resolution > move_distance:
                continue
            if (y_index - center_y) == 0 and (x_index - center_x) == 0:
                angle = heading_angle
            else:
                angle = atan2(y_index - center_y, x_index - center_x)
            x = [
                6.0 * (y_index - center_y) / (roi_height + 0.01), 6.0 * (x_index - center_x) / (roi_width + 0.01),
                5.0 * (angle - heading_angle) / math.pi
            ]
            probability = lookup(x[0]) * lookup(x[1]) * lookup(x[2])
            # probability = multivariate_normal.pdf(x, mean=mean, cov=cov)
            gmap[y_index, x_index] += probability
            sum_probability += probability
            loop_cnt += 1
    for y_index in range(roi_y_low_idx, roi_y_high_idx):
        for x_index in range(roi_x_low_idx, roi_x_high_idx):
            if math.hypot((x_index - center_x), (y_index - center_y)) * resolution > move_distance:
                continue
            gmap[y_index, x_index] = gmap[y_index, x_index] / sum_probability
    return gmap


@jit(nopython=True)
def build_all_human_map(human_states, width, height, x_offset, y_offset, resolution, time_step):
    global_gmap = np.zeros((width, height), dtype=np.float_)
    for index in range(human_states.shape[0]):
        gmap = np.zeros((width, height), dtype=np.float_)
        px = human_states[index, 0]
        py = human_states[index, 1]
        vx = human_states[index, 2]
        vy = human_states[index, 3]
        radius = human_states[index, 4]
        p_x = px + x_offset
        p_y = py + y_offset
        center_x = round(p_x / resolution)
        center_y = round(p_y / resolution)
        radius_pixel_len = round(radius / resolution)

        # movement in a time step
        delta_x = vx * time_step
        delta_y = vy * time_step
        move_distance = math.hypot(delta_x, delta_y)
        # move_radius = round((1.0 * time_step) / resolution)

        # calculate boundary of ROI
        roi_x_low_pos = p_x - move_distance
        roi_x_high_pos = p_x + move_distance

        roi_x_low_idx = round(roi_x_low_pos / resolution)
        roi_x_high_idx = round(roi_x_high_pos / resolution)

        roi_y_low_pos = p_y - move_distance
        roi_y_high_pos = p_y + move_distance

        roi_y_low_idx = round(roi_y_low_pos / resolution)
        roi_y_high_idx = round(roi_y_high_pos / resolution)

        roi_width = roi_x_high_idx - roi_x_low_idx
        roi_height = roi_y_high_idx - roi_y_low_idx

        heading_angle = atan2(delta_y, delta_x)  # rad
        # mean = [center_y, center_x, heading_angle]
        mean = [0.0, 0.0, 0.0]
        cov = np.eye(3)
        sum_probability = 0.0

        loop_cnt = 0
        for y_index in range(roi_y_low_idx, roi_y_high_idx):
            for x_index in range(roi_x_low_idx, roi_x_high_idx):
                if math.hypot((x_index - center_x), (y_index - center_y)) * resolution > move_distance:
                    continue
                if (y_index - center_y) == 0 and (x_index - center_x) == 0:
                    angle = heading_angle
                else:
                    angle = atan2(y_index - center_y, x_index - center_x)

                # warp to -pi pi
                delta_angle = angle - heading_angle
                if delta_angle <= -math.pi:
                    delta_angle += 2.0 * math.pi
                elif delta_angle >= math.pi:
                    delta_angle = 2.0 * math.pi - delta_angle

                x = [
                    6.0 * (y_index - center_y) / (roi_height + 0.01), 6.0 * (x_index - center_x) / (roi_width + 0.01),
                    5.0 * delta_angle / math.pi
                ]

                probability_1 = lookup(x[0]) * lookup(x[1]) * lookup(x[2])
                probability = probability_1
                gmap[y_index, x_index] += probability
                sum_probability += probability
                loop_cnt += 1

        for y_index in range(roi_y_low_idx, roi_y_high_idx):
            for x_index in range(roi_x_low_idx, roi_x_high_idx):
                if math.hypot((x_index - center_x), (y_index - center_y)) * resolution > move_distance:
                    continue
                if sum_probability == 0.0:
                    gmap[y_index, x_index] = 0.0
                else:
                    gmap[y_index, x_index] = gmap[y_index, x_index] / sum_probability
        global_gmap += gmap
    return global_gmap


class GaussianLookupTable(object):
    def __init__(self):
        self.table_size = 10000
        self.table_offset = 5000
        self.scale = 1000
        self.norm_lookuptable = np.zeros(self.table_size, dtype=np.float)  # -5000，5000，映射到 -5，5
        for index in range(self.table_size):
            int_x = index - self.table_offset
            float_x = index / self.scale
            self.norm_lookuptable[index] = norm.pdf(float_x, 0.0, 1)
            pass

    def lookup(self, raw_index):
        index = int(raw_index * self.scale + self.table_offset)
        if index > (self.table_size - 1) or index < 0:
            return 0.0
        else:
            return self.norm_lookuptable[index]


class gridmap(object):
    def __init__(self, xlimit, ylimit, resolution=0.05, time_step=1.0):
        self.time_step = time_step
        self.x_offset = round(xlimit / 2)
        self.y_offset = round(ylimit / 2)
        self.width = round(xlimit / resolution)
        self.height = round(ylimit / resolution)
        self.gmap = np.zeros((self.width, self.height), dtype=np.float)
        self.human_states = []
        self.resolution = resolution

        self.agent_center_x = 0
        self.agent_center_y = 0
        self.check_pixel_radius = 1

        self.norm = GaussianLookupTable()
        pass

    def clear_map(self):
        self.gmap = np.zeros((self.width, self.height), dtype=np.float)
        self.human_states = []

    def compute_occupied_probability(self, agent_state: FullState):
        agent_x = agent_state.px + self.x_offset
        agent_y = agent_state.py + self.y_offset
        center_x_idx = round(agent_x / self.resolution)
        center_y_idx = round(agent_y / self.resolution)

        self.agent_center_x = center_x_idx
        self.agent_center_y = center_y_idx

        if len(self.human_states) > 0:
            human_radius = self.human_states[0].radius
        else:
            human_radius = 0.0
        search_radius = agent_state.radius + human_radius
        x_low_raw = 0 if (agent_x - search_radius) < 0 else (agent_x - search_radius)
        x_low = round(x_low_raw / self.resolution)
        x_high = round((agent_x + search_radius) / self.resolution)

        y_low_raw = 0 if (agent_y - search_radius) < 0 else (agent_y - search_radius)
        y_low = round(y_low_raw / self.resolution)
        y_high = round((agent_y + search_radius) / self.resolution)

        self.check_pixel_radius = round(search_radius / self.resolution)

        sum_probability = 0.0
        for index_y in range(y_low, y_high):
            for index_x in range(x_low, x_high):
                if math.hypot((index_x - center_x_idx), (index_y - center_y_idx)) * self.resolution > search_radius:
                    continue
                if (index_x < self.width) and (index_y < self.height):
                    sum_probability += self.gmap[index_y, index_x]
        # if sum_probability > 0.0:
        #     print("collision probability:", sum_probability)
        return sum_probability

    def get_visualized_map(self):
        visual_map = np.zeros_like(self.gmap, dtype=np.uint8)
        for index_y in range(self.gmap.shape[0]):
            for index_x in range(self.gmap.shape[1]):
                pixel = self.gmap[index_y, index_x] * 255 * 30
                pixel = 255 if pixel > 255 else pixel
                visual_map[index_y, index_x] = 255 - pixel

        visual_map = cv2.circle(visual_map, (self.agent_center_y, self.agent_center_x),
                                self.check_pixel_radius,
                                color=[0, 0, 255],
                                thickness=2)
        return visual_map

    def build_map_cpu(self, human_states):
        self.human_states = human_states
        human_lists = []
        global_gmap = []
        for human in human_states:
            human_info = [human.px, human.py, human.vx, human.vy, human.radius]
            human_lists.append(human_info)
            global_gmap.append(None)
        human_info_array = np.array(human_lists)
        map = build_all_human_map(human_info_array, self.width, self.height, self.x_offset, self.y_offset,
                                  self.resolution, self.time_step)
        self.gmap = map
        return


class MultiHumanRL(CADRL):
    def __init__(self):
        super().__init__()
        self.map = gridmap(xlimit=12, ylimit=12, resolution=0.05, time_step=1.0)
        self.save_map_index = 0
        self.save_map_path_prefix = "debug_map/"
        self.use_rate_statistic = True
        self.statistic_info = [0 for index in range(20)]

    def get_act_statistic_info(self):
        return self.statistic_info

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
            need_build_map = True
            single_step_cnt = 0
            for action in self.action_space:
                next_self_state = self.propagate(state.self_state, action)

                if self.query_env:
                    next_human_states, reward, done, info = self.env.onestep_lookahead(action, need_build_map)
                    need_build_map = False
                else:
                    next_human_states = [
                        self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                        for human_state in state.human_states
                    ]
                    self.map.build_map_cpu(next_human_states)
                    collision_probability = self.map.compute_occupied_probability(next_self_state)
                    reward = self.compute_reward(next_self_state, next_human_states, collision_probability)

                batch_next_states = torch.cat([
                    torch.Tensor([next_self_state + next_human_state]).to(self.device)
                    for next_human_state in next_human_states
                ],
                                              dim=0)
                rotated_batch_input = self.rotate(batch_next_states).unsqueeze(0)
                if self.with_om:
                    if occupancy_maps is None:
                        occupancy_maps = self.build_occupancy_maps(next_human_states).unsqueeze(0)
                    rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps], dim=2)
                # VALUE UPDATE

                value, score = self.model(rotated_batch_input)
                next_state_value = value.data.item()

                value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value

                self.action_values.append(value)
                # ionly save the step cnt of used action
                if value > max_value:
                    max_value = value
                    max_action = action
                    if self.use_rate_statistic:
                        single_step_cnt = self.model.get_step_cnt()
                        
            if max_action is None:
                raise ValueError('Value network is not well trained. ')
            if self.use_rate_statistic:
                if single_step_cnt < len(self.statistic_info):
                    self.statistic_info[single_step_cnt] += 1
                else:
                    print("step count too large!!")
                pass

        if self.phase == 'train':
            self.last_state = self.transform(state)
        return max_action

    def compute_reward(self, nav, humans, collision_prob=0.0):
        # collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(humans):
            dist = np.linalg.norm((nav.px - human.px, nav.py - human.py)) - nav.radius - human.radius
            if dist < 0:
                collision = True
                break
            if dist < dmin:
                dmin = dist

        # check if reaching the goal
        reaching_goal = np.linalg.norm((nav.px - nav.gx, nav.py - nav.gy)) < nav.radius
        stationary_dist = np.linalg.norm((nav.px - nav.gx, nav.py + nav.gy))

        stationary_state = False
        if abs(nav.vx) <= 0.0001 and abs(nav.vy) <= 0.0001:
            stationary_state = True

        if collision:
            reward = -0.25
        elif reaching_goal:
            reward = 1
        elif (stationary_dist < self.env.stationary_penalty_dist) and (  # 给不动一个惩罚项
                self.env.agent_prev_vx is not None
                and self.env.agent_prev_vy is not None) and stationary_state and not (reaching_goal):
            reward = self.env.stationary_penalty * (self.env.stationary_penalty_dist -
                                                    stationary_dist) / self.env.stationary_penalty_dist
        else:
            reward = -0.25 * collision_prob * 4.0

        return reward

    def transform(self, state):
        """
        Take the state passed from agent and transform it to the input of value network

        :param state:
        :return: tensor of shape (# of humans, len(state))
        """
        state_tensor = torch.cat(
            [torch.Tensor([state.self_state + human_state]).to(self.device) for human_state in state.human_states],
            dim=0)
        if self.with_om:
            occupancy_maps = self.build_occupancy_maps(state.human_states)
            state_tensor = torch.cat([self.rotate(state_tensor), occupancy_maps], dim=1)
        else:
            state_tensor = self.rotate(state_tensor)
        return state_tensor

    def input_dim(self):
        return self.joint_state_dim + (self.cell_num**2 * self.om_channel_size if self.with_om else 0)

    def build_occupancy_maps(self, human_states):
        """

        :param human_states:
        :return: tensor of shape (# human - 1, self.cell_num ** 2)
        """
        occupancy_maps = []
        for human in human_states:
            other_humans = np.concatenate([
                np.array([(other_human.px, other_human.py, other_human.vx, other_human.vy)])
                for other_human in human_states if other_human != human
            ],
                                          axis=0)
            other_px = other_humans[:, 0] - human.px
            other_py = other_humans[:, 1] - human.py
            # new x-axis is in the direction of human's velocity
            human_velocity_angle = np.arctan2(human.vy, human.vx)
            other_human_orientation = np.arctan2(other_py, other_px)
            rotation = other_human_orientation - human_velocity_angle
            distance = np.linalg.norm([other_px, other_py], axis=0)
            other_px = np.cos(rotation) * distance
            other_py = np.sin(rotation) * distance

            # compute indices of humans in the grid
            other_x_index = np.floor(other_px / self.cell_size + self.cell_num / 2)
            other_y_index = np.floor(other_py / self.cell_size + self.cell_num / 2)
            other_x_index[other_x_index < 0] = float('-inf')
            other_x_index[other_x_index >= self.cell_num] = float('-inf')
            other_y_index[other_y_index < 0] = float('-inf')
            other_y_index[other_y_index >= self.cell_num] = float('-inf')
            grid_indices = self.cell_num * other_y_index + other_x_index
            occupancy_map = np.isin(range(self.cell_num**2), grid_indices)
            if self.om_channel_size == 1:
                occupancy_maps.append([occupancy_map.astype(int)])
            else:
                # calculate relative velocity for other agents
                other_human_velocity_angles = np.arctan2(other_humans[:, 3], other_humans[:, 2])
                rotation = other_human_velocity_angles - human_velocity_angle
                speed = np.linalg.norm(other_humans[:, 2:4], axis=1)
                other_vx = np.cos(rotation) * speed
                other_vy = np.sin(rotation) * speed
                dm = [list() for _ in range(self.cell_num**2 * self.om_channel_size)]
                for i, index in np.ndenumerate(grid_indices):
                    if index in range(self.cell_num**2):
                        if self.om_channel_size == 2:
                            dm[2 * int(index)].append(other_vx[i])
                            dm[2 * int(index) + 1].append(other_vy[i])
                        elif self.om_channel_size == 3:
                            dm[2 * int(index)].append(1)
                            dm[2 * int(index) + 1].append(other_vx[i])
                            dm[2 * int(index) + 2].append(other_vy[i])
                        else:
                            raise NotImplementedError
                for i, cell in enumerate(dm):
                    dm[i] = sum(dm[i]) / len(dm[i]) if len(dm[i]) != 0 else 0
                occupancy_maps.append([dm])

        return torch.from_numpy(np.concatenate(occupancy_maps, axis=0)).float()
