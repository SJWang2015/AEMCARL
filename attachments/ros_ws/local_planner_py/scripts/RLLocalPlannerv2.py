#! /usr/bin/env python
# coding=utf-8
import rospy
import tf2_ros
import tf2_geometry_msgs
import tf.transformations
import message_filters
import geometry_msgs.msg
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from helper.msg._ObjectArray import ObjectArray

import logging
import argparse
import configparser
import os
import torch
import numpy as np
from numpy.linalg import norm

# import threading
import math
import time

from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from envs.utils.robot import Robot
# from envs.policy.orca import ORCA
from envs.utils.human import Human
from CrowdRL import CrowdRL
from envs.utils.agent import Agent
from envs.utils.state import *

from error_code import ErrorCode
from node_state import NodeState

import scipy.interpolate as spi

import threading

delta_t = 0.1

def normalize_angle(angle):
    res = angle
    while res > math.pi:
        res -= 2.0 * math.pi
    while res < -math.pi:
        res += 2.0 * math.pi
    return res


class LocalPlannerSimpleRL:
    def __init__(self):
        rospy.init_node('local_planner_node')
        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)

        self.world_frame = 'odom'
        self.robot_frame = 'base_link'
        self.world_pose_width = 10

        self.parser = argparse.ArgumentParser('Parse configuration file')
        self.parser.add_argument('--env_config', type=str, default='output_gscarl-unicycle-v6/env.config')
        self.parser.add_argument('--policy_config', type=str, default='output_gscarl-unicycle-v6/policy.config')
        # self.parser.add_argument('--env_config', type=str, default='output_carla/env.config')
        # self.parser.add_argument('--policy_config', type=str, default='output_carla/policy.config')
        self.parser.add_argument('--policy', type=str, default='comcarl')
        self.parser.add_argument('--model_dir', type=str, default='/home/wang/Repositories/Hunter_ws/src/local_planner_py/scripts/output_gscarl-unicycle-v6/')
        # self.parser.add_argument('--model_dir', type=str, default='/home/wang/Repositories/Hunter_ws/src/local_planner_py/scripts/output_comcarl/')
        self.parser.add_argument('--il', default=False, action='store_true')
        self.parser.add_argument('--gpu', default=False)
        self.parser.add_argument('--phase', type=str, default='test')
        self.parser.add_argument('--square', default=True, action='store_true')
        self.parser.add_argument('--circle', default=False, action='store_true')
        # parser.add_argument('--video_file', type=str, default=None)
        # parser.add_argument('--traj', default=False, action='store_true')
        self.args = self.parser.parse_args()
        self.last_theta = 0.0
        self.diff_rad_threshold = 0.5

        if self.args.model_dir is not None:
            env_config_file = os.path.join(self.args.model_dir, os.path.basename(self.args.env_config))
            policy_config_file = os.path.join(self.args.model_dir, os.path.basename(self.args.policy_config))
            print(policy_config_file)
            if self.args.il:
                model_weights = os.path.join(self.args.model_dir, 'il_model.pth')
            else:
                if os.path.exists(os.path.join(self.args.model_dir, 'resumed_rl_model.pth')):
                    model_weights = os.path.join(self.args.model_dir, 'resumed_rl_model.pth')
                    print('Using resumed_rl_model.pth')
                else:
                    model_weights = os.path.join(self.args.model_dir, 'rl_model.pth')
        else:
            env_config_file = self.args.env_config
            policy_config_file = self.args.env_config

        # configure logging and device
        logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                            datefmt="%Y-%m-%d %H:%M:%S")
        # device = torch.device("cuda:0" if torch.cuda.is_available() and self.args.gpu else "cpu")
        # device = torch.device("cuda:0" if torch.cuda.is_available() and self.args.gpu else "cpu")
        device = torch.device("cpu")
        logging.info('Using device: %s', device)

        # configure policy
        self.policy = policy_factory[self.args.policy]()
        self.policy_config = configparser.RawConfigParser()
        self.policy_config.read(policy_config_file)
        self.policy.configure(self.policy_config)
        if self.policy.trainable:
            if self.args.model_dir is None:
                self.parser.error('Trainable policy must be specified with a model weights directory')
            # self.policy.get_model().load_state_dict(torch.load(model_weights))
            self.policy.get_model().load_state_dict(torch.load(model_weights, map_location={'cuda:6': 'cuda:0'}))
            # self.policy.get_model().load_state_dict(torch.load(model_weights))

        # configure environment
        self.env_config = configparser.RawConfigParser()
        self.env_config.read(env_config_file)
        
        self.robot = Robot(self.env_config, 'robot')
        self.robot.set(0, 0, 0, 0, 0, 0, np.pi / 2)
        self.robot.theta = np.pi / 2
        self.humans = []
        self.human_size = 0.3
        self.initHumans()
        # print(self.humans[0].px)
        # self.human = Human(env_config, 'humans')
        
        self.robot.set_policy(self.policy)
        self.robot.time_step = self.env_config.getfloat("env", "time_step")


        self.policy.set_phase(self.args.phase)
        self.policy.set_device(device)
        self.policy.time_step = self.robot.time_step

        self.last_pos = None
        self.done = False
        self.action = None
        self.goal = PoseStamped()
        self.last_cmd_vel_angular = 0.0
        # self.next_theta = 0.0
        # self.next_theta_rt = 0.0

        self.use_amcl_pose = True
        self.safe_dist = 0.8
        self.use_safe_policy = False
        self.use_hua = False
        self.r_ratio = 1.0
     
        self.odom_sub = rospy.Subscriber("/current_pose", PoseStamped, self.odom_callback, queue_size=1)
        self.object_array_sub = rospy.Subscriber('/tracking/objects', ObjectArray, self.object_callback, queue_size=11)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_callback, queue_size=2)

        self.cnt=1
        self.cmd_vel_pub = rospy.Publisher('/husky_velocity_controller/cmd_vel', Twist, queue_size=1)
        self.info_robot_marker_pub = rospy.Publisher('/robot_marker_info', Marker, queue_size=1)
        self.info_goal_marker_pub = rospy.Publisher('/goal_marker_info', Marker, queue_size=1)

        self.robot_marker = Marker()
        self.robot_info_marker = Marker()
        self.robot_info_marker.header.stamp = rospy.Time.now()
        self.robot_info_marker.action = self.robot_info_marker.ADD
        self.robot_info_marker.ns = 'basic_shape'
        self.robot_info_marker.type = self.robot_info_marker.TEXT_VIEW_FACING
        self.robot_info_marker.header.frame_id = 'world'
        self.robot_info_marker.id = 0
        self.robot_info_marker.pose.position.x = self.robot.px
        self.robot_info_marker.pose.position.y = self.robot.py
        self.robot_info_marker.pose.position.z = 0.5
        # self.robot_info_marker.pose.position.z += 0.4 * 1.5
        
        self.robot_info_marker.pose.orientation.x = 0
        self.robot_info_marker.pose.orientation.y = 0
        self.robot_info_marker.pose.orientation.z = 0
        self.robot_info_marker.pose.orientation.w = 1

        # self.robot_info_marker.scale.x = 2.0
        # self.robot_info_marker.scale.y = 2.0
        self.robot_info_marker.scale.z = 1.0
        self.robot_info_marker.color.a = 1.0
        self.robot_info_marker.color.r = 1
        self.robot_info_marker.color.g = 0
        self.robot_info_marker.color.b = 0

        ## Goal Marker
        self.goal_marker = Marker()
        self.goal_info_marker = Marker()
        self.goal_info_marker.header.stamp = rospy.Time.now()
        self.goal_info_marker.action = self.goal_info_marker.ADD
        self.goal_info_marker.ns = 'basic_shape'
        self.goal_info_marker.type = self.goal_info_marker.TEXT_VIEW_FACING
        self.goal_info_marker.text = "Goal"
        self.goal_info_marker.type = self.goal_info_marker.CUBE
        self.goal_info_marker.header.frame_id = 'world'
        self.goal_info_marker.id = 0
        self.goal_info_marker.pose.position.x = 0
        self.goal_info_marker.pose.position.y = 0
        self.goal_info_marker.pose.position.z = 0
        # self.goal_info_marker.pose.position.z += 0.4 * 1.5
        
        self.goal_info_marker.pose.orientation.x = 0
        self.goal_info_marker.pose.orientation.y = 0
        self.goal_info_marker.pose.orientation.z = 0
        self.goal_info_marker.pose.orientation.w = 1
        self.goal_info_marker.scale.x = 0.4
        self.goal_info_marker.scale.y = 0.4
        self.goal_info_marker.scale.z = 0.4
        self.goal_info_marker.color.a = 1.0
        self.goal_info_marker.color.r = 1.0
        self.goal_info_marker.color.g = 0
        self.goal_info_marker.color.b = 0

        # Robot Path
        self.info_robot_path_pub = rospy.Publisher('/robot_path_info', Path, queue_size=20)
        self.robot_path = Path()
        self.robot_path.header.stamp = rospy.Time.now()
        self.robot_path.header.frame_id = 'world'

        self.rate = rospy.Rate(self.cnt)
        # self.blue = lambda x: '\033[94m' + x + '\033[0m'
        self.lock = threading.Lock()
        self.planner_thread = threading.Thread(target=self.planning_thread(self.odom_sub, self.object_array_sub), name='PlanningThread')
        self.planner_thread.start()


    def initHumans(self, use_random = False):
        '''
            human.set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None)
            robot.set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None)
        '''
        human = ObservableState(-50.0, 0, 0, 0, self.human_size)
        if use_random:
            pos = np.random.randint(-5, 5, 2)
            v = np.random.randint(0, 1.5, 2)
            # print(pos)
            human = ObservableState(pos[0], pos[1], v[0], v[1], 0.6)
        self.humans.append(human)

    def object_callback(self, objects):
        self.lock.acquire()
        try:
            # print('Update objects')
            self.humans = []
            for idx in range(len(objects.list)):
                # human = Human(self.env_config, 'humans')
                human = objects.list[idx]
                theta = human.heading
                vx = human.velocity * math.cos(human.heading)
                vy = human.velocity * math.sin(human.heading)

                px = human.world_pose.point.x
                py = human.world_pose.point.y
            
                human = ObservableState(px, py, vx, vy, self.human_size)
                self.humans.append(human)
        finally:
            self.lock.release()

    def goal_callback(self, goal):
        self.lock.acquire()
        try:
            print('***********************Update goal*********************')
            self.goal = goal
            self.done = False
            self.robot.gx = self.goal.pose.position.x
            self.robot.gy = self.goal.pose.position.y

            self.goal_info_marker.text = "Goal"
            self.goal_info_marker.pose.position.x = self.goal.pose.position.x
            self.goal_info_marker.pose.position.y = self.goal.pose.position.y
            self.info_goal_marker_pub.publish(self.goal_info_marker)

            self.robot_path.poses = []
            self.info_robot_path_pub.publish(self.robot_path)

            cmd_vel_msg = Twist()
            # print("Planning!")
            # print('Robot Last Pos & Goal Pos:%f,%f, %f, %f' % (self.robot.px, self.robot.py, self.robot.gx, self.robot.gy))
            # begin = time.time()
            state = JointState(self.robot.get_full_state(), self.humans)
            self.action = self.policy.predict(state)
            # print("Time cost:", time.time()-begin)
            if self.use_hua:
                theta = self.robot.theta + np.arctan2(self.action.vy, self.action.vx)
                px = self.robot.px + self.action.vx * self.robot.time_step 
                py = self.robot.py + self.action.vy * self.robot.time_step
                self.robot.theta = np.arctan2(self.action.vy, self.action.vx)
            else:
                theta = self.robot.theta + self.action.r
                px = self.robot.px + np.cos(self.action.r) * self.action.v * self.robot.time_step 
                py = self.robot.py + np.sin(self.action.r) * self.action.v * self.robot.time_step
                self.robot.theta = self.action.r

            robot_pos = np.array([px, py]).reshape(-1,2)
            dist = []
            for i in range(len(self.humans)):
                dist.append(self.humans[i].position)
            dist = norm(np.array(dist) - robot_pos, axis=1) - self.safe_dist
            print('min dist: ' + '\033[94m' + str(min(dist)) + '\033[0m')

            # cmd_vel_msg.linear.x = self.action.v  # next_vx
            # cmd_vel_msg.linear.y = 0.0  # next_vy
        
            # cmd_vel_msg.angular.z = self.action.r
            if self.use_hua:
                action_v, action_w = self.omni2diff([self.action.vx, self.action.vy], self.robot.radius)
                cmd_vel_msg.linear.x = action_v # next_vx
                cmd_vel_msg.angular.z = action_w
            else:
                cmd_vel_msg.linear.x = self.action.v
                # cmd_vel_msg.linear.y = 0.0  # next_vy
                cmd_vel_msg.angular.z = self.action.r

            if min(dist) < 0.0 and self.use_safe_policy:
                cmd_vel_msg.linear.x = 0.0 # next_vx
                cmd_vel_msg.angular.z = 0.0
                # str_out = '\033[1;31;40m' + str(cmd_vel_msg.linear.x) + ', ' + str(cmd_vel_msg.angular.z) + '\033[0m'
                # print('Action: ' + str_out)
            else:
                pass
                # str_out = '\033[1;35;40m' + str(cmd_vel_msg.linear.x) + ', ' + str(cmd_vel_msg.angular.z) + '\033[0m'
                # print("Action:%f,%f" % (cmd_vel_msg.linear.x, cmd_vel_msg.angular.z))
                # print('Action: ' + str_out)
            self.last_pos = np.array(self.robot.get_position())
            self.cmd_vel_pub.publish(cmd_vel_msg)
        finally:
            self.lock.release()


    def odom_callback(self, odom):
    # def odom_update(self, odom):
        self.lock.acquire()
        try:
            # print('Update odom')
            if self.use_amcl_pose:
                self.robot.px = odom.pose.position.x
                self.robot.py = odom.pose.position.y
                # self.robot.vx = odom.twist.twist.linear.x
                # self.robot.vy = 0.0
                quaternion = (
                    odom.pose.orientation.x,
                    odom.pose.orientation.y,
                    odom.pose.orientation.z,
                    odom.pose.orientation.w)  
            else:
                self.robot.px = odom.pose.pose.position.x
                self.robot.py = odom.pose.pose.position.y
                quaternion = (
                    odom.pose.pose.orientation.x,
                    odom.pose.pose.orientation.y,
                    odom.pose.pose.orientation.z,
                    odom.pose.pose.orientation.w)
            euler = tf.transformations.euler_from_quaternion(quaternion)
            self.robot.theta = euler[2]

            self.robot_path.poses.append(odom)
            self.info_robot_path_pub.publish(self.robot_path)

            end_position = np.array([self.robot.px, self.robot.py])
            reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < self.robot.radius
            if reaching_goal:
                self.done = True
            else:
                self.done = False
        finally:
            self.lock.release()

        # print("Robot Pos: %f, %f" % (self.robot.px, self.robot.py))

    
    def omni2diff(self, vel_omni, radians_robot, w_max=0.1, guarantee_time=0.2):
        # vel_omni: [vx, vy]
        # global diff_rad_threshold

        # speed = vel_omni[0]#np.sqrt(vel_omni[0] ** 2 + vel_omni[1] ** 2)
        # radians = vel_omni[1]#np.arctan2(vel_omni[1], vel_omni[0])
        speed = np.sqrt(vel_omni[0] ** 2 + vel_omni[1] ** 2)
        radians = np.arctan2(vel_omni[1], vel_omni[0])

        diff_radians = radians_robot - radians

        if diff_radians > np.pi:
            diff_radians = diff_radians % (2 * np.pi) - 2 * np.pi
        elif diff_radians < -np.pi:
            diff_radians = diff_radians % (2 * np.pi) + 2 * np.pi

        if diff_radians < self.diff_rad_threshold and diff_radians > -self.diff_rad_threshold:
            w = 0
        else:
            w = -diff_radians / guarantee_time
            if w > w_max:
                w = w_max

        v = speed * np.cos(diff_radians)
        # v = speed
        if v < 0:
            v = 0

        if speed == 0:
            v = 0
            w = 0

        return v, w


    def planning_thread(self, odom, objects):
        # last_pos = np.array(self.robot.get_position())
        cmd_vel_msg = Twist()
        while not rospy.is_shutdown():
            self.info_goal_marker_pub.publish(self.goal_info_marker)
            if self.done:
                # print("Reached!")
                cmd_vel_msg = Twist()
                cmd_vel_msg.linear.x = 0
                # cmd_vel_msg.linear.y = 0
                cmd_vel_msg.angular.z = 0
                self.cmd_vel_pub.publish(cmd_vel_msg)
                # self.robot_info_marker.text = "Reached:0.0m/s,0.0"
                self.robot_info_marker.text = "Reached."
                self.robot_info_marker.pose.position.x = self.robot.px
                self.robot_info_marker.pose.position.y = self.robot.py
                self.cmd_vel_pub.publish(cmd_vel_msg)
                self.info_robot_marker_pub.publish(self.robot_info_marker)
            else:
                self.lock.acquire()
                self.info_goal_marker_pub.publish(self.goal_info_marker)
                try:
                    # print('Robot Last Pos & Goal Pos:%f,%f, %f, %f' % (self.robot.px, self.robot.py, self.robot.gx, self.robot.gy))
                    begin = time.time()
                    if len(self.humans) == 0:
                        self.initHumans()
                    # print('Size of Humans:', len(self.humans))
                    state = JointState(self.robot.get_full_state(), self.humans)
                    self.action = self.policy.predict(state)
                    if self.use_hua:
                        theta = self.robot.theta + np.arctan2(self.action.vy, self.action.vx)
                        px = self.robot.px + self.action.vx * self.robot.time_step 
                        py = self.robot.py + self.action.vy * self.robot.time_step
                        self.robot.theta = np.arctan2(self.action.vy, self.action.vx)
                    else:
                        theta = self.robot.theta + self.action.r
                        px = self.robot.px + np.cos(self.action.r) * self.action.v * self.robot.time_step 
                        py = self.robot.py + np.sin(self.action.r) * self.action.v * self.robot.time_step
                        self.robot.theta = self.action.r
                    robot_pos = np.array([px, py]).reshape(-1,2)
                    # robot_pos = np.array([self.robot.px, self.robot.py]).reshape(-1,2)
                    dist = []
                    for i in range(len(self.humans)):
                        dist.append(self.humans[i].position)
                    dist = norm(np.array(dist) - robot_pos, axis=1) - self.safe_dist
                    # print('min dist: ' + '\033[94m' + str(min(dist)) + '\033[0m')
                    
                    # print("Time cost:", time.time()-begin)
                    if self.use_hua:
                        action_v, action_w = self.omni2diff([self.action.vx, self.action.vy], self.robot.radius)
                        cmd_vel_msg.linear.x = action_v # next_vx
                        cmd_vel_msg.angular.z = action_w
                    else:
                        cmd_vel_msg.linear.x = self.action.v
                        # cmd_vel_msg.linear.y = 0.0  # next_vy
                        cmd_vel_msg.angular.z = self.action.r
                    if min(dist) < 0.0 and self.use_safe_policy:
                        cmd_vel_msg.linear.x = 0.0 # next_vx
                        cmd_vel_msg.angular.z = 0.0
                        str_out = '\033[1;31;40m' + str(cmd_vel_msg.linear.x) + ', ' + str(cmd_vel_msg.angular.z) + '\033[0m'
                        print('Action: ' + str_out)
                    else:
                        str_out = '\033[1;35;40m' + str(cmd_vel_msg.linear.x) + ', ' + str(cmd_vel_msg.angular.z) + '\033[0m'
                        print('Action: ' + str_out)
                    # print("Action:%f,%f" % (cmd_vel_msg.linear.x, cmd_vel_msg.angular.z))
                    self.cmd_vel_pub.publish(cmd_vel_msg)
                    print("Time cost:", time.time()-begin)
                    # self.robot_info_marker.text = "Planning:" + str(cmd_vel_msg.linear.x)[:4] + 'm/s,' + str(math.degrees(cmd_vel_msg.angular.z))[:5]
                    self.robot_info_marker.text = "Planning..."
                    self.robot_info_marker.pose.position.x = px
                    self.robot_info_marker.pose.position.y = py
                    # self.marker_pub.publish(self.robot_marker)
                    self.last_pos = np.array(self.robot.get_position())
                    self.info_robot_marker_pub.publish(self.robot_info_marker)

                finally:
                    self.lock.release()

if __name__ == '__main__':
    rospy.init_node('local_planner_node')
    local_planner = LocalPlannerSimpleRL()
    rospy.spin()
