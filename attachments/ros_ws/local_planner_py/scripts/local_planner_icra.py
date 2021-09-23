#!/usr/bin/env python
import math
import time
import threading
import rospy
import tf
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist

from node_state import NodeState
from error_code import ErrorCode

from roborts_msgs.msg import LocalPlannerAction


class LocalPlanner:
    def __init__(self):
        self.local_goal = PoseStamped()
        self.tmp_plan = Path()
        self.global_plan = Path()
        self.tf_listener = tf.TransformListener()  # About rospy-tf usage, see http://wiki.ros.org/tf/Tutorials/Writing%20a%20tf%20listener%20%28Python%29
        self.lock = threading.Lock()

    def initialize(self):
        print ("Initialize")
        error_code = ErrorCode.OK
        return error_code

    def set_plan(self, plan, local_goal):
        print ("Set Plan")
        try:
            self.lock.acquire()
            if not plan.poses:
                self.tmp_plan.poses.append(local_goal)
            else:
                self.tmp_plan = plan
                self.local_goal = local_goal
        finally:
            self.lock.release()

    def get_plan(self, plan):
        print ("Get Plan")
        try:
            self.lock.acquire()
            self.global_plan = plan
        finally:
            self.lock.release()

    def compute_velocity_commands(self):
        cmd_vel_msg = Twist()
        error_code = ErrorCode.OK
        cmd_vel_msg.linear.x = 0.1
        return cmd_vel_msg, error_code

    def is_goal_reached(self):
        return False

    def update_object_array(self, object_array):
        # Update object
        # Detail see SARosPerceptionKitti/helper/msg
        print ("Update object array!")
