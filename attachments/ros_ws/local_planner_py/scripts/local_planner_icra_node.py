#! /usr/bin/env python

import time, threading
import roslib
roslib.load_manifest('local_planner_py')
import rospy
import actionlib
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped

# from roborts_msgs.msg import LocalPlannerAction
# from roborts_msgs.msg import LocalPlannerFeedback
# from roborts_msgs.msg import LocalPlannerResult
from helper.msg import ObjectArray

from RLLocalPlanner import  RLLocalPlanner
from node_state import NodeState
from error_code import ErrorCode


class LocalPlannerNode:
    def __init__(self):
        self.server = actionlib.SimpleActionServer('/local_planner_node_action', LocalPlannerAction,
                                                   self.execute_callback, False)
        self.server.start()
        self.node_state = NodeState.IDLE
        self.error_code = ErrorCode.OK
        self.max_error = 5
        self.local_planner = RLLocalPlanner()
        self.frequency = 100
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.cmd_vel_msg = Twist()
        self.local_goal = PoseStamped()
        self.planner_thread = None
        self.lock = threading.Lock()
        self.object_array_sub = rospy.Subscriber('/tracking/objects', ObjectArray, self.object_callback)
        print ("LocalPlannerNode Init!")

    def execute_callback(self, command):
        print ("Receive command!")
        error_code = self.get_error_code()
        node_state = self.get_node_state()
        if node_state == NodeState.FAILURE:
            feedback = LocalPlannerFeedback()
            result = LocalPlannerResult()
            feedback.error_code = error_code
            result.error_code = error_code
            self.server.publish_feedback(feedback)
            self.server.set_aborted(result, "Error!")
            print("Initialization Failed, Failed to execute action!")
            return

        # TODO: Thread lock
        self.local_planner.set_plan(command.route, self.local_goal)

        print ("Send plan!")
        if node_state == NodeState.IDLE:
            self.start_planning()

        while not rospy.is_shutdown():
            # sleep 1ms
            if self.server.is_preempt_requested():
                print("Action Preempted")
                if self.server.is_new_goal_available():
                    self.server.set_preempted()
                    break
                else:
                    self.server.set_preempted()
                    self.stop_planning()
            node_state = self.get_node_state()
            error_code = self.get_error_code()

            if node_state == NodeState.RUNNING or node_state == NodeState.SUCCESS or node_state == NodeState.FAILURE:
                feedback = LocalPlannerFeedback()
                result = LocalPlannerResult()
                if error_code != ErrorCode.OK:
                    feedback.error_code = error_code
                    self.set_error_code(ErrorCode.OK)
                    self.server.publish_feedback(feedback)
                if node_state == NodeState.SUCCESS:
                    result.error_code = error_code
                    self.server.set_succeeded(result, "MSG")
                    self.stop_planning()
                    break
                if node_state == NodeState.FAILURE:
                    result.error_code = error_code
                    self.server.set_aborted(result, "MSG")
                    self.stop_planning()
                    break

    def start_planning(self):
        if self.planner_thread is not None:
            if not self.planner_thread.is_alive():
                self.planner_thread.join()
        self.node_state = NodeState.RUNNING
        self.planner_thread = threading.Thread(target=self.planning_thread, name='PlanningThread')
        self.planner_thread.start()

    def stop_planning(self):
        self.set_node_state(NodeState.IDLE)
        if not self.planner_thread.is_alive():
            self.planner_thread.join()

    def planning_thread(self):
        print("Planning thread start!")
        # # TODO: Initialize local planner here
        # error_code = self.local_planner.initialize()
        # if error_code == ErrorCode.OK:
        #     print ("Local Planner Initialize Success!")
        # else:
        #     print ("Local Planner Initialize Failed!")
        #     self.set_node_state(NodeState.FAILURE)
        #     self.set_error_code(error_code)

        error_count = 0
        while self.get_node_state() == NodeState.RUNNING:
            # TODO: Mutex
            # TODO: Begin time counting
            self.cmd_vel_msg, error_code = self.local_planner.compute_velocity_commands()
            # TODO: End time counting
            cost_time = 0
            need_time = 1000.0 / self.frequency
            sleep_time = need_time - cost_time
            if sleep_time <= 0:
                sleep_time = 0
            if error_code == ErrorCode.OK:
                error_count = 0
                self.cmd_vel_pub.publish(self.cmd_vel_msg)
                if self.local_planner.is_goal_reached():
                    self.set_node_state(NodeState.SUCCESS)
            elif error_count > self.max_error > 0:
                print("Can not finish plan with max retries!")
                self.set_error_code(ErrorCode.LP_MAX_ERROR_FAILURE)
                self.set_node_state(NodeState.FAILURE)
            else:
                error_count = error_count + 1
                print("Can not get cmd_vel for once.")
            self.set_error_code(error_code)
        self.cmd_vel_msg.linear.x = 0
        self.cmd_vel_msg.linear.y = 0
        self.cmd_vel_msg.angular.z = 0
        self.cmd_vel_pub.publish(self.cmd_vel_msg)

    def set_node_state(self, node_state):
        self.lock.acquire()
        try:
            self.node_state = node_state
        finally:
            self.lock.release()

    def set_error_code(self, error_code):
        self.lock.acquire()
        try:
            self.error_code = error_code
        finally:
            self.lock.release()

    def get_node_state(self):
        self.lock.acquire()
        node_state = None
        try:
            node_state = self.node_state
        finally:
            self.lock.release()
        return node_state

    def get_error_code(self):
        self.lock.acquire()
        error_code = None
        try:
            error_code = self.error_code
        finally:
            self.lock.release()
        return error_code

    def object_callback(self, object_array):
        self.local_planner.update_object_array(object_array)
        print ("Receive Object")


if __name__ == '__main__':
    rospy.init_node('local_planner_node')
    local_planner_node = LocalPlannerNode()
    rospy.spin()
