#!/usr/bin/env python 
import roslib
import rospy
import tf 
import geometry_msgs.msg
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovariance
import thread
import threading
import tf2_ros
import geometry_msgs.msg
import numpy
import tf_conversions
from helper.msg._ObjectArray import ObjectArray
from helper.msg._Object import Object

from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseArray
import math
import time

from tf.transformations import euler_from_quaternion, quaternion_from_euler

target_link_pose = geometry_msgs.msg.Pose()
target_model_pose = Odometry()
map2base_link_ts = geometry_msgs.msg.TransformStamped()
def xyz_to_mat44(pos):
            return tf.transformations.translation_matrix((pos.x, pos.y, pos.z))

def xyzw_to_mat44(ori):
    return tf.transformations.quaternion_matrix((ori.x, ori.y, ori.z, ori.w))
def TransPose(ps):
    global map2base_link_ts

    #parent: map, child:base_link
    translation = [map2base_link_ts.transform.translation.x, map2base_link_ts.transform.translation.y, map2base_link_ts.transform.translation.z]
    rotation = [map2base_link_ts.transform.rotation.x, map2base_link_ts.transform.rotation.y, map2base_link_ts.transform.rotation.z, map2base_link_ts.transform.rotation.w]
    mat44 = numpy.dot(tf.transformations.translation_matrix(translation), tf.transformations.quaternion_matrix(rotation))

    # pose44 is the given pose as a 4x4
    pose44 = numpy.dot(xyz_to_mat44(ps.pose.position), xyzw_to_mat44(ps.pose.orientation))

    # txpose is the new pose in target_frame as a 4x4
    txpose = numpy.dot(mat44, pose44)

    # xyz and quat are txpose's position and orientation
    xyz = tuple(tf.transformations.translation_from_matrix(txpose))[:3]
    quat = tuple(tf.transformations.quaternion_from_matrix(txpose))
    return xyz, quat
    
def TransformListenerFunc( threadName, delay):
    listener = tf.TransformListener()
    topic_rate = 50.0
    hunter_pos_pub = rospy.Publisher('/current_pose', PoseStamped, queue_size=1)
    rate = rospy.Rate(topic_rate)
    while not rospy.is_shutdown():
        try:
            #get transfrom , parent:base_link,child:odom
            #return [t.x, t.y, t.z], [r.x, r.y, r.z, r.w]
            (pos, rot) = listener.lookupTransform('base_link', 'odom', rospy.Time(0))
            #  ps: the geometry_msgs.msg.PoseStamped message
            odom_pose = geometry_msgs.msg.PoseStamped()
            odom_pose.header.frame_id = 'base_link'
            odom_pose.pose.position.x = pos[0]
            odom_pose.pose.position.y = pos[1]
            odom_pose.pose.position.z = pos[2]

            odom_pose.pose.orientation.x = rot[0]
            odom_pose.pose.orientation.y = rot[1]
            odom_pose.pose.orientation.z = rot[2]
            odom_pose.pose.orientation.w = rot[3]
            (xyz, xyzw) = TransPose(odom_pose)

            trans_broadcaster = tf2_ros.TransformBroadcaster()
            t = geometry_msgs.msg.TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "map"
            t.child_frame_id = 'odom'
            # tf.TransformerROS.transformPose("map",)
            t.transform.translation.x = xyz[0]
            t.transform.translation.y = xyz[1]
            t.transform.translation.z = xyz[2]
            # t.transform.translation.x = map2base_link_ts.transform.translation.x
            # t.transform.translation.y = map2base_link_ts.transform.translation.y
            # t.transform.translation.z = map2base_link_ts.transform.translation.z

            t.transform.rotation.x = xyzw[0]
            t.transform.rotation.y = xyzw[1]
            t.transform.rotation.z = xyzw[2]
            t.transform.rotation.w = xyzw[3]
            trans_broadcaster.sendTransform(t)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            # print('Error: Cannot transform \'base_link\' to \'map\'.')
            continue
        crnt_pos = geometry_msgs.msg.PoseStamped()
        crnt_pos.header.frame_id = 'map'
        crnt_pos.header.stamp = rospy.Time.now()
        crnt_pos.pose.position.x = map2base_link_ts.transform.translation.x
        crnt_pos.pose.position.y = map2base_link_ts.transform.translation.y
        crnt_pos.pose.position.z = map2base_link_ts.transform.translation.z
        crnt_pos.pose.orientation.x = map2base_link_ts.transform.rotation.x
        crnt_pos.pose.orientation.y = map2base_link_ts.transform.rotation.y
        crnt_pos.pose.orientation.z = map2base_link_ts.transform.rotation.z
        crnt_pos.pose.orientation.w = map2base_link_ts.transform.rotation.w

        hunter_pos_pub.publish(crnt_pos)
        rate.sleep()

def ModelMsgCb(ms_msg):
    global target_model_pose
    global map2base_link_ts
    target_model_pose = ms_msg

    trans_broadcaster = tf2_ros.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "map"
    t.child_frame_id = 'base_link_true'
    local_pose = target_model_pose.pose.pose

    # tf.TransformerROS.transformPose("map",)
    t.transform.translation.x = local_pose.position.x
    t.transform.translation.y = local_pose.position.y
    t.transform.translation.z = local_pose.position.z

    t.transform.rotation.x = local_pose.orientation.x
    t.transform.rotation.y = local_pose.orientation.y
    t.transform.rotation.z = local_pose.orientation.z
    t.transform.rotation.w = local_pose.orientation.w
    map2base_link_ts = t
    trans_broadcaster.sendTransform(t)

def DealObstacleLink(msg):
    return 0

agent_name_list = []
def DealObstacleModel(msg):
    local_msg = ModelStates()
    local_msg = msg

    target_id_list = []
    target_index_list = []
    target_model_pose_list = []
    target_model_twist_list = []
    target_index = 0
    for item in local_msg.name:
        for index in range(1, 13):
            if (item == "agent" + str(index)):
                target_id_list.append(index)
                target_index_list.append(target_index)
                target_model_pose_list.append(local_msg.pose[target_index])
                target_model_twist_list.append(local_msg.twist[target_index])
                break
        target_index += 1
        # print(item)

    b_time = time.time()
    obs_array = ObjectArray()
    obs_array.header.frame_id = 'world'
    obs_array.header.stamp = rospy.Time.now()
    obs_array.header.seq = 1
    for index in range(0, 12):
        ob = Object()
        #world pose
        pose = geometry_msgs.msg.Pose()
        twist = geometry_msgs.msg.Twist()
        pose = target_model_pose_list[index]
        twist = target_model_twist_list[index]
        ob.world_pose.header.stamp = rospy.Time.now()
        ob.world_pose.header.frame_id = 'world'
        
        ob.world_pose.point.x = pose.position.x
        ob.world_pose.point.y = pose.position.y
        ob.world_pose.point.z = pose.position.z

        (roll, pitch, yaw) = euler_from_quaternion([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        # print("yaw: ", yaw)
        ob.heading = yaw
        ob.velocity = numpy.sqrt(pow(twist.linear.x, 2) + pow(twist.linear.y, 2))

        ob.id = target_id_list[index]
        #velo pose
        #cam pose
        #others
        obs_array.list.append(ob)

    #pub marker for visualize
    ob_array = PoseArray()
    marker_array = MarkerArray()
    info_marker_array = MarkerArray()

    ob_array.header.frame_id='world'
    ob_array.header.stamp=rospy.Time.now()
    marker_id = 0
    for index in range(0, 12):
        #pose
        temp_pose = geometry_msgs.msg.Pose()
        temp_marker = Marker()

        pose = geometry_msgs.msg.Pose()
        twist = geometry_msgs.msg.Twist()
        pose = target_model_pose_list[index]
        twist = target_model_twist_list[index]

        temp_pose.position.x = pose.position.x
        temp_pose.position.y = pose.position.y
        angle = math.atan2(twist.linear.y,twist.linear.x)
        q = quaternion_from_euler(0,0,angle)
        temp_pose.orientation.x =q[0]
        temp_pose.orientation.y =q[1]
        temp_pose.orientation.z =q[2]
        temp_pose.orientation.w =q[3]
        ob_array.poses.append(temp_pose)

        #info marker:velocity info array
        temp_info_marker = Marker()
        temp_info_marker.header.stamp = rospy.Time.now()
        temp_info_marker.action = temp_info_marker.ADD
        temp_info_marker.ns = 'basic_shape'
        temp_info_marker.type = temp_info_marker.TEXT_VIEW_FACING
        temp_info_marker.header.frame_id = 'world'
        temp_info_marker.id = marker_id
        # temp_info_marker.pose = temp_pose
        
        temp_info_marker.pose.position.x = temp_pose.position.x
        temp_info_marker.pose.position.y = temp_pose.position.y
        temp_info_marker.pose.position.z = temp_pose.position.z + 1
        # temp_info_marker.pose.position.z += 0.4 * 1.5
        
        temp_info_marker.pose.orientation.x = 0
        temp_info_marker.pose.orientation.y = 0
        temp_info_marker.pose.orientation.z = 0
        temp_info_marker.pose.orientation.w = 1

        # temp_info_marker.scale.x = 2.0
        # temp_info_marker.scale.y = 2.0
        temp_info_marker.scale.z = 0.6

        temp_info_marker.color.a = 1.0
        temp_info_marker.color.r = 1
        temp_info_marker.color.g = 0
        temp_info_marker.color.b = 1

        car_x = map2base_link_ts.transform.translation.x
        car_y = map2base_link_ts.transform.translation.y
        car_z = map2base_link_ts.transform.translation.z
        dis2car = math.sqrt(pow(car_x - temp_pose.position.x, 2) + pow(car_y - temp_pose.position.y, 2))
        dis2car = (float)((int)(100 * dis2car)) / 100.0 - 0.6
        temp_info_marker.text = "dis:" + str(dis2car) + "m."
        info_marker_array.markers.append(temp_info_marker)
        # if dis2car <= 0.0:
        #     temp_info_marker.text = "Collision"
        # else:
        #     temp_info_marker.text = "dis:" + str(dis2car) + "m."
        #marker
        temp_marker.ns = 'basic_shape'
        temp_marker.header.stamp = rospy.Time.now()
        temp_marker.header.frame_id = 'world'
        temp_marker.type = temp_marker.SPHERE
        temp_marker.pose = temp_pose
        temp_marker.action = temp_marker.ADD
        temp_marker.scale.x = 0.4
        temp_marker.scale.y = 0.4
        temp_marker.scale.z = 0.4
        temp_marker.id = marker_id
        marker_id = marker_id + 1
        temp_marker.color.r = 0
        temp_marker.color.g = 1
        temp_marker.color.b = 0
        temp_marker.color.a = 1
        marker_array.markers.append(temp_marker)
    
    ob_pub.publish(ob_array)
    marker_pub.publish(marker_array)
    info_marker_pub.publish(info_marker_array)
    obstacle_pub.publish(obs_array)

if __name__ == '__main__':
    rospy.init_node('Hunter_tf_amclpose_listener')
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    static_transformStamped = geometry_msgs.msg.TransformStamped()
    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = "world"
    static_transformStamped.child_frame_id = "map"

    static_transformStamped.transform.translation.x = 0
    static_transformStamped.transform.translation.y = 0
    static_transformStamped.transform.translation.z = 0

    quat = tf.transformations.quaternion_from_euler(0, 0, 0)
    static_transformStamped.transform.rotation.x = quat[0]
    static_transformStamped.transform.rotation.y = quat[1]
    static_transformStamped.transform.rotation.z = quat[2]
    static_transformStamped.transform.rotation.w = quat[3]

    broadcaster.sendTransform(static_transformStamped)
    
    thread.start_new_thread(TransformListenerFunc, ("TransListener", 2,))
    # linkStates_sub = rospy.Subscriber("/gazebo/link_states", LinkStates, callback=LinkMsgCb, queue_size=2)
    modelStates_sub = rospy.Subscriber("/ground_truth/state", Odometry, callback=ModelMsgCb, queue_size=13)
    # obstacle_link_sub = rospy.Subscriber("/gazebo/link_states", LinkStates, callback=DealObstacleLink, queue_size=2)
    obstacle_model_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, callback=DealObstacleModel, queue_size=13)
    
    obstacle_pub = rospy.Publisher('/tracking/objects', ObjectArray, queue_size=13)
    marker_pub = rospy.Publisher('/ob_marker_array', MarkerArray, queue_size=13)
    info_marker_pub = rospy.Publisher('/info_marker_array', MarkerArray, queue_size=13)
    global ob_pub
    ob_pub = rospy.Publisher('/ob_state_array', PoseArray, queue_size=13)
    
    rospy.spin()
