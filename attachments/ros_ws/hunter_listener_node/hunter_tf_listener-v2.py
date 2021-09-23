#!/usr/bin/env python 
import roslib
# roslib.load_manifest('listening_tf')
import rospy
import tf 
import geometry_msgs.msg 
from geometry_msgs.msg import PoseStamped

if __name__=='__main__':
    rospy.init_node('Hunter_tf_amclpose_listener')
    topic_rate = 4
    listener = tf.TransformListener()
    hunter_pos_pub = rospy.Publisher('/current_pose', PoseStamped, queue_size=1)
    rate = rospy.Rate(topic_rate)

    while not rospy.is_shutdown():
        try:
            (pos,rot) = listener.lookupTransform('map', 'base_link', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            # print('Error: Cannot transform \'base_link\' to \'map\'.')
            continue

        crnt_pos = geometry_msgs.msg.PoseStamped()
        crnt_pos.pose.position.x = pos[0]
        crnt_pos.pose.position.y = pos[1]
        crnt_pos.pose.position.z = pos[2]
        crnt_pos.pose.orientation.x = rot[0]
        crnt_pos.pose.orientation.y = rot[1]
        crnt_pos.pose.orientation.z = rot[2]
        crnt_pos.pose.orientation.w = rot[3]

        hunter_pos_pub.publish(crnt_pos)
        rate.sleep()

