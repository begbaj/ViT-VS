#!/usr/bin/env python

import rospy
import tf
import math
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState


class GazeboBroadcaster:

    def __init__(self):

        self.cmd_vel_sub = rospy.Subscriber('/camera_vel', Twist, self.velCallback, queue_size=1)
        self.robot_set_state = ModelState()
        self.robot_set_state.model_name = 'realsense2_camera'
        self.robot_set_state.reference_frame = 'base_link'

        self.listener = tf.TransformListener()


    def cam_to_base(self):
        cam_to_base_trans, cam_to_base_rot = tf
        try:
            cam_to_base_trans, cam_to_base_rot = listener.lookupTransform('/camera_depth_optical_frame', '/base_link', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.loginfo('tf tree not available')


    def velCallback(self, data):
        # get the robot state
        try:
            self.model_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            self.robot_state = self.model_state_service('realsense2_camera','base_link')
            self.robot_set_state.pose = self.robot_state.pose
        except rospy.ServiceException:
            rospy.loginfo('GetModelState service failed')

        self.robot_set_state.twist = data
        rospy.loginfo(self.robot_set_state)

        try:
            self.model_state_set_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            result = self.model_state_set_service(self.robot_set_state)
        except rospy.ServiceException:
            rospy.loginfo('desired robot state was not set')
            
        try:
            real_state = self.model_state_service('realsense2_camera', '')
            rospy.loginfo(real_state)
        except rospy.ServiceException:
            rospy.loginfo('GetModelState service failed')   


def main():
    rospy.init_node('frame_controller')
    rate = rospy.Rate(50)  # 50 Hz

    gazebo_broadcaster = GazeboBroadcaster()

    # Loop until the node is stopped
    while not rospy.is_shutdown():
        rate.sleep()

if __name__ == '__main__':
    main()
