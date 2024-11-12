#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# Initialize ROS node
rospy.init_node('image_saver', anonymous=True)

# Initialize OpenCV bridge
bridge = CvBridge()

# Callback function to handle image messages
def image_callback(msg):
    try:
        # Convert ROS image message to OpenCV image
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            #cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            #normalized_depth_image = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # Save the image
        cv2.imwrite('goalrgb_new.jpg', cv_image)
            #cv2.imwrite('goalrgb.jpg', normalized_depth_image)
        rospy.loginfo("Image saved!")
    except Exception as e:
        rospy.logerr("Error processing image: %s", str(e))

# Subscribe to the image topic
rospy.Subscriber("/camera/color/image_raw", Image, image_callback)
    #rospy.Subscriber("/camera/depth/image_raw", Image, image_callback)

# Spin ROS
rospy.spin()
