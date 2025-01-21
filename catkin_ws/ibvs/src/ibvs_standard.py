# Core imports
import rospy
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Twist
from gazebo_msgs.srv import GetModelState, SpawnModel, SetModelState, DeleteModel
from sensor_msgs.msg import Image as ImageMsg
from cv_bridge import CvBridge

# Scientific computing
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import stats

# Deep learning & vision
import cv2
from PIL import Image

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

# System
import os
import sys
import yaml
import time
import types
from pathlib import Path
from typing import Union, List, Tuple
import argparse

# ROS transforms
import tf2_ros
import tf_conversions


def visualize_correspondences(image1, image2, points1, points2, save_path=None):
    """Visualize correspondences between two images."""
    if isinstance(image1, Image.Image):
        image1 = np.array(image1)
    if isinstance(image2, Image.Image):
        image2 = np.array(image2)

    points1 = np.array(points1)
    points2 = np.array(points2)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(image1)
    ax2.imshow(image2)

    ax1.axis('off')
    ax2.axis('off')

    colors = plt.cm.rainbow(np.linspace(0, 1, len(points1)))

    for i, ((y1, x1), (y2, x2), color) in enumerate(zip(points1, points2, colors)):
        ax1.plot(x1, y1, 'o', color=color, markersize=8)
        ax1.text(x1 + 5, y1 + 5, str(i), color=color, fontsize=8)

        ax2.plot(x2, y2, 'o', color=color, markersize=8)
        ax2.text(x2 + 5, y2 + 5, str(i), color=color, fontsize=8)

        con = ConnectionPatch(
            xyA=(x1, y1), xyB=(x2, y2),
            coordsA="data", coordsB="data",
            axesA=ax1, axesB=ax2, color=color, alpha=0.5
        )
        fig.add_artist(con)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

class Controller:
    def __init__(self, desired_position, desired_orientation, config_path, method='sift'):
        self.method = method.lower()
        self.config_path = config_path
        # Load parameters from YAML
        self.load_parameters()


        # Processing variables
        self.latest_image = None
        self.latest_image_depth = None
        self.latest_pil_image = None
        self.camera_position = None
        self.desired_position = desired_position
        self.desired_orientation = desired_orientation
        self.orientation_quaternion = None
        self.s_uv = None
        self.s_uv_star = None
        self.v_c = None
        self.iteration_count = 0

       # Histories and tracking variables
        self.velocity_history = []
        self.position_history = []
        self.orientation_history = []
        self.iteration_history = 0
        self.initial_position_error = None
        self.velocity_mean_100 = []
        self.velocity_mean_10 = []
        self.average_velocities = []
        self.velocity_vector_history = []

        # Velocity tracking
        self.applied_velocity_x = []
        self.applied_velocity_y = []
        self.applied_velocity_z = []
        self.applied_velocity_roll = []
        self.applied_velocity_pitch = []
        self.applied_velocity_yaw = []


        # Initialize EMA for velocity smoothing
        self.initialize_ema()

        # Initialize ROS
        if not rospy.get_node_uri():
            rospy.init_node('ibvs_controller', anonymous=True)

        # Initialize ROS communication
        self.bridge = CvBridge()
        self.setup_ros_communication()

        # Load goal image
        self.goal_image = self.load_goal_image(self.image_path)
        self.goal_image_array = np.array(self.goal_image)

        # Wait for first image
        rospy.loginfo("Waiting for the first image...")
        rospy.wait_for_message("/camera/color/image_raw", ImageMsg, timeout=10.0)
        rospy.loginfo("First image received!")

    def load_parameters(self):
        """Load parameters from a YAML configuration file."""
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Camera and image parameters
        self.u_max = config['u_max']  # Image width
        self.v_max = config['v_max']  # Image height
        self.f_x = config['f_x']  # Focal length x
        self.f_y = config['f_y']  # Focal length y
        self.c_x = self.u_max / 2  # Principal point x ONLY VALID FOR SIMULATION
        self.c_y = self.v_max / 2  # Principal point y ONLY VALID FOR SIMULATION

        # Control parameters
        self.lambda_ = config['lambda_']  # Control gain
        self.max_velocity = config.get('max_velocity', 1.0)
        self.min_error = config['min_error']
        self.max_error = config['max_error']
        self.num_pairs = config['num_pairs']  # Number of feature pairs to track

        # Sampling parameters
        self.num_samples = config['num_samples']
        self.num_circles = config['num_circles']
        self.circle_radius_aug = config['circle_radius_aug']

        # Convergence parameters
        self.velocity_convergence_threshold = config['velocity_convergence_threshold']
        self.velocity_threshold_translation = config['velocity_threshold_translation']
        self.velocity_threshold_rotation = config['velocity_threshold_rotation']
        self.error_threshold_ratio = config['error_threshold_ratio']
        self.error_threshold_absolute_translation = config['error_threshold_absolute_translation']
        self.error_threshold_absolute_rotation = config['error_threshold_absolute_rotation']

        # Iteration control
        self.min_iterations = config['min_iterations']
        self.max_iterations = config['max_iterations']
        self.max_velocity_vector_history = config.get('max_velocity_vector_history', 200)

        # this is processing resolution
        self.dino_input_size = config['dino_input_size']

        # EMA parameter
        self.ema_alpha = config.get('ema_alpha', 0.1)

        # Set the image path
        current_directory = os.path.dirname(__file__)
        self.image_path = os.path.join(current_directory, config['image_path'])

    def initialize_ema(self):
        """Initialize EMA for each velocity component."""
        self.ema_velocities = [None] * 6

    def update_ema(self, index, new_value):
        """Update EMA for a single velocity component."""
        if self.ema_velocities[index] is None:
            self.ema_velocities[index] = new_value
        else:
            self.ema_velocities[index] = self.ema_alpha * new_value + (1 - self.ema_alpha) * self.ema_velocities[index]
        return self.ema_velocities[index]

    def is_visual_servoing_done(self):
        """Check if visual servoing should stop based on error and velocity criteria"""
        if self.iteration_count < 300:  # Wait at least 300 iterations
            return False, False

        # Get current errors
        current_error_translation, current_error_rotation = self.calculate_end_error(self.desired_orientation)

        # Initialize initial errors if not already done
        if not hasattr(self, 'initial_error_translation'):
            self.initial_error_translation, self.initial_error_rotation = self.calculate_end_error(
                self.desired_orientation)

        # Check if current error is more than five times the initial error (divergence check)
        if current_error_translation > 5 * self.initial_error_translation:
            rospy.logerr("Aborting sample due to position error exceeding five times the initial error.")
            return True, False  # Done but not converged

        # Error-based convergence checks
        error_reduced_90_percent = ((current_error_translation / self.initial_error_translation) < 0.1 and
                                    (current_error_rotation / self.initial_error_rotation) < 0.1)

        error_below_absolute = (current_error_translation < 0.01 and  # 1cm
                                current_error_rotation < 1.0)  # 1 degree

        error_converged = error_reduced_90_percent or error_below_absolute

        # Velocity-based stopping check with sliding windows
        if len(self.velocity_vector_history) >= 200:  # Need 200 samples for two windows
            recent_velocities = np.array(self.velocity_vector_history[-200:])

            # Split into two 100-sample windows
            first_window = recent_velocities[:100]
            second_window = recent_velocities[100:]

            # Calculate means for both windows
            first_trans_velocities = np.linalg.norm(first_window[:, :3] * 1000.0, axis=1)  # mm/s
            first_rot_velocities = np.linalg.norm(np.degrees(first_window[:, 3:]), axis=1)  # deg/s

            second_trans_velocities = np.linalg.norm(second_window[:, :3] * 1000.0, axis=1)
            second_rot_velocities = np.linalg.norm(np.degrees(second_window[:, 3:]), axis=1)

            first_trans_mean = np.mean(first_trans_velocities)
            first_rot_mean = np.mean(first_rot_velocities)
            second_trans_mean = np.mean(second_trans_velocities)
            second_rot_mean = np.mean(second_rot_velocities)

            # Print velocity information every 50 iterations
            if self.iteration_count % 50 == 0:
                rospy.loginfo(f"\nVelocity Analysis (last 200 iterations):")
                rospy.loginfo(f"First window translation mean (mm/s): {first_trans_mean:.6f}")
                rospy.loginfo(f"Second window translation mean (mm/s): {second_trans_mean:.6f}")
                rospy.loginfo(f"First window rotation mean (deg/s): {first_rot_mean:.6f}")
                rospy.loginfo(f"Second window rotation mean (deg/s): {second_rot_mean:.6f}")

            # Check convergence conditions
            if first_trans_mean < 1.0 and first_rot_mean < 0.1:
                if second_trans_mean > first_trans_mean and second_rot_mean > first_rot_mean:
                    rospy.loginfo("Velocity trend indicates convergence - checking final error")
                    return True, error_reduced_90_percent

        # Print current errors periodically
        if self.iteration_count % 50 == 0:
            rospy.loginfo(f"Current translation error: {current_error_translation:.4f} cm")
            rospy.loginfo(f"Current rotation error: {current_error_rotation:.4f} degrees")

        # Check if maximum iterations reached
        if self.iteration_count >= self.max_iterations:
            rospy.loginfo("Maximum iterations reached")
            if error_reduced_90_percent:
                rospy.logwarn("Maximum iterations reached with 90% error reduction. Marking as converged.")
                return True, True
            else:
                rospy.logwarn("Maximum iterations reached without sufficient error reduction.")
                return True, False

        return False, False

    def setup_ros_communication(self):
        """Setup ROS publishers and subscribers."""
        # Subscribers
        self.image_sub_rgb = rospy.Subscriber("/camera/color/image_raw",
                                              ImageMsg,
                                              self.image_callback_rgb,
                                              queue_size=10)
        self.image_sub_depth = rospy.Subscriber("/camera/depth/image_raw",
                                                ImageMsg,
                                                self.image_callback_depth,
                                                queue_size=10)

        # Publishers
        self.pub = rospy.Publisher('/camera_vel', Twist, queue_size=10)
        self.image_pub = rospy.Publisher('/camera/image_processed', ImageMsg, queue_size=10)
        self.goal_image_pub = rospy.Publisher('/goal_image_processed', ImageMsg, queue_size=10)
        self.current_image_pub = rospy.Publisher('/current_image_processed', ImageMsg, queue_size=10)
        self.correspondence_pub = rospy.Publisher('/correspondence_visualization', ImageMsg, queue_size=10)

        rospy.loginfo("ROS publishers and subscribers initialized")

    def load_goal_image(self, image_path):
        """Load the goal image from the specified path."""
        try:
            goal_image = Image.open(image_path)
            goal_image = goal_image.convert('RGB')  # Ensure the image is in RGB format
            rospy.loginfo(".... got goal image")
            return goal_image
        except Exception as e:
            rospy.logerr(f"Failed to load image at {image_path}: {e}")
            sys.exit(1)

    def image_callback_rgb(self, msg):
        """Callback for RGB image messages."""
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.latest_pil_image = Image.fromarray(cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2RGB))

    def image_callback_depth(self, msg):
        """Callback for depth image messages."""
        self.latest_image_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def detect_features(self):
        """Detect features using the specified method (SIFT, ORB, or AKAZE)."""
        if self.latest_image is None:
            rospy.logwarn("No latest image available")
            return None, None

        try:
            # Use original resolution images
            goal_image_processed = np.array(self.goal_image)
            current_image_processed = np.array(self.latest_pil_image)

            # Convert images to grayscale
            goal_gray = cv2.cvtColor(goal_image_processed, cv2.COLOR_RGB2GRAY)
            current_gray = cv2.cvtColor(current_image_processed, cv2.COLOR_RGB2GRAY)

            # Initialize detector based on method
            if self.method == 'sift':
                detector = cv2.SIFT_create()
                norm_type = cv2.NORM_L2
            elif self.method == 'orb':
                detector = cv2.ORB_create(nfeatures=1000)
                norm_type = cv2.NORM_HAMMING
            elif self.method == 'akaze':
                detector = cv2.AKAZE_create()
                norm_type = cv2.NORM_HAMMING
            else:
                rospy.logerr(f"Unknown feature detection method: {self.method}")
                return None, None

            # Detect and compute keypoints and descriptors
            kp1, des1 = detector.detectAndCompute(goal_gray, None)
            kp2, des2 = detector.detectAndCompute(current_gray, None)

            if des1 is None or des2 is None:
                rospy.logwarn("No descriptors found in one or both images")
                return None, None

            #rospy.loginfo(
            #    f"{self.method.upper()} found {len(kp1)} keypoints in goal image and {len(kp2)} keypoints in current image")

            # Initialize BF matcher with appropriate norm type
            bf = cv2.BFMatcher(norm_type, crossCheck=True)
            matches = bf.match(des1, des2)

            if len(matches) < 4:  # Minimum required for visual servoing
                rospy.logwarn(f"Insufficient matches found: {len(matches)} < 4")
                return None, None

            # Get all distances
            distances = np.array([m.distance for m in matches])

            # Normalize and invert distances to [0,1] range where 1 is best match
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            normalized_distances = 1 - (distances - min_dist) / (max_dist - min_dist)

            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)

            # Determine number of pairs to use
            available_matches = len(matches)
            num_pairs_to_use = min(self.num_pairs, available_matches)

            if num_pairs_to_use < 4:
                rospy.logwarn("Not enough good matches for visual servoing")
                return None, None

            #if num_pairs_to_use < self.num_pairs:
            #    rospy.logwarn(f"Using reduced number of features: {num_pairs_to_use} (requested: {self.num_pairs})")

            # Select top matches
            matches = matches[:num_pairs_to_use]

            try:
                points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
                points2 = np.float32([kp2[m.trainIdx].pt for m in matches])

                # Visualize correspondences
                self.visualize_correspondences_with_lines(
                    goal_image_processed,
                    current_image_processed,
                    points1,
                    points2
                )

                # Create arrays directly from the detected points
                s_uv_star = np.round(points1).astype(int)
                s_uv = np.round(points2).astype(int)
                return (s_uv_star, s_uv), None  # Return tuple with None as second element to match expected format

            except Exception as e:
                rospy.logerr(f"Error extracting point locations: {str(e)}")
                return None, None

        except Exception as e:
            rospy.logerr(f"Error in detect_features: {str(e)}")
            return None, None

    def publish_figure(self, fig, publisher):
        """Convert a matplotlib figure to a ROS Image message and publish it."""
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
        ros_image_msg = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")
        publisher.publish(ros_image_msg)

        # Close the figure to free up memory
        plt.close(fig)

    def get_depth(self, current_points):
        """Get depth values for the current feature points."""
        if self.latest_image_depth is None:
            rospy.logwarn("No depth image received yet")
            return None

        z_values_meter = np.zeros((len(current_points), 1))
        height, width = self.latest_image_depth.shape

        for count, point in enumerate(current_points):
            x, y = int(point[0]), int(point[1])

            # Ensure the point is within the image bounds
            if 0 <= x < width and 0 <= y < height:
                depth_value = self.latest_image_depth[y, x]
                # Convert depth value to meters
                z_values_meter[count] = depth_value / 1000.0 if depth_value != 0 else 100
            else:
                z_values_meter[count] = 100  # Out of bounds point

        return z_values_meter

    def ibvs(self):
        if self.latest_image is None:
            print("No latest image available")
            return False

        start_time = time.time()

        # Check current error
        current_error_translation, _ = self.calculate_end_error(self.desired_orientation)
        if hasattr(self, 'initial_error_translation'):
            if current_error_translation > 5 * self.initial_error_translation:
                rospy.logerr(
                    f"Current error ({current_error_translation:.2f} cm) exceeds five times the initial error ({self.initial_error_translation:.2f} cm). Aborting sample.")
                return False
            
        try:
            (s_uv_star, s_uv), _ = self.detect_features()
            if s_uv_star is None or s_uv is None:
                if hasattr(self, 'v_c') and self.v_c is not None:
                    rospy.logwarn("Feature detection failed - continuing with last known velocities")
                    end_time = time.time()
                    print(
                        f"IBVS iteration executed in: {end_time - start_time:.2f} seconds ||| iteration count: {self.iteration_count}")
                    return True  # Continue with last known velocities
                else:
                    print("Feature detection failed and no previous velocities available")
                    return False

            self.draw_points(np.array(self.latest_pil_image), s_uv, s_uv_star)

            # Transform feature points to real-world coordinates
            s_xy, s_star_xy = self.transform_to_real_world(s_uv, s_uv_star)

            # Calculate error and interaction matrix
            e = s_xy - s_star_xy
            e = e.reshape((len(s_xy) * 2, 1))

            Z = self.get_depth(s_uv)
            if Z is None:
                if hasattr(self, 'v_c') and self.v_c is not None:
                    rospy.logwarn("Depth information unavailable - continuing with last known velocities")
                    return True  # Continue with last known velocities
                else:
                    print("Failed to get depth information and no previous velocities available")
                    return False

            L = self.calculate_interaction_matrix(s_xy, Z)
            v_c = -self.lambda_ * np.linalg.pinv(L.astype('float')) @ e

            # Update EMA for each velocity component
            self.v_c = np.array([self.update_ema(i, v) for i, v in enumerate(v_c.flatten())])

            # Store velocity history
            self.velocity_vector_history.append(self.v_c)
            if len(self.velocity_vector_history) > self.max_velocity_vector_history:
                self.velocity_vector_history.pop(0)

            end_time = time.time()
            print(
                f"IBVS iteration executed in: {end_time - start_time:.2f} seconds ||| iteration count: {self.iteration_count}")

            return True

        except Exception as e:
            rospy.logerr(f"Error in IBVS: {str(e)}")
            if hasattr(self, 'v_c') and self.v_c is not None:
                rospy.logwarn("Error in IBVS - continuing with last known velocities")
                return True  # Continue with last known velocities
            return False

    def transform_to_real_world(self, s_uv, s_uv_star):
        """Transform pixel feature points to real-world coordinates."""
        s_xy = []
        s_star_xy = []

        for uv, uv_star in zip(s_uv, s_uv_star):
            x = (uv[0] - self.c_x) / self.f_x
            y = (uv[1] - self.c_y) / self.f_y
            s_xy.append([x, y])

            x_star = (uv_star[0] - self.c_x) / self.f_x
            y_star = (uv_star[1] - self.c_y) / self.f_y
            s_star_xy.append([x_star, y_star])

        return np.array(s_xy), np.array(s_star_xy)

    def calculate_interaction_matrix(self, s_xy, Z):
        """Calculate the interaction matrix for the feature points."""
        L = np.zeros([2 * len(s_xy), 6], dtype=float)

        for count in range(len(s_xy)):
            x, y, z = s_xy[count, 0], s_xy[count, 1], Z[count, 0]
            L[2 * count, :] = [-1 / z, 0, x / z, x * y, -(1 + x ** 2), y]
            L[2 * count + 1, :] = [0, -1 / z, y / z, 1 + y ** 2, -x * y, -x]

        return L

    def publish_twist(self, v_c):
        """Publish velocity commands and store history."""
        twist_msg = Twist()

        # Apply velocity limits
        twist_msg.linear.x = np.clip(v_c[2], -self.max_velocity, self.max_velocity)
        twist_msg.linear.y = np.clip(-v_c[0], -self.max_velocity, self.max_velocity)
        twist_msg.linear.z = np.clip(-v_c[1], -self.max_velocity, self.max_velocity)
        twist_msg.angular.x = np.clip(v_c[5], -self.max_velocity, self.max_velocity)
        twist_msg.angular.y = np.clip(-v_c[3], -self.max_velocity, self.max_velocity)
        twist_msg.angular.z = np.clip(-v_c[4], -self.max_velocity, self.max_velocity)

        # Store velocity history
        self.applied_velocity_x.append(twist_msg.linear.x)
        self.applied_velocity_y.append(twist_msg.linear.y)
        self.applied_velocity_z.append(twist_msg.linear.z)
        self.applied_velocity_roll.append(twist_msg.angular.x)
        self.applied_velocity_pitch.append(twist_msg.angular.y)
        self.applied_velocity_yaw.append(twist_msg.angular.z)

        if np.any(np.abs(v_c) > self.max_velocity):
            rospy.logwarn("Velocity capped due to exceeding maximum allowed value.")

        self.pub.publish(twist_msg)

    def draw_points(self, image, current_points, goal_points):
        """Draw current and goal feature points on the image."""
        for x, y in current_points:
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)  # Current points in green
        for x, y in goal_points:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)  # Goal points in red

        ros_image = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.image_pub.publish(ros_image)

    def run(self):
        """Main visual servoing control loop."""
        rospy.loginfo("Starting visual servoing process...")

        # First, get initial camera pose
        initial_position, initial_orientation = self.get_current_camera_pose()
        if initial_position is None or initial_orientation is None:
            rospy.logerr("Failed to get initial camera pose. Aborting.")
            return self._create_failure_return()

        self.camera_position = initial_position
        self.orientation_quaternion = initial_orientation

        self.iteration_count = 0

        # Initialize histories
        self.position_history = []
        self.orientation_history = []
        self.velocity_history = []
        self.average_velocities = []
        self.velocity_mean_100 = []
        self.velocity_mean_10 = []
        self.velocity_mean_history = []
        self.velocity_vector_history = []

        # Initialize velocity tracking
        self.applied_velocity_x = []
        self.applied_velocity_y = []
        self.applied_velocity_z = []
        self.applied_velocity_roll = []
        self.applied_velocity_pitch = []
        self.applied_velocity_yaw = []

        # Initialize tracking variables
        lowest_position_error = float('inf')
        lowest_orientation_error = float('inf')

        # Calculate initial errors and set targets
        initial_position_error, initial_orientation_error = self.calculate_end_error(self.desired_orientation)
        self.initial_position_error = initial_position_error
        self.initial_error_translation = initial_position_error
        self.initial_error_rotation = initial_orientation_error

        try:
            while not rospy.is_shutdown():
                if self.latest_image is None:
                    rospy.logwarn_throttle(1, "Waiting for image...")
                    continue

                # Perform IBVS
                success = self.ibvs()
                if not success:
                    rospy.logwarn("IBVS iteration failed - returning failure state")
                    return self._create_failure_return()

                self.iteration_count += 1

                # Calculate and store average velocity
                avg_velocity = np.mean(np.abs(self.v_c))
                self.average_velocities.append(avg_velocity)
                self.velocity_history.append(avg_velocity)

                # Update velocity means
                if len(self.velocity_history) >= 100:
                    self.velocity_mean_100.append(np.mean(self.velocity_history[-100:]))
                else:
                    self.velocity_mean_100.append(np.mean(self.velocity_history))

                if len(self.velocity_history) >= 10:
                    self.velocity_mean_10.append(np.mean(self.velocity_history[-10:]))
                else:
                    self.velocity_mean_10.append(np.mean(self.velocity_history))

                # Apply control
                self.publish_twist(self.v_c)

                # Update position and orientation tracking
                self.camera_position, self.orientation_quaternion = self.get_current_camera_pose()
                if self.camera_position is None or self.orientation_quaternion is None:
                    rospy.logerr("Failed to get camera pose during iteration. Aborting.")
                    return self._create_failure_return()

                self.position_history.append(self.camera_position)
                self.orientation_history.append(self.orientation_quaternion)

                # Calculate current errors
                current_position_error, current_orientation_error = self.calculate_end_error(self.desired_orientation)

                # Update the lowest errors
                lowest_position_error = min(lowest_position_error, current_position_error)
                lowest_orientation_error = min(lowest_orientation_error, current_orientation_error)

                # Check if servoing is done
                done, converged = self.is_visual_servoing_done()
                if done:
                    rospy.loginfo(f"Visual servoing completed after {self.iteration_count} iterations.")
                    rospy.loginfo(f"Converged: {converged}")

                    # Calculate final errors
                    final_position_error, final_orientation_error = self.calculate_end_error(self.desired_orientation)

                    # Log final status
                    rospy.loginfo(f"Final Position Error: {final_position_error:.2f} cm")
                    rospy.loginfo(f"Final Orientation Error: {final_orientation_error:.2f} degrees")
                    rospy.loginfo(f"Lowest Position Error: {lowest_position_error:.2f} cm")
                    rospy.loginfo(f"Lowest Orientation Error: {lowest_orientation_error:.2f} degrees")

                    return (self.camera_position, self.orientation_quaternion, converged,
                            final_position_error, final_orientation_error,
                            np.array(self.position_history), np.array(self.orientation_history),
                            self.iteration_count,
                            lowest_position_error, lowest_orientation_error,
                            np.array(self.average_velocities),
                            np.array(self.velocity_mean_100),
                            np.array(self.velocity_mean_10),
                            np.array(self.applied_velocity_x),
                            np.array(self.applied_velocity_y),
                            np.array(self.applied_velocity_z),
                            np.array(self.applied_velocity_roll),
                            np.array(self.applied_velocity_pitch),
                            np.array(self.applied_velocity_yaw))

            # If interrupted, send zero velocity
            self.publish_twist(np.zeros(6))
            return self._create_failure_return()

        except Exception as e:
            rospy.logerr(f"Exception in run loop: {str(e)}")
            return self._create_failure_return()

    def _create_failure_return(self):
        """Helper method to create a properly formatted failure return tuple."""
        return (
            np.full_like(self.desired_position, np.nan),  # final_position
            np.full_like(self.desired_orientation, np.nan),  # final_quaternion
            False,  # converged
            np.nan,  # position_error
            np.nan,  # orientation_error
            np.array([]),  # position_history
            np.array([]),  # orientation_history
            0,  # iteration_history
            np.nan,  # lowest_position_error
            np.nan,  # lowest_orientation_error
            np.array([]),  # average_velocities
            np.array([]),  # velocity_mean_100
            np.array([]),  # velocity_mean_10
            np.array([]),  # applied_velocity_x
            np.array([]),  # applied_velocity_y
            np.array([]),  # applied_velocity_z
            np.array([]),  # applied_velocity_roll
            np.array([]),  # applied_velocity_pitch
            np.array([])  # applied_velocity_yaw
        )

    def calculate_end_error(self, desired_orientation):
        """Calculate position and orientation errors."""
        # Calculate position error in centimeters
        position_error = np.linalg.norm(self.camera_position - self.desired_position) * 100

        # Calculate orientation error in degrees
        current_orientation = R.from_quat(self.orientation_quaternion)
        desired_orientation = R.from_quat(desired_orientation)
        orientation_error = (current_orientation.inv() * desired_orientation).magnitude() * (180 / np.pi)

        return position_error, orientation_error

    def visualize_correspondences_with_lines(self, goal_image, current_image, points1, points2):
        """Create and publish visualization of correspondences between two images."""
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # Display images
        ax1.imshow(goal_image)
        ax2.imshow(current_image)

        # Force the aspect ratio and limits
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')

        # Set the exact limits for both axes
        ax1.set_xlim([0, 640])
        ax1.set_ylim([480, 0])  # Inverted because origin is at top-left
        ax2.set_xlim([0, 640])
        ax2.set_ylim([480, 0])  # Inverted because origin is at top-left

        # Plot correspondences with rainbow colors
        colors = plt.cm.rainbow(np.linspace(0, 1, len(points1)))
        for i, ((x1, y1), (x2, y2), color) in enumerate(zip(points1, points2, colors)):
            # Only plot points that are within bounds
            if 0 <= x1 <= 640 and 0 <= y1 <= 480 and 0 <= x2 <= 640 and 0 <= y2 <= 480:
                # Plot points and labels
                ax1.plot(x1, y1, 'o', color=color, markersize=8)
                ax1.text(min(x1 + 5, 635), min(y1 + 5, 475), str(i), color=color, fontsize=8)

                ax2.plot(x2, y2, 'o', color=color, markersize=8)
                ax2.text(min(x2 + 5, 635), min(y2 + 5, 475), str(i), color=color, fontsize=8)

                # Draw correspondence lines
                con = ConnectionPatch(
                    xyA=(x1, y1), xyB=(x2, y2),
                    coordsA="data", coordsB="data",
                    axesA=ax1, axesB=ax2, color=color, alpha=0.5
                )
                fig.add_artist(con)

        ax1.set_title("Goal Image")
        ax2.set_title("Current Image")
        plt.tight_layout()

        # Convert figure to ROS message and publish
        fig.canvas.draw()
        img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        ros_image = self.bridge.cv2_to_imgmsg(img_data, encoding="rgb8")

        # Publish all visualizations
        self.correspondence_pub.publish(ros_image)
        self.goal_image_pub.publish(self.bridge.cv2_to_imgmsg(np.array(goal_image), encoding="rgb8"))
        self.current_image_pub.publish(self.bridge.cv2_to_imgmsg(np.array(current_image), encoding="rgb8"))

        plt.close(fig)

    def get_current_camera_pose(self):
        """Get current camera pose from Gazebo."""
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            model_state = get_model_state('realsense2_camera', '')

            position = np.array([
                model_state.pose.position.x,
                model_state.pose.position.y,
                model_state.pose.position.z
            ])

            orientation = np.array([
                model_state.pose.orientation.x,
                model_state.pose.orientation.y,
                model_state.pose.orientation.z,
                model_state.pose.orientation.w
            ])

            return position, orientation
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to get camera pose: {e}")
            return None, None


def sample_camera_positions(volume_dimensions, num_samples, desired_position):
    """
    Sample random camera positions within a specified volume.

    Args:
        volume_dimensions (np.ndarray): The dimensions of the volume for sampling (width, height, depth).
        num_samples (int): The number of samples to generate.
        desired_position (np.ndarray): The desired central position to offset the samples from.

    Returns:
        np.ndarray: An array of sampled camera positions.
    """
    # Offset the volume to be centered around the desired position
    half_dims = volume_dimensions / 2
    min_bounds = desired_position - half_dims
    max_bounds = desired_position + half_dims

    # Sample positions uniformly within the defined bounds
    positions = np.random.uniform(min_bounds, max_bounds, size=(num_samples, 3))
    return positions


def sample_focal_points_original(num_samples, reference_point, num_circles, circle_radius_aug):
    """
    Sample focal points based on the original implementation in PoseLookingAtSamePointWithNoiseAndRotationZGenerator.

    Args:
        num_samples (int): The total number of samples to generate.
        reference_point (np.ndarray): The reference point (3D vector: [x, y, z]).
        num_circles (int): Number of circles to generate points on.
        circle_radius_aug (float): Radius augmentation factor for circles.

    Returns:
        np.ndarray: An array of sampled focal points (shape: [num_samples, 3]).
    """
    samples_per_circle = num_samples // num_circles
    looked_at_points = np.empty((num_samples, 3))

    for cn in range(num_circles):
        radius = circle_radius_aug * (cn + 1)
        istart = cn * samples_per_circle

        # Sample points on a circle
        rand_theta = np.random.uniform(-np.pi, np.pi, size=samples_per_circle)
        x = np.cos(rand_theta) * radius + reference_point[0]
        y = np.sin(rand_theta) * radius + reference_point[1]
        z = np.repeat(reference_point[2], samples_per_circle)

        points = np.column_stack((x, y, z))
        looked_at_points[istart: istart + samples_per_circle] = points

    return looked_at_points


def calculate_position_error(positions, desired_position):
    """
    Calculate the position error as the Euclidean distance from the desired position.

    Args:
        positions (np.ndarray): The sampled camera positions.
        desired_position (np.ndarray): The desired central position.

    Returns:
        tuple: The average error and standard deviation of the error in centimeters.
    """
    errors = np.linalg.norm(positions - desired_position, axis=1)
    average_error = np.mean(errors) * 100  # Convert to centimeters
    std_deviation = np.std(errors) * 100  # Convert to centimeters
    return average_error, std_deviation


def detect_trend(data, window_size=100, consecutive_increases=5):
    """
    Detect if there's an increasing trend in the data.

    Args:
    data (array): Input data
    window_size (int): Size of the sliding window for linear regression
    consecutive_increases (int): Number of consecutive positive slopes required to confirm trend

    Returns:
    tuple: (bool indicating if trend is increasing, index where trend starts)
    """
    slopes = []
    for i in range(len(data) - window_size):
        y = data[i:i + window_size]
        x = np.arange(window_size)
        slope, _, _, _, _ = stats.linregress(x, y)
        slopes.append(slope)

    increasing_count = 0
    for i, slope in enumerate(slopes):
        if slope > 0:
            increasing_count += 1
            if increasing_count >= consecutive_increases:
                return True, i + window_size - consecutive_increases
        else:
            increasing_count = 0

    return False, -1


def calculate_orientation_error(quaternion_list, desired_orientation):
    # Ensure quaternion_list is a numpy array
    quaternion_list = np.array(quaternion_list)  # is x y z w

    # Convert desired_orientation to a Rotation object
    desired_rotation = R.from_quat(desired_orientation)

    errors = []

    for quaternion in quaternion_list:
        # Convert the current quaternion to a Rotation object
        current_rotation = R.from_quat(quaternion)

        # Calculate the relative rotation from current to desired
        relative_rotation = current_rotation.inv() * desired_rotation

        # Get the angle of rotation (in radians)
        angle = relative_rotation.magnitude()

        # Convert to degrees
        error_degrees = np.degrees(angle)

        errors.append(error_degrees)

    errors = np.array(errors)

    # Calculate mean and standard deviation of the errors
    mean_error = np.mean(errors)
    std_dev_error = np.std(errors)

    return mean_error, std_dev_error


def place_red_box_at_focal_point(x, y, z=0.01, counter=1):
    """
    Place a red box model at the given focal point in Gazebo.

    Args:
        x (float): The x-coordinate of the focal point.
        y (float): The y-coordinate of the focal point.
        z (float): The z-coordinate (height above the poster).
        counter (int): A counter to create unique model names.
    """
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

        # Load the model XML from the file
        model_xml = open("/root/catkin_ws/src/ibvs/models/red_box/model.sdf", 'r').read()

        initial_pose = Pose()
        initial_pose.position.x = x
        initial_pose.position.y = y
        initial_pose.position.z = z

        # Create a unique model name using the counter
        model_name = f"red_box_{counter}"

        # Spawn the model in Gazebo
        spawn_model(model_name, model_xml, "", initial_pose, "world")
        print(f"Spawned {model_name} at position ({x}, {y}, {z})")
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")


def set_camera_pose(camera_position, orientation_quaternion):
    """
    Set the camera's pose in Gazebo with the given position and orientation.

    Args:
        camera_position (np.ndarray): The position of the camera.
        orientation_quaternion (np.ndarray): The orientation quaternion for the camera.
    """
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # Create a new state for the camera
        state = ModelState()
        state.model_name = 'realsense2_camera'
        state.pose.position.x = camera_position[0]
        state.pose.position.y = camera_position[1]
        state.pose.position.z = camera_position[2]
        state.pose.orientation.x = orientation_quaternion[0]
        state.pose.orientation.y = orientation_quaternion[1]
        state.pose.orientation.z = orientation_quaternion[2]
        state.pose.orientation.w = orientation_quaternion[3]
        state.reference_frame = 'world'

        # Set the new state in Gazebo
        set_state(state)
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")


def rotate_camera_x_axis(orientation_quaternion, angle_degrees):
    """
    Rotate the camera around its X-axis by the specified angle.

    Args:
        orientation_quaternion (np.ndarray): The original orientation quaternion.
        angle_degrees (float): The rotation angle in degrees.

    Returns:
        np.ndarray: The new orientation quaternion after rotation.
    """
    # Convert the original quaternion to a rotation object
    original_rotation = R.from_quat(orientation_quaternion)

    # Create a rotation around the X-axis
    x_rotation = R.from_euler('x', angle_degrees, degrees=True)

    # Combine the rotations
    new_rotation = original_rotation * x_rotation

    # Convert back to quaternion
    new_quaternion = new_rotation.as_quat()

    return new_quaternion


def find_and_set_best_pose(controller, camera_position, initial_quaternion):
    """
    Find the best pose by testing four different orientations and set the camera to that pose.

    Args:
        controller (Controller): The controller object.
        camera_position (np.ndarray): The initial camera position.
        initial_quaternion (np.ndarray): The initial orientation quaternion.

    Returns:
        tuple: A tuple containing the best camera position and orientation quaternion.
    """
    best_pose = None
    best_mean = float('-inf')
    current_mean = 0

    # Test four different orientations
    for angle in [0, 90, 180, 270]:
        if angle == 0:
            current_quaternion = initial_quaternion
        else:
            current_quaternion = rotate_camera_x_axis(initial_quaternion, angle)

        set_camera_pose(camera_position, current_quaternion)
        rospy.sleep(1)  # Wait for the pose to settle

        rospy.loginfo(f"Testing Camera Position ({angle}°): {camera_position}")
        rospy.loginfo(f"Testing Orientation Quaternion ({angle}°): {current_quaternion}")

        _, sim_selected_12 = controller.detect_features()
        current_mean = sim_selected_12.mean()
        rospy.loginfo(f"sim mean for {angle}° rotation: {current_mean}")

        # Update the best pose if current mean is higher
        if current_mean > best_mean:
            best_mean = current_mean
            best_pose = (camera_position, current_quaternion)

    # Set the camera to the best pose
    set_camera_pose(*best_pose)
    rospy.sleep(1)  # Wait for the pose to settle

    rospy.loginfo(f"Selected Best Camera Position: {best_pose[0]}")
    rospy.loginfo(f"Selected Best Orientation Quaternion: {best_pose[1]}")
    rospy.loginfo(f"Best sim mean: {best_mean}")

    return best_pose


def manage_gazebo_models(model_index):
    """
    Delete the current model and spawn a new perturbed model in Gazebo.

    Args:
        model_index (int): The index of the perturbed model to spawn (1-500).
    """
    # Delete the current model
    rospy.wait_for_service('/gazebo/delete_model')
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

        # If it's the first iteration, delete the original "resized" model
        if model_index == 1:
            model_to_delete = "resized"
        else:
            model_to_delete = f"resized{model_index - 1}"

        delete_model(model_to_delete)
        rospy.loginfo(f"Deleted model: {model_to_delete}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to delete model {model_to_delete}: {e}")

    # Spawn the new perturbed model
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

        # Construct the path to the new model
        model_path = f"/root/catkin_ws/src/ibvs/models/viso{model_index}/model.sdf"

        # Check if the model file exists
        if not os.path.exists(model_path):
            rospy.logerr(f"Model file not found: {model_path}")
            return

        with open(model_path, "r") as f:
            model_xml = f.read()

        # Set the pose for the new model
        initial_pose = Pose()
        initial_pose.position.x = 0
        initial_pose.position.y = 0
        initial_pose.position.z = 0.005

        # Convert Euler angles to quaternion
        quaternion = tf_conversions.transformations.quaternion_from_euler(1.5708, 0, 1.5708)
        initial_pose.orientation.x = quaternion[0]
        initial_pose.orientation.y = quaternion[1]
        initial_pose.orientation.z = quaternion[2]
        initial_pose.orientation.w = quaternion[3]

        # Spawn the model
        new_model_name = f"resized{model_index}"
        spawn_model(new_model_name, model_xml, "", initial_pose, "world")

        rospy.loginfo(f"Spawned perturbed model: {new_model_name}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to spawn model {new_model_name}: {e}")


def calculate_look_at_orientation(camera_positions, focal_points):
    """
    Calculate the rotation matrix and quaternion for the camera to look at the target position.

    Args:
        camera_positions (np.ndarray): The positions of the camera.
        focal_points (np.ndarray): The target positions (focal point).

    Returns:
        tuple: A tuple containing two numpy arrays:
               - An array of rotation matrices for each sample
               - An array of quaternions (w, x, y, z) for each sample
    """
    num_samples = len(camera_positions)
    rotation_matrices = np.empty((num_samples, 3, 3))
    quaternions = np.empty((num_samples, 4))

    for i in range(num_samples):
        # Calculate the forward vector (X-axis of camera)
        forward = focal_points[i] - camera_positions[i]
        forward = forward / np.linalg.norm(forward)

        # Calculate the right vector (negative Y-axis of camera)
        world_up = np.array([-1, 0, 0])  # Z is up in world space
        right = -np.cross(forward, world_up)
        right = right / np.linalg.norm(right)

        # Calculate the up vector (Z-axis of camera)
        up = np.cross(right, forward)

        # Construct the rotation matrix
        rotation_matrix = np.column_stack((forward, -right, up))
        rotation_matrices[i] = rotation_matrix

        # Convert rotation matrix to quaternion
        r = R.from_matrix(rotation_matrix)
        quat = r.as_quat()  # Returns in scalar-last format [x, y, z, w]
        quaternions[i] = r.as_quat()

    return rotation_matrices, quaternions


def apply_z_axis_rotation(rotation_matrices, num_circles, samples_per_circle, rz_max=np.radians(120)):
    """
    Apply a random rotation around the optical axis (z-axis) to the given rotation matrices.

    Args:
        rotation_matrices (np.ndarray): The initial rotation matrices.
        num_circles (int): Number of circles used in sampling.
        samples_per_circle (int): Number of samples per circle.
        rz_max (float): Maximum rotation angle around the optical axis in radians.

    Returns:
        np.ndarray: An array of quaternions (x, y, z, w) for each sample after z-axis rotation.
    """
    num_samples = len(rotation_matrices)
    quaternions = []

    for cn in range(num_circles):
        # Generate a sequence of rotation angles for this circle
        rz_values = np.linspace(-rz_max, rz_max, num=samples_per_circle)

        for i in range(samples_per_circle):
            idx = cn * samples_per_circle + i
            if idx >= num_samples:
                break

            # Create a rotation matrix for the optical axis rotation
            rz = rz_values[i]
            cos_rz = np.cos(rz)
            sin_rz = np.sin(rz)
            Rx = np.array([
                [1, 0, 0],
                [0, cos_rz, -sin_rz],
                [0, sin_rz, cos_rz]
            ])

            # Apply the optical axis rotation to the initial rotation matrix
            final_rotation_matrix = np.dot(rotation_matrices[idx], Rx)

            # Convert final rotation matrix to quaternion using scipy
            r = R.from_matrix(final_rotation_matrix)
            quaternion = r.as_quat()  # Returns in scalar-last format [x, y, z, w]

            # Reorder to [w, x, y, z] to match common conventions
            # quaternion = np.roll(quaternion, 1)

            quaternions.append(quaternion)

    return np.array(quaternions)


def main(args):
    # Initialize the ROS node
    rospy.init_node('visual_servoing_inference', anonymous=True)
    start_time = time.time()

    # Lists to store final camera positions, quaternions, convergence flags, and End Errors
    final_positions = []
    final_quaternions = []
    convergence_flags = []
    position_errors = []
    orientation_errors = []
    all_position_histories = []
    all_orientation_histories = []
    all_iteration_histories = []
    lowest_position_errors = []
    lowest_orientation_errors = []
    all_average_velocities = []
    all_velocity_mean_100 = []
    all_velocity_mean_10 = []
    all_applied_velocity_x = []
    all_applied_velocity_y = []
    all_applied_velocity_z = []
    all_applied_velocity_roll = []
    all_applied_velocity_pitch = []
    all_applied_velocity_yaw = []

    # Load parameters from YAML file
    current_directory = os.path.dirname(__file__)
    config_filename = args.config if args.config else 'config.yaml'
    config_path = os.path.join(current_directory, f'../config/{config_filename}')
    config_path = os.path.abspath(config_path)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Get parameters from config
    num_circles = config['num_circles']
    circle_radius_aug = config['circle_radius_aug']
    samples_per_circle = config['num_samples'] // num_circles
    num_samples = num_circles * samples_per_circle  # Ensure num_samples is exactly divisible

    rospy.loginfo(f"Processing {num_samples} samples ({num_circles} circles with {samples_per_circle} samples each)")

    # Define the desired position and orientation
    desired_position = np.array([0, 0, 0.61])
    desired_orientation = np.array([0, 0.7071068, 0, 0.7071068])
    box_sample_size = np.array([1.2, 1.2, 0.3])
    reference_point = np.array([0.0, 0.0, 0.01])

    # Set the random seed
    np.random.seed(41)

    # Sample the camera positions
    camera_positions = sample_camera_positions(box_sample_size, num_samples, desired_position)
    rospy.loginfo(f"Generated {len(camera_positions)} camera positions")

    # Sample focal points
    focal_points = sample_focal_points_original(num_samples, reference_point, num_circles, circle_radius_aug)
    rospy.loginfo(f"Generated {len(focal_points)} focal points")

    # Calculate look-at orientations for the cameras
    look_at_matrices, look_at_quaternions = calculate_look_at_orientation(camera_positions, focal_points)
    rospy.loginfo(f"Generated {len(look_at_matrices)} look-at matrices")

    # Pre-calculate position and orientation errors before processing
    avg_pos_error, std_pos_error = calculate_position_error(camera_positions, desired_position)
    rospy.loginfo(f"Average Position Error (before processing): {avg_pos_error:.2f} cm")
    rospy.loginfo(f"Standard Deviation of Position Error (before processing): {std_pos_error:.2f} cm")

    # Apply z-axis rotation to the look-at orientations
    orientations = apply_z_axis_rotation(look_at_matrices, num_circles, samples_per_circle)
    rospy.loginfo(f"Generated {len(orientations)} orientations")

    avg_orient_error, std_orient_error = calculate_orientation_error(orientations, desired_orientation)
    rospy.loginfo(f"Average Orientation Error (before processing): {avg_orient_error:.2f} degrees")
    rospy.loginfo(f"Standard Deviation of Orientation Error (before processing): {std_orient_error:.2f}")

    # Initialize controller object
    controller = Controller(desired_position, desired_orientation, config_path, method=args.method)

    # Verify array sizes before processing
    if not (len(camera_positions) == len(orientations) == num_samples):
        rospy.logerr(f"Array size mismatch: camera_positions={len(camera_positions)}, "
                     f"orientations={len(orientations)}, num_samples={num_samples}")
        return

    for i in range(num_samples):
        rospy.loginfo(f"Processing sample {i + 1}/{num_samples}")

        try:
            if args.perturbation:
                manage_gazebo_models(i + 1)
                rospy.sleep(1)

            # Set camera pose directly
            set_camera_pose(camera_positions[i], orientations[i])
            rospy.sleep(1)

            # Run visual servoing
            result = controller.run()

            # Check if result is None (indicating a failure)
            if result is None:
                rospy.logwarn(f"Sample {i + 1} failed - skipping")
                # Add placeholder values for failed sample
                final_positions.append(np.full_like(desired_position, np.nan))
                final_quaternions.append(np.full_like(desired_orientation, np.nan))
                convergence_flags.append(False)
                position_errors.append(np.nan)
                orientation_errors.append(np.nan)
                all_position_histories.append(np.array([]))
                all_orientation_histories.append(np.array([]))
                all_iteration_histories.append(0)
                lowest_position_errors.append(np.nan)
                lowest_orientation_errors.append(np.nan)
                all_average_velocities.append(np.array([]))
                all_velocity_mean_100.append(np.array([]))
                all_velocity_mean_10.append(np.array([]))
                all_applied_velocity_x.append(np.array([]))
                all_applied_velocity_y.append(np.array([]))
                all_applied_velocity_z.append(np.array([]))
                all_applied_velocity_roll.append(np.array([]))
                all_applied_velocity_pitch.append(np.array([]))
                all_applied_velocity_yaw.append(np.array([]))
                continue

            (final_position, final_quaternion, converged, position_error, orientation_error,
             position_history, orientation_history, iteration_history,
             lowest_position_error, lowest_orientation_error,
             average_velocities, velocity_mean_100, velocity_mean_10,
             applied_velocity_x, applied_velocity_y, applied_velocity_z,
             applied_velocity_roll, applied_velocity_pitch, applied_velocity_yaw) = result

            # Store results
            final_positions.append(final_position)
            final_quaternions.append(final_quaternion)
            convergence_flags.append(converged)
            position_errors.append(position_error)
            orientation_errors.append(orientation_error)
            all_position_histories.append(position_history)
            all_orientation_histories.append(orientation_history)
            all_iteration_histories.append(iteration_history)
            lowest_position_errors.append(lowest_position_error)
            lowest_orientation_errors.append(lowest_orientation_error)
            all_average_velocities.append(average_velocities)
            all_velocity_mean_100.append(velocity_mean_100)
            all_velocity_mean_10.append(velocity_mean_10)
            all_applied_velocity_x.append(applied_velocity_x)
            all_applied_velocity_y.append(applied_velocity_y)
            all_applied_velocity_z.append(applied_velocity_z)
            all_applied_velocity_roll.append(applied_velocity_roll)
            all_applied_velocity_pitch.append(applied_velocity_pitch)
            all_applied_velocity_yaw.append(applied_velocity_yaw)

            rospy.loginfo(f"Completed sample {i + 1} (Converged: {converged})")

        except Exception as e:
            rospy.logerr(f"Error processing sample {i + 1}: {str(e)}")
            # Add placeholder values for failed sample
            final_positions.append(np.full_like(desired_position, np.nan))
            final_quaternions.append(np.full_like(desired_orientation, np.nan))
            convergence_flags.append(False)
            position_errors.append(np.nan)
            orientation_errors.append(np.nan)
            all_position_histories.append(np.array([]))
            all_orientation_histories.append(np.array([]))
            all_iteration_histories.append(0)
            lowest_position_errors.append(np.nan)
            lowest_orientation_errors.append(np.nan)
            all_average_velocities.append(np.array([]))
            all_velocity_mean_100.append(np.array([]))
            all_velocity_mean_10.append(np.array([]))
            all_applied_velocity_x.append(np.array([]))
            all_applied_velocity_y.append(np.array([]))
            all_applied_velocity_z.append(np.array([]))
            all_applied_velocity_roll.append(np.array([]))
            all_applied_velocity_pitch.append(np.array([]))
            all_applied_velocity_yaw.append(np.array([]))
            continue

    end_time = time.time()
    total_execution_time = end_time - start_time

    config_name = os.path.splitext(os.path.basename(args.config))[0]
    method_name = args.method
    perturbation_str = "perturbed" if args.perturbation else "standard"
    results_filename = f"results_{config_name}_{method_name}_{perturbation_str}.npz"

    # Save all data to a file
    try:
        np.savez(results_filename,
                 initial_positions=camera_positions,
                 initial_orientations=orientations,
                 final_positions=np.array(final_positions),
                 final_quaternions=np.array(final_quaternions),
                 convergence_flags=np.array(convergence_flags),
                 position_errors=np.array(position_errors),
                 orientation_errors=np.array(orientation_errors),
                 all_position_histories=np.array(all_position_histories, dtype=object),
                 all_orientation_histories=np.array(all_orientation_histories, dtype=object),
                 all_iteration_histories=np.array(all_iteration_histories),
                 lowest_position_errors=np.array(lowest_position_errors),
                 lowest_orientation_errors=np.array(lowest_orientation_errors),
                 all_average_velocities=np.array(all_average_velocities, dtype=object),
                 all_velocity_mean_100=np.array(all_velocity_mean_100, dtype=object),
                 all_velocity_mean_10=np.array(all_velocity_mean_10, dtype=object),
                 all_applied_velocity_x=np.array(all_applied_velocity_x, dtype=object),
                 all_applied_velocity_y=np.array(all_applied_velocity_y, dtype=object),
                 all_applied_velocity_z=np.array(all_applied_velocity_z, dtype=object),
                 all_applied_velocity_roll=np.array(all_applied_velocity_roll, dtype=object),
                 all_applied_velocity_pitch=np.array(all_applied_velocity_pitch, dtype=object),
                 all_applied_velocity_yaw=np.array(all_applied_velocity_yaw, dtype=object),
                 total_execution_time=total_execution_time)

        rospy.loginfo(f"Results saved to {results_filename}")

    except Exception as e:
        rospy.logerr(f"Error saving results: {str(e)}")

    # Calculate and display final statistics
    try:
        # Calculate orientation error after processing (excluding NaN values)
        valid_quaternions = [q for q in final_quaternions if not np.any(np.isnan(q))]
        if valid_quaternions:
            avg_orient_error, std_orient_error = calculate_orientation_error(valid_quaternions, desired_orientation)
            rospy.loginfo(f"Average Orientation Error (after processing): {avg_orient_error:.2f} degrees")
            rospy.loginfo(f"Standard Deviation of Orientation Error (after processing): {std_orient_error:.2f} degrees")
        else:
            rospy.logwarn("No valid quaternions to calculate orientation error")

        # Calculate average End Errors for converged samples (excluding NaN values)
        valid_position_errors = [err for err, flag in zip(position_errors, convergence_flags)
                                 if flag and not np.isnan(err)]
        valid_orientation_errors = [err for err, flag in zip(orientation_errors, convergence_flags)
                                    if flag and not np.isnan(err)]

        if valid_position_errors:
            avg_position_error = np.nanmean(valid_position_errors)
            avg_orientation_error = np.nanmean(valid_orientation_errors)
            rospy.loginfo(f"Average End Position Error (for converged samples): {avg_position_error:.2f} cm")
            rospy.loginfo(f"Average End Orientation Error (for converged samples): {avg_orientation_error:.2f} degrees")
            rospy.loginfo(f"Number of converged samples: {len(valid_position_errors)} out of {num_samples}")
        else:
            rospy.logwarn("No valid converged samples to calculate average errors")

    except Exception as e:
        rospy.logerr(f"Error calculating final statistics: {str(e)}")

    rospy.loginfo(f"Total execution time: {total_execution_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Servoing Inference")
    parser.add_argument("--config", type=str, help="Name of the configuration YAML file")
    parser.add_argument("--perturbation", action="store_true", help="Enable image perturbation")
    parser.add_argument("--method", type=str, choices=['sift', 'orb', 'akaze'], default='sift',
                        help="Feature detection method to use (sift, orb, or akaze)")
    args = parser.parse_args()

    main(args)
