#!/usr/bin/env python

import rospy
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PointStamped
import tf.transformations as tf
import numpy as np
import pickle
from std_srvs.srv import Empty, EmptyResponse
import os
import datetime
import cv2
from cv_bridge import CvBridge


class data_saver:
    def __init__(self):
        self.rgb_topic = rospy.get_param("~rgb_topic", "")
        self.pose_topic = rospy.get_param("~pose_topic", "")
        self.wrench_topic = rospy.get_param("~wrench_topic", "")
        self.save_location = rospy.get_param("~save_location", "")
        self.task_name = rospy.get_param("~task_name", "")

        if self.rgb_topic == "" or \
            self.pose_topic == "" or \
            self.wrench_topic == "" or \
            self.save_location == "" or \
            self.task_name == "":
            
            rospy.logerr("One or more parameters are missing. Please set the parameters")
            rospy.signal_shutdown("Missing parameters")

        # Construct actual save directory
        self.task_save_path = os.path.join(self.save_location, self.task_name)
        if not os.path.exists(self.task_save_path):
            os.makedirs(self.task_save_path)
            rospy.loginfo(f"Created directory: {self.task_save_path}")

        self.bridge = CvBridge()

        self.latest_rgb = None
        self.latest_pose = None
        self.latest_wrench = None

        # Data storage buffers
        self.rgb_data = []
        self.pose_data = []
        self.wrench_data = []

        self.recording = False

        # Subscribers
        rospy.Subscriber(self.pose_topic, Pose, self.end_effector_callback)
        rospy.Subscriber(self.wrench_topic, WrenchStamped, self.force_sensor_callback)
        rospy.Subscriber(self.rgb_topic, Image, self.camera_callback)

        # Services for start/stop recording
        rospy.Service('/start_recording', Empty, self.start_recording)
        rospy.Service('/stop_recording', Empty, self.stop_recording)

        # Timer for periodic data collection (30Hz)
        rospy.Timer(rospy.Duration(1.0 / 30), self.timer_callback)

    def start_recording(self, req):
        self.recording = True
        self.rgb_data = []
        self.pose_data = []
        self.wrench_data = []
        rospy.loginfo("Started recording data.")
        return EmptyResponse()

    def stop_recording(self, req):
        """Service callback to stop recording and save data"""
        self.recording = False
        data = {
            "rgb_data": self.rgb_data,
            "pose_data": self.pose_data,
            "wrench_data": self.wrench_data
        }

        # Generate timestamped file name
        timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")  # Format: MMDDYY_HHMMSS
        file_path = os.path.join(self.task_save_path, f"{timestamp}.pkl")

        with open(file_path, "wb") as file:
            pickle.dump(data, file)

        rospy.loginfo(f"Recording stopped. Data saved to {file_path}")
        return EmptyResponse()
    
    def timer_callback(self, event):

        if self.latest_rgb is None:
            rospy.logwarn(f"Missing rgb data")
            return
        if self.latest_pose is None:
            rospy.logwarn(f"Missing pose data")
            return
        if self.latest_wrench is None:
            rospy.logwarn(f"Missing wrench data")
            return
        
        # If all data is available, record it
        if self.recording:
            self.rgb_data.append(self.latest_rgb)
            self.pose_data.append(self.latest_pose)
            self.wrench_data.append(self.latest_wrench)
        else:
            rospy.logwarn(f"Standby... ready to start recording")

    def end_effector_callback(self, eef_msg):
        self.latest_pose = np.array([
            eef_msg.position.x, eef_msg.position.y, eef_msg.position.z,
            eef_msg.orientation.x, eef_msg.orientation.y, eef_msg.orientation.z, eef_msg.orientation.w
        ])

    def force_sensor_callback(self, ft_msg):
        self.latest_wrench = np.array([
            ft_msg.wrench.force.x, ft_msg.wrench.force.y, ft_msg.wrench.force.z,
            ft_msg.wrench.torque.x, ft_msg.wrench.torque.y, ft_msg.wrench.torque.z
        ])

    def camera_callback(self, rgb_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            self.latest_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")

# Main function
def main():
    rospy.init_node('data_collector')
    data = data_saver()

    loop_rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        loop_rate.sleep()
    
    
if __name__ == '__main__':
    main()
