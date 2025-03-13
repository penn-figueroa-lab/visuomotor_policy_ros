#!/usr/bin/env python

import rospy
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PointStamped, PoseStamped
import tf.transformations as tf
import numpy as np
import pickle
from std_srvs.srv import Empty, EmptyResponse
import os
import datetime
import cv2
from cv_bridge import CvBridge
import time

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

        self.init_rgb = None
        self.latest_pose = None
        self.init_wrench = None


        # Data storage buffers
        self.rgb_data_with_timestamp = []
        # self.rgb_timestamp = []
        self.pose_data_with_timestamp = []
        # self.pose_timestamp = []
        self.wrench_data_with_timestamp = []
        # self.wrench_timestamp = []

        self.recording = False

        # Subscribers
        rospy.Subscriber(self.pose_topic, PoseStamped, self.end_effector_callback)
        rospy.Subscriber(self.wrench_topic, WrenchStamped, self.force_sensor_callback)
        rospy.Subscriber(self.rgb_topic, Image, self.camera_callback)

        # Services for start/stop recording
        rospy.Service('/start_recording', Empty, self.start_recording)
        rospy.Service('/stop_recording', Empty, self.stop_recording)

        # Timer for periodic data collection for end effector pose (300Hz)
        rospy.Timer(rospy.Duration(1.0 / 300), self.eef_pose_data_collection_callback)

        # Initialize episode number
        if not rospy.has_param('/episode_num'):
            rospy.set_param('/episode_num',0)

    def start_recording(self, req):
        self.recording = True
        self.rgb_data_with_timestamp = []
        # self.rgb_timestamp = []
        self.pose_data_with_timestamp = []
        # self.pose_timestamp = []
        self.wrench_data_with_timestamp = []
        # self.wrench_timestamp = []
        rospy.loginfo("Started recording data.")
        return EmptyResponse()

    def stop_recording(self, req):
        """Service callback to stop recording and save data"""
        self.recording = False
        data = {
            "rgb_data_with_timestamp": self.rgb_data_with_timestamp,
            "pose_data_with_timestamp":self.pose_data_with_timestamp,
            "wrench_data_with_timestamp": self.wrench_data_with_timestamp
        }

        # Generate timestamped file name
        episode_num = rospy.get_param('/episode_num')
        # timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")  # Format: MMDDYY_HHMMSS
        file_path = os.path.join(self.task_save_path, f"episode_{episode_num}.pkl")
        rospy.set_param('/episode_num',episode_num+1)

        with open(file_path, "wb") as file:
            pickle.dump(data, file)

        rospy.loginfo(f"Recording stopped. Data saved to {file_path}")
        return EmptyResponse()
    
    
    # def timer_callback(self, event):

    #     if self.latest_rgb is None:
    #         rospy.logwarn(f"Missing rgb data")
    #         return
    #     if self.latest_pose is None:
    #         rospy.logwarn(f"Missing pose data")
    #         return
    #     if self.latest_wrench is None:
    #         rospy.logwarn(f"Missing wrench data")
    #         return
        
    #     # If all data is available, record it
    #     if self.recording:
    #         self.rgb_data.append(self.latest_rgb)
    #         self.pose_data.append(self.latest_pose)
    #         self.wrench_data.append(self.latest_wrench)
    #         self.time_data.append(time.time())
    #     else:
    #         rospy.logwarn(f"Standby... ready to start recording")

    def end_effector_callback(self, eef_msg):
        # Use PoseStamped message for 
        # publish_time = eef_msg.header.stamp.to_sec()
        pose = np.array([
            eef_msg.pose.position.x, eef_msg.pose.position.y, eef_msg.pose.position.z,
            eef_msg.pose.orientation.x, eef_msg.pose.orientation.y, eef_msg.pose.orientation.z, eef_msg.pose.orientation.w
        ])
        self.latest_pose = pose
        if self.latest_pose is None:
            rospy.logwarn(f"Missing pose data")
        # self.pose_timestamp.append(publish_time)

    def eef_pose_data_collection_callback(self, event):
        received_time = rospy.Time.now().to_nsec()
        self.pose_data_with_timestamp.append((received_time,self.latest_pose))
       

    def force_sensor_callback(self, ft_msg):
        # publish_time = ft_msg.header.stamp.to_sec() 
        received_time = rospy.Time.now().to_nsec()
        wrench = np.array([
            ft_msg.wrench.force.x, ft_msg.wrench.force.y, ft_msg.wrench.force.z,
            ft_msg.wrench.torque.x, ft_msg.wrench.torque.y, ft_msg.wrench.torque.z
        ])
        self.wrench_data_with_timestamp.append((received_time,wrench))
        self.init_wrench = wrench
        if self.init_wrench is None:
            rospy.logwarn(f"Missing wrench data")

        # self.wrench_timestamp.append(publish_time)

    def camera_callback(self, rgb_msg):
        try:
            received_time = rospy.Time.now().to_nsec()
            # publish_time = rgb_msg.header.stamp.to_sec() 
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            self.rgb_data_with_timestamp.append((received_time,cv_image))
            self.init_rgb = cv_image
            if self.init_rgb is None:
                rospy.logwarn(f"Missing rgb data")
            # self.rgb_timestamp.append(publish_time)
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")

# Main function
def main():
    rospy.init_node('data_collector_without_align')
    data = data_saver()

    loop_rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        loop_rate.sleep()
    
    
if __name__ == '__main__':
    main()
