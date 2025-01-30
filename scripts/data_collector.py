#!/usr/bin/env python

import rospy
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from franka_msgs.msg import FrankaState
import tf.transformations as tf
import numpy as np
import pickle


class data_saver:
    def __init__(self):
        self.rgb_data_buffer=[]
        self.pose_data_buffer=[]
        self.wrench_data_buffer=[]

        self.rgb_data=[]
        self.pose_data=[]
        self.wrench_data=[]

    def add_new_rgb_data(self, rgb):
        self.rgb_data_buffer.append(rgb)
        
    def add_new_pose_data(self,pose):
        self.pose_data_buffer.append(pose)

    def add_new_wrench_data(self,wrench):
        self.wrench_data_buffer.append(wrench)

    def create_data_point(self):
        self.rgb_data.append(self.rgb_data_buffer[-1])
        self.pose_data.append(self.pose_data_buffer[-1])
        self.wrench_data.append(self.wrench_data_buffer[-1])

    def save_data(self, file_path):
        data = {"rgb_data": self.rgb_data, "pose_data": self.pose_data, "wrench_data": self.wrench_data}
        with open(file_path, "wb") as file:
            pickle.dump(data, file)
        print(f"Data saved to {file_path}")


    # Callback to process Franka state and publish force, torque, and pose
    def end_effector_callback(self, eef_msg):

        eef_pose = np.zeros(7)

        # Extract end-effector pose
        eef_pose[0] = eef_msg.position.x 
        eef_pose[1] = eef_msg.position.y
        eef_pose[2] = eef_msg.position.z

        # Extract quaternion from the Pose message
        eef_pose[3] = eef_msg.orientation.x
        eef_pose[4] = eef_msg.orientation.y
        eef_pose[5] = eef_msg.orientation.z
        eef_pose[6] = eef_msg.orientation.w

        self.add_new_pose_data(eef_pose)

        

    def force_sensor_callback(self,ft_msg):
        # Extract force and torque data
        wrench_data = np.zeros(6)

        wrench_data[0]=ft_msg.wrench.force.x
        wrench_data[1]=ft_msg.wrench.force.y
        wrench_data[2]=ft_msg.wrench.force.z
        wrench_data[3]=ft_msg.wrench.force.x
        wrench_data[4]=ft_msg.wrench.force.y
        wrench_data[5]=ft_msg.wrench.force.z

        self.add_new_wrench_data(wrench_data)


    def camera_callback(self,rgb_msg):
        self.add_new_rgb_data(rgb_msg)
        pass



# Main function
def main():
    rospy.init_node('data_collector')
    data = data_saver()

    # Subscribers
    rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, data.end_effector_callback)
    rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, data.force_sensor_callback)
    rospy.Subscriber('/camera/rgb', Image, data.camera_callback)


    loop_rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        data.create_data_point()
        loop_rate.sleep()
    
    
if __name__ == '__main__':
    
    main()
