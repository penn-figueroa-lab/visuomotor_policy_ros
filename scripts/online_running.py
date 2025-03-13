import sys
import os
from typing import Dict, Callable, Tuple, List

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../"))

import cv2
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import zarr
import rospy
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PointStamped, PoseStamped
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from cv_bridge import CvBridge




from online_model_rollout.common_format import raw_to_obs
import online_model_rollout.spacial_utility as su
from online_model_rollout.policy_inference_utility import ModelPredictiveControllerHybrid
# from PyriteUtility.planning_control.trajectory import LinearTransformationInterpolator
from online_model_rollout.model_io import load_policy
# from PyriteUtility.plotting.matplotlib_helpers import set_axes_equal
# from PyriteUtility.umi_utils.usb_util import reset_all_elgato_devices
# from PyriteUtility.common import GracefulKiller
from online_model_rollout.common_format import get_image_transform


if "PYRITE_CHECKPOINT_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_CHECKPOINT_FOLDERS")
# if "PYRITE_HARDWARE_CONFIG_FOLDERS" not in os.environ:
#     raise ValueError(
#         "Please set the environment variable PYRITE_HARDWARE_CONFIG_FOLDERS"
#     )
if "PYRITE_CONTROL_LOG_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_CONTROL_LOG_FOLDERS")

checkpoint_folder_path = os.environ.get("PYRITE_CHECKPOINT_FOLDERS")
hardware_config_folder_path = os.environ.get("PYRITE_HARDWARE_CONFIG_FOLDERS")
control_log_folder_path = os.environ.get("PYRITE_CONTROL_LOG_FOLDERS")

# id list, 0 for single arm
id_list = [0]

class ObservationDataBuffer:
    def __init__(self,query_sizes, cam_res, policy_obs_res):
        self.rgb_topic = rospy.get_param("~rgb_topic", "")
        self.pose_topic = rospy.get_param("~pose_topic", "")
        self.wrench_topic = rospy.get_param("~wrench_topic", "")

        if self.rgb_topic == "" or \
            self.pose_topic == "" or \
            self.wrench_topic == "":

            rospy.logerr("One or more parameters are missing. Please set the parameters")
            rospy.signal_shutdown("Missing parameters")


        # Subscribers
        rospy.Subscriber(self.pose_topic, PoseStamped, self.end_effector_callback)
        rospy.Subscriber(self.wrench_topic, WrenchStamped, self.force_sensor_callback)
        rospy.Subscriber(self.rgb_topic, Image, self.camera_callback)

        self.id_list = [0]

        # Sensor message initialization
        self.init_rgb = None
        self.init_pose = None
        self.init_wrench = None

        # Data Buffers for Sensors(Saved in Tuple format)
        self.stamped_rgb_data_buffer = []
        self.stamped_pose_data_buffer = []
        self.stamped_wrench_data_buffer = []

        # Query size for observation, in dictionary format, keys: rgb, wrench, pose
        self.query_sizes = query_sizes

        # Image Size Transformation Function
        self.bridge = CvBridge()
        self.image_transform = get_image_transform(input_res=cam_res, output_res=policy_obs_res, bgr_to_rgb=False)

        # Data Buffer for Observations
        self.rgb_buffer = [np.zeros((self.query_sizes["rgb"], cam_res[0], cam_res[1], 3),dtype=np.uint8)]
        self.rgb_timestamp_s = [np.array] * len(id_list)
        self.ts_pose_buffer = [np.array] * len(id_list)
        self.ts_pose_fb_timestamp_s = [np.array] * len(id_list)
        self.wrench_buffer = [np.array] * len(id_list)
        self.wrench_timestamp_s = [np.array] * len(id_list)


    def end_effector_callback(self, eef_msg):
        # Use PoseStamped message for 
        # publish_time = eef_msg.header.stamp.to_sec()
        received_time = rospy.Time.now().to_nsec()
        pose = np.array([
            eef_msg.pose.position.x, eef_msg.pose.position.y, eef_msg.pose.position.z,
            eef_msg.pose.orientation.x, eef_msg.pose.orientation.y, eef_msg.pose.orientation.z, eef_msg.pose.orientation.w
        ])
        self.stamped_pose_data_buffer.append((received_time,pose))
        self.init_pose = pose
        if self.init_pose is None:
            rospy.logwarn(f"Missing pose data")
       

    def force_sensor_callback(self, ft_msg):
        received_time = rospy.Time.now().to_nsec()
        wrench = np.array([
            ft_msg.wrench.force.x, ft_msg.wrench.force.y, ft_msg.wrench.force.z,
            ft_msg.wrench.torque.x, ft_msg.wrench.torque.y, ft_msg.wrench.torque.z
        ])
        self.stamped_wrench_data_buffer.append((received_time,wrench))
        self.init_wrench = wrench
        if self.init_wrench is None:
            rospy.logwarn(f"Missing wrench data")


    def camera_callback(self, rgb_msg):
        try:
            received_time = rospy.Time.now().to_nsec()

            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            rgb = cv_image[0:720,280:1000]
            rgb = cv2.resize(rgb, (224,224), interpolation=cv2.INTER_AREA)
            self.stamped_rgb_data_buffer.append((received_time,rgb))
            self.init_rgb = rgb
            if self.init_rgb is None:
                rospy.logwarn(f"Missing rgb data")
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")

    def get_obs(self):
        for id in self.id_list:
            # rgb data
            # print("self.stamped_rgb_data_buffer: ", self.stamped_rgb_data_buffer)
            # print("self.rgb_buffer[id]: ",self.rgb_buffer[id])
            self.rgb_timestamp_s[id] = np.array([tup[0] for tup in self.stamped_rgb_data_buffer[-self.query_sizes["rgb"]:]])/1e9
            self.rgb_buffer[id] = np.array([tup[1] for tup in self.stamped_rgb_data_buffer[-self.query_sizes["rgb"]:]])
            for i in range(self.query_sizes["rgb"]):
                self.rgb_buffer[id][i] = self.image_transform(
                    self.rgb_buffer[id][i]
                )
            
            # pose data
            self.ts_pose_fb_timestamp_s[id] = np.array([tup[0] for tup in self.stamped_pose_data_buffer[-self.query_sizes["ts_pose_fb"]:]])/1e9
            self.ts_pose_buffer[id] = np.array([tup[1] for tup in self.stamped_pose_data_buffer[-self.query_sizes["ts_pose_fb"]:]])
           
            # wrench data
            self.wrench_timestamp_s[id] = np.array([tup[0] for tup in self.stamped_wrench_data_buffer[-self.query_sizes["wrench"]:]])/1e9
            self.wrench_buffer[id] = np.array([tup[1] for tup in self.stamped_wrench_data_buffer[-self.query_sizes["wrench"]:]])
        
        results = {}
        for id in self.id_list:
            results[f"rgb_{id}"] = self.rgb_buffer[id]
            results[f"rgb_time_stamps_{id}"] = self.rgb_timestamp_s[id]
            results[f"ts_pose_fb_{id}"] = self.ts_pose_buffer[id]
            results[f"robot_time_stamps_{id}"] = self.ts_pose_fb_timestamp_s[id]
            results[f"wrench_{id}"] = self.wrench_buffer[id]
            results[f"wrench_time_stamps_{id}"] = self.wrench_timestamp_s[id]
        return results


class PolicyRollout:
    def __init__(self, device=torch.device("cuda"), dtype=torch.float32):
        if torch.cuda.is_available() and device == torch.device("cuda"):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        

        self.control_para = {
        "raw_time_step_s": 0.002,  # dt of raw data collection. Used to compute time step from time_s such that the downsampling according to shape_meta works.
        "slow_down_factor": 1.5,  # 3 for flipup, 1.5 for wiping
        "sparse_execution_horizon": 12,  # 12 for flipup, 8/24 for wiping
        "max_duration_s": 3500,
        "pausing_mode": False,
        }
        self.pipeline_para = {
            "save_low_dim_every_N_frame": 1,
            "save_visual_every_N_frame": 1,
            "ckpt_path": "/2025.03.04_13.43.38_flip_up_new_conv_230",
            "control_log_path": control_log_folder_path + "/temp/",
        }
        
        ckp_load_path = checkpoint_folder_path + self.pipeline_para["ckpt_path"]
        print("ckp_load_path: ", ckp_load_path)
        self.policy, self.shape_meta = load_policy(checkpoint_folder_path + self.pipeline_para["ckpt_path"], self.device)


        (policy_image_width, policy_image_height) = self.get_real_obs_resolution(self.shape_meta)

        # Observation Query Size
        rgb_query_size = (
            self.shape_meta["sample"]["obs"]["sparse"]["rgb_0"]["horizon"] - 1
        ) * self.shape_meta["sample"]["obs"]["sparse"]["rgb_0"]["down_sample_steps"] + 1
        ts_pose_query_size = (
            self.shape_meta["sample"]["obs"]["sparse"]["robot0_eef_pos"]["horizon"] - 1
        ) * self.shape_meta["sample"]["obs"]["sparse"]["robot0_eef_pos"]["down_sample_steps"] + 1
        wrench_query_size = (
            self.shape_meta["sample"]["obs"]["sparse"]["robot0_eef_wrench"]["horizon"] - 1
        ) * self.shape_meta["sample"]["obs"]["sparse"]["robot0_eef_wrench"][
            "down_sample_steps"
        ] + 1
        self.query_sizes = {
            "rgb": rgb_query_size,
            "ts_pose_fb": ts_pose_query_size,
            "wrench": wrench_query_size,
        }

        # Observation Data Buffer
        self.obs_data_buffer = ObservationDataBuffer(query_sizes=self.query_sizes, cam_res=(224,224), policy_obs_res=(policy_image_width, policy_image_height))
        time.sleep(2)

        # Action Timestamp Management
        self.p_timestep_s = self.control_para["raw_time_step_s"]
        sparse_action_down_sample_steps = self.shape_meta["sample"]["action"]["sparse"][
            "down_sample_steps"
        ]
        sparse_action_horizon = self.shape_meta["sample"]["action"]["sparse"]["horizon"]
        sparse_execution_horizon = (
            sparse_action_down_sample_steps * self.control_para["sparse_execution_horizon"]
        )
        self.sparse_action_timesteps_s = (
            np.arange(0, sparse_action_horizon)
            * sparse_action_down_sample_steps
            * self.p_timestep_s
            * self.control_para["slow_down_factor"]
        )

        # Lowlevel action
        self.id_list = [0]

        self.controller = ModelPredictiveControllerHybrid(
            shape_meta=self.shape_meta,
            id_list=id_list,
            policy=self.policy,
            sparse_execution_horizon=sparse_execution_horizon,
        )

        # Action Publisher
        self.action_publisher = rospy.Publisher('/adp/ref_pose_stiffness', Float64MultiArray, queue_size=10)
  


    def get_real_obs_resolution(self, shape_meta: dict) -> Tuple[int, int]:
        out_res = None
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            shape = attr.get("shape")
            if type == "rgb":
                co, ho, wo = shape
                if out_res is None:
                    out_res = (wo, ho)
                assert out_res == (wo, ho)
        return out_res
    
    def run_online_rollout(self):
        while not rospy.is_shutdown():
            rospy.loginfo("START ONLINE!!!!!")
            # Get observation
            obs_raw = self.obs_data_buffer.get_obs()
            # print("obs_raw: ",obs_raw)
            obs_task = dict()
            raw_to_obs(obs_raw, obs_task, self.shape_meta)

            # Run inference
            self.controller.set_observation(obs_task["obs"])
            (
                action_sparse_target_mats,
                action_sparse_vt_mats,
                action_stiffnesses,
            ) = self.controller.compute_sparse_control(self.device)

            # decode stiffness matrix
            outputs_ts_nominal_targets = [np.array] * len(id_list)
            outputs_ts_targets = [np.array] * len(id_list)
            outputs_ts_stiffnesses = [np.array] * len(id_list)
            for id in id_list:
                SE3_TW = su.SE3_inv(su.pose7_to_SE3(obs_raw[f"ts_pose_fb_{id}"][-1]))
                ts_targets_nominal = su.SE3_to_pose7(
                    action_sparse_target_mats[id].reshape([-1, 4, 4])
                )
                ts_targets_virtual = su.SE3_to_pose7(
                    action_sparse_vt_mats[id].reshape([-1, 4, 4])
                )

                ts_stiffnesses = np.zeros([6, 6 * ts_targets_virtual.shape[0]])
                for i in range(ts_targets_virtual.shape[0]):
                    SE3_target = action_sparse_target_mats[id][i].reshape([4, 4])
                    SE3_virtual_target = action_sparse_vt_mats[id][i].reshape([4, 4])
                    stiffness = action_stiffnesses[id][i]

                    # stiffness: 1. convert vt to tool frame
                    SE3_TVt = SE3_TW @ SE3_virtual_target
                    SE3_Ttarget = SE3_TW @ SE3_target

                    # stiffness: 2. compute stiffness matrix in the tool frame
                    compliance_direction_tool = (
                        SE3_TVt[:3, 3] - SE3_Ttarget[:3, 3]
                    ).reshape(3)

                    if np.linalg.norm(compliance_direction_tool) < 0.001:  #
                        compliance_direction_tool = np.array([1.0, 0.0, 0.0])

                    compliance_direction_tool /= np.linalg.norm(
                        compliance_direction_tool
                    )
                    X = compliance_direction_tool
                    Y = np.cross(X, np.array([0, 0, 1]))
                    Y /= np.linalg.norm(Y)
                    Z = np.cross(X, Y)

                    default_stiffness = 5000
                    default_stiffness_rot = 100
                    target_stiffness = stiffness

                    M = np.diag(
                        [target_stiffness, default_stiffness, default_stiffness]
                    )
                    S = np.array([X, Y, Z]).T
                    stiffness_matrix = S @ M @ np.linalg.inv(S)
                    stiffness_matrix_full = np.eye(6) * default_stiffness_rot
                    stiffness_matrix_full[:3, :3] = stiffness_matrix
                    ts_stiffnesses[:, 6 * i : 6 * i + 6] = stiffness_matrix_full

                outputs_ts_nominal_targets[id] = ts_targets_nominal
                outputs_ts_targets[id] = ts_targets_virtual
                outputs_ts_stiffnesses[id] = ts_stiffnesses
            
            action_start_time_s = obs_raw["robot_time_stamps_0"][-1]

            outputs_ts_targets = outputs_ts_targets[0].T  # N x 7 to 7 x N
            outputs_ts_stiffnesses = outputs_ts_stiffnesses[0]
            timestamps = self.sparse_action_timesteps_s + action_start_time_s

            joint_data_list = [outputs_ts_targets.tolist(), outputs_ts_stiffnesses.tolist()]
            flattened_data = np.concatenate(joint_data_list, axis=None).tolist()
 
            msg = Float64MultiArray()
            msg.data = flattened_data  # Store actual data

            msg.layout.dim.append(MultiArrayDimension(label="matrix_num", size=2, stride=len(outputs_ts_targets)*len(outputs_ts_targets[0])))  # 2 matrix
            msg.layout.dim.append(MultiArrayDimension(label="horizon", size=len(outputs_ts_targets), stride=1))  # 6 columns
            self.action_publisher.publish(msg)


def main():
    rospy.init_node('online_running')
    policy = PolicyRollout()
    policy.run_online_rollout()



if __name__ == "__main__":
    main()
    
