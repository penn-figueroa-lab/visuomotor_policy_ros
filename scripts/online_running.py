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



# from PyriteEnvSuites.envs.task.manip_server_env import ManipServerEnv
# from PyriteEnvSuites.utils.env_utils import ts_to_js_traj, pose9pose9s1_to_traj

from online_model_rollout.common_format import raw_to_obs
import online_model_rollout.spacial_utility as su
from online_model_rollout.policy_inference_utility import ModelPredictiveControllerHybrid
from PyriteUtility.planning_control.trajectory import LinearTransformationInterpolator
from online_model_rollout.model_io import load_policy
from PyriteUtility.plotting.matplotlib_helpers import set_axes_equal
from PyriteUtility.umi_utils.usb_util import reset_all_elgato_devices
from PyriteUtility.common import GracefulKiller

if "PYRITE_CHECKPOINT_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_CHECKPOINT_FOLDERS")
if "PYRITE_HARDWARE_CONFIG_FOLDERS" not in os.environ:
    raise ValueError(
        "Please set the environment variable PYRITE_HARDWARE_CONFIG_FOLDERS"
    )
if "PYRITE_CONTROL_LOG_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_CONTROL_LOG_FOLDERS")

checkpoint_folder_path = os.environ.get("PYRITE_CHECKPOINT_FOLDERS")
hardware_config_folder_path = os.environ.get("PYRITE_HARDWARE_CONFIG_FOLDERS")
control_log_folder_path = os.environ.get("PYRITE_CONTROL_LOG_FOLDERS")


def main():
    control_para = {
        "raw_time_step_s": 0.002,  # dt of raw data collection. Used to compute time step from time_s such that the downsampling according to shape_meta works.
        "slow_down_factor": 1.5,  # 3 for flipup, 1.5 for wiping
        "sparse_execution_horizon": 12,  # 12 for flipup, 8/24 for wiping
        "max_duration_s": 3500,
        "pausing_mode": False,
        "device": "cuda",
    }
    pipeline_para = {
        "save_low_dim_every_N_frame": 1,
        "save_visual_every_N_frame": 1,
        "ckpt_path": "/2024.09.25_21.10.43_flip_up_new_resnet_230",
        # "hardware_config_path": hardware_config_folder_path + "/manip_server_config_left_arm.yaml",
        "hardware_config_path": hardware_config_folder_path
        + "/manip_server_config_bimanual.yaml",
        "control_log_path": control_log_folder_path + "/temp/",
    }
    force_filtering_para = {
        "sampling_freq": 100,
        "cutoff_freq": 5,
        "order": 5,
    }
    verbose = 1

    episode_id = 0

    def get_real_obs_resolution(shape_meta: dict) -> Tuple[int, int]:
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

    def printOrNot(verbose, *args):
        if verbose >= 0:
            print(f"[Episode {episode_id}] ", *args)

    vbs_h1 = verbose + 1  # verbosity for header
    vbs_h2 = verbose  # verbosity for sub-header
    vbs_p = verbose - 1  # verbosity for paragraph

    reset_all_elgato_devices()

    # load policy
    print("Loading policy: ", checkpoint_folder_path + pipeline_para["ckpt_path"])
    device = torch.device(control_para["device"])
    policy, shape_meta = load_policy(
        checkpoint_folder_path + pipeline_para["ckpt_path"], device
    )

    # image size
    (image_width, image_height) = get_real_obs_resolution(shape_meta)

    # zarr img buffer size estimation: 20s, 1000hz step rate
    img_buffer_size_estimated = int(
        20 * 1000 / pipeline_para["save_visual_every_N_frame"]
    )
    rgb_buffer_shape_nhwc = (
        img_buffer_size_estimated,
        image_height,
        image_width,
        3,
    )

    rgb_query_size = (
        shape_meta["sample"]["obs"]["sparse"]["rgb_0"]["horizon"] - 1
    ) * shape_meta["sample"]["obs"]["sparse"]["rgb_0"]["down_sample_steps"] + 1
    ts_pose_query_size = (
        shape_meta["sample"]["obs"]["sparse"]["robot0_eef_pos"]["horizon"] - 1
    ) * shape_meta["sample"]["obs"]["sparse"]["robot0_eef_pos"]["down_sample_steps"] + 1
    wrench_query_size = (
        shape_meta["sample"]["obs"]["sparse"]["robot0_eef_wrench"]["horizon"] - 1
    ) * shape_meta["sample"]["obs"]["sparse"]["robot0_eef_wrench"][
        "down_sample_steps"
    ] + 1
    query_sizes = {
        "rgb": rgb_query_size,
        "ts_pose_fb": ts_pose_query_size,
        "wrench": wrench_query_size,
    }

    env = ManipServerEnv(
        camera_res_hw=(image_height, image_width),
        hardware_config_path=pipeline_para["hardware_config_path"],
        filter_params=force_filtering_para,
        query_sizes=query_sizes,
        compliant_dimensionality=3,
    )

    env.reset()

    p_timestep_s = control_para["raw_time_step_s"]

    sparse_action_down_sample_steps = shape_meta["sample"]["action"]["sparse"][
        "down_sample_steps"
    ]
    sparse_action_horizon = shape_meta["sample"]["action"]["sparse"]["horizon"]
    sparse_action_horizon_s = (
        sparse_action_horizon * sparse_action_down_sample_steps * p_timestep_s
    )
    sparse_execution_horizon = (
        sparse_action_down_sample_steps * control_para["sparse_execution_horizon"]
    )
    sparse_action_timesteps_s = (
        np.arange(0, sparse_action_horizon)
        * sparse_action_down_sample_steps
        * p_timestep_s
        * control_para["slow_down_factor"]
    )

    action_type = "pose9"  # "pose9" or "pose9pose9s1"
    id_list = [0]
    if shape_meta["action"]["shape"][0] == 9:
        action_type = "pose9"
    elif shape_meta["action"]["shape"][0] == 19:
        action_type = "pose9pose9s1"
    elif shape_meta["action"]["shape"][0] == 38:
        action_type = "pose9pose9s1"
        id_list = [0, 1]
    else:
        raise RuntimeError("unsupported")

    if action_type == "pose9pose9s1":
        action_to_trajectory = pose9pose9s1_to_traj
    else:
        raise RuntimeError("unsupported")

    printOrNot(vbs_h2, "Creating MPC.")
    controller = ModelPredictiveControllerHybrid(
        shape_meta=shape_meta,
        id_list=id_list,
        policy=policy,
        action_to_trajectory=action_to_trajectory,
        sparse_execution_horizon=sparse_execution_horizon,
    )
    controller.set_time_offset(env)

    # timestep_idx = 0
    # stiffness = None
    episode_initial_time_s = env.current_hardware_time_s
    execution_duration_s = (
        sparse_execution_horizon * p_timestep_s * control_para["slow_down_factor"]
    )
    printOrNot(vbs_h2, "Starting main loop.")

    if control_para["pausing_mode"]:
        plt.ion()  # to run GUI event loop
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        x = np.linspace(-0.02, 0.2, 20)
        y = np.linspace(-0.1, 0.1, 20)
        z = np.linspace(-0.1, 0.1, 20)
        ax.plot3D(x, y, z, color="blue", marker="o", markersize=3)
        ax.plot3D(x, y, z, color="red", marker="o", markersize=3)
        ax.set_title("Target and virtual target")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

    # log
    log_store = zarr.DirectoryStore(path=pipeline_para["control_log_path"])
    log = zarr.open(store=log_store, mode="w")  # w: overrite if exists

    horizon_count = 0
    print("test plotting RGB. Press q to continue.")
    while True:
        obs_raw = env.get_observation_from_buffer()

        # plot the rgb image
        if len(id_list) == 1:
            rgb = obs_raw["rgb_0"][-1]
        else:
            rgb = np.vstack([obs_raw[f"rgb_{i}"][-1] for i in id_list])
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", bgr)
        key = cv2.waitKey(10)
        if key == ord("q"):
            break

    ts_pose_initial = []
    for id in id_list:
        ts_pose_initial.append(obs_raw[f"ts_pose_fb_{id}"][-1])
    #########################################
    # main loop starts
    #########################################
    while True:
        printOrNot(vbs_h1, "New episode. Episode ID: ", episode_id)
        input("Press Enter to start the episode.")
        killer = GracefulKiller()
        while not killer.kill_now:
            horizon_initial_time_s = env.current_hardware_time_s
            printOrNot(vbs_h1, "Starting new horizon at ", horizon_initial_time_s)

            obs_raw = env.get_observation_from_buffer()

            # plot the rgb image
            if len(id_list) == 1:
                rgb = obs_raw["rgb_0"][-1]
            else:
                rgb = np.vstack([obs_raw[f"rgb_{i}"][-1] for i in id_list])
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("image", bgr)
            cv2.waitKey(10)

            obs_task = dict()
            raw_to_obs(obs_raw, obs_task, shape_meta)

            assert action_type == "pose9pose9s1"

            # Run inference
            controller.set_observation(obs_task["obs"])
            (
                action_sparse_target_mats,
                action_sparse_vt_mats,
                action_stiffnesses,
            ) = controller.compute_sparse_control(device)

            # for id in id_list:
            #     print(f"Stiffness {id}: ", action_stiffnesses[id])

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

            # the "now" when the observation is taken
            action_start_time_s = obs_raw["robot_time_stamps_0"][-1]
            timestamps = sparse_action_timesteps_s

            if control_para["pausing_mode"]:
                # plot the actions for this horizon using matplotlib
                ax.cla()
                for id in id_list:
                    ax.plot3D(
                        action_sparse_target_mats[id][..., 0, 3],
                        action_sparse_target_mats[id][..., 1, 3],
                        action_sparse_target_mats[id][..., 2, 3],
                        color="red",
                        marker="o",
                        markersize=3,
                    )

                    ax.plot3D(
                        action_sparse_vt_mats[id][..., 0, 3],
                        action_sparse_vt_mats[id][..., 1, 3],
                        action_sparse_vt_mats[id][..., 2, 3],
                        color="blue",
                        marker="o",
                        markersize=3,
                    )

                    ax.plot3D(
                        action_sparse_target_mats[id][
                            : control_para["sparse_execution_horizon"], 0, 3
                        ],
                        action_sparse_target_mats[id][
                            : control_para["sparse_execution_horizon"], 1, 3
                        ],
                        action_sparse_target_mats[id][
                            : control_para["sparse_execution_horizon"], 2, 3
                        ],
                        color="yellow",
                        linewidth=3,
                        marker="o",
                        markersize=4,
                    )

                    ax.plot3D(
                        action_sparse_vt_mats[id][
                            : control_para["sparse_execution_horizon"], 0, 3
                        ],
                        action_sparse_vt_mats[id][
                            : control_para["sparse_execution_horizon"], 1, 3
                        ],
                        action_sparse_vt_mats[id][
                            : control_para["sparse_execution_horizon"], 2, 3
                        ],
                        color="green",
                        linewidth=3,
                        marker="o",
                        markersize=4,
                    )

                    ax.plot3D(
                        obs_raw[f"ts_pose_fb_{id}"][-1][0],
                        obs_raw[f"ts_pose_fb_{id}"][-1][1],
                        obs_raw[f"ts_pose_fb_{id}"][-1][2],
                        color="black",
                        marker="o",
                        markersize=8,
                    )

                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")

                set_axes_equal(ax)

                plt.draw()

                input("Press Enter to start executing the plotted actions.")

                # log the whole action
                horizon_log = log.create_group(f"horizon_{horizon_count}")
                action_start_time_s = env.current_hardware_time_s
                for id in id_list:
                    horizon_log.create_dataset(
                        f"ts_virtual_targets_{id}", data=outputs_ts_targets[id]
                    )
                    horizon_log.create_dataset(
                        f"ts_nominal_targets_{id}", data=outputs_ts_nominal_targets[id]
                    )
                    horizon_log.create_dataset(
                        f"ts_stiffnesses_{id}", data=outputs_ts_stiffnesses[id]
                    )
                horizon_log.create_dataset(
                    "timestamps_s", data=timestamps + action_start_time_s
                )

                # cut the action and only keep the execution horizon
                for id in id_list:
                    outputs_ts_targets[id] = outputs_ts_targets[id][
                        : control_para["sparse_execution_horizon"], :
                    ]
                    outputs_ts_stiffnesses[id] = outputs_ts_stiffnesses[id][
                        :, : 6 * control_para["sparse_execution_horizon"]
                    ]
                timestamps = timestamps[: control_para["sparse_execution_horizon"]]
                action_start_time_s = env.current_hardware_time_s

            if len(id_list) == 1:
                outputs_ts_targets = outputs_ts_targets[0].T  # N x 7 to 7 x N
                outputs_ts_stiffnesses = outputs_ts_stiffnesses[0]
            else:
                outputs_ts_targets = np.hstack(
                    outputs_ts_targets
                ).T  # 2 x N x 7 to 14 x N
                outputs_ts_stiffnesses = np.vstack(
                    outputs_ts_stiffnesses
                )  # 6 x (6xN) to 12 x (6xN)

            env.schedule_controls(
                pose7_cmd=outputs_ts_targets,
                stiffness_matrices_6x6=outputs_ts_stiffnesses,
                timestamps=(timestamps + action_start_time_s) * 1000,
            )

            # # log the truncated action
            # horizon_log = log.create_group(f"horizon_{horizon_count}")
            # horizon_log.create_dataset("ts_targets", data=outputs_ts_targets)
            # horizon_log.create_dataset("timestamps", data=timestamps * 1000)
            # horizon_log.create_dataset("ts_stiffnesses", data=outputs_ts_stiffnesses)
            # horizon_log.create_dataset(
            #     "timestamps_s", data=timestamps + action_start_time_s
            # )
            horizon_count += 1

            if control_para["pausing_mode"]:
                c = input("Press Enter to start the next horizon. q to quit.")
                if c == "q":
                    break

            # # log
            # if warming_up_done:
            #     if timestep_idx % pipeline_para["save_low_dim_every_N_frame"] == 0:
            #         env.add_low_dim_observation(None, ts_target, time_s)
            #         # env.add_optional_observation(stiffness)
            #     if timestep_idx % pipeline_para["save_visual_every_N_frame"] == 0:
            #         env.add_visual_observation(time_s)

            time_s = env.current_hardware_time_s
            sleep_duration_s = horizon_initial_time_s + execution_duration_s - time_s

            printOrNot(vbs_h1, "sleep_duration_s", sleep_duration_s)
            time.sleep(max(0, sleep_duration_s))

            if not control_para["pausing_mode"]:
                # only check duration when not in pausing mode
                if time_s - episode_initial_time_s > control_para["max_duration_s"]:
                    break

        printOrNot(vbs_h1, "End of episode.")
        episode_id += 1

        print("Options:")
        print("     c: continue to next episode.")
        print("     r: reset to default pose, then continue.")
        print("     b: reset to default pose, then quit the program.")
        print("     others: quit the program.")
        c = input("Please select an option: ")
        if c == "r" or c == "b":
            print("Resetting to default pose.")
            obs_raw = env.get_observation_from_buffer()
            N = 100
            duration_s = 5
            timestamps = np.linspace(0, 1, N) * duration_s
            homing_ts_targets = np.zeros([7 * len(id_list), N])
            for id in id_list:
                ts_pose_fb = obs_raw[f"ts_pose_fb_{id}"][-1]
                SE3_waypoints = [
                    su.pose7_to_SE3(ts_pose_fb),
                    su.pose7_to_SE3(ts_pose_initial[id]),
                ]

                SE3_interpolator = LinearTransformationInterpolator(
                    x_wp=np.array([0, duration_s]),
                    y_wp=np.array(SE3_waypoints),
                )
                SE3_waypoints = SE3_interpolator(timestamps)

                for i in range(N):
                    wpi = SE3_waypoints[i]
                    pose7 = su.SE3_to_pose7(wpi)
                    homing_ts_targets[0 + id * 7 : 7 + id * 7, i] = pose7

            time_now_s = env.current_hardware_time_s
            env.schedule_controls(
                pose7_cmd=homing_ts_targets,
                timestamps=(timestamps + time_now_s) * 1000,
            )
        elif c == "c":
            pass
        else:
            print("Quitting the program.")
            break

        if c == "b":
            input("Press Enter to quit program.")
            break

        print("Continuing to execution.")
    env.cleanup()
    # end of episode


if __name__ == "__main__":
    main()
