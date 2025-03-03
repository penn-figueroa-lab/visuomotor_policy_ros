import sys
import os

sys.path.append(os.path.join(sys.path[0], ".."))  # PyriteUtility

from einops import rearrange, reduce
import time
import torch
import copy
import numpy as np
from typing import Optional, Callable

from PyriteConfig.tasks.common import common_type_conversions as task

from PyriteUtility.common import dict_apply
import PyriteUtility.spatial_math.spatial_utilities as su


def printOrNot(verbose, *args):
    if verbose >= 0:
        print(*args)


# fmt: off
class ModelPredictiveControllerHybrid():
    """Class that maintains IO buffering of a MPC controller with hybrid policy.
    args:
        shape_meta: dict
        policy: torch.nn.Module
        action_to_trajectory: A function that converts waypoints to a continuous trajectory
        execution_horizon: MPC execution horizon in number of steps
        obs_queue_size: size of the observation queue. Should make sure that the queue is large
                        enough for downsampling. Should not be too large to avoid waiting for
                        too long before MPC starts.
        low_dim_obs_upsample_ratio: The real feedback frequency is usually lower than that used in training. 
            for example, training may uses 500hz data, but real feedback is 100hz. In this case, we need to
            set low_dim_obs_upsample_ratio to 5. Then if shape_meta sparse_obs_low_dim_down_sample_steps = 5,
            the MPC will downsample feedback at step = 1, instead of step = 5.

    Important internal variables:
        obs_queue: stores observations specified in shape_meta['obs']
                   Data in this queue is supposed to be at raw frequency, and need to be downsampled
                   according to 'down_sample_steps' in shape_meta.
    """
    def __init__(self,
        shape_meta,
        id_list,
        policy,
        action_to_trajectory: Callable[[np.ndarray], Callable],
        sparse_execution_horizon=10,
    ):
        print("[MPC] Initializing")
        self.shape_meta = shape_meta
        self.id_list = id_list

        action_type = "pose9" # "pose9" or "pose9pose9s1"
        if shape_meta['action']['shape'][0] == 9:
            action_type = "pose9"
        elif shape_meta['action']['shape'][0] == 19:
            action_type = "pose9pose9s1"
        elif shape_meta['action']['shape'][0] == 38:
            action_type = "pose9pose9s1"
        else:
            raise RuntimeError('unsupported')

        if action_type == "pose9":
            action_postprocess = task.action9_postprocess
        elif action_type == "pose9pose9s1":
            action_postprocess = task.action19_postprocess
        else:
            raise RuntimeError('unsupported')
        self.action_type = action_type
        self.action_postprocess = action_postprocess

        self.policy = policy
        self.sparse_execution_horizon_time_step = sparse_execution_horizon
        self.action_to_trajectory = action_to_trajectory

        # internal variables
        self.time_offset = None
        self.sparse_obs_data = {}
        self.sparse_obs_last_timestamps = {}
        self.horizon_start_time_step = -np.inf
        self.sparse_action_traj = []
        self.SE3_WBase = None
        self.verbose_level = -1

        self.sparse_target_mats = None
        self.sparse_vt_mats = None
        self.stiffness = None

        # added just for debugging
        self.sparse_action = None

        print("[MPC] Done initializing")


    def set_time_offset(self, hardware):
        '''
        Set time offset such that timing in this controller is aligned with hardware time.
        hardware: a class with a property 'current_hardware_time_s'
        '''
        self.time_offset = hardware.current_hardware_time_s - time.perf_counter()

    def set_observation(self, obs_task):
        for key, attr in self.shape_meta['sample']['obs']['sparse'].items():
            data = obs_task[key]
            horizon = attr['horizon']
            down_sample_steps = attr['down_sample_steps']
            # sample 'horizon' number of latest obs from the queue
            assert len(data) >= (horizon-1) * down_sample_steps + 1
            self.sparse_obs_data[key] = data[-(horizon-1) * down_sample_steps - 1::down_sample_steps]

        for id in self.id_list:
            self.sparse_obs_last_timestamps[f"rgb_time_stamps_{id}"] = obs_task[f"rgb_time_stamps_{id}"][-1]
            self.sparse_obs_last_timestamps[f"robot_time_stamps_{id}"] = obs_task[f"robot_time_stamps_{id}"][-1]
            self.sparse_obs_last_timestamps[f"wrench_time_stamps_{id}"] = obs_task[f"wrench_time_stamps_{id}"][-1]

    def compute_sparse_control(self, device):
        """ Run sparse model inference once. Does not output control.
        """
        time_now = time.perf_counter() + self.time_offset
        for id in self.id_list:
            dt_rgb = time_now - self.sparse_obs_last_timestamps[f"rgb_time_stamps_{id}"]
            dt_ts_pose = time_now - self.sparse_obs_last_timestamps[f"robot_time_stamps_{id}"]
            dt_wrench = time_now - self.sparse_obs_last_timestamps[f"wrench_time_stamps_{id}"]
            print(f'[MPC] obs lagging for robot {id}: dt_rgb: {dt_rgb}, dt_ts_pose: {dt_ts_pose}, dt_wrench: {dt_wrench}')

        with torch.no_grad():
            s = time.time()
            obs_sample_np = {}
            obs_sample_np['sparse'], SE3_WBase = task.sparse_obs_to_obs_sample(
                obs_sparse=self.sparse_obs_data,
                shape_meta=self.shape_meta,
                reshape_mode='reshape',
                id_list=self.id_list,
                ignore_rgb=False,
            )
            self.SE3_WBase = SE3_WBase
            # add batch dimension
            obs_sample_np = dict_apply(obs_sample_np,
                lambda x: rearrange(x, '... -> 1 ...'))
            # convert to torch tensor
            obs_sample = dict_apply(obs_sample_np,
                lambda x: torch.from_numpy(x).to(device))

            result = self.policy.predict_action(obs_sample)
            raw_action = result['sparse'][0].detach().to('cpu').numpy()
            
            action = self.action_postprocess(raw_action, SE3_WBase, self.id_list)
            printOrNot(self.verbose_level, 'Sparse inference latency:', time.time() - s)
            return action

    def update_control(self, time_step, device):
        """ Update control per time step.
            A ValueError is raised if no IK solution is found.
        """
        sparse_action_horizon = self.shape_meta['sample']['action']['sparse']['horizon']
        sparse_action_down_sample_steps = self.shape_meta['sample']['action']['sparse']['down_sample_steps']

        if time_step > self.horizon_start_time_step + self.sparse_execution_horizon_time_step:
            # time for a new horizon
            
            # TODO: replace this mechanism with multi-threading, remove these two variables
            time0 = time.perf_counter()
            p_timestep_s = 0.01 # 0.002

            if self.action_type == "pose9":
                self.sparse_action = self.compute_sparse_control(device)
            elif self.action_type == "pose9pose9s1":
                self.sparse_target_mats, self.sparse_vt_mats, self.stiffness = self.compute_sparse_control(device)
            else:
                raise RuntimeError('unsupported')
            
            sparse_action_timesteps = np.arange(0, sparse_action_horizon) * sparse_action_down_sample_steps
            if self.action_type == "pose9":
                self.sparse_action_traj = self.action_to_trajectory(action_mats = self.sparse_action, time_steps = sparse_action_timesteps)
            elif self.action_type == "pose9pose9s1":
                self.sparse_action_traj = self.action_to_trajectory(target_mats = self.sparse_target_mats,
                                                                    vt_mats = self.sparse_target_mats, # debug self.sparse_vt_mats,
                                                                    stiffness = self.stiffness,
                                                                    time_steps = sparse_action_timesteps)

            time1 = time.perf_counter()
            
            self.horizon_start_time_step = time_step + (time1 - time0 - 0.012) / p_timestep_s

            print("[MPC debug] new horizon start time step: ", self.horizon_start_time_step, ", computation takes: ", time1 - time0, "s")
            return None # indicates a new horizon has started


        time_now = time_step - self.horizon_start_time_step
        print("[MPC debug] time_step in horizon: ", time_now)
        time_now = max(time_now, 0)
        target_now = self.sparse_action_traj(time_now)
        if (np.linalg.norm(target_now[1].reshape([4, 4])[:3, 3]) < 1e-3):
            print('Warning: sparse action is zero at time_now:', time_now)
        return target_now

    def get_SE3_targets(self):
        return self.sparse_target_mats, self.sparse_vt_mats

# fmt: on
