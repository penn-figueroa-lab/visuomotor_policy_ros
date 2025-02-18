import sys
import os

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../../"))

import online_model_rollout.spacial_utility as su
# from PyriteUtility.computer_vision.computer_vision_utility import get_image_transform
import numpy as np
from typing import Union, Dict, Optional
import zarr
import torch

##
## raw: keys used in the dataset. Each key contains data for a whole episode
## obs: keys used in inference. Needs some pre-processing before sending to the policy NN.
## obs_preprocessed: obs with normalized rgb keys. len = whole episode
## obs_sample: len = obs horizon, pose computed relative to current pose (id = -1)
## action: pose command in world frame. len = whole episode
## action_sample: len = action horizon, pose computed relative to current pose (id = 0)


def raw_to_obs(
    raw_data: Union[zarr.Group, Dict[str, np.ndarray]],
    episode_data: dict,
    shape_meta: dict,
):
    """convert shape_meta.raw data to shape_meta.obs.

    This function keeps image data as compressed zarr array in memory, while loads and decompresses
    low dim data.

    Args:
      raw_data: input, has keys from shape_meta.raw, each value is an ndarray of shape (t, ...)
      episode_data: output dictionary that matches shape_meta.obs
    """
    episode_data["obs"] = {}
    # obs.rgb: keep entry, keep as compressed zarr array in memory
    for key, attr in shape_meta["raw"].items():
        type = attr.get("type", "low_dim")
        if type == "rgb":
            # obs.rgb: keep as compressed zarr array in memory
            episode_data["obs"][key] = raw_data[key]

    # obs.low_dim: load entry, convert to obs.low_dim
    for id in shape_meta["id_list"]:
        pose7_fb = raw_data[f"ts_pose_fb_{id}"]
        wrench = raw_data[f"wrench_{id}"]

        pose9_fb = su.SE3_to_pose9(su.pose7_to_SE3(pose7_fb))

        episode_data["obs"][f"robot{id}_eef_pos"] = pose9_fb[..., :3]
        episode_data["obs"][f"robot{id}_eef_rot_axis_angle"] = pose9_fb[..., 3:]
        episode_data["obs"][f"robot{id}_eef_wrench"] = wrench[:]

        # optional: abs
        if "robot0_abs_eef_pos" in shape_meta["obs"].keys():
            episode_data["obs"][f"robot{id}_abs_eef_pos"] = pose9_fb[..., :3]
            episode_data["obs"][f"robot{id}_abs_eef_rot_axis_angle"] = pose9_fb[..., 3:]

        # timestamps
        episode_data["obs"][f"rgb_time_stamps_{id}"] = raw_data[
            f"rgb_time_stamps_{id}"
        ][:]
        episode_data["obs"][f"robot_time_stamps_{id}"] = raw_data[
            f"robot_time_stamps_{id}"
        ][:]
        episode_data["obs"][f"wrench_time_stamps_{id}"] = raw_data[
            f"wrench_time_stamps_{id}"
        ][:]
