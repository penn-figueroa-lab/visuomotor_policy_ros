import os
import zarr
import pickle
import numpy as np
from termcolor import cprint
import pathlib
import pandas as pd
import numpy as np
import pickle
import shutil
import cv2
import re
import sys
sys.path.append("/home/yihan/Documents/adaptive_compliance_policy")

input_dir = "/home/yihan/Documents/ft_sensor_ws/src/visuomotor_policy_ros/data/default_task_expert"

demo_dirs = [f for f in os.listdir(input_dir) if f.endswith(".pkl")]
episode_names = os.listdir(input_dir)

def check_one_episode(input_dir, episode_name):
    episode_dir = input_dir.joinpath(episode_name)
    with open(episode_dir, 'rb') as f:
        episode_data = pickle.load(f)
    
    rgb_data_with_time_from_pkl = episode_data["rgb_data_with_timestamp"]
    rgb_timestamp = np.array([tup[0] for tup in rgb_data_with_time_from_pkl])
    rgb_timestamp = rgb_timestamp/1e6
    pose_data_with_time_from_pkl = episode_data["pose_data_with_timestamp"]
    pose_timestamp = np.array([tup[0] for tup in pose_data_with_time_from_pkl])
    pose_timestamp = pose_timestamp/1e6
    wrench_data_with_time_from_pkl = episode_data["wrench_data_with_timestamp"]
    wrench_timestamp = np.array([tup[0] for tup in wrench_data_with_time_from_pkl])
    wrench_timestamp = wrench_timestamp/1e6
    
    
    
    
    # Synchronization Check using np.searchsorted
    for t in rgb_timestamp:
        pose_idx = np.searchsorted(pose_timestamp, t)
        wrench_idx = np.searchsorted(wrench_timestamp, t)

        # Ensure indices are within range
        pose_idx = min(pose_idx, len(pose_timestamp) - 1)
        wrench_idx = min(wrench_idx, len(wrench_timestamp) - 1)

        # Get closest timestamps
        pose_time = pose_timestamp[pose_idx]
        wrench_time = wrench_timestamp[wrench_idx]

        # Compute differences
        pose_diff = abs(pose_time - t)
        wrench_diff = abs(wrench_time - t)

        # Threshold for synchronization (in seconds)
        threshold = 0.01  # 10 ms

        if pose_diff > threshold:
            print(f"Warning: RGB timestamp {t:.6f} sec has mismatched pose timestamp {pose_time:.6f} sec (diff={pose_diff:.6f} sec)")

        if wrench_diff > threshold:
            print(f"Warning: RGB timestamp {t:.6f} sec has mismatched wrench timestamp {wrench_time:.6f} sec (diff={wrench_diff:.6f} sec)")

    print(f"Episode {episode_name} checked.")
    
    
    



