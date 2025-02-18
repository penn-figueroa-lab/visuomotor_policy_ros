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
from pathlib import Path
sys.path.append("/home/yihan/Documents/adaptive_compliance_policy")

input_dir = Path("/home/yihan/Documents/ft_sensor_ws/src/visuomotor_policy_ros/data/default_task_expert")

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
    
    time_offsets = []
    
    time_offsets.append(rgb_timestamp[0])
    time_offsets.append(pose_timestamp[0])
    time_offsets.append(wrench_timestamp[0])
    time_offset = np.min(time_offsets)

    rgb_timestamp -= time_offset
    pose_timestamp -= time_offset
    wrench_timestamp -= time_offset
    
    
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
        threshold = 10  # 10 ms

        if pose_diff > threshold:
            print(f"Warning: RGB timestamp {t:.6f} sec has mismatched pose timestamp {pose_time:.6f} sec (diff={pose_diff:.6f} sec)")

        if wrench_diff > threshold:
            print(f"Warning: RGB timestamp {t:.6f} sec has mismatched wrench timestamp {wrench_time:.6f} sec (diff={wrench_diff:.6f} sec)")

    print(f"Episode {episode_name} checked.")
    return rgb_timestamp,pose_timestamp,wrench_timestamp

def writedown_timestamp_source(rgb_timestamp, pose_timestamp, wrench_timestamp,output_txt):
    # Create sets of float timestamps for each sensor.
    sensor_timestamps = {
        "rgb": set(rgb_timestamp.tolist()),
        "pose": set(pose_timestamp.tolist()),
        "force": set(wrench_timestamp.tolist())  # using 'force' for wrench
    }

    # Build the union of all unique float timestamps and sort them.
    all_timestamps = sorted(sensor_timestamps["rgb"].union(sensor_timestamps["pose"]).union(sensor_timestamps["force"]))

    # Build the combined timeline lines.
    lines = []
    for t in all_timestamps:
        sensors = []
        for sensor_name, ts_set in sensor_timestamps.items():
            if t in ts_set:
                sensors.append(sensor_name)
        # Format the line as: "timestamp, sensor1, sensor2, ..."
        line = f"{t}, " + ", ".join(sensors)
        lines.append(line)

    # Write the combined timeline to a text file.
    with open(output_txt, "w") as f:
        for line in lines:
            f.write(line + "\n")



for name in episode_names:
    print("Check episode: ",name)
    rgb_timestamp,pose_timestamp,wrench_timestamp=check_one_episode(input_dir,name)
    writedown_timestamp_source(rgb_timestamp,pose_timestamp,wrench_timestamp,output_txt=f"src/data_process/sensor_output_{name}.txt")


    



