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


from PyriteUtility.data_pipeline.episode_data_buffer import (
    VideoData,
    EpisodeDataBuffer,
)
from PyriteUtility.planning_control.filtering import LiveLPFilter
import concurrent.futures


# check environment variables

''' 1. Input Data Path Setting'''
if "PYRITE_RAW_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_RAW_DATASET_FOLDERS")
input_dir = pathlib.Path(
    os.environ.get("PYRITE_RAW_DATASET_FOLDERS") + "/swipe_board"
)
# input_dir = "/home/yihan/Documents/ft_sensor_ws/src/visuomotor_policy_ros/data/default_task_expert"


''' 2. Output Data Path Setting'''
if "PYRITE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_DATASET_FOLDERS")
output_dir = pathlib.Path(os.environ.get("PYRITE_DATASET_FOLDERS") + "/flip_up_new_v5")
# output_dir = "/home/yihan/Documents/ft_sensor_ws/src/visuomotor_policy_ros/data/default_task_dataset"

robot_timestamp_dir = output_dir.joinpath("robot_timestamp")
wrench_timestamp_dir = output_dir.joinpath("wrench_timestamp")
rgb_timestamp_dir = output_dir.joinpath("rgb_timestamp")


''' 3. Robot Number Config '''
# specify the input and output directories
id_list = [0]  # single robot
# id_list = [0, 1] # bimanual


''' 4. Read pkl Data '''
image_arrays = []
pose_arrays = []
wrench_arrays = []


# clean and create output folders
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# open the zarr store
store = zarr.DirectoryStore(path=output_dir)
root = zarr.open(store=store, mode="a")

print("Reading data from input_dir: ", input_dir)
demo_dirs = [f for f in os.listdir(input_dir) if f.endswith(".pkl")]
episode_names = os.listdir(input_dir)


''' 5. Data Processing Function'''
def process_one_episode(root, episode_name, input_dir, id_list):
    if episode_name.startswith("."):
        return True

    # info about input
    
    # episode_id = episode_name[8:]
    match = re.search(r'\d+', episode_name)
    episode_id = match.group()
    print(f"episode_name: {episode_name}, episode_id: {episode_id}")
    episode_dir = input_dir.joinpath(episode_name)
    with open(episode_dir, 'rb') as f:
        episode_data = pickle.load(f)

    # read rgb
    data_rgb = []
    data_rgb_time_stamps = []
    rgb_data_shapes = []

    for id in id_list:
        rgb_data_with_time_from_pkl = episode_data["rgb_data_with_timestamp"]
        rgb_timastamp = np.array([tup[0] for tup in rgb_data_with_time_from_pkl])
        rgb_timastamp = rgb_timastamp/1e6
        rgb_data = np.array([tup[1] for tup in rgb_data_with_time_from_pkl]).astype(np.uint8)
        rgb_length = len(rgb_data)
        img = rgb_data[0]
        rgb_data_shapes.append((rgb_length,*img.shape))

        data_rgb.append(rgb_data)
        data_rgb_time_stamps.append(rgb_timastamp)


    # read low dim data
    data_ts_pose_fb = []
    data_robot_time_stamps = []
    data_wrench = []
    data_wrench_time_stamps = []
    print(f"Reading low dim data for : {episode_dir}")
    for id in id_list:
        # read pose data
        pose_data_with_time_from_pkl = episode_data["pose_data_with_timestamp"]
        pose_timestamp = np.array([tup[0] for tup in pose_data_with_time_from_pkl])
        pose_timestamp = pose_timestamp/1e6
        pose_data = np.array([tup[1] for tup in pose_data_with_time_from_pkl])
        print("pose_timestamp: ",len(pose_timestamp))
        print("pose_data shape: ",pose_data.shape)
        data_ts_pose_fb.append(pose_data)
        data_robot_time_stamps.append(pose_timestamp)

        # read wrench data
        wrench_data_with_time_from_pkl = episode_data["wrench_data_with_timestamp"]
        wrench_timestamp = np.array([tup[0] for tup in wrench_data_with_time_from_pkl])
        wrench_timestamp = wrench_timestamp/1e6
        wrench_data = np.array([tup[1] for tup in wrench_data_with_time_from_pkl])
        print("wrench_data shape: ",wrench_data.shape)
        data_wrench.append(wrench_data)
        data_wrench_time_stamps.append(wrench_timestamp)
        

    # get filtered force
    print(f"Computing filtered wrench for {episode_name}")
    force_filtering_para = {
        "sampling_freq": 100,
        "cutoff_freq": 5,
        "order": 5,
    }
    ft_filter = LiveLPFilter(
        fs=500,
        cutoff=5,
        order=5,
        dim=6,
    )
    data_wrench_filtered = []
    for id in id_list:
        data_wrench_filtered.append(np.array([ft_filter(y) for y in data_wrench[id]]))

    # make time stamps start from zero
    time_offsets = []
    for id in id_list:
        time_offsets.append(data_rgb_time_stamps[id][0])
        time_offsets.append(data_robot_time_stamps[id][0])
        time_offsets.append(data_wrench_time_stamps[id][0])
    time_offset = np.min(time_offsets)
    for id in id_list:
        data_rgb_time_stamps[id] -= time_offset
        data_robot_time_stamps[id] -= time_offset
        data_wrench_time_stamps[id] -= time_offset
    print("data_rgb_time_stamps0: ",data_rgb_time_stamps[id][0])
    print("data_robot_time_stamps0: ",data_robot_time_stamps[id][0])
    print("data_wrench_time_stamps0: ",data_wrench_time_stamps[id][0])

    # create output zarr
    print(f"Saving everything to : {output_dir}")
    recoder_buffer = EpisodeDataBuffer(
        store_path=output_dir,
        camera_ids=id_list,
        save_video=True,
        save_video_fps=60,
        data=root,
    )

    # save data using recoder_buffer
    rgb_data_buffer = {}
    for id in id_list:
        rgb_data = data_rgb[id]
        rgb_data_buffer.update({id: VideoData(rgb=rgb_data, camera_id=id)})
    recoder_buffer.create_zarr_groups_for_episode(rgb_data_shapes, id_list, episode_id)
    recoder_buffer.save_video_for_episode(
        visual_observations=rgb_data_buffer,
        visual_time_stamps=data_rgb_time_stamps,
        episode_id=episode_id,
    )
    recoder_buffer.save_low_dim_for_episode(
        ts_pose_command=data_ts_pose_fb,
        ts_pose_fb=data_ts_pose_fb,
        wrench=data_wrench,
        wrench_filtered=data_wrench_filtered,
        robot_time_stamps=data_robot_time_stamps,
        wrench_time_stamps=data_wrench_time_stamps,
        episode_id=episode_id,
    )
    return True



''' 6. Data Processing and Save Data as zarr Database'''

with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
    futures = [
        executor.submit(
            process_one_episode,
            root,
            episode_name,
            input_dir,
            id_list,
        )
        for episode_name in episode_names
    ]
    for future in concurrent.futures.as_completed(futures):
        if not future.result():
            raise RuntimeError("Multi-processing failed!")

print("Finished reading. Now start generating metadata")
from PyriteUtility.computer_vision.imagecodecs_numcodecs import register_codecs

register_codecs()
buffer = zarr.open(output_dir)
meta = buffer.create_group("meta", overwrite=True)
episode_robot_len = []
episode_wrench_len = []
episode_rgb_len = []

for id in id_list:
    episode_robot_len.append([])
    episode_wrench_len.append([])
    episode_rgb_len.append([])

count = 0
for key in buffer["data"].keys():
    episode = key
    ep_data = buffer["data"][episode]

    for id in id_list:
        episode_robot_len[id].append(ep_data[f"ts_pose_fb_{id}"].shape[0])
        episode_wrench_len[id].append(ep_data[f"wrench_{id}"].shape[0])
        episode_rgb_len[id].append(ep_data[f"rgb_{id}"].shape[0])
        print(
            f"Number {count}: {episode}: id = {id}: robot len: {episode_robot_len[id][-1]}, wrench_len: {episode_wrench_len[id][-1]} rgb len: {episode_rgb_len[id][-1]}"
        )
    count += 1

for id in id_list:
    meta[f"episode_robot{id}_len"] = zarr.array(episode_robot_len[id])
    meta[f"episode_wrench{id}_len"] = zarr.array(episode_wrench_len[id])
    meta[f"episode_rgb{id}_len"] = zarr.array(episode_rgb_len[id])

print(f"All done! Generated {count} episodes in {output_dir}")
print("The only thing left is to run postprocess_add_virtual_target_label.py")
