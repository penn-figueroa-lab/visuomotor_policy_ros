import os
import zarr
import pickle
import tqdm
import numpy as np
import torch
# import pytorch3d.ops as torch3d_ops
import torchvision
from termcolor import cprint
import re
import time
from cv_bridge import CvBridge
import pathlib

import numpy as np
import torch
# import pytorch3d.ops as torch3d_ops
import torchvision
import socket
import pickle
import cv2
import shutil


# check environment variables

''' 1. Input Data Path Setting'''
# if "PYRITE_RAW_DATASET_FOLDERS" not in os.environ:
#     raise ValueError("Please set the environment variable PYRITE_RAW_DATASET_FOLDERS")
# input_dir = pathlib.Path(
#     os.environ.get("PYRITE_RAW_DATASET_FOLDERS") + "/flip_up_new_v5"
# )
input_dir = "/home/yihan/Documents/ft_sensor_ws/src/visuomotor_policy_ros/data/default_task_expert"


''' 2. Output Data Path Setting'''
if "PYRITE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_DATASET_FOLDERS")
output_dir = pathlib.Path(os.environ.get("PYRITE_DATASET_FOLDERS") + "/flip_up_new_v5")
# output_dir = "/home/yihan/Documents/ft_sensor_ws/src/visuomotor_policy_ros/data/default_task_dataset"

robot_timestamp_dir = output_dir.joinpath("robot_timestamp")
wrench_timestamp_dir = output_dir.joinpath("wrench_timestamp")
rgb_timestamp_dir = output_dir.joinpath("rgb_timestamp")


''' 3. Read pkl Data '''
image_arrays = []
pose_arrays = []
wrench_arrays = []
bridge = CvBridge()

# clean and create output folders
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# open the zarr store
store = zarr.DirectoryStore(path=output_dir)
root = zarr.open(store=store, mode="a")

print("Reading data from input_dir: ", input_dir)
demo_dirs = [f for f in os.listdir(input_dir) if f.endswith(".pkl")]


''' 4. Data Processing'''
def process_one_episode(root, episode_name, input_dir, id_list):
    if episode_name.startswith("."):
        return True

    # info about input
    episode_id = episode_name[13:]
    print(f"episode_name: {episode_name}, episode_id: {episode_id}")
    episode_dir = input_dir.joinpath(episode_name)

    # read rgb
    data_rgb = []
    data_rgb_time_stamps = []
    rgb_data_shapes = []
    for id in id_list:
        rgb_dir = episode_dir.joinpath("rgb_" + str(id))
        rgb_file_list = os.listdir(rgb_dir)
        rgb_file_list.sort()  # important!
        num_raw_images = len(rgb_file_list)
        img = cv2.imread(str(rgb_dir.joinpath(rgb_file_list[0])))

        rgb_data_shapes.append((num_raw_images, *img.shape))
        data_rgb.append(np.zeros(rgb_data_shapes[id], dtype=np.uint8))
        data_rgb_time_stamps.append(np.zeros(num_raw_images))

        print(f"Reading rgb data from: {rgb_dir}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = set()
            for i in range(len(rgb_file_list)):
                futures.add(
                    executor.submit(
                        image_read,
                        rgb_dir,
                        rgb_file_list,
                        i,
                        data_rgb[id],
                        data_rgb_time_stamps[id],
                    )
                )

            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError("Failed to read image!")

    # read low dim data
    data_ts_pose_fb = []
    data_robot_time_stamps = []
    data_wrench = []
    data_wrench_time_stamps = []
    print(f"Reading low dim data for : {episode_dir}")
    for id in id_list:
        json_path = episode_dir.joinpath("robot_data_" + str(id) + ".json")
        df_robot_data = pd.read_json(json_path)
        data_robot_time_stamps.append(df_robot_data["robot_time_stamps"].to_numpy())
        data_ts_pose_fb.append(np.vstack(df_robot_data["ts_pose_fb"]))

        # read wrench data
        json_path = episode_dir.joinpath("wrench_data_" + str(id) + ".json")
        df_wrench_data = pd.read_json(json_path)
        data_wrench_time_stamps.append(df_wrench_data["wrench_time_stamps"].to_numpy())
        data_wrench.append(np.vstack(df_wrench_data["wrench"]))

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


def preprocess_pose_data(pose):
    pass

def preprocess_rgb_data(image):
    pass

def preprocess_wrench_data(wrench):
    pass


for demo in demo_dirs:
    demo_file = os.path.join(input_dir, demo)
    cprint('Processing {}'.format(demo), 'green')

    with open(demo_file, 'rb') as f:
        demo = pickle.load(f)

    demo_length = len(demo['rgb_data'])
    
    for index in range(demo_length):
        obs_image = demo['rgb_data'][index]
        obs_image = cv2.cvtColor(obs_image, cv2.COLOR_BGR2RGB)
        cv2.imshow("Gopro Image", obs_image)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break
        pose_data = demo['pose_data'][index]
        print(pose_data)
        wrench_data = demo['wrench_data'][index]
        print(wrench_data)
    cv2.destroyAllWindows()




