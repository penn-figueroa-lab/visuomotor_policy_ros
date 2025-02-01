import os
import zarr
import pickle
import tqdm
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import torchvision
from termcolor import cprint
import re
import time


import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import torchvision
import socket
import pickle


expert_data_path = "/home/yihan/Documents/ft_sensor_ws/src/visuomotor_policy_ros/data/default_task_expert"
save_data_path = "/home/yihan/Documents/ft_sensor_ws/src/visuomotor_policy_ros/data/default_task_dataset"
demo_dirs = [f for f in os.listdir(expert_data_path) if f.endswith(".pkl")]

image_arrays = []
pose_arrays = []
wrench_arrays = []


if os.path.exists(save_data_path):
    cprint('Data already exists at {}'.format(save_data_path), 'red')
    cprint("If you want to overwrite, delete the existing directory first.", "red")
    cprint("Do you want to overwrite? (y/n)", "red")
    user_input = 'y'
    if user_input == 'y':
        cprint('Overwriting {}'.format(save_data_path), 'red')
        os.system('rm -rf {}'.format(save_data_path))
    else:
        cprint('Exiting', 'red')
        exit()
os.makedirs(save_data_path, exist_ok=True)

def preprocess_pose_data(pose):
    pass

def preprocess_rgb_data(image):
    pass

def preprocess_wrench_data(wrench):
    pass


for demo in demo_dirs:
    demo_file = os.path.join(expert_data_path, demo)
    cprint('Processing {}'.format(demo), 'green')

    with open(demo_file, 'rb') as f:
        demo = pickle.load(f)

    demo_length = len(demo['rgb_data'])
    
    for index in range(demo_length):
        obs_image = demo['rgb_data'][index]
        pose_data = demo['pose_data'][index]
        wrench_data = demo['wrench_data'][index]





