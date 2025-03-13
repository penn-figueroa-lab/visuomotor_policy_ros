import os

import pickle

import numpy as np
import torch
# import pytorch3d.ops as torch3d_ops
import torchvision

import numpy as np
import torch
# import pytorch3d.ops as torch3d_ops
import torchvision

import pickle
import cv2

import sys




input_dir = "/Users/yihanli/Documents/Energy_MPPI/visuomotor_policy_ros/data"
sys.path.append(input_dir)
demo_dirs = [f for f in os.listdir(input_dir) if f.endswith(".pkl")]

for demo in demo_dirs:
    demo_file = os.path.join(input_dir, demo)

    with open(demo_file, 'rb') as f:
        demo_data = pickle.load(f)
        
    first_frame = demo_data['rgb_data'][0]
    height, width, channels = first_frame.shape
    demo_length = len(demo_data['rgb_data'])
    output_video_path = os.path.join(input_dir, "/Users/yihanli/Documents/Energy_MPPI/visuomotor_policy_ros/data/demonstration.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use 'XVID' for .avi files
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))  # 30 FPS

    
    for index in range(demo_length):
        obs_image = demo_data['rgb_data'][index]
        out.write(obs_image)
        cv2.imshow("Gopro Image", obs_image)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break
        pose_data = demo_data['pose_data'][index]
        print(pose_data)
        wrench_data = demo_data['wrench_data'][index]
        print(wrench_data)
    out.release()
cv2.destroyAllWindows()
