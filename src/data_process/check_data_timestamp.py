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



