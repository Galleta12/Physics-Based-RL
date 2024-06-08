

import json
import numpy as np
import sys
import os

# Append the parent directory of both utils and some_math to sys.path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# from util_data import BODY_DEFS, BODY_JOINTS_IN_DP_ORDER, DOF_DEF, BODY_JOINTS
# from transformations import euler_from_quaternion
# from some_math import *

from .util_data import BODY_DEFS, BODY_JOINTS_IN_DP_ORDER, DOF_DEF, BODY_JOINTS,BODY_HIERARCHY_JOINTS,BODY_INTIAL_XPOS_MUJOCO_XML,BODY_INITIAL_XQUAT_MUJOCO_XML, JOINTS_AXIS_ONEDOF

#from util_data import BODY_DEFS, BODY_JOINTS_IN_DP_ORDER, DOF_DEF, BODY_JOINTS,BODY_HIERARCHY_JOINTS,BODY_INTIAL_XPOS_MUJOCO_XML,BODY_INITIAL_XQUAT_MUJOCO_XML

from some_math.transformations import euler_from_quaternion
from some_math.math_utils import *


import jax
from jax import numpy as jp





class DataConverter(object):
    def __init__(self,file,duplicate) -> None:
        self.num_bodies = len(BODY_DEFS)
        self.file = file
        self.duplicate = duplicate
        
        self.create_more()
        
    
    def create_more(self):
        with open(self.file, 'r') as fin:
            data = json.load(fin)
            #grab the frames that are inside the frames key
            self.motions =np.array(data["Frames"])
            
            self.dt = self.motions[0][0]
        
        
        #prev_tim = self.motions[-2][0]
        self.motions[-1][0] = self.dt
        
        # Extract the original dt and the displacement for x and z components
        original_dt = self.motions[0][0]
        displacement_xz = self.motions[-1, [1, 3]] - self.motions[0, [1, 3]]

        # Initialize duplicated data
        duplicated_data = np.tile(self.motions, (self.duplicate, 1))
        
        for i in range(1, self.duplicate):
            start_idx = i * len(self.motions)
            end_idx = (i + 1) * len(self.motions)
            duplicated_data[start_idx:end_idx, 1] += i * displacement_xz[0]  # Adjust x
            duplicated_data[start_idx:end_idx, 3] += i * displacement_xz[1]  # Adjust z

        # Ensure the dt for the last frame is set back to zero
        duplicated_data[-1, 0] = 0.0

        output_data = {
            "Loop": data["Loop"],
            "Frames": duplicated_data.tolist()
        }

        output_file = self.file.replace(".txt", "_duplicated.txt")
        with open(output_file, 'w') as fout:
            json.dump(output_data, fout, indent=4)


if __name__ == "__main__":
    file_path = "motions/humanoid3d_punch.txt"
    s = DataConverter(file_path,2)
    print(s.motions)
    



