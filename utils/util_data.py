
import jax
from jax import numpy as jp
BODIES = ['world',
'root',
'chest',
'neck',
'right_shoulder',
'right_elbow',
'left_shoulder',
'left_elbow',
'right_hip',
'right_knee',
'right_ankle',
'left_hip',
'left_knee',
'left_ankle']


#in order for the mocap
BODY_JOINTS_IN_DP_ORDER = ["chest", "neck", "right_hip", "right_knee",
                        "right_ankle", "right_shoulder", "right_elbow", "left_hip", 
                        "left_knee", "left_ankle", "left_shoulder", "left_elbow"]


#right elbow, left elbow, right knee and left_nee one 1 D0Fs
BODY_JOINTS = ["chest", "neck", "right_shoulder", "right_elbow", 
            "left_shoulder", "left_elbow", "right_hip", "right_knee", 
            "right_ankle", "left_hip", "left_knee", "left_ankle"]

DOF_DEF = {"root": 3, "chest": 3, "neck": 3, "right_shoulder": 3, 
           "right_elbow": 1, "right_wrist": 0, "left_shoulder": 3, "left_elbow": 1, 
           "left_wrist": 0, "right_hip": 3, "right_knee": 1, "right_ankle": 3, 
           "left_hip": 3, "left_knee": 1, "left_ankle": 3}

PARAMS_KP_KD = {"chest": [1000, 100], "neck": [100, 10], "right_shoulder": [400, 40], "right_elbow": [300, 30], 
        "left_shoulder": [400, 40], "left_elbow": [300, 30], "right_hip": [500, 50], "right_knee": [500, 50], 
        "right_ankle": [400, 40], "left_hip": [500, 50], "left_knee": [500, 50], "left_ankle": [400, 40]}

        
BODY_DEFS = ["root", "chest", "neck", "right_hip", "right_knee", 
             "right_ankle", "right_shoulder", "right_elbow", "right_wrist", "left_hip", 
             "left_knee", "left_ankle", "left_shoulder", "left_elbow", "left_wrist"]      
        
#function to select an specific joint in this case we will work with the left shoulder
#joint

def get_joint_index(model,name_body,axis):
    index = model.body(name_body).jntadr[0]
    
    if axis == 'Y':
        index += 1
    elif axis == 'Z':
        index += 2
    
    return  index + 6


#so for vel the root joint will have 6 dofs that are 
#3 from the linear velocty and 3 from the angular velocity
def get_vel_indx(model,name_body, axis):
    index = model.body(name_body).jntadr[0]
    if axis == 'Y':
        index += 1
    elif axis == 'Z':
        index += 2
    return  index + 5


def generate_kp_kd_gains():
    kp, kd = [], []
    for each_joint in BODY_JOINTS:
        kp += [PARAMS_KP_KD[each_joint][0] for _ in range(DOF_DEF[each_joint])]
        kd += [PARAMS_KP_KD[each_joint][1] for _ in range(DOF_DEF[each_joint])]
    
    return jp.array(kp), jp.array(kd)        

#remember that nv is 34-6 is 28
#same size as the actuators that are for the kp,kd gains
def get_actuator_indx(model,name,axis):
    index = model.body(name).jntadr[0]
    if axis == 'Y':
        index +=1
    elif axis == 'Z':
        index +=2
    
    return index - 1