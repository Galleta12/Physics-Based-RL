
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


JOINTS_AXIS_ONEDOF ={'right_elbow':[0,-1,0],
                     'left_elbow':[0,-1,0],
                    'right_knee':[0,-1,0],
                     'left_knee':[0,-1,0]}



#the index is the index in the dict or on the bodies list of the parent joint
BODY_HIERARCHY_JOINTS = {'root': [None, 0],
                        'chest': ['root', 0],
                        'neck': ['chest', 1],
                        'right_shoulder': ['chest', 1],
                        'right_elbow': ['right_shoulder', 3],
                        'left_shoulder': ['chest', 1],
                        'left_elbow': ['left_shoulder', 5],
                        'right_hip': ['root', 0],
                        'right_knee': ['right_hip', 7],
                        'right_ankle': ['right_knee', 8],
                        'left_hip': ['root', 0],
                        'left_knee': ['left_hip', 10],
                        'left_ankle': ['left_knee', 11]}


BODY_INTIAL_XPOS_MUJOCO_XML={'root': [0. , 0. , 0.9],
 'chest': [0. ,0. ,0.236151],
 'neck': [0. ,0. ,0.223894],
 'right_shoulder': [-0.02405, -0.18311 ,0.2435],
 'right_elbow': [0., 0., -0.274788],
 'left_shoulder': [-0.02405, 0.18311, 0.2435],
 'left_elbow': [0., 0., -0.274788],
 'right_hip': [ 0., -0.084887, 0.  ],
 'right_knee': [ 0., 0., -0.421546],
 'right_ankle': [ 0, 0, -0.40987],
 'left_hip': [0., 0.084887, 0.   ],
 'left_knee': [0., 0., -0.421546],
 'left_ankle': [0., 0., -0.40987]}


# BODY_INTIAL_XPOS_MUJOCO_XML={'root': [0. , 0. , 0.9],
#  'chest': [0.      , 0.      , 1.136151],
#  'neck': [0.      , 0.      , 1.360045],
#  'right_shoulder': [-0.02405 , -0.18311 ,  1.379651],
#  'right_elbow': [-0.02405  , -0.18311  ,  1.1048629],
#  'left_shoulder': [-0.02405 ,  0.18311 ,  1.379651],
#  'left_elbow': [-0.02405  ,  0.18311  ,  1.1048629],
#  'right_hip': [ 0.      , -0.084887,  0.9     ],
#  'right_knee': [ 0.        , -0.084887  ,  0.47845396],
#  'right_ankle': [ 0.        , -0.084887  ,  0.06858397],
#  'left_hip': [0.      , 0.084887, 0.9     ],
#  'left_knee': [0.        , 0.084887  , 0.47845396],
#  'left_ankle': [0.        , 0.084887  , 0.06858397]}


#it start with the identity quaternion
BODY_INITIAL_XQUAT_MUJOCO_XML={'root': [1., 0., 0., 0.],
 'chest': [1., 0., 0., 0.],
 'neck': [1., 0., 0., 0.],
 'right_shoulder': [1., 0., 0., 0.],
 'right_elbow': [1., 0., 0., 0.],
 'left_shoulder': [1., 0., 0., 0.],
 'left_elbow': [1., 0., 0., 0.],
 'right_hip': [1., 0., 0., 0.],
 'right_knee': [1., 0., 0., 0.],
 'right_ankle': [1., 0., 0., 0.],
 'left_hip': [1., 0., 0., 0.],
 'left_knee': [1., 0., 0., 0.],
 'left_ankle': [1., 0., 0., 0.]}
    
#function to select an specific joint
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


def move_only_arm_vel(data_vel_mocap):
    # Create a copy of the input array
    new_data = jp.copy(data_vel_mocap)
    
    
    
    # Indices of the right arm
    idx_right_arm = jp.array([13, 14, 15, 16])
    
    
    # Step 2: Set all values to zero except those at the indices in idx_right_arm
    mask = jp.ones(new_data.shape[1], dtype=bool)
    mask = mask.at[idx_right_arm].set(False)
    
    new_data = jp.where(mask, 0, new_data)
    return new_data



def move_only_arm(data_pos_mocap):
    # Create a copy of the input array
    new_data = jp.copy(data_pos_mocap)
    
    # Define data for root
    data_for_root = jp.array([0. , 0. , 0.9, 1. , 0. , 0. , 0.])
    
    # Indices of the right arm
    idx_right_arm = jp.array([13, 14, 15, 16])
    
    # Step 1: Replace the first 7 values in each row with data_for_root
    #new_data = new_data.at[:, :7].set(data_for_root)
    
    # Step 2: Set all values to zero except those at the indices in idx_right_arm
    mask = jp.ones(new_data.shape[1], dtype=bool)
    mask = mask.at[idx_right_arm].set(False)
    
    new_data = jp.where(mask, 0, new_data)
    new_data = new_data.at[:, :7].set(data_for_root)
    return new_data

