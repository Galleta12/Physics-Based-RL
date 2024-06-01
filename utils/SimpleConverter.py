
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


class SimpleConverter(object):
    def __init__(self,file) -> None:
        self.num_bodies = len(BODY_DEFS)
        self.pos_dim = 3
        self.rot_dim =4
        self.file = file
        
        self.load_data()

        
    def load_data(self):
        with open(self.file, 'r') as fin:
            data = json.load(fin)
            #grab the frames that are inside the frames key
            self.motions =np.array(data["Frames"])

            m_shape =np.shape(self.motions)
            
            #for now this will be a numpy array
            #populate the data with nan values 
            self.data = np.full(m_shape, np.nan)
            
            #save the delta time which will be the first value
            self.dt = self.motions[0][0]
    
    def load_mocap(self):
        self.read_raw_data()
        self.convert_to_mujoco_data()
        self.create_dict_duration()
    
    def read_raw_data(self):
        self.all_states = []
        self.durations = []
        #remember the frames are the rows
        #and the col are the lenght of the frame
        self.total_time = 0.0
        for frame in self.motions:
            self.calculate_duration(frame[0])
            
            #start for the root            
            curr_idx = 1
            #up to
            offset_idx = 8
            state = {}
            
            #basically the first 7 values ignoring the first index that is dt
            #are 3 pos and 4 rot
            #get the root pos from index 1 to index 3
            state['root_pos'] = align_position(frame[curr_idx:curr_idx+3])
            #for this get from index 4 to index 7 since there are 4 positions that are the rotations quaternions
            state['root_rot'] = align_rotation(frame[curr_idx+3:offset_idx])
            #for the other joints
            
            self.align_joints(frame,curr_idx,offset_idx,state)
        
        self.all_states = self.all_states
        self.durations = np.array(self.durations)
        
    
    
      
    
    def get_global_pos_rot_joints(self,pos_matrix,rot_matrix,index,joint_name,local_quat):
        
        #get the parent rotation      
        parent_name,idx = BODY_HIERARCHY_JOINTS[joint_name]
        #the parent will be in quaternion
        parent_quat = rot_matrix[idx]
        
                
        #calculate the global rotation
        global_rotation = get_global_rotation_quat(parent_quat,local_quat)
        rot_matrix[index] = global_rotation
        
        
        
        # Save the global position
        pos_matrix[index] = self.get_global_pos(pos_matrix,idx,index,parent_quat)
        
        
         
    def get_global_pos(self,pos_matrix,idx,index,parent_rotation):
        global_pos_parent = pos_matrix[idx]
        local_pos = pos_matrix[index]
        #get the rotation in a matrix
        rotation_parent_matrix = quaternion_to_rotation_matrix(parent_rotation)
        #apply transformation
        transformation_matrix = apply_transformation(rotation_parent_matrix,global_pos_parent,local_pos)
        
        global_pos =transformation_matrix[:3]
        
        return global_pos

        
       
    
    #remember the angles are in radians
    def get_global_pos_rot_joints_axis(self,pos_matrix,rot_matrix,index,joint_name,local_angle):
        
        #get the parent rotation      
        parent_name,idx = BODY_HIERARCHY_JOINTS[joint_name]
        #the paretn will be in quaternion
        parent_quat = rot_matrix[idx]
        
        #now we need to convert the local_angle axis to quat
        local_quat = axis_angle_to_quat(np.array(JOINTS_AXIS_ONEDOF[joint_name]),local_angle)
        
        
        #calculate the global rotation
        global_rotation = get_global_rotation_quat(parent_quat,local_quat)
        
        rot_matrix[index] = global_rotation
        
        # Save the global position
        pos_matrix[index] = self.get_global_pos(pos_matrix,idx,index,parent_quat)
       
        
          
    
    def start_matrices_savers(self):
        
        xpos_matrix = np.array(list(BODY_INTIAL_XPOS_MUJOCO_XML.values()))
        xquat_matrix = np.array(list(BODY_INITIAL_XQUAT_MUJOCO_XML.values()))
        return xpos_matrix,xquat_matrix
    
    def convert_to_mujoco_data(self):
        self.data_vel = []
        self.data_pos = []
        self.data_xpos =[]
        self.data_xrot =[]
        
        #this will loop the shape row of the motions, since that is the lenght of all state
        for k in range(len(self.all_states)):
            #the dim are the rows the joint and 3 for the x,y,z pos
            #and 4 for the quaternions
            # x_pos_matrix = np.zeros((13,3))
            # x_rot_matrix = np.zeros((13,4))
            x_pos_matrix, x_rot_matrix = self.start_matrices_savers()
            
            
            #for the velocity
            tmp_vel = []
            tmp_angle = []
            #grab an all state dict, this is a dict inside a list
            state = self.all_states[k]
            #at the very beginning
            #this will be calculatino for velocity
            if k == 0:
                #for the time
                dura = self.durations[k]
            else:
                dura = self.durations[k-1]
            
            
            init_idx = 0
            offset_idx = 1
            #grab an index and populate it with dura
            #a populate it with the duration
            #k rows,
            #so basically we populate the data, row with dura, which is the delta time
            #so all the row at column one will be delta time
            self.data[k, init_idx:offset_idx] = dura
            
            # root pos indexing for grabbing the root pos
            init_idx = offset_idx
            offset_idx += 3
            #we move to the next 3 an populate it as
            #the root pos array, so on the matrix on each row, it will be
            #first delta time and then the position of the root
            self.data[k, init_idx:offset_idx] =np.array(state['root_pos'])
            
           
            #the velocity is just zero at the beginning
            if k == 0:
                tmp_vel += [0.0, 0.0, 0.0]
            else:
                #we calculate the velocity
                # the averge velocity formula, with the 1/dura to get the change of speed per unit of time
                #so this is for the root
                #remeber this will return 3 rotations
                #this is the linear velocity
                tmp_vel += ((self.data[k, init_idx:offset_idx] - self.data[k-1, init_idx:offset_idx])*1.0/dura).tolist()
            #save the pos, rember that is in quaterion representation
            tmp_angle += state['root_pos'].tolist()
            x_pos_matrix[0] = state['root_pos'].tolist()
            
            
            # root rot
            init_idx = offset_idx
            #we grab the 4 columns for the root rotations
            offset_idx += 4
            #we saved on the data matrix
            #now we compute for the root velocities
            self.data[k, init_idx:offset_idx] =np.array(state['root_rot'])
            if k == 0:
                tmp_vel += [0.0, 0.0, 0.0]
            else:
                #calculate the rotation velocity
                #so we pass the current root and the prev
                #for the root we saved the angular in quaternion format
                #this is the angular velocity
                tmp_vel += calc_rot_vel(self.data[k, init_idx:offset_idx], self.data[k-1, init_idx:offset_idx], dura)
                #print(tmp_vel)
            #saved the root rotation
            tmp_angle += state['root_rot'].tolist()
            x_rot_matrix[0] = state['root_rot'].tolist()
            
            
                #now do the same for the joints
            for index,each_joint in enumerate(BODY_JOINTS):
                init_idx = offset_idx
                #generate a tmp val for the joint
                #first grab the positions 4
                tmp_val = state[each_joint]
                
                if DOF_DEF[each_joint] == 1:
                    assert 1 == len(tmp_val)
                    offset_idx += 1
                    #grab the joints
                    self.data[k, init_idx:offset_idx] = state[each_joint]
                    if k == 0:
                        tmp_vel += [0.0]
                    #calculate the velocities
                    else:
                        tmp_vel += ((self.data[k, init_idx:offset_idx] - self.data[k-1, init_idx:offset_idx])*1.0/dura).tolist()
                    tmp_angle += state[each_joint].tolist()
                    
                    
                    #since here it start from,chest but we want to start from
                    #the root, Body joints start from chest
                    #pos_matrix[index+1] = list(euler_tuple)
                
                    self.get_global_pos_rot_joints_axis(x_pos_matrix,x_rot_matrix,index+1,each_joint,state[each_joint])
                    
                    
                    ##later
                    #x_pos_matrix[index] = 
                    
                #do the same for the joints with more dofs
                elif DOF_DEF[each_joint] == 3:
                    assert 4 == len(tmp_val)
                    offset_idx += 4
                    self.data[k, init_idx:offset_idx] = state[each_joint]
                    if k == 0:
                        tmp_vel += [0.0, 0.0, 0.0]
                    else:
                        tmp_vel += calc_rot_vel(self.data[k, init_idx:offset_idx], self.data[k-1, init_idx:offset_idx], dura)
                    #save the quat on each joint
                    quat = state[each_joint]
                    
                    
                    #index plus one since we start from the chest, but the matrix start from the root
                    self.get_global_pos_rot_joints(x_pos_matrix,x_rot_matrix,index+1,each_joint,quat)
                    
            
                    #change the format for the transformation library
                    #the last element the real
                    quat = np.array([quat[1], quat[2], quat[3], quat[0]])
                    #get an euler from the quaternion
                    #since the joints are not in quaternion but in x,y,z axis
                    euler_tuple = euler_from_quaternion(quat, axes='rxyz')
                    # add it to the tmp angle
                    tmp_angle += list(euler_tuple)
            
            
            #this is qvel
            self.data_vel.append(np.array(tmp_vel))
            #this is qpos
            self.data_pos.append(np.array(tmp_angle))

            self.data_xpos.append(x_pos_matrix)
            self.data_xrot.append(x_rot_matrix)

        
            
    def align_joints(self,frame,curr_idx,offset_idx,state):
        for joint in BODY_JOINTS_IN_DP_ORDER:
            #set limits of indext what we will grab
            curr_idx = offset_idx
            #degree of freedom
            dof = DOF_DEF[joint]
            #the offset is like a pointer we increment it
            if dof == 1:
                offset_idx += 1
                #save the frame in the state dict
                #so grab from the current idx to the offset plus 1
                #since this is a hinge joint we dont
                #need a to align
                state[joint] = frame[curr_idx:offset_idx]
            elif dof ==3:
                #since we want the rotations to be saved
                #remeber that the rotations are 4
                offset_idx +=4
                #we align the rotations
                #we dont need to align the pos of the joinst
                #since is relative to the parent
                #so as long as the root is align that is enough
                state[joint] = align_rotation(frame[curr_idx:offset_idx])
        self.all_states.append(state)
    
    #we pass the first frame
    def calculate_duration(self,frame):
        duration = frame
        frame = self.total_time
        self.total_time += duration
        self.durations.append(duration)
    
    
    def create_dict_duration(self):
        #create a list, where the first two values are the start time of the frame
        #and the second value is the duration the delta time

        #comulative sum list
        cummulative_durations = jp.cumsum(self.durations)
        #re format it since, the first index starts at time 0 so we can get rid of the last index
        cummulative_durations = jp.concatenate([jp.array([0.0]), cummulative_durations[:-1]])
        #so the keys are the index of each frame, then we will save an array,
        #where the first value, is the comulative duration and the second is the trajecotry duration

        self.duration_dict = {i: [float(cummulative_durations[i]), float(self.durations[i])]
                        for i in range(len(self.durations))}
        
        
            
        
    # def load_mocap(self)


if __name__ == "__main__":

    file_path = "motions/humanoid3d_punch.txt"
    s = SimpleConverter(file_path)
    s.load_mocap()
    
    #testing the global pos
    #print(s.data_xpos[0:3])
    print('rotation')
    print(s.data_xrot[0:3])
    
    
    
    
    # print("motino shape", s.motions.shape)
    # #shape of the row of the motions is the len of the durations
    # print("durations", s.durations)
    # print("duration shape", s.durations.shape)
    # #lenght of the motion, in the case of the walk is 1.26 sec
    # print("time", s.total_time)
    # print("all state root pos", s.all_states[-1]['chest'])
    # # print("len state", len(s.all_states))



    # np.set_printoptions(edgeitems=30, linewidth=1000, 
    #     formatter=dict(float=lambda x: "%.3g" % x))



    # print("data matrix", s.data[-1])
    # print("data shape", s.data.shape)

    # # print('qvel', s.data_pos)
    # # print('qpos', s.data_vel)

    # print('qvel', len(s.data_pos))
    # print('qpos', len(s.data_vel))





