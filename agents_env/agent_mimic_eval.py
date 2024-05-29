from datetime import datetime
import functools
from IPython.display import HTML
import jax
from jax import numpy as jp
import numpy as np
from typing import Any, Dict, Sequence, Tuple, Union
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model
from etils import epath
from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from jax import vmap
from jax import lax
from utils import util_data

from copy import deepcopy


from .agent_env_template import HumanoidDiff
from .losses import *

import sys
import os
# Append the parent directory of both utils and some_math to sys.path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)



from some_math.rotation6D import quaternion_to_rotation_6d
from some_math.math_utils_jax import *


class HumanoidEnvTrainEval(HumanoidDiff):
    def __init__(self, reference_trajectory_qpos, 
                 reference_trajectory_qvel, 
                 duration_trajectory,
                 dict_duration, 
                 model_path,
                 kp_gains,
                 kd_gains,
                 reference_x_pos,
                 reference_x_rot,
                 **kwargs):
        super().__init__(reference_trajectory_qpos, 
                         reference_trajectory_qvel, 
                         duration_trajectory, 
                         dict_duration, 
                         model_path, **kwargs)
        self.kp__gains = kp_gains
        self.kd__gains = kd_gains
        #the row lenght is the number of frames of the motion
        #thus it is the lenght, it will be the same with qvel
        self.rollout_lenght = reference_trajectory_qpos.shape[0]
        #I want to save another instance of the model for the refernece 
        self.sys_reference = deepcopy(self.sys)
        self.reference_x_pos = reference_x_pos
        self.reference_x_rot = reference_x_rot

        #for now this will be hardcode
        self.rot_weight =  0.5
        self.vel_weight =  0.01
        self.ang_weight =  0.01
        self.reward_scaling= 0.02
        #for now it will be the same size
        self.cycle_len = reference_trajectory_qpos.shape[0]
    
    #set pd callback
    def set_pd_callback(self,pd_control):
        self.pd_function = pd_control
    
    
    
    def get_reference_state(self,step_index):
        ref_qp = self.reference_trajectory_qpos[step_index]
        ref_qv = self.reference_trajectory_qvel[step_index]
        #now I will return a state depending on the index and the reference trajectory
        return self.pipeline_init(ref_qp,ref_qv)
    
    
    
    def reset(self, rng: jp.ndarray) -> State:
        
        #set this as zero
        reward, done, zero = jp.zeros(3)
        #start at the initial state
        
        #data = self.get_reference_state(0)
        
        
        qvel = jp.zeros(self.sys.nv)
        qpos =  self.sys.qpos0
        data = self.pipeline_init(qpos,qvel) 
        
        metrics = {'step_index': 0, 'pose_error': zero, 'fall': zero}
        obs = self._get_obs(data, 0)
        
        #jax.debug.print("obs shape{}", obs.shape)
        #size 193
        state = State(data, obs, reward, done, metrics)
           
        return state
    
    def _get_obs(self, data: mjx.Data, step_idx: jp.ndarray)-> jp.ndarray:
          
        current_step_inx =  jp.asarray(step_idx, dtype=jp.int32)
        #we take out the first index that is the world pos
        current_xpos = data.xpos[1:]
        current_xrot = data.xquat[1:]
        #get rid of the first index that is the root, we just want
        #pos relative to the root, thus the root will become zero
        relative_pos = (current_xpos - current_xpos[0])[1:].ravel()
        #jax.debug.print("relative shape{}", relative_pos.shape)
        
        #this is already in quat form
        current_qpos_root = data.qpos[3:7]
        #qpos of the joins this are scale values, since they are hinge joints
        current_qpos_joints = data.qpos[7:] 
        #now I will convert them into quaterions
        #first the joints that are onedofs, thus axis angle to quaterion
        hinge_quat = self.hinge_to_quat(current_qpos_joints)
        #now to get a 13x4 quaterion, we combine all the links
    
        local_quat = self.local_quat(current_qpos_root,current_xrot,hinge_quat,self.one_dofs_joints_idx,self.link_types_array_without_root)
        
        #now we convert it to a 6D matrix representation
        local_rot_6D= quaternion_to_rotation_6d(local_quat).ravel()
        
        # #remeber for now we have the linear vel of the root
        # linear_vel = data.qvel[0:3]
        # angular_vel = data.qvel[3:]
        # jax.debug.print("linear vel{}", linear_vel.shape)
        # jax.debug.print("angular vel{}", angular_vel.shape)
        cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])        
        #get the phi value 
        phi = ( current_step_inx% self.cycle_len) / self.cycle_len
        phi = jp.asarray(phi)
        #in theory it is mutiable if we do concatenate [] instead of
        #() since, one is a list and the other a tuple
        return jp.concatenate([relative_pos,local_rot_6D,cvel.vel.ravel(),cvel.ang.ravel(),phi[None]])
   
        #just with a custom target but not selected joints
    
    
     
    def set_ref_state_pipeline(self,step_index):
        ref_qp = self.reference_trajectory_qpos[step_index]
        ref_qv = self.reference_trajectory_qvel[step_index]
        #now I will return a state depending on the index and the reference trajectory
        return self._pipeline.init(self.sys_reference, ref_qp, ref_qv, self._debug)
        
    
    
    
    def step(self, state: State, action: jp.ndarray,
                                           custom_target,time) -> State:
        
        initial_idx = state.metrics['step_index']
        current_step_inx =  jp.asarray(initial_idx, dtype=jp.int32)
                
        current_state_ref = self.set_ref_state_pipeline(current_step_inx)
            
        #current qpos and qvel for the torque    
        qpos = state.pipeline_state.q
        qvel = state.pipeline_state.qd
        
        
        #this will be modified by a one without custom target   
        #torque = self.pd_function(custom_target,self.sys,state,qpos,qvel,
        #                         self.kp__gains,self.kd__gains,time,self.sys.dt) 
        timeEnv = state.pipeline_state.time
        #jax.debug.print("timeEnv: {}",timeEnv)
        
        torque = self.pd_function(custom_target,self.sys,state,qpos,qvel,
                                 self.kp__gains,self.kd__gains,time,self.sys.dt) 
        
        
        data = self.pipeline_step(state.pipeline_state,torque)
        
        #first get the values, values of the current state
        global_pos_state = data.x.pos
        global_rot_state = quaternion_to_rotation_6d(data.x.rot)
        global_vel_state = data.xd.vel
        global_ang_state = data.xd.ang
        #now for the reference trajectory
        global_pos_ref = self.reference_x_pos[current_step_inx]
        global_rot_ref = quaternion_to_rotation_6d(self.reference_x_rot[current_step_inx])
        global_vel_ref = current_state_ref.xd.vel
        global_ang_ref = current_state_ref.xd.ang
        
        reward = -1 * (mse_pos(global_pos_state, global_pos_ref) +
               self.rot_weight * mse_rot(global_rot_state, global_rot_ref) +
               self.vel_weight * mse_vel(global_vel_state, global_vel_ref) +
               self.ang_weight * mse_ang(global_ang_state, global_ang_ref)
               ) * self.reward_scaling
        
        #jax.debug.print("rewards: {}",reward)
        
        #here I will do the fall
        #on the z axis
        fall = jp.where(data.qpos[2] < 0.2, jp.float32(1), jp.float32(0))
        fall = jp.where(data.qpos[2] > 1.7, jp.float32(1), fall)
        
        #jax.debug.print("fall: {}",fall)
        #jax.debug.print("qpos: {}",data.qpos[0:3])
        
        
        #get the observations
        obs = self._get_obs(data, current_step_inx)
        
        #increment the step index to know in which episode and wrap
        #this is for cyclic motions but I may need to fix it
        next_step_index = (current_step_inx + 1) % self.rollout_lenght
        
        state.metrics.update(
            step_index=next_step_index,
            pose_error=loss_l2_relpos(global_pos_state, global_pos_ref),
            fall=fall,
        )
        
        
        return state.replace(
            pipeline_state= data, obs=obs, reward=reward
        )
        
        