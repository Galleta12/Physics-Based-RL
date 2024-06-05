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
from jax import jit
from functools import partial
from utils import util_data
from utils import SimpleConverter
from some_math.math_utils_jax import *
from copy import deepcopy



from .agent_template import HumanoidTemplate
from .losses import *


import sys
import os
from .losses import *
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from some_math.rotation6D import quaternion_to_rotation_6d
from some_math.math_utils_jax import *
from utils.SimpleConverter import SimpleConverter
from utils.util_data import generate_kp_kd_gains
import some_math.quaternion_diff as diff_quat


class HumanoidEvalTemplate(HumanoidTemplate):
    def __init__(self, reference_data: SimpleConverter, 
                 model_path, 
                 args, 
                 **kwargs):
        super().__init__(reference_data, 
                         model_path, 
                         args, 
                         **kwargs)
        

    #
    def step(self, state: State, action: jp.ndarray) -> State:
        initial_idx = state.metrics['step_index']
        current_step_inx =  jp.asarray(initial_idx, dtype=jp.int32) + 1
        
        current_state_ref = self.set_ref_state_pipeline(current_step_inx)
    
        #current qpos and qvel for the torque    
        qpos = state.pipeline_state.q
        qvel = state.pipeline_state.qd

        timeEnv = state.pipeline_state.time

        target_angles = action * jp.pi * 1.2
        #target_angles = action 
        
        torque = self.pd_function(target_angles,self.sys,state,qpos,qvel,
                                 self.kp_gains,self.kd_gains,timeEnv,self.sys.dt) 
        
        data = self.pipeline_step(state.pipeline_state,torque)
        #data = self.pipeline_init(current_state_ref.qpos, current_state_ref.qvel)
        
        
        #get the observations
        obs = self._get_obs(data, current_step_inx)
        
        reward = self.compute_rewards_diffmimic(data,current_state_ref)
        
        fall = jp.where(data.qpos[2] < 0.2, jp.float32(1), jp.float32(0))
        fall = jp.where(data.qpos[2] > 1.7, jp.float32(1), fall)
        
        global_pos_state = data.x.pos
        
        global_pos_ref = current_state_ref.x.pos
              
        pose_error=loss_l2_relpos(global_pos_state, global_pos_ref)
        
        state.metrics.update(
            step_index=current_step_inx,
            pose_error=pose_error,
            fall=fall,
        )
        
        
        
        return state.replace(
            pipeline_state= data, obs=obs, reward=reward, done=state.metrics['fall']
        )
        
    
    
    
    #class for testing step_custom where we just check the pd controller
    def step_custom(self, state: State, action: jp.ndarray) -> State:
        initial_idx = state.metrics['step_index']
        current_step_inx =  jp.asarray(initial_idx, dtype=jp.int32) + 1
        #jax.debug.print("current step idx{}", current_step_inx)
        current_state_ref = self.set_ref_state_pipeline(current_step_inx)
    
        #current qpos and qvel for the torque    
        qpos = state.pipeline_state.q
        qvel = state.pipeline_state.qd

        timeEnv = state.pipeline_state.time
        #target_angles = action * jp.pi * 1.2
        
        
        #jax.debug.print("time{}",timeEnv)
        torque = self.pd_function(current_state_ref.qpos[7:],self.sys,state,qpos,qvel,
                                 self.kp_gains,self.kd_gains,timeEnv,self.sys.dt) 
        data = self.pipeline_step(state.pipeline_state,torque)
        
        
        #get the observations
        obs = self._get_obs(data, current_step_inx)
        
        reward = self.compute_rewards_diffmimic(data,current_state_ref)
        
        fall = jp.where(data.qpos[2] < 0.2, jp.float32(1), jp.float32(0))
        fall = jp.where(data.qpos[2] > 1.7, jp.float32(1), fall)
        
        global_pos_state = data.x.pos
        
        global_pos_ref = current_state_ref.x.pos
              
        pose_error=loss_l2_relpos(global_pos_state, global_pos_ref)
        
        state.metrics.update(
            step_index=current_step_inx,
            pose_error=pose_error,
            fall=fall,
        )
        
        return state.replace(
            pipeline_state= data, obs=obs, reward=reward, done=state.metrics['fall']
        )
        
        
    
    
    #standard obs this will changed on other derived
    #classes from this
    def _get_obs(self, data: base.State, step_idx: jp.ndarray)-> jp.ndarray:
          
        current_step_inx =  jp.asarray(step_idx, dtype=jp.int32)
        
        
        relative_pos , local_rotations,local_vel,local_ang = self.convert_local(data)
        #relative_pos , local_rotations,local_vel,local_ang = self.convertLocaDiff(data)
        
        relative_pos = relative_pos[1:]
        #q_relative_pos,q_local_rotations, q_local_vel, q_local_ang = self.convertLocaDiff(data)
        
        #convert quat to 6d root
        rot_6D= quaternion_to_rotation_6d(local_rotations)
        
        
        # jax.debug.print("pos mine{}",relative_pos)
        # jax.debug.print("rot mine {}", local_rotations)
        # jax.debug.print("vel mine{}", local_vel)
        # jax.debug.print("ang mine{}", local_ang)
        
        # jax.debug.print("pos q{}",q_relative_pos)
        # jax.debug.print("rot q {}", q_local_rotations)
        # jax.debug.print("vel q{}", q_local_vel)
        # jax.debug.print("ang q{}", q_local_ang)
        
        
        #phi
        phi = ( current_step_inx % self.cycle_len) / self.cycle_len
        
        phi = jp.asarray(phi)
        
        
        #jax.debug.print("phi{}", phi)
        return jp.concatenate([relative_pos.ravel(),rot_6D.ravel(),
                               local_vel.ravel(),local_ang.ravel(),phi[None]])


    
    
    def convert_local(self,data: base.State):
        
        root_pos = data.x.pos[0]
        root_quat_inv =  math.quat_inv(data.x.rot[0])
        
        #inverted_root= math.quat_mul(root_quat_inv,data.x.rot[0])
        
        #apply root-relative transformation
        def transform_to_root_local(pos, rot,vel, ang):
            local_pos = pos - root_pos
            local_rot = math.quat_mul(root_quat_inv, rot)
            local_vel = math.rotate(vel,math.quat_inv(data.x.rot[0]))
            local_ang = math.rotate(ang,math.quat_inv(data.x.rot[0]))
            
            return local_pos, local_rot, local_vel,local_ang
        #Applying the transformation to all positions and orientations
        #this is relative to the root
        local_positions, local_rotations,local_vel,local_ang = jax.vmap(transform_to_root_local)(data.x.pos, data.x.rot,data.xd.vel,data.xd.ang)
        #get rid of the zero  root pos
        #relative_pos = local_positions[1:]
        
        return  local_positions,local_rotations,local_vel,local_ang

    
    
    def convertLocaDiff(self, data:base.State):
        pos, rot, vel, ang = data.x.pos, data.x.rot, data.xd.vel, data.xd.ang

        root_pos = pos[0] 

        relative_pos = pos - root_pos
        
        #re-arrange all elements of the root quaterion as as xyzw
        rot_xyzw_raw = rot[:, [1, 2, 3, 0]]
        
        #normalize quaternion to make it unit magnitude
        rot_xyzw = diff_quat.quat_normalize(rot_xyzw_raw)
        
        root_rot_xyzw = diff_quat.quat_normalize(rot_xyzw[0])  
        
        #root_inverse = quat_inverse(root_rot_xyzw)
        normalized_rot_xyzw = diff_quat.quat_mul_norm(diff_quat.quat_inverse(root_rot_xyzw), rot_xyzw)
        normalized_pos = diff_quat.quat_rotate(diff_quat.quat_inverse(root_rot_xyzw), relative_pos)
        normalized_vel = diff_quat.quat_rotate(diff_quat.quat_inverse(root_rot_xyzw), vel)
        normalized_ang = diff_quat.quat_rotate(diff_quat.quat_inverse(root_rot_xyzw), ang)

        normalized_rot = normalized_rot_xyzw[:, [3, 0, 1, 2]]
        
        return normalized_pos, normalized_rot, normalized_vel,normalized_ang
    
    
    
    
    
    
    
    def _com(self, pipeline_state: base.State) -> jax.Array:
        inertia = self.sys.link.inertia
        
        mass_sum = jp.sum(inertia.mass)
        x_i = pipeline_state.x.vmap().do(inertia.transform)
        com = (
            jp.sum(jax.vmap(jp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
        )
        return com, inertia, mass_sum, x_i  # pytype: disable=bad-return-type  # jax-ndarray
