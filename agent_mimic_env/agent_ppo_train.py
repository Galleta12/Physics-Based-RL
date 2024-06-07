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
from brax.mjx.pipeline import _reformat_contact
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
from etils import epath
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

from .agent_template import HumanoidTemplate
from .losses import *


class HumanoidPPOENV(HumanoidTemplate):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline_step = jax.checkpoint(self.pipeline_step, 
                policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
        args = kwargs.pop('args')
        
        
        self.w_p =  args.deep_mimic_reward_weights.w_p
        self.w_v =  args.deep_mimic_reward_weights.w_v
        self.w_e =  args.deep_mimic_reward_weights.w_e
        self.w_c= args.deep_mimic_reward_weights.w_c    
        
        self.w_pose =  args.deep_mimic_weights_factors.w_pose
        self.w_angular =  args.deep_mimic_weights_factors.w_angular
        self.w_efector =  args.deep_mimic_weights_factors.w_efector
        self.w_com= args.deep_mimic_weights_factors.w_com   

        
        
        
        
    
    def reset(self, rng: jp.ndarray) -> State:
        
        #set this as zero
        reward, done, zero = jp.zeros(3)
        #start at the initial state
        data = self.get_reference_state(0)       
        # qvel = jp.zeros(self.sys.nv)
        # qpos =  self.sys.qpos0
        # data = self.pipeline_init(qpos,qvel) 
        
        metrics = {'step_index': 0, 'pose_error': zero, 'fall': zero}
        obs = self._get_obs(data, 0)    
        #jax.debug.print("obs: {}",obs.shape)
        #the obs should be size 193?

        state_info = {
            'rng': rng,
            'steps': 0.0,
            'reward_tuple': {
                'reference_quaternions': 0.0,
                'reference_angular': 0.0,
                'reference_end_effector': 0.0,
                'reference_com': 0.0
            },
    
        }
        
        #metrics = {}
        #save the infor on the metrics
        for k in state_info['reward_tuple']:
            metrics[k] = state_info['reward_tuple'][k]
        
        state = State(data, obs, reward, done, metrics,state_info)
           
        return jax.lax.stop_gradient(state)
    
    
    def step(self, state: State, action: jp.ndarray) -> State:
        
        #perform the action of the policy
        #current qpos and qvel for the torque    
        qpos = state.pipeline_state.q
        qvel = state.pipeline_state.qd

        timeEnv = state.pipeline_state.time

        #deltatime of physics
        dt = self.sys.opt.timestep
        
            
        #target_angles = action * jp.pi * 1.2
        #exclude the root
        target_angles = self._inital_pos[7:] +(action * jp.pi * 1.2)
        #target_angles = action 
        
        torque = self.pd_function(target_angles,self.sys,state,qpos,qvel,
                                 self.kp_gains,self.kd_gains,timeEnv,dt) 
        
        data = self.pipeline_step(state.pipeline_state,torque)
                
        index_new =jp.array(state.info['steps']%self.cycle_len, int)
        
        #jax.debug.print("new idx: {}",index_new)
        
        
        
        initial_idx =  state.metrics['step_index']
        current_step_inx =  jp.asarray(initial_idx, dtype=jp.int64) + 1
        
        current_state_ref = self.set_ref_state_pipeline(current_step_inx,state.pipeline_state)

        fall=0.0
        fall = jp.where(data.qpos[2] < 0.3, 1.0, fall)
        fall = jp.where(data.qpos[2] > 1.7, 1.0, fall)
        
        #get the observations
        obs = self._get_obs(data, current_step_inx)
        
        reward, reward_tuple = self.compute_rewards_deepmimic(data,current_state_ref)
        
        #state mangement
        state.info['reward_tuple'] = reward_tuple
        
        for k in state.info['reward_tuple'].keys():
            state.metrics[k] = state.info['reward_tuple'][k]
        
        
        
        global_pos_state = data.x.pos
        global_pos_ref = current_state_ref.x.pos
        pose_error=loss_l2_relpos(global_pos_state, global_pos_ref)
        
        
        state.metrics.update(
            step_index=current_step_inx,
            pose_error=pose_error,
            fall=fall,
        )
        
        return state.replace(
            pipeline_state= data, obs=obs, reward=reward, done=fall
        )
        
    # def step(self, state: State, action: jp.ndarray) -> State:
        
    #     #this doesnt increment it might increment with the algorithms?
    #     index_new =jp.array(state.info['steps']%self.cycle_len, int)
        
    #     #jax.debug.print("new idx: {}",index_new)
        
    #     initial_idx = state.metrics['step_index']
    #     #we want to check the next idx
    #     current_step_inx =  jp.asarray(initial_idx, dtype=jp.int64) + 1
    #     #jax.debug.print("current_step_idx: {}",current_step_inx)
    #     #get the reference state
    #     current_state_ref = self.set_ref_state_pipeline(current_step_inx,state.pipeline_state)
        
    #     qpos = state.pipeline_state.q
    #     qvel = state.pipeline_state.qd

    #     timeEnv = state.pipeline_state.time
        
    #     #deltatime of physics
    #     dt = self.sys.opt.timestep
        
        
    #     torque = self.pd_function(current_state_ref.qpos[7:],self.sys,state,qpos,qvel,
    #                              self.kp_gains,self.kd_gains,timeEnv,dt) 
    #     data = self.pipeline_step(state.pipeline_state,torque)      
    #     #perform forward kinematics to do the same movement that is on the reference trajectory
    #     #this is just for demostration that the trajectory data is working properly
    #     #data = self.pipeline_init(current_state_ref.qpos, current_state_ref.qvel)
        
    #     #here I will do the fall
    #     #check on the z axis
    #     fall = jp.where(data.qpos[2] < 0.3, 1.0, 0.0)
    #     fall = jp.where(data.qpos[2] > 1.7, 1.0, fall)
        
        
    #     obs = self._get_obs(data, current_step_inx)
        
    #     reward, reward_tuple = self.compute_rewards_deepmimic(data,current_state_ref)
        
    #     #state mangement
    #     state.info['reward_tuple'] = reward_tuple
        
    #     for k in state.info['reward_tuple'].keys():
    #         state.metrics[k] = state.info['reward_tuple'][k]
        
        
    #     global_pos_state = data.x.pos
    #     global_pos_ref = current_state_ref.x.pos
    #     pose_error=loss_l2_relpos(global_pos_state, global_pos_ref)
        
        
    #     state.metrics.update(
    #         step_index=current_step_inx,
    #         pose_error=pose_error,
    #         fall=fall,
    #     )
        
    #     return state.replace(
    #         pipeline_state= data, obs=obs, reward=reward, done=fall
    #     )
    
    
 
        
    def compute_rewards_deepmimic(self,data,current_state_ref):
        
        #first get the local position
        #local_pos , local_rotations,local_vel,local_ang = self.convert_local(data)
        local_pos , local_rotations,local_vel,local_ang =self.convertLocaDiff(data)
        #ref_local_pos , ref_local_rotations,ref_local_vel,ref_local_ang = self.convert_local(current_state_ref)
        ref_local_pos , ref_local_rotations,ref_local_vel,ref_local_ang = self.convertLocaDiff(current_state_ref)
        
        
        #data for the end effector reward
        current_ee = data.geom_xpos[self.dict_ee]
        current_ref_ee = current_state_ref.geom_xpos[self.dict_ee]
               
        com_current = self.get_com(data)
        ref_com = self.get_com(current_state_ref)
        
        reward_tuple = {
            'reference_quaternions': (
                self.quat_diff(local_rotations,ref_local_rotations) * self.w_p 
            
            ),
            'reference_angular': (
                self.angular_diff(local_ang, ref_local_ang) * self.w_v 
            ),
            'reference_end_effector': (
                self.end_effector_diff(current_ee,current_ref_ee) * self.w_e
            ),
            'reference_com': (
                self.com_reward(com_current, ref_com) * self.w_c
            )   
        }
        

    
        reward = sum(reward_tuple.values())
        
        return reward, reward_tuple
        
    
    def com_reward(self,com_current, ref_com):
        
        
        # jax.debug.print("com result local:{}",com_current)
        # jax.debug.print("ref com result ref:{}",ref_com)
        
        
        com_dist = jp.linalg.norm(com_current - ref_com)         
        com_reward = jp.exp(-self.w_com * (com_dist**2))
        return com_reward
        
    
    
    def end_effector_diff(self,current_ee,current_ref_ee):
        # jax.debug.print("end diff result local:{}",current_ee)
        # jax.debug.print("end diff result ref:{}",current_ref_ee)
                
        ee_dist = jp.linalg.norm(current_ee - current_ref_ee)        
        #jax.debug.print("end dis:{}",ee_dist)
        
        ee_reward = jp.exp(-self.w_efector * (ee_dist**2))
        #jax.debug.print("ee_reward:{}",ee_reward)
        
        return ee_reward
        
    
    def angular_diff(self,local_ang,ref_local_ang):
        
        # jax.debug.print("ang diff result local:{}",local_ang)
        # jax.debug.print("ang diff result ref:{}",ref_local_ang)
        
        ang_dist = jp.linalg.norm(local_ang - ref_local_ang)
        
        ang_reward = jp.exp(-self.w_angular * (ang_dist**2))
        return ang_reward
        
    
    
    #relative quat ref to quatcurrent
    def quat_diff(self, local_rot, current_state_ref):
        #in the calculation we include the root
        
        #chenge the order to use the quat diff from insactor paper
        current_quat = local_rot[:, [1, 2, 3, 0]] 
        ref_quat = current_state_ref[:, [1, 2, 3, 0]] 
        
        current_quat_normalized = diff_quat.quat_normalize(current_quat) 
        ref_quat_normalized = diff_quat.quat_normalize(ref_quat)
        quat_diff= diff_quat.quat_mul_norm(current_quat_normalized,diff_quat.quat_inverse(ref_quat_normalized))

        
        # Get the scalar rotation to get difference displacement
        # Assuming the quaternion is [x, y, z, w], with w at the last position:
        angles = 2 * jp.arccos(jp.clip(jp.abs(quat_diff[:, 3]), -1.0, 1.0))
        
        
        norm = jp.linalg.norm(angles)
        
        quat_reward = jp.exp(-self.w_pose *(norm**2))
        
        #jax.debug.print("quat diff result:{}",quat_reward)
        
        return quat_reward
    
    
    
    def get_com(self,data):
          
        return data.subtree_com[1]
    
    
    def _get_obs(self, data: base.State, step_idx: jp.ndarray)-> jp.ndarray:
          
        current_step_inx =  jp.asarray(step_idx, dtype=jp.int64)
        
        
        #relative_pos , local_rotations,local_vel,local_ang = self.convert_local(data)
        relative_pos , local_rotations,local_vel,local_ang = self.convertLocaDiff(data)
        
        relative_pos = relative_pos[1:]
        #q_relative_pos,q_local_rotations, q_local_vel, q_local_ang = self.convertLocaDiff(data)
        
        local_rotations = local_rotations.at[0].set(data.x.rot[0])
        
        
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
        phi = (current_step_inx % self.cycle_len) / self.cycle_len
        
        phi = jp.asarray(phi)
        
        
        #jax.debug.print("phi{}", phi)
        return jp.concatenate([relative_pos.ravel(),rot_6D.ravel(),
                               local_vel.ravel(),local_ang.ravel(),phi[None]])


    
    