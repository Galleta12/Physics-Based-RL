from datetime import datetime
import functools
from IPython.display import HTML
import jax.numpy as jp
import numpy as np
import jax
from jax import config # Analytical gradients work much better with double precision.
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)
config.update('jax_default_matmul_precision', jax.lax.Precision.HIGH)
from brax import math
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
        
        reward, done, zero = jp.zeros(3)
        # Convert to float64
        #new_step_idx_float = jp.asarray(0, dtype=jp.float64)
        new_step_idx = jax.random.randint(rng, shape=(), minval=0, maxval=self.rollout_lenght, dtype=jp.int64)
        # Convert to float64
        new_step_idx_float = jp.asarray(new_step_idx, dtype=jp.float64)   
       
        data = self.get_reference_state(new_step_idx_float)       
        # qvel = jp.zeros(self.sys.nv)
        # qpos =  self.sys.qpos0
        # data = self.pipeline_init(qpos,qvel) 
        
        metrics = {'step_index': new_step_idx_float, 'pose_error': zero, 'fall': zero}
        
        #jax.debug.print("obs: {}",obs.shape)
        #the obs should be size 193?
        
        ref_qp = data.qpos

        state_info = {
            'rng': rng,
            'steps': 0.0,
            'reward_tuple': {
                'reference_quaternions': 0.0,
                'reference_angular': 0.0,
                'reference_end_effector': 0.0,
                'reference_com': 0.0
            },
            'default_pos': ref_qp,
            'last_action':jp.zeros(28),
            'kinematic_ref': ref_qp     
    
        }
        
        obs = self._get_obs(data,new_step_idx_float,state_info) 
        #metrics = {}
        #save the infor on the metrics
        for k in state_info['reward_tuple']:
            metrics[k] = state_info['reward_tuple'][k]
        
        state = State(data, obs, reward, done, metrics,state_info)
           
        #return jax.lax.stop_gradient(state)
        return state
    
  
    def step(self, state: State, action: jp.ndarray) -> State:
            
        qpos = state.pipeline_state.q
        qvel = state.pipeline_state.qd
        timeEnv = state.pipeline_state.time
        #deltatime of physics
        dt = self.sys.opt.timestep
        
        #action = jp.clip(action, -1, 1) # Raw action  
        target_angles = action * jp.pi * 1.2
        #exclude the root
        #target_angles = state.info['default_pos'][7:] +(action * jp.pi * 1.2)
        #target_angles = action 
        
        torque = self.pd_function(target_angles,self.sys,state,qpos,qvel,
                                 self.kp_gains,self.kd_gains,timeEnv,dt) 
        
        data = self.pipeline_step(state.pipeline_state,torque)
        #data = self.pipeline_step(state.pipeline_state,target_angles)
                
        index_new =jp.array(state.info['steps']%self.cycle_len, int)
                
        
        initial_idx = state.metrics['step_index'] +1.0
        current_step_inx =  jp.asarray(initial_idx%self.cycle_len, dtype=jp.float64)

        current_state_ref = self.set_ref_state_pipeline(current_step_inx,state.pipeline_state)

        
        fall=0.0
        fall = jp.where(data.qpos[2] < 0.5, 1.0, fall)
        fall = jp.where(data.qpos[2] > 1.7, 1.0, fall)
        
        reward, reward_tuple = self.compute_rewards_deepmimic(data,current_state_ref)
        
        #get the observations
        #state mangement
        state.info['last_action'] = action
        state.info['kinematic_ref'] = current_state_ref.qpos
        state.info['reward_tuple'] = reward_tuple
        
        obs = self._get_obs(data, current_step_inx,state.info)
        
        
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
        
    
    
    
    #class for testing step_custom where we just check the pd controller
    def step_custom(self, state: State, action: jp.ndarray) -> State:
        
        initial_idx = state.metrics['step_index'] +1.0
        current_step_inx =  jp.asarray(initial_idx%self.cycle_len, dtype=jp.float64)
        current_state_ref = self.set_ref_state_pipeline(current_step_inx,state.pipeline_state)
      
        qpos = state.pipeline_state.q
        qvel = state.pipeline_state.qd
        timeEnv = state.pipeline_state.time
        #deltatime of physics
        dt = self.sys.opt.timestep
        
        #target_angles = action * jp.pi * 1.2
        #exclude the root
        target_angles = state.info['default_pos'][7:] +(current_state_ref.qpos[7:]* jp.pi * 1.2)
        #target_angles = action 
        
        torque = self.pd_function(current_state_ref.qpos[7:],self.sys,state,qpos,qvel,
                                 self.kp_gains,self.kd_gains,timeEnv,dt) 
        
        data = self.pipeline_step(state.pipeline_state,torque)
        #data = self.pipeline_step(state.pipeline_state,target_angles)
        #data = self.pipeline_init(current_state_ref.qpos,current_state_ref.qvel)
        
        index_new =jp.array(state.info['steps']%self.cycle_len, int)
                
        
        
        fall=0.0
        fall = jp.where(data.qpos[2] < 0.5, 1.0, fall)
        fall = jp.where(data.qpos[2] > 1.7, 1.0, fall)
        
        reward, reward_tuple = self.compute_rewards_deepmimic(data,current_state_ref)
        
        #get the observations
        #state mangement
        state.info['last_action'] = action
        state.info['kinematic_ref'] = current_state_ref.qpos
        state.info['reward_tuple'] = reward_tuple
        
        obs = self._get_obs(data, current_step_inx,state.info)
        
        
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
        
    
    
 
        
    def compute_rewards_deepmimic(self,data,current_state_ref):
        
        #first get the local position
        #local_pos , local_rotations,local_vel,local_ang = self.convert_local(data)
        local_pos , local_rotations,local_vel,local_ang =self.convertLocaDiff(data)
        #ref_local_pos , ref_local_rotations,ref_local_vel,ref_local_ang = self.convert_local(current_state_ref)
        ref_local_pos , ref_local_rotations,ref_local_vel,ref_local_ang = self.convertLocaDiff(current_state_ref)
        
        #jax.debug.print("this is the local ang curret:{}",local_ang)
        #jax.debug.print("this is the local ang ref:{}",ref_local_ang)
                
        #jax.debug.print("this is the local ang rotation{}:",local_rotations)
        #jax.debug.print("this is the local ang rotation{}:",ref_local_rotations)

        
        
        #data for the end effector reward
        current_ee = data.geom_xpos[self.dict_ee]
        current_ref_ee = current_state_ref.geom_xpos[self.dict_ee]
        
        #jax.debug.print('current end{}',current_ee)
        #jax.debug.print('currentref end{}',current_ref_ee)
        
        
        com_current,_,_,_ = self._com(data)
        ref_com,_,_,_ = self._com(current_state_ref)
        # com_current = self.get_com(data)
        # ref_com = self.get_com(current_state_ref)
        #jax.debug.print('current com{}',com_current)
        #jax.debug.print('current com{}',ref_com)
        
        reward_tuple = {
            'reference_quaternions': (
                self.quat_diff(local_rotations,ref_local_rotations,
                               data,current_state_ref)  
            ),
            'reference_angular': (
                self.angular_diff(local_ang, ref_local_ang)  
            ),
            'reference_end_effector': (
                self.end_effector_diff(current_ee,current_ref_ee) 
            ),
            'reference_com': (
                self.com_diff(com_current, ref_com) 
            )   
        }
        
    
        quaterion_error = reward_tuple['reference_quaternions']
        angular_error = reward_tuple['reference_angular']
        end_error = reward_tuple['reference_end_effector']
        com_error = reward_tuple['reference_com']
        #reward = sum(reward_tuple.values())
        #jax.debug.print("quat error{}:",quaterion_error)
        #jax.debug.print("ang error{}:",angular_error)
        #jax.debug.print("end error{}:",end_error)
        #jax.debug.print("com error{}:",com_error)
        
        
        
        reward_1 = (self.w_p*jp.exp(-quaterion_error)) + (self.w_v * jp.exp(-angular_error))
        reward_2 = (self.w_e * jp.exp(-end_error)) + (self.w_c * jp.exp(-com_error))
        reward = (reward_1 + reward_2)
        #jax.debug.print("reward from env {}:",reward)
        
        
        return reward, reward_tuple
        
    
    def com_diff(self,com_current, ref_com):
        # jax.debug.print("com result local:{}",com_current)
        # jax.debug.print("ref com result ref:{}",ref_com)
        #shape 3x1    
        com_norm = jp.linalg.norm(ref_com - com_current,axis=-1)         
        com_dist = jp.sum(com_norm)
        com_reward = self.w_com * com_dist
        
        return com_reward
        
    
    
    def end_effector_diff(self,current_ee,current_ref_ee):
        # jax.debug.print("end diff result local:{}",current_ee)
        # jax.debug.print("end diff result ref:{}",current_ref_ee)        
        ee_norm = jp.linalg.norm(current_ref_ee - current_ee,axis=-1)        
        #jax.debug.print("end dis:{}",ee_dist)
        ee_dist = jp.sum(ee_norm)
        ee_reward = self.w_efector * ee_dist
        #jax.debug.print("ee_reward:{}",ee_reward)
        
        return ee_reward
        
    
    def angular_diff(self,local_ang,ref_local_ang):
        
        # jax.debug.print("ang diff result local:{}",local_ang)
        # jax.debug.print("ang diff result ref:{}",ref_local_ang)
        
        ang_norm = jp.linalg.norm(ref_local_ang - local_ang,axis=-1)
        ang_dist = jp.sum(ang_norm)
        ang_reward = self.w_angular * ang_dist
        
        return ang_reward
        
    
    
    #relative quat ref to quatcurrent
    def quat_diff(self, local_rotations,ref_local_rotations,data,current_state_ref):
        #in the calculation we include the root 
        current_rot6D = quaternion_to_rotation_6d(local_rotations[1:])
        ref_rot6D = quaternion_to_rotation_6d(ref_local_rotations[1:])
        
        #rot_dist = jp.linalg.norm(current_rot6D - ref_rot6D)
        
        #quat_reward = jp.exp(-self.w_pose * (rot_dist**2))
        #quat_reward = jp.exp(-self.w_pose * (rot_dist**2))

        
        #jax.debug.print("quat diff result:{}",quat_reward)
        
        #along the last axis which is the columns
        rot_norm = jp.linalg.norm(ref_rot6D-current_rot6D,axis=-1)
        
        rot_dist = jp.sum(rot_norm)
        
        quat_reward = self.w_pose * rot_dist
        
        
        return quat_reward
    
    
    
    def get_com(self,data):
          
        return data.subtree_com[1]
    
    
    def _get_obs(self, data: base.State, step_idx: jp.ndarray,state_info: Dict[str, Any])-> jp.ndarray:
          
        current_step_inx =  jp.asarray(step_idx, dtype=jp.int64)
                
        #relative_pos , local_rotations,local_vel,local_ang = self.convert_local(data)
        relative_pos , local_rotations,local_vel,local_ang = self.convertLocaDiff(data)
        
        relative_pos = relative_pos[1:]
        # #q_relative_pos,q_local_rotations, q_local_vel, q_local_ang = self.convertLocaDiff(data)
        
        local_rotations = local_rotations.at[0].set(data.x.rot[0])
        
        
        # #convert quat to 6d root
        rot_6D= quaternion_to_rotation_6d(local_rotations)
        
        phi = (current_step_inx % self.cycle_len) / self.cycle_len
        
        phi = jp.asarray(phi)
        
        
        #jax.debug.print("phi{}", phi)
        #I will add the last action
        last_ref = state_info['kinematic_ref']
        return jp.concatenate([data.qpos,data.qvel,phi[None]])
        
        
    
    
    