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
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
        self.pipeline_step = jax.checkpoint(self.pipeline_step, 
            policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    

    def step(self, state: State, action: jp.ndarray) -> State:
        
        #perform the action of the policy
        #current qpos and qvel for the torque    
        #perform the action of the policy
        #current qpos and qvel for the torque    
        qpos = state.pipeline_state.q
        qvel = state.pipeline_state.qd
        timeEnv = state.pipeline_state.time
        #deltatime of physics
        dt = self.sys.opt.timestep
        
        #action = jp.clip(action, -1, 1) # Raw action  
        target_angles = action * jp.pi * 1.2
        #exclude the root
        #target_angles = state.info['default_pos'][7:] +(action * jp.pi * 1.2)
        #target_angles = (action * jp.pi * 1.2)
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
        
        reward, reward_tuple = self.compute_rewards_diffmimic(data,current_state_ref)
        
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
                
        index_new =jp.array(state.info['steps']%self.cycle_len, int)
                
        
        fall=0.0
        fall = jp.where(data.qpos[2] < 0.5, 1.0, fall)
        fall = jp.where(data.qpos[2] > 1.7, 1.0, fall)
        
        reward, reward_tuple = self.compute_rewards_diffmimic(data,current_state_ref)
        
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
        
        
    
    
