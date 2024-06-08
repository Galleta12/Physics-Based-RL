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
from jax import lax


from .agent_template import HumanoidTemplate
from .agent_eval_template import HumanoidEvalTemplate
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


class HumanoidTrainTemplate(HumanoidEvalTemplate):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        args = kwargs.pop('args')
        
        self.err_threshold = args.threshold
        
    def reset(self, rng: jp.ndarray) -> State:
        
        #set this as zero
        reward, done, zero = jp.zeros(3)
        #start at random initial state
        #random state initialization (RSI)
        #set on jax int 64 so this is a int64
        new_step_idx = jax.random.randint(rng, shape=(), minval=0, maxval=self.rollout_lenght, dtype=jp.int64)
        # Convert to float64
        new_step_idx_float = jp.asarray(new_step_idx, dtype=jp.float64)
            
       
        data = self.get_reference_state(new_step_idx_float)       
        # qvel = jp.zeros(self.sys.nv)
        # qpos =  self.sys.qpos0
        # data = self.pipeline_init(qpos,qvel) 
        
        metrics = {'step_index': new_step_idx_float, 'pose_error': zero, 'fall': zero}
        obs = self._get_obs(data,new_step_idx_float)    
        
        #jax.debug.print("obs: {}",obs.shape)
        #the obs should be size 193?
        
        ref_qp = self.reference_trajectory_qpos[0]
        
        
        state_info = {
            'rng': rng,
            'steps': 0.0,
            'reward_tuple': {
                'reference_position': 0.0,
                'reference_rotation': 0.0,
                'reference_velocity': 0.0,
                'reference_angular': 0.0
            },
            # 'step_index':0.0,
            # 'pose_error':0.0,
            # 'fall': 0.0
            
        }
        
        #metrics = {}
        #save the infor on the metrics
        for k in state_info['reward_tuple']:
            metrics[k] = state_info['reward_tuple'][k]
        
        state = State(data, obs, reward, done, metrics,state_info)
           
        return jax.lax.stop_gradient(state)    
        
    # def _demo_replay(self, state,ref_data_pos,current_idx)-> State:
    #     global_pos_state = state.x.pos
    #     #jax.debug.print("pos state: {}",global_pos_state)
    #     #jax.debug.print("pos ref: {}",ref_data_pos)
    #     error = loss_l2_pos(global_pos_state, ref_data_pos)
    #     #jax.debug.print("error demoreplay: {}",error)
        
    #     replay = jp.where(error > self.err_threshold, jp.float32(1), jp.float32(0))
    #     #jax.debug.print("replay: {}",replay)
    #     #replace the ref_state data
    #       # Define the true and false branch functions for lax.cond
    #     def true_fun(_):
    #         # Update state to reference state and maintain step index
    #         return self.set_ref_state_pipeline(current_idx)
    #         #return self.set_ref_step(current_idx,state)
            

    #     def false_fun(_):
    #         # Return the original state with updated metrics
    #         return state
    #     # Use lax.cond to conditionally switch between states
    #     new_data = lax.cond(replay == 1, true_fun, false_fun, None)
        
    #     return new_data,replay
    
    
    def _demoreplay(self,data,ref_data):
        global_pos_state = data.x.pos
        ref_data_pos = ref_data.x.pos
        
        error = loss_l2_pos(global_pos_state, ref_data_pos)
        #jax.debug.print("error demo replay: {}",error)
        to_reference = jp.where(error > self.err_threshold, 1.0, 0.0)
        to_reference = jp.array(to_reference, dtype=int) # keeps output types same as input. 
        #jax.debug.print("to converted reference: {}",to_reference)
        #convert the data to mjxbrax
        ref_data = self._get_new_ref(ref_data,data)
        #get new data
        
        return ref_data,to_reference

    
    def _get_new_ref(self,ref_data,data):
        ref_qpos,ref_qvel = ref_data.qpos,ref_data.qvel
        ref_data = data.replace(qpos=ref_qpos, qvel=ref_qvel)
        ref_data = mjx.forward(self.sys, ref_data)
        ref_data = self.mjx_to_brax(ref_data)
        return ref_data
    
    
    def step(self, state: State, action: jp.ndarray) -> State:
        
        state = super(HumanoidTrainTemplate,self).step(state,action)
        # data = state.pipeline_state
        
        # #perform the demoreplay
        # idx = state.metrics['step_index']
        # idx_alg = jp.array(state.info['steps'], int)
        # current_step_inx =  jp.asarray(idx, dtype=jp.int64)
        
        # #get the coordinates of the current step 
        # #get qpos and q vel
        # ref_data = self.set_ref_state_pipeline(current_step_inx,data)
        # #ref_qpos,ref_qvel=self.get_ref_qdata(current_step_inx)
     
        # # ref_data = data.replace(qpos=ref_qpos, qvel=ref_qvel)
        # # ref_data = mjx.forward(self.sys, ref_data)
        # # ref_x, ref_xd = ref_data.x, ref_data.xd

        # #perform the demoreplay
        # new_ref_data,to_reference = self._demoreplay(data,ref_data)
        
        
        # data = jax.tree_util.tree_map(lambda x, y: 
        #                           jp.array((1-to_reference)*x + to_reference*y, x.dtype), data, new_ref_data)

        
        # obs = self._get_obs(data,current_step_inx)
        
        #return state.replace(pipeline_state=data, obs=obs) 
        return state 
    
    
    def step_custom(self, state: State, action: jp.ndarray) -> State:
        
        state = super(HumanoidTrainTemplate,self).step_custom(state,action)
        #grab the data
        data = state.pipeline_state
        
        #perform the demoreplay
        idx = state.metrics['step_index']
        idx_alg = jp.array(state.info['steps'], int)
        current_step_inx =  jp.asarray(idx, dtype=jp.int64)
        
        
        #get the coordinates of the current step 
        #get qpos and q vel
        
        ref_data = self.set_ref_state_pipeline(current_step_inx,data)
        #ref_qpos,ref_qvel=self.get_ref_qdata(current_step_inx)
     
        # ref_data = data.replace(qpos=ref_qpos, qvel=ref_qvel)
        # ref_data = mjx.forward(self.sys, ref_data)
        # ref_x, ref_xd = ref_data.x, ref_data.xd

        #perform the demoreplay
        new_ref_data,to_reference = self._demoreplay(data,ref_data)
        
        
        data = jax.tree_util.tree_map(lambda x, y: 
                                  jp.array((1-to_reference)*x + to_reference*y, x.dtype), data, new_ref_data)

        
        obs = self._get_obs(data,current_step_inx)
        
        return state.replace(pipeline_state=data, obs=obs) 
    
    
    
    
    def get_ref_qdata(self, idx):
        ref_qp = self.reference_trajectory_qpos[idx]
        ref_qv = self.reference_trajectory_qvel[idx]
        
        return ref_qp,ref_qv