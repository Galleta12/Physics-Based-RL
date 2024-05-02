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

from utils import util_data




from .agent_env_template2 import HumanoidDiff2


class HumanoidTrain2(HumanoidDiff2):
    def __init__(self, reference_trajectory_qpos, 
                 reference_trajectory_qvel, 
                 duration_trajectory,
                 dict_duration, 
                 model_path,
                 kp_gains,
                 kd_gains,
                 **kwargs):
        super().__init__(reference_trajectory_qpos, 
                         reference_trajectory_qvel, 
                         duration_trajectory, 
                         dict_duration, 
                         model_path, **kwargs)
        self.kp__gains = kp_gains
        self.kd__gains = kd_gains
        
    
    #set pd callback
    def set_pd_callback(self,pd_control):
        self.pd_function = pd_control
    
    #this step is vectorized, thus it will perferom the controllers on all the joints
    #with the given reference
    def step(self, state: State, action: jp.ndarray) -> State:
        #note this is how it looks a state
        """
        @struct.dataclass
        class State(base.Base):
            #Environment state for training and inference
            pipeline_state: Optional[base.State]
            obs: jax.Array
            reward: jax.Array
            done: jax.Array
            metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
            info: Dict[str, Any] = struct.field(default_factory=dict)
        """    
        time = state.pipeline_state.time
        done =1.0
        #grab the current q pos and qvel of the state,     
        qpos = state.pipeline_state.q.copy()
        qvel = state.pipeline_state.qd.copy()
        
        #perform pd
        #we just pass the actions, that are the targets     
        torque = self.pd_function(action,self.sys,state,qpos,qvel,
                                 self.kp__gains,self.kd__gains,time,self.sys.dt) 
        
        data = self.pipeline_step(state.pipeline_state,torque)
        
        reward = jp.zeros(3)
        
        obs = self._get_obs(data, action)
    
        
        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done
        )
        
        
    def step_selected_joints_custom_target_and_joints(self, state: State, action: jp.ndarray,
                                           custom_target,joints1,joints2,
                                           joints3,joints4,
                                           joints5,joints6,time) -> State:
        #note this is how it looks a state
        """
        @struct.dataclass
        class State(base.Base):
            #Environment state for training and inference
            pipeline_state: Optional[base.State]
            obs: jax.Array
            reward: jax.Array
            done: jax.Array
            metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
            info: Dict[str, Any] = struct.field(default_factory=dict)

        """              
        done =1.0
            
        qpos = state.pipeline_state.q
        qvel = state.pipeline_state.qd
    
        torque = self.pd_function(custom_target,self.sys,state,qpos,qvel,
                                 self.kp__gains,self.kd__gains,time,self.sys.dt) 
        
        #jax.debug.print("x: {}",pd)  
        
                       
        updated_ctrl = state.pipeline_state.ctrl.at[joints1].set(torque[joints1])
        updated_ctrl = updated_ctrl.at[joints2].set(torque[joints2])
        updated_ctrl = updated_ctrl.at[joints3].set(torque[joints3])
        updated_ctrl = updated_ctrl.at[joints4].set(torque[joints4])
        
        # updated_ctrl = updated_ctrl.at[joints5].set(torque[joints5])
        # updated_ctrl = updated_ctrl.at[joints6].set(torque[joints6])
        #updated_ctrl = state.pipeline_state.ctrl.at[:].set(pd)
       
        data = self.pipeline_step(state.pipeline_state,updated_ctrl)
        
        reward = jp.zeros(3)
        
        obs = self._get_obs(data, updated_ctrl)
        
        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done
        )
    
    
    
    
        
    def step_selected_joints_custom_target_and_joints6(self, state: State, action: jp.ndarray,
                                           custom_target,joints1,joints2,
                                           joints3,joints4,
                                           joints5,joints6,time) -> State:
        #note this is how it looks a state
        """
        @struct.dataclass
        class State(base.Base):
            #Environment state for training and inference
            pipeline_state: Optional[base.State]
            obs: jax.Array
            reward: jax.Array
            done: jax.Array
            metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
            info: Dict[str, Any] = struct.field(default_factory=dict)

        """              
        done =1.0
            
        qpos = state.pipeline_state.q
        qvel = state.pipeline_state.qd
    
        torque = self.pd_function(custom_target,self.sys,state,qpos,qvel,
                                 self.kp__gains,self.kd__gains,time,self.sys.dt) 
        
        #jax.debug.print("x: {}",pd)  
        
                       
        updated_ctrl = state.pipeline_state.ctrl.at[joints1].set(torque[joints1])
        updated_ctrl = updated_ctrl.at[joints2].set(torque[joints2])
        updated_ctrl = updated_ctrl.at[joints3].set(torque[joints3])
        updated_ctrl = updated_ctrl.at[joints4].set(torque[joints4])
        
        updated_ctrl = updated_ctrl.at[joints5].set(torque[joints5])
        updated_ctrl = updated_ctrl.at[joints6].set(torque[joints6])
        #updated_ctrl = state.pipeline_state.ctrl.at[:].set(pd)
       
        data = self.pipeline_step(state.pipeline_state,updated_ctrl)
        
        reward = jp.zeros(3)
        
        obs = self._get_obs(data, updated_ctrl)
        
        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done
        )
    
    
    
    
    
    
    
    
    #just with a custom target but not selected joints
    def step_custom_target(self, state: State, action: jp.ndarray,
                                           custom_target,time) -> State:
        #note this is how it looks a state
        """
        @struct.dataclass
        class State(base.Base):
            #Environment state for training and inference
            pipeline_state: Optional[base.State]
            obs: jax.Array
            reward: jax.Array
            done: jax.Array
            metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
            info: Dict[str, Any] = struct.field(default_factory=dict)

        """              
        done =1.0
            
        qpos = state.pipeline_state.q
        qvel = state.pipeline_state.qd
    
        torque = self.pd_function(custom_target,self.sys,state,qpos,qvel,
                                 self.kp__gains,self.kd__gains,time,self.sys.dt) 
        
        #jax.debug.print("x: {}",pd)  
        
    
        #updated_ctrl = state.pipeline_state.ctrl.at[:].set(pd)
       
        data = self.pipeline_step(state.pipeline_state,torque)
        
        reward = jp.zeros(3)
        
        obs = self._get_obs(data, torque)
        
        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done
        )
    