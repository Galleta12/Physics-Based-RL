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


class HumanoidTemplate(PipelineEnv):
    def __init__(
      self,
      reference_data: SimpleConverter,
      model_path,
      args,
      **kwargs,
    ):
        
        path = model_path
        mj_model = mujoco.MjModel.from_xml_path(path)
        
        sys = mjcf.load_model(mj_model)
        
        
        self._dt = 1/60
        optimal_timestep = self._dt/5  
        sys = sys.tree_replace({'opt.timestep': 0.002, 'dt': 0.002})
        n_frames = kwargs.pop('n_frames', int(self._dt / 0.002))
        super().__init__(sys, backend='mjx', n_frames=n_frames)
        
        #this is to keep separate the sys of the agent
        #and the sys for the reference
        self.sys_reference = deepcopy(self.sys)
        #data for the reference
        self.reference_trajectory_qpos = jp.asarray(reference_data.data_pos)
        self.reference_trajectory_qvel = jp.asarray(reference_data.data_vel)
        self.reference_x_pos = reference_data.data_xpos
        self.reference_x_rot = reference_data.data_xrot
        
        #a dictionary with the duration of the data
        self.duration_trajectory = reference_data.total_time
        self.dict_duration = reference_data.duration_dict
        
        #the gains for the pd controller
        self.kp_gains,self.kd_gains = generate_kp_kd_gains()
        
        #this it is the lenght, of the trajectory
        self.rollout_lenght = self.reference_trajectory_qpos.shape[0]
        
        self.rot_weight =  args.rot_weight
        self.vel_weight =  args.vel_weight
        self.ang_weight =  args.ang_weight
        self.reward_scaling= args.reward_scaling
        
        #for now it will be the same size
        self.cycle_len = args.cycle_len if args.cycle_len !=0 else self.rollout_lenght  

        
        
        
    def set_pd_callback(self,pd_control):
        self.pd_function = pd_control
    
    
    #get a reference state from the referece trajectory
    #this is used on the reset
    def get_reference_state(self,step_index):
        ref_qp = self.reference_trajectory_qpos[step_index]
        ref_qv = self.reference_trajectory_qvel[step_index]
        #now I will return a state depending on the index and the reference trajectory
        return self.pipeline_init(ref_qp,ref_qv)
    
    #set the reference state during the steps
    #this is imporant, since we keep two states at the same time
    #this will use its own sys
    def set_ref_state_pipeline(self,step_index):
        ref_qp = self.reference_trajectory_qpos[step_index]
        ref_qv = self.reference_trajectory_qvel[step_index]        
        #now I will return a state depending on the index and the reference trajectory
        return self._pipeline.init(self.sys_reference, ref_qp, ref_qv, self._debug)
    
    
    #the standard reset for all the agents derived from this class
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
        state = State(data, obs, reward, done, metrics)
           
        return state
    
    
    
    def step(self, state: State, action: jp.ndarray) -> State:
        initial_idx = state.metrics['step_index']
        #we want to check the next idx
        current_step_inx =  jp.asarray(initial_idx, dtype=jp.int32) + 1
        jax.debug.print("current_step_idx: {}",current_step_inx)
        #get the reference state
        current_state_ref = self.set_ref_state_pipeline(current_step_inx)
        
        #current qpos and qvel for the torque    
        qpos = state.pipeline_state.q
        qvel = state.pipeline_state.qd
        
        #perform forward kinematics to do the same movement that is on the reference trajectory
        #this is just for demostration that the trajectory data is working properly
        data = self.pipeline_init(current_state_ref.qpos, current_state_ref.qvel)
        
        
        obs = self._get_obs(data, current_step_inx)
        
        reward = self.compute_rewards_diffmimic(data,current_state_ref)
        
        #here I will do the fall
        #check on the z axis
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
        
        
        
        
        
        
        

        
    def compute_rewards_diffmimic(self, data,current_state_ref):
        
        global_pos_state = data.x.pos
        global_rot_state = quaternion_to_rotation_6d(data.x.rot)
                
        global_vel_state = data.xd.vel
        global_ang_state = data.xd.ang
        
        
        #now for the reference trajectory
        global_pos_ref = current_state_ref.x.pos
        global_rot_ref = quaternion_to_rotation_6d(current_state_ref.x.rot)
              
        global_vel_ref = current_state_ref.xd.vel        
        global_ang_ref = current_state_ref.xd.ang
         
        # jax.debug.print("rot weight: {}",self.rot_weight)
        # jax.debug.print("vel weight: {}",self.vel_weight)
        # jax.debug.print("ang weight: {}",self.ang_weight)
        # jax.debug.print("reward scaling: {}",self.reward_scaling)
        
        return -1 * (mse_pos(global_pos_state, global_pos_ref) +
            self.rot_weight * mse_rot(global_rot_state, global_rot_ref) +
            self.vel_weight * mse_vel(global_vel_state, global_vel_ref) +
            self.ang_weight * mse_ang(global_ang_state, global_ang_ref)
            ) * self.reward_scaling
            
            
    
    #standard obs this will changed on other derived
    #classes from this
    def _get_obs(self, data: mjx.Data, action: jp.ndarray)-> jp.ndarray:
          
          #in theory it is mutiable if we do concatenate [] instead of
          #() since, one is a list and the other a tuple
          return jp.concatenate([data.qpos,data.qvel])
   



    
    
    