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
from .agent_mimic_eval import HumanoidEnvTrainEval
from .losses import *

import sys
import os
# Append the parent directory of both utils and some_math to sys.path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)



from some_math.rotation6D import quaternion_to_rotation_6d
from some_math.math_utils_jax import *

class HumanoidEnvTrain(HumanoidEnvTrainEval):
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
                         model_path,
                         kp_gains,
                         kd_gains,
                         reference_x_pos,
                         reference_x_rot,
                         **kwargs)
        self.err_threshold = 0.4
  
    
    

    
    def _demo_replay(self, state,ref_data_pos,current_idx)-> State:
        global_pos_state = state.x.pos
        #jax.debug.print("pos state: {}",global_pos_state)
        #jax.debug.print("pos ref: {}",ref_data_pos)
        error = loss_l2_pos(global_pos_state, ref_data_pos)
        #jax.debug.print("error: {}",error)
        
        replay = jp.where(error > self.err_threshold, jp.float32(1), jp.float32(0))
        #jax.debug.print("replay: {}",replay)
        #replace the ref_state data
          # Define the true and false branch functions for lax.cond
        def true_fun(_):
            # Update state to reference state and maintain step index
            return self.set_ref_state_pipeline(current_idx)
            #return self.set_ref_step(current_idx,state)
            

        def false_fun(_):
            # Return the original state with updated metrics
            return state
        # Use lax.cond to conditionally switch between states
        new_data = lax.cond(replay == 1, true_fun, false_fun, None)
        
        return new_data,replay
        
    
        
    #just with a custom target but not selected joints
    def step(self, state: State, action: jp.ndarray) -> State:
        
        initial_idx = state.metrics['step_index']
        current_step_inx =  jp.asarray(initial_idx, dtype=jp.int32)
                
        #jax.debug.print("time ref: {}",self.new_ref_data.time)
        current_state_ref = self.set_ref_state_pipeline(current_step_inx)
    
        
        #current qpos and qvel for the torque    
        qpos = state.pipeline_state.q
        qvel = state.pipeline_state.qd
        
        timeEnv = state.pipeline_state.time
        #this will be modified by a one without custom target   
        torque = self.pd_function(action,self.sys,state,qpos,qvel,
                                 self.kp__gains,self.kd__gains,timeEnv,self.sys.dt) 
        
        data = self.pipeline_step(state.pipeline_state,torque)
        
        #first get the values, first value
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
        
        #here the demoreplay
        new_data,replay=self._demo_replay(data,self.reference_x_pos[current_step_inx],current_step_inx)

        data = data.replace(qpos=new_data.qpos, qvel=new_data.qvel, q=new_data.q,qd=new_data.qd,
                            xpos=new_data.xpos, xquat=new_data.xquat,x=new_data.x,xd=new_data.xd)
        #jax.debug.print("data time data: {}",data.time)
        
        
        #get the observations
        obs = self._get_obs(data, current_step_inx)
        
        #increment the step index to know in which episode and wrap
        next_step_index = (current_step_inx + 1) % self.rollout_lenght
        
        state.metrics.update(
            step_index=next_step_index,
            pose_error=loss_l2_relpos(global_pos_state, global_pos_ref),
            fall=fall,
            replay=replay
        )
        
        
        return state.replace(
            pipeline_state= data, obs=obs, reward=reward, done=state.metrics['fall']
        )
        
        