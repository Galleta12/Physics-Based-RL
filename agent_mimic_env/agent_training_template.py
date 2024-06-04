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
    def __init__(self, reference_data: SimpleConverter, 
                 model_path, 
                 args, **kwargs):
        super().__init__(reference_data, 
                         model_path, 
                         args, **kwargs)
        
        self.err_threshold = args.threshold
        
        
        
    def _demo_replay(self, state,ref_data_pos,current_idx)-> State:
        global_pos_state = state.x.pos
        #jax.debug.print("pos state: {}",global_pos_state)
        #jax.debug.print("pos ref: {}",ref_data_pos)
        error = loss_l2_pos(global_pos_state, ref_data_pos)
        #jax.debug.print("error demoreplay: {}",error)
        
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
    
    
    
    
        
    def step(self, state: State, action: jp.ndarray) -> State:
        
        state = super(HumanoidTrainTemplate,self).step(state,action)
        #grab the data
        data = state.pipeline_state
        
        #perform the demoreplay
        idx = state.metrics['step_index']
        
        current_step_inx =  jp.asarray(idx, dtype=jp.int32)
        
        current_state_ref =self.set_ref_state_pipeline(current_step_inx)
        
        global_pos_ref = current_state_ref.x.pos
        
        new_data,replay=self._demo_replay(data,global_pos_ref,current_step_inx)

        data = data.replace(qpos=new_data.qpos, qvel=new_data.qvel, q=new_data.q,qd=new_data.qd,
                              xpos=new_data.xpos, xquat=new_data.xquat,x=new_data.x,xd=new_data.xd)
        
        #jax.debug.print("Replay: {}",replay)
        
        obs = self._get_obs(data,current_step_inx)
        
        return state.replace(pipeline_state=data, obs=obs) 
        
    
    def step_custom(self, state: State, action: jp.ndarray) -> State:
        
        state = super(HumanoidTrainTemplate,self).step_custom(state,action)
        #grab the data
        data = state.pipeline_state
        
        #perform the demoreplay
        idx = state.metrics['step_index']
        
        current_step_inx =  jp.asarray(idx, dtype=jp.int32)
        
        
        current_state_ref =self.set_ref_state_pipeline(current_step_inx)
        
        global_pos_ref = current_state_ref.x.pos
        
        new_data,replay=self._demo_replay(data,global_pos_ref,current_step_inx)

        data = data.replace(qpos=new_data.qpos, qvel=new_data.qvel, q=new_data.q,qd=new_data.qd,
                              xpos=new_data.xpos, xquat=new_data.xquat,x=new_data.x,xd=new_data.xd)
        
        jax.debug.print("Replay: {}",replay)
        #jax.debug.print("Time: {}",data.time)
        
        obs = self._get_obs(data,current_step_inx)
        
        return state.replace(pipeline_state=data, obs=obs) 
        
    