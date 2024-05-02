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
from utils import SimpleConverter

#the template will have as input, the duration of the reference motion
#the reference motion itself, a reference motion dictionary with useful data

#I will create here the environemtn for my body
class HumanoidDiff2(PipelineEnv):
    def __init__(
      self,
      reference_trajectory_qpos,
      reference_trajectory_qvel,
      duration_trajectory,
      dict_duration,
      model_path,
      **kwargs,
    ):
        path = model_path
        mj_model = mujoco.MjModel.from_xml_path(path)
        
        sys = mjcf.load_model(mj_model)
        
        # Set the policy execution timestep (30Hz)
        self._dt = 1.0 / 30  # Policy execution frequency
                
        # Set the optimal physics simulation timestep (1.2kHz)
        optimal_timestep = 1.0 / 1200  # Physics simulation frequency 
        
        
        sys = sys.tree_replace({'opt.timestep': optimal_timestep, 'dt':  optimal_timestep })
        #the num of frames will be 40, make play with this more  
        
        n_frames = kwargs.pop('n_frames', int(self._dt / sys.opt.timestep))
       
      
    
        
        
        super().__init__(sys, backend='mjx', n_frames=n_frames)
                
        self.reference_trajectory_qpos = reference_trajectory_qpos
        self.reference_trajectory_qvel = reference_trajectory_qvel
        self.duration_trajectory = duration_trajectory
        self.dict_duration = dict_duration

    #just to change the trajectory if needed
    def set_new_trajectory(self,trajectory:SimpleConverter):
        
        self.reference_trajectory_qpos = jp.asarray(trajectory.data_pos)
        self.reference_trajectory_qvel = jp.asarray(trajectory.data_vel)
        self.duration_trajectory = trajectory.total_time
        self.dict_duration = trajectory.duration_dict

            
    def reset(self, rng: jp.ndarray) -> State:
        
        #set this as zero
        reward, done, zero = jp.zeros(3)
        
        metrics = {'step_index': zero, 'pose_error': zero, 'fall': zero}
        qvel = jp.zeros(self.sys.nv)
        qpos =  self.sys.qpos0
        
        #this is what init starts do
        #it seems that init do forward kinematics
        """Initializes physics data.
        Args:
            sys: a brax System
            q: (q_size,) joint angle vector
            qd: (qd_size,) joint velocity vector
            unused_debug: ignored
        """ 
        data = self.pipeline_init(qpos, qvel)
        
        obs = self._get_obs(data, jp.zeros(self.sys.nv))
        
        state = State(data, obs, reward, done, metrics)
        
        return state
    #step should be defined on each class
    # def step(self, state: State, action: jp.ndarray) -> State:
    #     #note this is how it looks a state
    #     """
    #     @struct.dataclass
    #     class State(base.Base):
    #         #Environment state for training and inference
    #         pipeline_state: Optional[base.State]
    #         obs: jax.Array
    #         reward: jax.Array
    #         done: jax.Array
    #         metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
    #         info: Dict[str, Any] = struct.field(default_factory=dict)

    #     """    
    #     data0 = state.pipeline_state
    #     done =1.0
    #     # #the second input needs to be the actutor dim for computing the
    #     # #torques
    #     # data = self.pipeline_step(data0, jp.zeros(self.sys.nv)[6:])
        
         
    #     qpos = data0.qpos
    #     qvel = data0.qvel
        
    #     #forward dynamics if you check the code u will see
    #     #the pipeline init perform forward
    #     data = self.pipeline_init(qpos, qvel)
        
        
    #     reward = jp.zeros(3)
        
    #     obs = self._get_obs(data, jp.zeros(self.sys.nv))
        
    #     return state.replace(
    #         pipeline_state=data, obs=obs, reward=reward, done=done
    #     )
    
    def _get_obs(self, data: mjx.Data, action: jp.ndarray)-> jp.ndarray:
          
          #in theory it is mutiable if we do concatenate [] instead of
          #() since, one is a list and the other a tuple
          return jp.concatenate([data.qpos,data.qvel])
   


