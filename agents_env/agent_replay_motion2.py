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


class HumanoidReplay2(HumanoidDiff2):
    def __init__(self, 
                 reference_trajectory_qpos, 
                 reference_trajectory_qvel, 
                 duration_trajectory, 
                 dict_duration, 
                 model_path, 
                 **kwargs):
        super().__init__(reference_trajectory_qpos, 
                         reference_trajectory_qvel, 
                         duration_trajectory, 
                         dict_duration, 
                         model_path, 
                         **kwargs)
    
    
    
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
        #save the data
        #pipeline_state is similar to mujoco_data
        data0 = state.pipeline_state
        done =1.0     
        qpos = data0.qpos
        qvel = data0.qvel
        
        #forward dynamics if you check the code you will see
        #the pipeline init perform forward kinematics
        data = self.pipeline_init(qpos, qvel)
        
        
        reward = jp.zeros(3)
        
        obs = self._get_obs(data, jp.zeros(self.sys.nv))
        
        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done
        )
    
    