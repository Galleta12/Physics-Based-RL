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
from jax.flatten_util import ravel_pytree

from some_math.rotation6D import quaternion_to_rotation_6d
from some_math.math_utils_jax import *
from utils.SimpleConverter import SimpleConverter
from utils.util_data import generate_kp_kd_gains
import some_math.quaternion_diff as diff_quat


def get_config():
  def get_default_rewards_config():
    default_config = config_dict.ConfigDict(
        dict(
            scales=config_dict.ConfigDict(
              dict(
                min_reference_tracking = -2.5 * 3e-3, # to equalize the magnitude
                reference_tracking = -1.0
              
                )
              )
            )
    )
    return default_config

  default_config = config_dict.ConfigDict(
      dict(rewards=get_default_rewards_config(),))

  return default_config

class HumanoidAPGTest(PipelineEnv):
    def __init__(
      self,
      termination_height: float=0.5,
      **kwargs,
    ):
        
        reference_data = kwargs.pop('referece_data')
        model_path = kwargs.pop('model_path')
        args = kwargs.pop('args')
        
        
        path = epath.Path(model_path).as_posix()
        mj_model = mujoco.MjModel.from_xml_path(path)
        
        # mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        # mj_model.opt.iterations = 6
        # mj_model.opt.ls_iterations = 6
            
        physics_steps_per_control_step = 10
       
        sys = mjcf.load_model(mj_model)
        
        
        self._dt = 1/60
        optimal_timestep = self._dt/5  
        
        sys = sys.tree_replace({'opt.timestep': 0.002})
        
        #n_frames = kwargs.pop('n_frames', int(self._dt / 0.002))
        n_frames = kwargs.pop('n_frames',  physics_steps_per_control_step)
        
        super().__init__(sys=sys, **kwargs)    
    
        
        #and the sys for the reference
        self.sys_reference = deepcopy(self.sys)
        self.termination_height = termination_height
        
        self.err_threshold = 0.4 # diffmimic; value from paper.
        self.reward_config = get_config()

        #self.feet_inds = jp.array([21,28,35,42]) # LF, RF, LH, RH
        self.rollout_lenght = args.ep_len
        
        data_pos = jp.asarray(reference_data.data_pos)
        data_vel= jp.asarray(reference_data.data_vel)
        self.kinematic_ref_qpos = jp.asarray(data_pos)
        self.kinematic_ref_qvel = jp.asarray(data_vel)
        
        
        self.cycle_len = jp.array(self.kinematic_ref_qpos.shape[0])
        

    
        # Can decrease jit time and training wall-clock time significantly.
        self.pipeline_step = jax.checkpoint(self.pipeline_step, 
        policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
        
    
    #get a reference state from the referece trajectory
    #this is used on the reset only on the reset
    def get_reference_state(self,step_index):
        
        step_index =  jp.asarray(step_index, dtype=jp.int64)
        ref_qp = self.kinematic_ref_qpos[step_index]
        ref_qv = self.kinematic_ref_qvel[step_index]
        #now I will return a state depending on the index and the reference trajectory
        return self.pipeline_init(ref_qp,ref_qv)
    
    
    def set_ref_state_pipeline(self,step_index):
        step_index =  jp.asarray(step_index, dtype=jp.int64)
        
        ref_qp = self.kinematic_ref_qpos[step_index]
        ref_qv = self.kinematic_ref_qpos[step_index]
        #now I will return a state depending on the index and the reference trajectory
        return self._pipeline.init(self.sys_reference, ref_qp, ref_qv, self._debug)
         
    
    def reset(self, rng: jp.ndarray) -> State:
        
        #set this as zero
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
               'reference_tracking': 0.0,
               'min_reference_tracking': 0.0,
            },
            'last_action':jp.zeros(28),
            'kinematic_ref': ref_qp,     
            'default_pos': ref_qp
        }
        
        obs = self._get_obs(data,new_step_idx_float,state_info)    
        #metrics = {}
        #save the infor on the metrics
        for k in state_info['reward_tuple']:
            metrics[k] = state_info['reward_tuple'][k]
        
        state = State(data, obs, reward, done, metrics,state_info)
           
        return jax.lax.stop_gradient(state)
  
    def step(self, state: State, action: jax.Array) -> State:
        action = jp.clip(action, -1, 1) # Raw action

        action = state.info['default_pos'][7:] +(action * jp.pi * 1.2)

        data = self.pipeline_step(state.pipeline_state, action)
        
        initial_idx = state.metrics['step_index'] +1.0
        current_step_inx =  jp.asarray(initial_idx%self.cycle_len, dtype=jp.float64)
        
        ref_qpos = self.kinematic_ref_qpos[jp.array(current_step_inx, int)]
        ref_qvel = self.kinematic_ref_qvel[jp.array(current_step_inx, int)]
        
        ref_data =self._pipeline.init(self.sys_reference, ref_qpos, ref_qvel, self._debug)
        
        # Calculate maximal coordinates
        # ref_data = data.replace(qpos=ref_qpos, qvel=ref_qvel)
        # ref_data = mjx.forward(self.sys, ref_data)
        ref_x, ref_xd = ref_data.x, ref_data.xd

        state.info['kinematic_ref'] = ref_qpos

        # observation data
        x, xd = data.x, data.xd
        obs = self._get_obs(data,current_step_inx,state.info)

        # Terminate if flipped over or fallen down.
        done = 0.0
        done = jp.where(x.pos[0, 2] < self.termination_height, 1.0, done)
       

        # reward
        reward_tuple = {
            'reference_tracking': (
            self._reward_reference_tracking(x, xd, ref_x, ref_xd)
            * self.reward_config.rewards.scales.reference_tracking
            ),
            'min_reference_tracking': (
            self._reward_min_reference_tracking(ref_qpos, ref_qvel, state)
            * self.reward_config.rewards.scales.min_reference_tracking
            )    
        }
        
       
        reward = sum(reward_tuple.values())
        reward = jp.nan_to_num(reward)
        obs = jp.nan_to_num(obs)
        flattened_vals, _ = ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        done = jp.where(num_nans > 0, 1.0, done)
        
        
        # state management
        state.info['reward_tuple'] = reward_tuple
        state.info['last_action'] = action # used for observation. 

        for k in state.info['reward_tuple'].keys():
            state.metrics[k] = state.info['reward_tuple'][k]

        global_pos_state = data.x.pos
        global_pos_ref = ref_data.x.pos
        pose_error=loss_l2_relpos(global_pos_state, global_pos_ref)
        
        state.metrics.update(
            step_index=current_step_inx,
            pose_error=pose_error,
            fall=done,
        )
        
        
        
        state = state.replace(
            pipeline_state=data, obs=obs, reward=reward,
            done=done)
        
       
        return state
    
    def _get_obs(self, data: base.State, step_idx: jp.ndarray,state_info: Dict[str, Any])-> jp.ndarray:

        current_step_inx =  jp.array(step_idx, int)
        relative_pos , local_rotations,local_vel,local_ang = self.convertLocaDiff(data)
        
        relative_pos = relative_pos[1:]
        local_rotations = local_rotations.at[0].set(data.x.rot[0])
        rot_6D= quaternion_to_rotation_6d(local_rotations)
        phi = (current_step_inx % self.cycle_len) / self.cycle_len
        
        phi = jp.asarray(phi)
        
        last_ref = state_info['kinematic_ref']
        
        
        return jp.concatenate([relative_pos.ravel(),rot_6D.ravel(),
                               local_vel.ravel(),local_ang.ravel(),
                               phi[None]])
        #return obs
  
    def mjx_to_brax(self, data):
        """ 
        Apply the brax wrapper on the core MJX data structure.
        """
        q, qd = data.qpos, data.qvel
        x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
        cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
        offset = data.xpos[1:, :] - data.subtree_com[self.sys.body_rootid[1:]]
        offset = Transform.create(pos=offset)
        xd = offset.vmap().do(cvel)
        data = _reformat_contact(self.sys, data)
        return data.replace(q=q, qd=qd, x=x, xd=xd)


    # ------------ reward functions----------------
    def _reward_reference_tracking(self, x, xd, ref_x, ref_xd):
        """
        Rewards based on inertial-frame body positions.
        Notably, we use a high-dimension representation of orientation.
        """

        f = lambda x, y: ((x - y) ** 2).sum(-1).mean()

        _mse_pos = f(x.pos,  ref_x.pos)
        _mse_rot = f(quaternion_to_rotation_6d(x.rot),
                    quaternion_to_rotation_6d(ref_x.rot))
        _mse_vel = f(xd.vel, ref_xd.vel)
        _mse_ang = f(xd.ang, ref_xd.ang)

        # Tuned to be about the same size.
        return _mse_pos      \
        + 0.1 * _mse_rot   \
        + 0.01 * _mse_vel  \
        + 0.001 * _mse_ang

    def _reward_min_reference_tracking(self, ref_qpos, ref_qvel, state):
        """ 
        Using minimal coordinates. Improves accuracy of joint angle tracking.
        """
        pos = jp.concatenate([
        state.pipeline_state.qpos[:3],
        state.pipeline_state.qpos[7:]])
        pos_targ = jp.concatenate([
        ref_qpos[:3],
        ref_qpos[7:]])
        pos_err = jp.linalg.norm(pos_targ - pos)
        vel_err = jp.linalg.norm(state.pipeline_state.qvel- ref_qvel)

        return pos_err + vel_err

    def _reward_feet_height(self, feet_pos, feet_pos_ref):
        return jp.sum(jp.abs(feet_pos - feet_pos_ref)) # try to dr
    
    def convertLocaDiff(self, data:base.State):
        pos, rot, vel, ang = data.x.pos, data.x.rot, data.xd.vel, data.xd.ang

        root_pos = pos[0] 

        relative_pos = pos - root_pos
        
        #re-arrange all elements of the root quaterion as as xyzw
        rot_xyzw_raw = rot[:, [1, 2, 3, 0]]
        
        #normalize quaternion to make it unit magnitude
        rot_xyzw = diff_quat.quat_normalize(rot_xyzw_raw)
        #normalize root rot
        root_rot_xyzw = diff_quat.quat_normalize(rot_xyzw[0])  
        
        #root_inverse = quat_inverse(root_rot_xyzw)
        normalized_rot_xyzw = diff_quat.quat_mul_norm(diff_quat.quat_inverse(root_rot_xyzw), rot_xyzw)
        normalized_pos = diff_quat.quat_rotate(diff_quat.quat_inverse(root_rot_xyzw), relative_pos)
        normalized_vel = diff_quat.quat_rotate(diff_quat.quat_inverse(root_rot_xyzw), vel)
        normalized_ang = diff_quat.quat_rotate(diff_quat.quat_inverse(root_rot_xyzw), ang)

        normalized_rot = normalized_rot_xyzw[:, [3, 0, 1, 2]]
        
        #we just pass the relative pos
        
        return relative_pos, normalized_rot, normalized_vel,normalized_ang
    
    