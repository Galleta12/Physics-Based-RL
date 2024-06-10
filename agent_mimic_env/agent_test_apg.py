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


def cos_wave(t, step_period, scale):
    _cos_wave = -jp.cos(((2*jp.pi)/step_period)*t)
    return _cos_wave * (scale/2) + (scale/2)

def dcos_wave(t, step_period, scale):
    """ 
    Derivative of the cos wave, for reference velocity
    """
    return ((scale*jp.pi) / step_period) * jp.sin(((2*jp.pi)/step_period)*t)

def make_kinematic_ref(sinusoid, step_k, scale=0.3, dt=1/50):
    """ 
    Makes trotting kinematics for the 12 leg joints.
    step_k is the number of timesteps it takes to raise and lower a given foot.
    A gait cycle is 2 * step_k * dt seconds long.
    """
    
    _steps = jp.arange(step_k)
    step_period = step_k * dt
    t = _steps * dt
    
    wave = sinusoid(t, step_period, scale)
    # Commands for one step of an active front leg
    fleg_cmd_block = jp.concatenate(
        [jp.zeros((step_k, 1)),
        wave.reshape(step_k, 1),
        -2*wave.reshape(step_k, 1)],
        axis=1
    )
    # Our standing config reverses front and hind legs
    h_leg_cmd_bloc = -1 * fleg_cmd_block

    block1 = jp.concatenate([
        jp.zeros((step_k, 3)),
        fleg_cmd_block,
        h_leg_cmd_bloc,
        jp.zeros((step_k, 3))],
        axis=1
    )

    block2 = jp.concatenate([
        fleg_cmd_block,
        jp.zeros((step_k, 3)),
        jp.zeros((step_k, 3)),
        h_leg_cmd_bloc],
        axis=1
    )
    # In one step cycle, both pairs of active legs have inactive and active phases
    step_cycle = jp.concatenate([block1, block2], axis=0)
    return step_cycle

def get_config():
  def get_default_rewards_config():
    default_config = config_dict.ConfigDict(
        dict(
            scales=config_dict.ConfigDict(
              dict(
                min_reference_tracking = -2.5 * 3e-3, # to equalize the magnitude
                reference_tracking = -1.0,
                feet_height = -1.0
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
      termination_height: float=0.25,
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
       
        kp = 230
        mj_model.actuator_gainprm[:, 0] = kp
        mj_model.actuator_biasprm[:, 1] = -kp 
        sys = mjcf.load_model(mj_model)
        
        
        self._dt = 1/60
        optimal_timestep = self._dt/5  
        
        sys = sys.tree_replace({'opt.timestep': 0.002})
        
        #n_frames = kwargs.pop('n_frames', int(self._dt / 0.002))
        n_frames = kwargs.pop('n_frames',  physics_steps_per_control_step)
        
        super().__init__(sys=sys, **kwargs)    
    
        self.termination_height = termination_height
        
        #self._init_q = mj_model.keyframe('standing').qpos
        
        self.err_threshold = 0.4 # diffmimic; value from paper.
        
        #self._default_ap_pose = mj_model.keyframe('standing').qpos[7:]
        self.reward_config = get_config()

        #self.action_loc = self._default_ap_pose
        #self.action_scale = jp.array([0.2, 0.8, 0.8] * 4)
        
        self.feet_inds = jp.array([21,28,35,42]) # LF, RF, LH, RH

        
        data_pos = jp.asarray(reference_data.data_pos)
        data_vel= jp.asarray(reference_data.data_vel)
        self.kinematic_ref_qpos = jp.asarray(data_pos[:,:28])
        self.kinematic_ref_qvel = jp.asarray(data_vel[:,:27])
        # self.kinematic_ref_qpos = jp.asarray(data_pos[:,:19])
        # self.kinematic_ref_qvel = jp.asarray(data_vel[:,:18])
        
        self.l_cycle = jp.array(self.kinematic_ref_qpos.shape[0])
        
        # Expand to entire state space.

        

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
    
    def reset(self, rng: jax.Array) -> State:
        # Deterministic init

        #qpos = jp.array(self._init_q)
        #qvel = jp.zeros(34)
        
        #data = self.pipeline_init(qpos, qvel)
        data = self.get_reference_state(0)

        # Position onto ground
        # pen = jp.min(data.contact.dist)
        # qpos = qpos.at[2].set(qpos[2] - pen)
        # data = self.pipeline_init(qpos, qvel)

        state_info = {
            'rng': rng,
            'steps': 0.0,
            'reward_tuple': {
                'reference_tracking': 0.0,
                'min_reference_tracking': 0.0,
            },
            'last_action': jp.zeros(21), # from MJX tutorial.
            'kinematic_ref': jp.zeros(28),
        }

        x, xd = data.x, data.xd
        obs = self._get_obs(data.qpos, x, xd, state_info)
        reward, done = jp.zeros(2)
        metrics = {}
        for k in state_info['reward_tuple']:
            metrics[k] = state_info['reward_tuple'][k]
        state = State(data, obs, reward, done, metrics, state_info)
        return jax.lax.stop_gradient(state)
  
    def step(self, state: State, action: jax.Array) -> State:
        action = jp.clip(action, -1, 1) # Raw action

        action = action

        data = self.pipeline_step(state.pipeline_state, action)
        
        # jax.debug.print('steps cycle: {}',state.info['steps']%self.l_cycle)
        # jax.debug.print('steps info: {}',state.info['steps'])
        
        
        ref_qpos = self.kinematic_ref_qpos[jp.array(state.info['steps']%self.l_cycle, int)]
        ref_qvel = self.kinematic_ref_qvel[jp.array(state.info['steps']%self.l_cycle, int)]
        
        # Calculate maximal coordinates
        ref_data = data.replace(qpos=ref_qpos, qvel=ref_qvel)
        ref_data = mjx.forward(self.sys, ref_data)
        ref_x, ref_xd = ref_data.x, ref_data.xd

        state.info['kinematic_ref'] = ref_qpos

        # observation data
        x, xd = data.x, data.xd
        obs = self._get_obs(data.qpos, x, xd, state.info)

        # Terminate if flipped over or fallen down.
        done = 0.0
        done = jp.where(x.pos[0, 2] < self.termination_height, 1.0, done)
        up = jp.array([0.0, 0.0, 1.0])
        done = jp.where(jp.dot(math.rotate(up, x.rot[0]), up) < 0, 1.0, done)

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

        state = state.replace(
            pipeline_state=data, obs=obs, reward=reward,
            done=done)
        
        #### Reset state to reference if it gets too far
        error = (((x.pos - ref_x.pos) ** 2).sum(-1)**0.5).mean()
        to_reference = jp.where(error > self.err_threshold, 1.0, 0.0)

        to_reference = jp.array(to_reference, dtype=int) # keeps output types same as input. 
        ref_data = self.mjx_to_brax(ref_data)

        data = jax.tree_util.tree_map(lambda x, y: 
                                    jp.array((1-to_reference)*x + to_reference*y, x.dtype), data, ref_data)
        
        x, xd = data.x, data.xd # Data may have changed.
        obs = self._get_obs(data.qpos, x, xd, state.info)
        
        return state.replace(pipeline_state=data, obs=obs)
    
    def _get_obs(self, qpos: jax.Array, x: Transform, xd: Motion,
               state_info: Dict[str, Any]) -> jax.Array:

        inv_base_orientation = math.quat_inv(x.rot[0])
        local_rpyrate = math.rotate(xd.ang[0], inv_base_orientation)

        obs_list = []
        # yaw rate
        obs_list.append(jp.array([local_rpyrate[2]]) * 0.25)
        # projected gravity
        obs_list.append(
            math.rotate(jp.array([0.0, 0.0, -1.0]), inv_base_orientation))
        # motor angles
        angles = qpos[7:19]
        obs_list.append(angles)
        # last action
        obs_list.append(state_info['last_action'])
        # kinematic reference
        kin_ref = self.kinematic_ref_qpos[jp.array(state_info['steps']%self.l_cycle, int)]
        obs_list.append(kin_ref[7:]) # First 7 indicies are fixed

        obs = jp.clip(jp.concatenate(obs_list), -100.0, 100.0)

        return obs
  
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