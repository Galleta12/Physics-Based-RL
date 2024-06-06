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
      **kwargs,
    ):
        
        reference_data = kwargs.pop('referece_data')
        model_path = kwargs.pop('model_path')
        args = kwargs.pop('args')
        
        
        path = epath.Path(model_path).as_posix()
        
        # physics_steps_per_control_step = 10
        # kwargs['n_frames'] = kwargs.get(
        # 'n_frames', physics_steps_per_control_step)

        mj_model = mujoco.MjModel.from_xml_path(path)
        
        sys = mjcf.load_model(mj_model)

       
        self._dt = 1/60
        optimal_timestep = self._dt/5  
        
        sys = sys.tree_replace({'opt.timestep': 0.002})
        
        n_frames = kwargs.pop('n_frames', int(self._dt / 0.002))
        
        super().__init__(sys, backend='mjx', n_frames=n_frames)
        
           
        #this is to keep separate the sys of the agent
        #and the sys for the reference
        self.sys_reference = deepcopy(self.sys)
        #data for the reference
        self.reference_trajectory_qpos = jp.array(reference_data.data_pos)
        self.reference_trajectory_qvel = jp.array(reference_data.data_vel)
        print("qpos init",self.reference_trajectory_qpos.shape)
        print("qvel init",self.reference_trajectory_qvel.shape)
        
        
        self.kp_gains,self.kd_gains = generate_kp_kd_gains()
        #this it is the lenght, of the trajectory
        self.rollout_lenght = jp.array(self.reference_trajectory_qpos.shape[0])
        
        self.rot_weight =  args.rot_weight
        self.vel_weight =  args.vel_weight
        self.ang_weight =  args.ang_weight
        self.reward_scaling= args.reward_scaling
        
        #for now it will be the same size
        self.cycle_len = jp.array(self.reference_trajectory_qpos.shape[0])  

        
        
        # self._init_q = mj_model.keyframe('standing').qpos
         
        # self._default_ap_pose = mj_model.keyframe('standing').qpos[7:]
        # # Expand to entire state space.


        # kinematic_ref_qpos = make_kinematic_ref(
        # cos_wave, 25, scale=0.3, dt=self.dt)
        # kinematic_ref_qvel = make_kinematic_ref(
        # dcos_wave, 25, scale=0.3, dt=self.dt)
        # self.l_cycle = jp.array(kinematic_ref_qpos.shape[0])

        # kinematic_ref_qpos += self._default_ap_pose
        # ref_qs = np.tile(self._init_q.reshape(1, 19), (self.l_cycle, 1))
        # ref_qs[:, 7:] = kinematic_ref_qpos
        # self.kinematic_ref_qpos = jp.array(ref_qs)
        
        # ref_qvels = np.zeros((self.l_cycle, 18))
        # ref_qvels[:, 6:] = kinematic_ref_qvel
        # self.kinematic_ref_qvel = jp.array(ref_qvels)


        # self.kinematic_ref_qpos = jp.zeros((self.cycle_len, 19))
        

        # self.kinematic_ref_qvel = jp.zeros((self.cycle_len, 18))
            
        
        # Can decrease jit time and training wall-clock time significantly.
        self.pipeline_step = jax.checkpoint(self.pipeline_step, 
        policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
            
        
   
    def reset(self, rng: jax.Array) -> State:
        # Deterministic init
        
        # qpos = self.sys.qpos0
        # qvel = jp.zeros(18)
        
        # data = self.pipeline_init(qpos, qvel)
        data = self.get_reference_state(0)
        # # Position onto ground
        # pen = jp.min(data.contact.dist)
        # qpos = qpos.at[2].set(qpos[2] - pen)
        # data = self.pipeline_init(qpos, qvel)
        state_info = {
            'rng': rng,
            'steps': 0.0,
            'reward_tuple': {
                'reference_position': 0.0,
                'reference_rotation': 0.0,
                'reference_velocity': 0.0,
                'reference_angular': 0.0
            },
           'index_step':0.0
        }

        x, xd = data.x, data.xd
        obs = self._get_obs(data,state_info)
        reward, done = jp.zeros(2)
        metrics = {}
        for k in state_info['reward_tuple']:
            metrics[k] = state_info['reward_tuple'][k]
        state = State(data, obs, reward, done, metrics, state_info)
        return jax.lax.stop_gradient(state)
    
    
    #get a reference state from the referece trajectory
    #this is used on the reset only on the reset
    def get_reference_state(self,step_index):
        ref_qp = self.reference_trajectory_qpos[step_index]
        ref_qv = self.reference_trajectory_qvel[step_index]
        
        # jax.debug.print("ref_qp{}", ref_qp.shape)
        # jax.debug.print("ref_qv{}", ref_qv.shape)
        
        #now I will return a state depending on the index and the reference trajectory
        return self.pipeline_init(ref_qp,ref_qv)
    
    
    def step(self, state: State, action: jax.Array) -> State:
        
            
        action = action
        
        #jax.debug.print("action shape:{}",action.shape)
        
        
        data = self.pipeline_step(state.pipeline_state, action)
        
              
        # jax.debug.print('steps cycle: {}',state.info['steps']%self.l_cycle)
        # jax.debug.print('steps info: {}',state.info['steps'])
        current_idx = jp.array(state.info['index_step'] + 1,int)
        #updated in the info
        state.info['index_step'] =  state.info['index_step'] + 1.0
        
        # ref_qpos = self.reference_trajectory_qpos[current_idx]
        # ref_qvel = self.reference_trajectory_qvel[current_idx]
  
        # ref_qpos = self.kinematic_ref_qpos[jp.array(state.info['steps']%self.l_cycle, int)]
        # ref_qvel = self.kinematic_ref_qvel[jp.array(state.info['steps']%self.l_cycle, int)]
    
  
        # ref_qpos = self.kinematic_ref_qpos[jp.array(state.info['steps']%self.cycle_len, int)]
        # ref_qvel = self.kinematic_ref_qvel[jp.array(state.info['steps']%self.cycle_len, int)]
    
        
        # Calculate maximal coordinates
        # ref_data = data.replace(qpos=ref_qpos, qvel=ref_qvel)
        # ref_data = mjx.forward(self.sys, ref_data)
        ref_x, ref_xd = data.x, data.xd

       

        # observation data
        x, xd = data.x, data.xd
        obs = self._get_obs(data, state.info)

        # Terminate if flipped over or fallen down.
        done = 0.0
        done = jp.where(x.pos[0, 2] < 0.25, 1.0, done)
        # up = jp.array([0.0, 0.0, 1.0])
        # done = jp.where(jp.dot(math.rotate(up, x.rot[0]), up) < 0, 1.0, done)


        reward_tuple = {
        'reference_position': (
          self._reward_position(x,ref_x)
            ),
        'reference_rotation': (
            self._reward_rotation(x,ref_x) * self.rot_weight
            ),
        'reference_velocity': (
            self._reward_velocity(xd,ref_xd) * self.vel_weight
            ),
        'reference_angular': (
            self._reward_angular(xd,ref_xd) * self.ang_weight
            ),
        
        }
        
        
        
        #reward = sum(reward_tuple.values())
        reward = -1* sum(reward_tuple.values()) * self.reward_scaling
        jax.debug.print("total rewards:{}", reward)


        # state management
        state.info['reward_tuple'] = reward_tuple
       
        for k in state.info['reward_tuple'].keys():
            state.metrics[k] = state.info['reward_tuple'][k]

        state = state.replace(
            pipeline_state=data, obs=obs, reward=reward,
            done=done)
        

        return state
    
    def _get_obs(self, data: base.State, state_info: Dict[str, Any]) -> jax.Array:

        
        current_idx = jp.array(state_info['index_step'],int)
        
        relative_pos , local_rotations,local_vel,local_ang = self.convert_local(data)
        #relative_pos , local_rotations,local_vel,local_ang = self.convertLocaDiff(data)
        
        relative_pos = relative_pos[1:]
        #q_relative_pos,q_local_rotations, q_local_vel, q_local_ang = self.convertLocaDiff(data)
        
        #convert quat to 6d root
        rot_6D= quaternion_to_rotation_6d(local_rotations)

        phi = (current_idx % self.cycle_len) / self.cycle_len

        phi = jp.array(phi)

        return jp.concatenate([relative_pos.ravel(),rot_6D.ravel(),
                               local_vel.ravel(),local_ang.ravel(),phi[None]])
  
    
    
    
    
      
    def convert_local(self,data: base.State):
        
        root_pos = data.x.pos[0]
        root_quat_inv =  math.quat_inv(data.x.rot[0])
        
        #inverted_root= math.quat_mul(root_quat_inv,data.x.rot[0])
        
        #apply root-relative transformation
        def transform_to_root_local(pos, rot,vel, ang):
            local_pos = pos - root_pos
            local_rot = math.quat_mul(root_quat_inv, rot)
            local_vel = math.rotate(vel,math.quat_inv(data.x.rot[0]))
            local_ang = math.rotate(ang,math.quat_inv(data.x.rot[0]))
            
            return local_pos, local_rot, local_vel,local_ang
        #Applying the transformation to all positions and orientations
        #this is relative to the root
        local_positions, local_rotations,local_vel,local_ang = jax.vmap(transform_to_root_local)(data.x.pos, data.x.rot,data.xd.vel,data.xd.ang)
        #get rid of the zero  root pos
        #relative_pos = local_positions[1:]
        
        return local_positions,local_rotations,local_vel,local_ang
    
    
    
  
  
  
  
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

    def _reward_position(self,x,ref_x):
        
        f = lambda x, y: ((x - y) ** 2).sum(-1).mean()
        _mse_pos = f(x.pos, ref_x.pos)
        jax.debug.print("_mse_pos:{}", _mse_pos)
        
        return _mse_pos
    def _reward_rotation(self,x,ref_x):
        
        f = lambda x, y: ((x - y) ** 2).sum(-1).mean()
        _mse_rot = f(quaternion_to_rotation_6d(x.rot),
                    quaternion_to_rotation_6d(ref_x.rot))
        
        jax.debug.print("_mse_rot:{}", _mse_rot)
        
        return _mse_rot
    
    def _reward_velocity(self,xd,ref_xd):
        
        f = lambda x, y: ((x - y) ** 2).sum(-1).mean()
        _mse_vel = f(xd.vel, ref_xd.vel)
        jax.debug.print("_mse_vel:{}", _mse_vel)
        
        
        return _mse_vel 
    def _reward_angular(self,xd,ref_xd):
        
        f = lambda x, y: ((x - y) ** 2).sum(-1).mean()
        _mse_ang = f(xd.ang, ref_xd.ang)
        
        jax.debug.print("_mse_ang:{}", _mse_ang)
        
        return _mse_ang
    
    
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

