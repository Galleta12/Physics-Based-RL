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

class HumanoidTemplate(PipelineEnv):
    def __init__(
      self,
      **kwargs,
    ):
        
        reference_data = kwargs.pop('referece_data')
        model_path = kwargs.pop('model_path')
        args = kwargs.pop('args')
        
        
        path = epath.Path(model_path).as_posix()
        mj_model = mujoco.MjModel.from_xml_path(path)
        
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        mj_model.opt.iterations = 1
        mj_model.opt.ls_iterations = 6
        
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
        
        #the gains for the pd controller
        self.kp_gains,self.kd_gains = generate_kp_kd_gains()
        
        #this it is the lenght, of the trajectory
        self.rollout_lenght = jp.array(self.reference_trajectory_qpos.shape[0])
        
        self.rot_weight =  args.rot_weight
        self.vel_weight =  args.vel_weight
        self.ang_weight =  args.ang_weight
        self.reward_scaling= args.reward_scaling
        
        #for now it will be the same size
        self.cycle_len = jp.array(args.cycle_len) if args.cycle_len !=0 else self.rollout_lenght  

        
        #ind for the end-effect reward of deepmimic
        #this is located on the geom_xpos
        #right_wrist,left_writst,right_ankle,left_ankle
        #the geom_xpos has a shape of 16x3
        # Can decrease jit time and training wall-clock time significantly.
        self.dict_ee = jp.array([6,9,12,15])
        
        self.pipeline_step = jax.checkpoint(self.pipeline_step, 
        policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
        
        
        
    def set_pd_callback(self,pd_control):
        self.pd_function = pd_control
    
    
    #get a reference state from the referece trajectory
    #this is used on the reset only on the reset
    def get_reference_state(self,idx):
        
        #step_index=jp.array(idx,int)
        step_index=idx
        
        ref_qp = self.reference_trajectory_qpos[step_index]
        ref_qv = self.reference_trajectory_qvel[step_index]
        #now I will return a state depending on the index and the reference trajectory
        return self.pipeline_init(ref_qp,ref_qv)
    
    #set the reference state during the steps
    #this is imporant, since we keep two states at the same time
    #this will use its own sys
    def set_ref_state_pipeline(self,step_index,data:mjx.Data):
        ref_qp = self.reference_trajectory_qpos[step_index]
        ref_qv = self.reference_trajectory_qvel[step_index]
        #calculate the position of the refernece motion with forward kinematics
        # ref_data = data.replace(qpos=ref_qp,qvel=ref_qv)
        # ref_data = mjx.forward(self.sys, ref_data)
        
        # return ref_data
        #now I will return a state depending on the index and the reference trajectory
        return self._pipeline.init(self.sys_reference, ref_qp, ref_qv, self._debug)
        
    
    
    #the standard reset for all the agents derived from this class
    def reset(self, rng: jax.Array) -> State:
        
        
        
        #set this as zero
        reward, done,zero = jp.zeros(3)
        #start at the initial state
        data = self.get_reference_state(0)       
        # qvel = jp.zeros(self.sys.nv)
        # qpos =  self.sys.qpos0
        # data = self.pipeline_init(qpos,qvel) 
        
        metrics = {'pose_error': 0.0}
        #jax.debug.print("obs: {}",obs.shape)
        #the obs should be size 193?
                
        state_info = {
            'rng': rng,
            'steps': 0.0,
            'reward_tuple': {
                'reference_position': 0.0,
                'reference_rotation': 0.0,
                'reference_velocity': 0.0,
                'reference_angular': 0.0
            },
            'index_step': 0.0,
        
        }
        
        obs = self._get_obs(data, state_info)    
        #metrics = {}
        #save the infor on the metrics
        for k in state_info['reward_tuple']:
            metrics[k] = state_info['reward_tuple'][k]
        
        
        state = State(data, obs, reward, done, metrics,state_info)
        
        return jax.lax.stop_gradient(state)
    
    
    
    def step(self, state: State, action: jp.ndarray) -> State:
        
        #this doesnt increment it might increment with the algorithms?
        index_new =jp.array(state.info['steps']%self.cycle_len, int)
        
        #jax.debug.print("new idx: {}",index_new)
        
        current_step_inx = jp.array(state.info['index_step'] + 1,int)
        #updated in the info
        state.info['index_step'] =  state.info['index_step'] + 1.0
        

        #get the reference state
        current_state_ref = self.set_ref_state_pipeline(current_step_inx,state.pipeline_state)
        
         
        #perform forward kinematics to do the same movement that is on the reference trajectory
        #this is just for demostration that the trajectory data is working properly
        data = self.pipeline_init(current_state_ref.qpos, current_state_ref.qvel)
        
        obs = self._get_obs(data, state.info)
        
        #here I will do the fall
        #check on the z axis
        fall=0.0
        fall = jp.where(data.qpos[2] < 0.2, 1.0, 0.0)
        fall = jp.where(data.qpos[2] > 1.7, 1.0, fall)
        
        
        
        reward, reward_tuple = self.compute_rewards_diffmimic(data,current_state_ref)
        
            
        #state mangement
        state.info['reward_tuple'] = reward_tuple
        
        for k in state.info['reward_tuple'].keys():
            state.metrics[k] = state.info['reward_tuple'][k]
        
        
        global_pos_state = data.x.pos
        global_pos_ref = current_state_ref.x.pos
        pose_error=loss_l2_relpos(global_pos_state, global_pos_ref)
        
        state.metrics['pose_error'] = pose_error
        
        
        return state.replace(
            pipeline_state= data, obs=obs, reward=reward, done=fall
        )
        
        
        
        
        
        
        

        
    def compute_rewards_diffmimic(self, data,current_state_ref):
        
        global_pos_state = data.x.pos
        global_rot_state = quaternion_to_rotation_6d(data.x.rot)
                
        global_vel_state = data.xd.vel
        global_ang_state = data.xd.ang
        
        
        #now for the reference trajectory
        global_pos_ref = current_state_ref.x.pos
        #jax.debug.print("global pos ref: {}",global_pos_ref)
        
        global_rot_ref = quaternion_to_rotation_6d(current_state_ref.x.rot)
              
        global_vel_ref = current_state_ref.xd.vel
        #jax.debug.print("global ref vel: {}",global_pos_ref)
                
        global_ang_ref = current_state_ref.xd.ang
        #jax.debug.print("global ref ang: {}",global_pos_ref)
         
        # jax.debug.print("rot weight: {}",self.rot_weight)
        # jax.debug.print("vel weight: {}",self.vel_weight)
        # jax.debug.print("ang weight: {}",self.ang_weight)
        # jax.debug.print("reward scaling: {}",self.reward_scaling)
        
        
        reward_tuple = {
            'reference_position': (
                mse_pos(global_pos_state, global_pos_ref)
            
            ),
            'reference_rotation': (
                mse_rot(global_rot_state, global_rot_ref)*self.rot_weight 
            ),
            'reference_velocity': (
                mse_vel(global_vel_state, global_vel_ref) *self.vel_weight
            ),
            'reference_angular': (
                mse_ang(global_ang_state, global_ang_ref)*  self.ang_weight
            )   
        }
        
        reward = -1*sum(reward_tuple.values()) * self.reward_scaling
        
        
        
    
        return reward, reward_tuple
            
    
    def compute_rewards_deepmimic(self,data,current_state_ref):
        
        pass
    
     
    
     #standard obs this will changed on other derived
    #classes from this
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

    
     
    

    
    
    def _com(self, pipeline_state: base.State) -> jax.Array:
        inertia = self.sys.link.inertia
        
        mass_sum = jp.sum(inertia.mass)
        x_i = pipeline_state.x.vmap().do(inertia.transform)
        com = (
            jp.sum(jax.vmap(jp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
        )
        return com, inertia, mass_sum, x_i  # pytype: disable=bad-return-type  # jax-ndarray


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

    
    