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
        
        # mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        # mj_model.opt.iterations = 6
        # mj_model.opt.ls_iterations = 6
            
        physics_steps_per_control_step = 10
        
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

        
        #initial position
        #the initial position
        self._inital_pos = self.reference_trajectory_qpos[0]
        
        
        #ind for the end-effect reward of deepmimic
        #this is located on the geom_xpos
        #right_wrist,left_writst,right_ankle,left_ankle
        #the geom_xpos has a shape of 16x3
        self.dict_ee = jp.array([6,9,12,15])
        
        
        
    def set_pd_callback(self,pd_control):
        self.pd_function = pd_control
    
    
    #get a reference state from the referece trajectory
    #this is used on the reset only on the reset
    def get_reference_state(self,step_index):
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
        
        ref_qp = self.reference_trajectory_qpos[0]
        
        
        state_info = {
            'rng': rng,
            'steps': 0.0,
            'reward_tuple': {
                'reference_position': 0.0,
                'reference_rotation': 0.0,
                'reference_velocity': 0.0,
                'reference_angular': 0.0
            },
            # 'step_index':0.0,
            # 'pose_error':0.0,
            # 'fall': 0.0
            
        }
        
        #metrics = {}
        #save the infor on the metrics
        for k in state_info['reward_tuple']:
            metrics[k] = state_info['reward_tuple'][k]
        
        state = State(data, obs, reward, done, metrics,state_info)
           
        return jax.lax.stop_gradient(state)
        #return state
    
    
    
    def step(self, state: State, action: jp.ndarray) -> State:
        
        #this doesnt increment it might increment with the algorithms?
        index_new =jp.array(state.info['steps']%self.cycle_len, int)
        
        #jax.debug.print("new idx: {}",index_new)
        
        initial_idx = state.metrics['step_index'] +1
        #we want to check the next idx
        current_step_inx =  jp.asarray(initial_idx%self.rollout_lenght, dtype=jp.int64)
        #jax.debug.print("current_step_idx: {}",current_step_inx)
        #get the reference state
        current_state_ref = self.set_ref_state_pipeline(current_step_inx,state.pipeline_state)
        
         
        #perform forward kinematics to do the same movement that is on the reference trajectory
        #this is just for demostration that the trajectory data is working properly
        data = self.pipeline_init(current_state_ref.qpos, current_state_ref.qvel)
        
        #here I will do the fall
        #check on the z axis
        fall = jp.where(data.qpos[2] < 0.5, 1.0, 0.0)
        fall = jp.where(data.qpos[2] > 1.7, 1.0, fall)
        
        
        obs = self._get_obs(data, current_step_inx)
        
        reward, reward_tuple = self.compute_rewards_diffmimic(data,current_state_ref)
        
        #state mangement
        state.info['reward_tuple'] = reward_tuple
        
        for k in state.info['reward_tuple'].keys():
            state.metrics[k] = state.info['reward_tuple'][k]
        
        
        global_pos_state = data.x.pos
        global_pos_ref = current_state_ref.x.pos
        pose_error=loss_l2_relpos(global_pos_state, global_pos_ref)
        
        # state.info['step_index'] = current_step_inx
        # state.info['pose_error'] = pose_error
        # state.info['fall'] = fall
        state.metrics.update(
            step_index=current_step_inx,
            pose_error=pose_error,
            fall=fall,
        )
        
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
        # return -1 * (mse_pos(global_pos_state, global_pos_ref) +
        #     self.rot_weight * mse_rot(global_rot_state, global_rot_ref) +
        #     self.vel_weight * mse_vel(global_vel_state, global_vel_ref) +
        #     self.ang_weight * mse_ang(global_ang_state, global_ang_ref)
        #     ) * self.reward_scaling
            
    

    
     
    
    #standard obs this will changed on other derived
    #classes from this
    def _get_obs(self, data: base.State, step_idx: jp.ndarray)-> jp.ndarray:
          
        current_step_inx =  jp.asarray(step_idx, dtype=jp.int64)
        
        
        #relative_pos , local_rotations,local_vel,local_ang = self.convert_local(data)
        relative_pos , local_rotations,local_vel,local_ang = self.convertLocaDiff(data)
        
        relative_pos = relative_pos[1:]
        #q_relative_pos,q_local_rotations, q_local_vel, q_local_ang = self.convertLocaDiff(data)
        
        local_rotations = local_rotations.at[0].set(data.x.rot[0])
        
        
        #convert quat to 6d root
        rot_6D= quaternion_to_rotation_6d(local_rotations)
        
        # jax.debug.print("pos mine{}",relative_pos)
        # jax.debug.print("rot mine {}", local_rotations)
        # jax.debug.print("vel mine{}", local_vel)
        # jax.debug.print("ang mine{}", local_ang)
        # jax.debug.print("pos q{}",q_relative_pos)
        # jax.debug.print("rot q {}", q_local_rotations)
        # jax.debug.print("vel q{}", q_local_vel)
        # jax.debug.print("ang q{}", q_local_ang)
        #phi
        phi = (current_step_inx % self.cycle_len) / self.cycle_len
        
        phi = jp.asarray(phi)
        
        
        #jax.debug.print("phi{}", phi)
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
    
    
    
    
    
    
    
    def _com(self, pipeline_state: base.State) -> jax.Array:
        inertia = self.sys.link.inertia
        if self.backend in ['spring', 'positional']:
            inertia = inertia.replace(
                i=jax.vmap(jp.diag)(
                    jax.vmap(jp.diagonal)(inertia.i)
                    ** (1 - self.sys.spring_inertia_scale)
                ),
                mass=inertia.mass ** (1 - self.sys.spring_mass_scale),
            )
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

    
    