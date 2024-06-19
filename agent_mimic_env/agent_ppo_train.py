from datetime import datetime
import functools
from IPython.display import HTML
import jax.numpy as jp
import numpy as np
import jax
from jax import config # Analytical gradients work much better with double precision.
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)
config.update('jax_default_matmul_precision', jax.lax.Precision.HIGH)
from brax import math
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
import some_math.math_utils_jax as math_jax
from utils.SimpleConverter import SimpleConverter
from utils.util_data import generate_kp_kd_gains
import some_math.quaternion_diff as diff_quat

from .agent_template import HumanoidTemplate
from .losses import *


class HumanoidPPOENV(HumanoidTemplate):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline_step = jax.checkpoint(self.pipeline_step, 
                policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
        args = kwargs.pop('args')
        
                
        self.w_p =  args.deep_mimic_reward_weights.w_p
        self.w_v =  args.deep_mimic_reward_weights.w_v
        self.w_e =  args.deep_mimic_reward_weights.w_e
        self.w_c= args.deep_mimic_reward_weights.w_c    
        
        self.w_pose =  args.deep_mimic_weights_factors.w_pose
        self.w_angular =  args.deep_mimic_weights_factors.w_angular
        self.w_efector =  args.deep_mimic_weights_factors.w_efector
        self.w_com= args.deep_mimic_weights_factors.w_com   

           
    
    def reset(self, rng: jp.ndarray) -> State:
        
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
                'reference_quaternions': 0.0,
                'reference_angular': 0.0,
                'reference_end_effector': 0.0,
                'reference_com': 0.0
            },
            'default_pos': ref_qp,
            'last_action':jp.zeros(28),
            'kinematic_ref': ref_qp     
    
        }
        
        obs = self._get_obs(data,new_step_idx_float,state_info) 
        #metrics = {}
        #save the infor on the metrics
        for k in state_info['reward_tuple']:
            metrics[k] = state_info['reward_tuple'][k]
        
        state = State(data, obs, reward, done, metrics,state_info)
           
        return jax.lax.stop_gradient(state)
        #return state
    
  
    def step(self, state: State, action: jp.ndarray) -> State:
            
        qpos = state.pipeline_state.q
        qvel = state.pipeline_state.qd
        timeEnv = state.pipeline_state.time
        #deltatime of physics
        dt = self.sys.opt.timestep
        
        action = jp.clip(action, -1, 1) # Raw action  
        #target_angles = action * jp.pi * 1.2
        #exclude the root
        target_angles = state.info['default_pos'][7:] +(action * jp.pi * 1.2)
        #target_angles = action 
        
        torque = self.pd_function(target_angles,self.sys,state,qpos,qvel,
                                 self.kp_gains,self.kd_gains,timeEnv,dt) 
        
        data = self.pipeline_step(state.pipeline_state,torque)
        #data = self.pipeline_step(state.pipeline_state,target_angles)
                
        index_new =jp.array(state.info['steps']%self.cycle_len, int)
                
        
        initial_idx = state.metrics['step_index'] +1.0
        current_step_inx =  jp.asarray(initial_idx%self.cycle_len, dtype=jp.float64)

        current_state_ref = self.set_ref_state_pipeline(current_step_inx,state.pipeline_state)

        
        fall=0.0
        fall = jp.where(data.qpos[2] < 0.5, 1.0, fall)
        fall = jp.where(data.qpos[2] > 1.7, 1.0, fall)
        
        reward, reward_tuple = self.compute_rewards_deepmimic(data,current_state_ref)
        
        #get the observations
        #state mangement
        state.info['last_action'] = action
        state.info['kinematic_ref'] = current_state_ref.qpos
        state.info['reward_tuple'] = reward_tuple
        
        obs = self._get_obs(data, current_step_inx,state.info)
        
        
        for k in state.info['reward_tuple'].keys():
            state.metrics[k] = state.info['reward_tuple'][k]
        
        
        
        global_pos_state = data.x.pos
        global_pos_ref = current_state_ref.x.pos
        pose_error=loss_l2_relpos(global_pos_state, global_pos_ref)
        
        
        state.metrics.update(
            step_index=current_step_inx,
            pose_error=pose_error,
            fall=fall,
        )
        
        return state.replace(
            pipeline_state= data, obs=obs, reward=reward, done=fall
        )
        
    
    
    
    #class for testing step_custom where we just check the pd controller
    def step_custom(self, state: State, action: jp.ndarray) -> State:
        
        initial_idx = state.metrics['step_index'] +1.0
        current_step_inx =  jp.asarray(initial_idx%self.cycle_len, dtype=jp.float64)
        current_state_ref = self.set_ref_state_pipeline(current_step_inx,state.pipeline_state)
      
        qpos = state.pipeline_state.q
        qvel = state.pipeline_state.qd
        timeEnv = state.pipeline_state.time
        #deltatime of physics
        dt = self.sys.opt.timestep
        
        #target_angles = action * jp.pi * 1.2
        #exclude the root
        target_angles = state.info['default_pos'][7:] +(current_state_ref.qpos[7:]* jp.pi * 1.2)
        #target_angles = action 
        
        torque = self.pd_function(current_state_ref.qpos[7:],self.sys,state,qpos,qvel,
                                 self.kp_gains,self.kd_gains,timeEnv,dt) 
        
        data = self.pipeline_step(state.pipeline_state,torque)
        #data = self.pipeline_step(state.pipeline_state,target_angles)
        #data = self.pipeline_init(current_state_ref.qpos,current_state_ref.qvel)
        
        index_new =jp.array(state.info['steps']%self.cycle_len, int)
                
        
        
        fall=0.0
        fall = jp.where(data.qpos[2] < 0.5, 1.0, fall)
        fall = jp.where(data.qpos[2] > 1.7, 1.0, fall)
        
        reward, reward_tuple = self.compute_rewards_deepmimic(data,current_state_ref)
        
        #get the observations
        #state mangement
        state.info['last_action'] = action
        state.info['kinematic_ref'] = current_state_ref.qpos
        state.info['reward_tuple'] = reward_tuple
        
        obs = self._get_obs(data, current_step_inx,state.info)
        
        
        for k in state.info['reward_tuple'].keys():
            state.metrics[k] = state.info['reward_tuple'][k]
        
        
        
        global_pos_state = data.x.pos
        global_pos_ref = current_state_ref.x.pos
        pose_error=loss_l2_relpos(global_pos_state, global_pos_ref)
        
        
        state.metrics.update(
            step_index=current_step_inx,
            pose_error=pose_error,
            fall=fall,
        )
        
        return state.replace(
            pipeline_state= data, obs=obs, reward=reward, done=fall
        )
        
    
    
 
        
    def compute_rewards_deepmimic(self,data,current_state_ref):
        
       
        #local_pos , local_rotations,local_vel,local_ang =self.convertLocaDiff(data)
        #ref_local_pos , ref_local_rotations,ref_local_vel,ref_local_ang = self.convertLocaDiff(current_state_ref)
        
        current_joint_rotations = self.get_local_joints_rotation(data) 
        ref_joint_rotations = self.get_local_joints_rotation(current_state_ref) 
        # #we only want angular velocities
        current_joints_ang =  data.qvel[3:]
        ref_ang =   current_state_ref.qvel[3:]
       
       
        #jax.debug.print("this is the local ang curret:{}",local_ang)
        #jax.debug.print("this is the local ang ref:{}",ref_local_ang)
                
        #jax.debug.print("this is the local ang rotation{}:",local_rotations)
        #jax.debug.print("this is the local ang rotation{}:",ref_local_rotations)

        
        
        #data for the end effector reward
        current_ee = data.geom_xpos[self.dict_ee]
        current_ref_ee = current_state_ref.geom_xpos[self.dict_ee]
        
        #jax.debug.print('current end{}',current_ee)
        #jax.debug.print('currentref end{}',current_ref_ee)
        
        
        com_current,_,_,_ = self._com(data)
        ref_com,_,_,_ = self._com(current_state_ref)
        # com_current = self.get_com(data)
        # ref_com = self.get_com(current_state_ref)
        #jax.debug.print('current com{}',com_current)
        #jax.debug.print('current com{}',ref_com)
        
        reward_tuple = {
            'reference_quaternions': (
                self.quat_diff(current_joint_rotations,ref_joint_rotations,
                               data,current_state_ref)  
            ),
            'reference_angular': (
                self.angular_diff(current_joints_ang , ref_ang)  
            ),
            'reference_end_effector': (
                self.end_effector_diff(current_ee,current_ref_ee) 
            ),
            'reference_com': (
                self.com_diff(com_current, ref_com) 
            )   
        }
        
    
       
        reward_quat = self.w_p *jp.exp(-reward_tuple['reference_quaternions']) 
        reward_ang = self.w_v *jp.exp(-reward_tuple['reference_angular']) 
        reward_end = self.w_e *jp.exp(-reward_tuple['reference_end_effector']) 
        reward_com = self.w_c *jp.exp(-reward_tuple['reference_com']) 
       
        #reward = sum(reward_tuple.values())
        #jax.debug.print("quat error{}:",quaterion_error)
        #jax.debug.print("ang error{}:",angular_error)
        #jax.debug.print("end error{}:",end_error)
        #jax.debug.print("com error{}:",com_error)
        reward = reward_quat + reward_ang + reward_end + reward_com
        
        return reward, reward_tuple
        
    
    def com_diff(self,com_current, ref_com):
        # jax.debug.print("com result local:{}",com_current)
        # jax.debug.print("ref com result ref:{}",ref_com)
        #shape 3x1    
        com_norm = jp.linalg.norm(ref_com - com_current,ord=2)         
        com_reward = self.w_com * com_norm
        
        return com_reward
        
    
    
    def end_effector_diff(self,current_ee,current_ref_ee):
        # jax.debug.print("end diff result local:{}",current_ee)
        # jax.debug.print("end diff result ref:{}",current_ref_ee)        
        ee_norm = jp.linalg.norm(current_ref_ee - current_ee,ord=2,axis=-1)        
        #jax.debug.print("end dis:{}",ee_dist)
        ee_dist = jp.sum(ee_norm**2)
        ee_reward = self.w_efector * ee_dist
        #jax.debug.print("ee_reward:{}",ee_reward)
        
        return ee_reward
        
    
    def angular_diff(self,local_ang,ref_local_ang):
        
        # jax.debug.print("ang diff result local:{}",local_ang)
        # jax.debug.print("ang diff result ref:{}",ref_local_ang)
        ang_norm = jp.linalg.norm(ref_local_ang - local_ang,ord=2,axis=-1)
        ang_dist = jp.sum(ang_norm**2)
        ang_reward = self.w_angular * ang_dist
        
        return ang_reward
        
    
    
    #relative quat ref to quatcurrent
    def quat_diff(self, local_rotations,ref_local_rotations,data,current_state_ref):
        #in the calculation we include the root 
        current_rot6D = quaternion_to_rotation_6d(local_rotations)
        ref_rot6D = quaternion_to_rotation_6d(ref_local_rotations)
        
        #rot_dist = jp.linalg.norm(current_rot6D - ref_rot6D)
        
        #quat_reward = jp.exp(-self.w_pose * (rot_dist**2))
        #quat_reward = jp.exp(-self.w_pose * (rot_dist**2))

        
        #jax.debug.print("current rot 6d:{}",current_rot6D)
        
        #along the last axis which is the columns
        rot_norm = jp.linalg.norm(ref_rot6D-current_rot6D,ord=2,axis=-1)
        
        rot_dist = jp.sum(rot_norm**2)
        
        quat_reward = self.w_pose * rot_dist
        
        
        return quat_reward
    
    
    
    def get_com(self,data):
          
        return data.subtree_com[1]
    
    
    # def _get_obs(self, data: base.State, step_idx: jp.ndarray,state_info: Dict[str, Any])-> jp.ndarray:
          
    #     current_step_inx =  jp.asarray(step_idx, dtype=jp.int64)
                        
    #     phi = (current_step_inx % self.cycle_len) / self.cycle_len
        
    #     phi = jp.asarray(phi)
    #     #jax.debug.print("phi{}", phi)
    #     #I will add the last action
    #     last_ref = state_info['kinematic_ref']
    #     return jp.concatenate([data.qpos[2:],data.qvel,phi[None]])
        
    def get_local_joints_rotation(self,data):
        
        current_xrot = data.xquat[1:]
        #this is already in quat form
        current_qpos_root = data.qpos[3:7]
        #qpos of the joins this are scale values, since they are hinge joints
        current_qpos_joints = data.qpos[7:] 
        #now I will convert them into quaterions
        #first the joints that are onedofs, thus axis angle to quaterion
        hinge_quat = self.hinge_to_quat(current_qpos_joints)
        
        local_quat = self.local_quat(current_qpos_root,current_xrot,hinge_quat,self.one_dofs_joints_idx,self.link_types_array_without_root)
        
        return local_quat
    
    def hinge_to_quat(self,current_joints):
        #jnt_axis without free joint
        axis_hinge = self.sys.jnt_axis[1:]
        vmap_compute_quaternion = jax.vmap(math_jax.compute_quaternion_for_joint, in_axes=(0, 0))     
        hinge_quat = vmap_compute_quaternion(axis_hinge,current_joints) 
        return hinge_quat
    
    def local_quat(self,current_root_rot,current_x_quat,hinge_quat,one_dofs_joints_idx, link_types_array_without_root):
        # Mask to filter out the quaternions that are not to be combined in triples
        mask = jp.ones(hinge_quat.shape[0], dtype=bool)
        mask = mask.at[one_dofs_joints_idx].set(False)
        
        # #keep track of the true and false values
        # true_mask = jp.sum(mask)
        # false_mask = mask.size - true_mask
        
        # Indices that can be combined
        combinable_indices = jp.nonzero(mask, size=mask.size)[0]
#       #since the size will be 28, the same as the mask we only select
        #the first values and get rid of the last values, for now
        #24 and 4 is static. I wish to do this dynamic
        combinable_indices = combinable_indices[0:24]
        
        # Indices that cannot be combined
        non_combinable_indices = jp.nonzero(~mask, size=mask.size)[0]
        non_combinable_indices = non_combinable_indices[0:4]
             
        no_grouped_quaterions = hinge_quat[non_combinable_indices]

        #select and store these indices
        # Assuming the remaining quaternions are multiple of three
        # Reshape the array to (-1, 3, 4) where 3 is the number of quaternions to be combined and 4 is the quaternion dimension
        grouped_quaternions = hinge_quat[combinable_indices].reshape(-1, 3, 4)
        #this will be applied on the first axis
        vmap_combined_quat = jax.vmap(math_jax.combine_quaterions_joint_3DOFS)

        quat_combined = vmap_combined_quat(grouped_quaternions)
        #there are 13 links -12, since we will merge the root at the end the shape 1 is 4 for the quat
        quat_loc_all_joints = jp.zeros((current_x_quat.shape[0]-1,current_x_quat.shape[1]))

        #Create a mask where each position is True if the corresponding link type is 3
        link_types_mask = link_types_array_without_root == 3
        
        filter_out_jnt_type_3_idx = jp.nonzero(link_types_mask,size=link_types_mask.size)
        
        #the first row is where is the data the shape is 1,12
        #there are 8 indices for the 3 dofs and 4 for the one dofs
        filter_out_jnt_type_3_idx = jp.array(filter_out_jnt_type_3_idx)[0][0:8]
        
        filter_out_jnt_type_one_dofs = jp.nonzero(~link_types_mask,size=link_types_mask.size)
        
        filter_out_jnt_type_one_dofs = jp.array(filter_out_jnt_type_one_dofs)[0][0:4]
        
        
        quat_loc_all_joints = quat_loc_all_joints.at[filter_out_jnt_type_3_idx].set(quat_combined)

        quat_loc_all_joints = quat_loc_all_joints.at[filter_out_jnt_type_one_dofs].set(no_grouped_quaterions)

        #we reshape the currenr root, rot to get a 1x4 and avoid errors
        quat_loc_all_joints = jp.concatenate([current_root_rot.reshape(1,-1),quat_loc_all_joints],axis=0)

        return quat_loc_all_joints
    