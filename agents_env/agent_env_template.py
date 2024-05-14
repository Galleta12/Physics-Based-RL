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
#the template will have as input, the duration of the reference motion
#the reference motion itself, a reference motion dictionary with useful data

#I will create here the environemtn for my body
class HumanoidDiff(PipelineEnv):
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
        #self._dt = 1.0 / 30  # Policy execution frequency
                
        # # Set the optimal physics simulation timestep (1.2kHz)
        # optimal_timestep = 1.0 / 1200  # Physics simulation frequency 
        
        
        # sys = sys.tree_replace({'opt.timestep': optimal_timestep, 'dt':  optimal_timestep })
        # #the num of frames will be 40, make play with this more  
        
        # n_frames = kwargs.pop('n_frames', int(self._dt / sys.opt.timestep))
        
        self._dt = 1/60
        # arbitray selection _dt/5 we can also use 0.0333 like diffmimic
        #optimal_timestep = self._dt/5  
        optimal_timestep = self._dt/5   
        #sys = sys.tree_replace({'opt.timestep': optimal_timestep, 'dt': optimal_timestep})
        sys = sys.tree_replace({'opt.timestep': 0.002, 'dt': 0.002})
        
      
        n_frames = kwargs.pop('n_frames', int(self._dt / 0.002))
        
        
        
        super().__init__(sys, backend='mjx', n_frames=n_frames)
                
        self.reference_trajectory_qpos = reference_trajectory_qpos
        self.reference_trajectory_qvel = reference_trajectory_qvel
        self.duration_trajectory = duration_trajectory
        self.dict_duration = dict_duration
        self.link_type()
        #save the joints idx of one dofs
        self.set_joints_one_idx()
    #just to change the trajectory if needed
    def set_new_trajectory(self,trajectory:SimpleConverter):
        
        self.reference_trajectory_qpos = jp.asarray(trajectory.data_pos)
        self.reference_trajectory_qvel = jp.asarray(trajectory.data_vel)
        self.duration_trajectory = trajectory.total_time
        self.dict_duration = trajectory.duration_dict

    #return a list of joint type without root
    def link_type(self):
        # Remove the first character ('f') and convert the rest to a list of integers
        link_types_numbers = [int(char) for char in self.sys.link_types[1:]]

        # Convert the list of integers to a JAX array
        self.link_types_array_without_root = jp.array(link_types_numbers)

    #here I will set the joints indexes 
    #the joints that only have one dofs
    #the last paremeter X doesnt matter it doesnt mean anything
    #the actuator idx is depending on the nu dim, so is the qpos
    #after getting rid of the first 7 values
    def set_joints_one_idx(self):
        self.right_elbow_joint = util_data.get_actuator_indx(self.sys.mj_model,'right_elbow','X')
        self.left_elbow_joint = util_data.get_actuator_indx(self.sys.mj_model,'left_elbow','X')
        self.right_knee_joint = util_data.get_actuator_indx(self.sys.mj_model,'right_knee','X')
        self.left_knee_joint = util_data.get_actuator_indx(self.sys.mj_model,'left_knee','X')

        self.one_dofs_joints_idx = jp.array([self.right_elbow_joint,self.left_elbow_joint,
                                             self.right_knee_joint,self.left_knee_joint])     

    def hinge_to_quat(self,current_joints):
        #jnt_axis without free joint
        axis_hinge = self.sys.jnt_axis[1:]
        vmap_compute_quaternion = jax.vmap(compute_quaternion_for_joint, in_axes=(0, 0))     
        hinge_quat = vmap_compute_quaternion(axis_hinge,current_joints) 
        return hinge_quat
    
    
    #@partial(jit, static_argnums=(1, 3,))
    def local_quat(self,current_root_rot,current_x_quat,hinge_quat,one_dofs_joints_idx, link_types_array_without_root):
        # Mask to filter out the quaternions that are not to be combined in triples
        mask = jp.ones(hinge_quat.shape[0], dtype=bool)
        mask = mask.at[one_dofs_joints_idx].set(False)

        #index 0 since is a tuple and we want the indices that are index 0
        combinable_indices = jp.where(mask)[0]
        #negate it to get the inverse
        non_combinable_indices = jp.where(~mask)[0]

        no_grouped_quaterions = hinge_quat[non_combinable_indices]

        #select and store these indices
        # Assuming the remaining quaternions are multiple of three
        # Reshape the array to (-1, 3, 4) where 3 is the number of quaternions to be combined and 4 is the quaternion dimension
        grouped_quaternions = hinge_quat[combinable_indices].reshape(-1, 3, 4)
        #this will be applied on the first axs
        vmap_combined_quat = jax.vmap(combine_quaterions_joint_3DOFS)

        quat_combined = vmap_combined_quat(grouped_quaternions)
        
        #there are 13 links -12, since we will merge the root at the end the shape 1 is 4 for the quat
        quat_loc_all_joints = jp.zeros((current_x_quat[0]-1,current_x_quat[1]))

        # Create a mask where each position is True if the corresponding link type is 3
        link_types_mask = link_types_array_without_root == 3

        filter_out_jnt_type_idx = jp.where(link_types_mask)

        filter_out_jnt_type_one_dofs = jp.where(~link_types_mask)

        quat_loc_all_joints = quat_loc_all_joints.at[filter_out_jnt_type_idx].set(quat_combined)

        quat_loc_all_joints = quat_loc_all_joints.at[filter_out_jnt_type_one_dofs].set(no_grouped_quaterions)

        #we reshape the currenr root, rot to get a 1x3 and avoid errors
        quat_loc_all_joints = jp.concatenate([current_root_rot.reshape(1,-1),quat_loc_all_joints],axis=0)

        return quat_loc_all_joints

            
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
   


