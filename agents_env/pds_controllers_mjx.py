
import jax
from jax import numpy as jp
import numpy as np
import mujoco
from mujoco import mjx
from brax import base
from brax.envs.base import Env, PipelineEnv, State
from jax.scipy.linalg import cho_factor, cho_solve
from some_math.math_utils import compute_cubic_trajectory


#first compute a standard pd_controller
#for now the angular velocity target will be zero
#since we dont want to add an extra angular velocity.
#we want to focus more on the angles
def standard_pd_controller_mjx(target,mj_model,mj_data,mjx_data,
                           q,qdot,kp_array,kd_array,time,dt):
    #extract the root from the q pos and qdot
    q = q[7:]
    
    qdot = qdot[6:]
    
    #the error pos
    error = q - target
    
    error_angular = -kd_array* qdot
    
    error_pos = -kp_array*error
    
    
    return error_pos + error_angular







def feedback_pd_controller_mjx(target,mj_model,
                           mj_data,mjx_data,q,qdot,kp_array,kd_array,time,dt):
    
    
    q = q[7:]
    qdot = qdot[6:]
    #calculate error position
    error = q - target
    
    error_pos = -kp_array*error
    
    #calculate angular eeor
    error_angular = -kd_array* qdot
    
    #Calculate pd control
    pd_control = error_pos + error_angular
    
    #concatenate to work with the corilis and Mass matrix
    #since the mass and F(corolis and gravity)
    pd_control = jp.concatenate((jp.zeros(6),pd_control))
        
    #corolis plus gravity, the dim is nv x 1 so the same as velocity
    #dim nv 34
    F = mjx_data.qfrc_bias.copy()
    
    #now get the mass matrix nv x nv
    M = np.zeros((mj_model.nv,mj_model.nv)) 
    
    mujoco.mj_fullM(mj_model, M, mj_data.qM)
    
    M = jax.device_put(M)
    F = jax.device_put(F)
    
    #we add an extra dummy dim on pd_control to avoid eror on the multiplication
    tau_pd = M @ pd_control[:,None]
    #the same witht the F
    tau = tau_pd + F[:,None]    
    #get it back to one dimensional array
    #we get rid of the 6 values, since we want to be in dim nu=28
    #since this is the dim of the actuators   
    return tau.squeeze()[6:]


#initialize corilis, mass and external forces
#and set kp and kd as diagonal matrices
def init_corolis_mass_external_diagonal(mj_model,mj_data,mjx_data,kp_array,kd_array):
    #get the centrifugal force
    #dim (nv)
    C = mjx_data.qfrc_bias.copy()
    
    tau_ext = mjx_data.qfrc_applied.copy()  # This should represent the external forces
    #get the mass matrix fore the inertia
    M = np.zeros((mj_model.nv,mj_model.nv))
    mujoco.mj_fullM(mj_model, M,mj_data.qM)
    M = jax.device_put(M)
    C = jax.device_put(C)
    
    KP = jp.diag(kp_array)
    KD = jp.diag(kd_array)
    
    return C,M,KP,KD, tau_ext

#calculate the new mass,

def calculate_new_mass_check_if_invertible(M,KD,dt,time):
    jax.debug.print("M: {}", M)
    
    new_mass = M + (KD * dt)
    
    # add regularization at the begging to avoid error with Nan
    if time == 1:
        # regularization is called Tikhonov
        epsilon = 1e-6  # Small regularization term
        new_mass = new_mass + epsilon * jp.eye(new_mass.shape[0])
        
    # Calculate the condition number
    cond_number = jp.linalg.cond(new_mass)
    
    # Now check the condition number
    if cond_number < 1/jp.finfo(new_mass.dtype).eps:
        # It is likely invertible and well-conditioned
        pass
    else:
        # The matrix may be singular or near-singular
        print(f"Warning: High condition number ({cond_number}) indicates potential numerical instability. current time {time}")
    
    return new_mass


@jax.jit
def compute_acceleration(q_error,qdot,C,tau_ext,new_mass,KP,KD):
    
    # we add none on the indexing to add a dimension and avoid errors
    # add a dimension to vectors to fit matrix operation requirements
   
    C = C[:, None]
    q_error = q_error[:, None]
    qdot = qdot[:, None]
    tau_ext = tau_ext[:,None]
    
    # Calculate the proportional and damping forces
    prop_force = KP @ q_error
    damp_force = KD @ qdot
    
    # Combine forces for the equation
    combined_forces = -C - prop_force - damp_force + tau_ext
      
  
    # solve the equation
    #we can also solve it with this line of code, but I will use the cholesky way
    qdot_dot = jp.linalg.solve(new_mass, combined_forces)
    
    #return it back to one dim array
    return qdot_dot.squeeze()
    
#perform the stable pd controller
def stable_pd_controller_custom_trajectory_mjx(target,mj_model,mj_data,mjx_data,q,qdot,kp_array,kd_array,time,dt):
    
    #this could change more when doing it with the real angles
    target_q_next= compute_cubic_trajectory(time+dt,target)
    
    #calculate the next q erro, this is on the paper stable pd
    #this is have dim nu
    error_q_next =(q[7:] + (qdot[6:]*dt) )-target_q_next
    
    #first reshape the kp and kd by adding elemnts at the beginning 6
    #remember the size of the kp and kd is 28, and we want 34 to match
    #the dofs nv size
    kp_matrix = jp.concatenate([jp.zeros(6),kp_array])
    kd_matrix = jp.concatenate([jp.zeros(6),kd_array])
    
    #save the angular error, that is the velocity itself
    angular_error = qdot
    
    # add 6 elements to math the dim of the mass and corolis
    error_pos = jp.concatenate([jp.zeros(6), error_q_next])
    
    
    #initialize the variables for getting the acceleration equation
    C,M,KP,KD,tau_ext=init_corolis_mass_external_diagonal(mj_model,mj_data,mjx_data,kp_matrix,kd_matrix)
    #get the mass inertia matrix with the added kd diagonal
    new_mass =calculate_new_mass_check_if_invertible(M,KD,dt,time)
    
    #calculate the predicted acceleration   
    qdot_dot = compute_acceleration(error_pos,angular_error,C,tau_ext,new_mass,KP,KD)
    
    # add the predicted error for the qdot, and then add than on the principal equation
    #so this is like the  next angular error on the stable pd paper
    angular_error = angular_error + (qdot_dot*dt)
    #now get the torque avoiding the root
    tau = -kp_array * error_pos[6:] - kd_array * angular_error[6:]
    return tau
