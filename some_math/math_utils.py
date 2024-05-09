from pyquaternion import Quaternion
import numpy as np
import jax
from jax import numpy as jp
from math import sqrt, pi, sin, cos, asin, acos, atan2, exp, log
#here I will change the coordinate frame
#since mujoco uses the z axis as the up vector
#x as the right vector and the y axis as the forward vector
#but my position is set as 
#since this mocap data, the x axis we can keep the same
#the y axis we need to converted to the negative Z axis
#and the z axis of pos to a y axis
#remember mujoco uses the right hand rule
#and mocap left hand, rule.
#and on mocap y is up and on mujoco z is up
def align_position(pos):
    #is one dimensional array 3,1
    assert pos.shape[0] == 3
    transformation_matrix = np.array([[1.0, 0.0, 0.0], 
                                      [0.0, 0.0, -1.0], 
                                      [0.0, 1.0, 0.0]])
    
    return np.matmul(transformation_matrix, pos)
    
#the rotation is a quat input from the txt of mocap
def align_rotation(rot):
    #the first element is the real component the other ones are the
    #imaginary components
    q_input = Quaternion(rot[0], rot[1], rot[2], rot[3])
    #transformation matrix to convert to a right handed system
    #this convert a left system to right sysmem with z up
    convert_left = Quaternion(matrix=np.array([[1.0, 0.0, 0.0], 
                                               [0.0, 0.0, -1.0], 
                                               [0.0, 1.0, 0.0]]))
    #this is the inverse, of the left
    #so a point converted to the right, by the matrix left
    #can get back to the left system with the inverse
    inverse_right = Quaternion(matrix=np.array([[1.0, 0.0, 0.0], 
                                                [0.0, 0.0, 1.0], 
                                                [0.0, -1.0, 0.0]]))
    q_output = convert_left * q_input * inverse_right
    
    return q_output.elements

#so we calculate the angular velocity
#
def calc_rot_vel(seg_0, seg_1, dura):
    #so seg_0 is the new and seg 1 is the prev
    q_0 = Quaternion(seg_0[0], seg_0[1], seg_0[2], seg_0[3])
    q_1 = Quaternion(seg_1[0], seg_1[1], seg_1[2], seg_1[3])
    
    #remember the conjugate is when we invert the imaginary components
    #so this represents the rotatiom from the prev to the new
    q_diff =  q_0.conjugate * q_1
    #we get the axis
    axis = q_diff.axis
    #the angle in radians
    angle = q_diff.angle
    
    #calculate the angular velocty, relative to the axis of rotation
    tmp_diff = angle/dura * axis
    
    diff_angular = [tmp_diff[0], tmp_diff[1], tmp_diff[2]]
    
    
    return diff_angular




def get_global_rotation_quat(parent,local):
    parent_quat = Quaternion(parent[0], parent[1], parent[2], parent[3])
    local_quat = Quaternion(local[0], local[1], local[2], local[3])
    global_rot = (parent_quat * local_quat) 
    
    return global_rot.elements



#got this from brax
def rotate_vec_quat(vec, quat):
  """Rotates a vector vec by a unit quaternion quat.

  Args:
    vec: (3,) a vector
    quat: (4,) a quaternion

  Returns:
    ndarray(3) containing vec rotated by quat.
  """
  if len(vec.shape) != 1:
    raise ValueError('vec must have no batch dimensions.')
  s, u = quat[0], quat[1:]
  r = 2 * (np.dot(u, vec) * u) + (s * s -np.dot(u, u)) * vec
  r = r + 2 * s * np.cross(u, vec)
  return r


def apply_transformation(rot_matrix, traslation, vec):
    #4x4
    transformation_matrix = np.eye(4)
    #set the top with the rot matrix
    transformation_matrix[:3,:3] = rot_matrix
    #apply the traslation on the right
    transformation_matrix[:3,3] = traslation 
    vec_homogeneous = np.append(vec,1)
    #perform the transformation
    return transformation_matrix @ vec_homogeneous

#formulat to get a rotation matrix from a quaterion
def quaternion_to_rotation_matrix(quaternion):
    """Convert a quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = quaternion
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])


#got it from the pyquaterion
def axis_angle_to_quat(axis, angle):
        """Initialise from axis and angle representation

        Create a Quaternion by specifying the 3-vector rotation axis and rotation
        angle (in radians) from which the quaternion's rotation should be created.

        Params:
            axis: a valid numpy 3-vector
            angle: a real valued angle in radians
        """
        mag_sq = np.dot(axis, axis)
        if mag_sq == 0.0:
            raise ZeroDivisionError("Provided rotation axis has no length")
        # Ensure axis is in unit vector form
        if (abs(1.0 - mag_sq) > 1e-12):
            axis = axis / sqrt(mag_sq)
        theta = angle / 2.0
        r = cos(theta)
        i = axis * sin(theta)

        quaterion = [r, i[0], i[1], i[2]]
        return np.array(quaterion)




#this function is for the pd, to generate custom
#trajectories
#I will use a cubic trajectory so it can be smooth
#for the real pd-controller I will not use that. I got this idea from mujoco-python vids on 
#youtube
def generate_trajectory(t0, tf, q0, qf):
    tf_t0_3 = (tf - t0)**3
    a0 = qf*(t0**2)*(3*tf-t0) + q0*(tf**2)*(tf-3*t0)
    a0 = a0 / tf_t0_3

    a1 = 6 * t0 * tf * (q0 - qf)
    a1 = a1 / tf_t0_3

    a2 = 3 * (t0 + tf) * (qf - q0)
    a2 = a2 / tf_t0_3

    a3 = 2 * (q0 - qf)
    a3 = a3 / tf_t0_3

    #return a0, a1, a2, a3
    return jp.array([a0, a1, a2, a3])



def start_trajectories(trajectory_dict,use_dummy=True):
    #dummy trajectory to initialize everything at zero
    if use_dummy:
        #standard trajectory
        single_trajectory = generate_trajectory(1, 3, 0, 0)
        # Repeat this trajectory 28 times to form a 28x4 array
        #28 nu, number of actuators
        trajectories = jp.tile(single_trajectory, (28, 1))
    
    #now on the specified indices changes the jp array with the ones
    #on the trajectory dict
    # Convert dictionary to index and value arrays
    indices = jp.array(list(trajectory_dict.keys()))
    new_trajectories = jp.stack(list(trajectory_dict.values()))
    # Perform the update in a single operation
    trajectories = trajectories.at[indices].set(new_trajectories)
    
    return trajectories

#compute the cubic trajectory array
def compute_cubic_trajectory(time,trajectory):
    #trajectory = start_trajectories()
    #only for the pos/angles since we dont want trajectory for the vel
    #we use the coma to specify that we want to include all rows and the 0
    #it self
    cubic_trajectory = trajectory[:,0] + trajectory[:,1] * time + \
        trajectory[:,2] * (time **2) + trajectory[:,3]* (time**3)
    return cubic_trajectory
