from pyquaternion import Quaternion
import numpy as np

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
