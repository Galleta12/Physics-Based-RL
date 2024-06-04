import jax
from jax import numpy as jp
from some_math.rotation6D import quaternion_to_rotation_6d


import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from some_math.quaternion_diff import *

#got this from brax
#global space
def loss_l2_relpos(pos, ref_pos):
    relpos, ref_relpos = (pos - pos[0])[1:], (ref_pos - ref_pos[0])[1:]
    relpos_loss = (((relpos - ref_relpos) ** 2).sum(-1) ** 0.5).mean()
        
    #relpos_loss = relpos_loss
        
    return relpos_loss


def loss_l2_pos(pos, ref_pos):
    pos_loss = (((pos - ref_pos) ** 2).sum(-1)**0.5).mean()
    return pos_loss

#position in global
def mse_pos(pos, ref_pos):
    #jax.debug.print("global pos: {}", pos)
    #jax.debug.print("ref pos: {}", ref_pos)
    
    pos_loss = ((pos - ref_pos) ** 2).sum(-1).mean()
    
    
    #jax.debug.print("mse_pos: {}", pos_loss)
    
    return pos_loss

#rotation this is already in 6D form
def mse_rot(rot, ref_rot):
    rot_loss = ((rot - ref_rot) ** 2).sum(-1).mean()
    #jax.debug.print("mse_rot: {}", rot_loss)
    
    return rot_loss

def mse_vel(vel, ref_vel):
    vel_loss = ((vel - ref_vel) ** 2).sum(-1).mean()
    
    #jax.debug.print("vel loss: {}", vel_loss)
    
    return vel_loss

def mse_ang(ang, ref_ang):
    ang_loss = ((ang - ref_ang) ** 2).sum(-1).mean()
    
    #jax.debug.print("ang_loss: {}", ang_loss)
    
    return ang_loss



# def convert_to_local(qp, remove_height=False):
#     #make sure that the data is unchanged
#     pos, rot, vel, ang = qp.pos * 1., qp.rot * 1., qp.vel * 1., qp.ang * 1.
#     # normalize
#     if not remove_height:
#         root_pos = pos[:1] * jp.array([1., 1., 0.])  # shape (1, 3)
    
    
#     normalized_pos = pos - root_pos

#     # normalize
#     rot_xyzw_raw = rot[:, [1, 2, 3, 0]]  # wxyz -> xyzw
#     rot_xyzw = quat_normalize(rot_xyzw_raw)

#     root_rot_xyzw = quat_normalize(rot_xyzw[:1] * jp.array([0., 0., 1., 1.]))  # [x, y, z, w] shape (1, 4)

#     normalized_rot_xyzw = quat_mul_norm(quat_inverse(root_rot_xyzw), rot_xyzw)
#     normalized_pos = quat_rotate(quat_inverse(root_rot_xyzw), normalized_pos)
#     normalized_vel = quat_rotate(quat_inverse(root_rot_xyzw), vel)
#     normalized_ang = quat_rotate(quat_inverse(root_rot_xyzw), ang)

#     normalized_rot = normalized_rot_xyzw[:, [3, 0, 1, 2]]

#     qp_normalized = QP(pos=normalized_pos, rot=normalized_rot, vel=normalized_vel, ang=normalized_ang)

#     return qp_normalized

