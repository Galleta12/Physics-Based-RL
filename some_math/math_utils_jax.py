import jax
from jax import numpy as jp


#got it from jax
def axis_angle_to_quat(axis: jax.Array, angle: jax.Array) -> jax.Array:
  """Provides a quaternion that describes rotating around axis by angle.

  Args:
    axis: (3,) axis (x,y,z)
    angle: () float angle to rotate by

  Returns:
    A quaternion that rotates around axis by angle
  """
  s, c = jp.sin(angle * 0.5), jp.cos(angle * 0.5)
  return jp.insert(axis * s, 0, c)


def quat_norm(q):
    """Calculate the norm of the quaternion."""
    return jp.sqrt(jp.sum(q**2))


def quat_normalize(q):
    """Normalize the quaternion."""
    return q / quat_norm(q)

#Jax quat
def quat_inv(q):
    """Calculates the inverse of any quaternion."""
    norm_q = quat_norm(q)
    return jp.array([q[0], -q[1], -q[2], -q[3]]) / norm_q**2


def quat_mul(u: jax.Array, v: jax.Array) -> jax.Array:
  """Multiplies two quaternions.

  Args:
    u: (4,) quaternion (w,x,y,z)
    v: (4,) quaternion (w,x,y,z)

  Returns:
    A quaternion u * v.
  """
  return jp.array([
      u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
      u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
      u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
      u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
  ])




def compute_quaternion_for_joint(joint_axis,joint_angle):
    
    joint_quat = axis_angle_to_quat(joint_axis,joint_angle)
    return joint_quat



def combine_quaterions_joint_3DOFS(quats):
    # Combine rotations in XYZ order, ensure the multiplication reflects this
    #first element is w the real number
    combined_quat = quat_mul(quat_mul(quats[0], quats[1]),quats[2])
    return combined_quat
