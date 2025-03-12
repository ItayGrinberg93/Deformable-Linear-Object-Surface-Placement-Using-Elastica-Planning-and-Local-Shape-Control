"""
UR robot uses an axis-angle convention to describe the orientation Rx,Ry,Rz (i.e rotation vector).
This file includes useful functions for controlling orientation and converting angles representation
"""

from scipy.spatial.transform import Rotation as R
import numpy as np
import quaternion as qt
from numpy import linalg as LA


def delta_rotation_vector(q_start,q_goal):
    """
    The function computes the rotation which is required to go from start orientation to goal orientation.
    :paramq_start: quaternions of rotation which represents the start orientation.
    :paramq_goal: quaternions of rotation which represents the goal orientation
    :return: vec_delta (3D ndarray) - the delta rotation vector,
             theta (float) - the angle of rotation in the range [-pi, pi], n_hat - the direction of rotation.
    """
    # Fill in your implenatation of delta_rotation_vector() function.
    # Compute the delta rotation vector, the angle of rotation and the direction of rotation.
    # There are always 2 rotations that can be applied to go from start to goal orientation.
    # Make sure you compute the rotation that its angle is within the range [-pi, pi].

    q_delta =q_goal *q_start**-1

    w_cliped = max( min(q_delta.w, 1), -1 )  # Insure that w is within the range [-1,1] so arccos could be applied.
    theta = 2 * np.arccos(w_cliped) # The same as using  q_delta.angle()
    # Notice that the use of 2 * np.arccos(q_delta.w) or 2 * q_delta.angle() returns angle in the range [0, 2*pi] instead of [-pi, pi].
    # To insure that theta would be in [-pi, pi] (atan2 can be use instead):
    if theta > np.pi:
        theta = theta - 2*np.pi
    elif theta < -np.pi:
        theta = theta + 2*np.pi

    # Compute the direction of the rotation vector - n_hat:
    sin_theta_div_2 = np.sin(theta/2)
    n_size = abs(sin_theta_div_2)  # The size of the vector is the absolute value of the sin(theta/2)
    n_size = max(n_size, 1e-10)   # Avoid size of 0 by replace it with a very small value
    n_vec = np.array([q_delta.x, q_delta.y, q_delta.z])
    n_hat = n_vec/n_size

    # # Another equivalent option is to use:
    # n_size = max(LA.norm(q_delta.vec), 1e-10)   # Avoid size of 0 by replace it with a very small value
    # n_hat = q_delta.vec / n_size

    # Compute the rotation vector that corresponds to q_delta :
    vec_delta = theta * n_hat

    # # Notice that the built-in function - quaternion.as_rotation_vector(q_delta) can return a rotation vector with |angle| > pi in an oposite direction.
    # # quaternion.as_rotation_vector(q_delta) is the same as applying q_delta.angle() * n_hat
    # # To use the built-in as_rotation_vector() you should apply this block of if-statements:
    # vec_delta = qt.as_rotation_vector(q_delta)
    # abs_theta = LA.norm(vec_delta)
    # n_hat = vec_delta/abs_theta
    # if abs_theta > np.pi and abs_theta < 2*np.pi:
    #     n_hat = - vec_delta/abs_theta
    #     abs_theta = abs(abs_theta - 2*np.pi)
    #     vec_delta = abs_theta * n_hat
    # theta = abs_theta

    return vec_delta, theta, n_hat


def orientation_error(self, desired, current):
    """
    This function calculates a 3-dimensional orientation error vector for use in the
    impedance controller. It does this by computing the delta rotation between the
    inputs and converting that rotation to exponential coordinates (axis-angle
    representation, where the 3d vector is axis * angle).
    See https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation for more information.
    Optimized function to determine orientation error from matrices

    Args:
        desired (np.array): 2d array representing target orientation matrix
        current (np.array): 2d array representing current orientation matrix

    Returns:
        np.array: 2d array representing orientation error as a matrix
    """
    rc1 = current[0:3, 0]
    rc2 = current[0:3, 1]
    rc3 = current[0:3, 2]
    rd1 = desired[0:3, 0]
    rd2 = desired[0:3, 1]
    rd3 = desired[0:3, 2]

    error = 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))

    return error


def Robot2Euler(orientation):
    """
    Convert axis-angle to euler xyz convention
    :param orient_array: np.array([Rx,Ry,Rz]) from the robot pose
    :return: euler angles in [rad]
    """

    temp = R.from_rotvec(orientation)
    euler = temp.as_euler("xyz", degrees=False)
    return np.array(euler)


def Euler2Robot(euler_angles):
    """
    Convert euler zyx angle to axis-angle
    :param: array of euler angles in xyz convention
    :return:  np.array([Rx,Ry,Rz])
    """
    temp2 = R.from_euler('xyz', euler_angles, degrees=False)
    axis_angles = temp2.as_rotvec()
    return np.array(axis_angles)


def Axis2Vector(axis_angles):
    """
    Convert axis-angle representation to the rotation vector form
    :param axis_angles: [Rx,Ry,Rz]
    :return: rot = [theta*ux,theta*uy,theta*uz] where:
    size is "theta"
    direction [ux,uy,uz] is a rotation vector
    """
    # axis_deg = np.rad2deg(axis_angles)
    size = np.linalg.norm(axis_angles)  # np.linalg.norm(axis_deg)
    if size > 1e-8:
        direction = axis_angles / size  # axis_deg/size
    else:
        direction = np.array([0,0,0])
    return size, direction


def Rot_matrix(angle, axis):
    if axis == 'x':
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    elif axis == 'y':
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == 'z':
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    raise RuntimeError('!!Error in Rot_matrix(angle,axis): axis must take the values: "x","y",or "z" as characters!!')
    return 'Error in Rot_matrix(angle,axis)'


def RotationVector(current_orientation, desired_orientation, angles_format='axis'):
    """
    Outputs the rotation vector needed to rotate from
    """
    if angles_format == 'axis':
        R01 = R.from_rotvec(current_orientation).as_matrix()
        R10 = R01.T
        R02 = R.from_rotvec(desired_orientation).as_matrix()
        R12 = np.dot(R10, R02)

        Vrot1 = R.from_matrix(R12).as_rotvec()
        Vrot0 = np.dot(R01, Vrot1)  #   Vrot = R01*R12=R02 (in a vector form)
        return Vrot0

    raise ValueError('!!Error in rotation_vector(): argument "angles_format" must get the values: "axis"!!')
    return 'Error'


def Rot_marix_to_axis_angles(Rot_matrix):
    Rotvec = R.from_matrix(Rot_matrix).as_rotvec()
    return Rotvec


def Gripper2Base_matrix(axis_angles_reading):
    R0t = R.from_rotvec(axis_angles_reading).as_matrix()
    return R0t


def Base2Gripper_matrix(axis_angles_reading):
    Rt0 = R.from_rotvec(axis_angles_reading).as_matrix().T
    return Rt0


def Tool2Base_vec(axis_angles_reading,vector):
    # "vector" is the vector that one would like to transform from Tool coordinate sys to the Base coordinate sys
    R0t = R.from_rotvec(axis_angles_reading).as_matrix()
    return R0t@vector


def Base2Tool_vec(axis_angles_reading,vector):
    # "vector" is the vector that one would like to transform from Base coordinate sys to the Tool coordinate sys.
    Rt0 = (R.from_rotvec(axis_angles_reading).as_matrix()).T
    return Rt0@vector


def Tool2Base_multiple_vectors(axis_angles_reading,matrix):
    # "matrix" is matrix with all the vectors the one want to translate from Tool sys to Base sys.
    # matrix.shape should be nX3, when n is any real number of vectors.
    R0t = R.from_rotvec(axis_angles_reading).as_matrix()
    return (R0t@(matrix.T)).T


def Base2Tool_multiple_vectors(axis_angles_reading,matrix):
    # "matrix" is matrix with all the vectors the one want to translate from Base sys to Tool sys.
    # matrix.shape should be nX3, when n is any real number of vectors
    Rt0 = (R.from_rotvec(axis_angles_reading).as_matrix()).T
    return (Rt0@(matrix.T)).T

def Base2Tool_sys_converting(coordinate_sys,pose_real,pose_ref,vel_real,vel_ref,F_internal,F_external,force_reading):
    # "coordinate_sys" is the axis_angle vector which represent the rotation vector between Base sys to Tool sys.

    REAL_DATA_tool = Base2Tool_multiple_vectors(coordinate_sys, np.block(
        [[pose_real[:3]], [pose_real[3:]], [vel_real[:3]], [vel_real[3:]]]))
    [pose_real[:3], pose_real[3:], vel_real[:3], vel_real[3:]] = [REAL_DATA_tool[0], REAL_DATA_tool[1],
                                                                  REAL_DATA_tool[2], REAL_DATA_tool[3]]
    REF_DATA_tool = Base2Tool_multiple_vectors(coordinate_sys, np.block(
        [[pose_ref[:3]], [pose_ref[3:]], [vel_ref[:3]], [vel_ref[3:]]]))
    [pose_ref[:3], pose_ref[3:], vel_ref[:3], vel_ref[3:]] = [REF_DATA_tool[0], REF_DATA_tool[1], REF_DATA_tool[2], REF_DATA_tool[3]]

    FORCE_DATA_tool = Base2Tool_multiple_vectors(coordinate_sys,
                                                    np.block([[force_reading[:3]], [F_internal[:3]], [F_external[:3]]]))
    [force_reading[:3], F_internal[:3], F_external[:3]] = [FORCE_DATA_tool[0], FORCE_DATA_tool[1], FORCE_DATA_tool[2]]

    MOMENT_DATA_tool = Base2Tool_multiple_vectors(coordinate_sys, np.block(
        [[force_reading[3:]], [F_internal[3:]], [F_external[3:]]]))
    [force_reading[3:], F_internal[3:], F_external[3:]] = [MOMENT_DATA_tool[0], MOMENT_DATA_tool[1],
                                                           MOMENT_DATA_tool[2]]
    return pose_real,pose_ref,vel_real,vel_ref,F_internal,F_external,force_reading
