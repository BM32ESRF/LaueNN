# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:46:40 2022

@author: PURUSHOT
"""
__author__ = "Ravi raj purohit PURUSHOTTAM RAJ PUROHIT, CRG-IF BM32 @ ESRF"

import numpy as np
from scipy.spatial.transform import Rotation as R

def misorientation_axis_from_delta(delta):
    """Compute the misorientation axis from the misorientation matrix.
    """
    n = np.array([delta[1, 2] - delta[2, 1], delta[2, 0] - delta[0, 2], delta[0, 1] - delta[1, 0]])
    n /= np.sqrt(
        (delta[1, 2] - delta[2, 1]) ** 2 + (delta[2, 0] - delta[0, 2]) ** 2 + (delta[0, 1] - delta[1, 0]) ** 2)
    return n

def misorientation_angle_from_delta(delta):
    """Compute the misorientation angle from the misorientation matrix.
    Compute the angle assocated with this misorientation matrix :math:`\\Delta g`.
    It is defined as :math:`\\omega = \\arccos(\\text{trace}(\\Delta g)/2-1)`.
    To avoid float rounding error, the argument is rounded to 1. if it is within 1 and 1 plus 32 bits floating 
    point precison.
    """
    cw = 0.5 * (delta.trace() - 1)
    if cw > 1. and cw - 1. < np.finfo('float32').eps:
        print('cw=%.20f, rounding to 1.' % cw)
        cw = 1.
    omega = np.arccos(cw)
    return omega

def disorientation(orientation, orientation1, symmetries):
    """Compute the disorientation another crystal orientation.
    Considering all the possible crystal symmetries, the disorientation
    is defined as the combination of the minimum misorientation angle
    and the misorientation axis lying in the fundamental zone, which
    can be used to bring the two lattices into coincidence.
    """
    the_angle = np.pi
    (gA, gB) = (orientation, orientation1)  # nicknames
    for (g1, g2) in [(gA, gB), (gB, gA)]:
        for j in range(symmetries.shape[0]):
            sym_j = symmetries[j]
            oj = np.dot(sym_j, g1)  # the crystal symmetry operator is left applied
            for i in range(symmetries.shape[0]):
                sym_i = symmetries[i]
                oi = np.dot(sym_i, g2)
                delta = np.dot(oi, oj.T)
                mis_angle = misorientation_angle_from_delta(delta)
                if mis_angle < the_angle:
                    # now compute the misorientation axis, should check if it lies in the fundamental zone
                    mis_axis = misorientation_axis_from_delta(delta)
                    the_angle = mis_angle
                    the_axis = mis_axis
                    the_axis_xyz = np.dot(oi.T, the_axis)
    the_angle = np.rad2deg(the_angle)
    return (the_angle, the_axis, the_axis_xyz)

def rotation_matrix(eulerang, order='xyz'):
    """
    input
        theta1, theta2, theta3 = rotation angles in rotation order (degrees)
        oreder = rotation order of x,y,z　e.g. XZY rotation -- 'xzy'
    output
        3x3 rotation matrix (numpy array)
    """
    theta1, theta2, theta3 = eulerang
    c1 = np.cos(theta1 * np.pi / 180)
    s1 = np.sin(theta1 * np.pi / 180)
    c2 = np.cos(theta2 * np.pi / 180)
    s2 = np.sin(theta2 * np.pi / 180)
    c3 = np.cos(theta3 * np.pi / 180)
    s3 = np.sin(theta3 * np.pi / 180)

    if order == 'xzx':
        matrix=np.array([[c2, -c3*s2, s2*s3],
                         [c1*s2, c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3],
                         [s1*s2, c1*s3+c2*c3*s1, c1*c3-c2*s1*s3]])
    elif order=='xyx':
        matrix=np.array([[c2, s2*s3, c3*s2],
                         [s1*s2, c1*c3-c2*s1*s3, -c1*s3-c2*c3*s1],
                         [-c1*s2, c3*s1+c1*c2*s3, c1*c2*c3-s1*s3]])
    elif order=='yxy':
        matrix=np.array([[c1*c3-c2*s1*s3, s1*s2, c1*s3+c2*c3*s1],
                         [s2*s3, c2, -c3*s2],
                         [-c3*s1-c1*c2*s3, c1*s2, c1*c2*c3-s1*s3]])
    elif order=='yzy':
        matrix=np.array([[c1*c2*c3-s1*s3, -c1*s2, c3*s1+c1*c2*s3],
                         [c3*s2, c2, s2*s3],
                         [-c1*s3-c2*c3*s1, s1*s2, c1*c3-c2*s1*s3]])
    elif order=='zyz':
        matrix=np.array([[c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3, c1*s2],
                         [c1*s3+c2*c3*s1, c1*c3-c2*s1*s3, s1*s2],
                         [-c3*s2, s2*s3, c2]])
    elif order=='zxz':
        matrix=np.array([[c1*c3-c2*s1*s3, -c1*s3-c2*c3*s1, s1*s2],
                         [c3*s1+c1*c2*s3, c1*c2*c3-s1*s3, -c1*s2],
                         [s2*s3, c3*s2, c2]])
    elif order=='xyz':
        matrix=np.array([[c2*c3, -c2*s3, s2],
                         [c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1],
                         [s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2]])
    elif order=='xzy':
        matrix=np.array([[c2*c3, -s2, c2*s3],
                         [s1*s3+c1*c3*s2, c1*c2, c1*s2*s3-c3*s1],
                         [c3*s1*s2-c1*s3, c2*s1, c1*c3+s1*s2*s3]])
    elif order=='yxz':
        matrix=np.array([[c1*c3+s1*s2*s3, c3*s1*s2-c1*s3, c2*s1],
                         [c2*s3, c2*c3, -s2],
                         [c1*s2*s3-c3*s1, c1*c3*s2+s1*s3, c1*c2]])
    elif order=='yzx':
        matrix=np.array([[c1*c2, s1*s3-c1*c3*s2, c3*s1+c1*s2*s3],
                         [s2, c2*c3, -c2*s3],
                         [-c2*s1, c1*s3+c3*s1*s2, c1*c3-s1*s2*s3]])
    elif order=='zyx':
        matrix=np.array([[c1*c2, c1*s2*s3-c3*s1, s1*s3+c1*c3*s2],
                         [c2*s1, c1*c3+s1*s2*s3, c3*s1*s2-c1*s3],
                         [-s2, c2*s3, c2*c3]])
    elif order=='zxy':
        matrix=np.array([[c1*c3-s1*s2*s3, -c2*s1, c1*s3+c3*s1*s2],
                         [c3*s1+c1*s2*s3, c1*c2, s1*s3-c1*c3*s2],
                         [-c2*s3, s2, c2*c3]])

    return matrix

def OrientationMatrix2Euler(g):
    """
    Compute the Euler angles from the orientation matrix.
    This conversion follows the paper of Rowenhorst et al. :cite:`Rowenhorst2015`.
    In particular when :math:`g_{33} = 1` within the machine precision,
    there is no way to determine the values of :math:`\phi_1` and :math:`\phi_2`
    (only their sum is defined). The convention is to attribute
    the entire angle to :math:`\phi_1` and set :math:`\phi_2` to zero.
    :param g: The 3x3 orientation matrix
    :return: The 3 euler angles in degrees.
    """
    eps = np.finfo('float').eps
    (phi1, Phi, phi2) = (0.0, 0.0, 0.0)
    # treat special case where g[2, 2] = 1
    if np.abs(g[2, 2]) >= 1 - eps:
        if g[2, 2] > 0.0:
            phi1 = np.arctan2(g[0][1], g[0][0])
        else:
            phi1 = -np.arctan2(-g[0][1], g[0][0])
            Phi = np.pi
    else:
        Phi = np.arccos(g[2][2])
        zeta = 1.0 / np.sqrt(1.0 - g[2][2] ** 2)
        phi1 = np.arctan2(g[2][0] * zeta, -g[2][1] * zeta)
        phi2 = np.arctan2(g[0][2] * zeta, g[1][2] * zeta)
    # ensure angles are in the range [0, 2*pi]
    if phi1 < 0.0:
        phi1 += 2 * np.pi
    if Phi < 0.0:
        Phi += 2 * np.pi
    if phi2 < 0.0:
        phi2 += 2 * np.pi
    return np.degrees([phi2, Phi, phi1])

def rotation_angles(matrix, order):
    """
    input
        matrix = 3x3 rotation matrix (numpy array)
        oreder(str) = rotation order of x, y, z : e.g, rotation XZY -- 'xzy'
    output
        theta1, theta2, theta3 = rotation angles in rotation order
    """
    r11, r12, r13 = matrix[0]
    r21, r22, r23 = matrix[1]
    r31, r32, r33 = matrix[2]

    if order == 'xzx':
        theta1 = np.arctan(r31 / r21)
        theta2 = np.arctan(r21 / (r11 * np.cos(theta1)))
        theta3 = np.arctan(-r13 / r12)

    elif order == 'xyx':
        theta1 = np.arctan(-r21 / r31)
        theta2 = np.arctan(-r31 / (r11 *np.cos(theta1)))
        theta3 = np.arctan(r12 / r13)

    elif order == 'yxy':
        theta1 = np.arctan(r12 / r32)
        theta2 = np.arctan(r32 / (r22 *np.cos(theta1)))
        theta3 = np.arctan(-r21 / r23)

    elif order == 'yzy':
        theta1 = np.arctan(-r32 / r12)
        theta2 = np.arctan(-r12 / (r22 *np.cos(theta1)))
        theta3 = np.arctan(r23 / r21)

    elif order == 'zyz':
        theta1 = np.arctan(r23 / r13)
        theta2 = np.arctan(r13 / (r33 *np.cos(theta1)))
        theta3 = np.arctan(-r32 / r31)

    elif order == 'zxz':
        theta1 = np.arctan(-r13 / r23)
        theta2 = np.arctan(-r23 / (r33 *np.cos(theta1)))
        theta3 = np.arctan(r31 / r32)

    elif order == 'xzy':
        theta1 = np.arctan(r32 / r22)
        theta2 = np.arctan(-r12 * np.cos(theta1) / r22)
        theta3 = np.arctan(r13 / r11)

    elif order == 'xyz':
        theta1 = np.arctan(-r23 / r33)
        theta2 = np.arctan(r13 * np.cos(theta1) / r33)
        theta3 = np.arctan(-r12 / r11)

    elif order == 'yxz':
        theta1 = np.arctan(r13 / r33)
        theta2 = np.arctan(-r23 * np.cos(theta1) / r33)
        theta3 = np.arctan(r21 / r22)

    elif order == 'yzx':
        theta1 = np.arctan(-r31 / r11)
        theta2 = np.arctan(r21 * np.cos(theta1) / r11)
        theta3 = np.arctan(-r23 / r22)

    elif order == 'zyx':
        theta1 = np.arctan(r21 / r11)
        theta2 = np.arctan(-r31 * np.cos(theta1) / r11)
        theta3 = np.arctan(r32 / r33)

    elif order == 'zxy':
        theta1 = np.arctan(-r12 / r22)
        theta2 = np.arctan(r32 * np.cos(theta1) / r22)
        theta3 = np.arctan(-r31 / r33)

    theta1 = theta1 * 180 / np.pi
    theta2 = theta2 * 180 / np.pi
    theta3 = theta3 * 180 / np.pi

    return (theta1, theta2, theta3)

## OR using scipy functions
def rot_mat_to_euler(rot_mat): 
    r = R.from_matrix(rot_mat)
    return r.as_euler('zxz')* 180/np.pi

## Cubic symmetric matrix
sym = np.zeros((24, 3, 3), dtype=float)
sym[0] = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
sym[1] = np.array([[0., 0., -1.], [0., -1., 0.], [-1., 0., 0.]])
sym[2] = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
sym[3] = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
sym[4] = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
sym[5] = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
sym[6] = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
sym[7] = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
sym[8] = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
sym[9] = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
sym[10] = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
sym[11] = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
sym[12] = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
sym[13] = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
sym[14] = np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])
sym[15] = np.array([[0., 1., 0.], [0., 0., -1.], [-1., 0., 0.]])
sym[16] = np.array([[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]])
sym[17] = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
sym[18] = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
sym[19] = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
sym[20] = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
sym[21] = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])
sym[22] = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
sym[23] = np.array([[-1., 0., 0.], [0., 0., -1.], [0., -1., 0.]])

euler_ang = 96.61, 90.29, 185.24 ## order of angle is important Phi2, PHI, Phi1 ?
euler_ang1 = 6.58, 95.24, 269.71

# euler_ang = 90.29, 96.61, 185.24 
# euler_ang = 185.24 , 90.29, 96.61
# euler_ang = 185.24 , 96.61, 90.29

rot_mat = rotation_matrix(euler_ang, order='zxz')
rot_mat1 = rotation_matrix(euler_ang1, order='zxz')


print(disorientation(rot_mat, rot_mat1, sym))



#%%
euler_back = OrientationMatrix2Euler(rot_mat)
euler_back1 = rotation_angles(rot_mat, order='zxz')

print(euler_ang)
# print(euler_back)
print(euler_back1)




for syms in sym:
    Osym = np.dot(syms, rot_mat)
    print(OrientationMatrix2Euler(rot_mat))





