# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:08:04 2022

@author: PURUSHOT

UBmatrix (LaueTools reference frame) to Euler angle conversion
"""

import numpy as np
# =============================================================================
# Function to go from UB to Euler angles
# =============================================================================
from scipy.spatial.transform import Rotation as R
def rot_mat_to_euler(rot_mat): 
    r = R.from_matrix(rot_mat)
    return r.as_euler('zxz')* 180/np.pi

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

# =============================================================================
# =============================================================================

orientation_matrix_uncorrected = np.eye(3) #Identity matrix   


orientation_matrix_uncorrected = np.array([[0.7207019,0.2365380,-0.6516429],[-0.2644371,0.9627174,0.0569927],[0.6408289,0.1312438,0.7563817]])
## rotate orientation by 40degrees to bring in Sample RF from LaueTools RF
omega = np.deg2rad(-40)
cw = np.cos(omega)
sw = np.sin(omega)
##For this we have to do a rotation along the Y axis
mat_from_lab_to_sample_frame = np.array([[cw, 0.0, sw], [0.0, 1.0, 0.0], [-sw, 0, cw]]) #Y

##Compute the new transformed UB matrix in Sample frame
orientation_matrix_transformed = np.dot(mat_from_lab_to_sample_frame.T, orientation_matrix_uncorrected)

##Verify determinant
if np.linalg.det(orientation_matrix_transformed) < 0:
    orientation_matrix_transformed = -orientation_matrix_transformed
    

euler_angles_pymicro_uncorrected = OrientationMatrix2Euler(orientation_matrix_uncorrected)
euler_angles_scipy_uncorrected = rot_mat_to_euler(orientation_matrix_uncorrected)


euler_angles_pymicro_transformed = OrientationMatrix2Euler(orientation_matrix_transformed)
euler_angles_scipy_transformed = rot_mat_to_euler(orientation_matrix_transformed)


# =============================================================================
# Convert UB matrix to RGB Color (careful only for Cubic systems)
# =============================================================================
sym = np.zeros((48, 3, 3), dtype=np.float)
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

def get_field_color(orientation_matrix, axis=np.array([0., 0., 1.]), syms=sym):
    """Compute the IPF (inverse pole figure) colour for this orientation.
    Given a particular axis expressed in the laboratory coordinate system,
    one can compute the so called IPF colour based on that direction
    expressed in the crystal coordinate system as :math:`[x_c,y_c,z_c]`.
    There is only one tuple (u,v,w) such that:
    .. math::
      [x_c,y_c,z_c]=u.[0,0,1]+v.[0,1,1]+w.[1,1,1]
    and it is used to assign the RGB colour.
    :param ndarray axis: the direction to use to compute the IPF colour.
    :param syms : the symmetry operator to use.
    :return tuple: a tuple contining the RGB values.
    """
    for sym in syms:
        Osym = np.dot(sym, orientation_matrix)
        Vc = np.dot(Osym, axis)
        if Vc[2] < 0:
            Vc *= -1.  # using the upward direction
        uvw = np.array([Vc[2] - Vc[1], Vc[1] - Vc[0], Vc[0]])
        uvw /= np.linalg.norm(uvw)
        uvw /= max(uvw)
        if (uvw[0] >= 0. and uvw[0] <= 1.0) and (uvw[1] >= 0. and uvw[1] <= 1.0) and (
                uvw[2] >= 0. and uvw[2] <= 1.0):
            break
    uvw = uvw / uvw.max()
    return uvw

color = get_field_color(orientation_matrix_transformed)

