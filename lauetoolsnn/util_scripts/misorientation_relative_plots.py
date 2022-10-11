# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 19:04:03 2022

@author: PURUSHOT
"""
import numpy as np


# =============================================================================
# PYMICRO FUNCTION IMPORTS
# =============================================================================

def move_rotation_to_FZ(g, symmetry_operators = None):
    """Compute the rotation matrix in the Fundamental Zone of a given
    `Symmetry` instance.

    :param g: a 3x3 matrix representing the rotation.
    :param verbose: flag for verbose mode.
    :return: a new 3x3 matrix for the rotation in the fundamental zone.
    """
    omegas = []  # list to store all the rotation angles
    syms = symmetry_operators
    for sym in syms:
        # apply the symmetry operator
        om = np.dot(sym, g)
        cw = 0.5 * (om.trace() - 1)
        omega = np.arccos(cw)
        omegas.append(omega)
    index = np.argmin(omegas)
    return np.dot(syms[index], g)

def misorientation_axis_from_delta(delta):
    """Compute the misorientation axis from the misorientation matrix.

    :param delta: The 3x3 misorientation matrix.
    :returns: the misorientation axis (normalised vector).
    """
    n = np.array([delta[1, 2] - delta[2, 1], delta[2, 0] -
                  delta[0, 2], delta[0, 1] - delta[1, 0]])
    n /= np.sqrt((delta[1, 2] - delta[2, 1]) ** 2 +
                 (delta[2, 0] - delta[0, 2]) ** 2 +
                 (delta[0, 1] - delta[1, 0]) ** 2)
    return n

def misorientation_angle_from_delta(delta):
    """Compute the misorientation angle from the misorientation matrix.

    Compute the angle associated with this misorientation matrix :math:`\\Delta g`.
    It is defined as :math:`\\omega = \\arccos(\\text{trace}(\\Delta g)/2-1)`.
    To avoid float rounding error, the argument is rounded to 1.0 if it is
    within 1 and 1 plus 32 bits floating point precison.

    .. note::

      This does not account for the crystal symmetries. If you want to
      find the disorientation between two orientations, use the
      :py:meth:`~pymicro.crystal.microstructure.Orientation.disorientation`
      method.

    :param delta: The 3x3 misorientation matrix.
    :returns float: the misorientation angle in radians.
    """
    cw = 0.5 * (delta.trace() - 1)
    if cw > 1. and cw - 1. < 10 * np.finfo('float32').eps:
        cw = 1.
    omega = np.arccos(cw)
    return omega

def disorientation(orientation_matrix, orientation_matrix1, symmetry_operators=None):
    """Compute the disorientation another crystal orientation.

    Considering all the possible crystal symmetries, the disorientation
    is defined as the combination of the minimum misorientation angle
    and the misorientation axis lying in the fundamental zone, which
    can be used to bring the two lattices into coincidence.

    .. note::

     Both orientations are supposed to have the same symmetry. This is not
     necessarily the case in multi-phase materials.

    :param orientation: an instance of
        :py:class:`~pymicro.crystal.microstructure.Orientation` class
        describing the other crystal orientation from which to compute the
        angle.
    :param crystal_structure: an instance of the `Symmetry` class
        describing the crystal symmetry, triclinic (no symmetry) by
        default.
    :returns tuple: the misorientation angle in radians, the axis as a
        numpy vector (crystal coordinates), the axis as a numpy vector
        (sample coordinates).
    """
    # gA = move_rotation_to_FZ(orientation_matrix, symmetry_operators=symmetry_operators)
    # gB = move_rotation_to_FZ(orientation_matrix1, symmetry_operators=symmetry_operators)
    the_angle = np.pi
    symmetries = symmetry_operators
    (gA, gB) = (orientation_matrix, orientation_matrix1)  # nicknames
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
    return np.rad2deg(the_angle), the_axis, the_axis_xyz

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

def OrientationMatrix2Rodrigues(g):
    """
    Compute the rodrigues vector from the orientation matrix.

    :param g: The 3x3 orientation matrix representing the rotation.
    :returns: The Rodrigues vector as a 3 components array.
    """
    t = g.trace() + 1
    if np.abs(t) < np.finfo(g.dtype).eps:
        print('warning, returning [0., 0., 0.], consider using axis, angle '
              'representation instead')
        return np.zeros(3)
    else:
        r1 = (g[1, 2] - g[2, 1]) / t
        r2 = (g[2, 0] - g[0, 2]) / t
        r3 = (g[0, 1] - g[1, 0]) / t
    return np.array([r1, r2, r3])

def Rodrigues2OrientationMatrix(rod):
    """
    Compute the orientation matrix from the Rodrigues vector.

    :param rod: The Rodrigues vector as a 3 components array.
    :returns: The 3x3 orientation matrix representing the rotation.
    """
    r = np.linalg.norm(rod)
    I = np.diagflat(np.ones(3))
    if r < np.finfo(r.dtype).eps:
        # the rodrigues vector is zero, return the identity matrix
        return I
    theta = 2 * np.arctan(r)
    n = rod / r
    omega = np.array([[0.0, n[2], -n[1]],
                      [-n[2], 0.0, n[0]],
                      [n[1], -n[0], 0.0]])
    g = I + np.sin(theta) * omega + (1 - np.cos(theta)) * omega.dot(omega)
    return g

def compute_mean_orientation(orientation_matrix, symmetry_operators=None):
    """Compute the mean orientation.

    This function computes a mean orientation from several data points
    representing orientations. Each orientation is first moved to the
    fundamental zone, then the corresponding Rodrigues vectors can be
    averaged to compute the mean orientation.

    :param ndarray rods: a (n, 3) shaped array containing the Rodrigues
    vectors of the orientations.
    :param `Symmetry` symmetry: the symmetry used to move orientations
    to their fundamental zone (cubic by default)
    :returns: the mean orientation as an `Orientation` instance.
    """
    rods_fz = np.zeros((len(orientation_matrix),3))
    for i in range(len(orientation_matrix)):
        # g_fz = move_rotation_to_FZ(orientation_matrix[i], symmetry_operators=symmetry_operators)
        rods_fz[i] = OrientationMatrix2Rodrigues(orientation_matrix[i])
    mean_orientation = Rodrigues2OrientationMatrix(np.mean(rods_fz, axis=0))
    return mean_orientation
#%%
import matplotlib.pyplot as plt
import os
import _pickle as cPickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import trange

folder = os.getcwd()
with open("results.pickle", "rb") as input_file:
    best_match, \
    mat_global, rotation_matrix1, strain_matrix, strain_matrixs,\
        col, colx, coly, match_rate, files_treated,\
            lim_x, lim_y, spots_len, iR_pix, fR_pix, material_, \
                material1_, lattice, lattice1, symmetry0, symmetry1,\
                    crystal, crystal1 = cPickle.load(input_file)
match_tol = 0
fR_tol = 10000
matnumber = 1
rangemin = -0.1
rangemax = 0.1
bins = 100
rangeval = len(match_rate)
material_id = [material_, material1_]

for matid in range(matnumber):
    for index in range(len(rotation_matrix1)):
        mean_orientation = compute_mean_orientation(rotation_matrix1[index][0], np.array(crystal._hklsym))
        
        misorientation_angles = np.zeros(len(rotation_matrix1[index][0]))
        misorientation_crystalaxis = np.zeros((len(rotation_matrix1[index][0]),3))
        misorientation_sampleaxis = np.zeros((len(rotation_matrix1[index][0]),3))
        for i in trange(len(rotation_matrix1[index][0])):
            result = disorientation(mean_orientation, rotation_matrix1[index][0][i,:,:], np.array(crystal._hklsym))
            misorientation_angles[i] = result[0]
            misorientation_crystalaxis[i,:] = result[1]
            misorientation_sampleaxis[i,:] = result[2]
        
        # nan based on misorientations
        nan_index12 = np.where(misorientation_angles > 15)[0]
        nan_index13 = np.where(misorientation_angles < -15)[0]
        
        nan_index11 = np.where(match_rate[index][0] < match_tol)[0]
        nan_index1 = np.where(fR_pix[index][0] > fR_tol)[0]
        mat_id_index = np.where(mat_global[index][0] != matid+1)[0]
        nan_index = np.hstack((mat_id_index,nan_index1,nan_index11,nan_index12,nan_index13))
        nan_index = np.unique(nan_index)
    
        misorientation_angles_plot = np.copy(misorientation_angles)
        misorientation_angles_plot[nan_index] = np.nan             
                
        fig = plt.figure(figsize=(11.69,8.27), dpi=100)
        bottom, top = 0.1, 0.9
        left, right = 0.1, 0.8
        fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
        
        if np.isnan(np.max(misorientation_angles_plot)):
            vmax = 3
        else:
            vmax = np.max(misorientation_angles_plot) + 0.1
            
        if np.isnan(np.min(misorientation_angles_plot)):
            vmin = -3
        else:
            vmin = np.min(misorientation_angles_plot) - 0.1
        
        max_vm = np.max((np.abs(vmax),np.abs(vmin)))
        vmin = -max_vm
        vmax = max_vm
        axs = fig.subplots(1, 1)
        axs.set_title(r"Misorientation map (in $\degree$)", loc='center', fontsize=8)
        im=axs.imshow(misorientation_angles_plot.reshape((lim_x,lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
        axs.set_xticks([])
        axs.set_yticks([])
        divider = make_axes_locatable(axs)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.ax.tick_params(labelsize=8) 
        
        axs.label_outer()
        plt.savefig(folder+ "//"+'figure_misorientation_from_mean_'+str(matid)+"_"+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
        plt.close(fig)
        
#TODO misorientation from one pixel to another (for last pixel do it with row below)






