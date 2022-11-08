# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 19:04:03 2022

@author: PURUSHOT

Script to test misorientation functions

Functions modified from https://github.com/jacione/lauepy
to work out regarding the applicability
and perhaps extend it to triclinic and monoclinic systems

FOr now, only CUBIC!!
"""
from scipy.spatial.transform import Rotation
import numpy as np

def euler_as_matrix(angles):
    """
    Return the active rotation matrix
    """
    # NOTE:
    #   It is not recommended to directly associated Euler angles with
    #   other common transformation concept due to its unique passive
    #   nature.
    phi2, phi, phi1 = angles
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c, s = np.cos(phi), np.sin(phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)
    return np.array([
                        [c1 * c2 - s1 * c * s2, -c1 * s2 - s1 * c * c2, s1 * s],
                        [s1 * c2 + c1 * c * s2, -s1 * s2 + c1 * c * c2, -c1 * s],
                        [s * s2, s * c2, c],
                    ])

def quats_as_eulers(qs):
    """
    Quaternion to Euler angles
    """
    # NOTE: assuming Bunge Euler angle (z->x->z)
    #   w = cos(Phi/2) * cos(phi1/2 + phi2/2)
    #   x = sin(Phi/2) * cos(phi1/2 - phi2/2)
    #   y = sin(Phi/2) * sin(phi1/2 - phi2/2)
    #   z = cos(Phi/2) * sin(phi1/2 + phi2/2)
    w,x,y,z = qs
    phi2 = np.arctan2(z, w) - np.arctan2(y, x)
    PHI = 2 * np.arcsin(np.sqrt(x ** 2 + y ** 2))
    phi1 = np.arctan2(z, w) + np.arctan2(y, x)
    return phi2, PHI, phi1
    
def normalize(qs):
    # standardize the quaternion
    # 1. rotation angle range: [0, pi] -> self.w >= 0
    # 2. |q| === 1
    w,x,y,z = qs
    _sgn = -1 if w < 0 else 1
    _norm = np.linalg.norm([w, x, y, z]) * _sgn
    w /= _norm
    x /= _norm
    y /= _norm
    z /= _norm
    return (w,x,y,z)

def average_quaternions(qs):
    """
    Description
    -----------
    Return the average quaternion based on algorithm published in
        F. Landis Markley et.al.
        Averaging Quaternions,
        doi: 10.2514/1.28949
    Parameters
    ----------
    qs: quaternions for average
    
    Returns
    -------
    Quaternion
        average quaternion of the given list
    Note:
    This method only provides an approximation, with about 1% error. 
    """
    _sum = np.sum([np.outer(q, q) for q in qs], axis=0)
    _eigval, _eigvec = np.linalg.eig(_sum / len(qs))
    return np.real(_eigvec.T[_eigval.argmax()])

# convert euler angles to quaternions
def rmat_2_quat(rmat):
    r = Rotation.from_matrix(rmat)
    # Invert to match the massif convention, where the Bunge Euler/Rotation is a
    # transformation from sample to crystal frame.
    r1 = r.inv()
    quat = r1.as_quat()
    if quat[3] < 0:
        quat = -1.0 * quat
    quat = np.roll(quat, 1)
    return quat

# =============================================================================
# One scalar and one array
# =============================================================================
# compute quaternion product
def QuadProd(p, q):
    p0 = p[0] #np.reshape(p[:, 0], (p[:, 0].shape[0], 1))
    q0 = np.reshape(q[:, 0], (q[:, 0].shape[0], 1))
    l = np.sum(p[1:]*q[:, 1:], 1) #np.sum(p[:, 1:]*q[:, 1:], 1)
    prod1 = (p0*q0).flatten() - l
    prod2 = p0*q[:, 1:] + q0*p[1:] + np.cross(p[1:], q[:, 1:])
    m = np.transpose(np.stack([prod1, prod2[:,0], prod2[:,1], prod2[:,2]]))
    return m

# invert quaternion
def invQuat(p):
    q = np.transpose(np.stack([-p[0], p[1], p[2], p[3]]))
    return q

# calculate the disorientation between two sets of quaternions ps and qs
def calc_disorient(y_true, y_pred):
    # sort quaternion for cubic symmetry trick
    p = np.sort(np.abs(QuadProd(invQuat(y_true), y_pred)))
    # calculate last component of two other options
    p1 = (p[:,2] + p[:,3]) / 2 ** (1 / 2)
    p2 = (p[:,0] + p[:,1] + p[:,2] + p[:,3]) / 2
    vals = np.transpose(np.stack([p[:,-1], p1, p2]))
    # pick largest value and find angle
    max_val = np.amax(vals, axis=1)
    mis = (2 * np.arccos(max_val))
    return np.degrees(replacenan(mis))

def replacenan(t):
    t[np.isnan(t)] = 0
    return t
#%%
import matplotlib.pyplot as plt
import os
import _pickle as cPickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import trange

folder = os.getcwd()
with open(r"C:\Users\purushot\Desktop\SiC\results_triangle_5UBs\results.pickle", "rb") as input_file:
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

print("Number of Phases present", len(np.unique(np.array(mat_global)))-1)

    
#%%
for matid in range(matnumber):
    for index in range(len(rotation_matrix1)):
        quats = []
        rotation_matrix_transformed = []
        for om_ind in range(len(rotation_matrix1[index][0])):
            ##convert the UB matrix to sample reference frame
            orientation_matrix = rotation_matrix1[index][0][om_ind]
            ##Transformation not needed, so we dont have to reapply this to index in Lauetools frame
            # ## rotate orientation by 40degrees to bring in Sample RF
            # omega = np.deg2rad(-40)
            # cw = np.cos(omega)
            # sw = np.sin(omega)
            # mat_from_lab_to_sample_frame = np.array([[cw, 0.0, sw], [0.0, 1.0, 0.0], [-sw, 0, cw]]) #Y
            # orientation_matrix = np.dot(mat_from_lab_to_sample_frame.T, orientation_matrix)
    
            # if np.linalg.det(orientation_matrix) < 0:
            #     orientation_matrix = -orientation_matrix
            #make quaternion from UB matrix
            quats.append(rmat_2_quat(orientation_matrix))
            rotation_matrix_transformed.append(orientation_matrix)
        quats = np.array(quats)
        
        misori = []
        for ii in trange(len(quats)):
            misori.append(calc_disorient(quats[ii,:], quats))
        
        misori = np.array(misori)
        
        mr = match_rate[index][0]
        mr = mr.flatten()
        if np.max(mr) == 0:
            continue
        
        ref_index = np.where(mr == np.max(mr))[0][0] ##choose from best matching rate
        
        ##Lets bin the angles to know how many significant grains are present
        max_ang = int(np.max(misori[ref_index,:])) + 1
        zz, binzz = np.histogram(misori[ref_index,:], bins=max_ang) #1°bin
        
        bin_index = np.where(zz> 0.05*np.max(zz))[0]
        bin_angles = binzz[bin_index]
        
        
        rotation_matrix_transformed = np.array(rotation_matrix_transformed)
        
        average_UB = []
        grains = np.copy(misori[ref_index,:])
        for kk, jj in enumerate(bin_angles):
            if jj ==0:
                cond = (grains<jj+1)
            else:
                cond = (grains<jj+1) * (grains>jj-1)
                
            grain_om = rotation_matrix_transformed[cond, :, :]
            grain_quats = quats[cond, :]
            avg_quat_grain = average_quaternions(grain_quats)
            avg_quat_grain = normalize(avg_quat_grain)
            avg_euler_grain = quats_as_eulers(avg_quat_grain)
            avg_om_grain = euler_as_matrix(avg_euler_grain).T
            average_UB.append(avg_om_grain)
            ## mask the grain pixels
            grains[cond] = 360 + kk
        
        grains[grains<350] = np.nan
        grains = grains.reshape((lim_x,lim_y))
        grains = grains - 360
        
        ####Average UBs misorientation with each other
        average_UB = np.array(average_UB)
        quats_UB = []
        for om_ind in range(len(average_UB)):
            quats_UB.append(rmat_2_quat(average_UB[om_ind, :, :]))
        quats_UB = np.array(quats_UB)
        misori_avg = []
        for ii in range(len(quats_UB)):
            misori_avg.append(calc_disorient(quats_UB[ii,:], quats_UB))
        misori_avg = np.array(misori_avg)
        
        fig = plt.figure(figsize=(11.69,8.27), dpi=100)
        bottom, top = 0.1, 0.9
        left, right = 0.1, 0.8
        fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
        axs = fig.subplots(1, 1)
        axs.set_title(r"Grain map", loc='center', fontsize=8)
        im=axs.imshow(grains, origin='lower', cmap=plt.cm.jet)
        axs.set_xticks([])
        axs.set_yticks([])
        divider = make_axes_locatable(axs)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.ax.tick_params(labelsize=8) 
        axs.label_outer()
        plt.show()
        # plt.savefig(folder+ "//"+'figure_misorientation_'+str(matid)+"_"+str(index)+'.png', 
        #             bbox_inches='tight',format='png', dpi=1000) 
        # plt.close(fig)
        






