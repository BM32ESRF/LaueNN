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


###other symmetry misorientations
# Step_1: get the symmetry operators
# sym_ops = sym_operator(lattice)
# Step_2: make sure both are in the same frame
# Step_3: calculate misorientations among all possible pairs
# _drs = [other.q.conjugate * self.q * op for op in sym_ops]
# # Step_4: Locate the one pair with the smallest rotation angle
# _dr = _drs[np.argmin([me.rot_angle for me in _drs])]
# return (_dr.rot_angle, _dr.rot_axis)

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

def disorientation_array(orientation_matrix, orientation_matrix1, symmetry_operators=None):
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
    the_angle = np.pi
    symmetries = symmetry_operators
    (g1, g2) = (orientation_matrix, orientation_matrix1)  # nicknames

    ##vectorize
    oj = np.dot(symmetries, g1) #shape of len(symm),3,3
    ojT = np.transpose(oj, axes=(0,2,1))
    oi = np.dot(symmetries, g2)
    # delta = np.dot(oi, ojT)
    delta = np.einsum('ijkl,nlm->kinjm', oi, ojT)
    #### misorientation_angle_from_delta
    # tr = np.trace(delta, axis1=1, axis2=3)
    tr = np.trace(delta, axis1=3, axis2=4)
    cw = 0.5 * (tr - 1)
    cond = (cw>1.) * (cw - 1. < 10 * np.finfo('float32').eps)
    cw[cond] = 1.0
    mis_angle = np.arccos(cw)
    mis_angle[np.isnan(mis_angle)] = np.pi
    
    the_angle = mis_angle.min(axis=(1,2))
    the_angle = np.rad2deg(the_angle)
    return the_angle

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

def symmetryQuats(lattice):
    """List of symmetry operations as quaternions."""
    import math
    if lattice == 'cubic':
        symQuats =  [
                    [ 1.0,              0.0,              0.0,              0.0              ],
                    [ 0.0,              1.0,              0.0,              0.0              ],
                    [ 0.0,              0.0,              1.0,              0.0              ],
                    [ 0.0,              0.0,              0.0,              1.0              ],
                    [ 0.0,              0.0,              0.5*math.sqrt(2), 0.5*math.sqrt(2) ],
                    [ 0.0,              0.0,              0.5*math.sqrt(2),-0.5*math.sqrt(2) ],
                    [ 0.0,              0.5*math.sqrt(2), 0.0,              0.5*math.sqrt(2) ],
                    [ 0.0,              0.5*math.sqrt(2), 0.0,             -0.5*math.sqrt(2) ],
                    [ 0.0,              0.5*math.sqrt(2),-0.5*math.sqrt(2), 0.0              ],
                    [ 0.0,             -0.5*math.sqrt(2),-0.5*math.sqrt(2), 0.0              ],
                    [ 0.5,              0.5,              0.5,              0.5              ],
                    [-0.5,              0.5,              0.5,              0.5              ],
                    [-0.5,              0.5,              0.5,             -0.5              ],
                    [-0.5,              0.5,             -0.5,              0.5              ],
                    [-0.5,             -0.5,              0.5,              0.5              ],
                    [-0.5,             -0.5,              0.5,             -0.5              ],
                    [-0.5,             -0.5,             -0.5,              0.5              ],
                    [-0.5,              0.5,             -0.5,             -0.5              ],
                    [-0.5*math.sqrt(2), 0.0,              0.0,              0.5*math.sqrt(2) ],
                    [ 0.5*math.sqrt(2), 0.0,              0.0,              0.5*math.sqrt(2) ],
                    [-0.5*math.sqrt(2), 0.0,              0.5*math.sqrt(2), 0.0              ],
                    [-0.5*math.sqrt(2), 0.0,             -0.5*math.sqrt(2), 0.0              ],
                    [-0.5*math.sqrt(2), 0.5*math.sqrt(2), 0.0,              0.0              ],
                    [-0.5*math.sqrt(2),-0.5*math.sqrt(2), 0.0,              0.0              ],
                  ]
    elif lattice == 'hexagonal':
        symQuats =  [
                    [ 1.0,0.0,0.0,0.0 ],
                    [-0.5*math.sqrt(3), 0.0, 0.0,-0.5 ],
                    [ 0.5, 0.0, 0.0, 0.5*math.sqrt(3) ],
                    [ 0.0,0.0,0.0,1.0 ],
                    [-0.5, 0.0, 0.0, 0.5*math.sqrt(3) ],
                    [-0.5*math.sqrt(3), 0.0, 0.0, 0.5 ],
                    [ 0.0,1.0,0.0,0.0 ],
                    [ 0.0,-0.5*math.sqrt(3), 0.5, 0.0 ],
                    [ 0.0, 0.5,-0.5*math.sqrt(3), 0.0 ],
                    [ 0.0,0.0,1.0,0.0 ],
                    [ 0.0,-0.5,-0.5*math.sqrt(3), 0.0 ],
                    [ 0.0, 0.5*math.sqrt(3), 0.5, 0.0 ],
                  ]
    elif lattice == 'tetragonal':
        symQuats =  [
                        [ 1.0,0.0,0.0,0.0 ],
                        [ 0.0,1.0,0.0,0.0 ],
                        [ 0.0,0.0,1.0,0.0 ],
                        [ 0.0,0.0,0.0,1.0 ],
                        [ 0.0, 0.5*math.sqrt(2), 0.5*math.sqrt(2), 0.0 ],
                        [ 0.0,-0.5*math.sqrt(2), 0.5*math.sqrt(2), 0.0 ],
                        [ 0.5*math.sqrt(2), 0.0, 0.0, 0.5*math.sqrt(2) ],
                        [-0.5*math.sqrt(2), 0.0, 0.0, 0.5*math.sqrt(2) ],
                      ]
    elif lattice == 'orthorhombic':
        symQuats =  [
                        [ 1.0,0.0,0.0,0.0 ],
                        [ 0.0,1.0,0.0,0.0 ],
                        [ 0.0,0.0,1.0,0.0 ],
                        [ 0.0,0.0,0.0,1.0 ],
                      ]
    else:
        symQuats =  [
                        [ 1.0,0.0,0.0,0.0 ],
                      ]
      
    return symQuats



#%%
import matplotlib.pyplot as plt
import os
import _pickle as cPickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import trange

folder = os.getcwd()
with open(r"D:\some_projects\GaN\Si_GaN_nanowires\results_Si_2022-07-08_18-22-15\results.pickle", "rb") as input_file:
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

print("Number of Phases present (includes non indexed phase zero also)", len(np.unique(np.array(mat_global))))

    
#%%
average_UB = []
nb_pixels = []
mat_index = []

for index in range(len(rotation_matrix1)):
    for val in np.unique(mat_global[index][0]):
        if val == 0:
            continue ##skipping no indexed patterns
        
        if val == 1:
            crystal_symm = np.array(crystal._hklsym)
        elif val == 2:
            crystal_symm = np.array(crystal1._hklsym)
            
        mat_index1 = mat_global[index][0]
        mask_ = np.where(mat_index1 != val)[0]
        
        quats = []
        for om_ind in range(len(rotation_matrix1[index][0])):
            ##convert the UB matrix to sample reference frame
            orientation_matrix = rotation_matrix1[index][0][om_ind]
            #make quaternion from UB matrix
            quats.append(rmat_2_quat(orientation_matrix))
        quats = np.array(quats)
        
        mr = match_rate[index][0]
        mr = mr.flatten()
        
        mr[mask_] = 0
        if np.max(mr) == 0:
            continue
        
        ref_index = np.where(mr == np.max(mr))[0][0] ##choose from best matching rate
        ##Quats approach
        misori_quats = calc_disorient(quats[ref_index,:], quats)
        ##Pymicro approach
        misori_UB = disorientation_array(rotation_matrix1[index][0][ref_index,:,:],
                                                    rotation_matrix1[index][0],
                                                    np.unique(crystal_symm, axis=0))
        misori = misori_UB
        
        ##Lets bin the angles to know how many significant grains are present
        max_ang = int(np.max(replacenan(misori))) + 5
        
        misori[mask_] = -1
        bins = np.arange(0,max_ang, 1)
        zz, binzz = np.histogram(misori, bins=bins) #1°bin
        
        bin_index = np.where(zz> 0.1*np.max(zz))[0]
        bin_angles = binzz[bin_index]
                
        grains = np.copy(misori)
        grains_segmentation = np.zeros_like(misori)
        
        grain_tol = 2.5
        for kk, jj in enumerate(bin_angles):
            if jj == 0:
                cond = (grains<=jj+grain_tol)* (grains>=90-grain_tol)
            else:
                cond = (grains<=jj+grain_tol) * (grains>=jj-grain_tol)            
            
            ### Apply more filters to select good data only here
            if np.all(cond == False):
                continue
            
            grain_om = rotation_matrix1[index][0][cond,:,:]
            avg_om_grain = compute_mean_orientation(grain_om)
            
            grain_quats = quats[cond, :]
            avg_quat_grain = average_quaternions(grain_quats)
            avg_quat_grain = normalize(avg_quat_grain)
            avg_euler_grain = quats_as_eulers(avg_quat_grain)
            avg_om_grain_quat = euler_as_matrix(avg_euler_grain).T
            average_UB.append(avg_om_grain)
            nb_pixels.append(len(cond[cond]))
            mat_index.append(val)
            ## mask the grain pixels
            for ll in range(len(cond)):
                if cond[ll]:
                    if grains_segmentation[ll] == 0:
                        grains_segmentation[ll] = kk
        
        grains = grains_segmentation.reshape((lim_x,lim_y))
        
        fig = plt.figure(figsize=(11.69,8.27), dpi=100)
        bottom, top = 0.1, 0.9
        left, right = 0.1, 0.8
        fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
        axs = fig.subplots(1, 2)
        axs[0].set_title(r"Grain map", loc='center', fontsize=8)
        im=axs[0].imshow(grains, origin='lower', cmap=plt.cm.jet)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.ax.tick_params(labelsize=8) 
        axs[0].label_outer()
        
        axs[1].set_title(r"Misorientation map", loc='center', fontsize=8)
        im=axs[1].imshow(misori.reshape((lim_x,lim_y)), origin='lower', cmap=plt.cm.jet)
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.ax.tick_params(labelsize=8) 
        axs[1].label_outer()
        
        plt.show()
        # plt.savefig(folder+ "//"+'figure_misorientation_'+str(matid)+"_"+str(index)+'.png', 
        #             bbox_inches='tight',format='png', dpi=1000) 
        # plt.close(fig)
        
####Average UBs misorientation with each other
average_UB = np.array(average_UB)
nb_pixels = np.array(nb_pixels)
mat_index = np.array(mat_index)
quats_UB = []
for om_ind in range(len(average_UB)):
    quats_UB.append(rmat_2_quat(average_UB[om_ind, :, :]))
quats_UB = np.array(quats_UB)
misori_avg = []
for ii in range(len(quats_UB)):
    misori_avg.append(calc_disorient(quats_UB[ii,:], quats_UB))
misori_avg = np.array(misori_avg)


s_ix = np.argsort(nb_pixels)[::-1]
average_UB = average_UB[s_ix]
nb_pixels = nb_pixels[s_ix]
#############################
save_directory_ = r"D:\some_projects\GaN\Si_GaN_nanowires\results_Si_2022-07-08_18-22-15"
## save a average_rot_mat.txt file
text_file = open(os.path.join(save_directory_,"average_rot_mat.txt"), "w")
text_file.write("# ********** Average UB matrix from Misorientation computation *************\n")

for imat in range(len(average_UB)):
    local_ub = average_UB[imat,:,:].flatten()
    string_ = ",".join(map(str, local_ub))
    string_ = string_ + ","+str(mat_index[imat])
    text_file.write("# ********** UB MATRIX "+str(imat+1)+" ********** \n")
    text_file.write("# Nb of pixel occupied "+str(nb_pixels[imat])+"/"+str(lim_x*lim_y)+" ********** \n")
    text_file.write(string_ + " \n")
text_file.close()
    
    
    
    
    
    
    
    
    



