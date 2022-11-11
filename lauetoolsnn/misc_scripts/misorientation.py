# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 14:22:01 2022

@author: PURUSHOT

Compute misorientation between two orientations
"""
__author__ = "Ravi raj purohit PURUSHOTTAM RAJ PUROHIT, CRG-IF BM32 @ ESRF"

import numpy as np

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


## SPOT 0 and SPOT 11
#hkl
orientation = np.array([[-0.644008485053909, -0.754499173210708, -0.126428117137471],
                        [-0.75505310803185 ,  0.600293107909911,  0.263710046542227],
                        [-0.123074857217915,  0.265291079442643, -0.956280932931817]])
#h-kl
orientation1 = np.array([[ 0.204880676004153,  0.754499173210709,  0.623502130088132],
                         [-0.168585913503182, -0.600293107909911,  0.781810062844012],
                         [ 0.96415918059706 , -0.265291079442643,  0.00420923253857 ]])
#-hk-l	
orientation2 = np.array([[-0.062142576871985, -0.727132460085605,  0.683678788342571],
                         [ 0.251299526525604,  0.651679033810394,  0.715655632870976],
                         [-0.965971248643296,  0.216096115512392,  0.142133794908061]])
#-h-k-l	  
orientation3 = np.array([[ 0.659980357238595,  0.727132460085605, -0.188955850793439],
                         [ 0.750034954288652, -0.651679033810395,  0.112969041056257],
                         [-0.041101042636329, -0.216096115512393, -0.97550662383946 ]])


a, b, c, alpha, beta, gamma = 5.1505, 5.2116, 5.3173, 90, 99.23, 90

symmetry_operators = np.array([[[ 1., -0., -0.],
                                [-0.,  1., -0.],
                                [-0.,  0.,  1.]],
                         
                               [[-1.,  0.,  0.],
                                [-0.,  1., -0.],
                                [ 0.,  0., -1.]],
                         
                               [[-1.,  0.,  0.],
                                [ 0., -1.,  0.],
                                [ 0., -0., -1.]],
                         
                               [[ 1., -0., -0.],
                                [ 0., -1.,  0.],
                                [-0., -0.,  1.]]])

print("*************hkl and h-kl****************************")
mis_angle, cry_axis, sam_axis = disorientation(orientation, orientation1, symmetry_operators)
print("Misorientation angle in deg is ", np.round(mis_angle,4))
print("Crystal misorientation axis is ", np.round(cry_axis,4))
print("Sample misorientation axis is ", np.round(sam_axis,4))

print("*************hkl and -hk-l****************************")
mis_angle, cry_axis, sam_axis = disorientation(orientation, orientation2, symmetry_operators)
print("Misorientation angle in deg is ", np.round(mis_angle,4))
print("Crystal misorientation axis is ", np.round(cry_axis,4))
print("Sample misorientation axis is ", np.round(sam_axis,4))

print("*************hkl and -h-k-l****************************")
mis_angle, cry_axis, sam_axis = disorientation(orientation, orientation3, symmetry_operators)
print("Misorientation angle in deg is ", np.round(mis_angle,4))
print("Crystal misorientation axis is ", np.round(cry_axis,4))
print("Sample misorientation axis is ", np.round(sam_axis,4))

#%% Other reflection
import LaueTools.findorient as FindO
B = np.array([[ 1.967027323100055e-01, -3.085660161325959e-17,3.056100612146954e-02],
               [ 0.000000000000000e+00,  1.918796530815872e-01,-1.151568276331365e-17],
               [ 0.000000000000000e+00,  0.000000000000000e+00,1.880653715231414e-01]])
tth_chi_spot1 = np.array([55.92449951, 26.80540085]) #spot 78
tth_chi_spot2 = np.array([64.20510101, 31.64730072]) #spot 84

#hkl
hkl0 , hkl1 = np.array([0,4,3]), np.array([0,5,3])
orientation = FindO.OrientMatrix_from_2hkl(hkl0, tth_chi_spot1, \
                             hkl1, tth_chi_spot2, B)
#h-kl    
hkl0 , hkl1 = np.array([0,-4,3]), np.array([0,-5,3])
orientation1 = FindO.OrientMatrix_from_2hkl(hkl0, tth_chi_spot1, \
                             hkl1, tth_chi_spot2, B)   
#-hk-l	
hkl0 , hkl1 = np.array([0,4,-3]), np.array([0,5,-3])
orientation2 = FindO.OrientMatrix_from_2hkl(hkl0, tth_chi_spot1, \
                             hkl1, tth_chi_spot2, B)
#-h-k-l	
hkl0 , hkl1 = np.array([0,-4,-3]), np.array([0,-5,-3])
orientation3 = FindO.OrientMatrix_from_2hkl(hkl0, tth_chi_spot1, \
                             hkl1, tth_chi_spot2, B)

print("*************hkl and h-kl****************************")
mis_angle, cry_axis, sam_axis = disorientation(orientation, orientation1, symmetry_operators)
print("Misorientation angle in deg is ", np.round(mis_angle,4))
print("Crystal misorientation axis is ", np.round(cry_axis,4))
print("Sample misorientation axis is ", np.round(sam_axis,4))

print("*************hkl and -hk-l****************************")
mis_angle, cry_axis, sam_axis = disorientation(orientation, orientation2, symmetry_operators)
print("Misorientation angle in deg is ", np.round(mis_angle,4))
print("Crystal misorientation axis is ", np.round(cry_axis,4))
print("Sample misorientation axis is ", np.round(sam_axis,4))

print("*************hkl and -h-k-l****************************")
mis_angle, cry_axis, sam_axis = disorientation(orientation, orientation3, symmetry_operators)
print("Misorientation angle in deg is ", np.round(mis_angle,4))
print("Crystal misorientation axis is ", np.round(cry_axis,4))
print("Sample misorientation axis is ", np.round(sam_axis,4))

#%%
import numpy as np
import LaueTools.findorient as FindO
DEG = np.pi / 180.0
RAD = 1 / DEG
IDENTITYMATRIX = np.eye(3)

def dlat_to_rlat(dlat, angles_in_deg=1, setvolume=False):
    r"""
    Computes RECIPROCAL lattice parameters from DIRECT lattice parameters `dlat`
    :param dlat: [a,b,c, alpha, beta, gamma] angles are in degrees
    :param angles_in_deg: 1 when last three parameters are angle in degrees
    (then results angles are in degrees)
    :returns: [a*,b*,c*, alpha*, beta*, gamma*] angles are in degrees
    .. note::
        dlat stands for DIRECT (real space) lattice, rlat for RECIPROCAL lattice
    .. todo:: To remove setvolume
    """
    rlat = np.zeros(6)
    dlat = np.array(dlat)

    if angles_in_deg:
        dlat[3:] *= DEG

    if not setvolume:
        dvolume = (dlat[0] * dlat[1] * dlat[2] * np.sqrt(1
                        + 2 * np.cos(dlat[3]) * np.cos(dlat[4]) * np.cos(dlat[5])
                        - np.cos(dlat[3]) * np.cos(dlat[3])
                        - np.cos(dlat[4]) * np.cos(dlat[4])
                        - np.cos(dlat[5]) * np.cos(dlat[5])))
    elif setvolume == 1:
        dvolume = 1
    elif setvolume != 1:
        dvolume = setvolume
    elif setvolume == "a**3":
        dvolume = dlat[0] ** 3
    elif setvolume == "b**3":
        dvolume = dlat[1] ** 3
    elif setvolume == "c**3":
        dvolume = dlat[2] ** 3
    # compute reciprocal lattice parameters
    rlat[0] = dlat[1] * dlat[2] * np.sin(dlat[3]) / dvolume
    rlat[1] = dlat[0] * dlat[2] * np.sin(dlat[4]) / dvolume
    rlat[2] = dlat[0] * dlat[1] * np.sin(dlat[5]) / dvolume
    rlat[3] = np.arccos((np.cos(dlat[4]) * np.cos(dlat[5]) - np.cos(dlat[3]))
                        / (np.sin(dlat[4]) * np.sin(dlat[5])))
    rlat[4] = np.arccos((np.cos(dlat[3]) * np.cos(dlat[5]) - np.cos(dlat[4]))
                        / (np.sin(dlat[3]) * np.sin(dlat[5])))
    rlat[5] = np.arccos((np.cos(dlat[3]) * np.cos(dlat[4]) - np.cos(dlat[5]))
                        / (np.sin(dlat[3]) * np.sin(dlat[4])))
    if angles_in_deg:
        rlat[3:] *= RAD
        # convert radians into degrees
    return rlat

def calc_B_RR(latticeparameters, directspace=1, setvolume=False):
    r"""
    * Calculate B0 matrix (columns = vectors a*,b*,c*) from direct (real) space lattice parameters (directspace=1)
    * Calculate a matrix (columns = vectors a,b,c) from direct (real) space lattice parameters (directspace=0)
    :math:`\boldsymbol q_{ortho}=B_0 {\bf G^*}` where :math:`{\bf G^*}=h{\bf a^*}+k{\bf b^*}+l{\bf c^*}`
    :param latticeparameters:
        * [a,b,c, alpha, beta, gamma]    (angles are in degrees) if directspace=1
        * [a*,b*,c*, alpha*, beta*, gamma*] (angles are in degrees) if directspace=0
    :param directspace:
        * 1 (default) converts  (reciprocal) direct lattice parameters
            to (direct) reciprocal space calculates "B" matrix in the reciprocal space of input latticeparameters
        * 0  converts  (reciprocal) direct lattice parameters to (reciprocal) direct space
            calculates "B" matrix in same space of  input latticeparameters
    :param setvolume:
        * False, sets direct unit cell volume to the true volume from lattice parameters
        * 1,      sets direct unit cell volume to 1
        * 'a**3',  sets direct unit cell volume to a**3
        * 'b**3', sets direct unit cell volume to b**3
        * 'c**3',  sets direct unit cell volume to c**3
    :return: B Matrix (triangular up) from  crystal (reciprocal space) frame to orthonormal frame matrix
    :rtype: numpy array
    B matrix is used in q=U B G* formula or
        as B0  in q= (UB) B0 G*
    after Busing Levy, Acta Crysta 22 (1967), p 457
    .. math::
            \left( \begin{matrix}
            a^*  & b^*\cos \gamma^* & c^*\cos beta^*\\
            0  & b^*\sin \gamma^* &-c^*\sin \beta^*\cos \alpha\\
            0 &  0    &      c^*\sin \beta^*\sin \alpha\\
                    \end{matrix} \right)
    with
    .. math :: \cos(\alpha)=(\cos \beta^*\cos \gamma^*-\cos \alpha^*)/(\sin \beta^*\sin \gamma^*)
    and
    .. math :: c^* \sin \beta^* \sin \alpha = 1/c
    """
    B = np.zeros((3, 3), dtype=float)
    lat = 1.0 * np.array(latticeparameters)
    if directspace:  # from lattice param in one space to a matrix in other space
        rlat = dlat_to_rlat(lat, setvolume=setvolume)
        rlat[3:] *= DEG  # convert angles elements in radians
        B[0, 0] = rlat[0]
        B[0, 1] = rlat[1] * np.cos(rlat[5])
        B[1, 1] = rlat[1] * np.sin(rlat[5])
        B[0, 2] = rlat[2] * np.cos(rlat[4])
        B[1, 2] = -rlat[2] * np.sin(rlat[4]) * np.cos(lat[3] * DEG)
        B[2, 2] = rlat[2] * np.sin(rlat[4]) * np.sin(lat[3] * DEG)
        return B
    else:  # from lattice parameters in one space to a matrix in the same space
        lat = np.array(lat)
        lat[3:] *= DEG  # convert angles elements in radians
        B[0, 0] = lat[0]
        B[0, 1] = lat[1] * np.cos(lat[5])  # gamma angle
        B[1, 1] = lat[1] * np.sin(lat[5])
        B[0, 2] = lat[2] * np.cos(lat[4])  # beta angle
        B[1, 2] = (lat[2] / np.sin(lat[5]) * (np.cos(lat[3]) - np.cos(lat[5]) * np.cos(lat[4])))
        B[2, 2] = lat[2] * np.sqrt(1.0 - B[0, 2] ** 2 - B[1, 2] ** 2)
        return B
    
#☺ 2th, chi, X, Y, I
# 49.463101   -6.651456   2189.480000   3698.680000   28098.160
# 60.628779   -3.895113   2090.550000   3058.740000   8038.380
# 88.274917   5.143545   1758.360000   1901.790000   6359.930
# 62.080597   14.109168   1400.280000   3014.480000   6031.100
# 75.481662   -31.621051   3281.040000   2507.910000   5143.790
# 49.439918   -6.694806   2189.480000   3698.680000   28098.160
# 52.426207   16.596216 
latticeparams = [3.640666542,3.640666542,5.27828,90.0,90,90] #ZrO2 Tetragonal
B =  calc_B_RR(latticeparams)

B = np.array([[ 1.967027323100055e-01, -3.085660161325959e-17,3.056100612146954e-02],
               [ 0.000000000000000e+00,  1.918796530815872e-01,-1.151568276331365e-17],
               [ 0.000000000000000e+00,  0.000000000000000e+00,1.880653715231414e-01]])

tth_chi_spot1 = np.array([49.439918,   -6.694806]) #spot 0
tth_chi_spot2 = np.array([52.426207,   16.596216 ]) #spot 1

#hkl
hkl0 , hkl1 = np.array([2,1,-3]), np.array([3,1,2])
orientation = FindO.OrientMatrix_from_2hkl(hkl0, tth_chi_spot1, \
                             hkl1, tth_chi_spot2, B)
print(orientation)


[[-0.209421479191001, -0.97770818155381,   0.015144496564611],[ 0.676758371367223, -0.230793835796587,  0.699093922261295],[ 0.952674972306491,  0.042875525306998, -0.300951967048689]]

#%% Convert Euler angles to rotation matrix and then compute their equivalent suymmetries and convert back to Euler angles

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

euler_ang = 96.61, 90.29, 185.24











