# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 20:16:23 2022

@author: PURUSHOT

Get average orientations from a list of orientations

Also filter them with low misorientation angles

"""
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import trange
# import sys
# sys.path.append(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\lauetools")
# import CrystalParameters as CP
# import dict_LaueTools as dictLT

### Please enter the path of the results.npz file (which is extracted from the LauetoolsNN results directory)
result_directory = r"C:\Users\purushot\Desktop\SiC\results_SiC_2022-08-29_16-03-21"
save_direc = result_directory + "\\" + "additional_plots"

if not os.path.exists(save_direc):
    os.makedirs(save_direc)
    
results_obj = np.load(result_directory + "//results.npz", allow_pickle=True)
print("Results loaded successfully")

best_match = results_obj["arr_0"] # contains all the details regarding the indexation
mat_global = results_obj["arr_1"] # material id of each pixel (to identify if it is Si or Cu)
rotation_matrix1 = results_obj["arr_2"] # 3x3 rotation matrix for each pixel (in Lauetools frame)
strain_matrix = results_obj["arr_3"] # 3x3 strain matrix of each UB matrix for each pixel (crystal reference frame)
strain_matrixs = results_obj["arr_4"] # 3x3 strain matrix of each UB matrix for each pixel (sample reference frame)
col = results_obj["arr_5"] # Color in sample Z direction confroming the IPF color (be careful)
colx = results_obj["arr_6"] # Color in sample X direction confroming the IPF color (be careful)
coly = results_obj["arr_7"] # Color in sample Y direction confroming the IPF color (be careful)
match_rate = results_obj["arr_8"] # Matching rate of each indexation
files_treated = results_obj["arr_9"] # File names of each pixel
lim_x = results_obj["arr_10"] # Limit X of raster scan
lim_y = results_obj["arr_11"] # Limit Y of raster scan
spots_len = results_obj["arr_12"] # Number of spots per pattern indexed
iR_pix = results_obj["arr_13"] # initial pixel residues before strain refinement
fR_pix = results_obj["arr_14"] # final residue pixel after strain refinement
ub_matricies = len(rotation_matrix1) # Total UB matricies
material_ = str(results_obj["arr_15"]) # Material name for mat id 1
material1_ = str(results_obj["arr_16"]) # Material name for mat id 2
material_id = [material_, material1_]

match_tol = 70
pixel_residues = 2
mat = 0
bins = 30

#%%
### lets build a filtered version of UB matricies
rotation_matrix = []
for i in range(len(rotation_matrix1)):
    temp_mat = rotation_matrix1[i][0]
    temp_matglobal = mat_global[i][0]
    temp_mr = match_rate[i][0]
    temp_fr = fR_pix[i][0]
    for j in range(len(temp_mat)):
        if (temp_mr[j] < match_tol) or (temp_fr[j] > pixel_residues):
            continue
        orientation_matrix = temp_mat[j,:,:]
        
        ## rotate orientation by 40degrees to bring in Sample RF
        omega = np.deg2rad(-40.0)
        # rotation de -omega autour de l'axe x (or Y?) pour repasser dans Rsample
        cw = np.cos(omega)
        sw = np.sin(omega)
        mat_from_lab_to_sample_frame = np.array([[cw, 0.0, sw], [0.0, 1.0, 0.0], [-sw, 0, cw]])
        orientation_matrix = np.dot(mat_from_lab_to_sample_frame, orientation_matrix)
        if np.linalg.det(orientation_matrix) < 0:
            orientation_matrix = -orientation_matrix
        
        rotation_matrix.append(orientation_matrix)
rotation_matrix = np.array(rotation_matrix)
#%%
##Import all functions
import numpy as np
import sys, time

from numba import njit, prange

@njit(parallel=True)
def misorientation_with_matrices(A, B, C):
    """
    Calculates all possible misorientation combinations between
    A and B using matrix operations
    
    Parameters
    ----------
    A: numpy ndarray(n, 3, 3)
        3D array listin n rotation matrices
    B: numpy ndarray(m, 3, 3)
        3D array listing m rotation matrices
    C: numpy ndarray(24, 3, 3)
        List of cubic symmetry operators
    
    Returns
    -------
    mis: numpy ndarray(n, m)
        Misorientation angle in radians
    """
    size_A, size_B, size_C = len(A), len(B), len(C)
    # Initialize empty mis array
    mis = np.zeros((size_A, size_B))
    # prange is used for parallelizing 
    for i in prange(size_A):
        for j in prange(size_B):
            trmax = -1.0
            for k in range(size_C):
                # T = C * A * B^T
                # tr = T_11 + T_22 + T_33
                tr = np.trace(C[k].dot(A[i]).dot(B[j].T))
                # Finds maximum value of trace (minimum angle)
                if tr > trmax:
                    trmax = tr
            # Due to rounding, trmax is slightly greater than 3, which is not allowed
            if trmax > 3.0:
                mis[i, j] = 0.0
            else:
                mis[i, j] = np.arccos((trmax - 1.0)/2.0)
    return mis

def list_cubic_symmetry_operators():
    """
    Lists symmetry matrices for cubic symmetry group
    """
    axis_angle = np.array([
        # Identity
        [0., 1., 0., 0.],
        # 2-fold on <100>
        [np.pi, 1., 0., 0.],
        [np.pi, 0., 1., 0.],
        [np.pi, 0., 0., 1.],
        # 4-fold on <100>
        [np.pi/2., 1., 0., 0.],
        [np.pi/2., 0., 1., 0.],
        [np.pi/2., 0., 0., 1.],
        [np.pi/2., -1., 0., 0.],
        [np.pi/2., 0., -1., 0.],
        [np.pi/2., 0., 0., -1.],
        # 2-fold on <110>
        [np.pi, 1., 1., 0.],
        [np.pi, 1., 0., 1.],
        [np.pi, 0., 1., 1.],
        [np.pi, 1., -1., 0.],
        [np.pi, -1., 0., 1.],
        [np.pi, 0., 1., -1.],
        # 3-fold on <111>
        [np.pi*2./3., 1., 1., 1.],
        [np.pi*2./3., 1., -1., 1.],
        [np.pi*2./3., -1., 1., 1.],
        [np.pi*2./3., -1., -1., 1.],
        [np.pi*2./3., 1., 1., -1.],
        [np.pi*2./3., 1., -1., -1.],
        [np.pi*2./3., -1., 1., -1.],
        [np.pi*2./3., -1., -1., -1.]])

    # Round and convert float to int. The elements of the operators are
    # the integers 0, 1, and -1
    return axis_angle_to_rotation_matrix(axis_angle[:, 1:], axis_angle[:, 0]).round(0).astype(int)

def axis_angle_to_rotation_matrix(axis, theta):
    theta_dim = np.ndim(theta)
    axis_dim = np.ndim(axis)

    if axis_dim != theta_dim + 1:
        raise Exception('Invalid shapes of theta or axis')

    if theta_dim == 0:
        theta = np.asarray(theta).reshape(-1)
        axis = np.asarray(axis).reshape(-1, 3)

    axis = axis/np.linalg.norm(axis, axis=1).reshape(-1, 1)

    N = len(theta)
    R = np.ndarray((N, 3, 3))

    ctheta = np.cos(theta)
    ctheta1 = 1 - ctheta
    stheta = np.sin(theta)

    R[:, 0, 0] = ctheta1*axis[:, 0]**2. + ctheta
    R[:, 0, 1] = ctheta1*axis[:, 0]*axis[:, 1] - axis[:, 2]*stheta
    R[:, 0, 2] = ctheta1*axis[:, 0]*axis[:, 2] + axis[:, 1]*stheta
    R[:, 1, 0] = ctheta1*axis[:, 1]*axis[:, 0] + axis[:, 2]*stheta
    R[:, 1, 1] = ctheta1*axis[:, 1]**2. + ctheta
    R[:, 1, 2] = ctheta1*axis[:, 1]*axis[:, 2] - axis[:, 0]*stheta
    R[:, 2, 0] = ctheta1*axis[:, 2]*axis[:, 0] - axis[:, 1]*stheta
    R[:, 2, 1] = ctheta1*axis[:, 2]*axis[:, 1] + axis[:, 0]*stheta
    R[:, 2, 2] = ctheta1*axis[:, 2]**2. + ctheta

    if theta_dim == 0:
        R = R.reshape(3, 3)

    return R

def euler_angles_to_rotation_matrix(phi1, Phi, phi2, conv='zxz', **kwargs):
    """
    Given 3 Euler angles, calculates rotation R (active description).
    Please notice that the Euler angles in the ang files follow passive
    description.
    Parameters:
    -----------
    phi1 : float or list, tuple, or array(N)
    Phi : float or list, tuple, or array(N)
    phi2 : float or list, tuple, or array(N)
        Euler angles
    conv : string (optional)
        Rotation convention
        Default: zxz (Bunge notation)
    **kwargs :
        verbose : boolean
            If True (default), print calculation time
    Returns
    -------
    R : numpy ndarray
        Rotation matrix shape(3,3) list of rotation matrices (shape(N, 3, 3))
        describing the rotation from the crystal coordinate frame to the
        sample coordinate frame (EBSD system). Notice that R is different of
        M = R^-1, which represents the rotation from the crystal coordinate
        frame to the sample coordinate frame
    """
    verbose = kwargs.pop('verbose', True)
    if verbose:
        t0 = time.time()
        sys.stdout.write('Calculating rotation matrices... ')
        sys.stdout.flush()

    if np.ndim(phi1) == 0:
        N = 1
    else:
        phi1 = np.asarray(phi1)
        Phi = np.asarray(Phi)
        phi2 = np.asarray(phi2)
        if len(phi1) == len(Phi) and len(phi1) == len(phi2):
            N = len(phi1)
        else:
            raise Exception('Lengths of phi1, Phi, and phi2 differ')

    cphi1, sphi1 = np.cos(phi1), np.sin(phi1)
    cPhi, sPhi = np.cos(Phi), np.sin(Phi)
    cphi2, sphi2 = np.cos(phi2), np.sin(phi2)
    R = np.ndarray((N, 3, 3))

    conv = conv.lower()
    if conv == 'zxz':
        R[:, 0, 0] = cphi1*cphi2 - sphi1*cPhi*sphi2
        R[:, 0, 1] = -cphi1*sphi2 - sphi1*cPhi*cphi2
        R[:, 0, 2] = sphi1*sPhi
        R[:, 1, 0] = sphi1*cphi2 + cphi1*cPhi*sphi2
        R[:, 1, 1] = -sphi1*sphi2 + cphi1*cPhi*cphi2
        R[:, 1, 2] = -cphi1*sPhi
        R[:, 2, 0] = sPhi*sphi2
        R[:, 2, 1] = sPhi*cphi2
        R[:, 2, 2] = cPhi
    elif conv == 'xyz':
        R[:, 0, 0] = cPhi*cphi1
        R[:, 0, 1] = -cPhi*sphi1
        R[:, 0, 2] = sPhi
        R[:, 1, 0] = cphi2*sphi1 + sphi2*sPhi*cphi1
        R[:, 1, 1] = cphi2*cphi1 - sphi2*sPhi*sphi1
        R[:, 1, 2] = -sphi2*cPhi
        R[:, 2, 0] = sphi2*sphi1 - cphi2*sPhi*cphi1
        R[:, 2, 1] = sphi2*cphi1 + cphi2*sPhi*sphi1
        R[:, 2, 2] = cphi2*cPhi
    else:
        raise Exception('"{}" convention not supported'.format(conv))

    if np.ndim(phi1) == 0:
        R = R.reshape(3, 3)

    if verbose:
        sys.stdout.write('{:.2f} s\n'.format(time.time() - t0))
        sys.stdout.flush()

    return R


def rotation_matrix_to_euler_angles(R, conv='zxz', avg=False, **kwargs):
    """
    Calculates the Euler angles from a rotation matrix or a sequence
    of rotation matrices (active description).
    Please notice that the Euler angles in the ang files follow passive
    description.
    Parameters:
    -----------
    R : numpy array shape(3, 3) or shape(N, 3, 3)
        Rotation matrix or list or rotation matrices
    conv : string (optional)
        Rotation convention
        Default: zxz (Bunge notation)
    **kwargs :
        verbose : boolean
            If True (default), print calculation time
        avg : boolean
            If True, calculates the Euler angles corresponding to the
            average orientation.
            If False (default), simply calculates the Euler angles for
            each rotation matrix provided.
    """

    Rdim = np.ndim(R)
    if Rdim == 2:
        R = R.reshape(1, 3, 3)
        
    if avg:
        print("Entering in avg orientation section")
        verbose = kwargs.pop('verbose', True)
        if verbose:
            t0 = time.time()
            sys.stdout.write('Calculating Euler angles... ')
            sys.stdout.flush()

        Phi = np.arccos(R[:, 2, 2])
        sPhi = np.sin(Phi)
        cphi1, cphi2 = -R[:, 1, 2]/sPhi, R[:, 2, 1]/sPhi
        sphi1, sphi2 = R[:, 0, 2]/sPhi, R[:, 2, 0]/sPhi

        # arctan2 returns value in the range [-pi,pi].
        phi1, phi2 = np.arctan2(sphi1, cphi1), np.arctan2(sphi2, cphi2)
        neg1, neg2 = phi1 < 0, phi2 < 0
        if np.ndim(neg1) > 0:
            # phi1 and phi2 to range [0, 2pi]
            phi1[neg1] = phi1[neg1] + 2.*np.pi
            phi2[neg2] = phi2[neg2] + 2.*np.pi
        else:
            if neg1:
                phi1 += 2.*np.pi
            if neg2:
                phi2 += 2.*np.pi

        if Rdim == 2:
            phi1, Phi, phi2 = phi1[0], Phi[0], phi2[0]

        if verbose:
            sys.stdout.write('{:.2f} s\n'.format(time.time() - t0))
            sys.stdout.flush()
    else:
        Phi = np.arccos(np.mean(R[:, 2, 2]))
        sPhi = np.sin(Phi)
        cphi1, cphi2 = -np.mean(R[:, 1, 2])/sPhi, np.mean(R[:, 2, 1])/sPhi
        sphi1, sphi2 = np.mean(R[:, 0, 2])/sPhi, np.mean(R[:, 2, 0])/sPhi
        phi1, phi2 = np.arctan2(sphi1, cphi1), np.arctan2(sphi2, cphi2)
        R_avg = euler_angles_to_rotation_matrix(phi1, Phi, phi2, verbose=False)
        # n=kwargs.pop('n', 5), maxdev=kwargs.pop('maxdev', .25)
        R_avg = minimize_disorientation(R, R_avg, **kwargs)
        phi1, Phi, phi2 = rotation_matrix_to_euler_angles(R_avg, verbose=False, avg=True)  # recursive

    return phi1, Phi, phi2

def minimize_disorientation(V, V0, **kwargs):
    """
    Calculates the orientation that truly minimizes the disorientation
    between the list of orientations V and a single orientation V0.
    """
    n = kwargs.pop('n', 5)  # grid size
    maxdev = kwargs.pop('maxdev', .25)  # maximum deviation in degrees
    maxdev = np.radians(maxdev)  # maxdev in radians
    it = kwargs.pop('it', 3)  # number of iterations
    verbose = kwargs.pop('verbose', False)
    if verbose:
        sys.stdout.write('\nMinimizing disorientation...\n')
        sys.stdout.flush()
    for i in range(it):
        t = np.linspace(-maxdev, maxdev, n)
        # Euler space in XYZ convention
        theta, phi, psi = np.meshgrid(t, t, t)
        theta, phi, psi = theta.ravel(), phi.ravel(), psi.ravel()
        # Rotation matrices from euler angles shape(n^3, 3, 3)
        A = euler_angles_to_rotation_matrix(theta, phi, psi, conv='xyz', verbose=False)
        # Rotate V0 by A. Resulting B is shape(n^3, 3, 3)
        B = np.tensordot(A, V0, axes=[[-1], [-2]])
        # Calculate rotation from B to V. Resulting D is shape(len(V), n^3, 3, 3)
        D = np.tensordot(V, B.transpose(0, 2, 1), axes=[[-1], [-2]]).transpose([0, 2, 1, 3])
        # Average (mean) trace of D along axis 0 shape(n^3)
        tr = np.abs(np.trace(D, axis1=2, axis2=3)).mean(axis=0)
        # Index of maximum trace value
        imax = np.argmax(tr)
        if verbose:
            dth = np.degrees(np.arccos((np.trace(A[imax])-1.)/2.))
            sys.stdout.write('{:2d} : {:g}, {:g}, {:g}; mis = {:g} deg\n'.format(
                i+1, np.degrees(theta[imax]), np.degrees(phi[imax]), np.degrees(psi[imax]), dth))
        V0 = A[imax].dot(V0)
        maxdev /= n
    del A, B, D, tr

    return V0
#%%
### Lets compute the mutual misorientation of these UB matricies
sym = list_cubic_symmetry_operators().astype(float)

eul_ang = rotation_matrix_to_euler_angles(rotation_matrix, avg=True)
mat_back = euler_angles_to_rotation_matrix(*eul_ang)

mis = np.rad2deg(misorientation_with_matrices(mat_back, mat_back, sym))

# misorientation_angles = np.zeros((len(rotation_matrix), len(rotation_matrix)))
# misorientation_crystalaxis = np.zeros((len(rotation_matrix), len(rotation_matrix), 3))
# misorientation_sampleaxis = np.zeros((len(rotation_matrix), len(rotation_matrix), 3))

# for i in trange(len(rotation_matrix)):
#     if i >10:
#         continue
#     for j in range(len(rotation_matrix)):
        
#         result = disorientation(rotation_matrix[i,:,:], rotation_matrix[j,:,:], sym)
#         misorientation_angles[i,j] = result[0]
#         misorientation_crystalaxis[i,j,:] = result[1]
#         misorientation_sampleaxis[i,j,:] = result[2]




### club together the <1 deg misoriented UBs 


### average the clubbed UBs


### Write out the UB matrix text file to be later used with LaueNN for induvidual orientation mapping










