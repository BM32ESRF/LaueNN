# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:15:37 2022

@author: PURUSHOT

Convert uniform orientation ori format to MTEX readable plot

"""

import numpy as np

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

folder = r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\util_scripts"

orientations = np.loadtxt(folder + "//n500-id1.ori")
orientations = orientations.reshape((len(orientations),3,3))
phase_euler_angles = np.ones(len(orientations), dtype=np.int16)

lim_x = len(orientations)
lim_y = 1

material0_LG = "11"
header = [
        "Channel Text File",
        "Prj     lauetoolsnn",
        "Author    [Ravi raj purohit]",
        "JobMode    Grid",
        "XCells    "+str(lim_x),
        "YCells    "+str(lim_y),
        "XStep    1.0",
        "YStep    1.0",
        "AcqE1    0",
        "AcqE2    0",
        "AcqE3    0",
        "Euler angles refer to Sample Coordinate system (CS0)!    Mag    100    Coverage    100    Device    0    KV    15    TiltAngle    40    TiltAxis    0",
        "Phases    1",
        str(36)+";"+str(36)+";"+\
        str(36)+"\t"+str(90)+";"+\
            str(90)+";"+str(90)+"\t"+"Material1"+ "\t"+material0_LG+ "\t"+"????"+"\t"+"????",
        "Phase    X    Y    Bands    Error    Euler1    Euler2    Euler3    MAD    BC    BS"
        ]

euler_angles = np.zeros((len(orientations),3))    
for i in range(len(orientations)):
    euler_angles[i,:] = OrientationMatrix2Euler(orientations[i,:])
# =================CALCULATION OF POSITION=====================================
euler_angles = euler_angles.reshape((lim_x,lim_y,3))
phase_euler_angles = phase_euler_angles.reshape((lim_x,lim_y,1))

a = euler_angles
filename125 = folder+"//Cu_MTEX_UBmat_training.ctf"

f = open(filename125, "w")
for ij in range(len(header)):
    f.write(header[ij]+" \n")

for i123 in range(euler_angles.shape[1]):
    y_step = 1 * i123
    for j123 in range(euler_angles.shape[0]):
        x_step = 1 * j123
        phase_id = int(phase_euler_angles[j123,i123,0])
        eul =  str(phase_id)+'\t' + "%0.4f" % x_step +'\t'+"%0.4f" % y_step+'\t8\t0\t'+ \
                            "%0.4f" % a[j123,i123,0]+'\t'+"%0.4f" % a[j123,i123,1]+ \
                                '\t'+"%0.4f" % a[j123,i123,2]+'\t0.0001\t180\t0\n'
        string = eul
        f.write(string)
f.close()




