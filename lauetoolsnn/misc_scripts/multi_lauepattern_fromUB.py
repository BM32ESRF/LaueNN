# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 13:43:56 2022

@author: PURUSHOT

Misoriented crystal for Laue

"""
import numpy as np
import matplotlib.pyplot as plt
from random import random as rand1
from math import acos

try:
    from lauetoolsnn.utils_lauenn import Euler2OrientationMatrix
except:
    from utils_lauenn import Euler2OrientationMatrix

try:
    import lauetoolsnn.lauetools.CrystalParameters as CP
    import lauetoolsnn.lauetools.lauecore as LT
except:
    from lauetools import CrystalParameters as CP
    from lauetools import lauecore as LT

    
def simulatemultiplepatterns(nbUBs, key_material=None, emin=5, emax=23, detectorparameters=None, pixelsize=None,
                             sortintensity = False, dim1=2048, dim2=2048, removeharmonics=1):
    
    detectordiameter = pixelsize * dim1
    g = np.zeros((nbUBs, 3, 3))
    for igr in range(nbUBs):
        phi1 = rand1() * 360.
        phi = 180. * acos(2 * rand1() - 1) / np.pi
        phi2 = rand1() * 360.
        g[igr] = Euler2OrientationMatrix((phi1, phi, phi2))

    l_tth, l_chi, l_miller_ind, l_posx, l_posy, l_E, l_intensity = [],[],[],[],[],[],[]
 
    for grainind in range(nbUBs):
        UBmatrix = g[grainind]
        grain = CP.Prepare_Grain(key_material, UBmatrix)
        s_tth, s_chi, s_miller_ind, \
            s_posx, s_posy, s_E= LT.SimulateLaue_full_np(grain, emin, emax,
                                                        detectorparameters,
                                                        pixelsize=pixelsize,
                                                        dim=(dim1, dim2),
                                                        detectordiameter=detectordiameter,
                                                        removeharmonics=removeharmonics)
        s_miller_ind = np.c_[s_miller_ind, np.zeros(len(s_miller_ind))]
        s_intensity = 1./s_E
        l_tth.append(s_tth)
        l_chi.append(s_chi)
        l_miller_ind.append(s_miller_ind)
        l_posx.append(s_posx)
        l_posy.append(s_posy)
        l_E.append(s_E)
        l_intensity.append(s_intensity)

    #flat_list = [item for sublist in l for item in sublist]
    s_tth = np.array([item for sublist in l_tth for item in sublist])
    s_chi = np.array([item for sublist in l_chi for item in sublist])
    s_miller_ind = np.array([item for sublist in l_miller_ind for item in sublist])
    s_posx = np.array([item for sublist in l_posx for item in sublist])
    s_posy = np.array([item for sublist in l_posy for item in sublist])
    s_E = np.array([item for sublist in l_E for item in sublist])
    s_intensity=np.array([item for sublist in l_intensity for item in sublist])
    
    if sortintensity:
        indsort = np.argsort(s_intensity)[::-1]
        s_tth=np.take(s_tth, indsort)
        s_chi=np.take(s_chi, indsort)
        s_miller_ind=np.take(s_miller_ind, indsort, axis=0)
        s_posx=np.take(s_posx, indsort)
        s_posy=np.take(s_posy, indsort)
        s_E=np.take(s_E, indsort)
        s_intensity=np.take(s_intensity, indsort)
        
    return s_tth, s_chi, s_miller_ind, s_posx, s_posy, s_intensity

detectorparameters = [79.50900, 977.9000, 931.8900, 0.3570000, 0.4370000]
pixelsize = 0.0734 # 0.079142 #
ccd_label_global = "sCMOS" #"MARCCD165" #"Cor"#
dim1 = 2018 #2048 #
dim2 = 2016 #2048 #
emax_global = 23
emin_global = 5

s_tth, s_chi, s_miller_ind, _, _, _ = simulatemultiplepatterns(1, key_material="Cu", 
                                                                    emin=5, emax=23,
                                                                    detectorparameters=detectorparameters,
                                                                    pixelsize=pixelsize,
                                                                    sortintensity = 1, 
                                                                    dim1=dim1, dim2=dim2, 
                                                                    removeharmonics=1)


## Plot the Laue pattern
fig = plt.figure()
plt.scatter(s_tth, s_chi, c='k')
plt.ylabel(r'$\chi$ (in deg)',fontsize=8)
plt.xlabel(r'2$\theta$ (in deg)', fontsize=10)
plt.grid(linestyle='--', linewidth=0.5)

plt.show()


