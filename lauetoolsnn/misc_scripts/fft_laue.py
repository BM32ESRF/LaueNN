# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 15:15:31 2022

@author: PURUSHOT

Trial of using IFT and FT to the Laue image and Laue spots

"""
import sys
sys.path.append(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\lauetools")
import imageprocessing as ImProc


## read a Laue image
import numpy as np
from matplotlib import pyplot as plt

example_img = r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\util_scripts\img\HS261120b_SH2_S5_B_0000.tif"

dataimage = plt.imread(example_img)
framedim = dataimage.shape


# correct background of the image
backgroundimage = ImProc.compute_autobackground_image(dataimage, boxsizefilter=10)
corrected_data = ImProc.computefilteredimage(dataimage, backgroundimage, "sCMOS", usemask=True,
                                                    formulaexpression="A-B")
corrected_data = np.copy(backgroundimage)
## plot the image and visualize it
# plt.imshow(corrected_data, cmap='gray')
# plt.imshow(np.log(corrected_data), cmap='gray')

# plt.imshow(corrected_data[1500:1750,:], cmap='gray')
# plt.imshow(np.log(corrected_data[1500:1750,:]), cmap='gray')


#%% FFT
from scipy.fft import fftfreq
from scipy.fft import fft, ifft, fft2, ifft2 

img_FT = fft2(corrected_data)
# plt.imshow(np.log(np.abs(img_FT)), cmap='gray', vmin=2, vmax=15)
plt.imshow(np.abs(img_FT), cmap='gray', vmax=1e3)

plt.colorbar()



# img_alt = np.abs(ifft2(corrected_data[750:1000,950:1050]))
# img_alt = np.abs(ifft2(corrected_data))

# # plt.imshow(img_alt, cmap='gray')
# plt.imshow(np.log(img_alt), cmap='gray')
# plt.colorbar()





#%%
import sys
sys.path.append(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\lauetools")
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
## LaueTools import
import generaltools as GT
import lauecore as LT
import CrystalParameters as CP
## External libraries
## for faster binning of histogram
## C version of hist
# from fast_histogram import histogram1d
# import os
# from adjustText import adjust_text
from scipy.ndimage import convolve
from math import acos

def Euler2OrientationMatrix(euler):
    """Compute the orientation matrix :math:`\mathbf{g}` associated with
    the 3 Euler angles :math:`(\phi_1, \Phi, \phi_2)`.
    :param euler: The triplet of the Euler angles (in degrees).
    :return g: The 3x3 orientation matrix.
    """
    (rphi1, rPhi, rphi2) = np.radians(euler)
    c1 = np.cos(rphi1)
    s1 = np.sin(rphi1)
    c = np.cos(rPhi)
    s = np.sin(rPhi)
    c2 = np.cos(rphi2)
    s2 = np.sin(rphi2)
    # rotation matrix g
    g11 = c1 * c2 - s1 * s2 * c
    g12 = s1 * c2 + c1 * s2 * c
    g13 = s2 * s
    g21 = -c1 * s2 - s1 * c2 * c
    g22 = -s1 * s2 + c1 * c2 * c
    g23 = c2 * s
    g31 = s1 * s
    g32 = -c1 * s
    g33 = c
    g = np.array([[g11, g12, g13], [g21, g22, g23], [g31, g32, g33]])
    return g

DEG = np.pi / 180.0

key_material = 'ZrO2'
patch = 'square'
r = 5
detectorparameters = [70, 1039, 944, 0.747, 0.071]
pixelsize = 0.079142

nbUBs = 1
seed = 20
np.random.seed(seed)
UBelemagnles = np.random.random((nbUBs,3))*360-180.
###########################################################
# UBelemagnles[1,:] =UBelemagnles[0,:] 
# UBelemagnles[1,2] = UBelemagnles[1,2] + 1
g = np.zeros((nbUBs, 3, 3))
for igr in range(nbUBs):
    if igr == 0:
        np.random.seed(seed)
        phi1 = np.random.random(1) * 360.
        np.random.seed(seed+5)
        phi = 180. * acos(2 * np.random.random(1) - 1) / np.pi
        np.random.seed(seed+10)
        phi2 = np.random.random(1) * 360.
    if igr == 1:
        phi1 = phi1
        phi = phi
        phi2 = phi2+1
    g[igr] = Euler2OrientationMatrix((float(phi1), float(phi), float(phi2)))

###########################################################
img_array = np.zeros((2048,2048), dtype=np.uint8)
app_len=0

l_miller_ind = np.zeros((1,3))
l_tth = np.zeros(1)
l_chi = np.zeros(1)
l_posx = np.zeros(1)
l_posy = np.zeros(1)
l_E = np.zeros(1)
l_intensity = np.zeros(1)

for grainind in range(nbUBs):
    UBmatrix = g[grainind]
    
    grain = CP.Prepare_Grain(key_material, UBmatrix)

    s_tth, s_chi, s_miller_ind, s_posx, s_posy, s_E= LT.SimulateLaue_full_np(grain, 5, 23,
                                                                              detectorparameters,
                                                                              detectordiameter = pixelsize * 2048*3,
                                                                              pixelsize=pixelsize,
                                                                              removeharmonics=1)
    s_intensity = 1./s_E
    
    l_tth = np.hstack((l_tth, s_tth))
    l_chi = np.hstack((l_chi, s_chi))
    l_posx = np.hstack((l_posx, s_posx))
    l_posy = np.hstack((l_posy, s_posy))
    l_E = np.hstack((l_E, s_E))
    l_intensity = np.hstack((l_intensity, s_intensity))
    l_miller_ind = np.vstack((l_miller_ind, s_miller_ind))

    Xpix = s_posx
    Ypix = s_posy
    app_len = app_len + len(s_tth)
    
    # print("Number of peaks in the grain ",g_id," is ",len(Xpix))
    # make circle peak patch (currently square pattern) =DONE
    if patch == "circular": ##fast version
        y,x = np.ogrid[-r: r+1, -r: r+1]
        mask = x**2+y**2 <= r**2
        mask = 1*mask.astype(float)
        for i in range(len(Xpix)):
            X_pix = int(Xpix[i])
            Y_pix = int(Ypix[i])
            if X_pix<0 or X_pix>=2048 or Y_pix<0 or Y_pix>=2048:
                continue
            init_img = np.zeros((2048,2048), dtype=np.uint8)
            init_img[X_pix, Y_pix] = 1
            b = convolve(init_img, mask)-sum(sum(mask)) + 1
            img_array[:, :] = img_array[:, :] + b
    elif patch == "square":
        for i in range(len(Xpix)):
            X_pix = int(Xpix[i])
            Y_pix = int(Ypix[i])
            img_array[X_pix-r:X_pix+r, Y_pix-r:Y_pix+r] = 1

l_tth = np.delete(l_tth, 0, axis=0)
l_chi = np.delete(l_chi, 0, axis=0)
l_posx = np.delete(l_posx, 0, axis=0)
l_posy = np.delete(l_posy, 0, axis=0)
l_E = np.delete(l_E, 0, axis=0)
l_intensity = np.delete(l_intensity, 0, axis=0)
l_miller_ind = np.delete(l_miller_ind, 0, axis=0)
###########################################################
fig = plt.figure()
plt.scatter(l_tth, l_chi, c='k')
plt.ylabel(r'$\chi$ (in deg)',fontsize=8)
plt.xlabel(r'2$\theta$ (in deg)', fontsize=10)
plt.grid(linestyle='--', linewidth=0.5)
plt.show(fig)
##########################################################

##convert scatter data to image data
x_, x_idx = np.unique(np.ravel(l_tth), return_inverse=True)
y_, y_idx = np.unique(np.ravel(l_chi), return_inverse=True)
newArray = np.zeros((len(y_),len(x_)), dtype=l_tth.dtype)
newArray[x_idx, y_idx] = 100
for i in range(1,10):
    
    try:
        newArray[x_idx+i, y_idx] = 1
    except:
        continue
    try:
        newArray[x_idx, y_idx+i] = 1
    except:
        continue
    try:
        newArray[x_idx+i, y_idx+i] = 1
    except:
        continue
    
    try:
        newArray[x_idx-i, y_idx] = 1
    except:
        continue
    try:
        newArray[x_idx, y_idx-i] = 1
    except:
        continue
    try:
        newArray[x_idx-i, y_idx-i] = 1
    except:
        continue
    try:
        newArray[x_idx+i, y_idx-i] = 1
    except:
        continue
    try:
        newArray[x_idx-i, y_idx+i] = 1
    except:
        continue
fig10 = plt.figure()
plt.imshow(newArray, cmap='gray')#,vmax=50)
plt.colorbar()
plt.show(fig10)


from scipy.fft import fftfreq
from scipy.fft import fft, ifft, fft2, ifft2 

fig10 = plt.figure()
plt.imshow(img_array, cmap='gray')#,vmax=50)
plt.colorbar()
plt.show(fig10)


img_FT = fft2(img_array)

fig1 = plt.figure()
plt.imshow(np.abs(img_FT), cmap='gray',vmax=10)
plt.colorbar()
plt.show(fig1)



img_alt = np.abs(ifft2(img_array))

fig100 = plt.figure()
plt.imshow(img_alt, cmap='gray')
# plt.imshow(np.log(img_alt), cmap='gray')
plt.colorbar()
plt.show(fig100)












