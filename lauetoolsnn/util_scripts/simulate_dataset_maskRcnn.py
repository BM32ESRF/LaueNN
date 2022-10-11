# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 23:41:32 2022

@author: PURUSHOT

GroundTruth generation for blob detection

## Gaussian function for spots
## Poisson bg noise
## energy controls the spot size (hihg energy-->small spot)
        --> calculate structure factor of each HKL and use that for intensity also
        
## Make smaller 256x256 images from big Laue image
## to convert back to big image after prediction on 256x256 images
## Annote the dataset for maskRCNN network --> TODO
"""

import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import lauetoolsnn.lauetools.lauecore as LT
import lauetoolsnn.lauetools.CrystalParameters as CP
from random import random as rand1
from math import acos
from lauetoolsnn.utils_lauenn import Euler2OrientationMatrix

DEG = np.pi / 180.0
# =============================================================================
# INPUT
# =============================================================================
key_material = 'ZrO2'

detectorparameters = [79.61200, 977.8100, 932.1700, 0.4770000, 0.4470000]
pixelsize = 0.07340000

nbUBs = 1

# =============================================================================
# FUNCTIONS
# =============================================================================
def makeGaussian2(intensity=500, x_center=50, y_center=50, theta=0, sigma_x = 10, sigma_y=10, x_size=100, y_size=100):
    # x_center and y_center will be the center of the gaussian, theta will be the rotation angle
    # sigma_x and sigma_y will be the stdevs in the x and y axis before rotation
    # x_size and y_size give the size of the frame 
    theta = 2*np.pi*theta/360
    x = np.arange(0, x_size, 1, float)
    y = np.arange(0, y_size, 1, float)
    y = y[:,np.newaxis]
    # rotation
    a=np.cos(theta)*x -np.sin(theta)*y
    b=np.sin(theta)*x +np.cos(theta)*y
    a0=np.cos(theta)*x_center -np.sin(theta)*y_center
    b0=np.sin(theta)*x_center +np.cos(theta)*y_center
    return np.sqrt(intensity)* (np.exp(-(((a-a0)**2)/(2*(sigma_x**2)) + ((b-b0)**2) /(2*(sigma_y**2)))))

def convert_binary(image_matrix, thresh_val):
    white = 255
    black = 0
    initial_conv = np.where((image_matrix <= thresh_val), image_matrix, white)
    final_conv = np.where((initial_conv > thresh_val), initial_conv, black)
    return final_conv

g = np.zeros((nbUBs, 3, 3))
for igr in range(nbUBs):
    phi1 = rand1() * 360.
    phi = 180. * acos(2 * rand1() - 1) / np.pi
    phi2 = rand1() * 360.
    g[igr] = Euler2OrientationMatrix((phi1, phi, phi2))

# =============================================================================
# Poisson noise
# =============================================================================
mask_sum_image = np.zeros((2018,2016), dtype=np.uint8)
## poisson noise
intensity_start = 50
myimage = intensity_start * np.ones((2018,2016))
imageplusnoise = np.random.poisson(lam=myimage, size=None)        
noiseonlyimage = imageplusnoise - myimage
##base noisy image
img_array = np.copy(noiseonlyimage)


app_len=0
l_miller_ind = np.zeros((1,3))
l_tth = np.zeros(1)
l_chi = np.zeros(1)
l_posx = np.zeros(1)
l_posy = np.zeros(1)
l_E = np.zeros(1)
l_intensity = np.zeros(1)

for grainind in trange(nbUBs):
    UBmatrix = g[grainind]
    grain = CP.Prepare_Grain(key_material, UBmatrix)
    s_tth, s_chi, s_miller_ind, s_posx, s_posy, s_E= LT.SimulateLaue_full_np(grain, 5, 23,
                                                                              detectorparameters,
                                                                              detectordiameter = pixelsize*2018*1.6,
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
    app_len = app_len + len(s_tth)
    print("Number of peaks in the grain ",grainind+1," is ",len(s_posx))
    
    for i in range(len(s_posx)):
        energy = 100 * (1./s_E[i])
        spot_size = int(energy//2)
        if spot_size < 2:
            ## not including very high energy i.e very tiny spots
            continue
        r = spot_size
        intensity = intensity_start
        mask = makeGaussian2(intensity=100*intensity*spot_size, 
                             theta = np.random.randint(0,180,1), 
                             sigma_x = r, 
                             sigma_y = r)
        ##for groundtruth in MaskRCnn
        mask_binary = convert_binary(mask, 10)
        
        X_pix = int(s_posx[i])
        Y_pix = int(s_posy[i])
        if 0<=X_pix<2018 and 0<=Y_pix<2016:
            xshap, yshap = img_array[X_pix-50 :X_pix+50+1, Y_pix-50 :Y_pix+50+1].shape
            xmin = X_pix-50
            xmax = X_pix+50
            ymin = Y_pix-50
            ymax = Y_pix+50
            img_array[xmin:xmax, ymin:ymax] = img_array[xmin:xmax, ymin:ymax] + mask[:xshap, :yshap]

mask_image = convert_binary(img_array, 10)

fig,ax = plt.subplots(1)
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
ax.axis('off')
ax.imshow(img_array)
ax.axis('off')
plt.show()
# plt.savefig('annote_lauePattern_ZrO2.png', bbox_inches='tight', dpi = 300, pad_inches=0.0)

fig,ax = plt.subplots(1)
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
ax.axis('off')
ax.imshow(mask_image, cmap='gray')
ax.axis('off')
plt.show()
# plt.savefig('annote_lauePattern_ZrO2.png', bbox_inches='tight', dpi = 300, pad_inches=0.0)

#%%
# =============================================================================
# ## sub cases of gaussian overlaps
# =============================================================================
mask = None
##case 1 Elongated Gaussian with random angles
intensity = intensity_start
mask = makeGaussian2(intensity=100*intensity*spot_size, 
                     theta = np.random.randint(0,180,1), 
                     sigma_x = 0.5*r, 
                     sigma_y = 2.5*r)
##for groundtruth in MaskRCnn
mask_binary = convert_binary(mask, 10)

mask = mask + noiseonlyimage[:100,:100]

fig,ax = plt.subplots(1,2)
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
ax[0].imshow(mask, cmap='hot')
ax[1].imshow(mask_binary, cmap='gray')
plt.show()

#%%
##case 2 2Gaussians
intensity = intensity_start
mask0 = makeGaussian2(intensity=100*intensity*spot_size, 
                     theta = np.random.randint(0,180,1), 
                     sigma_x = r, 
                     sigma_y = r)
##for groundtruth in MaskRCnn
mask_binary0 = convert_binary(mask0, 10)

mask1 = makeGaussian2(intensity=100*intensity*spot_size, 
                      x_center=55, y_center=55,
                      theta = np.random.randint(0,180,1), 
                      sigma_x = 0.5*r, 
                      sigma_y = 2*r)
##for groundtruth in MaskRCnn
mask_binary1 = convert_binary(mask1, 10)
mask = mask0 + mask1
mask = mask + noiseonlyimage[:100,:100]
mask_binary = mask_binary0+mask_binary1

fig,ax = plt.subplots(1,2)
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
ax[0].imshow(mask, cmap='hot')
ax[1].imshow(mask_binary, cmap='gray')
plt.show()

#%%
##case 2 Gaussian clusters
intensity = intensity_start
mask0 = makeGaussian2(intensity=100*intensity*spot_size, 
                     theta = np.random.randint(0,180,1), 
                     sigma_x = r, 
                     sigma_y = r)
##for groundtruth in MaskRCnn
mask_binary0 = convert_binary(mask0, 10)

mask1 = makeGaussian2(intensity=100*intensity*spot_size, 
                      x_center=55, y_center=55,
                      theta = np.random.randint(0,180,1), 
                      sigma_x = r, 
                      sigma_y = r)
##for groundtruth in MaskRCnn
mask_binary1 = convert_binary(mask1, 10)

mask2 = makeGaussian2(intensity=100*intensity*spot_size, 
                      x_center=45, y_center=45,
                      theta = np.random.randint(0,180,1), 
                      sigma_x = r, 
                      sigma_y = r)
##for groundtruth in MaskRCnn
mask_binary2 = convert_binary(mask2, 10)

mask3 = makeGaussian2(intensity=100*intensity*spot_size, 
                      x_center=45, y_center=50,
                      theta = np.random.randint(0,180,1), 
                      sigma_x = r, 
                      sigma_y = r)
##for groundtruth in MaskRCnn
mask_binary3 = convert_binary(mask3, 10)

mask4 = makeGaussian2(intensity=100*intensity*spot_size, 
                      x_center=50, y_center=45,
                      theta = np.random.randint(0,180,1), 
                      sigma_x = 3.5*r, 
                      sigma_y = r)
##for groundtruth in MaskRCnn
mask_binary4 = convert_binary(mask4, 10)

mask = mask0 + mask1 + mask2 + mask3 + mask4
mask = mask + noiseonlyimage[:100,:100]
mask_binary = mask_binary0+mask_binary1+mask_binary2+mask_binary3+mask_binary4

fig,ax = plt.subplots(1,2)
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
ax[0].imshow(mask, cmap='hot')
ax[1].imshow(mask_binary, cmap='gray')
plt.show()

#%%
##case 2 Gaussian clusters
intensity = intensity_start
mask0 = makeGaussian2(intensity=100*intensity*spot_size, 
                     theta = np.random.randint(0,180,1), 
                     sigma_x = r, 
                     sigma_y = r)
##for groundtruth in MaskRCnn
mask_binary0 = convert_binary(mask0, 10)

mask1 = makeGaussian2(intensity=100*intensity*spot_size, 
                      x_center=53, y_center=53,
                      theta = np.random.randint(0,180,1), 
                      sigma_x = r, 
                      sigma_y = r)
##for groundtruth in MaskRCnn
mask_binary1 = convert_binary(mask1, 10)

mask2 = makeGaussian2(intensity=100*intensity*spot_size, 
                      x_center=47, y_center=47,
                      theta = np.random.randint(0,180,1), 
                      sigma_x = r, 
                      sigma_y = r)
##for groundtruth in MaskRCnn
mask_binary2 = convert_binary(mask2, 10)

mask3 = makeGaussian2(intensity=100*intensity*spot_size, 
                      x_center=45, y_center=50,
                      theta = np.random.randint(0,180,1), 
                      sigma_x = r, 
                      sigma_y = r)
##for groundtruth in MaskRCnn
mask_binary3 = convert_binary(mask3, 10)



mask = mask0 + mask1 + mask2 + mask3
mask = mask + noiseonlyimage[:100,:100]
mask_binary = mask_binary0+mask_binary1+mask_binary2+mask_binary3

fig,ax = plt.subplots(1,2)
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
ax[0].imshow(mask, cmap='hot')
ax[1].imshow(mask_binary, cmap='gray')
plt.show()
#%%
##case 2 Gaussian clusters
intensity = intensity_start
mask0 = makeGaussian2(intensity=100*intensity*spot_size, 
                     theta = np.random.randint(0,180,1), 
                     sigma_x = r, 
                     sigma_y = r)
##for groundtruth in MaskRCnn
mask_binary0 = convert_binary(mask0, 10)

mask1 = makeGaussian2(intensity=100*intensity*spot_size, 
                      x_center=55, y_center=50,
                      theta = np.random.randint(0,180,1), 
                      sigma_x = r, 
                      sigma_y = r)
##for groundtruth in MaskRCnn
mask_binary1 = convert_binary(mask1, 10)

mask2 = makeGaussian2(intensity=100*intensity*spot_size, 
                      x_center=50, y_center=55,
                      theta = np.random.randint(0,180,1), 
                      sigma_x = r, 
                      sigma_y = r)
##for groundtruth in MaskRCnn
mask_binary2 = convert_binary(mask2, 10)

mask3 = makeGaussian2(intensity=100*intensity*spot_size, 
                      x_center=55, y_center=55,
                      theta = np.random.randint(0,180,1), 
                      sigma_x = r, 
                      sigma_y = r)
##for groundtruth in MaskRCnn
mask_binary3 = convert_binary(mask3, 10)



mask = mask0 + mask1 + mask2 + mask3
mask = mask + noiseonlyimage[:100,:100]
mask_binary = mask_binary0+mask_binary1+mask_binary2+mask_binary3

fig,ax = plt.subplots(1,2)
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
ax[0].imshow(mask, cmap='hot')
ax[1].imshow(mask_binary, cmap='gray')
plt.show()

#%%
# N = 512
# thetas = np.linspace(start=0, stop=360, num=N, endpoint=False)

# classRound = np.zeros((N, L, L))
# classOval = np.zeros((N, L, L))
# classYOvalL = np.zeros((N, L, L))
# classYOvalR = np.zeros((N, L, L))

# for i, theta in enumerate(thetas):
#     classRound[i] = np.asarray(PILImage.fromarray(round_disc).rotate(theta))
#     classOval[i] = np.asarray(PILImage.fromarray(oval_disc).rotate(theta))
#     classYOvalL[i] = np.asarray(PILImage.fromarray(yoval_discL).rotate(theta))
#     classYOvalR[i] = np.asarray(PILImage.fromarray(yoval_discR).rotate(theta))
    
    
    
#%%
## Some useful function from old EBSD UNET project
### Split image into multiple subsets of size 256 x 256
## generate images and masks for each subset
## train on these subsets and function to convert the iamge back into big one after prediction
# def make_prediction_img(x, model, target_size=100):
#     quarter_target_size = target_size // 4
#     half_target_size = target_size // 2    
#     _, sample_prediction = predict(model, x[0:target_size, 0:target_size, :], colors=[[127,127,127],[0,0,0],[255,255,255]])    
#     nb_channels = sample_prediction.shape[2]
#     dtype = sample_prediction.dtype
#     pad_width = ((quarter_target_size, target_size), (quarter_target_size, target_size), (0, 0))
#     pad_x = np.pad(x, pad_width, 'edge')
#     pad_y = np.zeros((pad_x.shape[0], pad_x.shape[1], nb_channels), dtype=dtype)

#     def update_prediction_center(row_begin, row_end, col_begin, col_end):
#         """Just update the center half of the window."""
#         x_window = pad_x[row_begin:row_end, col_begin:col_end, :]
#         _, y_window = predict(model, x_window, colors=[[127,127,127],[0,0,0],[255,255,255]])
#         y_window_center = y_window[quarter_target_size:target_size - quarter_target_size, quarter_target_size:target_size - quarter_target_size, :]
#         pad_y[row_begin + quarter_target_size:row_end - quarter_target_size, col_begin + quarter_target_size:col_end - quarter_target_size, :] = y_window_center

#     for row_begin in range(0, pad_x.shape[0], half_target_size):
#         for col_begin in range(0, pad_x.shape[1], half_target_size):
#             row_end = row_begin + target_size
#             col_end = col_begin + target_size
#             if row_end <= pad_x.shape[0] and col_end <= pad_x.shape[1]:
#                 update_prediction_center(row_begin, row_end, col_begin, col_end)

#     y = pad_y[quarter_target_size:quarter_target_size+x.shape[0], quarter_target_size:quarter_target_size+x.shape[1], :]
#     return y

def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points

def blockshaped(arr, split_height, split_width, overlap_kernal = 0.0):
    image_set = []
    img_h, img_w = arr.shape[:2]
    X_points = start_points(img_w, split_width, overlap_kernal)
    Y_points = start_points(img_h, split_height, overlap_kernal)
    count = 0
    
    if len(arr.shape) == 2:
        for i in Y_points:
            for j in X_points:
                split = arr[i:i+split_height, j:j+split_width]
                image_set.append(split)
                count += 1
        return count, np.array(image_set)
    
    elif len(arr.shape) == 3:
        for i in Y_points:
            for j in X_points:
                split = arr[i:i+split_height, j:j+split_width, :]
                image_set.append(split)
                count += 1
        return count, np.array(image_set)






