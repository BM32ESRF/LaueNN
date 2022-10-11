# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 12:06:15 2022

@author: PURUSHOT

Convert pixels to Gnemonic and do Hough transform
"""
import numpy as np
from tqdm import trange
import LaueTools.generaltools as GT
import LaueTools.lauecore as LT
import LaueTools.CrystalParameters as CP
# from adjustText import adjust_text
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage


DEG = np.pi / 180.0
def ComputeGnomon_singledata(tth, chi, CenterProjection=(45 * DEG, 0 * DEG)):
    data_theta = tth / 2.0
    data_chi = chi
    lat = np.arcsin(np.cos(data_theta * DEG) * np.cos(data_chi * DEG))  # in rads
    longit = np.arctan(
        -np.sin(data_chi * DEG) / np.tan(data_theta * DEG))  # + ones(len(data_chi))*(np.pi)
    centerlat, centerlongit = CenterProjection
    slat0 = np.sin(centerlat)
    clat0 = np.cos(centerlat)
    longit0 = centerlongit
    slat = np.sin(lat)
    clat = np.cos(lat)
    cosanguldist = slat * slat0 + clat * clat0 * np.cos(longit - longit0)
    Xgno = clat * np.sin(longit0 - longit) / cosanguldist
    Ygno = (slat * clat0 - clat * slat0 * np.cos(longit - longit0)) / cosanguldist
    NbptsGno = 300
    maxsize = max(Xgno,Ygno,-Xgno,-Ygno)+.0
    xgnomin,xgnomax,ygnomin,ygnomax=(-0.8,0.8,-0.5,0.5)
    xgnomin,xgnomax,ygnomin,ygnomax=(-maxsize,maxsize,-maxsize,maxsize)
    XGNO = int((Xgno-xgnomin)/(xgnomax-xgnomin)*NbptsGno)
    YGNO = int((Ygno-ygnomin)/(ygnomax-ygnomin)*NbptsGno)
    return np.array((XGNO, YGNO))

def ComputeGnomon_arraydata(tth, chi, CenterProjection=(45 * DEG, 0 * DEG)):
    data_theta = tth / 2.0
    data_chi = chi
    lat = np.arcsin(np.cos(data_theta * DEG) * np.cos(data_chi * DEG))  # in rads
    longit = np.arctan(
        -np.sin(data_chi * DEG) / np.tan(data_theta * DEG))  # + ones(len(data_chi))*(np.pi)
    centerlat, centerlongit = CenterProjection
    slat0 = np.sin(centerlat)
    clat0 = np.cos(centerlat)
    longit0 = centerlongit
    slat = np.sin(lat)
    clat = np.cos(lat)
    cosanguldist = slat * slat0 + clat * clat0 * np.cos(longit - longit0)
    Xgno = clat * np.sin(longit0 - longit) / cosanguldist
    Ygno = (slat * clat0 - clat * slat0 * np.cos(longit - longit0)) / cosanguldist
    NbptsGno = 300
    maxsize = max(Xgno.max(),Ygno.max(),-Xgno.min(),-Ygno.min())+.0
    xgnomin,xgnomax,ygnomin,ygnomax=(-0.8,0.8,-0.5,0.5)
    xgnomin,xgnomax,ygnomin,ygnomax=(-maxsize,maxsize,-maxsize,maxsize)
    XGNO = np.array((Xgno-xgnomin)/(xgnomax-xgnomin)*NbptsGno, dtype=int)
    YGNO = np.array((Ygno-ygnomin)/(ygnomax-ygnomin)*NbptsGno, dtype=int)
    return np.array((XGNO, YGNO)).T

def ComputeGnomon_2(TwiceTheta_Chi, CenterProjection=(45 * DEG, 0 * DEG)):
    data_theta = TwiceTheta_Chi[0] / 2.0
    data_chi = TwiceTheta_Chi[1]
    lat = np.arcsin(np.cos(data_theta * DEG) * np.cos(data_chi * DEG))  # in rads
    longit = np.arctan(
        -np.sin(data_chi * DEG) / np.tan(data_theta * DEG))  # + ones(len(data_chi))*(np.pi)
    centerlat, centerlongit = CenterProjection
    slat0 = np.ones(len(data_chi)) * np.sin(centerlat)
    clat0 = np.ones(len(data_chi)) * np.cos(centerlat)
    longit0 = np.ones(len(data_chi)) * centerlongit
    slat = np.sin(lat)
    clat = np.cos(lat)
    cosanguldist = slat * slat0 + clat * clat0 * np.cos(longit - longit0)
    _gnomonx = clat * np.sin(longit0 - longit) / cosanguldist
    _gnomony = (slat * clat0 - clat * slat0 * np.cos(longit - longit0)) / cosanguldist
    return _gnomonx, _gnomony

def computeGnomonicImage(TwiceTheta,Chi):
    DEG = np.pi/180.
    # CenterProjectionAngleTheta = 50#45
    TwiceTheta_Chi = TwiceTheta,Chi
    Xgno,Ygno = ComputeGnomon_2(TwiceTheta_Chi, CenterProjection=(45 * DEG, 0 * DEG))
    pts =(np.array([Xgno,Ygno]).T)
    nbpeaks=len(pts)
    NbptsGno = 300
    maxsize = max(Xgno.max(),Ygno.max(),-Xgno.min(),-Ygno.min())+.0
    xgnomin,xgnomax,ygnomin,ygnomax=(-0.8,0.8,-0.5,0.5)
    xgnomin,xgnomax,ygnomin,ygnomax=(-maxsize,maxsize,-maxsize,maxsize)
    
    halfdiagonal = np.sqrt(xgnomax**2+ygnomax**2)*NbptsGno
    XGNO = np.array((Xgno-xgnomin)/(xgnomax-xgnomin)*NbptsGno, dtype=int)
    YGNO = np.array((Ygno-ygnomin)/(ygnomax-ygnomin)*NbptsGno, dtype=int)
    imageGNO=np.zeros((NbptsGno+1,NbptsGno+1))
    imageGNO[XGNO,YGNO]=100
    return imageGNO, nbpeaks, halfdiagonal

patch = 'square'
r = 10
detectorparameters = [70, 1039, 944, 0.747, 0.071]
pixelsize = 0.079142
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
colu = []

key_material = 'Cu'

nbUBs = 5
sec_mat=True
seed = 10
np.random.seed(seed)
UBelemagnles = np.random.random((nbUBs,3))*360-180.

for angle_X, angle_Y, angle_Z in UBelemagnles:
    UBmatrix = GT.fromelemangles_toMatrix([angle_X, angle_Y, angle_Z])
    grain = CP.Prepare_Grain(key_material, UBmatrix)
    s_tth, s_chi, s_miller_ind, s_posx, s_posy, s_E= LT.SimulateLaue_full_np(grain, 5, 23,
                                                                             detectorparameters,
                                                                             detectordiameter = pixelsize * 2048,
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
    for _ in range(len(s_tth)):
        colu.append("k")
    Xpix = s_posx
    Ypix = s_posy
    app_len = app_len + len(s_tth)    
    if patch == "square":
        for i in range(len(Xpix)):
            X_pix = int(Xpix[i])
            Y_pix = int(Ypix[i])
            img_array[X_pix-r:X_pix+r, Y_pix-r:Y_pix+r] = 1

if sec_mat:
    key_material = 'Si'
    nbUBs = 1
    seed = 15
    np.random.seed(seed)
    UBelemagnles = np.random.random((nbUBs,3))*360-180.
    for angle_X, angle_Y, angle_Z in UBelemagnles:
        UBmatrix = GT.fromelemangles_toMatrix([angle_X, angle_Y, angle_Z])
        grain = CP.Prepare_Grain(key_material, UBmatrix)
        s_tth, s_chi, s_miller_ind, s_posx, s_posy, s_E= LT.SimulateLaue_full_np(grain, 5, 23,
                                                                                 detectorparameters,
                                                                                 detectordiameter = pixelsize * 2048,
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
        for _ in range(len(s_tth)):
            colu.append("r")
        Xpix = s_posx
        Ypix = s_posy
        app_len = app_len + len(s_tth)    
        if patch == "square":
            for i in range(len(Xpix)):
                X_pix = int(Xpix[i])
                Y_pix = int(Ypix[i])
                img_array[X_pix-r:X_pix+r, Y_pix-r:Y_pix+r] = 2

print("Number of spots: " + str(app_len))
l_tth = np.delete(l_tth, 0, axis=0)
l_chi = np.delete(l_chi, 0, axis=0)
l_posx = np.delete(l_posx, 0, axis=0)
l_posy = np.delete(l_posy, 0, axis=0)
l_E = np.delete(l_E, 0, axis=0)
l_intensity = np.delete(l_intensity, 0, axis=0)
l_miller_ind = np.delete(l_miller_ind, 0, axis=0)

fig = plt.figure()
plt.imshow(img_array)
plt.ylabel(r'Y pixel',fontsize=10)
plt.xlabel(r'X pixel', fontsize=10)
plt.show()
# plt.savefig(key_material+'_'+str(seed)+'_two_images.png', bbox_inches='tight',format='png', dpi=1000) 
# plt.close(fig)
#%% Load 2th and chi

##♣ read cor file and get 2th and chi
filename_ = r"C:\Users\purushot\Anaconda3\envs\lauenn\Lib\site-packages\LaueTools\Examples\UO2\dat_UO2_A163_2_0028_LT_0.cor"
import LaueTools.IOLaueTools as IOLT
data_theta, data_chi = IOLT.readfile_cor(filename_)[1:3]

l_tth = data_theta * 2.
l_chi = data_chi
colu = []
for _ in range(len(l_tth)):
    colu.append("r")
###########################################################
fig = plt.figure()
plt.scatter(l_tth, l_chi, c=colu, s=15)
plt.ylabel(r'$\chi$ (in deg)',fontsize=8)
plt.xlabel(r'2$\theta$ (in deg)', fontsize=10)
plt.grid(linestyle='--', linewidth=0.5)
plt.show()
# plt.savefig('two_images.png', bbox_inches='tight',format='png', dpi=1000) 
# plt.close(fig)
###########################################################
# #%% 2D histogram ??
# ang_maxx = 120
# step = 1
# angbins = np.arange(0, ang_maxx+step, step)
# counts, xedges, yedges = np.histogram2d(l_tth, l_chi, bins=angbins, density=True)

# plt.scatter(l_tth, l_chi)
# plt.show()
# plt.imshow(counts)
# plt.show()

#%% Gnemonic image
imageGNO, nbpeaks, halfdiagonal = computeGnomonicImage(l_tth, l_chi)
plt.imshow(imageGNO)
plt.show()

#%%SKIMAGE approach
import numpy as np
from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
from skimage.feature import canny
import matplotlib.pyplot as plt
from matplotlib import cm

image = imageGNO

# Classic straight-line Hough transform
h, theta, d = hough_line(image)

# Generating figure 1
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')
ax[0].set_axis_off()

ax[1].imshow(np.log(1 + h),
             extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
             cmap=cm.gray, aspect=1/1.5)
ax[1].set_title('Hough transform')
ax[1].set_xlabel('Angles (degrees)')
ax[1].set_ylabel('Distance (pixels)')
ax[1].axis('image')

ax[2].imshow(image, cmap=cm.gray)
for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    print(angle, dist)
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
    
    # p1 = np.array((0,y0))
    # p2 = np.array((image.shape[1], y1))
    # p3 = np.array((158, 299))
    # distance = np.abs(np.cross(p2-p1, p3-p1)) / np.linalg.norm(p2-p1)
    # # print(distance)
    # # if np.isnan(distance):
    # #     ax[2].plot((0, image.shape[1]), (y0, y1), '-r', lw=0.5)
    # #     print(angle, dist, y0, y1)
    # # if distance < 50:
    # #     ax[2].plot((0, image.shape[1]), (y0, y1), '-r', lw=0.5)
    
    # i,j = 50,100
    # p3_0 = ComputeGnomon_singledata(l_tth[i], l_chi[i])
    # p3_1 = ComputeGnomon_singledata(l_tth[j], l_chi[j])

    # distance_0 = np.abs(np.cross(p2-p1, p3_0-p1)) / np.linalg.norm(p2-p1)
    # distance_1 = np.abs(np.cross(p2-p1, p3_1-p1)) / np.linalg.norm(p2-p1)
    
    # # print(distance_0, distance_1)
    # dist_ = 40
    # if distance_0 < dist_ and distance_1 < dist_:
    ax[2].plot((0, image.shape[1]), (y0, y1), '-r', lw=0.5)
    
ax[2].set_xlim((0, image.shape[1]))
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_axis_off()
ax[2].set_title('Detected lines')

plt.tight_layout()
plt.show()

# Line finding using the Probabilistic Hough Transform
edges = canny(image)
lines = probabilistic_hough_line(edges, threshold=10, line_length=100,
                                  line_gap=100)
# Generating figure 2
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')

ax[1].imshow(edges, cmap=cm.gray)
ax[1].set_title('Canny edges')

ax[2].imshow(edges)
for line in lines:
    p0, p1 = line
    ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]), lw=0.5)
ax[2].set_xlim((0, image.shape[1]))
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_title('Probabilistic Hough')

for a in ax:
    a.set_axis_off()

plt.tight_layout()
plt.show()

#%%
## get back the gnemonic coordinates from the lines image
#(216, 231)  (197, 130)

xarr, yarr = ComputeGnomon_arraydata(l_tth, l_chi)
p3=np.array((xarr,yarr))

for i in range(len(xarr)):
    pos_arr = i
    if np.all(pos_arr == np.array((216,231))) or np.all(pos_arr == np.array((231,216))):
        print("BINGO")

# distance of point to line (shortest distance)
# p1 = 
# p2 = 
# p3 = 
# np.abs(np.cross(p2-p1, p3-p1)) / norm(p2-p1))
#%% Hough transform for lines
# import numpy as np
# import math
# import matplotlib.pyplot as plt

# def rgb2gray(rgb):
#     return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

# def hough_line(img, angle_step=1, lines_are_white=True, value_threshold=5):
#     """
#     Hough transform for lines
#     Input:
#     img - 2D binary image with nonzeros representing edges
#     angle_step - Spacing between angles to use every n-th angle
#                  between -90 and 90 degrees. Default step is 1.
#     lines_are_white - boolean indicating whether lines to be detected are white
#     value_threshold - Pixel values above or below the value_threshold are edges
#     Returns:
#     accumulator - 2D array of the hough transform accumulator
#     theta - array of angles used in computation, in radians.
#     rhos - array of rho values. Max size is 2 times the diagonal
#            distance of the input image.
#     """
#     # Rho and Theta ranges
#     thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
#     width, height = img.shape
#     diag_len = int(round(math.sqrt(width * width + height * height)))
#     rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

#     # Cache some resuable values
#     cos_t = np.cos(thetas)
#     sin_t = np.sin(thetas)
#     num_thetas = len(thetas)

#     # Hough accumulator array of theta vs rho
#     accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
#     # (row, col) indexes to edges
#     are_edges = img > value_threshold if lines_are_white else img < value_threshold
#     y_idxs, x_idxs = np.nonzero(are_edges)

#     # Vote in the hough accumulator
#     for i in range(len(x_idxs)):
#         x = x_idxs[i]
#         y = y_idxs[i]

#         for t_idx in range(num_thetas):
#             # Calculate rho. diag_len is added for a positive index
#             rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
#             accumulator[rho, t_idx] += 1
#     return accumulator, thetas, rhos

# def show_hough_line(img, accumulator, thetas, rhos, save_path=None):
#     plt.imshow(accumulator, aspect='auto', cmap='jet', extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
#     if save_path is not None:
#         plt.savefig(save_path, bbox_inches='tight')
#     plt.show()


# img = imageGNO
# if img.ndim == 3:
#     img = rgb2gray(img)
# accumulator, thetas, rhos = hough_line(img, angle_step=1, value_threshold=0)
# show_hough_line(img,
#                 accumulator,
#                 thetas, rhos,
#                 save_path=None)

# #○peak finding
# idx = np.argmax(accumulator)
# rho = rhos[int(round(idx / accumulator.shape[1],0))]
# theta = thetas[int(round(idx % accumulator.shape[1],0))]
# #%% detect peak maximas from hough space
# import scipy.ndimage.filters as filters

# neighborhood_size = 20
# threshold = 2

# data_max = filters.maximum_filter(accumulator, neighborhood_size)
# maxima = (accumulator == data_max)

# data_min = filters.minimum_filter(accumulator, neighborhood_size)
# diff = ((data_max - data_min) > threshold)
# maxima[diff == 0] = 0

# labeled, num_objects = ndimage.label(maxima)
# slices = ndimage.find_objects(labeled)

# x, y = [], []
# for dy,dx in slices:
#     x_center = (dx.start + dx.stop - 1)/2
#     x.append(x_center)
#     y_center = (dy.start + dy.stop - 1)/2    
#     y.append(y_center)

# plt.imshow(accumulator,aspect='auto', cmap='jet')
# plt.autoscale(False)
# plt.plot(x,y, 'ro')
# plt.show()
# # plt.savefig('hough_space_maximas.png', bbox_inches = 'tight')
# # plt.close()

# #%%
# line_index = 1

# x_max = imageGNO.shape[0]
# y_max = imageGNO.shape[1]

# r_max = 900
# theta_max = 200

# r_dim = accumulator.shape[0]
# theta_dim = accumulator.shape[1]

# for i,j in zip(y, x):
#     r = round( (1.0 * i * r_max ) / r_dim,1)
#     theta = round( (1.0 * j * theta_max) / theta_dim,1)

#     fig, ax = plt.subplots()
#     ax.imshow(imageGNO)
#     # ax.autoscale(False)
#     px = []
#     py = []
#     for i in range(0,y_max+40,1):
#         px.append( math.cos(-theta) * i - math.sin(-theta) * r ) 
#         py.append( math.sin(-theta) * i + math.cos(-theta) * r )
#     ax.plot(px,py, linewidth=1)
#     plt.show()
#     line_index = line_index + 1

# #%%
# import cv2
# import numpy as np
# import imutils

# im = cv2.imread('../data/test1.jpg')
# gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 60, 150, apertureSize=3)

# img = im.copy()
# lines = cv2.HoughLines(edges,1,np.pi/180,200)

# for line in lines:
#     for rho,theta in line:
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a*rho
#         y0 = b*rho
#         x1 = int(x0 + 3000*(-b))
#         y1 = int(y0 + 3000*(a))
#         x2 = int(x0 - 3000*(-b))
#         y2 = int(y0 - 3000*(a))
#         cv2.line(img,(x1,y1),(x2,y2),(0,255,0),10)

# cv2.imshow('houghlines',imutils.resize(img, height=650))
# cv2.waitKey(0)
# cv2.destroyAllWindows()