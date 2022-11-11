# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:15:46 2022

@author: PURUSHOT

Compress a Laue Image with Fourier Transform

To validate a compression --> do peak search on original and compressed image
"""
# =============================================================================
# Compression
# =============================================================================
# In this section, we produce 6 different compressed versions of an image. 
# Compression is done by thresholding the coefficients’ magnitude and take 
# only the largest percentile of them. Therefore, we assign zero to any 
# point with fourier coefficient value smaller than the calculated threshold.
#  We also notice that the percentage reduction in the size of the compressed 
#  files is linearly correlated to the percentage of the points we zero out. 
#  At 30%, we notice the image is reconstructed with barely no quality loss. 
#  At 50%, we notice a very slight loss in the details. At 95%, even though 
#  most of the points were zeroed out, 
# however the image still keeps most of its details, but we can notice the loss in quality.

import matplotlib.pyplot as plt
from scipy import sparse
import os
import numpy as np

img = plt.imread(r"D:\some_projects\GaN\BLC12834\NW1\nw1_0000.tif")

transformed_original = np.fft.fft2(img)
data_csr = sparse.csr_matrix(transformed_original)
sparse.save_npz("original.npz", data_csr)
size = os.path.getsize("original.npz")/1e6


plt.figure(figsize=(15,5))
plt.subplot(231)
plt.imshow(img, cmap="gray", vmin=1000, vmax=2000)
plt.title("Original, size={} Mb".format(size)), plt.xticks([]), plt.yticks([])
compression_factors= [30, 50, 80, 90, 95]
index_count=2
for factor in compression_factors:
    transformed = transformed_original.copy()
    thresh = np.percentile(abs(transformed), factor)
    transformed[abs(transformed) < thresh] = 0
    data_csr = sparse.csr_matrix(transformed)
    fileName = "@{}%.npz".format(factor)
    sparse.save_npz(fileName, data_csr)
    size = os.path.getsize(fileName)/1e6
    back =  np.fft.ifft2(transformed).real
    plt.subplot(2,3,index_count), plt.imshow(back, cmap="gray", vmin=1000, vmax=2000)
    plt.title("@ {}%, size={} Mb".format(factor, size)), plt.xticks([]), plt.yticks([])
    index_count=index_count+1
plt.suptitle("Compression Levels",fontsize=22)
plt.show()


#%%
# The numpy packages provides a Fast Fourier Transform in 2D,
# and its inverse (the iFFT). FFTshift and iFFTshift
# are just there to get nicer, centered plots:
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from imageio.v3 import imread       # Load .png and .jpg images

def display_2(im_1, title_1, im_2, title_2, vmin=None, vmax=None):
    """
    Displays two images side by side; typically, an image and its Fourier transform.
    """
    plt.figure(figsize=(12,6))                    # Rectangular blackboard
    plt.subplot(1,2,1) ; plt.title(title_1)       # 1x2 waffle plot, 1st cell
    if vmin!=None and vmax!=None:
        plt.imshow(im_1, cmap="gray", vmin=vmin, vmax=vmax) # Auto-equalization
    else:
        plt.imshow(im_1, cmap="gray")
    plt.subplot(1,2,2) ; plt.title(title_2)       # 1x2 waffle plot, 2nd cell
    plt.imshow(im_2, cmap="gray", vmin=-7, vmax=15)  


def Fourier_bandpass(fI, fmin, fmax, vmin=None, vmax=None, save=False) :
    """
    Truncates a Fourier Transform fI, before reconstructing a bandpassed image.
    """
    Y, X = np.mgrid[:fI.shape[0], :fI.shape[1]]  # Horizontal and vertical gradients
    radius = (X - fI.shape[0]/2) ** 2 + (Y - fI.shape[1]/2) ** 2 # Squared distance to the middle point
    radius = ifftshift( np.sqrt(radius) )    # Reshape to be fft-compatible
    fI_band = fI.copy()               # Create a copy of the Fourier transform
    fI_band[ radius <=fmin ] = 0      # Remove all the low frequencies
    fI_band[ radius > fmax ] = 0      # Remove all the high frequencies
    I_band = np.real(ifft2(fI_band))  # Invert the new transform...
    display_2(I_band, "Image", fftshift( np.log(1e-7 + abs(fI_band)) ), "Fourier Transform", vmin=vmin, vmax=vmax )
    if save:
        np.save_npz(str(fmin)+"_"+str(fmax)+".npz", fI_band)


I = imread(r"D:\some_projects\GaN\BLC12834\NW1\nw1_0000.tif")  # Import as a grayscale array

fI = fft2(I)  # Compute the Fourier transform of our Laue image

# Display the logarithm of the amplitutde of Fourier coefficients.
# The "fftshift" routine allows us to put the zero frequency in
# the middle of the spectrum, thus centering the right plot as expected.
display_2( I, "Image", fftshift( np.log(1e-7 + abs(fI)) ), "Fourier Transform" , vmin=1000, vmax=2000)


Fourier_bandpass(fI, 0, 300, vmin=1000, vmax=2000, save=False)









