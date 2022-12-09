# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 19:39:54 2022

@author: PURUSHOT

Simple script to create a mask for Laue images and resaving them with mask to avoid unrealistic peaks

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import imageio
import os, re
import glob

def create_circular_mask(h, w, center=None, radius=None, current_mask=None, value=0):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    current_mask[mask==True] = value
    return current_mask

path = r"C:\Users\purushot\Desktop\Petr_Lukes\data"

mask_file_directory = os.path.join(path, "masked_images")
if not os.path.exists(mask_file_directory):
    os.makedirs(mask_file_directory)
    
    
list_of_files = glob.glob(path+'//'+'*.tif')
## sort files
## TypeError: '<' not supported between instances of 'str' and 'int'
list_of_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

verbose = True

## read, mask and write images back
for filename in list_of_files:
    data_raw = plt.imread(filename)
    unmasked_img = np.copy(data_raw)
    ## apply mask on the image
    h, w = data_raw.shape
    masked_img = create_circular_mask(h, w, 
                                    center=[425.57, 292.8],
                                    radius=50, 
                                    current_mask=data_raw, 
                                    value=int(0.5*np.std(unmasked_img)))
    
    ## save the images
    imageio.imwrite(os.path.join(mask_file_directory, filename.split(os.sep)[-1]), masked_img)
    # plt.imsave(os.path.join(mask_file_directory, filename.split(os.sep)[-1]), masked_img.astype('uint8'))
    
    read_image = plt.imread(os.path.join(mask_file_directory, filename.split(os.sep)[-1]))
    assert np.all(read_image==masked_img), "Image array and Image written on disk do not match, verify!!"
    ## plot the images
    if verbose:
        fig, axes = plt.subplots(1, 2)
        axes[0].title.set_text("UnMasked image")
        axes[0].imshow(unmasked_img, cmap='gray', norm=LogNorm())
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[1].title.set_text("Masked image")
        axes[1].imshow(masked_img, cmap='gray', norm=LogNorm())
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        plt.show()
        
        
        
        
