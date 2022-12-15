# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 09:02:26 2022

@author: PURUSHOT

Script to define user conformed training dataset

saves the newly generated dataset into the already exisiting model (before training)
So the new data is included in the training process

Note: Supports max of 2 phase material model
"""

import numpy as np
import math
import _pickle as cPickle
import os
import matplotlib.pyplot as plt

from lauetoolsnn.utils_lauenn import _round_indices, Euler2OrientationMatrix
import lauetoolsnn.lauetools.generaltools as GT
import lauetoolsnn.lauetools.lauecore as LT
import lauetoolsnn.lauetools.CrystalParameters as CP

diameter_factor = 1

def simulatemultimatpatterns(nbUBs, key_material, seed=10, 
                             emin=5, emax=23, detectorparameters=[79.51, 977.9, 931.9, 0.36, 0.44], 
                             pixelsize=0.0734, dim1=2018, dim2=2016, kf_dir="Z>0",
                             removeharmonics=1):
    """
    Parameters
    ----------
    nbUBs : list
        list of grains per material.
    seed : int, optional
        seed for random orientation generation. The default is 10.
    key_material : list
        DESCRIPTION. The default is None.
    emin : TYPE, optional
        DESCRIPTION. The default is 5.
    emax : TYPE, optional
        DESCRIPTION. The default is 23.
    detectorparameters : list, optional
        list containing the 5 detector parameter. The default is [79.51, 977.9, 931.9, 0.36, 0.44].
    pixelsize : float
        Pixel size of the detector. The default is 0.0734 (sCMOS bin2).
    dim1 : int, optional
        Dimension of the detector. The default is 2018.
    dim2 : int, optional
        Dimension of the detector. The default is 2016.
    removeharmonics : 0 or 1, optional
        To remove or include harmonics when Laue spots are geenrated. The default is 1.

    Returns array of descriptors describing the simulated Laue spots
    -------
    s_tth : array 
    s_chi : array
    s_miller_ind : array
    s_posx : array
    s_posy : array
    s_intensity : array
    """
    l_tth, l_chi, l_miller_ind, l_posx, l_posy, l_E, l_intensity = [],[],[],[],[],[],[]
    detectordiameter = pixelsize * dim1*diameter_factor
    np.random.seed(seed)

    for no, i in enumerate(nbUBs):
        if i == 0:
            continue
        for igr in range(i):
            # =============================================================================
            # ## random Euler angles
            # =============================================================================
            # phi1 = np.random.rand() * 360.
            # phi = 180. * math.acos(2 * np.random.rand() - 1) / np.pi
            # phi2 = np.random.rand() * 360.
            # UBmatrix = Euler2OrientationMatrix((phi1, phi, phi2))
            # =============================================================================
            #     ## or define your own UB matrix here
            # =============================================================================
            UBmatrix = np.eye(3)
            
            grain = CP.Prepare_Grain(key_material[no], UBmatrix)
            s_tth, s_chi, s_miller_ind, \
                s_posx, s_posy, s_E= LT.SimulateLaue_full_np(grain, emin, emax,
                                                            detectorparameters,
                                                            kf_direction=kf_dir,
                                                            pixelsize=pixelsize,
                                                            dim=(dim1, dim2),
                                                            detectordiameter=detectordiameter,
                                                            removeharmonics=removeharmonics)
            s_miller_ind = np.c_[s_miller_ind, np.ones(len(s_miller_ind))*no]
            s_intensity = 1./s_E
            l_tth.append(s_tth)
            l_chi.append(s_chi)
            l_miller_ind.append(s_miller_ind)
            l_posx.append(s_posx)
            l_posy.append(s_posy)
            l_E.append(s_E)
            l_intensity.append(s_intensity)
    
    s_tth = np.array([item for sublist in l_tth for item in sublist])
    s_chi = np.array([item for sublist in l_chi for item in sublist])
    s_miller_ind = np.array([item for sublist in l_miller_ind for item in sublist])
    s_posx = np.array([item for sublist in l_posx for item in sublist])
    s_posy = np.array([item for sublist in l_posy for item in sublist])
    s_E = np.array([item for sublist in l_E for item in sublist])
    s_intensity = np.array([item for sublist in l_intensity for item in sublist])
    
    ### Sort by intensity the arrays
    indsort = np.argsort(s_intensity)[::-1]
    s_tth=np.take(s_tth, indsort)
    s_chi=np.take(s_chi, indsort)
    s_miller_ind=np.take(s_miller_ind, indsort, axis=0)
    s_posx=np.take(s_posx, indsort)
    s_posy=np.take(s_posy, indsort)
    s_E=np.take(s_E, indsort)
    s_intensity=np.take(s_intensity, indsort)
    return s_tth, s_chi, s_miller_ind, s_posx, s_posy, s_intensity

def getMMpatterns_(nb, material_, 
                   emin=5, emax=23, detectorparameters=[79.51, 977.9, 931.9, 0.36, 0.44], pixelsize=0.0734, 
                   dim1=2018, dim2=2016,kf_dir="Z>0",
                   ang_maxx = 45, step = 0.5, 
                   seed = 10, noisy_data=False, remove_peaks=False, user_augmentation=False, 
                   normal_hkl=None, index_hkl=None,
                   removeharmonics=1, file_index=None, 
                   save_directory_=None, plot_data=False):

    s_tth, s_chi, s_miller_ind, \
        s_posx, s_posy, s_intensity = simulatemultimatpatterns(nb, 
                                                                seed=seed, 
                                                                key_material=material_, 
                                                                emin=emin, 
                                                                emax=emax,
                                                                detectorparameters=detectorparameters,
                                                                pixelsize=pixelsize,
                                                                dim1=dim1, dim2=dim2, 
                                                                kf_dir=kf_dir,
                                                                removeharmonics=removeharmonics
                                                                 )
    # =============================================================================
    #  Data augmentation
    # =============================================================================
    if noisy_data:
        ## apply random gaussian type noise to the data (tth and chi)
        ## So adding noise to the angular distances
        ## Instead of adding noise to all HKL's ... Add to few selected HKLs
        ## Adding noise to randomly 30% of the HKLs
        ## Realistic way of introducting strains is through Pixels and not 2theta
        noisy_pixel = 0.15
        indices_noise = np.random.choice(len(s_tth), int(len(s_tth)*0.3), replace=False)
        noise_ = np.random.normal(0,noisy_pixel,len(indices_noise))
        s_tth[indices_noise] = s_tth[indices_noise] + noise_
        noise_ = np.random.normal(0,noisy_pixel,len(indices_noise)) 
        s_chi[indices_noise] = s_chi[indices_noise] + noise_
        
    if remove_peaks:
        len_mi = np.array([iq for iq in range(len(s_miller_ind))])
        len_mi = len_mi[int(0.6*len(s_miller_ind)):]
        indices_remove = np.random.choice(len_mi, int(len(len_mi)*0.5), replace=False)
        ## delete randomly selected less intense peaks
        ## to simulate real peak detection, where some peaks may not be
        ## well detected
        ## Include maybe Intensity approach: Delete peaks based on their SF and position in detector
        if len(indices_remove) !=0:
            s_tth = np.delete(s_tth, indices_remove)
            s_chi = np.delete(s_chi, indices_remove)
            s_miller_ind = np.delete(s_miller_ind, indices_remove, axis=0)
    
    if user_augmentation:
        # nb_random_spots = 500
        ## add random two theta and chi spots to the dataset
        # to simulate noise in the patterns (in reality these 
        # are additional peaks from partial Laue patterns).
        # we can do 2D sampling of 2theta and chi from one Cor file;
        # but apparantly the chi is uniform from (-40 to +40)
        # while 2theta has a distribution
        pass
        
    # =============================================================================
    #  Classify the spots 
    # =============================================================================        
    # replace all hkl class with relevant hkls
    location = []
    skip_hkl = []
    for j, i in enumerate(s_miller_ind):
        if np.all(i[:3] == 0):
            skip_hkl.append(j)
            continue
        new_hkl = _round_indices(i[:3])
        mat_index = int(i[3])
        temp_ = np.all(new_hkl == normal_hkl[mat_index], axis=1)
        if len(np.where(temp_)[0]) == 1:
            ind_ = np.where(temp_)[0][0]
            location.append(index_hkl[mat_index][ind_])
        elif len(np.where(temp_)[0]) == 0:
            skip_hkl.append(j)
        elif len(np.where(temp_)[0]) > 1:
            ## first check if they both are same class or not
            class_output = []
            for ij in range(len(np.where(temp_)[0])):
                indc = index_hkl[mat_index][np.where(temp_)[0][ij]]
                class_output.append(indc)
            if len(set(class_output)) <= 1:
                location.append(class_output[0])
            else:
                skip_hkl.append(j)
                print(i)
                print(np.where(temp_)[0])
                for ij in range(len(np.where(temp_)[0])):
                    indc = index_hkl[mat_index][np.where(temp_)[0][ij]]
                    # print(classhkl[mat_index][indc])
                print("Skipping HKL as something is not proper with equivalent HKL module")
                print("This should never happen, if it did, then the implementation is wrong, duplicate output classes!!")

    allspots_the_chi = np.transpose(np.array([s_tth/2., s_chi]))
    tabledistancerandom = np.transpose(GT.calculdist_from_thetachi(allspots_the_chi, allspots_the_chi))
    # =============================================================================
    #  Generate histograms
    # =============================================================================    
    codebars = []
    angbins = np.arange(0,ang_maxx+step,step)
    for i in range(len(tabledistancerandom)):
        if i in skip_hkl: ## not saving skipped HKL
            continue
        angles = tabledistancerandom[i]
        spots_delete = [i] ## remove self bin from the histogram
        angles = np.delete(angles, spots_delete)
        fingerprint = np.histogram(angles, bins=angbins)[0]
        ## same normalization as old training dataset
        max_codebars = np.max(fingerprint)
        fingerprint = fingerprint/ max_codebars
        codebars.append(fingerprint)

    # =============================================================================
    # Plot the data
    # =============================================================================
    if plot_data:
        ## plot image data
        r = 5
        Xpix = s_posx
        Ypix = s_posy
        img_array = np.zeros((dim1,dim2), dtype=np.uint8)
        for i in range(len(Xpix)):
            X_pix = int(Xpix[i])
            Y_pix = int(Ypix[i])
            img_array[X_pix-r:X_pix+r, Y_pix-r:Y_pix+r] = 1
        
        fig, axes = plt.subplots(1, 2)
        axes[0].title.set_text("Pixel space")
        axes[0].imshow(img_array, cmap='gray')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        
        axes[1].title.set_text("Angular space")
        axes[1].scatter(s_tth, s_chi, s=5)
        axes[1].set_ylabel(r'$\chi$ (in deg)',fontsize=8)
        axes[1].set_xlabel(r'2$\theta$ (in deg)', fontsize=10)
        plt.show()
        
        ## plot fingerprints
        spot_plot = 0
        angles = tabledistancerandom[spot_plot]
        spots_delete = [spot_plot] ## remove self bin from the histogram
        angles = np.delete(angles, spots_delete)
        fingerprint = np.histogram(angles, bins=angbins)[0]
        print("Total spots in bins:", np.sum(fingerprint))
        fig, ax = plt.subplots(1,2)
        ax[0].title.set_text("Angular distribution of ("+str(int(s_miller_ind[spot_plot][0]))+" "+str(int(s_miller_ind[spot_plot][1]))+" "+str(int(s_miller_ind[spot_plot][2]))+")")
        ax[0].hist(angles, bins=angbins, density=False)
        ax[0].set_ylabel(r'Density',fontsize=8)
        ax[0].set_xlabel(r'Angular distance bins', fontsize=10)
        
        list_ = np.where(tabledistancerandom[spot_plot,:] > ang_maxx)[0]
        print(list_)
        ind_del = np.where(list_==spot_plot)[0]
        print(ind_del)
        list_ = np.delete(list_, ind_del, axis=0)
        
        
        ax[1].title.set_text("Spot ("+str(int(s_miller_ind[spot_plot][0]))+" "+str(int(s_miller_ind[spot_plot][1]))+" "+str(int(s_miller_ind[spot_plot][2]))+")")
        ax[1].scatter(s_tth, s_chi, c='k')
        ax[1].scatter(s_tth[spot_plot], s_chi[spot_plot], c='r')
        ax[1].scatter(s_tth[list_], s_chi[list_], c='b')
        ax[1].scatter(s_tth[list_], s_chi[list_], c='b')
        segments = []
        for i in range(len(s_tth)):
            if i in list_ or i == spot_plot:
                continue
            segments.append(1)
            ax[1].plot( [s_tth[spot_plot],s_tth[i]], [s_chi[spot_plot], s_chi[i]], c='r', ls="solid", lw=0.3)
        print("Total connections:", len(segments))
        ax[1].set_ylabel(r'$\chi$ (in deg)',fontsize=8)
        ax[1].set_xlabel(r'2$\theta$ (in deg)', fontsize=10)
        plt.show()
    
    # =============================================================================
    #  Save the file for training
    # =============================================================================
    suffix_ = "_user_defined"
    if len(codebars) != 0:
        mat_prefix = ""
        for no, i in enumerate(nb):
            if i != 0:
                mat_prefix = mat_prefix + material_[no]

        np.savez_compressed(save_directory_+'//'+mat_prefix+'_grain_'+str(file_index)+"_"+\
                            suffix_+"_nb"+"".join(str(param) for param in nb)+'.npz', \
                            codebars, location, [], [], 0, s_tth, s_chi, s_miller_ind)
    else:
        print("Skipping a simulation file: "+save_directory_+'//grain_'+\
                            str(file_index)+"_"+suffix_+'.npz'+"; Due to no data conforming user settings")
            
#%%            
############## Lets create some additional data

# =============================================================================
# ## user defined detector parameters
# =============================================================================
detectorparameters=[79.51, 977.9, 931.9, 0.36, 0.44]
pixelsize=0.0734
dim1=2018
dim2=2016
emin=5
emax=23
kf_dir="Z>0"
# =============================================================================
# Material and simulation definition
# =============================================================================
material_ = ["GaN", "Si"]
nb_grains = [1, 1]
# =============================================================================
# Training histogram generation setting
# this has to be the same as the original training dataset, other shape mismatch error
# =============================================================================
maximum_angle_to_search = 90
step_for_binning = 0.1

model_directory_ = r"C:\Users\purushot\Anaconda3\envs\py310\Lib\site-packages\lauetoolsnn\models\GaN_Si"

training_data_directory = os.path.join(model_directory_, "training_data")
# =============================================================================
# Extract useful objects from saved files in the model directory
# =============================================================================
hkl_all_class = []
for imat in material_:
    with open(model_directory_+"//classhkl_data_nonpickled_"+imat+".pickle", "rb") as input_file:
        hkl_all_class_mat = cPickle.load(input_file)[0]
        hkl_all_class.append(hkl_all_class_mat)
     
## make comprehensive list of dictionary
normal_hkl = []
index_hkl = []
for ino, imat in enumerate(material_):
    normal_hkl_ = np.zeros((1,3))
    for j in hkl_all_class[ino].keys():
        normal_hkl_ = np.vstack((normal_hkl_, hkl_all_class[ino][j]))
    normal_hkl1_ = np.delete(normal_hkl_, 0, axis =0)
    normal_hkl.append(normal_hkl1_)
    if ino > 0:
        ind_offset = index_hkl[ino-1][-1] + 1
        index_hkl_ = [ind_offset+j for j,k in enumerate(hkl_all_class[ino].keys()) for i in range(len(hkl_all_class[ino][k]))]
    else:
        index_hkl_ = [j for j,k in enumerate(hkl_all_class[ino].keys()) for i in range(len(hkl_all_class[ino][k]))]
    index_hkl.append(index_hkl_)

file_identifier = 50 

getMMpatterns_(nb_grains, material_, 
                emin, emax, detectorparameters, pixelsize, dim1, dim2, kf_dir,
                maximum_angle_to_search, step_for_binning, 
                seed = 100,#np.random.randint(1e6), 
                noisy_data=True, remove_peaks=True, user_augmentation=True,
                normal_hkl=normal_hkl, index_hkl=index_hkl,
                removeharmonics=1, file_index=file_identifier, 
                save_directory_=training_data_directory, plot_data=True)















