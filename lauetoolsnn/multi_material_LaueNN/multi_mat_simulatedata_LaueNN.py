#!/usr/bin/env python
# coding: utf-8

# # Notebook script for generating simulated laue patterns to be used to verify prediction of Laue hkl (step 3) using the Trained model from step 2
# 
# ### Define material of interest for which the simulated data is generated (angular coordinates data is generated based on the defined detector geometry);
# ### Simulate Laue patterns of required complexity

# In[1]:
if __name__ == '__main__':     #enclosing required because of multiprocessing


    ## Import modules used for this Notebook
    import os
    import numpy as np
    import random
    from random import random as rand1
    from math import acos
    from tqdm import trange
    
    ## if LaueToolsNN is properly installed
    try:
        from lauetoolsnn.utils_lauenn import get_multimaterial_detail, Euler2OrientationMatrix
        from lauetoolsnn.lauetools import dict_LaueTools as dictLT
        from lauetoolsnn.lauetools import IOLaueTools as IOLT
        from lauetoolsnn.lauetools import lauecore as LT
        from lauetoolsnn.lauetools import CrystalParameters as CP
        from lauetoolsnn.lauetools import generaltools as GT
    except:
        # else import from a path where LaueToolsNN files are
        import sys
        sys.path.append(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn")
        from utils_lauenn import get_multimaterial_detail, Euler2OrientationMatrix
        sys.path.append(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\lauetools")
        import dict_LaueTools as dictLT
        import IOLaueTools as IOLT
        import lauecore as LT
        import CrystalParameters as CP
        import generaltools as GT
    
    
    # ## step 1: define material and other parameters
    
    # In[2]:
    # =============================================================================
    ## User Input dictionary with parameters
    ## In case of only one phase/material, keep same value for material_ and material1_ key
    # =============================================================================
    input_params = {
                    # =============================================================================
                    #       GENERATION OF DATASET              
                    # =============================================================================
                    "material_": ["Zr_alpha",
                                  "ZrO2_mono",
                                  "ZrO2_tet",
                                  "ZrO2_cub",
                                  "Zr_Nb_Fe",
                                  ],             ## same key as used in dict_LaueTools
                    "prefix" : "",                 ## prefix for the folder to be created for training dataset
                    "symmetry": ["hexagonal",
                                 "monoclinic",
                                 "tetragonal",
                                 "cubic",
                                 "hexagonal",
                                 ],           ## crystal symmetry of material_
                    "SG": [194,
                           14,
                           137,
                           225,
                           194,
                           ],                     ## Space group of material_ (None if not known)
                    # =============================================================================
                    #        Detector parameters (roughly) of the Experimental setup
                    # =============================================================================
                    ## Sample-detector distance, X center, Y center, two detector angles
                    "detectorparameters" :  [79.553,979.32,932.31,0.37,0.447], 
                    "pixelsize" : 0.0734,          ## Detector pixel size
                    "dim1":2018,                   ## Dimensions of detector in pixels
                    "dim2":2016,
                    "emin" : 5,                    ## Minimum and maximum energy to use for simulating Laue Patterns
                    "emax" : 22,
                    # =============================================================================
                    #       simulation parameters             
                    # =============================================================================
                    "experimental_directory": "",
                    "experimental_prefix": "",
                    "use_simulated_dataset": True,  ## Use simulated dataset (generated at step 3a) incase no experimental data to verify the trained model
                    "grid_size_x" : 25,            ## Grid X and Y limit to generate the simulated dataset (a rectangular scan region)
                    "grid_size_y" : 25,  
                    "grains_max" :[1,
                                   1,
                                   1,
                                   1,
                                   1,
                                   ], ## Maximum number of grains to simulate (randomly generate between 1 and grains_max parameters)
                    }
    
    # ## Step 2: Get material parameters 
    
    # In[3]:
    material_= input_params["material_"]
    detectorparameters = input_params["detectorparameters"]
    pixelsize = input_params["pixelsize"]
    emax = input_params["emax"]
    emin = input_params["emin"]
    dim1 = input_params["dim1"]
    dim2 = input_params["dim2"]
    symm_ = input_params["symmetry"]
    SG = input_params["SG"]
    grains_sim = input_params["grains_max"]
    grid = input_params["grid_size_x"]*input_params["grid_size_y"] 
    
    if len(material_) > 1:
        prefix_mat = material_[0]
        for ino, imat in enumerate(material_):
            if ino == 0:
                continue
            prefix_mat = prefix_mat + "_" + imat
    else:
        prefix_mat = material_
    
    save_directory = os.getcwd()+"//"+prefix_mat+input_params["prefix"]

    save_directory_sim_data = save_directory + "//simulated_dataset"
    
    print("save directory is : "+save_directory)
    print("Simulated data save directory is : "+save_directory_sim_data)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    if not os.path.exists(save_directory_sim_data):
        os.makedirs(save_directory_sim_data)
    
    ## get unit cell parameters and other details required for simulating Laue patterns
    rules, symmetry, lattice_material, \
        crystal, SG = get_multimaterial_detail(material_, SG, symm_)

    
    # ## Step 3: Generate Laue patterns 
    
    # In[4]:
    
    text_file = open(save_directory_sim_data+"//filecreation_stats_"+prefix_mat+"_v2.txt", "w")
    
    detector_label = "sCMOS" ## by default (no need to modify; used later to get detector bounds)
    for ii in trange(grid):
        
        #time.sleep(1) ## 1second pause to replicate Experiment time in case 
                       ## we want to do live prediction while data is being written
        
        grains_to_sim = [random.randint(0,igrain) for igrain in grains_sim]
        while True:
            if np.all(np.array(grains_to_sim) == 0):
                grains_to_sim = [random.randint(0,igrain) for igrain in grains_sim]
            else:
                break
                
        l_tth, l_chi, l_miller_ind, l_posx, l_posy, l_E, l_intensity = [],[],[],[],[],[],[]
        detectordiameter = pixelsize * dim1
        prefix_cor_header = []
        nbgrains = []
        g = []
        for no, i in enumerate(grains_to_sim):
            if i != 0:
                nbgrains.append(i)
                for igr in range(i):
                    prefix_cor_header.append(material_[no])
                    phi1 = rand1() * 360.
                    phi = 180. * acos(2 * rand1() - 1) / np.pi
                    phi2 = rand1() * 360.
                    UBmatrix = Euler2OrientationMatrix((phi1, phi, phi2))
                    g.append(UBmatrix)
                    
                    grain = CP.Prepare_Grain(material_[no], UBmatrix)
                    s_tth, s_chi, s_miller_ind, \
                        s_posx, s_posy, s_E= LT.SimulateLaue_full_np(grain, emin, emax,
                                                                    detectorparameters,
                                                                    pixelsize=pixelsize,
                                                                    dim=(dim1, dim2),
                                                                    detectordiameter=detectordiameter,
                                                                    removeharmonics=1)
                    s_miller_ind = np.c_[s_miller_ind, np.ones(len(s_miller_ind))*no]
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
        
        #sortintensity
        indsort = np.argsort(s_intensity)[::-1]
        s_tth=np.take(s_tth, indsort)
        s_chi=np.take(s_chi, indsort)
        s_miller_ind=np.take(s_miller_ind, indsort, axis=0)
        s_posx=np.take(s_posx, indsort)
        s_posy=np.take(s_posy, indsort)
        s_E=np.take(s_E, indsort)
        s_intensity=np.take(s_intensity, indsort)
        
        # considering all spots
        allspots_the_chi = np.transpose(np.array([s_tth/2., s_chi]))
        tabledistancerandom = np.transpose(GT.calculdist_from_thetachi(allspots_the_chi, allspots_the_chi))
        # ground truth
        hkl_sol = s_miller_ind
        
        framedim = dictLT.dict_CCD[detector_label][0]
        dict_dp={}
        dict_dp['kf_direction']='Z>0'
        dict_dp['detectorparameters']=detectorparameters
        dict_dp['detectordistance']=detectorparameters[0]
        dict_dp['detectordiameter']=pixelsize*dim1
        dict_dp['pixelsize']=pixelsize
        dict_dp['dim']=framedim
        dict_detector = detectorparameters
        CCDcalib = {"CCDLabel":"cor",
                    "dd":dict_detector[0], 
                    "xcen":dict_detector[1], 
                    "ycen":dict_detector[2], 
                    "xbet":dict_detector[3], 
                    "xgam":dict_detector[4],
                    "pixelsize": pixelsize}

        IOLT.writefile_cor(save_directory_sim_data+"//"+prefix_mat+"_"+str(ii), s_tth, s_chi, s_posx, s_posy, s_intensity,
                           param=CCDcalib, sortedexit=0)    
        
        text_file.write("####### File : "+save_directory_sim_data+"//"+prefix_mat+"_"+str(ii) + ".cor generated \n")
        for ino, rm in enumerate(g):
            if np.all(rm == 0):
                continue
            text_file.write("# Phase "+prefix_cor_header[ino]+":  "+str(nbgrains[ino]) + " grains \n")
            temp_ = rm.flatten()
            string1 = "[["+str(temp_[0])+","+str(temp_[1])+","+str(temp_[2])+"],"+                      "["+str(temp_[3])+","+str(temp_[4])+","+str(temp_[5])+"],"+                          "["+str(temp_[6])+","+str(temp_[7])+","+str(temp_[8])+"]]"+ " \n"  
            text_file.write(string1)
        text_file.write("# ********** \n \n")
    text_file.close()
    
    
    # ## Generate a config file for the simulated dataset to be used with GUI
    
    # In[6]:
    
    ## make calib text file
    calib_file = save_directory_sim_data+"//calib.det"
    text_file = open(calib_file, "w")
    
    text_file.write("79.553, 979.32, 932.31, 0.37, 0.447, 0.07340000, 2018, 2016 \n")
    text_file.write("Sample-Detector distance(IM), xO, yO, angle1, angle2, pixelsize, dim1, dim2 \n")
    text_file.write("Calibration done with Ge at Wed Sep 22 14:31:38 2021 with LaueToolsGUI.py \n")
    text_file.write("Experimental Data file: G:\\bm32\\SH1\\Ge_0001_LT_2.dat \n")
    text_file.write("Orientation Matrix: \n")
    text_file.write("[[0.0969250,0.6153840,-0.7822455],[-0.9616391,0.2605486,0.0858177],[0.2566238,0.7439200,0.6170310]] \n")
    text_file.write("# Material : Ge \n")
    text_file.write("# dd : 79.553 \n")
    text_file.write("# xcen : 979.32 \n")
    text_file.write("# ycen : 932.31 \n")
    text_file.write("# xbet : 0.37 \n")
    text_file.write("# xgam : 0.447 \n")
    text_file.write("# pixelsize : 0.0734 \n")
    text_file.write("# xpixelsize : 0.0734 \n")
    text_file.write("# ypixelsize : 0.0734 \n")
    text_file.write("# CCDLabel : cor \n")
    text_file.write("# framedim : (2018, 2016) \n")
    text_file.write("# detectordiameter : 162.93332000000004 \n")
    text_file.write("# kf_direction : Z>0")
    text_file.close()
    
 
# In[ ]:




