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
    from tqdm import trange
    
    ## if LaueToolsNN is properly installed
    try:
        from lauetoolsnn.utils_lauenn import get_material_detail,prepare_LP_NB
        from lauetoolsnn.lauetools import dict_LaueTools as dictLT
        from lauetoolsnn.lauetools import IOLaueTools as IOLT
    except:
        # else import from a path where LaueToolsNN files are
        import sys
        sys.path.append(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn")
        from utils_lauenn import get_material_detail,prepare_LP_NB
        sys.path.append(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\lauetools")
        import dict_LaueTools as dictLT
        import IOLaueTools as IOLT
    
    
    # ## step 1: define material and other parameters
    
    # In[2]:
    
    
    # =============================================================================
    ## User Input dictionary with parameters
    ## In case of only one phase/material, keep same value for material_ and material1_ key
    # =============================================================================
    input_params = {
                    "material_": "Cu",             ## same key as used in dict_LaueTools
                    "material1_": "Cu",            ## same key as used in dict_LaueTools
                    "prefix" : "",                 ## prefix for the folder to be created for training dataset
                    "symmetry": "cubic",           ## crystal symmetry of material_
                    "symmetry1": "cubic",          ## crystal symmetry of material1_
                    "SG": 225,                     ## Space group of material_ (None if not known)
                    "SG1": 225,                    ## Space group of material1_ (None if not known)
                    ## Detector parameters (roughly) of the Experimental setup
                    ## Sample-detector distance, X center, Y center, two detector angles
                    "detectorparameters" :  [79.553,979.32,932.31,0.37,0.447], 
                    "pixelsize" : 0.0734,          ## Detector pixel size
                    "dim1":2018,                   ## Dimensions of detector in pixels
                    "dim2":2016,
                    "emin" : 5,                    ## Minimum and maximum energy to use for simulating Laue Patterns
                    "emax" : 22,
                    "grains_max" : 5,              ## Maximum number of grains to simulate (randomly generate between 1 and grains_max parameters)
                    "grid_size_x" : 5,            ## Grid X and Y limit to generate the simulated dataset (a rectangular scan region)
                    "grid_size_y" : 5,
                    }
    generate_config_file_GUI = True
    
    
    # ## Step 2: Get material parameters 
    
    # In[3]:
    
    
    material_= input_params["material_"]
    material1_= input_params["material1_"]
    detectorparameters = input_params["detectorparameters"]
    pixelsize = input_params["pixelsize"]
    emax = input_params["emax"]
    emin = input_params["emin"]
    dim1 = input_params["dim1"]
    dim2 = input_params["dim2"]
    symm_ = input_params["symmetry"]
    symm1_ = input_params["symmetry1"]
    SG = input_params["SG"]
    SG1 = input_params["SG1"]
    grains_sim = input_params["grains_max"]
    grid = input_params["grid_size_x"]*input_params["grid_size_y"] 
    
    if material_ != material1_:
        save_directory = os.getcwd()+"//"+material_+"_"+material1_+input_params["prefix"]
    else:
        save_directory = os.getcwd()+"//"+material_+input_params["prefix"]
        
    save_directory_sim_data = save_directory + "//simulated_dataset"
    
    print("save directory is : "+save_directory)
    print("Simulated data save directory is : "+save_directory_sim_data)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    if not os.path.exists(save_directory_sim_data):
        os.makedirs(save_directory_sim_data)
        
    ## get unit cell parameters and other details required for simulating Laue patterns
    rules, symmetry, lattice_material,     crystal, SG, rules1, symmetry1,    lattice_material1, crystal1, SG1 = get_material_detail(material_, SG, symm_,
                                                               material1_, SG1, symm1_)
    
    
    # ## Step 3: Generate Laue patterns 
    
    # In[4]:
    
    
    if material_ != material1_:
        prefix1 = material_+"_"+material1_
    else:
        prefix1 = material_
        
    if material_ != material1_:
        text_file = open(save_directory_sim_data+"//filecreation_stats_"+material_+"_"+material1_+"_v2.txt", "w")
    else:
        text_file = open(save_directory_sim_data+"//filecreation_stats_"+material_+"_v2.txt", "w")
    
    detector_label = "sCMOS" ## by default (no need to modify; used later to get detector bounds)
    for ii in trange(grid):
        #time.sleep(0.5) ## 1second pause to replicate Experiment time in case we want to do live prediction while data is being written
        noisy_data = False #bool(random.getrandbits(1)) 
        remove_peaks = False #bool(random.getrandbits(1)) 
        
        if grains_sim != 1:
            nbgrains = np.random.randint(1,high=grains_sim) ## material0
        else:
            nbgrains = np.random.randint(1,high=grains_sim+1) ## material0
        nbgrains1 = np.random.randint(0,high=grains_sim) ## material1
        if material_ == material1_:
            nbgrains1 = 0 ## material1
            
        verbose = 0
        
        #print("Progress %i/%i ; Generating %i grains in a single Laue Pattern" %(ii, grid, nbgrains+nbgrains1))
        seednumber = np.random.randint(1e6)
        tabledistancerandom, hkl_sol,\
        s_posx, s_posy, s_I, s_tth, s_chi, g, g1  = prepare_LP_NB(nbgrains, nbgrains1,
                                                                        material_, verbose,
                                                                        material1_ = material1_,
                                                                        seed = seednumber,sortintensity=True,
                                                                        detectorparameters=detectorparameters, 
                                                                        pixelsize=pixelsize,
                                                                        dim1=dim1, dim2=dim2, 
                                                                        emin=emin, emax=emax,
                                                                        flag = 10, noisy_data=noisy_data,
                                                                        remove_peaks = remove_peaks) 
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

        IOLT.writefile_cor(save_directory_sim_data+"//"+prefix1+"_"+str(ii), s_tth, s_chi, s_posx, s_posy, s_I,
                           param=CCDcalib, sortedexit=0)    
        
        text_file.write("# File : "+save_directory_sim_data+"//"+prefix1+"_"+str(ii) + ".cor generated \n")
        text_file.write("# Phase "+material_+":  "+str(nbgrains) + " grains \n")    
        for rm in g:
            if np.all(rm == 0):
                continue
            temp_ = rm.flatten()
            string1 = "[["+str(temp_[0])+","+str(temp_[1])+","+str(temp_[2])+"],"+                      "["+str(temp_[3])+","+str(temp_[4])+","+str(temp_[5])+"],"+                          "["+str(temp_[6])+","+str(temp_[7])+","+str(temp_[8])+"]]"+ " \n"  
            text_file.write(string1)
        
        if material_ != material1_:
            text_file.write("# Phase "+material1_+":  "+str(nbgrains1) + " grains \n")
            for rm in g1:
                if np.all(rm == 0):
                    continue
                temp_ = rm.flatten()
                string1 = "[["+str(temp_[0])+","+str(temp_[1])+","+str(temp_[2])+"],"+                          "["+str(temp_[3])+","+str(temp_[4])+","+str(temp_[5])+"],"+                              "["+str(temp_[6])+","+str(temp_[7])+","+str(temp_[8])+"]]"+ " \n"  
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
    
    ## write config file for GUI 
    
    save_directory_sim_data_config = save_directory_sim_data + "//config"
    
    print("config file directory is : "+save_directory_sim_data_config)
    if not os.path.exists(save_directory_sim_data_config):
        os.makedirs(save_directory_sim_data_config)
        
    if material_ != material1_:
        text_file = open(save_directory_sim_data_config+"//config_"+material_+"_"+material1_+".txt", "w")
    else:
        text_file = open(save_directory_sim_data_config+"//config_"+material_+".txt", "w")
    
    text_file.write("### config file for LaueNeuralNetwork \n")
    text_file.write("[CPU]\n")
    text_file.write("n_cpu = 8\n")
    text_file.write("\n")
    text_file.write("[GLOBAL_DIRECTORY]\n")
    text_file.write("prefix = "+input_params["prefix"]+" \n")
    text_file.write("## directory where all training related data and results will be saved \n")
    text_file.write("main_directory = "+os.getcwd()+"\n")
    text_file.write("\n")
    text_file.write("[MATERIAL]\n")
    text_file.write("## same material key as lauetools (see dictlauetools.py for complete key)\n")
    text_file.write("## as of now symmetry can be cubic, hexagonal, orthorhombic, tetragonal, trigonal, monoclinic, triclinic\n")
    text_file.write("\n")
    text_file.write("material = "+material_+"\n")
    text_file.write("symmetry = "+symm_+"\n")
    text_file.write("space_group = "+str(SG)+"\n")
    text_file.write("general_diffraction_rules = false\n")
    text_file.write("\n")
    text_file.write("## if second phase is present, else none\n")
    text_file.write("material1 = "+material1_+"\n")
    text_file.write("symmetry1 = "+symm1_+"\n")
    text_file.write("space_group1 = "+str(SG1)+"\n")
    text_file.write("general_diffraction_rules1 = false\n")
    text_file.write("\n")
    text_file.write("[DETECTOR]\n")
    text_file.write("## path to detector calibration file (.det)\n")
    text_file.write("detectorfile = "+calib_file+" \n")
    text_file.write("## Max and Min energy to be used for generating training dataset, as well as for calcualting matching rate\n")
    text_file.write("emax = 22\n")
    text_file.write("emin = 5\n")
    text_file.write("\n")
    text_file.write("[TRAINING]\n")
    text_file.write("## classes_with_frequency_to_remove: HKL class with less appearance than specified will be ignored in output\n")
    text_file.write("## desired_classes_output : can be all or an integer: to limit the number of output classes\n")
    text_file.write("## max_HKL_index : can be auto or integer: Maximum index of HKL to build output classes\n")
    text_file.write("## max_nb_grains : Maximum number of grains to simulate per lauepattern\n")
    text_file.write("####### Material 0\n")
    text_file.write("classes_with_frequency_to_remove = 100\n")
    text_file.write("desired_classes_output = all\n")
    text_file.write("max_HKL_index = 5\n")
    text_file.write("max_nb_grains = 1\n")
    text_file.write("####### Material 1\n")
    text_file.write("## HKL class with less appearance than specified will be ignored in output\n")
    text_file.write("classes_with_frequency_to_remove1 = 100\n")
    text_file.write("desired_classes_output1 = all\n")
    text_file.write("max_HKL_index1 = 5\n")
    text_file.write("max_nb_grains1 = 1\n")
    text_file.write("\n")
    text_file.write("max_simulations = 500\n")
    text_file.write("include_small_misorientation = false\n")
    text_file.write("angular_distance = 120\n")
    text_file.write("step_size = 0.1\n")
    text_file.write("batch_size = 50\n")
    text_file.write("epochs = 5\n")
    text_file.write("\n")
    text_file.write("[PREDICTION]\n")
    text_file.write("UB_matrix_to_detect = 2\n")
    text_file.write("\n")
    text_file.write("matrix_tolerance = 0.5\n")
    text_file.write("matrix_tolerance1 = 0.5\n")
    text_file.write("\n")
    text_file.write("material0_limit = 1000\n")
    text_file.write("material1_limit = 1000\n")
    text_file.write("\n")
    text_file.write("model_weight_file = none\n")
    text_file.write("softmax_threshold_global = 0.90\n")
    text_file.write("mr_threshold_global = 1.00\n")
    text_file.write("cap_matchrate = 0.01\n")
    text_file.write("coeff = 0.3\n")
    text_file.write("coeff_overlap = 0.05\n")
    text_file.write("mode_spotCycle = graphmode\n")
    text_file.write("use_previous = false\n")
    text_file.write("\n")
    text_file.write("[EXPERIMENT]\n")
    text_file.write("experiment_directory = "+save_directory_sim_data+"\n")
    text_file.write("experiment_file_prefix = "+prefix1+"_\n")
    text_file.write("image_grid_x = "+str(input_params["grid_size_x"])+"\n")
    text_file.write("image_grid_y = "+str(input_params["grid_size_y"])+"\n")
    text_file.write("\n")
    text_file.write("[PEAKSEARCH]\n")
    text_file.write("intensity_threshold = 90\n")
    text_file.write("boxsize = 15\n")
    text_file.write("fit_peaks_gaussian = 1\n")
    text_file.write("FitPixelDev = 15\n")
    text_file.write("NumberMaxofFits = 3000\n")
    text_file.write("\n")
    text_file.write("[STRAINCALCULATION]\n")
    text_file.write("strain_compute = true\n")
    text_file.write("tolerance_strain_refinement = 0.5,0.4,0.3,0.2\n")
    text_file.write("tolerance_strain_refinement1 = 0.5,0.4,0.3,0.2\n")
    text_file.write("free_parameters = b,c,alpha,beta,gamma\n")
    text_file.write("\n")
    text_file.write("[CALLER]\n")
    text_file.write("residues_threshold=0.5\n")
    text_file.write("nb_spots_global_threshold=8\n")
    text_file.write("option_global = v2\n")
    text_file.write("nb_spots_consider = 500\n")
    text_file.write("use_om_user = false\n")
    text_file.write("path_user_OM = none\n")
    text_file.write("\n")
    text_file.write("[DEVELOPMENT]\n")
    text_file.write("write_MTEX_file = true\n")
    text_file.close()
    
    
# In[ ]:




