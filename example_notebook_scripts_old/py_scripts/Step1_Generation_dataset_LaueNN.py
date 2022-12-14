#!/usr/bin/env python
# coding: utf-8

# # Notebook script for generation of training dataset (supports single and two phase material)
# 
# ## For case of more than two phase, the code below can be adapted
# 
# ## Different steps of data generation is outlined in this notebook (LaueToolsNN GUI does the same thing)
# 
# ### Define material of interest
# ### Generate class hkl data for Neural Network model (these are the output neurons)
# ### Clean up generated dataset

# In[1]:

if __name__ == '__main__':     #enclosing required because of multiprocessing

    ## Import modules used for this Notebook
    import os
    
    ## if LaueToolsNN is properly installed
    try:
        from lauetoolsnn.utils_lauenn import generate_classHKL, generate_dataset, rmv_freq_class, get_material_detail
    except:
        # else import from a path where LaueToolsNN files are
        import sys
        sys.path.append(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn")
        from utils_lauenn import generate_classHKL, generate_dataset, rmv_freq_class, get_material_detail
    
    
        # ## step 1: define material and other parameters for simulating Laue patterns
        
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
                    "hkl_max_identify" : 5,        ## Maximum hkl index to classify in a Laue pattern
                    "maximum_angle_to_search":120, ## Angle of radial distribution to reconstruct the histogram (in deg)
                    "step_for_binning" : 0.1,      ## bin widht of angular radial distribution in degree
                    "nb_grains_per_lp" : 5,        ## max grains to be generated in a Laue Image
                    "grains_nb_simulate" : 100,    ## Number of orientations to generate (takes advantage of crystal symmetry)
                    ## Detector parameters (roughly) of the Experimental setup
                    ## Sample-detector distance, X center, Y center, two detector angles
                    "detectorparameters" :  [79.553,979.32,932.31,0.37,0.447], 
                    "pixelsize" : 0.0734,          ## Detector pixel size
                    "dim1":2018,                   ## Dimensions of detector in pixels
                    "dim2":2016,
                    "emin" : 5,                    ## Minimum and maximum energy to use for simulating Laue Patterns
                    "emax" : 22,
                    }
    
    
    # ## Step 2: Get material parameters 
    # ### Generates a folder with material name and gets material unit cell parameters and symmetry object from the get_material_detail function
    
    # In[3]:
    
    
    material_= input_params["material_"]
    material1_= input_params["material1_"]
    n = input_params["hkl_max_identify"]
    maximum_angle_to_search = input_params["maximum_angle_to_search"]
    step_for_binning = input_params["step_for_binning"]
    nb_grains_per_lp = input_params["nb_grains_per_lp"]
    grains_nb_simulate = input_params["grains_nb_simulate"]
    detectorparameters = input_params["detectorparameters"]
    pixelsize = input_params["pixelsize"]
    emax = input_params["emax"]
    emin = input_params["emin"]
    symm_ = input_params["symmetry"]
    symm1_ = input_params["symmetry1"]
    SG = input_params["SG"]
    SG1 = input_params["SG1"]
    
    if material_ != material1_:
        save_directory = os.getcwd()+"//"+material_+"_"+material1_+input_params["prefix"]
    else:
        save_directory = os.getcwd()+"//"+material_+input_params["prefix"]
    print("save directory is : "+save_directory)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    ## get unit cell parameters and other details required for simulating Laue patterns
    rules, symmetry, lattice_material,     crystal, SG, rules1, symmetry1,    lattice_material1, crystal1, SG1 = get_material_detail(material_, SG, symm_,
                                                               material1_, SG1, symm1_)
    
    
    # ## Step 3: Generate Neural network output classes (Laue spot hkls) using the generate_classHKL function
    
    # In[4]:
    
    
    ## procedure for generation of GROUND TRUTH classes
    # general_diff_cond = True will eliminate the hkl index that does not satisfy the general reflection conditions
    generate_classHKL(n, rules, lattice_material, symmetry, material_, crystal=crystal, SG=SG, general_diff_cond=True,
              save_directory=save_directory, write_to_console=print, ang_maxx = maximum_angle_to_search, 
              step = step_for_binning)
    
    if material_ != material1_:
        generate_classHKL(n, rules1, lattice_material1, symmetry1, material1_, crystal=crystal1, SG=SG1, general_diff_cond=True,
                  save_directory=save_directory, write_to_console=print, ang_maxx = maximum_angle_to_search, 
                  step = step_for_binning)
    
    
    # ## Step 4: Generate Training and Testing dataset only for the output classes (Laue spot hkls) calculated in the Step 3
    # ### Uses multiprocessing library
    
    # In[5]:


    ############ GENERATING TRAINING DATA ##############
    # data_realism =True ; will introduce noise and partial Laue patterns in the training dataset
    # modelp can have either "random" for random orientation generation or "uniform" for uniform orientation generation
    # include_scm (if True; misorientation_angle parameter need to be defined): this parameter introduces misoriented crystal of specific angle along a crystal axis in the training dataset
    generate_dataset(material_=material_, material1_=material1_, ang_maxx=maximum_angle_to_search,
                         step=step_for_binning, mode=0, 
                         nb_grains=nb_grains_per_lp, nb_grains1=nb_grains_per_lp, 
                         grains_nb_simulate=grains_nb_simulate, data_realism = True, 
                         detectorparameters=detectorparameters, pixelsize=pixelsize, type_="training_data",
                         var0 = 1, dim1=input_params["dim1"], dim2=input_params["dim2"], 
                         removeharmonics=1, save_directory=save_directory,
                        write_to_console=print, emin=emin, emax=emax, modelp = "random",
                        misorientation_angle = 1, general_diff_rules = False, 
                        crystal = crystal, crystal1 = crystal1, include_scm=False,)
    
    ############ GENERATING TESTING DATA ##############
    factor = 5 # validation split for the training dataset  --> corresponds to 20% of total training dataset
    generate_dataset(material_=material_, material1_=material1_, ang_maxx=maximum_angle_to_search,
                         step=step_for_binning, mode=0, 
                         nb_grains=nb_grains_per_lp, nb_grains1=nb_grains_per_lp, 
                         grains_nb_simulate=grains_nb_simulate//factor, data_realism = True, 
                         detectorparameters=detectorparameters, pixelsize=pixelsize, type_="testing_data",
                         var0 = 1, dim1=input_params["dim1"], dim2=input_params["dim2"], 
                         removeharmonics=1, save_directory=save_directory,
                        write_to_console=print, emin=emin, emax=emax, modelp = "random",
                        misorientation_angle = 1, general_diff_rules = False, 
                        crystal = crystal, crystal1 = crystal1, include_scm=False,)
    
    ## Updating the ClassHKL list by removing the non-common HKL or less frequent HKL from the list
    ## The non-common HKL can occur as a result of the detector position and energy used
    # freq_rmv: remove output hkl if the training dataset has less tha 100 occurances of the considered hkl (freq_rmv1 for second phase)
    # Weights (penalty during training) are also calculated based on the occurance
    rmv_freq_class(freq_rmv = 500, freq_rmv1 = 500,
                        save_directory=save_directory, material_=material_, 
                        material1_=material1_, write_to_console=print)
    
    ## End of data generation for Neural network training: all files are saved in the same folder to be later used for training and prediction


# In[ ]:




