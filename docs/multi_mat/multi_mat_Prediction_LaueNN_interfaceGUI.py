#!/usr/bin/env python
# coding: utf-8

# # Notebook script for Prediction of Laue spot hkl using the Trained model from step 2 (supports n phase material)
# # This notebook also includes complete indexation process from the predicted spot hkl
# 
# ## Different steps of loading model to predicting the hkl of spots is outlined in this notebook (LaueToolsNN GUI does the same thing)
# 
# ### Define material of interest and path to experimental data; the path to trained model will be extracted automatically by default
# ### Load the trained model 
# ### Prediction of Laue spots hkl 
# ### Constructing orientation matrix from the predicted hkl (i.e. index Laue Patterns)


# =============================================================================
# As of 12/04/2022 Multi_mat scripts only suport "slow" mode of UB matrix computation
# =============================================================================

# In[1]:
if __name__ == "__main__":    
    ## Import modules used for this Notebook
    import os
    from multiprocessing import cpu_count
    import configparser
    try:
        from lauetoolsnn.utils_lauenn import get_multimaterial_detail, resource_path
        from lauetoolsnn.lauetools import dict_LaueTools as dictLT
        from lauetoolsnn.GUI_multi_mat_LaueNN import start
    except:
        import sys
        sys.path.append(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn")
        from utils_lauenn import get_multimaterial_detail, resource_path
        from GUI_multi_mat_LaueNN import start
        sys.path.append(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\lauetools")
        import dict_LaueTools as dictLT
    
    ncpu = cpu_count()
    print("Number of CPUs available : ", ncpu)
    
    # ## step 1: define material and path to data and trained model
    # =============================================================================
    ## User Input dictionary with parameters
    ## In case of only one phase/material, keep same value for material_ and material1_ key
    # =============================================================================
    input_params = {
                    # =============================================================================
                    #       GENERATION OF DATASET              
                    # =============================================================================
                    "prefix" : "",
                    "material_": ["Al2TiO5",
                                  ],             ## same key as used in dict_LaueTools
                    "symmetry": ["orthorhombic",
                                 ],           ## crystal symmetry of material_
                    "SG": [63,
                           ],                     ## Space group of material_ (None if not known)
                    # =============================================================================
                    #        Detector parameters (roughly) of the Experimental setup
                    # =============================================================================
                    ## Sample-detector distance, X center, Y center, two detector angles
                    "detectorparameters" :  [79.50800, 977.8200, 931.9600, 0.3600000, 0.4370000], 
                    "pixelsize" : 0.0734,          ## Detector pixel size
                    "dim1":2018,                   ## Dimensions of detector in pixels
                    "dim2":2016,
                    "emin" : 5,                    ## Minimum and maximum energy to use for simulating Laue Patterns
                    "emax" : 22,
                    # =============================================================================
                    #       Prediction paarmeters             
                    # =============================================================================
                    "experimental_directory": r"F:\bm32_Rene_Al2O3\VF_P1\VF_P1_R4",
                    "experimental_prefix": "VF_P1_R4_",
                    "grid_size_x" : 31,           ## Grid X and Y limit to generate the simulated dataset (a rectangular scan region)
                    "grid_size_y" : 141,  
                    "UB_tolerance": [
                                     0.5,
                                     ],
                    "tolerance_strain": [
                                        [0.6,0.5,0.4,0.3,0.2],
                                        ],
                    "strain_free_parameters": ["b","c","alpha","beta","gamma"],
                    "material_ub_limit": [
                                          1,
                                          ],
                    "UB_matrix_detect": 1,
                    "material_phase_always_present": [1] ## in case if one phase is always 
                                                                # present in a Laue pattern (useful for substrate cases)
                                                                # or for pretty plots
                    }
    #%% ## Step 2: Get material parameters 
    # ### Get model and data paths from the input
    # ### User input parameters for various algorithms to compute the orientation matrix
    material_= input_params["material_"]
    detectorparameters = input_params["detectorparameters"]
    pixelsize = input_params["pixelsize"]
    emax = input_params["emax"]
    emin = input_params["emin"]
    dim1 = input_params["dim1"]
    dim2 = input_params["dim2"]
    symm_ = input_params["symmetry"]
    SG = input_params["SG"]
    tolerance = input_params["UB_tolerance"]
    tolerance_strain = input_params["tolerance_strain"]
    strain_free_parameters = input_params["strain_free_parameters"]
    material_limit = input_params["material_ub_limit"]
    material_phase_always_present = input_params["material_phase_always_present"]
    # =============================================================================
    # Experimental file extension "cor" for simulated dataset or "sCMOS" if experimental
    # =============================================================================
    ccd_label_global = "sCMOS"
    ## Experimental peak search parameters in case of RAW LAUE PATTERNS from detector
    intensity_threshold = 150
    boxsize = 10
    fit_peaks_gaussian = 1
    FitPixelDev = 15
    NumberMaxofFits = 2000 ### Max peaks per LP
    bkg_treatment = "A-B"
    
    ## Requirements
    ubmat = input_params["UB_matrix_detect"] # How many orientation matrix to detect per Laue pattern
    mode_spotCycle = "graphmode" ## mode of calculation
    use_previous_UBmatrix_name = False ## Try previous indexation solutions to speed up the process
    strain_calculation = True ## Strain refinement is required or no
    
    ## Parameters to control the orientation matrix indexation
    softmax_threshold_global = 0.80 # softmax_threshold of the Neural network to consider
    mr_threshold_global = 0.90 # match rate threshold to accept a solution immediately
    cap_matchrate = 0.45 * 100 ## any UB matrix providing MR less than this will be ignored
    coeff = 0.10            ## coefficient to calculate the overlap of two solutions
    coeff_overlap = 0.10   ##10% spots overlap is allowed with already indexed orientation

    ## Additional parameters to refine the orientation matrix construction process
    use_om_user = "false"
    nb_spots_consider = 350
    residues_threshold=0.35
    nb_spots_global_threshold=8
    option_global = "v2"
    additional_expression = ["none"] # for strain assumptions, like a==b for HCP
    # =========================================================================
    # END OF USER INPUT    
    # =========================================================================
    if len(material_) > 1:
        prefix_mat = material_[0]
        for ino, imat in enumerate(material_):
            if ino == 0:
                continue
            prefix_mat = prefix_mat + "_" + imat
    else:
        prefix_mat = material_[0]
    
    model_direc = os.getcwd()+"//"+prefix_mat+input_params["prefix"]
    model_weights = model_direc + "//model_"+prefix_mat+".h5"
    json_file = model_direc + "//model_"+prefix_mat+".json"
    model_annote = "CNN"
    
    if not os.path.exists(model_direc):
        print("The directory doesn't exists; please veify the path")
    else:
        print("Directory where trained model is stored : "+model_direc)
        
    ## get unit cell parameters and other details required for simulating Laue patterns
    rules, symmetry, lattice_material, \
        crystal, SG = get_multimaterial_detail(material_, SG, symm_)

    if input_params["experimental_directory"] == "" and input_params["experimental_prefix"] == "":
        filenameDirec =  model_direc + "//simulated_dataset"
        experimental_prefix = prefix_mat+"_"
        lim_x, lim_y = input_params["grid_size_x"], input_params["grid_size_y"]
        format_file = "cor"
    else:
        filenameDirec = input_params["experimental_directory"]
        experimental_prefix = input_params["experimental_prefix"]
        lim_x, lim_y = input_params["grid_size_x"], input_params["grid_size_y"] 
        format_file = dictLT.dict_CCD["sCMOS"][7]
    
    config_setting = configparser.ConfigParser()
    filepath = resource_path('settings.ini')
    print("Writing settings file in " + filepath)
    config_setting.read(filepath)
    config_setting.set('CALLER', 'residues_threshold',str(residues_threshold))
    config_setting.set('CALLER', 'nb_spots_global_threshold',str(nb_spots_global_threshold))
    config_setting.set('CALLER', 'option_global',option_global)
    config_setting.set('CALLER', 'use_om_user',use_om_user)
    config_setting.set('CALLER', 'nb_spots_consider',str(nb_spots_consider))
    config_setting.set('CALLER', 'path_user_OM',"none")
    config_setting.set('CALLER', 'intensity', str(intensity_threshold))
    config_setting.set('CALLER', 'boxsize', str(boxsize))
    config_setting.set('CALLER', 'pixdev', str(FitPixelDev))
    config_setting.set('CALLER', 'cap_softmax', str(softmax_threshold_global))
    config_setting.set('CALLER', 'cap_mr', str(cap_matchrate/100.))
    config_setting.set('CALLER', 'strain_free_parameters', ",".join(strain_free_parameters))
    config_setting.set('CALLER', 'additional_expression', ",".join(additional_expression))
    with open(filepath, 'w') as configfile:
        config_setting.write(configfile)
    
    if strain_calculation:
        strain_label_global = "YES"
    else:
        strain_label_global = "NO"
        
        
        
    ##Start the GUI plots
    start(        
            model_direc,
            material_,
            emin,
            emax,
            symmetry,
            detectorparameters,
            pixelsize,
            lattice_material,
            mode_spotCycle,
            softmax_threshold_global,
            mr_threshold_global,
            cap_matchrate,
            coeff,
            coeff_overlap,
            fit_peaks_gaussian,
            FitPixelDev,
            NumberMaxofFits,
            tolerance_strain,
            material_limit,
            use_previous_UBmatrix_name,
            material_phase_always_present,
            crystal,
            strain_free_parameters,
            additional_expression,
            strain_label_global, 
            ubmat, 
            boxsize, 
            intensity_threshold,
            ccd_label_global, 
            experimental_prefix, 
            lim_x, 
            lim_y,
            tolerance, 
            filenameDirec, 
            model_weights,
            model_annote
            )

