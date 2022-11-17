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
    import numpy as np
    import os
    import multiprocessing
    from multiprocessing import cpu_count
    import time, datetime
    import glob, re
    import configparser
    from itertools import accumulate
    ## if LaueToolsNN is properly installed
    try:
        from lauetoolsnn.utils_lauenn import get_multimaterial_detail, new_MP_multimat_function, resource_path, global_plots_MM
        from lauetoolsnn.lauetools import dict_LaueTools as dictLT
        from lauetoolsnn.NNmodels import read_hdf5
    except:
        # else import from a path where LaueToolsNN files are
        import sys
        sys.path.append(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn")
        from utils_lauenn import get_multimaterial_detail, new_MP_multimat_function, resource_path, global_plots_MM
        from NNmodels import read_hdf5        
        sys.path.append(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\lauetools")
        import dict_LaueTools as dictLT
    
    import _pickle as cPickle
    from tqdm import tqdm
    
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
                    "material_": ["Zr_alpha",
                                  "ZrO2_mono",
                                  ],             ## same key as used in dict_LaueTools
                    "prefix" : "",                 ## prefix for the folder to be created for training dataset
                    "symmetry": ["hexagonal",
                                 "monoclinic",
                                 ],           ## crystal symmetry of material_
                    "SG": [194,
                           14,
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
                    #       Prediction paarmeters             
                    # =============================================================================
                    "experimental_directory": r"C:\Users\purushot\Desktop\Guillou_Laue\scan_0012",
                    "experimental_prefix": "ech1_map2D_3_",
                    "grid_size_x" : 1,            ## Grid X and Y limit to generate the simulated dataset (a rectangular scan region)
                    "grid_size_y" : 5,  
                    "UB_tolerance": [0.6,
                                     0.6,
                                     ],
                    "tolerance_strain": [
                                        [0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15],
                                        [0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15],
                                        ],
                    "strain_free_parameters": ["b","c","alpha","beta","gamma"],
                    "material_ub_limit": [10,
                                          10,
                                          ],
                    
                    "UB_matrix_detect": 12,
                    "material_phase_always_present": [1,1,1,1,1,1,1,1,1,2,2,2],
                    ## in case if one phase is always present in a Laue pattern (useful for substrate cases)
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
    model_annote = "from_file"
    
    if len(material_) > 1:
        prefix_mat = material_[0]
        for ino, imat in enumerate(material_):
            if ino == 0:
                continue
            prefix_mat = prefix_mat + "_" + imat
    else:
        prefix_mat = material_[0]
    
    model_direc = os.getcwd()+"//"+prefix_mat+input_params["prefix"]
    
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
    
    hkl_all_class0 = []
    for ino, imat in enumerate(material_):
        with open(model_direc+"//classhkl_data_nonpickled_"+imat+".pickle", "rb") as input_file:
            hkl_all_class_load = cPickle.load(input_file)[0]
        hkl_all_class0.append(hkl_all_class_load)
        
    ## Experimental peak search parameters in case of RAW LAUE PATTERNS from detector
    intensity_threshold = 150
    boxsize = 10
    fit_peaks_gaussian = 1
    FitPixelDev = 18
    NumberMaxofFits = 2000 ### Max peaks per LP
    bkg_treatment = "A-B"

    ## Requirements
    ubmat = input_params["UB_matrix_detect"] # How many orientation matrix to detect per Laue pattern
    mode_spotCycle = "graphmode" ## mode of calculation
    use_previous_UBmatrix_name = False ## Try previous indexation solutions to speed up the process
    strain_calculation = True ## Strain refinement is required or not
    ccd_label_global = "sCMOS"

    ## Parameters to control the orientation matrix indexation
    softmax_threshold_global = 0.80 # softmax_threshold of the Neural network to consider
    mr_threshold_global = 0.70 # match rate threshold to accept a solution immediately
    cap_matchrate = 0.20 * 100 ## any UB matrix providing MR less than this will be ignored
    coeff = 0.10            ## coefficient to calculate the overlap of two solutions
    coeff_overlap = 0.10    ##10% spots overlap is allowed with already indexed orientation
    
    ## Additional parameters to refine the orientation matrix construction process
    use_om_user = "false"
    nb_spots_consider = 500
    residues_threshold=0.5
    nb_spots_global_threshold=8
    option_global = "v2"
    additional_expression = ["none"] # for strain assumptions, like a==b for HCP
    
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
    
    ## load model related files and generate the model
    classhkl = np.load(model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_0"]
    angbins = np.load(model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_1"]
    ind_mat_all = np.load(model_direc+"//MOD_grain_classhkl_angbin.npz",allow_pickle=True)["arr_5"]
    ind_mat = []
    for inni in ind_mat_all:
        ind_mat.append(len(inni))
    ind_mat = [item for item in accumulate(ind_mat)]
    
    # json_file = open(model_direc+"//model_"+prefix_mat+".json", 'r')
    load_weights = model_direc + "//model_"+prefix_mat+".h5"
    wb = read_hdf5(load_weights)
    temp_key = list(wb.keys())

    ct = time.time()
    now = datetime.datetime.fromtimestamp(ct)
    c_time = now.strftime("%Y-%m-%d_%H-%M-%S")   
    
    #%% ## Step 3: Initialize variables and prepare arguments for multiprocessing module
    
    col = [[] for i in range(int(ubmat))]
    colx = [[] for i in range(int(ubmat))]
    coly = [[] for i in range(int(ubmat))]
    rotation_matrix = [[] for i in range(int(ubmat))]
    strain_matrix = [[] for i in range(int(ubmat))]
    strain_matrixs = [[] for i in range(int(ubmat))]
    match_rate = [[] for i in range(int(ubmat))]
    spots_len = [[] for i in range(int(ubmat))]
    iR_pix = [[] for i in range(int(ubmat))]
    fR_pix = [[] for i in range(int(ubmat))]
    mat_global = [[] for i in range(int(ubmat))]
    best_match = [[] for i in range(int(ubmat))]
    spots1_global = [[] for i in range(int(ubmat))]
    for i in range(int(ubmat)):
        col[i].append(np.zeros((lim_x*lim_y,3)))
        colx[i].append(np.zeros((lim_x*lim_y,3)))
        coly[i].append(np.zeros((lim_x*lim_y,3)))
        rotation_matrix[i].append(np.zeros((lim_x*lim_y,3,3)))
        strain_matrix[i].append(np.zeros((lim_x*lim_y,3,3)))
        strain_matrixs[i].append(np.zeros((lim_x*lim_y,3,3)))
        match_rate[i].append(np.zeros((lim_x*lim_y,1)))
        spots_len[i].append(np.zeros((lim_x*lim_y,1)))
        iR_pix[i].append(np.zeros((lim_x*lim_y,1)))
        fR_pix[i].append(np.zeros((lim_x*lim_y,1)))
        mat_global[i].append(np.zeros((lim_x*lim_y,1)))
        best_match[i].append([[] for jk in range(lim_x*lim_y)])
        spots1_global[i].append([[] for jk in range(lim_x*lim_y)])

    # =============================================================================
    #         ## Multi-processing routine
    # =============================================================================        
    ## Number of files to generate
    grid_files = np.zeros((lim_x,lim_y))
    filenm = np.chararray((lim_x,lim_y), itemsize=1000)
    grid_files = grid_files.ravel()
    filenm = filenm.ravel()
    count_global = lim_x * lim_y
    list_of_files = glob.glob(filenameDirec+'//'+experimental_prefix+'*.'+format_file)
    ## sort files
    list_of_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    
    if len(list_of_files) == count_global:
        for ii in range(len(list_of_files)):
            grid_files[ii] = ii
            filenm[ii] = list_of_files[ii]     
        print("expected "+str(count_global)+" files based on the XY grid ("+str(lim_x)+","+str(lim_y)+") defined by user")
        print("and found "+str(len(list_of_files))+" files")
    else:
        print("expected "+str(count_global)+" files based on the XY grid ("+str(lim_x)+","+str(lim_y)+") defined by user")
        print("But found "+str(len(list_of_files))+" files (either all data is not written yet or maybe XY grid definition is not proper)")
        digits = len(str(count_global))
        digits = max(digits,4)
        # Temp fix
        for ii in range(count_global):
            text = str(ii)
            if ii < 10000:
                string = text.zfill(4)
            else:
                string = text.zfill(5)
            file_name_temp = filenameDirec+'//'+experimental_prefix + string+'.'+format_file
            ## store it in a grid 
            filenm[ii] = file_name_temp
    
    check = np.zeros((count_global,int(ubmat)))
    # =============================================================================
    blacklist = None
    
    ### Create a COR directory to be loaded in LaueTools
    cor_file_directory = filenameDirec + "//" + experimental_prefix+"CORfiles"
    if list_of_files[0].split(".")[-1] in ['cor',"COR","Cor"]:
        cor_file_directory = filenameDirec 
    if not os.path.exists(cor_file_directory):
        os.makedirs(cor_file_directory)
    
    try_prevs = False
    files_treated = []
    
    valu12 = [[ filenm[ii].decode(), 
                ii,
                rotation_matrix,
                strain_matrix,
                strain_matrixs,
                col,
                colx,
                coly,
                match_rate,
                spots_len, 
                iR_pix, 
                fR_pix,
                best_match,
                mat_global,
                check,
                detectorparameters,
                pixelsize,
                angbins,
                classhkl,
                hkl_all_class0,
                emin,
                emax,
                material_,
                symmetry,
                lim_x,
                lim_y,
                strain_calculation, 
                ind_mat, 
                model_direc, 
                tolerance,
                int(ubmat), ccd_label_global, 
                None,
                float(intensity_threshold),
                int(boxsize),
                bkg_treatment,
                filenameDirec, 
                experimental_prefix,
                blacklist,
                None,
                files_treated,
                try_prevs, ## try previous is kept true, incase if its stuck in loop
                wb,
                temp_key,
                cor_file_directory,
                mode_spotCycle,
                softmax_threshold_global,
                mr_threshold_global,
                cap_matchrate,
                tolerance_strain,
                NumberMaxofFits,
                fit_peaks_gaussian,
                FitPixelDev,
                coeff,
                coeff_overlap,
                material_limit,
                use_previous_UBmatrix_name,
                material_phase_always_present,
                crystal,
                strain_free_parameters,
                model_annote] for ii in range(count_global)]
    
    # start_time = time.time()
    # # Launch on single file to verify
    # results = new_MP_multimat_function(valu12[0])
    # print('Took ',time.time()-start_time, "seconds")
    
    # print("matching rate")
    # temp_0 = [results[6][i][0][0] for i in range(len(results[6]))]
    # print(temp_0)
    
    # print("Orientation metrix")
    # temp_0 = [results[2][i][0][0].ravel() for i in range(len(results[6]))]
    # for itemm in temp_0:
    #     print(",".join(str(x) for x in itemm))
        
    #% Launch multiprocessing prediction     
    if 1:
        args = zip(valu12)
        with multiprocessing.Pool(ncpu) as pool:
            results = pool.starmap(new_MP_multimat_function, tqdm(args, total=len(valu12)))
            
            for r in results:
                r_message_mpdata = r
                strain_matrix_mpdata, strain_matrixs_mpdata, rotation_matrix_mpdata, col_mpdata,\
                colx_mpdata, coly_mpdata, match_rate_mpdata, mat_global_mpdata,\
                    cnt_mpdata, meta_mpdata, files_treated_mpdata, spots_len_mpdata, \
                        iR_pixel_mpdata, fR_pixel_mpdata, best_match_mpdata, check_mpdata = r_message_mpdata
        
                for i_mpdata in files_treated_mpdata:
                    files_treated.append(i_mpdata)
        
                for intmat_mpdata in range(int(ubmat)):
                    check[cnt_mpdata,intmat_mpdata] = check_mpdata[cnt_mpdata,intmat_mpdata]
                    mat_global[intmat_mpdata][0][cnt_mpdata] = mat_global_mpdata[intmat_mpdata][0][cnt_mpdata]
                    strain_matrix[intmat_mpdata][0][cnt_mpdata,:,:] = strain_matrix_mpdata[intmat_mpdata][0][cnt_mpdata,:,:]
                    strain_matrixs[intmat_mpdata][0][cnt_mpdata,:,:] = strain_matrixs_mpdata[intmat_mpdata][0][cnt_mpdata,:,:]
                    rotation_matrix[intmat_mpdata][0][cnt_mpdata,:,:] = rotation_matrix_mpdata[intmat_mpdata][0][cnt_mpdata,:,:]
                    col[intmat_mpdata][0][cnt_mpdata,:] = col_mpdata[intmat_mpdata][0][cnt_mpdata,:]
                    colx[intmat_mpdata][0][cnt_mpdata,:] = colx_mpdata[intmat_mpdata][0][cnt_mpdata,:]
                    coly[intmat_mpdata][0][cnt_mpdata,:] = coly_mpdata[intmat_mpdata][0][cnt_mpdata,:]
                    match_rate[intmat_mpdata][0][cnt_mpdata] = match_rate_mpdata[intmat_mpdata][0][cnt_mpdata]
                    spots_len[intmat_mpdata][0][cnt_mpdata] = spots_len_mpdata[intmat_mpdata][0][cnt_mpdata]
                    iR_pix[intmat_mpdata][0][cnt_mpdata] = iR_pixel_mpdata[intmat_mpdata][0][cnt_mpdata]
                    fR_pix[intmat_mpdata][0][cnt_mpdata] = fR_pixel_mpdata[intmat_mpdata][0][cnt_mpdata]
                    best_match[intmat_mpdata][0][cnt_mpdata] = best_match_mpdata[intmat_mpdata][0][cnt_mpdata]
                
        #% Save results
        save_directory_ = filenameDirec+"//results_"+prefix_mat+"_"+c_time
        if not os.path.exists(save_directory_):
            os.makedirs(save_directory_)
        
        ## intermediate saving of pickle objects with results
        np.savez_compressed(save_directory_+ "//results.npz", 
                            best_match, mat_global, rotation_matrix, strain_matrix, 
                            strain_matrixs, col, colx, coly, match_rate, files_treated,
                            lim_x, lim_y, spots_len, iR_pix, fR_pix,
                            material_)
        ## intermediate saving of pickle objects with results
        with open(save_directory_+ "//results.pickle", "wb") as output_file:
                cPickle.dump([best_match, mat_global, rotation_matrix, strain_matrix, 
                              strain_matrixs, col, colx, coly, match_rate, files_treated,
                              lim_x, lim_y, spots_len, iR_pix, fR_pix,
                              material_, lattice_material,
                              symmetry, crystal], output_file)
        print("data saved in ", save_directory_)

        try:
            global_plots_MM(lim_x, lim_y, rotation_matrix, strain_matrix, strain_matrixs, 
                         col, colx, coly, match_rate, mat_global, spots_len, 
                         iR_pix, fR_pix, save_directory_, material_,
                         match_rate_threshold=5, bins=30)
        except:
            print("Error in the global plots module")