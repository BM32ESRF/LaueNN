# coding: utf-8
"""
Created on June 18 06:54:04 2021

Script routine for Laue neural network training and prediction
script for generation of training dataset (supports single and two phase material)
For case of more than two phase, please refere to Multi-mat lauenn

@author: Ravi raj purohit PURUSHOTTAM RAJ PUROHIT (purushot@esrf.fr)

Credits:
Lattice and symmetry routines are extracted and adapted from the PYMICRO and Xrayutilities repository

"""
__author__ = "Ravi raj purohit PURUSHOTTAM RAJ PUROHIT, CRG-IF BM32 @ ESRF"

if __name__ == '__main__':     #enclosing required because of multiprocessing

    ## Import modules used for this Notebook
    import os    
    ## Get the path of the lauetoolsnn library
    import lauetoolsnn
    laueNN_path = os.path.dirname(lauetoolsnn.__file__)
    print("LaueNN path is", laueNN_path)
    # =============================================================================
    # Step 0: Define the dictionary with all parameters 
    # =============================================================================
    ## User Input dictionary with parameters
    ## In case of only one phase/material, keep same value for material_ and material1_ key
    input_params = {
                    "global_path" : os.getcwd(),
                    "prefix" : "_2phase",                 ## prefix for the folder to be created for training dataset

                    "material_": ["Sn_beta", "Si"],             ## same key as used in dict_LaueTools
                    "symmetry": ["tetragonal", "cubic"],           ## crystal symmetry of material_
                    "SG": [141, 227], #186                    ## Space group of material_ (None if not known)
                    "hkl_max_identify" : [7,5],        ## Maximum hkl index to classify in a Laue pattern
                    "nb_grains_per_lp_mat" : [4,1],        ## max grains to be generated in a Laue Image

                    ## hkl_max_identify : can be "auto" or integer: Maximum index of HKL to build output classes
                    
                    # =============================================================================
                    # ## Data generation settings
                    # =============================================================================
                    "grains_nb_simulate" : 1000,    ## Number of orientations to generate (takes advantage of crystal symmetry)
                    "classes_with_frequency_to_remove": [250,250], ## classes_with_frequency_to_remove: HKL class with less appearance than 
                                                                            ##  specified will be ignored in output
                    "desired_classes_output": ["all","all"], ## desired_classes_output : can be all or an integer: to limit the number of output classes

                    "include_small_misorientation": False, ## to include additional data with small angle misorientation
                    "misorientation": 5, ##only used if "include_small_misorientation" is True
                    "maximum_angle_to_search":90, ## Angle of radial distribution to reconstruct the histogram (in deg)
                    "step_for_binning" : 0.1,      ## bin widht of angular radial distribution in degree
                    
                    # =============================================================================
                    #  ## Training parameters
                    # =============================================================================
                    "orientation_generation": "uniform", ## can be random or uniform
                    "batch_size":100,               ## batches of files to use while training
                    "epochs":8,                    ## number of epochs for training

                    # =============================================================================
                    # ## Detector parameters of the Experimental setup
                    # =============================================================================
                    ## Sample-detector distance, X center, Y center, two detector angles
                    ## MODIFY
                    "detectorparameters" :  [79.237,972.19,938.75,0.396,0.494],
                    "pixelsize" : 0.0734,          ## Detector pixel size
                    "dim1":2018,                   ## Dimensions of detector in pixels
                    "dim2":2016,
                    "emin" : 5,                    ## Minimum and maximum energy to use for simulating Laue Patterns
                    "emax" : 22,
                    "ccd_label" : "cor",
                    
                    # =============================================================================
                    # ## Prediction parameters
                    # =============================================================================
                    ## MODIFY
                    "experimental_directory": r"/home/esrf/vives/helene1408/HeleneV1/Sn_beta_Si/results_Sn_beta_Si_mardi_mat-whisk2_2022-09-21_11-03-26/image_cor",
                    "experimental_prefix": r"img_",
                    "grid_size_x" : 81,            ## Grid X and Y limit to generate the simulated dataset (a rectangular scan region)
                    "grid_size_y" : 81,
                    
                    # =============================================================================
                    # ## Prediction Settings
                    # =============================================================================
                    # model_weight_file: if none, it will select by default the latest H5 weight file, else provide a specific model
                    # softmax_threshold_global: thresholding to limit the predicted spots search zone
                    # cap_matchrate: any UB matrix providing MR less than this will be ignored
                    # coeff: should be same as cap_matchrate or no? (this is for try previous UB matrix)
                    # coeff_overlap: coefficient to limit the overlapping between spots; if more than this, new solution will be computed
                    # mode_spotCycle: How to cycle through predicted spots (slow or graphmode )
                    #TO MODIFY
                    "UB_matrix_to_detect" : 20,
                    "matrix_tolerance" : [0.6, 0.6],
                    "material_limit" : [1000, 0],
                    "material_phase_always_present" : None,
                    "softmax_threshold_global" : 0.85,
                    "cap_matchrate" : 0.40,
                    "coeff_overlap" : 0.3,
                    "mode_spotCycle" : "graphmode",
                    ##true for few crystal and prefered texture case, otherwise time consuming; advised for single phase alone
                    "use_previous" : False,
                    
                    # =============================================================================
                    # # [PEAKSEARCH]
                    # =============================================================================
                    "intensity_threshold" : 2,## for skimage this is of image standard deviation
                    "boxsize" : 10,## for skimage this is box size to fit
                    "fit_peaks_gaussian" : 1,## for skimage this is of no sense
                    "FitPixelDev" : 15, ## for skimage this is distance between peaks to avoid
                    "NumberMaxofFits" : 3000,## for skimage this is maximum leastquare attempts before giving up
                    "mode": "skimage",

                    # =============================================================================
                    # # [STRAINCALCULATION]
                    # =============================================================================
                    "strain_compute" : True,
                    "tolerance_strain_refinement" : [[0.6,0.5,0.4,0.3,0.2,0.1,0.05],
                                                     [0.6,0.5,0.4,0.3,0.2,0.1,0.05]],
                    "free_parameters" : ["b","c","alpha","beta","gamma"],
                    
                    # =============================================================================
                    # # [Additional settings]
                    # =============================================================================
                    "residues_threshold":0.25,
                    "nb_spots_global_threshold":8,
                    "nb_spots_consider" : 500,
                    # User defined orientation matrix supplied in a file
                    ## MODIFY
                    "use_om_user" : True,
                    "path_user_OM" : r"/home/esrf/vives/Desktop/LaueNN_script/Sn_beta_Si_2phase/results__2phase_2022-11-21_16-10-09/average_rot_mat.txt",
                    }

    run_prediction = True
    # =============================================================================
    # END OF USER INPUT
    # =============================================================================
    # global_path: path where all model related files will be saved
    global_path = input_params["global_path"]
    
    ## verify the length of material key
    if len(input_params["material_"]) > 2:
        print("This script uses modules of LaueNN that only supports maximum of two materials; please use multi-material script incase of more than 2phases")
        print("The script will run, however, will only use the first 2 materials")
    
    if len(input_params["material_"]) == 1:
        print("only one material is defined")
        ## modify the dictionary for two phase
        input_params["material_"].append(input_params["material_"][0])
        input_params["symmetry"].append(input_params["symmetry"][0])
        input_params["SG"].append(input_params["SG"][0])
        input_params["hkl_max_identify"].append(input_params["hkl_max_identify"][0])
        input_params["nb_grains_per_lp_mat"].append(input_params["nb_grains_per_lp_mat"][0])
        
        input_params["classes_with_frequency_to_remove"].append(input_params["classes_with_frequency_to_remove"][0])
        input_params["desired_classes_output"].append(input_params["desired_classes_output"][0])
        input_params["matrix_tolerance"].append(input_params["matrix_tolerance"][0])
        input_params["material_limit"].append(input_params["material_limit"][0])
        input_params["tolerance_strain_refinement"].append(input_params["tolerance_strain_refinement"][0])
        
        
    if run_prediction:
        prediction_GUI = False
        if prediction_GUI:
            print("Prediction with GUI is selected, please load the generated config file in LaueNN GUI and continue the prediction process")
        else:
            import numpy as np
            import multiprocessing, os, time, datetime
            import configparser, glob, re
            ## if LaueToolsNN is properly installed
            from lauetoolsnn.utils_lauenn import  get_material_detail, new_MP_function, resource_path, global_plots,\
                write_average_orientation, convert_pickle_to_hdf5, write_prediction_stats, write_MTEXdata, new_MP_function_v1
            from lauetoolsnn.lauetools import dict_LaueTools as dictLT
            from lauetoolsnn.NNmodels import read_hdf5
            from keras.models import model_from_json
            import _pickle as cPickle
            from tqdm import tqdm
            from multiprocessing import Process, Queue
            
            ncpu = multiprocessing.cpu_count()
            print("Number of CPUs available : ", ncpu)
            
            # ## Get material parameters 
            # ### Get model and data paths from the input
            # ### User input parameters for various algorithms to compute the orientation matrix
    
            material_= input_params["material_"][0]
            material1_= input_params["material_"][1]
            detectorparameters = input_params["detectorparameters"]
            pixelsize = input_params["pixelsize"]
            emax = input_params["emax"]
            emin = input_params["emin"]
            dim1 = input_params["dim1"]
            dim2 = input_params["dim2"]
            symm_ = input_params["symmetry"][0]
            symm1_ = input_params["symmetry"][1]
            SG = input_params["SG"][0]
            SG1 = input_params["SG"][1]
            
            if material_ != material1_:
                model_direc = os.path.join(global_path,material_+"_"+material1_+input_params["prefix"])
            else:
                model_direc = os.path.join(global_path,material_+input_params["prefix"])
                
            if not os.path.exists(model_direc):
                print("The directory doesn't exists; please veify the path")
                pass
            else:
                print("Directory where trained model is stored : "+model_direc)
            
            if material_ != material1_:
                prefix1 = material_+"_"+material1_
            else:
                prefix1 = material_
            
            filenameDirec = input_params["experimental_directory"]
            experimental_prefix = input_params["experimental_prefix"]
            lim_x, lim_y = input_params["grid_size_x"], input_params["grid_size_y"] 
            format_file = dictLT.dict_CCD[input_params["ccd_label"]][7]
            ## Experimental peak search parameters in case of RAW LAUE PATTERNS from detector
            intensity_threshold = input_params["intensity_threshold"]
            boxsize = input_params["boxsize"]
            fit_peaks_gaussian = input_params["fit_peaks_gaussian"]
            FitPixelDev = input_params["FitPixelDev"]
            NumberMaxofFits = input_params["NumberMaxofFits"]
            bkg_treatment = "A-B"
            mode_peaksearch = input_params["mode"]
            
            ## get unit cell parameters and other details required for simulating Laue patterns
            rules, symmetry, lattice_material,\
                crystal, SG, rules1, symmetry1,\
                    lattice_material1, crystal1, SG1 = get_material_detail(material_, SG, symm_,
                                                                       material1_, SG1, symm1_)
            
            ## Requirements
            ubmat = input_params["UB_matrix_to_detect"] # How many orientation matrix to detect per Laue pattern
            mode_spotCycle = input_params["mode_spotCycle"] ## mode of calculation
            use_previous_UBmatrix_name = input_params["use_previous"] ## Try previous indexation solutions to speed up the process
            strain_calculation = input_params["strain_compute"] ## Strain refinement is required or not
            ccd_label_global = input_params["ccd_label"]
                    
            ## tolerance angle to match simulated and experimental spots for two materials
            tolerance = input_params["matrix_tolerance"][0]
            tolerance1 = input_params["matrix_tolerance"][1]
            
            ## tolerance angle for strain refinements
            tolerance_strain = input_params["tolerance_strain_refinement"][0]
            tolerance_strain1 = input_params["tolerance_strain_refinement"][1]
            strain_free_parameters = input_params["free_parameters"]
            
            ## Parameters to control the orientation matrix indexation
            softmax_threshold_global = input_params["softmax_threshold_global"] # softmax_threshold of the Neural network to consider
            mr_threshold_global = 0.90 # match rate threshold to accept a solution immediately
            cap_matchrate = input_params["cap_matchrate"] * 100 ## any UB matrix providing MR less than this will be ignored
            coeff = 0.30            ## coefficient to calculate the overlap of two solutions
            coeff_overlap = input_params["coeff_overlap"]    ##10% spots overlap is allowed with already indexed orientation
            material0_limit = input_params["material_limit"][0]  ## how many UB can be proposed for first material
            material1_limit = input_params["material_limit"][1] ## how many UB can be proposed for second material; this forces the orientation matrix deduction algorithm to find only a required materials matrix
            material_phase_always_present = input_params["material_phase_always_present"] ## in case if one phase is always present in a Laue pattern (useful for substrate cases)
            
            ## Additional parameters to refine the orientation matrix construction process
            use_om_user = str(input_params["use_om_user"]).lower()
            path_user_OM = input_params["path_user_OM"]
            nb_spots_consider = input_params["nb_spots_consider"]
            residues_threshold = input_params["residues_threshold"]
            nb_spots_global_threshold = input_params["nb_spots_global_threshold"]
            option_global = "v1"
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
            config_setting.set('CALLER', 'path_user_OM',str(path_user_OM))
            config_setting.set('CALLER', 'intensity', str(intensity_threshold))
            config_setting.set('CALLER', 'boxsize', str(boxsize))
            config_setting.set('CALLER', 'pixdev', str(FitPixelDev))
            config_setting.set('CALLER', 'cap_softmax', str(softmax_threshold_global))
            config_setting.set('CALLER', 'cap_mr', str(cap_matchrate/100.))
            config_setting.set('CALLER', 'strain_free_parameters', ",".join(strain_free_parameters))
            config_setting.set('CALLER', 'additional_expression', ",".join(additional_expression))
            config_setting.set('CALLER', 'mode_peaksearch', str(mode_peaksearch))
            with open(filepath, 'w') as configfile:
                config_setting.write(configfile)
                
            ## load model related files and generate the model
            json_file = open(model_direc+"//model_"+prefix1+".json", 'r')
            classhkl = np.load(model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_0"]
            angbins = np.load(model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_1"]
            if material_ != material1_:
                ind_mat = np.load(model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_5"]
                ind_mat1 = np.load(model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_6"] 
            else:
                ind_mat = None
                ind_mat1 = None
            load_weights = model_direc + "//model_"+prefix1+".h5"
            wb = read_hdf5(load_weights)
            temp_key = list(wb.keys())
            
            # # load json and create model
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            print("Constructing model")
            model.load_weights(load_weights)
            print("Uploading weights to model")
            print("All model files found and loaded")
            
            ct = time.time()
            now = datetime.datetime.fromtimestamp(ct)
            c_time = now.strftime("%Y-%m-%d_%H-%M-%S")
            
            hkl_all_class1 = None
            with open(model_direc+"//classhkl_data_nonpickled_"+material_+".pickle", "rb") as input_file:
                hkl_all_class0 = cPickle.load(input_file)[0]
            
            if material_ != material1_:
                with open(model_direc+"//classhkl_data_nonpickled_"+material1_+".pickle", "rb") as input_file:
                    hkl_all_class1 = cPickle.load(input_file)[0]
    
            # ## Initialize variables and prepare arguments for multiprocessing module
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
            
            ##hack for multiprocessing, but very time consuming
            if use_previous_UBmatrix_name:
                np.savez_compressed(model_direc+'//rotation_matrix_indexed_1.npz', rotation_matrix, mat_global, match_rate, 0.0)
            
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
            
            ##making a big argument list for each CPU
            valu12 = [[filenm[ii].decode(), ii,
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
                        hkl_all_class1,
                        emin,
                        emax,
                        material_,
                        material1_,
                        symmetry,
                        symmetry1,   
                        lim_x,
                        lim_y,
                        strain_calculation, 
                        ind_mat, ind_mat1,
                        model_direc, float(tolerance),
                        float(tolerance1),
                        int(ubmat), ccd_label_global, 
                        None,
                        float(intensity_threshold),
                        int(boxsize),bkg_treatment,
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
                        tolerance_strain1,
                        NumberMaxofFits,
                        fit_peaks_gaussian,
                        FitPixelDev,
                        coeff,
                        coeff_overlap,
                        material0_limit,
                        material1_limit,
                        use_previous_UBmatrix_name,
                        material_phase_always_present,
                        crystal,
                        crystal1,
                        strain_free_parameters] for ii in range(count_global)]
            
            if ubmat > 10:
                def chunker_list(seq, size):
                    return (seq[i::size] for i in range(size))
                # =============================================================================
                # Below code when lot of UB matrix is demanded
                # =============================================================================
                _inputs_queue = Queue()
                _outputs_queue = Queue()
                _worker_processes = {}
                for i in range(ncpu):
                    _worker_processes[i]= Process(target=new_MP_function_v1, args=(_inputs_queue, _outputs_queue, i+1))
                for i in range(ncpu):
                    _worker_processes[i].start()
                    
                    
                chunks = chunker_list(valu12, ncpu)
                chunks_mp = list(chunks)

                meta = {'t1':time.time()}
                for ijk in range(int(ncpu)):
                    _inputs_queue.put((chunks_mp[ijk], ncpu, meta))
                ### Update data from multiprocessing
                pbar = tqdm(total=count_global)
                
                
                unique_count = []
                while True:
                    time.sleep(0.1)
                    
                    if len(np.unique(unique_count)) == count_global:
                        print("All files have been treated")
                        break
                                    
                    if not _outputs_queue.empty():            
                        n_range = _outputs_queue.qsize()
                        for _ in range(n_range):
                            r_message_mpdata = _outputs_queue.get()
                                         
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
                        
                            if cnt_mpdata not in unique_count:
                                pbar.update(1)
                            unique_count.append(cnt_mpdata)            
            else:      
                # =============================================================================
                #   Below code, not good when UB matrix is > 10          
                # =============================================================================
                import time
                args = zip(valu12)
                with multiprocessing.Pool(ncpu) as pool:
                    results = pool.starmap(new_MP_function, tqdm(args, total=len(valu12)), chunksize=1)
                    
                    for r_message_mpdata in results:
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
                
                time.sleep(1)
            ### Save files and results
            #% Save results
            save_directory_ = model_direc+"//results_"+input_params["prefix"]+"_"+c_time
            if not os.path.exists(save_directory_):
                os.makedirs(save_directory_)
            
            np.savez_compressed(save_directory_+ "//results.npz", 
                                best_match, mat_global, rotation_matrix, strain_matrix, 
                                strain_matrixs, col, colx, coly, match_rate, files_treated,
                                lim_x, lim_y, spots_len, iR_pix, fR_pix,
                                material_, material1_)
            ## intermediate saving of pickle objects with results
            with open(save_directory_+ "//results.pickle", "wb") as output_file:
                    cPickle.dump([best_match, mat_global, rotation_matrix, strain_matrix, 
                                  strain_matrixs, col, colx, coly, match_rate, files_treated,
                                  lim_x, lim_y, spots_len, iR_pix, fR_pix,
                                  material_, material1_, lattice_material, lattice_material1,
                                  symmetry, symmetry1, crystal, crystal1], output_file)
            print("data saved in ", save_directory_)
            
            ## Lets save also a set of average UB matrix in text file to be used with user_OM setting    
            try:
                write_average_orientation(save_directory_, mat_global, rotation_matrix,
                                              match_rate, lim_x, lim_y, crystal, crystal1,
                                              radius=10, grain_ang=5, pixel_grain_definition=5)
            except:
                print("Error with Average orientation and grain index calculation")
                
            try:
                convert_pickle_to_hdf5(save_directory_, files_treated, rotation_matrix, strain_matrix, 
                                       strain_matrixs, match_rate, spots_len, iR_pix, 
                                       fR_pix, colx, coly, col, mat_global,
                                       material_, material1_, lim_x, lim_y)
            except:
                print("Error writting H5 file")
                print("Make sure you have pandas and pytables")
            
            try:
                write_prediction_stats(save_directory_, material_, material1_, files_treated,\
                                        lim_x, lim_y, best_match, strain_matrixs, strain_matrix, iR_pix,\
                                        fR_pix,  mat_global)
            except:
                print("Error writting prediction statistics file")
                
            try:
                write_MTEXdata(save_directory_, material_, material1_, rotation_matrix,\
                                   lattice_material, lattice_material1, lim_x, lim_y, mat_global,\
                                    input_params["symmetry"][0], input_params["symmetry"][1])
            except:
                print("Error writting MTEX orientation file")
                
            try:
                global_plots(lim_x, lim_y, rotation_matrix, strain_matrix, strain_matrixs, 
                              col, colx, coly, match_rate, mat_global, spots_len, 
                              iR_pix, fR_pix, save_directory_, material_, material1_,
                              match_rate_threshold=5, bins=30)
            except:
                print("Error in the global plots module") 
        
        
        
        
        
        
