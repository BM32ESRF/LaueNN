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
    import json
    
    ## Get the path of the lauetoolsnn library
    import lauetoolsnn
    laueNN_path = os.path.dirname(lauetoolsnn.__file__)
    print("LaueNN path is", laueNN_path)
    
    ## Load the json of material and extinctions
    with open(os.path.join(laueNN_path, 'lauetools\material.json'),'r') as f:
        dict_Materials = json.load(f)
    with open(os.path.join(laueNN_path, 'lauetools\extinction.json'),'r') as f:
        extinction_json = json.load(f)

    ## Modify the dictionary values to add new entries
    dict_Materials["GaN"] = ["GaN", [3.189, 3.189, 5.185, 90, 90, 120], "wurtzite"]
    dict_Materials["Si"] = ["Si", [5.4309, 5.4309, 5.4309, 90, 90, 90], "dia"]

    extinction_json["wurtzite"] = "wurtzite"
    extinction_json["dia"] = "dia"

    ## verify if extinction is present in CrystalParameters.py file of lauetools (Manually done for now)

    ## dump the json back with new values
    with open(os.path.join(laueNN_path, 'lauetools\material.json'), 'w') as fp:
        json.dump(dict_Materials, fp)
    with open(os.path.join(laueNN_path, 'lauetools\extinction.json'), 'w') as fp:
        json.dump(extinction_json, fp)

    ## Verify if the material is added to the library or not;
    from lauetoolsnn.lauetools.dict_LaueTools import dict_Materials
    ## if not, restart the console
    print(dict_Materials["GaN"])
    print(dict_Materials["Si"])
    
    # =============================================================================
    # Step 0: Define the dictionary with all parameters 
    # =============================================================================
    ## User Input dictionary with parameters
    ## In case of only one phase/material, keep same value for material_ and material1_ key
    input_params = {
                    "global_path" : r"C:\Users\purushot\Desktop\LaueNN_script",
                    "prefix" : "_MMrand",                 ## prefix for the folder to be created for training dataset

                    "material_": ["GaN", "Si"],             ## same key as used in dict_LaueTools
                    "symmetry": ["hexagonal", "cubic"],           ## crystal symmetry of material_
                    "SG": [191, 227], #186                    ## Space group of material_ (None if not known)
                    "hkl_max_identify" : [7,5],        ## Maximum hkl index to classify in a Laue pattern
                    "nb_grains_per_lp" : [2,1],        ## max grains to be generated in a Laue Image

                    ## hkl_max_identify : can be "auto" or integer: Maximum index of HKL to build output classes
                    
                    # =============================================================================
                    # ## Data generation settings
                    # =============================================================================
                    "grains_nb_simulate" : 500,    ## Number of orientations to generate (takes advantage of crystal symmetry)
                    "classes_with_frequency_to_remove": [100,100], ## classes_with_frequency_to_remove: HKL class with less appearance than 
                                                                            ##  specified will be ignored in output
                    "desired_classes_output": ["all","all"], ## desired_classes_output : can be all or an integer: to limit the number of output classes
                    "list_hkl_keep" : None, #[[(0,0,1)],[(0,0,0)]],
                    "maximum_angle_to_search":60, ## Angle of radial distribution to reconstruct the histogram (in deg)
                    "step_for_binning" : 0.1,      ## bin widht of angular radial distribution in degree
                    
                    # =============================================================================
                    #  ## Training parameters
                    # =============================================================================
                    "orientation_generation": "uniform", ## could be "uniform" or "random"
                    "batch_size":100,               ## batches of files to use while training
                    "epochs":10,                    ## number of epochs for training

                    # =============================================================================
                    # ## Detector parameters of the Experimental setup
                    # =============================================================================
                    ## Sample-detector distance, X center, Y center, two detector angles
                    "detectorparameters" :  [79.61200, 977.8100, 932.1700, 0.4770000, 0.4470000], 
                    "pixelsize" : 0.0734,          ## Detector pixel size
                    "dim1":2018,                   ## Dimensions of detector in pixels
                    "dim2":2016,
                    "emin" : 5,                    ## Minimum and maximum energy to use for simulating Laue Patterns
                    "emax" : 22,
                    "ccd_label" : "sCMOS",
                    
                    # =============================================================================
                    # ## Prediction parameters
                    # =============================================================================
                    "experimental_directory": r"D:\some_projects\GaN\BLC12834\NW1",
                    "experimental_prefix": r"nw1_",
                    "grid_size_x" : 61,            ## Grid X and Y limit to generate the simulated dataset (a rectangular scan region)
                    "grid_size_y" : 61,
                    
                    # =============================================================================
                    # ## Prediction Settings
                    # =============================================================================
                    # model_weight_file: if none, it will select by default the latest H5 weight file, else provide a specific model
                    # softmax_threshold_global: thresholding to limit the predicted spots search zone
                    # cap_matchrate: any UB matrix providing MR less than this will be ignored
                    # coeff: should be same as cap_matchrate or no? (this is for when use_previous is True)
                    # coeff_overlap: coefficient to limit the overlapping between spots; if more than this, new solution will be computed
                    # mode_spotCycle: How to cycle through predicted spots (slow or graphmode )
                    "UB_matrix_to_detect" : 3,
                    "matrix_tolerance" : [0.6, 0.6],
                    "material_limit" : [2, 1],
                    "material_phase_always_present" : [2,1,1],
                    "softmax_threshold_global" : 0.85,
                    "cap_matchrate" : 0.30,
                    "coeff_overlap" : 0.3,
                    "mode_spotCycle" : "graphmode",
                    ##true for few crystal and prefered texture case, otherwise time consuming; advised for single phase alone
                    "use_previous" : False,
                    
                    # =============================================================================
                    # # [PEAKSEARCH]
                    # =============================================================================
                    "intensity_threshold" : 1,
                    "boxsize" : 10,
                    "fit_peaks_gaussian" : 1,
                    "FitPixelDev" : 15,
                    "NumberMaxofFits" : 3000,
                    "mode": "skimage",

                    # =============================================================================
                    # # [STRAINCALCULATION]
                    # =============================================================================
                    "strain_compute" : True,
                    "tolerance_strain_refinement" : [[0.6,0.5,0.4,0.3,0.2,0.1],
                                                     [0.6,0.5,0.4,0.3,0.2,0.1]],
                    "free_parameters" : ["b","c","alpha","beta","gamma"],
                    
                    # =============================================================================
                    # # [Additional settings]
                    # =============================================================================
                    "residues_threshold":0.25,
                    "nb_spots_global_threshold":8,
                    "nb_spots_consider" : 500,
                    # User defined orientation matrix supplied in a file
                    "use_om_user" : False,
                    "path_user_OM" : "",
                    }

    generate_dataset_MM = False
    train_network_MM = False
    run_prediction_MM = True
    prediction_GUI = False
    
    # =============================================================================
    # END OF USER INPUT
    # =============================================================================
    # global_path: path where all model related files will be saved
    global_path = input_params["global_path"]
    
    if generate_dataset_MM:
        import os
        from tqdm import trange
        ## if LaueToolsNN is properly installed
        from lauetoolsnn.utils_lauenn import generate_classHKL, generate_multimat_dataset, \
                                        rmv_freq_class_MM, get_multimaterial_detail
                            
        material_= input_params["material_"]
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
        SG = input_params["SG"]
        
        if len(material_) > 1:
            prefix_mat = material_[0]
            for ino, imat in enumerate(material_):
                if ino == 0:
                    continue
                prefix_mat = prefix_mat + "_" + imat
        else:
            prefix_mat = material_[0]
        
        save_directory = os.path.join(global_path,prefix_mat+input_params["prefix"])

        print("save directory is : "+save_directory)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        ## get unit cell parameters and other details required for simulating Laue patterns
        rules, symmetry, lattice_material, \
                                crystal, SG = get_multimaterial_detail(material_, SG, symm_)
            
        ### generate_classHKL_multimat
        ## procedure for generation of GROUND TRUTH classes
        # general_diff_cond = True will eliminate the hkl index that does not satisfy the general reflection conditions
        # mat_listHKl: provide a numpy array of hkls to be added to the list of output hkl
        for ino in trange(len(material_)):
            generate_classHKL(n[ino], rules[ino], lattice_material[ino], \
                              symmetry[ino], material_[ino], \
                              crystal=crystal[ino], SG=SG[ino], general_diff_cond=False,
                              save_directory=save_directory, write_to_console=print, \
                              ang_maxx = maximum_angle_to_search, \
                              step = step_for_binning, mat_listHKl = None)
        
        # ## Generate Training and Testing dataset only for the output classes (Laue spot hkls) calculated in the Step 3
        # ### Uses multiprocessing library
        ############ GENERATING MULTI MATERIAL TRAINING DATA ##############
        # data_realism =True ; will introduce noise and partial Laue patterns in the training dataset
        # modelp can have either "random" for random orientation generation or "uniform" for uniform orientation generation
        # include_scm (if True; misorientation_angle parameter need to be defined): this parameter introduces misoriented crystal of 
        # specific angle along a crystal axis in the training dataset    
        generate_multimat_dataset(material_=material_, 
                                 ang_maxx=maximum_angle_to_search,
                                 step=step_for_binning, 
                                 nb_grains=nb_grains_per_lp, 
                                 grains_nb_simulate=grains_nb_simulate, 
                                 data_realism = True, 
                                 detectorparameters=detectorparameters, 
                                 pixelsize=pixelsize, 
                                 type_="training_data",
                                 var0 = 1, 
                                 dim1=input_params["dim1"], 
                                 dim2=input_params["dim2"], 
                                 removeharmonics=1, 
                                 save_directory=save_directory,
                                 write_to_console=print, 
                                 emin=emin, 
                                 emax=emax, 
                                 modelp = input_params["orientation_generation"],
                                 general_diff_rules = False, 
                                 crystal = crystal,)
        
        ############ GENERATING TESTING DATA ##############
        factor = 5 # validation split for the training dataset  --> corresponds to 20% of total training dataset
        generate_multimat_dataset(material_=material_, 
                                 ang_maxx=maximum_angle_to_search,
                                 step=step_for_binning, 
                                 nb_grains=nb_grains_per_lp, 
                                 grains_nb_simulate=grains_nb_simulate//factor, 
                                 data_realism = True, 
                                 detectorparameters=detectorparameters, 
                                 pixelsize=pixelsize, 
                                 type_="testing_data",
                                 var0 = 1, 
                                 dim1=input_params["dim1"], 
                                 dim2=input_params["dim2"], 
                                 removeharmonics=1, 
                                 save_directory=save_directory,
                                 write_to_console=print, 
                                 emin=emin, 
                                 emax=emax, 
                                 modelp = input_params["orientation_generation"],
                                 general_diff_rules = False, 
                                 crystal = crystal,)
        ### Updating the ClassHKL list by removing the non-common HKL or less frequent HKL from the list
        ## The non-common HKL can occur as a result of the detector position and energy used
        # freq_rmv: remove output hkl if the training dataset has less tha 100 occurances of the considered hkl (freq_rmv1 for second phase)
        # Weights (penalty during training) are also calculated based on the occurance

        freq_rmv = input_params["classes_with_frequency_to_remove"]
        elements = input_params["desired_classes_output"]
        list_hkl_keep = input_params["list_hkl_keep"]
        
        rmv_freq_class_MM(freq_rmv = freq_rmv, elements = elements,
                          save_directory = save_directory, material_ = material_,
                          write_to_console = print, progress=None, qapp=None,
                          list_hkl_keep = list_hkl_keep)
        ## End of data generation for Neural network training: all files are saved in the same folder 
        ## to be later used for training and prediction
    
    if train_network_MM:
        import os
        import numpy as np
        import _pickle as cPickle
        import itertools
        from keras.callbacks import EarlyStopping, ModelCheckpoint
        import matplotlib.pyplot as plt
        ## if LaueToolsNN is properly installed
        from lauetoolsnn.utils_lauenn import array_generator, array_generator_verify, vali_array
        from lauetoolsnn.NNmodels import model_arch_general, LoggingCallback
        
        material_= input_params["material_"]
        epochs = input_params["epochs"]
        batch_size = input_params["batch_size"] 
        # ### number of files it will generate fro training
        nb_grains_list = []
        for ino, imat in enumerate(material_):
            nb_grains_list.append(list(range(input_params["nb_grains_per_lp"][ino]+1)))
        list_permute = list(itertools.product(*nb_grains_list))
        list_permute.pop(0)
        print(len(list_permute)*input_params["grains_nb_simulate"])
        # ## Get material parameters 
        # ### Generates a folder with material name and gets material unit cell parameters and symmetry object 
        # from the get_material_detail function        
        if len(material_) > 1:
            prefix_mat = material_[0]
            for ino, imat in enumerate(material_):
                if ino == 0:
                    continue
                prefix_mat = prefix_mat + "_" + imat
        else:
            prefix_mat = material_[0]
        
        save_directory = os.path.join(global_path,prefix_mat+input_params["prefix"])

        print("save directory is : "+save_directory)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        classhkl = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_0"]
        angbins = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_1"]
        loc_new = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_2"]
        with open(save_directory+"//class_weights.pickle", "rb") as input_file:
            class_weights = cPickle.load(input_file)
        class_weights = class_weights[0]
        
        # ##  Training
        # model save directory and filename
        if len(material_) > 1:
            prefix_mat = material_[0]
            for ino, imat in enumerate(material_):
                if ino == 0:
                    continue
                prefix_mat = prefix_mat + "_" + imat
        else:
            prefix_mat = material_[0]
            
        model_name = os.path.join(save_directory,"model_"+prefix_mat)
        
        # Define model and train
        model = model_arch_general( len(angbins)-1, len(classhkl),
                                    kernel_coeff = 1e-5,
                                    bias_coeff = 1e-6,
                                    lr = 1e-3,)

        # Save model config and weights
        model_json = model.to_json()
        with open(model_name+".json", "w") as json_file:
            json_file.write(model_json)  
    
        ## temp function to quantify the spots and classes present in a batch
        trainy_inbatch = array_generator_verify(save_directory+"//training_data", batch_size, 
                                                len(classhkl), loc_new, print)
        print("Number of spots in a batch of %i files : %i" %(batch_size, len(trainy_inbatch)))
        print("Min, Max class ID is %i, %i" %(np.min(trainy_inbatch), np.max(trainy_inbatch)))
        
        ## Batch loading for numpy grain files (Keep low value to avoid overcharging the RAM)
        nb_grains_list = []
        for ino, imat in enumerate(material_):
            nb_grains_list.append(list(range(input_params["nb_grains_per_lp"][ino]+1)))
        list_permute = list(itertools.product(*nb_grains_list))
        list_permute.pop(0)
        steps_per_epoch = len(list_permute)*(input_params["grains_nb_simulate"])//batch_size        
        val_steps_per_epoch = int(steps_per_epoch / 5)
        if steps_per_epoch == 0:
            steps_per_epoch = 1
        if val_steps_per_epoch == 0:
            val_steps_per_epoch = 1 
            
        ## Load generator objects from filepaths (iterators for Training and Testing datasets)
        training_data_generator = array_generator(save_directory+"//training_data", batch_size,                                          
                                                  len(classhkl), loc_new, print)
        testing_data_generator = array_generator(save_directory+"//testing_data", batch_size,                                           
                                                 len(classhkl), loc_new, print)
        
        
        ######### TRAIN THE DATA
        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=2)
        ms = ModelCheckpoint(save_directory+"//best_val_acc_model.h5", monitor='val_accuracy', 
                              mode='max', save_best_only=True)
        lc = LoggingCallback(None, None, None, model, model_name)
        ## Fitting function
        stats_model = model.fit(
                                training_data_generator, 
                                epochs=epochs, 
                                steps_per_epoch=steps_per_epoch,
                                validation_data=testing_data_generator,
                                validation_steps=val_steps_per_epoch,
                                verbose=1,
                                class_weight=class_weights,
                                callbacks=[es, ms, lc]
                                )          
        # serialize weights to HDF5
        model.save_weights(model_name+".h5")
        print("Saved model to disk")
        
        print( "Training Accuracy: "+str( stats_model.history['accuracy'][-1]))
        print( "Training Loss: "+str( stats_model.history['loss'][-1]))
        print( "Validation Accuracy: "+str( stats_model.history['val_accuracy'][-1]))
        print( "Validation Loss: "+str( stats_model.history['val_loss'][-1]))
        
        # Plot the accuracy/loss v Epochs
        epochs = range(1, len(model.history.history['loss']) + 1)
        fig, ax = plt.subplots(1,2)
        ax[0].plot(epochs, model.history.history['loss'], 'r', label='Training loss')
        ax[0].plot(epochs, model.history.history['val_loss'], 'r', ls="dashed", label='Validation loss')
        ax[0].legend()
        ax[1].plot(epochs, model.history.history['accuracy'], 'g', label='Training Accuracy')
        ax[1].plot(epochs, model.history.history['val_accuracy'], 'g', ls="dashed", label='Validation Accuracy')
        ax[1].legend()
        plt.savefig(save_directory+"//loss_accuracy_"+prefix_mat+".png", bbox_inches='tight',format='png', dpi=1000)
        plt.close()
        
        text_file = open(save_directory+"//loss_accuracy_logger_"+prefix_mat+".txt", "w")
        text_file.write("# EPOCH, LOSS, VAL_LOSS, ACCURACY, VAL_ACCURACY" + "\n")
        for inj in range(len(epochs)):
            string1 = str(epochs[inj]) + ","+ str(model.history.history['loss'][inj])+\
                            ","+str(model.history.history['val_loss'][inj])+","+str(model.history.history['accuracy'][inj])+\
                            ","+str(model.history.history['val_accuracy'][inj])+" \n"  
            text_file.write(string1)
        text_file.close() 

        # ## Stats on the trained model with sklearn metrics
        from sklearn.metrics import classification_report
        ## verify the statistics
        x_test, y_test = vali_array(save_directory+"//testing_data", 50, len(classhkl), loc_new, print)
        y_test = np.argmax(y_test, axis=-1)
        y_pred = np.argmax(model.predict(x_test), axis=-1)
        print(classification_report(y_test, y_pred))
        
    if run_prediction_MM:
        ## Import modules used for this Notebook
        import numpy as np
        import os
        import multiprocessing
        import time, datetime
        import glob, re
        import configparser
        from itertools import accumulate
        ## if LaueToolsNN is properly installed
        from lauetoolsnn.utils_lauenn import get_multimaterial_detail, new_MP_multimat_function, resource_path, global_plots_MM
        from lauetoolsnn.lauetools import dict_LaueTools as dictLT
        from lauetoolsnn.NNmodels import read_hdf5
        
        import _pickle as cPickle
        from tqdm import tqdm
        ncpu = multiprocessing.cpu_count()
        
        material_= input_params["material_"]
        detectorparameters = input_params["detectorparameters"]
        pixelsize = input_params["pixelsize"]
        emax = input_params["emax"]
        emin = input_params["emin"]
        dim1 = input_params["dim1"]
        dim2 = input_params["dim2"]
        symm_ = input_params["symmetry"]
        SG = input_params["SG"]
        tolerance = input_params["matrix_tolerance"]
        tolerance_strain = input_params["tolerance_strain_refinement"]
        strain_free_parameters = input_params["free_parameters"]
        material_limit = input_params["material_limit"]
        material_phase_always_present = input_params["material_phase_always_present"]
        model_annote = "DNN"

        ## Requirements
        ## Experimental peak search parameters in case of RAW LAUE PATTERNS from detector
        intensity_threshold = input_params["intensity_threshold"]
        boxsize = input_params["boxsize"]
        fit_peaks_gaussian = input_params["fit_peaks_gaussian"]
        FitPixelDev = input_params["FitPixelDev"]
        NumberMaxofFits = input_params["NumberMaxofFits"]
        bkg_treatment = "A-B"
        mode_peaksearch = input_params["mode"]
        
        ubmat = input_params["UB_matrix_to_detect"] # How many orientation matrix to detect per Laue pattern
        mode_spotCycle = input_params["mode_spotCycle"] ## mode of calculation
        use_previous_UBmatrix_name = input_params["use_previous"] ## Try previous indexation solutions to speed up the process
        strain_calculation = input_params["strain_compute"] ## Strain refinement is required or not
        ccd_label_global = input_params["ccd_label"]
        
        ## Parameters to control the orientation matrix indexation
        softmax_threshold_global = input_params["softmax_threshold_global"] # softmax_threshold of the Neural network to consider
        mr_threshold_global = 0.90 # match rate threshold to accept a solution immediately
        cap_matchrate = input_params["cap_matchrate"] * 100 ## any UB matrix providing MR less than this will be ignored
        coeff = 0.30            ## coefficient to calculate the overlap of two solutions
        coeff_overlap = input_params["coeff_overlap"]    ##10% spots overlap is allowed with already indexed orientation

        ## Additional parameters to refine the orientation matrix construction process
        use_om_user = str(input_params["use_om_user"]).lower()
        path_user_OM = input_params["path_user_OM"]
        nb_spots_consider = input_params["nb_spots_consider"]
        residues_threshold = input_params["residues_threshold"]
        nb_spots_global_threshold = input_params["nb_spots_global_threshold"]
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

        if len(material_) > 1:
            prefix_mat = material_[0]
            for ino, imat in enumerate(material_):
                if ino == 0:
                    continue
                prefix_mat = prefix_mat + "_" + imat
        else:
            prefix_mat = material_[0]
        
        model_direc = os.path.join(global_path,prefix_mat+input_params["prefix"])
        
        if not os.path.exists(model_direc):
            print("The directory doesn't exists; please veify the path")
        else:
            print("Directory where trained model is stored : "+model_direc)
            
        ## get unit cell parameters and other details required for simulating Laue patterns
        rules, symmetry, lattice_material, \
                            crystal, SG = get_multimaterial_detail(material_, SG, symm_)

        filenameDirec = input_params["experimental_directory"]
        experimental_prefix = input_params["experimental_prefix"]
        lim_x, lim_y = input_params["grid_size_x"], input_params["grid_size_y"] 
        format_file = dictLT.dict_CCD[ccd_label_global][7]
        
        hkl_all_class0 = []
        for ino, imat in enumerate(material_):
            with open(model_direc+"//classhkl_data_nonpickled_"+imat+".pickle", "rb") as input_file:
                hkl_all_class_load = cPickle.load(input_file)[0]
            hkl_all_class0.append(hkl_all_class_load)
            
        ## load model related files and generate the model
        classhkl = np.load(model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_0"]
        angbins = np.load(model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_1"]
        
        # from two phase training dataset FIX
        # if len(material_) <= 2: 
        #     ind_mat0 = np.load(model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_5"]
        #     ind_mat1 = np.load(model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_6"] 
        #     ind_mat = [ind_mat0, ind_mat0+ind_mat1]
        # else:
        ##Below lines are for dataset generated with multimat code
        ind_mat_all = np.load(model_direc+"//MOD_grain_classhkl_angbin.npz",allow_pickle=True)["arr_5"]
        ind_mat = []
        for inni in ind_mat_all:
            ind_mat.append(len(inni))
        ind_mat = [int(item) for item in accumulate(ind_mat)]
        
        # json_file = open(model_direc+"//model_"+prefix_mat+".json", 'r')
        load_weights = model_direc + "//model_"+prefix_mat+".h5"
        wb = read_hdf5(load_weights)
        temp_key = list(wb.keys())
        
        ct = time.time()
        now = datetime.datetime.fromtimestamp(ct)
        c_time = now.strftime("%Y-%m-%d_%H-%M-%S")   
        
        
        if prediction_GUI:
            from lauetoolsnn.GUI_multi_mat_LaueNN import start
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
                load_weights,
                model_annote
                )
        
        else:
            ## Step 3: Initialize variables and prepare arguments for multiprocessing module
            
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
    
            #% Launch multiprocessing prediction     
            args = zip(valu12)
            with multiprocessing.Pool(ncpu) as pool:
                results = pool.starmap(new_MP_multimat_function, tqdm(args, total=len(valu12)), chunksize=1)
                
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
            save_directory_ = model_direc+"//results_"+prefix_mat+"_"+c_time
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
        
