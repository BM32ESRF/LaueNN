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
                    "global_path" : r"C:\Users\purushot\Desktop\python_animation\mayavi_version\LaueNN_model",
                    "prefix" : "",                 ## prefix for the folder to be created for training dataset

                    "material_": "Cu",             ## same key as used in dict_LaueTools
                    "symmetry": "cubic",           ## crystal symmetry of material_
                    "SG": 227,                     ## Space group of material_ (None if not known)
                    "hkl_max_identify" : 5,        ## Maximum hkl index to classify in a Laue pattern
                    "nb_grains_per_lp_mat0" : 1,        ## max grains to be generated in a Laue Image

                    "material1_": "Cu",            ## same key as used in dict_LaueTools
                    "symmetry1": "cubic",          ## crystal symmetry of material1_
                    "SG1": 227,                    ## Space group of material1_ (None if not known)
                    "hkl_max_identify1" : 5,        ## Maximum hkl index to classify in a Laue pattern
                    "nb_grains_per_lp_mat1" : 1,        ## max grains to be generated in a Laue Image
                    ## hkl_max_identify : can be "auto" or integer: Maximum index of HKL to build output classes
                    
                    # =============================================================================
                    # ## Data generation settings
                    # =============================================================================
                    "grains_nb_simulate" : 500,    ## Number of orientations to generate (takes advantage of crystal symmetry)
                    "classes_with_frequency_to_remove": 100, ## classes_with_frequency_to_remove: HKL class with less appearance than 
                                                                            ##  specified will be ignored in output
                    "desired_classes_output": "all", ## desired_classes_output : can be all or an integer: to limit the number of output classes
                    "classes_with_frequency_to_remove1": 100,## classes_with_frequency_to_remove: HKL class with less appearance than 
                                                                            ##  specified will be ignored in output
                    "desired_classes_output1": "all", ## desired_classes_output : can be all or an integer: to limit the number of output classes
                    "include_small_misorientation": False, ## to include additional data with small angle misorientation
                    "misorientation": 5, ##only used if "include_small_misorientation" is True
                    "maximum_angle_to_search":20, ## Angle of radial distribution to reconstruct the histogram (in deg)
                    "step_for_binning" : 0.1,      ## bin widht of angular radial distribution in degree
                    
                    # =============================================================================
                    #  ## Training parameters
                    # =============================================================================
                    "orientation_generation": "uniform", ## can be random or uniform
                    "batch_size":50,               ## batches of files to use while training
                    "epochs":20,                    ## number of epochs for training

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
                    # coeff: should be same as cap_matchrate or no? (this is for try previous UB matrix)
                    # coeff_overlap: coefficient to limit the overlapping between spots; if more than this, new solution will be computed
                    # mode_spotCycle: How to cycle through predicted spots (slow or graphmode )
                    "UB_matrix_to_detect" : 3,
                    "matrix_tolerance" : 0.6,
                    "matrix_tolerance1" : 0.6,
                    "material0_limit" : 2,
                    "material1_limit" : 1,
                    "material_phase_always_present" : 2,
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
                    "tolerance_strain_refinement" : [0.6,0.5,0.4,0.3,0.2,0.1],
                    "tolerance_strain_refinement1" : [0.6,0.5,0.4,0.3,0.2,0.1],
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

    generate_dataset = True
    train_network = True
    write_config_GUI = False
    run_prediction = False
    prediction_GUI = False
    # =============================================================================
    # END OF USER INPUT
    # =============================================================================
    # global_path: path where all model related files will be saved
    global_path = input_params["global_path"]
    
    if write_config_GUI:
        ## write config file for GUI 
        if input_params["material_"] != input_params["material1_"]:
            text_file = open(os.path.join(global_path,input_params["material_"]+"_"+input_params["material1_"]+input_params["prefix"]+".lauenn"), "w")
        else:
            text_file = open(os.path.join(global_path,input_params["material_"]+"_"+input_params["prefix"]+".lauenn"), "w")

        text_file.write("### config file for LaueNeuralNetwork GUI \n")
        text_file.write("[GLOBAL_DIRECTORY]\n")
        text_file.write("prefix = "+input_params["prefix"]+" \n")
        text_file.write("main_directory = "+global_path+"\n")
        text_file.write("\n")
        text_file.write("[MATERIAL]\n")
        text_file.write("material = "+input_params["material_"]+"\n")
        text_file.write("symmetry = "+input_params["symmetry"]+"\n")
        text_file.write("space_group = "+str(input_params["SG"])+"\n")
        text_file.write("general_diffraction_rules = false\n")
        text_file.write("\n")
        text_file.write("material1 = "+input_params["material1_"]+"\n")
        text_file.write("symmetry1 = "+input_params["symmetry1"]+"\n")
        text_file.write("space_group1 = "+str(input_params["SG1"])+"\n")
        text_file.write("general_diffraction_rules1 = false\n")
        text_file.write("\n")
        text_file.write("[DETECTOR]\n")
        text_file.write("detectorfile = user_input"+" \n")
        text_file.write("params ="+ ",".join(str(param) for param in input_params["detectorparameters"])+","+str(input_params["pixelsize"])+","+str(input_params["dim1"])+","+str(input_params["dim2"])+","+str(input_params["ccd_label"])+" \n")
        text_file.write("emax = "+str(input_params["emax"])+"\n")
        text_file.write("emin = "+str(input_params["emin"])+"\n")
        text_file.write("\n")
        text_file.write("[TRAINING]\n")
        text_file.write("classes_with_frequency_to_remove = "+str(input_params["classes_with_frequency_to_remove"])+"\n")
        text_file.write("desired_classes_output = "+str(input_params["desired_classes_output"])+"\n")
        text_file.write("max_HKL_index = "+str(input_params["hkl_max_identify"])+"\n")
        text_file.write("max_nb_grains = "+str(input_params["nb_grains_per_lp_mat0"])+"\n")
        text_file.write("classes_with_frequency_to_remove1 = "+str(input_params["classes_with_frequency_to_remove1"])+"\n")
        text_file.write("desired_classes_output1 = "+str(input_params["desired_classes_output1"])+"\n")
        text_file.write("max_HKL_index1 = "+str(input_params["hkl_max_identify1"])+"\n")
        text_file.write("max_nb_grains1 = "+str(input_params["nb_grains_per_lp_mat1"])+"\n")
        text_file.write("max_simulations = "+str(input_params["grains_nb_simulate"])+"\n")
        text_file.write("include_small_misorientation = "+str(input_params["include_small_misorientation"]).lower()+"\n")
        text_file.write("misorientation_angle = 1 \n")
        text_file.write("angular_distance = "+str(input_params["maximum_angle_to_search"])+"\n")
        text_file.write("step_size = "+str(input_params["step_for_binning"])+"\n")
        text_file.write("batch_size = "+str(input_params["batch_size"])+"\n")
        text_file.write("epochs = "+str(input_params["epochs"])+"\n")
        text_file.write("\n")
        text_file.write("[PREDICTION]\n")
        text_file.write("UB_matrix_to_detect = "+str(input_params["UB_matrix_to_detect"])+"\n")
        text_file.write("matrix_tolerance = "+str(input_params["matrix_tolerance"])+"\n")
        text_file.write("matrix_tolerance1 = "+str(input_params["matrix_tolerance1"])+"\n")
        text_file.write("material0_limit = "+str(input_params["material0_limit"])+"\n")
        text_file.write("material1_limit = "+str(input_params["material1_limit"])+"\n")
        text_file.write("softmax_threshold_global = "+str(input_params["softmax_threshold_global"])+"\n")
        text_file.write("cap_matchrate = "+str(input_params["cap_matchrate"])+"\n")
        text_file.write("coeff = 0.3\n")
        text_file.write("coeff_overlap = "+str(input_params["coeff_overlap"])+"\n")
        text_file.write("mode_spotCycle = "+str(input_params["mode_spotCycle"])+"\n")
        text_file.write("use_previous = "+str(input_params["use_previous"]).lower()+"\n")
        text_file.write("\n")
        text_file.write("[EXPERIMENT]\n")
        text_file.write("experiment_directory = "+input_params["experimental_directory"]+"\n")
        text_file.write("experiment_file_prefix = "+input_params["experimental_prefix"]+"\n")
        text_file.write("image_grid_x = "+str(input_params["grid_size_x"])+"\n")
        text_file.write("image_grid_y = "+str(input_params["grid_size_y"])+"\n")
        text_file.write("\n")
        text_file.write("[PEAKSEARCH]\n")
        text_file.write("intensity_threshold = "+str(input_params["intensity_threshold"])+"\n")
        text_file.write("boxsize = "+str(input_params["boxsize"])+"\n")
        text_file.write("fit_peaks_gaussian = "+str(input_params["fit_peaks_gaussian"])+"\n")
        text_file.write("FitPixelDev = "+str(input_params["FitPixelDev"])+"\n")
        text_file.write("NumberMaxofFits = "+str(input_params["NumberMaxofFits"])+"\n")
        text_file.write("mode = "+str(input_params["mode"])+"\n")
        text_file.write("\n")
        text_file.write("[STRAINCALCULATION]\n")
        text_file.write("strain_compute = "+str(input_params["strain_compute"]).lower()+"\n")
        text_file.write("tolerance_strain_refinement ="+ ",".join(str(param) for param in input_params["tolerance_strain_refinement"])+"\n")
        text_file.write("tolerance_strain_refinement1 ="+ ",".join(str(param) for param in input_params["tolerance_strain_refinement1"])+"\n")
        text_file.write("free_parameters ="+ ",".join(str(param) for param in input_params["free_parameters"])+"\n")
        text_file.write("\n")
        text_file.write("[DEVELOPMENT]\n")
        text_file.write("material_phase_always_present="+str(input_params["material_phase_always_present"])+"\n")
        text_file.write("write_MTEX_file= true \n")
        text_file.write("[CALLER]\n")
        text_file.write("residues_threshold="+str(input_params["residues_threshold"])+"\n")
        text_file.write("nb_spots_global_threshold="+str(input_params["nb_spots_global_threshold"])+"\n")
        text_file.write("option_global = v2 \n")
        text_file.write("nb_spots_consider = "+str(input_params["nb_spots_consider"])+"\n")
        text_file.write("use_om_user = "+str(input_params["use_om_user"]).lower()+"\n")
        text_file.write("path_user_OM = "+str(input_params["path_user_OM"])+"\n")
        text_file.close()
    
    if generate_dataset:
        # =============================================================================
        # Step 1 
        # =============================================================================
        ## if LaueToolsNN is properly installed
        from lauetoolsnn.utils_lauenn import get_material_detail, generate_classHKL, generate_dataset, rmv_freq_class
        '''
        get_material_detail : Extract material parameters
        generate_classHKL : Generates List of output HKL for the neural network
        generate_dataset : Will simulate laue patterns and will build histogram for the hkl generated by classHKL
        rmv_freq_class : cleans the output class dataset
        '''
        # ## Get material parameters 
        # ### Generates a folder with material name and gets material unit cell parameters 
        # ### and symmetry object from the get_material_detail function
        material_= input_params["material_"]
        material1_= input_params["material1_"]
        n = input_params["hkl_max_identify"]
        n1 = input_params["hkl_max_identify1"]
        maximum_angle_to_search = input_params["maximum_angle_to_search"]
        step_for_binning = input_params["step_for_binning"]
        nb_grains_per_lp0 = input_params["nb_grains_per_lp_mat0"]
        nb_grains_per_lp1 = input_params["nb_grains_per_lp_mat1"]
        grains_nb_simulate = input_params["grains_nb_simulate"]
        detectorparameters = input_params["detectorparameters"]
        pixelsize = input_params["pixelsize"]
        emax = input_params["emax"]
        emin = input_params["emin"]
        symm_ = input_params["symmetry"]
        symm1_ = input_params["symmetry1"]
        SG = input_params["SG"]
        SG1 = input_params["SG1"]
        
        ## read hkl information from a fit file in case too large HKLs
        manual_hkl_list=False
        if manual_hkl_list:
            import numpy as np
            temp = np.loadtxt(r"img_0000_LT_1.fit")
            hkl_array = temp[:,2:5]
            hkl_array1 = None
        else:
            hkl_array = None
            hkl_array1 = None
            
        if material_ != material1_:
            save_directory = os.path.join(global_path,material_+"_"+material1_+input_params["prefix"])
        else:
            save_directory = os.path.join(global_path,material_+input_params["prefix"])
        print("save directory is : "+save_directory)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        ## get unit cell parameters and other details required for simulating Laue patterns
        rules, symmetry, lattice_material,\
            crystal, SG, rules1, symmetry1,\
            lattice_material1, crystal1, SG1 = get_material_detail(material_, SG, symm_,
                                                                   material1_, SG1, symm1_)
        
        # ## Generate Neural network output classes (Laue spot hkls) using the generate_classHKL function
        ## procedure for generation of GROUND TRUTH classes
        # general_diff_cond = True will eliminate the hkl index that does not 
        # satisfy the general reflection conditions, otherwise they will be eliminated in the next stage
        generate_classHKL(n, rules, lattice_material, symmetry, material_, 
                          crystal=crystal, SG=SG, general_diff_cond=False,
                          save_directory=save_directory, write_to_console=print, 
                          ang_maxx = maximum_angle_to_search, 
                          step = step_for_binning, mat_listHKl=hkl_array)
        
        if material_ != material1_:
            generate_classHKL(n1, rules1, lattice_material1, symmetry1, material1_, 
                              crystal=crystal1, SG=SG1, general_diff_cond=False,
                              save_directory=save_directory, write_to_console=print, 
                              ang_maxx = maximum_angle_to_search, 
                              step = step_for_binning, mat_listHKl=hkl_array1)
        
        
        # ## Generate Training and Testing dataset only for the output classes (Laue spot hkls) calculated in the Step 3
        # ### Uses multiprocessing library
        ############ GENERATING TRAINING DATA ##############
        # data_realism =True ; will introduce noise and partial Laue patterns in the training dataset
        # modelp can have either "random" for random orientation generation or 
        # "uniform" for uniform orientation generation
        # include_scm (if True; misorientation_angle parameter need to be defined): 
        #this parameter introduces misoriented crystal of specific angle along a crystal axis in the training dataset
        generate_dataset(material_=material_, material1_=material1_, ang_maxx=maximum_angle_to_search,
                         step=step_for_binning, mode=0, 
                         nb_grains=nb_grains_per_lp0, nb_grains1=nb_grains_per_lp1, 
                         grains_nb_simulate=grains_nb_simulate, data_realism = True, 
                         detectorparameters=detectorparameters, pixelsize=pixelsize, 
                         type_="training_data",
                         var0 = 1, dim1=input_params["dim1"], dim2=input_params["dim2"], 
                         removeharmonics=1, save_directory=save_directory,
                         write_to_console=print, 
                         emin=emin, emax=emax, 
                         modelp = input_params["orientation_generation"],
                         misorientation_angle = input_params["misorientation"], 
                         general_diff_rules = False, 
                         crystal = crystal, crystal1 = crystal1, 
                         include_scm=input_params["include_small_misorientation"],
                         mat_listHKl=hkl_array, mat_listHKl1=hkl_array1)
        
        ############ GENERATING TESTING DATA ##############
        factor = 5 # validation split for the training dataset  --> corresponds to 20% of total training dataset
        generate_dataset(material_=material_, material1_=material1_, ang_maxx=maximum_angle_to_search,
                         step=step_for_binning, mode=0, 
                         nb_grains=nb_grains_per_lp0, nb_grains1=nb_grains_per_lp1, 
                         grains_nb_simulate=grains_nb_simulate//factor, data_realism = True, 
                         detectorparameters=detectorparameters, pixelsize=pixelsize, 
                         type_="testing_data",
                         var0 = 1, dim1=input_params["dim1"], dim2=input_params["dim2"], 
                         removeharmonics=1, save_directory=save_directory,
                         write_to_console=print, 
                         emin=emin, emax=emax, 
                         modelp = input_params["orientation_generation"],
                         misorientation_angle = input_params["misorientation"], 
                         general_diff_rules = False, 
                         crystal = crystal, crystal1 = crystal1, 
                         include_scm=input_params["include_small_misorientation"],
                         mat_listHKl=hkl_array, mat_listHKl1=hkl_array1)
        
        ## Updating the ClassHKL list by removing the non-common HKL or less frequent HKL from the list
        ## The non-common HKL can occur as a result of the detector position and energy used
        # freq_rmv: remove output hkl if the training dataset has less tha 100 occurances of 
        # the considered hkl (freq_rmv1 for second phase)
        # Weights (penalty during training) are also calculated based on the occurance
        rmv_freq_class(freq_rmv = input_params["classes_with_frequency_to_remove"], 
                       freq_rmv1 = input_params["classes_with_frequency_to_remove1"], 
                       save_directory=save_directory, 
                       material_=material_, 
                       material1_=material1_, 
                       write_to_console=print,
                       list_hkl_keep=None, list_hkl_keep1=None)
        
        ## End of data generation for Neural network training: all files are saved in the 
        ## same folder to be later used for training and prediction

    if train_network:
        import numpy as np
        import os
        import _pickle as cPickle
        import itertools
        from keras.callbacks import EarlyStopping, ModelCheckpoint
        import matplotlib.pyplot as plt
        ## if LaueToolsNN is properly installed
        from lauetoolsnn.utils_lauenn import array_generator, array_generator_verify, vali_array
        # ## Defining a neural network architecture or load a predefined one from NNmodels
        from lauetoolsnn.NNmodels import LoggingCallback
        
        import tensorflow as tf
        import keras
        from keras.models import Sequential
        from keras.layers import Dense, Activation, Dropout
        from keras.regularizers import l2
        from keras.models import model_from_json
        metricsNN = [
                    keras.metrics.FalseNegatives(name="fn"),
                    keras.metrics.FalsePositives(name="fp"),
                    keras.metrics.TrueNegatives(name="tn"),
                    keras.metrics.TruePositives(name="tp"),
                    keras.metrics.Precision(name="precision"),
                    keras.metrics.Recall(name="accuracy"),
                    ]
        def model_arch_general(n_bins, n_outputs, kernel_coeff = 0.0005, bias_coeff = 0.0005, lr=None, verbose=1,
                               write_to_console=None):
            model = Sequential()
            model.add(keras.Input(shape=(n_bins,)))
            ## Hidden layer 1
            model.add(Dense(150, kernel_regularizer=l2(kernel_coeff), bias_regularizer=l2(bias_coeff)))
            model.add(Activation('relu'))
            model.add(Dropout(0.3)) ## Adding dropout as we introduce some uncertain data with noise
            ## Hidden layer 2
            model.add(Dense(100, kernel_regularizer=l2(kernel_coeff), bias_regularizer=l2(bias_coeff)))
            model.add(Activation('relu'))
            model.add(Dropout(0.3))
            ## Hidden layer 3
            model.add(Dense(50, kernel_regularizer=l2(kernel_coeff), bias_regularizer=l2(bias_coeff)))
            model.add(Activation('relu'))
            model.add(Dropout(0.3))
            ## Output layer 
            model.add(Dense(n_outputs, activation='softmax'))
            ## Compile model
            otp = tf.keras.optimizers.Adam(learning_rate=lr)
            model.compile(loss='categorical_crossentropy', optimizer=otp, metrics=[metricsNN])
            if verbose == 1:
                model.summary()
                stringlist = []
                model.summary(print_fn=lambda x: stringlist.append(x))
                short_model_summary = "\n".join(stringlist)
                if write_to_console!=None:
                    write_to_console(short_model_summary)
            return model
        
        material_= input_params["material_"]
        material1_= input_params["material1_"]
        nb_grains_per_lp = input_params["nb_grains_per_lp_mat0"]
        nb_grains_per_lp1 = input_params["nb_grains_per_lp_mat1"]
        grains_nb_simulate = input_params["grains_nb_simulate"]
        
        if material_ != material1_:
            save_directory = os.path.join(global_path,material_+"_"+material1_+input_params["prefix"])
        else:
            save_directory = os.path.join(global_path,material_+input_params["prefix"])
        
        if not os.path.exists(save_directory):
            print("The directory doesn't exists; please veify the path")
            pass
        else:
            print("Directory where training dataset is stored is : "+save_directory)
        
        # ## Load the necessary files generated in Step 1 script
        # ### Loading the Output class and ground truth        
        classhkl = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_0"]
        angbins = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_1"]
        loc_new = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_2"]
        with open(save_directory+"//class_weights.pickle", "rb") as input_file:
            class_weights = cPickle.load(input_file)
        class_weights = class_weights[0]
        
        # ## Training  
        # load model and train
        model = model_arch_general(len(angbins)-1, len(classhkl),
                                    kernel_coeff = 1e-5,
                                    bias_coeff = 1e-6,
                                    lr = 1e-3)
        
        ## temp function to quantify the spots and classes present in a batch
        batch_size = input_params["batch_size"] 
        trainy_inbatch = array_generator_verify(save_directory+"//training_data", 
                                                batch_size, 
                                                len(classhkl), 
                                                loc_new, 
                                                print)
        print("Number of spots in a batch of %i files : %i" %(batch_size, len(trainy_inbatch)))
        print("Min, Max class ID is %i, %i" %(np.min(trainy_inbatch), np.max(trainy_inbatch)))
        
        epochs = input_params["epochs"] 
        ## Batch loading for numpy grain files (Keep low value to avoid overcharging the RAM)
        if material_ != material1_:
            nb_grains_list = list(range(nb_grains_per_lp+1))
            nb_grains1_list = list(range(nb_grains_per_lp1+1))
            list_permute = list(itertools.product(nb_grains_list, nb_grains1_list))
            list_permute.pop(0)
            steps_per_epoch = (len(list_permute) * grains_nb_simulate)//batch_size
        else:
            steps_per_epoch = int((nb_grains_per_lp * grains_nb_simulate) / batch_size)
        
        val_steps_per_epoch = int(steps_per_epoch / 5)
        if steps_per_epoch == 0:
            steps_per_epoch = 1
        if val_steps_per_epoch == 0:
            val_steps_per_epoch = 1 

        ## Load generator objects from filepaths (iterators for Training and Testing datasets)
        training_data_generator = array_generator(save_directory+"//training_data", batch_size,                                           len(classhkl), loc_new, print)
        testing_data_generator = array_generator(save_directory+"//testing_data", batch_size,                                           len(classhkl), loc_new, print)
        
        # model save directory and filename
        if material_ != material1_:
            model_name = save_directory+"//model_"+material_+"_"+material1_
        else:
            model_name = save_directory+"//model_"+material_
            
        ######### TRAIN THE DATA
        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=2)
        ms = ModelCheckpoint(save_directory+"//best_val_acc_model.h5", 
                             monitor='val_accuracy', 
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
        
        # Save model config and weights
        model_json = model.to_json()
        with open(model_name+".json", "w") as json_file:
            json_file.write(model_json)            
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
        if material_ != material1_:
            plt.savefig(save_directory+"//loss_accuracy_"+material_+"_"+material1_+".png", 
                        bbox_inches='tight',format='png', dpi=1000)
        else:
            plt.savefig(save_directory+"//loss_accuracy_"+material_+".png", 
                        bbox_inches='tight',format='png', dpi=1000)
        plt.close()
        
        if material_ != material1_:
            text_file = open(save_directory+"//loss_accuracy_logger_"+material_+"_"+material1_+".txt", "w")
        else:
            text_file = open(save_directory+"//loss_accuracy_logger_"+material_+".txt", "w")
        
        text_file.write("# EPOCH, LOSS, VAL_LOSS, ACCURACY, VAL_ACCURACY" + "\n")
        for inj in range(len(epochs)):
            string1 = str(epochs[inj]) + ","
            string1 = string1 + str(model.history.history['loss'][inj])
            string1 = string1 + ","+str(model.history.history['val_loss'][inj])
            string1 = string1 + ","+str(model.history.history['accuracy'][inj])
            string1 = string1 + ","+str(model.history.history['val_accuracy'][inj])+" \n"  
            text_file.write(string1)
        text_file.close() 
        
        # ## Stats on the trained model with sklearn metrics
        from sklearn.metrics import classification_report
        x_test, y_test = vali_array(save_directory+"//testing_data", 50, len(classhkl), loc_new, print)
        y_test = np.argmax(y_test, axis=-1)
        y_pred = np.argmax(model.predict(x_test), axis=-1)
        print(classification_report(y_test, y_pred))
    
    if run_prediction:
        if prediction_GUI:
            print("Prediction with GUI is selected, please load the generated config file in LaueNN GUI and continue the prediction process")
        else:
            import numpy as np
            import multiprocessing, os, time, datetime
            import configparser, glob, re
            ## if LaueToolsNN is properly installed
            from lauetoolsnn.utils_lauenn import  get_material_detail, new_MP_function, resource_path, global_plots
            from lauetoolsnn.lauetools import dict_LaueTools as dictLT
            from lauetoolsnn.NNmodels import read_hdf5
            from keras.models import model_from_json
            import _pickle as cPickle
            from tqdm import tqdm
            
            ncpu = multiprocessing.cpu_count()
            print("Number of CPUs available : ", ncpu)
            
            # ## Get material parameters 
            # ### Get model and data paths from the input
            # ### User input parameters for various algorithms to compute the orientation matrix
    
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
            format_file = dictLT.dict_CCD["sCMOS"][7]
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
            
            ## get proper Laue group to compute the inverse pole figure colors and write MTEX output file for orientation analysis
            material0_lauegroup = "3"
            ## incase of same material
            material1_lauegroup = "5"
            
            ## Requirements
            ubmat = input_params["UB_matrix_to_detect"] # How many orientation matrix to detect per Laue pattern
            mode_spotCycle = input_params["mode_spotCycle"] ## mode of calculation
            use_previous_UBmatrix_name = input_params["use_previous"] ## Try previous indexation solutions to speed up the process
            strain_calculation = input_params["strain_compute"] ## Strain refinement is required or not
            ccd_label_global = input_params["ccd_label"]
                    
            ## tolerance angle to match simulated and experimental spots for two materials
            tolerance = input_params["matrix_tolerance"]
            tolerance1 = input_params["matrix_tolerance1"]
            
            ## tolerance angle for strain refinements
            tolerance_strain = input_params["tolerance_strain_refinement"]
            tolerance_strain1 = input_params["tolerance_strain_refinement1"]
            strain_free_parameters = input_params["free_parameters"]
            
            ## Parameters to control the orientation matrix indexation
            softmax_threshold_global = input_params["softmax_threshold_global"] # softmax_threshold of the Neural network to consider
            mr_threshold_global = 0.90 # match rate threshold to accept a solution immediately
            cap_matchrate = input_params["cap_matchrate"] * 100 ## any UB matrix providing MR less than this will be ignored
            coeff = 0.30            ## coefficient to calculate the overlap of two solutions
            coeff_overlap = input_params["coeff_overlap"]    ##10% spots overlap is allowed with already indexed orientation
            material0_limit = input_params["material0_limit"]  ## how many UB can be proposed for first material
            material1_limit = input_params["material1_limit"] ## how many UB can be proposed for second material; this forces the orientation matrix deduction algorithm to find only a required materials matrix
            material_phase_always_present = input_params["material_phase_always_present"] ## in case if one phase is always present in a Laue pattern (useful for substrate cases)
            
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
                
            ## load model related files and generate the model
            json_file = open(model_direc+"//model_"+prefix1+".json", 'r')
            classhkl = np.load(model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_0"]
            angbins = np.load(model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_1"]
            ind_mat = np.load(model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_5"]
            ind_mat1 = np.load(model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_6"]  
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
            
            # results = new_MP_function(valu12[0])
            # best = results[-2][0][0][0]
            
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
            
            try:
                global_plots(lim_x, lim_y, rotation_matrix, strain_matrix, strain_matrixs, 
                              col, colx, coly, match_rate, mat_global, spots_len, 
                              iR_pix, fR_pix, save_directory_, material_, material1_,
                              match_rate_threshold=5, bins=30)
            except:
                print("Error in the global plots module") 
        
        
        
        
        
        
