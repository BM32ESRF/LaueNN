#!/usr/bin/env python
# coding: utf-8

# # Notebook script for generation of training dataset (supports n phase material)
# 
# ## For case of one or two phase, GUI works
# 
# ## Different steps of data generation is outlined in this notebook (LaueToolsNN GUI does the same thing)
# 
# ### Define material of interest
# ### Generate class hkl data for Neural Network model (these are the output neurons)
# ### Clean up generated dataset

# In[1]:

if __name__ == '__main__':     #enclosing required because of multiprocessing

    ## If material key does not exist in Lauetoolsnn dictionary
    ## you can modify its JSON materials file before import or starting analysis
    import json
    
    ## Load the json of material and extinctions
    with open(r'C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\lauetools\material.json','r') as f:
        dict_Materials = json.load(f)

    with open(r'C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\lauetools\extinction.json','r') as f:
        extinction_json = json.load(f)
        
    ## Modify the dictionary values to add new entries
    dict_Materials["ZrO2_mono"] = ["ZrO2_mono", [5.1471,  5.2125, 5.3129, 90, 99.23, 90], "VO2_mono"]
    dict_Materials["ZrO2_tet"] = ["ZrO2_tet", [3.64,  3.64, 5.27, 90, 90, 90], "VO2_mono2tet"]
    dict_Materials["ZrO2_cub"] = ["ZrO2_cub", [4.625,  4.625, 4.625, 90, 90, 90], "225"]
    dict_Materials["Zr_alpha"] = ["Zr_alpha", [3.23,  3.23, 5.15, 90, 90, 120], "hcp"]
    dict_Materials["Zr_beta"] = ["Zr_beta", [3.62,  3.62, 3.62, 90, 90, 90], "229"]
    dict_Materials["Nb_beta"] = ["Nb_beta", [3.585,  3.585, 3.585, 90, 90, 90], "229"]
    dict_Materials["Zr_Nb_Fe"] = ["Zr_Nb_Fe", [4.879,  4.879, 7.992, 90, 90, 120], "hcp"]
    dict_Materials["Cr"] = ["Cr", [2.87,  2.87, 2.87, 90, 90, 90], "229"]
    dict_Materials["C14"] = ["C14", [5.05,  5.05, 8.24, 90, 90, 120], "hcp"]
    dict_Materials["C15"] = ["C15", [7.15,  7.15, 7.15, 90, 90, 90], "227"]
    
    extinction_json["VO2_mono2tet"] = "VO2_mono2tet"
    extinction_json["VO2_mono"] = "VO2_mono"
    extinction_json["hcp"] = "hcp"
    extinction_json["229"] = "229"
    extinction_json["227"] = "227"
    extinction_json["225"] = "225"
    
    ## dump the json back with new values
    with open(r'C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\lauetools\material.json', 'w') as fp:
        json.dump(dict_Materials, fp)

    with open(r'C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\lauetools\extinction.json', 'w') as fp:
        json.dump(extinction_json, fp)

    ## Import modules used for this Notebook
    import os
    import numpy as np
    import _pickle as cPickle
    import itertools
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    import matplotlib.pyplot as plt
    from tqdm import trange
    ## if LaueToolsNN is properly installed
    try:
        from lauetoolsnn.utils_lauenn import generate_classHKL, generate_multimat_dataset, \
                                        rmv_freq_class_MM, get_multimaterial_detail,\
                                        array_generator, array_generator_verify, vali_array
        from lauetoolsnn.NNmodels import model_arch_general_optimized, LoggingCallback,\
                                    model_arch_general_onelayer, model_arch_CNN_DNN_optimized
    except:
        # else import from a path where LaueToolsNN files are
        import sys
        sys.path.append(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn")
        from utils_lauenn import generate_classHKL, generate_multimat_dataset, \
                                rmv_freq_class_MM, get_multimaterial_detail,\
                                array_generator, array_generator_verify, vali_array
        from NNmodels import model_arch_general_optimized, LoggingCallback,\
                                    model_arch_general_onelayer, model_arch_CNN_DNN_optimized
                                
                                
    
    # ## step 1: define material and other parameters for simulating Laue patterns
        
    # In[2]:
    # =============================================================================
    ## User Input dictionary with parameters
    ## In case of only one phase/material, keep same value for material_ and material1_ key
    # =============================================================================
    input_params = {
                    # =============================================================================
                    #       GENERATION OF DATASET              
                    # =============================================================================
                    "prefix" : "",
                    "material_": ["Zr_alpha",
                                  "ZrO2_mono",
                                  ],             ## same key as used in dict_LaueTools
                    "symmetry": ["hexagonal",
                                 "monoclinic",
                                 ],           ## crystal symmetry of material_
                    "SG": [194,
                           14,
                           ],                     ## Space group of material_ (None if not known)
                    "hkl_max_identify" : [6,
                                          8,
                                          ],        ## Maximum hkl index to classify in a Laue pattern
                    "nb_grains_per_lp" : [4,
                                          4,
                                          ],        ## max grains to be generated in a Laue Image
                    "grains_nb_simulate" : 500,    ## Number of orientations to generate (takes advantage of crystal symmetry)
                    "maximum_angle_to_search":120, ## Angle of radial distribution to reconstruct the histogram (in deg)
                    "step_for_binning" : 0.1,      ## bin widht of angular radial distribution in degree
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
                    #       Training paarmeters             
                    # =============================================================================
                    "freq_rmv_classhkl" : [100,
                                           100,
                                           ],
                    "keep_length_classhkl" : [50,
                                              100,
                                              ],
                    "list_hkl_keep" : [
                                        [(0,0,1)],
                                        [(0,0,0)]                        
                                        ],
                    "batch_size":50,               ## batches of files to use while training
                    "epochs":30,   
                    }
    
    generate_data = False
    train_model = True
    
    # ### number of files it will generate fro training
    nb_grains_list = []
    for ino, imat in enumerate(input_params["material_"]):
        nb_grains_list.append(list(range(input_params["nb_grains_per_lp"][ino]+1)))
    list_permute = list(itertools.product(*nb_grains_list))
    list_permute.pop(0)
    print(len(list_permute)*input_params["grains_nb_simulate"])
    # ## Step 2: Get material parameters 
    # ### Generates a folder with material name and gets material unit cell parameters and symmetry object 
    # from the get_material_detail function
    
    # In[3]:
     
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
    
    save_directory = os.getcwd()+"//"+prefix_mat+input_params["prefix"]

    print("save directory is : "+save_directory)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    ## get unit cell parameters and other details required for simulating Laue patterns
    rules, symmetry, lattice_material, \
        crystal, SG = get_multimaterial_detail(material_, SG, symm_)
    
    
    # ## Step 3: Generate Neural network output classes (Laue spot hkls) using the generate_classHKL function
    
    # In[4]:
    if generate_data:
        ### generate_classHKL_multimat
        ## procedure for generation of GROUND TRUTH classes
        # general_diff_cond = True will eliminate the hkl index that does not satisfy the general reflection conditions
        for ino in trange(len(material_)):
            generate_classHKL(n[ino], rules[ino], lattice_material[ino], \
                              symmetry[ino], material_[ino], \
                              crystal=crystal[ino], SG=SG[ino], general_diff_cond=False,
                              save_directory=save_directory, write_to_console=print, \
                              ang_maxx = maximum_angle_to_search, \
                              step = step_for_binning)
        
        # ## Step 4: Generate Training and Testing dataset only for the output classes (Laue spot hkls) calculated in the Step 3
        # ### Uses multiprocessing library
        
        # In[5]:
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
                                 modelp = "random",
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
                                 modelp = "random",
                                 general_diff_rules = False, 
                                 crystal = crystal,)
        
        #%%# Updating the ClassHKL list by removing the non-common HKL or less frequent HKL from the list
        ## The non-common HKL can occur as a result of the detector position and energy used
        # freq_rmv: remove output hkl if the training dataset has less tha 100 occurances of the considered hkl (freq_rmv1 for second phase)
        # Weights (penalty during training) are also calculated based on the occurance
        
        freq_rmv = input_params["freq_rmv_classhkl"]
        elements = input_params["keep_length_classhkl"]
        list_hkl_keep = input_params["list_hkl_keep"]
        
        rmv_freq_class_MM(freq_rmv = freq_rmv, elements = elements,
                          save_directory = save_directory, material_ = material_,
                          write_to_console = print, progress=None, qapp=None,
                          list_hkl_keep = list_hkl_keep)
        
        
        ## End of data generation for Neural network training: all files are saved in the same folder 
        ## to be later used for training and prediction
    
        # ## Step 2: Load the necessary files generated in Step 1 script
        # ### Loading the Output class and ground truth

    # In[3]:
    if train_model:
        
        classhkl = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_0"]
        angbins = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_1"]
        loc_new = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_2"]
        with open(save_directory+"//class_weights.pickle", "rb") as input_file:
            class_weights = cPickle.load(input_file)
        class_weights = class_weights[0]
        n_bins = len(angbins)-1
        n_outputs = len(classhkl)
        print(n_bins, n_outputs)
        
        # ## Step 4: Training  
            
        # In[5]:
    
        epochs = input_params["epochs"]
        batch_size = input_params["batch_size"] 
        
        # model save directory and filename
        if len(material_) > 1:
            prefix_mat = material_[0]
            for ino, imat in enumerate(material_):
                if ino == 0:
                    continue
                prefix_mat = prefix_mat + "_" + imat
        else:
            prefix_mat = material_[0]
            
        model_name = save_directory+"//model_"+prefix_mat
        
        # Define model and train
        # neurons_multiplier is a list with number of neurons per layer, the first 
        # value is input shape and last value is output shape, 
        # inbetween are the number of neurons per hidden layers
        
        # model = model_arch_general_optimized(  n_bins, n_outputs,
        #                                         kernel_coeff = 1e-5,
        #                                         bias_coeff = 1e-6,
        #                                         lr = 1e-3,
        #                                         )
        # model = model_arch_general_onelayer(  n_bins, n_outputs,
        #                                         kernel_coeff = 1e-5,
        #                                         bias_coeff = 1e-6,
        #                                         lr = 1e-3,
        #                                         )
        
        model = model_arch_CNN_DNN_optimized(
                                                (n_bins, 1), 
                                                layer_activation="relu", 
                                                output_activation="softmax",
                                                dropout=0.2,
                                                stride = [5,2],
                                                kernel_size = [10,3],
                                                pool_size=[2,1],
                                                CNN_layers = 2,
                                                CNN_filters = [128,128],
                                                DNN_layers = 0,
                                                DNN_filters = [1000,500],
                                                output_neurons = n_outputs,
                                                learning_rate = 0.001,
                                                output="DNN"
                                                )
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
            nb_grains_list.append(list(range(nb_grains_per_lp[ino]+1)))
        list_permute = list(itertools.product(*nb_grains_list))
        list_permute.pop(0)
        steps_per_epoch = len(list_permute)*(grains_nb_simulate)//batch_size        
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
        # In[6]:
        from sklearn.metrics import classification_report
        ## verify the statistics
        x_test, y_test = vali_array(save_directory+"//testing_data", 50, len(classhkl), loc_new, print)
        y_test = np.argmax(y_test, axis=-1)
        y_pred = np.argmax(model.predict(x_test), axis=-1)
        print(classification_report(y_test, y_pred))





