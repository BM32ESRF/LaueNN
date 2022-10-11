# -*- coding: utf-8 -*-

## as of now only test imports
## testing of GUI to be done after installation manually with example case
# import pytest
# if __name__ == '__main__':     #enclosing required because of multiprocessing

import keras
print("Keras passed import")

import scipy
print("scipy passed import")

import numpy
print("numpy passed import")

import h5py
print("h5py passed import")

import tensorflow
print("tensorflow passed import")

import PyQt5
print("PyQt5 passed import")

import sklearn
print("sklearn passed import")

import skimage
print("skimage passed import")

import fabio
print("fabio passed import")

import networkx
print("networkx passed import")

import tqdm
print("tqdm passed import")


import pkg_resources

print("scipy :", pkg_resources.require("scipy")[0].version)
print("numpy :", pkg_resources.require("numpy")[0].version)
print("h5py :", pkg_resources.require("h5py")[0].version)
print("keras :", pkg_resources.require("keras")[0].version)
print("tensorflow :", pkg_resources.require("tensorflow")[0].version)
print("PyQt5 :", pkg_resources.require("PyQt5")[0].version)
print("sklearn :", pkg_resources.require("scikit-learn")[0].version)
print("skimage :", pkg_resources.require("scikit-image")[0].version)
print("fabio :", pkg_resources.require("fabio")[0].version)
print("networkx :", pkg_resources.require("networkx")[0].version)
print("tqdm :", pkg_resources.require("tqdm")[0].version)


def test_method1():
    ##DUmmy test 
    	a = 6
    	b = 8
    	assert a+2== b, "test failed"
    	assert b-2 == a, "test failed"
    
    #TODO add an example test case that verifies all the functionality of GUI
    # For eample run the automated scripts from the example notebook directory
    # need to add prediction routine , with simulated data ?

def test_lauenn_GenerationandTraining_module():
    ## Import modules used for this Notebook
    import os
    ## if LaueToolsNN is properly installed
    try:
        from lauetoolsnn.utils_lauenn import generate_classHKL, generate_dataset, rmv_freq_class, get_material_detail
    except:
        print("relative import failing")
        return
    # ## step 1: define material and other parameters for simulating Laue patterns
    # =============================================================================
    ## User Input dictionary with parameters
    ## In case of only one phase/material, keep same value for material_ and material1_ key
    # =============================================================================
    input_params = {
                    "material_": "Si",             ## same key as used in dict_LaueTools
                    "material1_": "GaN",            ## same key as used in dict_LaueTools
                    "prefix" : "",                 ## prefix for the folder to be created for training dataset
                    "symmetry": "cubic",           ## crystal symmetry of material_
                    "symmetry1": "hexagonal",          ## crystal symmetry of material1_
                    "SG": 227,                     ## Space group of material_ (None if not known)
                    "SG1": 186,                    ## Space group of material1_ (None if not known)
                    "hkl_max_identify" : 3,        ## Maximum hkl index to classify in a Laue pattern
                    "hkl_max_identify1" : 3,        ## Maximum hkl index to classify in a Laue pattern
                    "maximum_angle_to_search":120, ## Angle of radial distribution to reconstruct the histogram (in deg)
                    "step_for_binning" : 0.1,      ## bin widht of angular radial distribution in degree
                    "nb_grains_per_lp_mat0" : 1,        ## max grains to be generated in a Laue Image
                    "nb_grains_per_lp_mat1" : 1,        ## max grains to be generated in a Laue Image
                    "grains_nb_simulate" : 500,    ## Number of orientations to generate (takes advantage of crystal symmetry)
                    ## Detector parameters (roughly) of the Experimental setup
                    ## Sample-detector distance, X center, Y center, two detector angles
                    "detectorparameters" :  [79.26200, 972.2800, 937.7200, 0.4160000, 0.4960000], 
                    "pixelsize" : 0.0734,          ## Detector pixel size
                    "dim1":2018,                   ## Dimensions of detector in pixels
                    "dim2":2016,
                    "emin" : 5,                    ## Minimum and maximum energy to use for simulating Laue Patterns
                    "emax" : 22,
                    }
    
    
    # ## Step 2: Get material parameters 
    # ### Generates a folder with material name and gets material unit cell parameters and symmetry object from the get_material_detail function  
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
    
    hkl_array = None
    hkl_array1 = None
    
    if material_ != material1_:
        save_directory = os.getcwd()+"//"+material_+"_"+material1_+input_params["prefix"]
    else:
        save_directory = os.getcwd()+"//"+material_+input_params["prefix"]
    print("save directory is : "+save_directory)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    ## get unit cell parameters and other details required for simulating Laue patterns
    rules, symmetry, lattice_material, crystal, SG, rules1, symmetry1,\
        lattice_material1, crystal1, SG1 = get_material_detail(material_, SG, symm_,
                                                               material1_, SG1, symm1_)
    
    # ## Step 3: Generate Neural network output classes (Laue spot hkls) using the generate_classHKL function        
    ## procedure for generation of GROUND TRUTH classes
    # general_diff_cond = True will eliminate the hkl index that does not satisfy the general reflection conditions
    generate_classHKL(n, rules, lattice_material, symmetry, material_, crystal=crystal, SG=SG, general_diff_cond=False,
              save_directory=save_directory, write_to_console=print, ang_maxx = maximum_angle_to_search, 
              step = step_for_binning, mat_listHKl=hkl_array)
    
    if material_ != material1_:
        generate_classHKL(n1, rules1, lattice_material1, symmetry1, material1_, crystal=crystal1, SG=SG1, general_diff_cond=False,
                  save_directory=save_directory, write_to_console=print, ang_maxx = maximum_angle_to_search, 
                  step = step_for_binning, mat_listHKl=hkl_array1)
    
    
    # ## Step 4: Generate Training and Testing dataset only for the output classes (Laue spot hkls) calculated in the Step 3
    # ### Uses multiprocessing library

    ############ GENERATING TRAINING DATA ##############
    # data_realism =True ; will introduce noise and partial Laue patterns in the training dataset
    # modelp can have either "random" for random orientation generation or "uniform" for uniform orientation generation
    # include_scm (if True; misorientation_angle parameter need to be defined): this parameter introduces misoriented crystal of specific angle along a crystal axis in the training dataset
    generate_dataset(material_=material_, material1_=material1_, ang_maxx=maximum_angle_to_search,
                         step=step_for_binning, mode=0, 
                         nb_grains=nb_grains_per_lp0, nb_grains1=nb_grains_per_lp1, 
                         grains_nb_simulate=grains_nb_simulate, data_realism = True, 
                         detectorparameters=detectorparameters, pixelsize=pixelsize, type_="training_data",
                         var0 = 1, dim1=input_params["dim1"], dim2=input_params["dim2"], 
                         removeharmonics=1, save_directory=save_directory,
                        write_to_console=print, emin=emin, emax=emax, modelp = "random",
                        misorientation_angle = 1, general_diff_rules = False, 
                        crystal = crystal, crystal1 = crystal1, include_scm=False,
                        mat_listHKl=hkl_array, mat_listHKl1=hkl_array1)
    
    ############ GENERATING TESTING DATA ##############
    factor = 5 # validation split for the training dataset  --> corresponds to 20% of total training dataset
    generate_dataset(material_=material_, material1_=material1_, ang_maxx=maximum_angle_to_search,
                         step=step_for_binning, mode=0, 
                         nb_grains=nb_grains_per_lp0, nb_grains1=nb_grains_per_lp1, 
                         grains_nb_simulate=grains_nb_simulate//factor, data_realism = True, 
                         detectorparameters=detectorparameters, pixelsize=pixelsize, type_="testing_data",
                         var0 = 1, dim1=input_params["dim1"], dim2=input_params["dim2"], 
                         removeharmonics=1, save_directory=save_directory,
                        write_to_console=print, emin=emin, emax=emax, modelp = "random",
                        misorientation_angle = 1, general_diff_rules = False, 
                        crystal = crystal, crystal1 = crystal1, include_scm=False,
                        mat_listHKl=hkl_array, mat_listHKl1=hkl_array1)
    
    ## Updating the ClassHKL list by removing the non-common HKL or less frequent HKL from the list
    ## The non-common HKL can occur as a result of the detector position and energy used
    # freq_rmv: remove output hkl if the training dataset has less tha 100 occurances of the considered hkl (freq_rmv1 for second phase)
    # Weights (penalty during training) are also calculated based on the occurance
    rmv_freq_class(freq_rmv = 1, freq_rmv1 = 1,
                        save_directory=save_directory, material_=material_, 
                        material1_=material1_, write_to_console=print)
    
    ## End of data generation for Neural network training: all files are saved in the same folder to be later used for training and prediction

    ## Import modules used for this Notebook
    import numpy as np
    import os
    import _pickle as cPickle
    import itertools
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    import matplotlib.pyplot as plt
    
    ## if LaueToolsNN is properly installed
    from lauetoolsnn.utils_lauenn import array_generator, array_generator_verify, vali_array
    from lauetoolsnn.NNmodels import model_arch_general
    # ## step 1: define material and path to access the training dataset generated using Step 1 script
    # =============================================================================
    ## User Input dictionary with parameters (reduced but same as the one used in STEP 1)
    ## In case of only one phase/material, keep same value for material_ and material1_ key
    # =============================================================================
    input_params = {
                    "material_": "Si",             ## same key as used in dict_LaueTools
                    "material1_": "GaN",            ## same key as used in dict_LaueTools
                    "prefix" : "",                 ## prefix for the folder to be created for training dataset  
                    "nb_grains_per_lp_mat0" : 1,        ## max grains to be generated in a Laue Image
                    "nb_grains_per_lp_mat1" : 1,        ## max grains to be generated in a Laue Image
                    "grains_nb_simulate" : 500,    ## Number of orientations to generate (takes advantage of crystal symmetry)
                    "batch_size":50,               ## batches of files to use while training
                    "epochs":5,                    ## number of epochs for training
                    }
    
    material_= input_params["material_"]
    material1_= input_params["material1_"]
    nb_grains_per_lp = input_params["nb_grains_per_lp_mat0"]
    nb_grains_per_lp1 = input_params["nb_grains_per_lp_mat1"]
    grains_nb_simulate = input_params["grains_nb_simulate"]
    
    if material_ != material1_:
        save_directory = os.getcwd()+"//"+material_+"_"+material1_+input_params["prefix"]
    else:
        save_directory = os.getcwd()+"//"+material_+input_params["prefix"]
    
    if not os.path.exists(save_directory):
        print("The directory doesn't exists; please veify the path")
    else:
        print("Directory where training dataset is stored is : "+save_directory)
    
    
    # ## Step 2: Load the necessary files generated in Step 1 script
    # ### Loading the Output class and ground truth        
    classhkl = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_0"]
    angbins = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_1"]
    loc_new = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_2"]
    with open(save_directory+"//class_weights.pickle", "rb") as input_file:
        class_weights = cPickle.load(input_file)
    class_weights = class_weights[0]
    
    # ## Step 3: Defining a neural network architecture
    
    ###from NNmodel.py script or define here
    
    # ## Step 4: Training  

    # load model and train
    #neurons_multiplier is a list with number of neurons per layer, the first value is input shape and last value is output shape, inbetween are the number of neurons per hidden layers
    model = model_arch_general(  len(angbins)-1, len(classhkl),
                                           kernel_coeff = 1e-5,
                                           bias_coeff = 1e-6,
                                           lr = 1e-3,
                                            )
    ## temp function to quantify the spots and classes present in a batch
    batch_size = input_params["batch_size"] 
    trainy_inbatch = array_generator_verify(save_directory+"//training_data", batch_size, 
                                            len(classhkl), loc_new, print)
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
    
    ######### TRAIN THE DATA
    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=2)
    ms = ModelCheckpoint(save_directory+"//best_val_acc_model.h5", monitor='val_accuracy', 
                          mode='max', save_best_only=True)
    
    # model save directory and filename
    if material_ != material1_:
        model_name = save_directory+"//model_"+material_+"_"+material1_
    else:
        model_name = save_directory+"//model_"+material_
    
    ## Fitting function
    stats_model = model.fit(
                            training_data_generator, 
                            epochs=epochs, 
                            steps_per_epoch=steps_per_epoch,
                            validation_data=testing_data_generator,
                            validation_steps=val_steps_per_epoch,
                            verbose=1,
                            class_weight=class_weights,
                            callbacks=[es, ms]
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
        plt.savefig(save_directory+"//loss_accuracy_"+material_+"_"+material1_+".png", bbox_inches='tight',format='png', dpi=1000)
    else:
        plt.savefig(save_directory+"//loss_accuracy_"+material_+".png", bbox_inches='tight',format='png', dpi=1000)
    plt.close()
    
    if material_ != material1_:
        text_file = open(save_directory+"//loss_accuracy_logger_"+material_+"_"+material1_+".txt", "w")
    else:
        text_file = open(save_directory+"//loss_accuracy_logger_"+material_+".txt", "w")
    
    text_file.write("# EPOCH, LOSS, VAL_LOSS, ACCURACY, VAL_ACCURACY" + "\n")
    for inj in range(len(epochs)):
        string1 = str(epochs[inj]) + ","+ str(model.history.history['loss'][inj])+            ","+str(model.history.history['val_loss'][inj])+","+str(model.history.history['accuracy'][inj])+            ","+str(model.history.history['val_accuracy'][inj])+" \n"  
        text_file.write(string1)
    text_file.close() 
    
    # ## Stats on the trained model with sklearn metrics
    from sklearn.metrics import classification_report
    
    ## verify the 
    x_test, y_test = vali_array(save_directory+"//testing_data", 50, len(classhkl), loc_new, print)
    y_test = np.argmax(y_test, axis=-1)
    y_pred = np.argmax(model.predict(x_test), axis=-1)
    print(classification_report(y_test, y_pred))




    
    
    
    
    
    