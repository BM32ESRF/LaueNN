#!/usr/bin/env python
# coding: utf-8

# # Deep learning for HKL classification
# ## TODO: Test for Triclinic --> if the model works for two extreme symmetry case, then it should work for intermediate symmetries as well
# 
# ## Now can predict with > 95% accuracy for multi grain Laue Patterns
# ## If you have model save files; go to cell 45 to load and start prediction
# 
# ## Pros: Impressive speed for prediction; results not dependent on the statistical descriptor (ex: correlation function or distance measurment)
# ## Cons: Building reliable test data can take few hours (this will significantly increase for less symmetry crystals) --> multiprocessing to reduce time

# ## Library import

# In[51]:


# Keras library is required 
# "conda install -c conda-forge keras" for anaconda distribution
# Currently CPU calculation is more than enough. GPU not really needed


# In[82]:
import numpy as np
import os

import sys
sys.path.append(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn")
from utils_lauenn import generate_classHKL, generate_dataset, \
                        array_generator, rmv_freq_class, \
                        array_generator_verify, vali_array, get_material_detail

if __name__ == '__main__':                    
    # =============================================================================
    ## Make more general creation of Laue pattern; so no need to generate data everytime
    ## later the ang bins can be adjusted
    # =============================================================================
    generate_data = False
    train_model = False
    optimize_model = True
    
    input_params = {
                    "material_": "Cu", ## same key as used in LaueTools
                    "material1_": "Cu", ## same key as used in LaueTools
                    "prefix" : "",
                    "symmetry": "cubic",
                    "symmetry1": "cubic",
                    "hkl_max_identify" : 5, # can be "auto" or an index i.e 12
                    "maximum_angle_to_search" : 90,
                    "step_for_binning" : 0.1,
                    "nb_grains_per_lp" : 5, ## max grains to expect in a LP
                    "grains_nb_simulate" : 100, #500
                    "detectorparameters" :  [79.553,979.32,932.31,0.37,0.447],
                    "pixelsize" : 0.0734,
                    "dim1":2018,
                    "dim2":2016,
                    "emin" : 5,
                    "emax" : 22,
                    "batch_size":20, ## batches of files to use while training
                    "epochs":5, ## number of epochs for training
                    }
        
    #%% TEMP TEST to know the max HKL present in the camera for a material
    material_= input_params["material_"]
    material1_= input_params["material1_"]
    ## generate reference HKL library
    ############ USER INPUT
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
    
    if material_ != material1_:
        save_directory = os.getcwd()+"//"+material_+"_"+material1_+input_params["prefix"]
    else:
        save_directory = os.getcwd()+"//"+material_+input_params["prefix"]
        
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    # =============================================================================
    # Symmetry input
    # =============================================================================
    SG = 225 #14
    SG1 = 225 #14
    
    rules, symmetry, lattice_material, \
        crystal, SG, rules1, symmetry1,\
        lattice_material1, crystal1, SG1 = get_material_detail(material_, SG, symm_,
                                                               material1_, SG1, symm1_)
    #%%
    if generate_data:
        ## procedure for generation of GROUND TRUTH classes
        # =============================================================================
        # VERY IMPORTANT; TAKES Significant time
        # =============================================================================
        generate_classHKL(n, rules, lattice_material, symmetry, material_, crystal=crystal, SG=SG, general_diff_cond=False,
                  save_directory=save_directory, write_to_console=print, ang_maxx = maximum_angle_to_search, 
                  step = step_for_binning)
        if material_ != material1_:
            generate_classHKL(n, rules1, lattice_material1, symmetry1, material1_, crystal=crystal1, SG=SG1, general_diff_cond=False,
                      save_directory=save_directory, write_to_console=print, ang_maxx = maximum_angle_to_search, 
                      step = step_for_binning)
        
    # # Lets train the model
    # ## generate data for model learning
    # In[10]:
    if generate_data:
        factor = 5 # validation split for the training dataset
        ############ GENERATING TRAINING DATA
        generate_dataset(material_=material_, material1_=material1_, ang_maxx=maximum_angle_to_search,
                             step=step_for_binning, mode=0, 
                             nb_grains=nb_grains_per_lp, nb_grains1=nb_grains_per_lp, 
                             grains_nb_simulate=grains_nb_simulate, data_realism = True, 
                             detectorparameters=detectorparameters, pixelsize=pixelsize, type_="training_data",
                             var0 = 1, dim1=input_params["dim1"], dim2=input_params["dim2"], 
                             removeharmonics=1, save_directory=save_directory,
                            write_to_console=print, emin=emin, emax=emax, modelp = "random",
                            misorientation_angle = 1, general_diff_rules = False, 
                            crystal = crystal, crystal1 = crystal1, include_scm=False, 
                            matrix_phase_always_present=None)
        
        generate_dataset(material_=material_, material1_=material1_, ang_maxx=maximum_angle_to_search,
                             step=step_for_binning, mode=0, 
                             nb_grains=nb_grains_per_lp, nb_grains1=nb_grains_per_lp, 
                             grains_nb_simulate=grains_nb_simulate//factor, data_realism = True, 
                             detectorparameters=detectorparameters, pixelsize=pixelsize, type_="testing_data",
                             var0 = 1, dim1=input_params["dim1"], dim2=input_params["dim2"], 
                             removeharmonics=1, save_directory=save_directory,
                            write_to_console=print, emin=emin, emax=emax, modelp = "random",
                            misorientation_angle = 1, general_diff_rules = False, 
                            crystal = crystal, crystal1 = crystal1, include_scm=False, 
                            matrix_phase_always_present=None)
    # In[11]:
    if generate_data:
        ## Updating the ClassHKL list by removing the non-common HKL from the list
        ## The non-common HKL can occur as a result of the detector position and energy used
        rmv_freq_class(freq_rmv = 100, elements="all", freq_rmv1 = 100, elements1="all",
                            save_directory=save_directory, material_=material_, 
                            material1_=material1_, write_to_console=print,
                            progress=None, qapp=None)
    
    #%% Train model
    import _pickle as cPickle

    classhkl = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_0"]
    angbins = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_1"]
    loc_new = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_2"]
    with open(save_directory+"//class_weights.pickle", "rb") as input_file:
        class_weights = cPickle.load(input_file)
    class_weights = class_weights[0]
    
    n_bins = len(angbins)-1
    n_outputs = len(classhkl)
    
    if train_model:
        from sklearn.metrics import classification_report
        import itertools
        from keras.callbacks import EarlyStopping, ModelCheckpoint
        import matplotlib.pyplot as plt
        from model_NN import model_arch_general
        
        factor = 5 # validation split for the training dataset
        
        # load model and train
        model = model_arch_general( 
                                    kernel_coeff = 0.0005, bias_coeff = 0.0005, lr=0.0001,
                                    neurons_multiplier= [8, 15], layers=3, optimizer="adam", 
                                    batch_norm=True, activation="relu",
                                    dropout=0.3
                                    )
        ## temp function to quantify the spots and classes present in a batch
        batch_size = input_params["batch_size"] 
        trainy_inbatch = array_generator_verify(save_directory+"//training_data", batch_size, 
                                                len(classhkl), loc_new, print)
        print("Number of spots in a batch of %i files : %i" %(batch_size, len(trainy_inbatch)))
        print("Min, Max class ID is %i, %i" %(np.min(trainy_inbatch), np.max(trainy_inbatch)))
        # try varying batch size and epochs
        epochs = input_params["epochs"] 
        ## Batch loading for numpy grain files (Keep low value to avoid overcharging the RAM)
        nb_grains_per_lp1 = nb_grains_per_lp
        if material_ != material1_:
            nb_grains_list = list(range(nb_grains_per_lp+1))
            nb_grains1_list = list(range(nb_grains_per_lp1+1))
            list_permute = list(itertools.product(nb_grains_list, nb_grains1_list))
            list_permute.pop(0)
            steps_per_epoch = (len(list_permute) * grains_nb_simulate)//batch_size
        else:
            steps_per_epoch = int((nb_grains_per_lp * grains_nb_simulate) / batch_size)
            
        val_steps_per_epoch = int(steps_per_epoch / factor)
        if steps_per_epoch == 0:
            steps_per_epoch = 1
        if val_steps_per_epoch == 0:
            val_steps_per_epoch = 1 
        ## Load generator objects from filepaths
        training_data_generator = array_generator(save_directory+"//training_data", batch_size, \
                                                  len(classhkl), loc_new, print)
        testing_data_generator = array_generator(save_directory+"//testing_data", batch_size, \
                                                  len(classhkl), loc_new, print)
        ######### TRAIN THE DATA
        # from clr_callback import CyclicLR
        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=2)
        ms = ModelCheckpoint(save_directory+"//best_val_acc_model.h5", monitor='val_accuracy', 
                              mode='max', save_best_only=True)
        
        # model save directory and filename
        if material_ != material1_:
            model_name = save_directory+"//model_"+material_+"_"+material1_
        else:
            model_name = save_directory+"//model_"+material_
            
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
        # Save model config and weightsp
        ## new trained model, save files
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
            string1 = str(epochs[inj]) + ","+ str(model.history.history['loss'][inj])+\
                    ","+str(model.history.history['val_loss'][inj])+","+str(model.history.history['accuracy'][inj])+\
                    ","+str(model.history.history['val_accuracy'][inj])+" \n"  
            text_file.write(string1)
        text_file.close()
        
        x_test, y_test = vali_array(save_directory+"//testing_data", 50, len(classhkl), loc_new, print)
        y_test = np.argmax(y_test, axis=-1)
        y_pred = np.argmax(model.predict(x_test), axis=-1)
        print(classification_report(y_test, y_pred))    
    #%% Tuning hyperparameters GRID SEARCH
    if optimize_model:
        # grid search regularization values for moons dataset
        from sklearn.model_selection import GridSearchCV
        from keras.wrappers.scikit_learn import KerasClassifier
        
        ## Tuning on small dataset of Cu
        x_training, y_training = vali_array(save_directory+"//training_data", 20, 
                                            len(classhkl), 
                                            loc_new, print, tocategorical=False)
        if material_ != material1_:
            text_file = open(save_directory+"//grid_optimizer_logger_"+material_+"_"+material1_+".txt", "w")
        else:
            text_file = open(save_directory+"//grid_optimizer_logger_"+material_+".txt", "w")
        
        from model_NN import model_arch_general
        # =============================================================================
        # Tuning batch size and epochs
        # =============================================================================
        # Wrap Keras model so it can be used by scikit-learn
        model_ann = KerasClassifier(build_fn=model_arch_general, verbose=2)
        # Create hyperparameter space
        optimizer = ['Adam']
        learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
        init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
        activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
        dropout_rate = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
        weight_constraint = [None] #[1, 2, 3, 4, 5, 6, 7, 8, 9]
        layers = [1,2,3,4,5,6]
        values = [None]
        
        first_layer = int(n_outputs * 2)
        neurons_multiplier = [[n_bins, n_bins, first_layer*5, first_layer*10, first_layer*15, first_layer*20, first_layer*25, n_outputs],
                              [n_bins, n_bins, first_layer*10, first_layer*15, first_layer*20, first_layer*25, first_layer*30,n_outputs],
                              [n_bins, n_bins, first_layer*15, first_layer*20, first_layer*25, first_layer*30, first_layer*35,n_outputs],
                              [n_bins, n_bins, first_layer*20, first_layer*25, first_layer*30, first_layer*35, first_layer*40,n_outputs],
                              [n_bins, n_bins, first_layer*25, first_layer*30, first_layer*15, first_layer*20, first_layer*25,n_outputs],
                              [n_bins, n_bins, first_layer//4, first_layer//5, first_layer//6, first_layer//7, first_layer//8,n_outputs],
                              [n_bins, n_bins, first_layer, first_layer, first_layer, first_layer, first_layer,n_outputs],
                              [n_bins, n_bins, 32, 64, 128, 64, 32,n_outputs],
                              [n_bins, n_bins, n_bins, n_bins, n_bins, n_bins, n_bins, n_outputs],]
        batch_norm = [False]
        
        # Create hyperparameter options
        hyperparameters = dict( 
                                kernel_coeff=values,
                                bias_coeff=values,
                                init_mode=init_mode,
                                learning_rate=learning_rate,
                                neurons_multiplier=neurons_multiplier, 
                                layers=layers,
                                batch_norm=batch_norm,
                                optimizer=optimizer,
                                activation=activation,
                                dropout_rate=dropout_rate,
                                weight_constraint=weight_constraint,
                                )
        
        grid = GridSearchCV(estimator=model_ann, cv=5, param_grid=hyperparameters, n_jobs=-1)
        # Fit grid search
        grid_result = grid.fit(x_training, y_training)
        # View hyperparameters of best neural network
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        text_file.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_) + "\n")
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
            text_file.write("%f (%f) with: %r" % (mean, stdev, param) + "\n")
        
        
        # from model_NN import model_arch_general_noarg
        # # =============================================================================
        # # Tuning batch size and epochs
        # # =============================================================================
        # # Wrap Keras model so it can be used by scikit-learn
        # model_ann = KerasClassifier(build_fn=model_arch_general_noarg, verbose=0)
        # # Create hyperparameter space
        # epoch_values = [1, 3, 5, 7, 9]
        # batches = [10, 20, 30, 40, 50]
        # # Create hyperparameter options
        # hyperparameters = dict(
        #                         batch_size=batches,
        #                         epochs=epoch_values
        #                         )
        # grid = GridSearchCV(estimator=model_ann, cv=5, param_grid=hyperparameters, n_jobs=-1)
        # # Fit grid search
        # grid_result = grid.fit(x_training, y_training)
        # # View hyperparameters of best neural network
        # # summarize results
        # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # text_file.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_) + "\n")
        # means = grid_result.cv_results_['mean_test_score']
        # stds = grid_result.cv_results_['std_test_score']
        # params = grid_result.cv_results_['params']
        # for mean, stdev, param in zip(means, stds, params):
        #     print("%f (%f) with: %r" % (mean, stdev, param))
        #     text_file.write("%f (%f) with: %r" % (mean, stdev, param) + "\n")
    
        
        # from model_NN import model_arch_general_optimizer
        # # =============================================================================
        # # Tuning optimizers
        # # =============================================================================
        # # Wrap Keras model so it can be used by scikit-learn
        # model_ann = KerasClassifier(build_fn=model_arch_general_optimizer, verbose=0)
        # # Create hyperparameter space
        # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        # # Create hyperparameter options
        # hyperparameters = dict(optimizer=optimizer)
        # grid = GridSearchCV(estimator=model_ann, cv=5, param_grid=hyperparameters, n_jobs=-1)
        # # Fit grid search
        # grid_result = grid.fit(x_training, y_training)
        # # View hyperparameters of best neural network
        # # summarize results
        # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # text_file.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_) + "\n")
        # means = grid_result.cv_results_['mean_test_score']
        # stds = grid_result.cv_results_['std_test_score']
        # params = grid_result.cv_results_['params']
        # for mean, stdev, param in zip(means, stds, params):
        #     print("%f (%f) with: %r" % (mean, stdev, param))
        #     text_file.write("%f (%f) with: %r" % (mean, stdev, param) + "\n")
        
        
        # from model_NN import model_arch_general_optimizer_lr
        # # =============================================================================
        # # Tuning Learning rate for Adam
        # # =============================================================================
        # # Wrap Keras model so it can be used by scikit-learn
        # model_ann = KerasClassifier(build_fn=model_arch_general_optimizer_lr, verbose=0)
        # # Create hyperparameter space
        # learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
        # # Create hyperparameter options
        # hyperparameters = dict(learning_rate=learning_rate)
        # grid = GridSearchCV(estimator=model_ann, cv=5, param_grid=hyperparameters, n_jobs=-1)
        # # Fit grid search
        # grid_result = grid.fit(x_training, y_training)
        # # View hyperparameters of best neural network
        # # summarize results
        # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # text_file.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_) + "\n")
        # means = grid_result.cv_results_['mean_test_score']
        # stds = grid_result.cv_results_['std_test_score']
        # params = grid_result.cv_results_['params']
        # for mean, stdev, param in zip(means, stds, params):
        #     print("%f (%f) with: %r" % (mean, stdev, param))
        #     text_file.write("%f (%f) with: %r" % (mean, stdev, param) + "\n")
            
        
        # from model_NN import model_arch_general_weight
        # # =============================================================================
        # # Tuning Weight initialization for Kernal
        # # =============================================================================
        # # Wrap Keras model so it can be used by scikit-learn
        # model_ann = KerasClassifier(build_fn=model_arch_general_weight, verbose=0)
        # # Create hyperparameter space
        # init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
        # # Create hyperparameter options
        # hyperparameters = dict(init_mode=init_mode)
        # grid = GridSearchCV(estimator=model_ann, cv=5, param_grid=hyperparameters, n_jobs=-1)
        # # Fit grid search
        # grid_result = grid.fit(x_training, y_training)
        # # View hyperparameters of best neural network
        # # summarize results
        # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # text_file.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_) + "\n")
        # means = grid_result.cv_results_['mean_test_score']
        # stds = grid_result.cv_results_['std_test_score']
        # params = grid_result.cv_results_['params']
        # for mean, stdev, param in zip(means, stds, params):
        #     print("%f (%f) with: %r" % (mean, stdev, param))
        #     text_file.write("%f (%f) with: %r" % (mean, stdev, param) + "\n")
            
        
        # from model_NN import model_arch_general_activation
        # # =============================================================================
        # # Tuning Weight initialization for Kernal
        # # =============================================================================
        # # Wrap Keras model so it can be used by scikit-learn
        # model_ann = KerasClassifier(build_fn=model_arch_general_activation, verbose=0)
        # # Create hyperparameter space
        # activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
        # # Create hyperparameter options
        # hyperparameters = dict(activation=activation)
        # grid = GridSearchCV(estimator=model_ann, cv=5, param_grid=hyperparameters, n_jobs=-1)
        # # Fit grid search
        # grid_result = grid.fit(x_training, y_training)
        # # View hyperparameters of best neural network
        # # summarize results
        # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # text_file.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_) + "\n")
        # means = grid_result.cv_results_['mean_test_score']
        # stds = grid_result.cv_results_['std_test_score']
        # params = grid_result.cv_results_['params']
        # for mean, stdev, param in zip(means, stds, params):
        #     print("%f (%f) with: %r" % (mean, stdev, param))
        #     text_file.write("%f (%f) with: %r" % (mean, stdev, param) + "\n")
         
            
        # from model_NN import model_arch_general_dropout
        # # =============================================================================
        # # Tuning Weight initialization for Kernal
        # # =============================================================================
        # # Wrap Keras model so it can be used by scikit-learn
        # model_ann = KerasClassifier(build_fn=model_arch_general_dropout, verbose=0)
        # # Create hyperparameter space
        # dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # # Create hyperparameter options
        # hyperparameters = dict(dropout_rate=dropout_rate)
        # grid = GridSearchCV(estimator=model_ann, cv=5, param_grid=hyperparameters, n_jobs=-1)
        # # Fit grid search
        # grid_result = grid.fit(x_training, y_training)
        # # View hyperparameters of best neural network
        # # summarize results
        # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # text_file.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_) + "\n")
        # means = grid_result.cv_results_['mean_test_score']
        # stds = grid_result.cv_results_['std_test_score']
        # params = grid_result.cv_results_['params']
        # for mean, stdev, param in zip(means, stds, params):
        #     print("%f (%f) with: %r" % (mean, stdev, param))
        #     text_file.write("%f (%f) with: %r" % (mean, stdev, param) + "\n")
         
            
        # from model_NN import model_arch_general_dropoutweight
        # # =============================================================================
        # # Tuning Weight initialization for Kernal
        # # =============================================================================
        # # Wrap Keras model so it can be used by scikit-learn
        # model_ann = KerasClassifier(build_fn=model_arch_general_dropoutweight, verbose=0)
        # # Create hyperparameter space
        # dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # weight_constraint = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        # # Create hyperparameter options
        # hyperparameters = dict(dropout_rate=dropout_rate,
        #                        weight_constraint=weight_constraint)
        # grid = GridSearchCV(estimator=model_ann, cv=5, param_grid=hyperparameters, n_jobs=-1)
        # # Fit grid search
        # grid_result = grid.fit(x_training, y_training)
        # # View hyperparameters of best neural network
        # # summarize results
        # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # text_file.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_) + "\n")
        # means = grid_result.cv_results_['mean_test_score']
        # stds = grid_result.cv_results_['std_test_score']
        # params = grid_result.cv_results_['params']
        # for mean, stdev, param in zip(means, stds, params):
        #     print("%f (%f) with: %r" % (mean, stdev, param))
        #     text_file.write("%f (%f) with: %r" % (mean, stdev, param) + "\n")
            
            
        # from model_NN import model_arch_general_neurons
        # # =============================================================================
        # # Tuning Weight initialization for Kernal
        # # =============================================================================
        # # Wrap Keras model so it can be used by scikit-learn
        # model_ann = KerasClassifier(build_fn=model_arch_general_neurons, verbose=0)
        # # Create hyperparameter space
        # neurons = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
        # # Create hyperparameter options
        # hyperparameters = dict(neurons=neurons)
        # grid = GridSearchCV(estimator=model_ann, cv=5, param_grid=hyperparameters, n_jobs=-1)
        # # Fit grid search
        # grid_result = grid.fit(x_training, y_training)
        # # View hyperparameters of best neural network
        # # summarize results
        # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # text_file.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_) + "\n")
        # means = grid_result.cv_results_['mean_test_score']
        # stds = grid_result.cv_results_['std_test_score']
        # params = grid_result.cv_results_['params']
        # for mean, stdev, param in zip(means, stds, params):
        #     print("%f (%f) with: %r" % (mean, stdev, param))
        #     text_file.write("%f (%f) with: %r" % (mean, stdev, param) + "\n")
        
        
        # from model_NN import model_arch_general_layers
        # # =============================================================================
        # # Tuning Weight initialization for Kernal
        # # =============================================================================
        # # Wrap Keras model so it can be used by scikit-learn
        # model_ann = KerasClassifier(build_fn=model_arch_general_layers, verbose=0)
        # # Create hyperparameter space
        # layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # # Create hyperparameter options
        # hyperparameters = dict(layers=layers)
        # grid = GridSearchCV(estimator=model_ann, cv=5, param_grid=hyperparameters, n_jobs=-1)
        # # Fit grid search
        # grid_result = grid.fit(x_training, y_training)
        # # View hyperparameters of best neural network
        # # summarize results
        # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # text_file.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_) + "\n")
        # means = grid_result.cv_results_['mean_test_score']
        # stds = grid_result.cv_results_['std_test_score']
        # params = grid_result.cv_results_['params']
        # for mean, stdev, param in zip(means, stds, params):
        #     print("%f (%f) with: %r" % (mean, stdev, param))
        #     text_file.write("%f (%f) with: %r" % (mean, stdev, param) + "\n")
            
            
        text_file.close()
        



