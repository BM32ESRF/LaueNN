#!/usr/bin/env python
# coding: utf-8

# # Notebook script for Grid search optimization of hyperperameters of neural network architecture
# 
# ### Load the data generated in Step 1
# ### Define the Neural network architecture (with hyper parameters to be optimized)
# ### Run Sklearn GridSearch
# 

# In[ ]:

if __name__ == '__main__':     #enclosing required because of multiprocessing

    ## Import modules used for this Notebook
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    
    ## if LaueToolsNN is properly installed
    try:
        from lauetoolsnn.utils_lauenn import vali_array
    except:
        # else import from a path where LaueToolsNN files are
        import sys
        sys.path.append(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn")
        from utils_lauenn import vali_array
    
    
    # ## step 1: define material and path to access the training dataset generated using Step 1 script
    
    # In[ ]:
    
    
    # =============================================================================
    ## User Input dictionary with parameters (reduced but same as the one used in STEP 1)
    ## In case of only one phase/material, keep same value for material_ and material1_ key
    # =============================================================================
    input_params = {
                    "material_": "Cu",             ## same key as used in dict_LaueTools
                    "material1_": "Cu",            ## same key as used in dict_LaueTools
                    "prefix" : "",                 ## prefix for the folder to be created for training dataset
                    "nb_grains_per_lp" : 5,        ## max grains to be generated in a Laue Image
                    "grains_nb_simulate" : 100,    ## Number of orientations to generate (takes advantage of crystal symmetry)
                    "batch_size":50,               ## batches of files to use while training
                    "epochs":5,                    ## number of epochs for training
                    }
    
    material_= input_params["material_"]
    material1_= input_params["material1_"]
    nb_grains_per_lp = input_params["nb_grains_per_lp"]
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
    # ### Loading the Output class and ground truth; loading the training dataset of user defined batch number
    
    # In[ ]:
    
    
    classhkl = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_0"]
    angbins = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_1"]
    loc_new = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_2"]
    n_bins = len(angbins)-1
    n_outputs = len(classhkl)
    print(n_bins, n_outputs)
          
     ## Tuning on small dataset of Cu (20 corresponds to number of files)
    x_training, y_training = vali_array(save_directory+"//training_data", 20, 
                                        len(classhkl), 
                                        loc_new, print, tocategorical=False)
    print("Number of spots in a batch of %i files : %i" %(20, len(x_training)))
    print("Min, Max class ID is %i, %i" %(np.min(y_training), np.max(y_training)))
    
    
    # ## Step 3: Defining a neural network architecture with hyperparameters as free parameters
    
    # In[13]:
    
    
    import tensorflow as tf
    from tensorflow.keras.layers import BatchNormalization
    import keras
    from keras.regularizers import l2
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout
    from keras.constraints import maxnorm
    
    ####################################################################################
    ## General architecture with all free parameters
    ####################################################################################
    def model_arch_general(kernel_coeff = 0.0005, 
                           bias_coeff = 0.0005, 
                           init_mode='uniform',
                           learning_rate=0.0001, 
                           neurons_multiplier= [1200, 1200, (32*2*7)+(1200//2), (32*2*15), 32], 
                           layers=3, 
                           batch_norm=False, 
                           optimizer="adam",
                           activation='relu',
                           dropout_rate=0.0, 
                           weight_constraint=0):
        model = Sequential()
        # Input layer
        model.add(keras.Input(shape=(int(neurons_multiplier[0]),)))
        
        if layers > 0:
            for lay in range(layers):
                ## Hidden layer n
                if kernel_coeff == None and bias_coeff == None and                                weight_constraint == None and init_mode == None:
                    model.add(Dense(int(neurons_multiplier[lay+1]),))
                    
                elif kernel_coeff == None and bias_coeff == None and                                weight_constraint == None and init_mode != None:
                    model.add(Dense(int(neurons_multiplier[lay+1]), 
                                    kernel_initializer=init_mode))
                    
                elif kernel_coeff == None and bias_coeff == None and                                init_mode != None:
                    model.add(Dense(int(neurons_multiplier[lay+1]), 
                                    kernel_initializer=init_mode,
                                    kernel_constraint=maxnorm(weight_constraint)))
                
                elif weight_constraint == None and init_mode != None:
                    model.add(Dense(int(neurons_multiplier[lay+1]), 
                                    kernel_initializer=init_mode,
                                    kernel_regularizer=l2(kernel_coeff), 
                                    bias_regularizer=l2(bias_coeff),))
                
                elif init_mode == None and weight_constraint == None:
                    model.add(Dense(int(neurons_multiplier[lay+1]), 
                                    kernel_regularizer=l2(kernel_coeff), 
                                    bias_regularizer=l2(bias_coeff),))
                    
                elif kernel_coeff != None and bias_coeff != None and                                weight_constraint != None and init_mode != None:
                    model.add(Dense(int(neurons_multiplier[lay+1]), 
                                    kernel_initializer=init_mode,
                                    kernel_regularizer=l2(kernel_coeff), 
                                    bias_regularizer=l2(bias_coeff), 
                                    kernel_constraint=maxnorm(weight_constraint)))
                else:
                    print("condition not satisfied")
                
                if batch_norm:
                    model.add(BatchNormalization())
                model.add(Activation(activation))
                model.add(Dropout(dropout_rate))
        ## Output layer 
        model.add(Dense(int(neurons_multiplier[-1]), activation='softmax'))
        if (optimizer == "adam" or optimizer == "Adam") and learning_rate != None:
            opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        else:
            opt = optimizer
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
        model.summary()
        return model
    
    ####################################################################################
    ## General architecture with dropout as free parameters
    ####################################################################################
    def model_arch_general_dropout(n_bins, n_outputs, dropout = 0.0):
        """
        Very simple and straight forward Neural Network with few hyperparameters
        straighforward RELU activation strategy with cross entropy to identify the HKL
        Tried BatchNormalization --> no significant impact
        Tried weighted approach --> not better for HCP
        Trying Regularaization 
        l2(0.001) means that every coefficient in the weight matrix of the layer 
        will add 0.001 * weight_coefficient_value**2 to the total loss of the network
        """
        if n_outputs >= n_bins:
            param = n_bins
            if param*15 < (2*n_outputs): ## quick hack; make Proper implementation
                param = (n_bins + n_outputs)//2
        else:
            # param = n_outputs ## More reasonable ???
            param = n_outputs*2 ## More reasonable ???
            # param = n_bins//2
            
        model = Sequential()
        model.add(keras.Input(shape=(n_bins,)))
        ## Hidden layer 1
        model.add(Dense(n_bins, kernel_regularizer=l2(0.0005), bias_regularizer=l2( 0.0005)))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dropout)) ## Adding dropout as we introduce some uncertain data with noise
        ## Hidden layer 2
        model.add(Dense(((param)*15 + n_bins)//2, kernel_regularizer=l2( 0.0005), bias_regularizer=l2( 0.0005)))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
        ## Hidden layer 3
        model.add(Dense((param)*15, kernel_regularizer=l2( 0.0005), bias_regularizer=l2( 0.0005)))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
        ## Output layer 
        model.add(Dense(n_outputs, activation='softmax'))
        ## Compile model
        otp = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=otp, metrics=["accuracy"])
        # model.summary()
        return model
    
    
    # ## Step 4: Define hyperparameters and launch grid search  
    
    # In[14]:
    
    
    from sklearn.model_selection import GridSearchCV
    from keras.wrappers.scikit_learn import KerasClassifier
    
    if material_ != material1_:
        text_file = open(save_directory+"//grid_optimizer_logger_"+material_+"_"+material1_+".txt", "w")
    else:
        text_file = open(save_directory+"//grid_optimizer_logger_"+material_+".txt", "w")

    # =============================================================================
    # Tuning neural network parameters
    # Play with fixing and freeing different parameters
    # =============================================================================

    # Wrap Keras model so it can be used by scikit-learn
    model_ann = KerasClassifier(build_fn=model_arch_general_dropout, verbose=0)
    
    # Create hyperparameter space
    optimizer = ['adam']
    learning_rate = [0.001] #[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    init_mode = [None] #['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    activation = ['relu'] # ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    dropout_rate = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    weight_constraint = [None] #[1, 2, 3, 4, 5, 6, 7, 8, 9]
    layers = [3]
    kernel_coeff = [0.0005]
    bias_coeff = [0.0005]
    first_layer = int(n_outputs * 2)
    neurons_multiplier = [[int(n_bins), int(n_bins), int(first_layer*7+n_bins/2), int(first_layer*15), int(n_outputs)],]
    batch_norm = [False]

    # Create hyperparameter options
    #hyperparameters = dict( 
    #                        kernel_coeff=kernel_coeff,
    #                        bias_coeff=bias_coeff,
    #                        init_mode=init_mode,
    #                        learning_rate=learning_rate,
    #                        neurons_multiplier=neurons_multiplier, 
    #                        layers=layers,
    #                        batch_norm=batch_norm,
    #                        optimizer=optimizer,
    #                        activation=activation,
    #                        dropout_rate=dropout_rate,
    #                        weight_constraint=weight_constraint,
    #                        )
    hyperparameters = dict( 
                            n_bins=[n_bins],
                            n_outputs=[n_outputs],
                            dropout=dropout_rate,
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
    text_file.close()


# In[ ]:




