#!/usr/bin/env python
# coding: utf-8

# # Notebook script for Training the neural network (supports single and two phase material)
# 
# ## Different steps of neural network training is outlined in this notebook (LaueToolsNN GUI does the same thing)
# 
# ### Load the data generated in Step 1
# ### Define the Neural network architecture
# ### Train the network
# 

# In[1]:

if __name__ == '__main__':     #enclosing required because of multiprocessing

    ## Import modules used for this Notebook
    import numpy as np
    import os
    import _pickle as cPickle
    import itertools
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    import matplotlib.pyplot as plt
    
    ## if LaueToolsNN is properly installed
    try:
        from lauetoolsnn.utils_lauenn import array_generator, array_generator_verify, vali_array
    except:
        # else import from a path where LaueToolsNN files are
        import sys
        sys.path.append(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn")
        from utils_lauenn import array_generator, array_generator_verify, vali_array
    
    
    # ## step 1: define material and path to access the training dataset generated using Step 1 script
    
    # In[2]:
    
    
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
    # ### Loading the Output class and ground truth
    
    # In[3]:
    
    
    classhkl = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_0"]
    angbins = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_1"]
    loc_new = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_2"]
    with open(save_directory+"//class_weights.pickle", "rb") as input_file:
        class_weights = cPickle.load(input_file)
    class_weights = class_weights[0]
    
    n_bins = len(angbins)-1
    n_outputs = len(classhkl)
    print(n_bins, n_outputs)
    
    
    # ## Step 3: Defining a neural network architecture
    
    # In[4]:
    
    
    import tensorflow as tf
    import keras
    from keras.regularizers import l2
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout
    
    metricsNN = [
                keras.metrics.FalseNegatives(name="fn"),
                keras.metrics.FalsePositives(name="fp"),
                keras.metrics.TrueNegatives(name="tn"),
                keras.metrics.TruePositives(name="tp"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="accuracy"),
                ]
    
    def model_arch_general_optimized(n_bins, n_outputs, kernel_coeff = 0.0005, bias_coeff = 0.0005, lr=None, verbose=1,
                           write_to_console=None):
        """
        Very simple and straight forward Neural Network with few hyperparameters
        straighforward RELU activation strategy with cross entropy to identify the HKL
        Tried BatchNormalization --> no significant impact
        Tried weighted approach --> not better for HCP
        Trying Regularaization 
        l2(0.001) means that every coefficient in the weight matrix of the layer 
        will add 0.001 * weight_coefficient_value**2 to the total loss of the network
        1e-3,1e-5,1e-6
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
        model.add(Dense(n_bins, kernel_regularizer=l2(kernel_coeff), bias_regularizer=l2(bias_coeff)))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3)) ## Adding dropout as we introduce some uncertain data with noise
        ## Hidden layer 2
        model.add(Dense(((param)*15 + n_bins)//2, kernel_regularizer=l2(kernel_coeff), bias_regularizer=l2(bias_coeff)))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        ## Hidden layer 3
        model.add(Dense((param)*15, kernel_regularizer=l2(kernel_coeff), bias_regularizer=l2(bias_coeff)))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        ## Output layer 
        model.add(Dense(n_outputs, activation='softmax'))
        ## Compile model
        if lr != None:
            otp = tf.keras.optimizers.Adam(learning_rate=lr)
            model.compile(loss='categorical_crossentropy', optimizer=otp, metrics=[metricsNN])
        else:
            model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=[metricsNN])
        
        if verbose == 1:
            model.summary()
            stringlist = []
            model.summary(print_fn=lambda x: stringlist.append(x))
            short_model_summary = "\n".join(stringlist)
            if write_to_console!=None:
                write_to_console(short_model_summary)
        return model
    
    
    # ## Step 4: Training  
    
    # In[5]:
    
    
    # load model and train
    #neurons_multiplier is a list with number of neurons per layer, the first value is input shape and last value is output shape, inbetween are the number of neurons per hidden layers
    model = model_arch_general_optimized(  n_bins, n_outputs,
                                           kernel_coeff = 1e-5,
                                           bias_coeff = 1e-6,
                                           lr = 1e-3,
                                            )
    #model = model_arch_general_optimized(n_bins, n_outputs, kernel_coeff = 0.0005, bias_coeff = 0.0005, lr=None, verbose=1,)
    ## temp function to quantify the spots and classes present in a batch
    batch_size = input_params["batch_size"] 
    trainy_inbatch = array_generator_verify(save_directory+"//training_data", batch_size, 
                                            len(classhkl), loc_new, print)
    print("Number of spots in a batch of %i files : %i" %(batch_size, len(trainy_inbatch)))
    print("Min, Max class ID is %i, %i" %(np.min(trainy_inbatch), np.max(trainy_inbatch)))
    
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
    
    # In[6]:
    
    
    from sklearn.metrics import classification_report
    
    ## verify the 
    x_test, y_test = vali_array(save_directory+"//testing_data", 50, len(classhkl), loc_new, print)
    y_test = np.argmax(y_test, axis=-1)
    y_pred = np.argmax(model.predict(x_test), axis=-1)
    print(classification_report(y_test, y_pred))


# In[ ]:




