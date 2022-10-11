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
        from lauetoolsnn.NNmodels import model_arch_general
    except:
        # else import from a path where LaueToolsNN files are
        import sys
        sys.path.append(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn")
        from utils_lauenn import array_generator, array_generator_verify, vali_array
        from NNmodels import model_arch_general
    
    # ## step 1: define material and path to access the training dataset generated using Step 1 script
    
    # In[2]:
    
    
    # =============================================================================
    ## User Input dictionary with parameters (reduced but same as the one used in STEP 1)
    ## In case of only one phase/material, keep same value for material_ and material1_ key
    # =============================================================================
    input_params = {
                    "material_": "alpha_MoO3",             ## same key as used in dict_LaueTools
                    "material1_": "PMNPT",            ## same key as used in dict_LaueTools
                    "prefix" : "",                 ## prefix for the folder to be created for training dataset  
                    "nb_grains_per_lp_mat0" : 1,        ## max grains to be generated in a Laue Image
                    "nb_grains_per_lp_mat1" : 1,        ## max grains to be generated in a Laue Image
                    "grains_nb_simulate" : 1000,    ## Number of orientations to generate (takes advantage of crystal symmetry)
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
    
    # In[3]:
    
    
    classhkl = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_0"]
    angbins = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_1"]
    loc_new = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_2"]
    with open(save_directory+"//class_weights.pickle", "rb") as input_file:
        class_weights = cPickle.load(input_file)
    class_weights = class_weights[0]
    
    # ## Step 3: Defining a neural network architecture
    
    ###from NNmodel.py script or define here
    
    
    # ## Step 4: Training  
    
    # In[5]:
    
    
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
    
    # In[6]:
    
    
    from sklearn.metrics import classification_report
    
    ## verify the 
    x_test, y_test = vali_array(save_directory+"//testing_data", 50, len(classhkl), loc_new, print)
    y_test = np.argmax(y_test, axis=-1)
    y_pred = np.argmax(model.predict(x_test), axis=-1)
    print(classification_report(y_test, y_pred))


# In[ ]:




