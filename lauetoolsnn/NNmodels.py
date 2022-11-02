# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 23:04:13 2022

@author: PURUSHOT

NN models and their prediction with numpy based calculations
Keras and tensorflow gets locked for prediction in Multi-thread mode

Numpy version of prediction for both DNN and CNN models are implemented, but bit slow
but we can use multiprocessing

"""
__author__ = "Ravi raj purohit PURUSHOTTAM RAJ PUROHIT, CRG-IF BM32 @ ESRF"

import numpy as np
import os
## Keras import
tensorflow_keras = True
try:
    import tensorflow as tf
    import keras
    from keras.models import Sequential
    from tensorflow.keras.callbacks import Callback
    from keras.layers import Dense, Activation, Dropout
    from keras.regularizers import l2
    from keras.models import model_from_json
    from tensorflow.keras import Model
    # from tf.keras.layers.normalization import BatchNormalization
except:
    print("tensorflow not loaded; Training will not work")
    tensorflow_keras = False

try:
    import h5py
except:
    print("H5PY loading failed, prediction may not work")
    print("Issue with conda's installation of H5py due to incompatibility between HDF5 and H5py package")
    print("Please uninstall h5py with conda remove --force h5py")
    print("And install via pip with pip install h5py, this should solve the issue")

## GPU Nvidia drivers needs to be installed! Ughh
## if wish to use only CPU set the value to -1 else set it to 0 for GPU
## CPU training is suggested (as the model requires more RAM)
try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

metricsNN = [
            keras.metrics.FalseNegatives(name="fn"),
            keras.metrics.FalsePositives(name="fp"),
            keras.metrics.TrueNegatives(name="tn"),
            keras.metrics.TruePositives(name="tp"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="accuracy"),
            ]

class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """
    def __init__(self, print_fcn, progress_func, qapp, model, fn_model):
        Callback.__init__(self)
        self.print_fcn = print_fcn
        self.progress_func = progress_func
        self.batch_count = 0
        self.qapp = qapp
        self.model = model
        self.model_name = fn_model
    
    def on_batch_end(self, batch, logs={}):
        self.batch_count += 1
        if self.progress_func != None:
            self.progress_func.setValue(self.batch_count)
        if self.qapp != None:
            self.qapp.processEvents()
        
    def on_epoch_end(self, epoch, logs={}):
        msg = "{Epoch: %i} %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
        if self.print_fcn != None:
            self.print_fcn(msg)
        model_json = self.model.to_json()
        with open(self.model_name+".json", "w") as json_file:
            json_file.write(model_json)            
        # serialize weights to HDF5
        self.model.save_weights(self.model_name+"_"+str(epoch)+".h5")
        
def model_arch_general(n_bins, n_outputs, kernel_coeff = 0.0005, bias_coeff = 0.0005, lr=None, verbose=1,
                       write_to_console=None):
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
        param = n_outputs ## More reasonable ???
        # param = n_outputs*2 ## More reasonable ???
        # param = n_bins//2
        
    model = Sequential()
    model.add(keras.Input(shape=(n_bins,)))
    ## Hidden layer 1
    if kernel_coeff != None and bias_coeff !=None:
        model.add(Dense(n_bins, kernel_regularizer=l2(kernel_coeff), bias_regularizer=l2(bias_coeff)))
    else:
        model.add(Dense(n_bins,))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3)) ## Adding dropout as we introduce some uncertain data with noise
    ## Hidden layer 2
    if kernel_coeff != None and bias_coeff !=None:
        model.add(Dense(((param)*15 + n_bins)//2, kernel_regularizer=l2(kernel_coeff), bias_regularizer=l2(bias_coeff)))
    else:
        model.add(Dense(((param)*15 + n_bins)//2,))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    ## Hidden layer 3
    if kernel_coeff != None and bias_coeff !=None:
        model.add(Dense((param)*15, kernel_regularizer=l2(kernel_coeff), bias_regularizer=l2(bias_coeff)))
    else:
        model.add(Dense((param)*15,))
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

def model_arch_CNN_DNN_optimized(shape, 
                                 layer_activation="relu", 
                                 output_activation="softmax",
                                 dropout=0.3,
                                 stride = [1,1],
                                 kernel_size = [5,5],
                                 pool_size=[2,2],
                                 CNN_layers = 2,
                                 CNN_filters = [32,64],
                                 DNN_layers = 3,
                                 DNN_filters = [1000,500,100],
                                 output_neurons = 11,
                                 learning_rate = 0.001,
                                 output="DNN", 
                                 write_to_console=None, 
                                 verbose=1):            
    inputs = keras.layers.Input(shape, name="InputLayer")
    
    for lay in range(CNN_layers):
        if lay == 0:
            conv1 = keras.layers.Conv1D(filters=CNN_filters[lay], kernel_size=kernel_size[lay], 
                                        strides=stride[lay], 
                                        activation=layer_activation, name="Conv_"+str(lay+1))(inputs)
            if pool_size[lay] != 1 or pool_size[lay] != 0:
                pool1 = keras.layers.MaxPooling1D(pool_size=pool_size[lay], \
                                                  name="Pool_"+str(lay+1))(conv1)
        else:
            conv1 = keras.layers.Conv1D(filters=CNN_filters[lay], kernel_size=kernel_size[lay], 
                                        strides=stride[lay], 
                                        activation=layer_activation, name="Conv_"+str(lay+1))(pool1)
            if pool_size[lay] != 1 or pool_size[lay] != 0:
                pool1 = keras.layers.MaxPooling1D(pool_size=pool_size[lay], \
                                                  name="Pool_"+str(lay+1))(conv1)
    flatten = keras.layers.Flatten(name="Flatten")(pool1)

    for lay in range(DNN_layers):
        if lay == 0:
            ppKL = keras.layers.Dense(DNN_filters[lay], activation=layer_activation, \
                                      name="Dense_"+str(lay+1))(flatten)
            ppKL = keras.layers.Dropout(dropout, name="Dropout"+str(lay+1))(ppKL)   
        else:
            
            ppKL = keras.layers.Dense(DNN_filters[lay], activation=layer_activation, \
                                      name="Dense_"+str(lay+1))(ppKL)
            ppKL = keras.layers.Dropout(dropout, name="Dropout"+str(lay+1))(ppKL) 
    ## Output layer 
    if output != "CNN":
        if DNN_layers == 0:
            outputs = keras.layers.Dense(output_neurons, activation=output_activation, name="Dense_out")(flatten)
        else:
            outputs = keras.layers.Dense(output_neurons, activation=output_activation, name="Dense_out")(ppKL)
    else:
        outputs = keras.layers.Conv1D(filters=output_neurons, kernel_size=1, 
                                    strides=1, activation=output_activation, name="Conv_out")(flatten)
    model = Model(inputs, outputs)
    ## Compile model
    otp = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=otp, metrics=[metricsNN])

    if verbose == 1:
        model.summary()
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        if write_to_console!=None:
            write_to_console(short_model_summary)
    return model

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
        param = n_outputs
        
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

def model_arch_general_onelayer(n_bins, n_outputs, 
                                kernel_coeff = 0.0005, 
                                bias_coeff = 0.0005, lr=None, verbose=1,
                                write_to_console=None):
    """
    model_arch_general_onelayer
    Very simple and straight forward Neural Network with few hyperparameters
    straighforward RELU activation strategy with cross entropy to identify the HKL
    Tried BatchNormalization --> no significant impact
    Tried weighted approach --> not better for HCP
    Trying Regularaization 
    l2(0.001) means that every coefficient in the weight matrix of the layer 
    will add 0.001 * weight_coefficient_value**2 to the total loss of the network
    1e-3,1e-5,1e-6
    """
    model = Sequential()
    model.add(keras.Input(shape=(n_bins,)))
    ## Hidden layer 1
    model.add(Dense(n_bins, kernel_regularizer=l2(kernel_coeff), bias_regularizer=l2(bias_coeff)))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3)) ## Adding dropout as we introduce some uncertain data with noise
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

def read_hdf5(path):
    weights = {}
    keys = []
    with h5py.File(path, 'r') as f: # open file
        f.visit(keys.append) # append all keys to list
        for key in keys:
            if ':' in key: # contains data if ':' in key
                weights[f[key].name] = f[key][:]
    return weights

def softmax(x):
    return (np.exp(x).T / np.sum(np.exp(x).T, axis=0)).T

def predict_DNN(x, wb, temp_key):
    # first layer
    layer0 = np.dot(x, wb[temp_key[1]]) + wb[temp_key[0]] 
    layer0 = np.maximum(0, layer0) ## ReLU activation
    # Second layer
    layer1 = np.dot(layer0, wb[temp_key[3]]) + wb[temp_key[2]] 
    layer1 = np.maximum(0, layer1)
    # Third layer
    layer2 = np.dot(layer1, wb[temp_key[5]]) + wb[temp_key[4]]
    layer2 = np.maximum(0, layer2)
    # Output layer
    layer3 = np.dot(layer2, wb[temp_key[7]]) + wb[temp_key[6]]
    layer3 = softmax(layer3) ## output softmax activation
    return layer3

def predict_DNN_onelayer(x, wb, temp_key):
    # first layer
    layer0 = np.dot(x, wb[temp_key[1]]) + wb[temp_key[0]] 
    layer0 = np.maximum(0, layer0) ## ReLU activation
    # Second layer
    layer1 = np.dot(layer0, wb[temp_key[3]]) + wb[temp_key[2]] 
    layer3 = softmax(layer1) ## output softmax activation
    return layer3

def predict_with_file(x, model_direc=None, material_=None):
    if model_direc!=None and material_!=None:
        if len(material_) > 1:
            prefix_mat = material_[0]
            for ino, imat in enumerate(material_):
                if ino == 0:
                    continue
                prefix_mat = prefix_mat + "_" + imat
        else:
            prefix_mat = material_[0]
        
        json_file = open(model_direc+"//model_"+prefix_mat+".json", 'r')
        load_weights = model_direc + "//model_"+prefix_mat+".h5"
        # # load json and create model
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(load_weights)
        return model.predict(x)

# =============================================================================
# 1D CNN numpy implementation
# =============================================================================
    
def asStride(arr, sub_shape, stride):
    '''Get a strided sub-matrices view of an ndarray.
    Args:
        arr (ndarray): input array of rank 4, with shape (m, hi, wi, ci).
        sub_shape (tuple): window size: (f1, f2).
        stride (int): stride of windows in both 2nd and 3rd dimensions.
    Returns:
        subs (view): strided window view.
    This is used to facilitate a vectorized 3d convolution.
    The input array <arr> has shape (m, hi, wi, ci), and is transformed
    to a strided view with shape (m, ho, wo, f, f, ci). where:
        m: number of records.
        hi, wi: height and width of input image.
        ci: channels of input image.
        f: kernel size.
    The convolution kernel has shape (f, f, ci, co).
    Then the vectorized 3d convolution can be achieved using either an einsum()
    or a tensordot():
        conv = np.einsum('myxfgc,fgcz->myxz', arr_view, kernel)
        conv = np.tensordot(arr_view, kernel, axes=([3, 4, 5], [0, 1, 2]))
    See also skimage.util.shape.view_as_windows()
    '''
    if len(arr.shape) == 2:
        m, hi  = arr.shape
        ci = 1
        sm, sh = arr.strides
        sc = 1
    else:
        m, hi, ci  = arr.shape
        sm, sh, sc = arr.strides    
    view_shape = (m, 1+(hi-sub_shape)//stride, sub_shape, ci)
    strides = (sm, stride*sh, sh, sc)
    subs = np.lib.stride_tricks.as_strided(arr, view_shape, strides=strides, writeable=False)
    return subs

def calculate_output_dims(input_dims, _stride, _w, _padding):
    """
    :param input_dims - 4 element tuple (n, h_in, w_in, c)
    :output 4 element tuple (n, h_out, w_out, n_f)
    ------------------------------------------------------------------------
    n - number of examples in batch
    w_in - width of input volume
    h_in - width of input volume
    w_out - width of input volume
    h_out - width of input volume
    c - number of channels of the input volume
    n_f - number of filters in filter volume
    """
    if len(input_dims) == 2:
        n, h_in = input_dims
    elif len(input_dims) == 3:
        n, h_in, _ = input_dims
    else:
        print("unsupported input shape")
        
    h_f, w_f, n_f = _w.shape
    if _padding == 'same':
        return n, h_in, n_f
    elif _padding == 'valid':
        h_out = (h_in - h_f) // _stride + 1
        return n, h_out, n_f
    else:
        print(f"Unsupported padding value: {_padding}")

def ReluLayer_forward_pass(a_prev):
    """
    :param a_prev - ND tensor with shape (n, ..., channels)
    :output ND tensor with shape (n, ..., channels)
    ------------------------------------------------------------------------
    n - number of examples in batch
    """
    _z = np.maximum(0, a_prev)
    return _z

def softmax_forward_pass(a_prev):
    """
    :param a_prev - 2D tensor with shape (n, k)
    :output 2D tensor with shape (n, k)
    ------------------------------------------------------------------------
    n - number of examples in batch
    k - number of classes
    (np.exp(x).T / np.sum(np.exp(x).T, axis=0)).T
    """
    e = np.exp(a_prev - a_prev.max(axis=1, keepdims=True))
    _z = e / np.sum(e, axis=1, keepdims=True)
    return _z

def flatten_forward_pass(a_prev):
    """
    :param a_prev - ND tensor with shape (n, ..., channels)
    :output - 1D tensor with shape (n, 1)
    ------------------------------------------------------------------------
    n - number of examples in batch
    """
    return np.ravel(a_prev).reshape(a_prev.shape[0], -1)

def pooling_forward_pass(a_prev, _stride, _pool_size):
    """
    :param a_prev - 4D tensor with shape(n, h_in, w_in, c)
    :output 4D tensor with shape(n, h_out, w_out, c)
    ------------------------------------------------------------------------
    n - number of examples in batch
    w_in - width of input volume
    h_in - width of input volume
    c - number of channels of the input/output volume
    w_out - width of output volume
    h_out - width of output volume
    """
    subs = asStride(a_prev, _pool_size, _stride)
    output = subs.max(axis=2)
    return output

def forward_pass_DNN(a_prev, _w, _b):
    """
    :param a_prev - 2D tensor with shape (n, units_prev)
    :output - 2D tensor with shape (n, units_curr)
    ------------------------------------------------------------------------
    n - number of examples in batch
    units_prev - number of units in previous layer
    units_curr -  number of units in current layer    
    """
    return np.dot(a_prev, _w) + _b

def forward_pass_CNN(a_prev, _w, _b, _stride):
    """
    :param a_prev - 4D tensor with shape (n, h_in, w_in, c)
    :output 4D tensor with shape (n, h_out, w_out, n_f)
    ------------------------------------------------------------------------
    n - number of examples in batch
    w_in - width of input volume
    h_in - width of input volume
    w_out - width of input volume
    h_out - width of input volume
    c - number of channels of the input volume
    n_f - number of filters in filter volume
    """
    output_shape = calculate_output_dims(a_prev.shape, _stride, _w, 'valid')        
    _, h_out, _ = output_shape
    h_f, _, _ = _w.shape
    subs = asStride(a_prev, h_f, _stride)
    # output = np.einsum("ijkl,klm->ijm", subs, _w) + _b
    output = np.tensordot(subs, _w, axes=([2, 3], [0, 1])) + _b
    return output


def predict_CNN_DNN(x, wb, temp_key):
    """
    2 CNN layers[128,128] (stride[5,2], kernel[10,3] + maxpooling[2,1]) + Relu activation
    flatten
    Outpur DNN layer (softmax activation)
    
    """
    x = forward_pass_CNN(x, wb[temp_key[1]], wb[temp_key[0]], 5)
    x = ReluLayer_forward_pass(x)
    x = pooling_forward_pass(x, 2, 2)
    
    x = forward_pass_CNN(x, wb[temp_key[3]], wb[temp_key[2]], 2)
    x = ReluLayer_forward_pass(x)
    x = pooling_forward_pass(x, 1, 1)
    
    x = flatten_forward_pass(x)
    
    x = forward_pass_DNN(x, wb[temp_key[5]], wb[temp_key[4]])
    x = softmax_forward_pass(x)
    return x


# =============================================================================
# USER DEFINED MODEL
# =============================================================================
def user_defined_model(n_bins, n_outputs, 
                                kernel_coeff = 0.0005, 
                                bias_coeff = 0.0005, lr=None, verbose=1,
                                write_to_console=None):
    """
    model_arch_general_onelayer as user defined mode
    
    Please add any different architecture that you would like to experiment
    
    Dont forget also to update the prediction for it with numpy functions
    to use the multiprocessing. otherwise Keras implementation of predictions
    will lock the cpu and will not yield any results
    
    To see some implementation with numpy please check the predict_DNN function
    """
    model = Sequential()
    model.add(keras.Input(shape=(n_bins,)))
    ## Hidden layer 1
    model.add(Dense(n_bins, kernel_regularizer=l2(kernel_coeff), bias_regularizer=l2(bias_coeff)))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3)) ## Adding dropout as we introduce some uncertain data with noise
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



# if __name__ == "__main__":
#     pass
    # # test of numpy prediction
    # codebar = np.load(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\multi_mat_LaueNN\Zr_alpha_ZrO2_mono\model_Zr_alpha_ZrO2_mono.npz", allow_pickle=True)["arr_0"]
    # prediction = np.load(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\multi_mat_LaueNN\Zr_alpha_ZrO2_mono\model_Zr_alpha_ZrO2_mono.npz", allow_pickle=True)["arr_1"]

    # # =============================================================================
    # #     ## keras tensorflow format
    # # =============================================================================
    # json_file = open(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\multi_mat_LaueNN\Zr_alpha_ZrO2_mono\model_Zr_alpha_ZrO2_mono.json", 'r')
    # load_weights = r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\multi_mat_LaueNN\Zr_alpha_ZrO2_mono\model_Zr_alpha_ZrO2_mono.h5"
    # # # load json and create model
    # loaded_model_json = json_file.read()
    # json_file.close()
    # model = model_from_json(loaded_model_json)
    # model.load_weights(load_weights)
    # prediction1 = model.predict(codebar)
    
    # assert np.all(prediction == prediction1)
    
    # max_pred = np.max(prediction, axis = 1)
    # class_predicted = np.argmax(prediction, axis = 1)
    
    # # =============================================================================
    # #     ### Numpy format CNN
    # # =============================================================================
    # ##Model weights 
    # model_weights = read_hdf5(load_weights)
    # model_key = list(model_weights.keys())
    # prediction2 = predict_CNN_DNN(codebar, model_weights, model_key)
    
    # max_pred1 = np.max(prediction2, axis = 1)
    # class_predicted1 = np.argmax(prediction2, axis = 1)
    
    # assert np.all(class_predicted == class_predicted1)



