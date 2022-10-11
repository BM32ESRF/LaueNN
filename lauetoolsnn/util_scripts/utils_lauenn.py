# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 14:47:33 2021

@author: PURUSHOT

Functions for lauetoolsneuralnetwork
"""
__author__ = "Ravi raj purohit PURUSHOTTAM RAJ PUROHIT, CRG-IF BM32 @ ESRF"

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import logging
logger = logging.getLogger()
old_level = logger.level
logger.setLevel(100)

# import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Qt5Agg')
# matplotlib.rcParams.update({'font.size': 14})
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
from random import random as rand1
from math import acos
import time
import enum
import functools
import math
from numpy import pi, dot
import scipy
# from scipy.spatial.transform import Rotation as R
import _pickle as cPickle
import configparser

from multiprocessing import Process, Queue, cpu_count
from tqdm import tqdm

from skimage.transform import (hough_line, hough_line_peaks)
# =============================================================================
# Additonal networkx module
import networkx as nx
# =============================================================================
## LaueTools import
try:
    from lauetools import dict_LaueTools as dictLT
    from lauetools import IOLaueTools as IOLT
    from lauetools import generaltools as GT
    from lauetools import CrystalParameters as CP
    from lauetools import lauecore as LT
    from lauetools import LaueGeometry as Lgeo
    from lauetools import readmccd as RMCCD
    from lauetools import FitOrient as FitO
    from lauetools import findorient as FindO
    from lauetools import IOimagefile as IOimage   
except:
    import lauetoolsnn.lauetools.dict_LaueTools as dictLT
    import lauetoolsnn.lauetools.IOLaueTools as IOLT
    import lauetoolsnn.lauetools.generaltools as GT
    import lauetoolsnn.lauetools.CrystalParameters as CP
    import lauetoolsnn.lauetools.lauecore as LT
    import lauetoolsnn.lauetools.LaueGeometry as Lgeo
    import lauetoolsnn.lauetools.readmccd as RMCCD
    import lauetoolsnn.lauetools.FitOrient as FitO
    import lauetoolsnn.lauetools.findorient as FindO
    import lauetoolsnn.lauetools.IOimagefile as IOimage

from collections import OrderedDict
from math import cos, radians, sin, sqrt
import fractions
import collections
import random, itertools
import re

## Keras import
tensorflow_keras = True
try:
    import tensorflow as tf
    import keras
    from keras.models import Sequential
    from tensorflow.keras.callbacks import Callback
    from keras.layers import Dense, Activation, Dropout
    from tensorflow.keras.utils import to_categorical
    from keras.regularizers import l2
    from keras.models import model_from_json
    # from tf.keras.layers.normalization import BatchNormalization
except:
    print("tensorflow not loaded; Training and prediction will not work")
    tensorflow_keras = False
    Callback = "None"
    
try:
    from lauetools.wyckpos_lauetools import wp, eqhkl_default, eqhkl_custom, sgrp_sym, sgrp_name,\
                            sgrp_params, testhklcond_generalrules_array
except:
    from lauetoolsnn.lauetools.wyckpos_lauetools import wp, eqhkl_default, eqhkl_custom, sgrp_sym, sgrp_name,\
                            sgrp_params, testhklcond_generalrules_array
## for faster binning of histogram
## C version of hist
# from fast_histogram import histogram1d
import h5py

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

def resource_path(relative_path, verbose=0):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = os.path.dirname(__file__)
    if verbose:
        print("Base path of the library: ",base_path)
    return os.path.join(base_path, relative_path)

try:
    metricsNN = [
                keras.metrics.FalseNegatives(name="fn"),
                keras.metrics.FalsePositives(name="fp"),
                keras.metrics.TrueNegatives(name="tn"),
                keras.metrics.TruePositives(name="tp"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="accuracy"),
                ]
except:
    metricsNN = None

ACCEPTABLE_FORMATS = [".npz"]
gui_state = np.random.randint(1e6)
DIGITS = int(abs(np.log10(1e-08)))
CST_ENERGYKEV = 12.398
ACCEPTABLE_FORMATS = [".npz"]

hklcond_group = re.compile(r'([-hkil0-9\(\)]+): ([-+hklnor1-8=\s,]+)(?:, |$)')
DEG = np.pi / 180.0
dist_threshold = 50
# residues_threshold=0.5
# nb_spots_global_threshold=8
# option_global = "v2"
# use_om_user = True
# nb_spots_consider = 100
##v1 same as strains
##v2 ambigious spots all
##v3 ambigious spots with uniqueness

# if you wish to plot the training and testing dataset images
plot_images = False
try:
    ##make sure you have adjustText library
    from adjustText import adjust_text
except:
    plot_images = False
    

def call_global():
    global residues_threshold, nb_spots_global_threshold, option_global, \
            use_om_user, nb_spots_consider, path_user_OM, intensity_threshold, \
            FitPixelDev_global123, boxsize, softmax_threshold_global123, cap_matchrate123,\
            strain_free_parameters, additional_expression
    ## read a test config file and update the variables.
    config_setting = configparser.ConfigParser()
    filepath = resource_path('settings.ini')
    config_setting.read(filepath)
    residues_threshold = float(config_setting.get('CALLER', 'residues_threshold'))
    nb_spots_global_threshold = int(float(config_setting.get('CALLER', 'nb_spots_global_threshold')))
    option_global = config_setting.get('CALLER', 'option_global')
    use_om_user = config_setting.get('CALLER', 'use_om_user') == "true"
    nb_spots_consider = int(float(config_setting.get('CALLER', 'nb_spots_consider')))
    path_user_OM = config_setting.get('CALLER', 'path_user_OM')
    intensity_threshold = int(float(config_setting.get('CALLER', 'intensity')))
    boxsize = int(float(config_setting.get('CALLER', 'boxsize')))
    FitPixelDev_global123 = int(float(config_setting.get('CALLER', 'pixdev')))
    softmax_threshold_global123 = float(config_setting.get('CALLER', 'cap_softmax'))
    cap_matchrate123 = float(config_setting.get('CALLER', 'cap_mr'))
    strain_free_parameters = config_setting.get('CALLER', 'strain_free_parameters').split(",")
    additional_expression = config_setting.get('CALLER', 'additional_expression').split(",")
    if cap_matchrate123 < 1:
        cap_matchrate123 = cap_matchrate123 *100.0
    
def rmv_freq_class(freq_rmv = 0, elements="all", freq_rmv1 = 0, elements1="all",
                   save_directory="", material_=None, material1_=None, write_to_console=None,
                   progress=None, qapp=None):
    classhkl0 = np.load(save_directory+"//grain_classhkl_angbin.npz")["arr_0"]
    if write_to_console != None:
        write_to_console("First material index length: " + str(len(classhkl0)))
    ind_mat = np.array([ij for ij in range(len(classhkl0))])
    
    if material_ != material1_:
        classhkl1 = np.load(save_directory+"//grain_classhkl_angbin1.npz")["arr_0"]
        if write_to_console != None:
            write_to_console("Second material index length: " + str(len(classhkl1)))
        pre_ind = ind_mat[-1] + 1
        ind_mat1 = np.array([pre_ind+ij for ij in range(len(classhkl1))])
        classhkl = np.vstack((classhkl0, classhkl1))
    else:
        classhkl = classhkl0
        # ind_mat = None
        ind_mat1 = None     
        elements1 = "all"
        freq_rmv1 = 0
    
    angbins = np.load(save_directory+"//grain_classhkl_angbin.npz")["arr_1"]
    loc = np.array([ij for ij in range(len(classhkl))])
    trainy_ = array_generatorV2(save_directory+"//training_data", 0, progress, qapp)
    
    if material_ != material1_:
        ## split trainy_ for two materials index
        trainy_mat0 = []
        trainy_mat1 = []
        for ijnode in trainy_:
            if ijnode in ind_mat:
                trainy_mat0.append(ijnode)
            elif ijnode in ind_mat1:
                trainy_mat1.append(ijnode)
        trainy_mat0 = np.array(trainy_mat0)
        trainy_mat1 = np.array(trainy_mat1)
    else:
        trainy_mat0 = trainy_
        trainy_mat1 = None
    
    if write_to_console != None:            
        write_to_console("Class ID and frequency; check for data imbalance and select \
                            appropriate LOSS function for training the model")
    
    ## lets extract the least common occuring classes to simply the training dataset
    if elements == "all":
        most_common0 = collections.Counter(trainy_mat0).most_common()
    else:
        most_common0 = collections.Counter(trainy_mat0).most_common()[:elements]
        
    if material_ != material1_:
        if elements1 =="all":
            most_common1 = collections.Counter(trainy_mat1).most_common()
        else:
            most_common1 = collections.Counter(trainy_mat1).most_common()[:elements1]
    else:
        most_common1 = []
            
    most_common = most_common0 + most_common1       
    print(most_common)

    class_present = [most_common[i][0] for i in range(len(most_common))]
    rmv_indices = []
    count = 0
    for i in loc:
        if i not in class_present:
            rmv_indices.append(i)
        elif i in class_present:
            ind_ = np.where(np.array(class_present)==i)[0]
            ij = most_common[ind_[0]]

            if material_ != material1_:
                if (ij[0] in ind_mat) and (ij[1] <= freq_rmv):
                    rmv_indices.append(int(ij[0]))
                if (ij[0] in ind_mat1) and (ij[1] <= freq_rmv1):
                    rmv_indices.append(int(ij[0]))
            else:
                if (ij[1] <= freq_rmv):
                    rmv_indices.append(int(ij[0]))
        else:
            if write_to_console != None:
                write_to_console("Something Fishy in Remove Freq Class module")
    
    if material_ != material1_:
        for i in rmv_indices:
            if i in ind_mat:
                indd = np.where(ind_mat == i)[0]
                ind_mat = np.delete(ind_mat, indd, axis=0)
            elif i in ind_mat1:
                indd = np.where(ind_mat1 == i)[0]
                ind_mat1 = np.delete(ind_mat1, indd, axis=0)
    else:
        for i in rmv_indices:
            if i in ind_mat:
                indd = np.where(ind_mat == i)[0]
                ind_mat = np.delete(ind_mat, indd, axis=0)
                
    loc_new = np.delete(loc, rmv_indices)

    occurances = [most_common[i][1] for i in range(len(most_common)) if int(most_common[i][0]) in loc_new]
    occurances = np.array(occurances)
    
    class_weight = {}
    class_weight_temp = {}
    count = 0
    for i in loc_new:
        for ij in most_common:
            if int(ij[0]) == i:
                class_weight[count] = int(np.max(occurances)/ij[1]) ##+99 a quick hack to influence the weights
                class_weight_temp[int(ij[0])] = int(np.max(occurances)/ij[1])
                count += 1
    
    for occ in range(len(most_common)):
        if int(most_common[occ][0]) in loc_new:
            if write_to_console != None:
                if int(most_common[occ][0]) == -100:
                    write_to_console("Unclassified HKL (-100); occurance : "+str(most_common[occ][1])+\
                                        ": NN_weights : 0.0")
                else:
                    write_to_console("HKL : " +str(classhkl[int(most_common[occ][0])])+"; occurance : "+\
                                        str(most_common[occ][1])+\
                                          ": NN_weights : "+ str(class_weight_temp[int(most_common[occ][0])]))
    if write_to_console != None:
        write_to_console(str(len(rmv_indices))+ " classes removed from the classHKL object [removal frequency: "+\
                            str(freq_rmv)+"] (before:"+str(len(classhkl))+", now:"+str(len(classhkl)-len(rmv_indices))+")")
    print(str(len(rmv_indices))+ " classes removed from the classHKL object [removal frequency: "+\
                        str(freq_rmv)+"] (before:"+str(len(classhkl))+", now:"+str(len(classhkl)-len(rmv_indices))+")")
            
    classhkl = np.delete(classhkl, rmv_indices, axis=0)
    ## save the altered classHKL object
    if material_ != material1_:
        np.savez_compressed(save_directory+'//MOD_grain_classhkl_angbin.npz', classhkl, angbins, loc_new, 
                            rmv_indices, freq_rmv, len(ind_mat), len(ind_mat1))
    else:
        np.savez_compressed(save_directory+'//MOD_grain_classhkl_angbin.npz', classhkl, angbins, loc_new, 
                            rmv_indices, freq_rmv)
    with open(save_directory + "//class_weights.pickle", "wb") as output_file:
        cPickle.dump([class_weight], output_file)
    if write_to_console != None:
        write_to_console("Saved class weights data")

def array_generator(path_, batch_size, n_classes, loc_new, write_to_console=None, tocategorical=True):
    """
    Assign a new class to data that is removed (to include in the training anyway)
    """
    array_pairs = get_path(path_, ver=0)
    random.shuffle(array_pairs)
    zipped = itertools.cycle(array_pairs)
    while True:
        temp_var = False
        for bs in range(batch_size):
            array_path = next(zipped)
            obj = np.load(array_path)
            trainX = obj["arr_0"]
            loc1 = obj["arr_1"]
            
            if len(trainX) == 0 or len(loc1) == 0:
                if write_to_console != None:
                    write_to_console("Skipping File: "+ array_path+"; No data is found")
                if bs == 0:
                    temp_var = True
                continue                
            ## remove the non frequent class and rearrange the data
            loc1_new = []
            loc1_new_rmv = []
            for k, i in enumerate(loc1):
                temp_loc = np.where(loc_new==i)[0]
                if len(temp_loc) == 1:
                    loc1_new.append(temp_loc)
                else:
                    loc1_new_rmv.append(k)   
               
            loc1_new = np.array(loc1_new).ravel()
            loc1_new_rmv = np.array(loc1_new_rmv).ravel() 
            
            if len(trainX) != len(loc1_new):
                if len(loc1_new_rmv) > 0:
                    trainX = np.delete(trainX, loc1_new_rmv, axis=0) 

            if bs == 0 or temp_var:
                trainX1 = np.copy(trainX)
                trainY1 = np.copy(loc1_new)
            else:
                trainX1 = np.vstack((trainX1, trainX))
                trainY1 = np.hstack((trainY1, loc1_new))

        ## To normalize the size of one hot encoding
        count = 0
        if np.min(trainY1) != 0:
            trainY1 = np.append(trainY1, 0)
            count += 1
        if np.max(trainY1) != (n_classes-1):
            trainY1 = np.append(trainY1, n_classes-1)
            count += 1
        
        if tocategorical:
            trainY1 = to_categorical(trainY1)
        if count == 1:
            trainY1 = np.delete(trainY1, [len(trainY1)-1] ,axis=0)
        elif count == 2:
            trainY1 = np.delete(trainY1, [len(trainY1)-1,len(trainY1)-2] ,axis=0)

        yield trainX1, trainY1
        
def vali_array(path_, batch_size, n_classes, loc_new, write_to_console=None, tocategorical=True):
    array_pairs = get_path(path_, ver=0)
    random.shuffle(array_pairs)
    zipped = itertools.cycle(array_pairs)
    temp_var = False
    for bs in range(batch_size):
        array_path = next(zipped)
        obj = np.load(array_path)
        trainX = obj["arr_0"]
        loc1 = obj["arr_1"]
        
        if len(trainX) == 0 or len(loc1) == 0:
            if write_to_console != None:
                write_to_console("Skipping File: "+ array_path+"; No data is found")
            if bs == 0:
                temp_var = True
            continue
        
        ## remove the non frequent class and rearrange the data
        loc1_new = []
        loc1_new_rmv = []
        for k, i in enumerate(loc1):
            temp_loc = np.where(loc_new==i)[0]
            if len(temp_loc) == 1:
                loc1_new.append(temp_loc)
            else:
                loc1_new_rmv.append(k)
        
        loc1_new = np.array(loc1_new).ravel()
        loc1_new_rmv = np.array(loc1_new_rmv).ravel()
        
        if len(trainX) != len(loc1_new):
            if len(loc1_new_rmv) > 0:
                trainX = np.delete(trainX, loc1_new_rmv, axis=0)
            
        if bs == 0 or temp_var:
            trainX1 = trainX
            trainY1 = loc1_new
        else:
            trainX1 = np.vstack((trainX1, trainX))
            trainY1 = np.hstack((trainY1, loc1_new))
    
    count = 0
    if np.min(trainY1) != 0:
        trainY1 = np.append(trainY1, 0)
        count += 1
    if np.max(trainY1) != (n_classes-1):
        trainY1 = np.append(trainY1, n_classes-1)
        count += 1
    
    if tocategorical:
        trainY1 = to_categorical(trainY1)
        
    if count == 1:
        trainY1 = np.delete(trainY1, [len(trainY1)-1] ,axis=0)
    elif count == 2:
        trainY1 = np.delete(trainY1, [len(trainY1)-1,len(trainY1)-2] ,axis=0)

    return trainX1, trainY1

def get_path(path_, ver=0):
    image_files = []
    for dir_entry in os.listdir(path_):
        if os.path.isfile(os.path.join(path_, dir_entry)) and \
                os.path.splitext(dir_entry)[1] in ACCEPTABLE_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            image_files.append((file_name, file_extension,
                                os.path.join(path_, dir_entry)))
    return_value = []
    for image_file, _, image_full_path in image_files:
        if image_file == "grain_classhkl_angbin":
            continue
        if image_file == "grain_classhkl_angbin1":
            continue
        if ver == 1 and image_file == "grain_init":
            continue
        if ver == 1 and image_file == "grain_init1":
            continue
        return_value.append((image_full_path))
    return return_value

def array_generator_verify(path_, batch_size, n_classes, loc_new, write_to_console=None):
    array_pairs = get_path(path_, ver=1)
    random.shuffle(array_pairs)
    zipped = itertools.cycle(array_pairs)
    while True:
        temp_var = False
        for bs in range(batch_size):
            array_path = next(zipped)
            obj = np.load(array_path)
            loc1 = obj["arr_1"]            
            if len(loc1) == 0:
                if write_to_console !=None:
                    write_to_console("Skipping File: "+ array_path+"; No data is found")
                if bs == 0:
                    temp_var = True
                continue             
            ## remove the non frequent class and rearrange the data
            loc1_new = []
            for k, i in enumerate(loc1):
                temp_loc = np.where(loc_new==i)[0]
                if len(temp_loc) == 1:
                    loc1_new.append(temp_loc)     
            loc1_new = np.array(loc1_new).ravel()
            if bs == 0 or temp_var:
                trainY1 = np.copy(loc1_new)
            else:
                trainY1 = np.hstack((trainY1, loc1_new)) 
        return trainY1

def create_additional_data(path_, write_to_console=None, material=None, material1=None):
    """array_generator_verify(self.save_directory+"//training_data", batch_size, 
          len(self.classhkl), self.loc_new, self.write_to_console)
            if generate_additional_data==True"""
    array_pairs = get_path(path_, ver=1)
    
    for ijk in array_pairs:
        
        if ijk.split("\\")[-1].startswith(material+"_grain_"):
            obj = np.load(ijk)
            
            loc1 = obj["arr_1"]
            for kji in array_pairs:
                
                pass #TODO

    # random.shuffle(array_pairs)
    # zipped = itertools.cycle(array_pairs)
    # while True:
    #     temp_var = False
    #     for bs in range(batch_size):
    #         array_path = next(zipped)
    #         obj = np.load(array_path)
    #         loc1 = obj["arr_1"]            
    #         if len(loc1) == 0:
    #             if write_to_console !=None:
    #                 write_to_console("Skipping File: "+ array_path+"; No data is found")
    #             if bs == 0:
    #                 temp_var = True
    #             continue             
    #         ## remove the non frequent class and rearrange the data
    #         loc1_new = []
    #         for k, i in enumerate(loc1):
    #             temp_loc = np.where(loc_new==i)[0]
    #             if len(temp_loc) == 1:
    #                 loc1_new.append(temp_loc)     
    #         loc1_new = np.array(loc1_new).ravel()
    #         if bs == 0 or temp_var:
    #             trainY1 = np.copy(loc1_new)
    #         else:
    #             trainY1 = np.hstack((trainY1, loc1_new)) 
    #     return trainY1
    # open each npz file and combine two grains to form another Laue pattern
    ##save all data then open them and combine into one laue pattern --> better for two phase material
    # s_tth, s_chi, s_miller_ind, _, _, _, \
    #     ori_mat, ori_mat1
    # s_tth = np.array([item for sublist in l_tth for item in sublist])
    # s_chi = np.array([item for sublist in l_chi for item in sublist])
    # s_miller_ind = np.array([item for sublist in l_miller_ind for item in sublist])
    # s_posx = np.array([item for sublist in l_posx for item in sublist])
    # s_posy = np.array([item for sublist in l_posy for item in sublist])
    # s_E = np.array([item for sublist in l_E for item in sublist])
    # s_intensity=np.array([item for sublist in l_intensity for item in sublist])
    
    # if sortintensity:
    #     indsort = np.argsort(s_intensity)[::-1]
    #     s_tth=np.take(s_tth, indsort)
    #     s_chi=np.take(s_chi, indsort)
    #     s_miller_ind=np.take(s_miller_ind, indsort, axis=0)
    #     s_posx=np.take(s_posx, indsort)
    #     s_posy=np.take(s_posy, indsort)
    #     s_E=np.take(s_E, indsort)
    #     s_intensity=np.take(s_intensity, indsort)

def array_generatorV2(path_, ver=1, progress=None, qapp=None):
    array_pairs = get_path(path_, ver=ver)
    random.shuffle(array_pairs)
    if progress !=None:
        progress.setMaximum(len(array_pairs))
    for bs in range(len(array_pairs)):
        loc1 = np.load(array_pairs[bs])["arr_1"]           
        if bs == 0:
            trainY1 = loc1
        if bs > 0:
            trainY1 = np.hstack((trainY1, loc1))
        if progress !=None:
            progress.setValue(bs+1)
        if qapp !=None:
            qapp.processEvents()
    return trainY1

def printProgressBar(iteration, total, prefix = '', suffix = 'Complete', 
                      decimals = 1, length = 50, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        
def mse_images(pathA, pathB, ix, iy, ccd_label, progressbar=False, iteration=None, total=None):
   	# the 'Mean Squared Error' between the two images is the
   	# sum of the squared difference between the two images;
   	# NOTE: the two images must have the same dimension
    imageA, _, _ = IOimage.readCCDimage(pathA, stackimageindex=-1,
                                      CCDLabel=ccd_label,
                                      dirname=None, verbose=0)
    imageB, _, _ = IOimage.readCCDimage(pathB, stackimageindex=-1,
                                      CCDLabel=ccd_label,
                                      dirname=None, verbose=0)
    err = np.sum((imageA.astype("int") - imageB.astype("int")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    if progressbar:
        printProgressBar(iteration, total-1)
    
    return err, ix, iy
    

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

def generate_classHKL(n, rules, lattice_material, symmetry, material_, crystal=None, SG=None, general_diff_cond=False,
         save_directory="", write_to_console=None, progress=None, qapp=None, ang_maxx = None, step = None):  
    temp_ = GT.threeindices_up_to(int(n))
    classhkl_ = temp_
    
    
    if general_diff_cond:
        classhkl_ = crystal.hkl_allowed_array(classhkl_)
        
    if write_to_console !=None:
        write_to_console("Generating HKL objects")
        
    # generate HKL object
    if progress !=None:
        progress.setMaximum(len(classhkl_))
        
    hkl_all = {}
    # another_method = False
    for i in range(len(classhkl_)):
        new_hkl = classhkl_[i,:]

        new_rounded_hkl = _round_indices(new_hkl)
        mul_family = crystal.equivalent_hkls(new_rounded_hkl)
        
        family = []
        for sym in mul_family:
            family.append(sym)
        hkl_all[str(new_rounded_hkl)] = {"hkl":new_rounded_hkl, 
                                 "family": family}
        
        if progress !=None:
            progress.setValue(i+1)
        if qapp !=None:
            qapp.processEvents() 

    ## FAST IMPLEMENTATION
    ## make comprehensive list of dictionary
    equ_hkl = np.zeros((1,3))
    for j in hkl_all.keys():
        equ_hkl = np.vstack((equ_hkl, hkl_all[j]["family"]))
    equ_hkl = np.delete(equ_hkl, 0, axis =0)

    index_hkl = [j for j,k in enumerate(hkl_all.keys()) for i in range(len(hkl_all[k]["family"]))]
    
    if write_to_console !=None:
        write_to_console("Removing harmonics and building equivalent HKL objects")
    if progress !=None:
        progress.setMaximum(len(hkl_all.keys()))
        
    ind_rmv = []
    for j1, i1 in enumerate(hkl_all.keys()):
        hkl_1 = hkl_all[i1]["hkl"]
        temp1_ = np.all(hkl_1 == equ_hkl, axis=1)
        if len(np.where(temp1_)[0]) != 0:
            ind_ = np.where(temp1_)[0]
            for inin in ind_:
                if index_hkl[inin] > j1:
                    ind_rmv.append(i1)
                    break
        if progress !=None:
            progress.setValue(j1+1)
        if qapp !=None:
            qapp.processEvents()

    if len(ind_rmv) != 0:
        for inrmv in ind_rmv:
            _ = hkl_all.pop(inrmv, None)
    
    #Check same class HKL and remove them to avoid conflict
    #ADD the removed class as Multiplicity for the non removed class
    classhkl = np.zeros((len(hkl_all),3))
    keys_rmv = []
    for j1, i1 in enumerate(hkl_all.keys()):
        hkl_object = hkl_all[i1]["hkl"]
        classhkl[j1,:] = hkl_object
        keys_rmv.append(i1)
    if ang_maxx == None:
        ang_maxx= 90
    if step == None:
        step=0.1
        
    codebars, angbins = get_material_data(material_ = material_, ang_maxx = ang_maxx, step = step,
                                               hkl_ref=n, classhkl=classhkl)
    # if write_to_console !=None:
    #     write_to_console("Verifying if two different HKL class have same angular distribution (can be very time consuming depending on the symmetry)")
    list_appended = []
    list_remove = []
    for i, j in enumerate(codebars):
        for k, l in enumerate(codebars):
            # if i in list_appended and k in list_appended:
            #     continue
            if i != k and np.all(j == l):
                # string0 = "HKL's "+ str(classhkl[i])+" and "+str(classhkl[k])+" have exactly the same angular distribution."
                # if write_to_console !=None:
                #     write_to_console(string0)
                if keys_rmv[i] in list_remove or keys_rmv[k] in list_remove:
                    if write_to_console !=None:
                        continue
                        # write_to_console("list already added")                        
                else:
                    list_remove.append(keys_rmv[i])
                    ind_rmv.append(keys_rmv[i])
                    for ijk in hkl_all[keys_rmv[i]]['family']:
                        hkl_all[keys_rmv[k]]['family'].append(ijk)
            list_appended.append(i)
            list_appended.append(k)
    if len(list_remove) != 0:
        for inrmv in list_remove:
            _ = hkl_all.pop(inrmv, None)
            
    if write_to_console !=None:
        write_to_console("Finalizing the HKL objects")
    hkl_all_class = hkl_all
    hkl_millerindices = {}
    classhkl = np.zeros((len(hkl_all),3))
    for j1, i1 in enumerate(hkl_all.keys()):
        hkl_object = hkl_all[i1]["hkl"]
        classhkl[j1,:] = hkl_object
        family = hkl_all_class[i1]["family"]
        hkl_millerindices[i1] =  np.array([ii for ii in family])

    tempdict = hkl_millerindices

    with open(save_directory + "//classhkl_data_"+material_+".pickle", "wb") as output_file:
        cPickle.dump([classhkl, classhkl_, ind_rmv, n, temp_, \
                      hkl_all_class, hkl_all, lattice_material, symmetry], output_file)
    
    with open(save_directory + "//classhkl_data_nonpickled_"+material_+".pickle", "wb") as output_file:
        cPickle.dump([tempdict], output_file)       
    if write_to_console !=None:
        write_to_console("Saved class HKL data in : "+save_directory + "//classhkl_data_"+material_+".pickle")
    
def write_training_testing_dataMTEX(save_directory,material_, material1_, lattice_material, lattice_material1,
                                    material0_lauegroup, material1_lauegroup):
    for imh in ["training_data", "testing_data"]:
        image_files = []
        path_ = save_directory+"//"+imh
        for dir_entry in os.listdir(path_):
            if os.path.isfile(os.path.join(path_, dir_entry)) and \
                    os.path.splitext(dir_entry)[1] in ACCEPTABLE_FORMATS:
                file_name, file_extension = os.path.splitext(dir_entry)
                image_files.append((file_name, file_extension,
                                    os.path.join(path_, dir_entry)))
        return_value = []
        for image_file, _, image_full_path in image_files:
            if image_file == "grain_classhkl_angbin" or image_file == "grain_classhkl_angbin1" or\
                image_file == "grain_init" or image_file == "grain_init1":
                continue
            return_value.append((image_full_path))

        ori_array1 = np.zeros((1,3,3))
        if material_ != material1_:
            ori_array2 = np.zeros((1,3,3))
        for bs in return_value:
            obj = np.load(bs)
            ori1 = obj["arr_2"]
            ori2 = obj["arr_3"] 
            flag = obj["arr_4"] 
            ## flag 0 is random data
            ## flag 1, 2, 3 are small angle miori data
            if flag == 0:
                if len(ori1) != 0:
                    ori_array1 = np.vstack((ori_array1,ori1))
                if material_ != material1_:
                    if len(ori2) != 0:
                        ori_array2 = np.vstack((ori_array2,ori2))
                    
        ori_array1 = np.delete(ori_array1, 0, axis=0)
        phase_ori1 = np.ones(len(ori_array1))
        
        ori_array = ori_array1
        phase_ori = phase_ori1
        if material_ != material1_:
            ori_array2 = np.delete(ori_array2, 0, axis=0)         
            phase_ori2 = np.ones(len(ori_array2))*2
            ori_array = np.vstack((ori_array, ori_array2))
            phase_ori = np.hstack((phase_ori, phase_ori2))
        
        if material_ == material1_:
            lattice = lattice_material
            material0_LG = material0_lauegroup
            header = [
                    "Channel Text File",
                    "Prj     lauetoolsnn",
                    "Author    [Ravi raj purohit]",
                    "JobMode    Grid",
                    "XCells    "+str(len(ori_array)),
                    "YCells    "+str(1),
                    "XStep    1.0",
                    "YStep    1.0",
                    "AcqE1    0",
                    "AcqE2    0",
                    "AcqE3    0",
                    "Euler angles refer to Sample Coordinate system (CS0)!    Mag    100    Coverage    100    Device    0    KV    15    TiltAngle    40    TiltAxis    0",
                    "Phases    1",
                    str(round(lattice._lengths[0]*10,5))+";"+str(round(lattice._lengths[1]*10,5))+";"+\
                    str(round(lattice._lengths[2]*10,5))+"\t"+str(round(lattice._angles[0],5))+";"+\
                        str(round(lattice._angles[1],5))+";"+str(round(lattice._angles[2],5))+"\t"+"Material1"+ "\t"+material0_LG+ "\t"+"????"+"\t"+"????",
                    "Phase    X    Y    Bands    Error    Euler1    Euler2    Euler3    MAD    BC    BS"]
        else:
            lattice = lattice_material
            lattice1 = lattice_material1
            material0_LG = material0_lauegroup
            material1_LG = material1_lauegroup
            header = [
                    "Channel Text File",
                    "Prj     lauetoolsnn",
                    "Author    [Ravi raj purohit]",
                    "JobMode    Grid",
                    "XCells    "+str(len(ori_array)),
                    "YCells    "+str(1),
                    "XStep    1.0",
                    "YStep    1.0",
                    "AcqE1    0",
                    "AcqE2    0",
                    "AcqE3    0",
                    "Euler angles refer to Sample Coordinate system (CS0)!    Mag    100    Coverage    100    Device    0    KV    15    TiltAngle    40    TiltAxis    0",
                    "Phases    2",
                    str(round(lattice._lengths[0]*10,5))+";"+str(round(lattice._lengths[1]*10,5))+";"+\
                    str(round(lattice._lengths[2]*10,5))+"\t"+str(round(lattice._angles[0],5))+";"+\
                        str(round(lattice._angles[1],5))+";"+str(round(lattice._angles[2],5))+"\t"+"Material1"+ "\t"+material0_LG+ "\t"+"????"+"\t"+"????",
                    str(round(lattice1._lengths[0]*10,5))+";"+str(round(lattice1._lengths[1]*10,5))+";"+\
                    str(round(lattice1._lengths[2]*10,5))+"\t"+str(round(lattice1._angles[0],5))+";"+\
                        str(round(lattice1._angles[1],5))+";"+str(round(lattice1._angles[2],5))+"\t"+"Material2"+ "\t"+material1_LG+ "\t"+"????"+"\t"+"????",
                    "Phase    X    Y    Bands    Error    Euler1    Euler2    Euler3    MAD    BC    BS"]
        # =================CALCULATION OF POSITION=====================================
        euler_angles = np.zeros((len(ori_array),3))
        phase_euler_angles = np.zeros(len(ori_array))
        for i in range(len(ori_array)):                
            # euler_angles[i,:] = rot_mat_to_euler(ori_array[i,:,:])
            euler_angles[i,:] = OrientationMatrix2Euler(ori_array[i,:,:])
            phase_euler_angles[i] = phase_ori[i]        

        a = euler_angles
        if material_ != material1_:
            filename125 = save_directory+ "//"+material_+"_"+material1_+"_MTEX_UBmat_"+imh+".ctf"
        else:
            filename125 = save_directory+ "//"+material_+"_MTEX_UBmat_"+imh+".ctf"
            
        f = open(filename125, "w")
        for ij in range(len(header)):
            f.write(header[ij]+" \n")
                
        for j123 in range(euler_angles.shape[0]):
            y_step = 1
            x_step = 1 * j123
            phase_id = int(phase_euler_angles[j123])
            eul =  str(phase_id)+'\t' + "%0.4f" % x_step +'\t'+"%0.4f" % y_step+'\t8\t0\t'+ \
                                "%0.4f" % a[j123,0]+'\t'+"%0.4f" % a[j123,1]+ \
                                    '\t'+"%0.4f" % a[j123,2]+'\t0.0001\t180\t0\n'
            string = eul
            f.write(string)
        f.close()
        
def get_material_data(material_="Cu", ang_maxx = 45, step = 0.5, hkl_ref=13, classhkl = None):
    a, b, c, alpha, beta, gamma = dictLT.dict_Materials[material_][1]
    Gstar = CP.Gstar_from_directlatticeparams(a, b, c, alpha, beta, gamma)
    rules = dictLT.dict_Materials[material_][-1]
    
    hkl2 = GT.threeindices_up_to(int(hkl_ref))
    hkl2 = CP.ApplyExtinctionrules(hkl2,rules)
    hkl2 = hkl2.astype(np.int16)

    query_angle = ang_maxx/2.
    angle_tol = ang_maxx/2.
    metrics = Gstar

    hkl1 = classhkl
    H1 = hkl1
    n1 = hkl1.shape[0]
    H2 = hkl2
    n2 = hkl2.shape[0]
    dstar_square_1 = np.diag(np.inner(np.inner(H1, metrics), H1))
    dstar_square_2 = np.diag(np.inner(np.inner(H2, metrics), H2))
    scalar_product = np.inner(np.inner(H1, metrics), H2) * 1.0
    d1 = np.sqrt(dstar_square_1.reshape((n1, 1))) * 1.0
    d2 = np.sqrt(dstar_square_2.reshape((n2, 1))) * 1.0
    outy = np.outer(d1, d2)
    
    ratio = scalar_product / outy
    ratio = np.round(ratio, decimals=7)
    tab_angulardist = np.arccos(ratio) / (np.pi / 180.0)
    np.putmask(tab_angulardist, np.abs(tab_angulardist) < 0.001, 400)
    
    # self.write_to_console("Calculating Mutual angular distances")
    # self.progress.setMaximum(len(tab_angulardist))
    closest_angles_values = []
    for ang_ in range(len(tab_angulardist)):
        tab_angulardist_ = tab_angulardist[ang_,:]
        angles_set = np.ravel(tab_angulardist_)  # 1D array
        sorted_ind = np.argsort(angles_set)
        sorted_angles = angles_set[sorted_ind]
        
        angle_query = angle_tol
        if isinstance(query_angle, (list, np.ndarray, tuple)):
            angle_query = query_angle[0]
        
        array_angledist = np.abs(sorted_angles - angle_query)
        pos_min = np.argmin(array_angledist)
        closest_angle = sorted_angles[pos_min]
        
        if np.abs(closest_angle - query_angle) > angle_tol:
            if angle_query > 0.5:
                pass
            print("TODO function get_material_data")
            
        condition = array_angledist <= angle_tol
        closest_index_in_sorted_angles_raw = np.where(condition)[0]
        closest_angles_values.append(np.take(sorted_angles, closest_index_in_sorted_angles_raw))
        # self.progress.setValue(ang_+1)
        # QApplication.processEvents() 
    
    # self.write_to_console("Constructing histograms")
    # self.progress.setMaximum(len(closest_angles_values))
    codebars = []
    angbins = np.arange(0, ang_maxx+step, step)
    for i in range(len(closest_angles_values)):
        angles = closest_angles_values[i]
        fingerprint = np.histogram(angles, bins=angbins)[0]
        # fingerprint = histogram1d(angles, range=[min(angbins),max(angbins)], bins=len(angbins)-1)
        ## Normalize the histogram by its maximum: simple way 
        ## Maybe better normalization is possible.. to be seen
        max_codebars = np.max(fingerprint)
        fingerprint = fingerprint/ max_codebars
        codebars.append(fingerprint)
        # self.progress.setValue(i+1)
        # QApplication.processEvents() 
    # self.progress.setValue(0)
    return codebars, angbins

def Euler2OrientationMatrix(euler):
    """Compute the orientation matrix :math:`\mathbf{g}` associated with
    the 3 Euler angles :math:`(\phi_1, \Phi, \phi_2)`.
    :param euler: The triplet of the Euler angles (in degrees).
    :return g: The 3x3 orientation matrix.
    """
    (rphi1, rPhi, rphi2) = np.radians(euler)
    c1 = np.cos(rphi1)
    s1 = np.sin(rphi1)
    c = np.cos(rPhi)
    s = np.sin(rPhi)
    c2 = np.cos(rphi2)
    s2 = np.sin(rphi2)
    # rotation matrix g
    g11 = c1 * c2 - s1 * s2 * c
    g12 = s1 * c2 + c1 * s2 * c
    g13 = s2 * s
    g21 = -c1 * s2 - s1 * c2 * c
    g22 = -s1 * s2 + c1 * c2 * c
    g23 = c2 * s
    g31 = s1 * s
    g32 = -c1 * s
    g33 = c
    g = np.array([[g11, g12, g13], [g21, g22, g23], [g31, g32, g33]])
    return g

def getpatterns_(nb, nb1, material_=None, material1_=None, emin=5, emax=23, detectorparameters=None, pixelsize=None, 
                 sortintensity = False, ang_maxx = 45, step = 0.5, classhkl = None, classhkl1 = None, noisy_data=False, 
                 remove_peaks=False, seed = None,hkl_all=None, lattice_material=None, family_hkl=None,
                 normal_hkl=None, index_hkl=None, hkl_all1=None, lattice_material1=None, family_hkl1=None,
                 normal_hkl1=None, index_hkl1=None, dim1=2048, dim2=2048, removeharmonics=1, flag = 0,
                 img_i=None, img_j=None, save_directory_=None, odf_data=None, odf_data1=None, modelp=None,
                 misorientation_angle=None, max_millerindex=0, max_millerindex1=0, general_diff_cond=False, crystal=None, crystal1=None,
                 phase_always_present=None):
    
    s_tth, s_chi, s_miller_ind, _, _, _, \
        ori_mat, ori_mat1 = simulatemultiplepatterns(nb, nb1, seed=seed, key_material=material_, 
                                                                        key_material1=material1_,
                                                                        emin=emin, emax=emax,
                                                                         detectorparameters=detectorparameters,
                                                                         pixelsize=pixelsize,
                                                                         sortintensity = sortintensity, 
                                                                         dim1=dim1, dim2=dim2, 
                                                                         removeharmonics=removeharmonics,
                                                                         flag=flag, odf_data=odf_data,
                                                                         odf_data1=odf_data1, mode=modelp,
                                                                         misorientation_angle=misorientation_angle,
                                                                         phase_always_present=phase_always_present)
    if noisy_data:
        ## apply random gaussian type noise to the data (tth and chi)
        ## So adding noise to the angular distances
        ## Instead of adding noise to all HKL's ... Add to few selected HKLs
        ## Adding noise to randomly 30% of the HKLs
        ## Realistic way of introducting strains is through Pixels and not 2theta
        noisy_pixel = 0.15
        indices_noise = np.random.choice(len(s_tth), int(len(s_tth)*0.3), replace=False)
        noise_ = np.random.normal(0,noisy_pixel,len(indices_noise))
        s_tth[indices_noise] = s_tth[indices_noise] + noise_
        noise_ = np.random.normal(0,noisy_pixel,len(indices_noise)) 
        s_chi[indices_noise] = s_chi[indices_noise] + noise_
        
    if remove_peaks:
        len_mi = np.array([iq for iq in range(len(s_miller_ind))])
        len_mi = len_mi[int(0.6*len(s_miller_ind)):]
        indices_remove = np.random.choice(len_mi, int(len(len_mi)*0.3), replace=False)
        ## delete randomly selected less intense peaks
        ## to simulate real peak detection, where some peaks may not be
        ## well detected
        ## Include maybe Intensity approach: Delete peaks based on their SF and position in detector
        if len(indices_remove) !=0:
            s_tth = np.delete(s_tth, indices_remove)
            s_chi = np.delete(s_chi, indices_remove)
            s_miller_ind = np.delete(s_miller_ind, indices_remove, axis=0)
        else:
            print(nb, nb1, material_, material1_, odf_data, odf_data1)

    # replace all hkl class with relevant hkls
    ## skip HKLS that dont follow the general diffraction rules
    location = []
    skip_hkl = []
    delete_spots = []
    for j, i in enumerate(s_miller_ind):

        new_hkl = _round_indices(i[:3])

        if i[3] == 0: ##material 1
            
            if general_diff_cond:
                cond_proceed = crystal.hkl_allowed(i[:3], returnequivalents=False)
            else:
                cond_proceed = True
            
            if not cond_proceed:
                delete_spots.append(j)
                continue
            
            if np.any(np.abs(new_hkl)>max_millerindex):
                skip_hkl.append(j)
                continue
            
            temp_ = np.all(new_hkl == normal_hkl, axis=1)
            if len(np.where(temp_)[0]) == 1:
                ind_ = np.where(temp_)[0][0]
                location.append(index_hkl[ind_])
            elif len(np.where(temp_)[0]) == 0:
                # print("Entering -100 for "+ str(i) + "\n")
                skip_hkl.append(j)
            elif len(np.where(temp_)[0]) > 1:
                ## first check if they both are same class or not
                class_output = []
                for ij in range(len(np.where(temp_)[0])):
                    indc = index_hkl[np.where(temp_)[0][ij]]
                    class_output.append(indc)
                if len(set(class_output)) <= 1:
                    location.append(class_output[0])
                else:
                    skip_hkl.append(j)
                    print(i)
                    print(np.where(temp_)[0])
                    for ij in range(len(np.where(temp_)[0])):
                        indc = index_hkl[np.where(temp_)[0][ij]]
                        print(classhkl[indc])
                    print("Entering -500: Skipping HKL as something is not proper with equivalent HKL module")
            
        elif i[3] == 1: ##material 2
            
            if general_diff_cond:
                cond_proceed1 = crystal1.hkl_allowed(i[:3], returnequivalents=False)
            else:
                cond_proceed1 = True
            
            if not cond_proceed1:
                delete_spots.append(j)
                continue
            
            if np.any(np.abs(new_hkl)>max_millerindex1):
                skip_hkl.append(j)
                continue
            
            temp_ = np.all(new_hkl == normal_hkl1, axis=1)
            if len(np.where(temp_)[0]) == 1:
                ind_ = np.where(temp_)[0][0]
                location.append(index_hkl1[ind_])
            elif len(np.where(temp_)[0]) == 0:
                # print("Entering -100 for "+ str(i) + "\n")
                skip_hkl.append(j)
            elif len(np.where(temp_)[0]) > 1:
                ## first check if they both are same class or not
                class_output = []
                for ij in range(len(np.where(temp_)[0])):
                    indc = index_hkl1[np.where(temp_)[0][ij]]
                    class_output.append(indc)
                if len(set(class_output)) <= 1:
                    location.append(class_output[0])
                else:
                    skip_hkl.append(j)
                    print(i)
                    print(np.where(temp_)[0])
                    for ij in range(len(np.where(temp_)[0])):
                        indc = index_hkl1[np.where(temp_)[0][ij]]
                        print(classhkl[indc])
                    print("Entering -500: Skipping HKL as something is not proper with equivalent HKL module")
    
    allspots_the_chi = np.transpose(np.array([s_tth/2., s_chi]))
    tabledistancerandom = np.transpose(GT.calculdist_from_thetachi(allspots_the_chi, allspots_the_chi))
    
    codebars = []
    angbins = np.arange(0,ang_maxx+step,step)
    for i in range(len(tabledistancerandom)):
        if i in skip_hkl or i in delete_spots: ## not saving skipped HKL
            continue
        angles = tabledistancerandom[i]
        spots_delete = [i]
        for del_spts in delete_spots:
            spots_delete.append(del_spts)
        angles = np.delete(angles, spots_delete)
        # angles = np.delete(angles, i)# removing the self distance
        fingerprint = np.histogram(angles, bins=angbins)[0]
        # fingerprint = histogram1d(angles, range=[min(angbins),max(angbins)], bins=len(angbins)-1)
        ## same normalization as before
        max_codebars = np.max(fingerprint)
        fingerprint = fingerprint/ max_codebars
        codebars.append(fingerprint)
    
    if phase_always_present != None:
        suffix_ = "_development"
    else:
        suffix_ = ""
    ###########################################################
    if flag in [0,1,2,3] and plot_images:
        fig = plt.figure()
        plt.scatter(s_tth, s_chi, c='k')
        plt.ylabel(r'$\chi$ (in deg)',fontsize=8)
        plt.xlabel(r'2$\theta$ (in deg)', fontsize=10)
        plt.grid(linestyle='--', linewidth=0.5)
        texts1=[]
        for i, txt_hkl in enumerate(s_miller_ind):
            txt = _round_indices(txt_hkl[:3])
            # print("Actual hkl: "+str(txt_hkl[:3])+" ;  Rounded hkl: "+str(txt[:3]))
            if txt_hkl[3] == 0:
                if np.any(np.abs(txt) > max_millerindex):
                    continue
            elif txt_hkl[3] == 1:
                if np.any(np.abs(txt) > max_millerindex1):
                    continue
            txt = txt_hkl 
            texts1.append(plt.text(s_tth[i], s_chi[i], str(int(txt[0]))+" "+str(int(txt[1]))+" "+str(int(txt[2])), size=8))
        adjust_text(texts1, only_move={'points':'y', 'text':'y'})
    ###########################################################
    if flag == 0:
        if plot_images:
            plt.savefig(save_directory_+'//grain_'+str(img_i)+"_"+\
                                str(img_j)+suffix_+'.png', bbox_inches='tight',format='png', dpi=1000) 
            plt.close(fig)
        if len(codebars) != 0:
            if nb == 0:
                np.savez_compressed(save_directory_+'//'+material1_+'_grain_'+str(img_i)+"_"+\
                                    str(img_j)+suffix_+'.npz', codebars, location, ori_mat, ori_mat1, flag,\
                                    s_tth, s_chi, s_miller_ind)
            elif nb1 == 0:
                np.savez_compressed(save_directory_+'//'+material_+'_grain_'+str(img_i)+"_"+\
                                    str(img_j)+suffix_+'.npz', codebars, location, ori_mat, ori_mat1, flag,\
                                    s_tth, s_chi, s_miller_ind)
            else:
                np.savez_compressed(save_directory_+'//'+material_+"_"+material1_+'_grain_'+str(img_i)+"_"+\
                                    str(img_j)+suffix_+'.npz', codebars, location, ori_mat, ori_mat1, flag,\
                                    s_tth, s_chi, s_miller_ind)
        else:
            print("Skipping a simulation file: "+save_directory_+'//grain_'+\
                                str(img_i)+"_"+str(img_j)+suffix_+'.npz'+"; Due to no data conforming user settings")
    elif flag == 1:
        if plot_images:
            plt.savefig(save_directory_+'//grain_'+str(img_j)+suffix_+'_smo.png', bbox_inches='tight',format='png', dpi=1000) 
            plt.close(fig)
        if len(codebars) != 0:
            np.savez_compressed(save_directory_+'//grain_'+str(img_j)+suffix_+'_smo.npz', \
                                codebars, location, ori_mat, ori_mat1, flag,\
                                s_tth, s_chi, s_miller_ind)
        else:
            print("Skipping a simulation file: "+save_directory_+'//grain_'+\
                                str(img_j)+'_smo.npz'+"; Due to no data conforming user settings")
    elif flag == 2:
        if plot_images:
            plt.savefig(save_directory_+'//grain_'+str(img_j)+suffix_+'_smo1.png', bbox_inches='tight',format='png', dpi=1000) 
            plt.close(fig)
        if len(codebars) != 0:
            np.savez_compressed(save_directory_+'//grain_'+str(img_j)+suffix_+'_smo1.npz', \
                                codebars, location, ori_mat, ori_mat1, flag,\
                                s_tth, s_chi, s_miller_ind)
        else:
            print("Skipping a simulation file: "+save_directory_+'//grain_'+\
                                str(img_j)+'_smo1.npz'+"; Due to no data conforming user settings")
    elif flag == 3:
        if plot_images:
            plt.savefig(save_directory_+'//grain_'+str(img_j)+suffix_+'_smo2.png', bbox_inches='tight',format='png', dpi=1000) 
            plt.close(fig)
        if len(codebars) != 0:
            np.savez_compressed(save_directory_+'//grain_'+str(img_j)+suffix_+'_smo2.npz', \
                                codebars, location, ori_mat, ori_mat1, flag,\
                                s_tth, s_chi, s_miller_ind)
        else:
            print("Skipping a simulation file: "+save_directory_+'//grain_'+\
                                str(img_j)+suffix_+'_smo2.npz'+"; Due to no data conforming user settings")

def simulatemultiplepatterns(nbUBs, nbUBs1, seed=123, key_material=None, key_material1=None, 
                             emin=5, emax=23, detectorparameters=None, pixelsize=None,
                             sortintensity = False, dim1=2048, dim2=2048, removeharmonics=1, flag = 0,
                             odf_data=None, odf_data1=None, mode="random", misorientation_angle = 1,
                             phase_always_present=None):
    
    detectordiameter = pixelsize * dim1 #TODO * 2.0
    # UBelemagnles = np.random.random((3,nbUBs))*360-180
    orientation_send = []
    orientation_send1 = []
    if flag == 0:
        g = np.zeros((nbUBs, 3, 3))
        if key_material != key_material1:
            g1 = np.zeros((nbUBs1, 3, 3))

        if mode == "random":
            if key_material != key_material1:
                for igr in range(nbUBs1):    
                    phi1 = rand1() * 360.
                    phi = 180. * acos(2 * rand1() - 1) / np.pi
                    phi2 = rand1() * 360.
                    g1[igr] = Euler2OrientationMatrix((phi1, phi, phi2))
                    orientation_send1.append(g1[igr])
                    
            for igr in range(nbUBs):
                phi1 = rand1() * 360.
                phi = 180. * acos(2 * rand1() - 1) / np.pi
                phi2 = rand1() * 360.
                g[igr] = Euler2OrientationMatrix((phi1, phi, phi2))
                orientation_send.append(g[igr])
                
        elif  mode == "uniform":
            if key_material != key_material1:
                g1 = odf_data1
                for igr in range(len(g1)):
                    orientation_send1.append(g1[igr])
            g = odf_data
            for igr in range(len(g)):
                orientation_send.append(g[igr])
            
    elif flag == 1 or flag == 2 or flag == 3:
        nbUBs = 2
        g = np.zeros((nbUBs, 3, 3))
        for igr in range(nbUBs):
            if igr == 0:
                phi1 = rand1() * 360.
                phi = 180. * acos(2 * rand1() - 1) / np.pi
                phi2 = rand1() * 360.
                g[igr] = Euler2OrientationMatrix((phi1, phi, phi2))
                orientation_send.append(g[igr])
            elif igr == 1:
                phi2 = phi2 + misorientation_angle ## adding user defined deg misorientation along phi2
                g[igr] = Euler2OrientationMatrix((phi1, phi, phi2))
                orientation_send1.append(g[igr])                

    l_tth, l_chi, l_miller_ind, l_posx, l_posy, l_E, l_intensity = [],[],[],[],[],[],[]
    
    if flag == 1:
        for grainind in range(nbUBs):
            UBmatrix = g[grainind]
            grain = CP.Prepare_Grain(key_material, UBmatrix)
            s_tth, s_chi, s_miller_ind, \
                s_posx, s_posy, s_E= LT.SimulateLaue_full_np(grain, emin, emax,
                                                            detectorparameters,
                                                            pixelsize=pixelsize,
                                                            dim=(dim1, dim2),
                                                            detectordiameter=detectordiameter,
                                                            removeharmonics=removeharmonics)
            s_miller_ind = np.c_[s_miller_ind, np.zeros(len(s_miller_ind))]
            s_intensity = 1./s_E
            l_tth.append(s_tth)
            l_chi.append(s_chi)
            l_miller_ind.append(s_miller_ind)
            l_posx.append(s_posx)
            l_posy.append(s_posy)
            l_E.append(s_E)
            l_intensity.append(s_intensity)
            
    elif flag == 2:
        for grainind in range(nbUBs):
            UBmatrix = g[grainind]
            grain = CP.Prepare_Grain(key_material1, UBmatrix)
            s_tth, s_chi, s_miller_ind, \
                s_posx, s_posy, s_E= LT.SimulateLaue_full_np(grain, emin, emax,
                                                            detectorparameters,
                                                            pixelsize=pixelsize,
                                                            dim=(dim1, dim2),
                                                            detectordiameter=detectordiameter,
                                                            removeharmonics=removeharmonics)
            s_miller_ind = np.c_[s_miller_ind, np.ones(len(s_miller_ind))]
            s_intensity = 1./s_E
            l_tth.append(s_tth)
            l_chi.append(s_chi)
            l_miller_ind.append(s_miller_ind)
            l_posx.append(s_posx)
            l_posy.append(s_posy)
            l_E.append(s_E)
            l_intensity.append(s_intensity)
    
    elif flag == 3:
        for grainind in range(nbUBs):
            UBmatrix = g[grainind]
            if grainind == 0:
                grain = CP.Prepare_Grain(key_material, UBmatrix)
            else:
                grain = CP.Prepare_Grain(key_material1, UBmatrix)
            s_tth, s_chi, s_miller_ind, \
                s_posx, s_posy, s_E= LT.SimulateLaue_full_np(grain, emin, emax,
                                                            detectorparameters,
                                                            pixelsize=pixelsize,
                                                            dim=(dim1, dim2),
                                                            detectordiameter=detectordiameter,
                                                            removeharmonics=removeharmonics)
            s_miller_ind = np.c_[s_miller_ind, np.ones(len(s_miller_ind))]
            s_intensity = 1./s_E
            l_tth.append(s_tth)
            l_chi.append(s_chi)
            l_miller_ind.append(s_miller_ind)
            l_posx.append(s_posx)
            l_posy.append(s_posy)
            l_E.append(s_E)
            l_intensity.append(s_intensity)
    
    else:
        for grainind in range(nbUBs):
            UBmatrix = g[grainind]
            grain = CP.Prepare_Grain(key_material, UBmatrix)
            s_tth, s_chi, s_miller_ind, \
                s_posx, s_posy, s_E= LT.SimulateLaue_full_np(grain, emin, emax,
                                                            detectorparameters,
                                                            pixelsize=pixelsize,
                                                            dim=(dim1, dim2),
                                                            detectordiameter=detectordiameter,
                                                            removeharmonics=removeharmonics)
            s_miller_ind = np.c_[s_miller_ind, np.zeros(len(s_miller_ind))]
            s_intensity = 1./s_E
            l_tth.append(s_tth)
            l_chi.append(s_chi)
            l_miller_ind.append(s_miller_ind)
            l_posx.append(s_posx)
            l_posy.append(s_posy)
            l_E.append(s_E)
            l_intensity.append(s_intensity)
            
        if (key_material != key_material1):
            for grainind in range(nbUBs1):
                # print(nbUBs, nbUBs1, key_material, key_material1, flag)
                UBmatrix = g1[grainind]
                grain = CP.Prepare_Grain(key_material1, UBmatrix)
                s_tth, s_chi, s_miller_ind, \
                    s_posx, s_posy, s_E= LT.SimulateLaue_full_np(grain, emin, emax,
                                                                detectorparameters,
                                                                pixelsize=pixelsize,
                                                                dim=(dim1, dim2),
                                                                detectordiameter=detectordiameter,
                                                                removeharmonics=removeharmonics)
                s_miller_ind = np.c_[s_miller_ind, np.ones(len(s_miller_ind))]
                s_intensity = 1./s_E
                l_tth.append(s_tth)
                l_chi.append(s_chi)
                l_miller_ind.append(s_miller_ind)
                l_posx.append(s_posx)
                l_posy.append(s_posy)
                l_E.append(s_E)
                l_intensity.append(s_intensity)
    
    ## add constant UB matrix to the simulated data
    if phase_always_present != None:
        UBmatrix, key_material_new = phase_always_present.split(';')
        UBmat = []
        for kk in UBmatrix.split(","):
            UBmat.append(float(kk))
        UBmat = np.array(UBmat)
        UBmatrix = UBmat.reshape((3,3))
        grain = CP.Prepare_Grain(key_material_new, UBmatrix)
        s_tth, s_chi, s_miller_ind, \
            s_posx, s_posy, s_E= LT.SimulateLaue_full_np(grain, emin, emax,
                                                        detectorparameters,
                                                        pixelsize=pixelsize,
                                                        dim=(dim1, dim2),
                                                        detectordiameter=detectordiameter,
                                                        removeharmonics=removeharmonics)
            
        if key_material_new == key_material:
            s_miller_ind = np.c_[s_miller_ind, np.zeros(len(s_miller_ind))]
        elif key_material_new == key_material1:
            s_miller_ind = np.c_[s_miller_ind, np.ones(len(s_miller_ind))]
        s_intensity = 1./s_E
        l_tth.append(s_tth)
        l_chi.append(s_chi)
        l_miller_ind.append(s_miller_ind)
        l_posx.append(s_posx)
        l_posy.append(s_posy)
        l_E.append(s_E)
        l_intensity.append(s_intensity)
    
    #flat_list = [item for sublist in l for item in sublist]
    s_tth = np.array([item for sublist in l_tth for item in sublist])
    s_chi = np.array([item for sublist in l_chi for item in sublist])
    s_miller_ind = np.array([item for sublist in l_miller_ind for item in sublist])
    s_posx = np.array([item for sublist in l_posx for item in sublist])
    s_posy = np.array([item for sublist in l_posy for item in sublist])
    s_E = np.array([item for sublist in l_E for item in sublist])
    s_intensity=np.array([item for sublist in l_intensity for item in sublist])
    
    if sortintensity:
        indsort = np.argsort(s_intensity)[::-1]
        s_tth=np.take(s_tth, indsort)
        s_chi=np.take(s_chi, indsort)
        s_miller_ind=np.take(s_miller_ind, indsort, axis=0)
        s_posx=np.take(s_posx, indsort)
        s_posy=np.take(s_posy, indsort)
        s_E=np.take(s_E, indsort)
        s_intensity=np.take(s_intensity, indsort)
        
    return s_tth, s_chi, s_miller_ind, s_posx, s_posy, s_intensity, orientation_send, orientation_send1

def chunker_list(seq, size):
    return (seq[i::size] for i in range(size))

def worker_generation(inputs_queue, outputs_queue, proc_id):
    while True:
        time.sleep(0.01)
        if not inputs_queue.empty():
            message = inputs_queue.get()
            num1, _, meta = message
            flag1 = meta['flag']
            for ijk in range(len(num1)):
                nb, nb1, material_, material1_, emin, emax, detectorparameters, pixelsize, \
                 sortintensity, ang_maxx, step, classhkl, classhkl1, noisy_data, \
                 remove_peaks, seed,hkl_all, lattice_material, family_hkl,\
                 normal_hkl, index_hkl, hkl_all1, lattice_material1, family_hkl1,\
                 normal_hkl1, index_hkl1, dim1, dim2, removeharmonics, flag,\
                 img_i, img_j, save_directory_, odf_data, odf_data1, modelp,\
                     misorientation_angle, max_millerindex, max_millerindex1,\
                         general_diff_cond, crystal, crystal1, phase_always_present = num1[ijk]


                getpatterns_(nb, nb1, material_, material1_, emin, emax, detectorparameters, pixelsize, \
                                         sortintensity, ang_maxx, step, classhkl, classhkl1, noisy_data, \
                                         remove_peaks, seed,hkl_all, lattice_material, family_hkl,\
                                         normal_hkl, index_hkl, hkl_all1, lattice_material1, family_hkl1,\
                                         normal_hkl1, index_hkl1, dim1, dim2, removeharmonics, flag,\
                                         img_i, img_j, save_directory_, odf_data, odf_data1, modelp, \
                                         misorientation_angle, max_millerindex, max_millerindex1, general_diff_cond, crystal, \
                                             crystal1, phase_always_present)
                    
                if ijk%10 == 0 and ijk!=0:
                    outputs_queue.put(11)
            if flag1 == 1:
                break
            
def ComputeGnomon_singledata(tth, chi, CenterProjection=(45 * DEG, 0 * DEG)):
    data_theta = tth / 2.0
    data_chi = chi
    lat = np.arcsin(np.cos(data_theta * DEG) * np.cos(data_chi * DEG))  # in rads
    longit = np.arctan(
        -np.sin(data_chi * DEG) / np.tan(data_theta * DEG))  # + ones(len(data_chi))*(np.pi)
    centerlat, centerlongit = CenterProjection
    slat0 = np.sin(centerlat)
    clat0 = np.cos(centerlat)
    longit0 = centerlongit
    slat = np.sin(lat)
    clat = np.cos(lat)
    cosanguldist = slat * slat0 + clat * clat0 * np.cos(longit - longit0)
    Xgno = clat * np.sin(longit0 - longit) / cosanguldist
    Ygno = (slat * clat0 - clat * slat0 * np.cos(longit - longit0)) / cosanguldist
    NbptsGno = 300
    maxsize = max(Xgno,Ygno,-Xgno,-Ygno)+.0
    xgnomin,xgnomax,ygnomin,ygnomax=(-0.8,0.8,-0.5,0.5)
    xgnomin,xgnomax,ygnomin,ygnomax=(-maxsize,maxsize,-maxsize,maxsize)
    XGNO = int((Xgno-xgnomin)/(xgnomax-xgnomin)*NbptsGno)
    YGNO = int((Ygno-ygnomin)/(ygnomax-ygnomin)*NbptsGno)
    return np.array((XGNO, YGNO))

def ComputeGnomon_2(TwiceTheta_Chi, CenterProjection=(45 * DEG, 0 * DEG)):
    data_theta = TwiceTheta_Chi[0] / 2.0
    data_chi = TwiceTheta_Chi[1]
    lat = np.arcsin(np.cos(data_theta * DEG) * np.cos(data_chi * DEG))  # in rads
    longit = np.arctan(
        -np.sin(data_chi * DEG) / np.tan(data_theta * DEG))  # + ones(len(data_chi))*(np.pi)
    centerlat, centerlongit = CenterProjection
    slat0 = np.ones(len(data_chi)) * np.sin(centerlat)
    clat0 = np.ones(len(data_chi)) * np.cos(centerlat)
    longit0 = np.ones(len(data_chi)) * centerlongit
    slat = np.sin(lat)
    clat = np.cos(lat)
    cosanguldist = slat * slat0 + clat * clat0 * np.cos(longit - longit0)
    _gnomonx = clat * np.sin(longit0 - longit) / cosanguldist
    _gnomony = (slat * clat0 - clat * slat0 * np.cos(longit - longit0)) / cosanguldist
    return _gnomonx, _gnomony

def computeGnomonicImage(TwiceTheta,Chi):
    DEG = np.pi/180.
    # CenterProjectionAngleTheta = 50#45
    TwiceTheta_Chi = TwiceTheta,Chi
    Xgno,Ygno = ComputeGnomon_2(TwiceTheta_Chi, CenterProjection=(45 * DEG, 0 * DEG))
    pts =(np.array([Xgno,Ygno]).T)
    nbpeaks=len(pts)
    NbptsGno = 300
    maxsize = max(Xgno.max(),Ygno.max(),-Xgno.min(),-Ygno.min())+.0
    xgnomin,xgnomax,ygnomin,ygnomax=(-0.8,0.8,-0.5,0.5)
    xgnomin,xgnomax,ygnomin,ygnomax=(-maxsize,maxsize,-maxsize,maxsize)
    
    halfdiagonal = np.sqrt(xgnomax**2+ygnomax**2)*NbptsGno
    XGNO = np.array((Xgno-xgnomin)/(xgnomax-xgnomin)*NbptsGno, dtype=np.int)
    YGNO = np.array((Ygno-ygnomin)/(ygnomax-ygnomin)*NbptsGno, dtype=np.int)
    imageGNO=np.zeros((NbptsGno+1,NbptsGno+1))
    imageGNO[XGNO,YGNO]=100
    return imageGNO, nbpeaks, halfdiagonal

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

def predict(x, wb, temp_key):
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
    
    
    # # first layer
    # layer0 = np.dot(x, wb[temp_key[1]]) + wb[temp_key[0]] 
    # layer0 = np.maximum(0, layer0) ## ReLU activation
    # # Second layer
    # layer1 = np.dot(layer0, wb[temp_key[3]]) + wb[temp_key[2]] 
    # layer3 = softmax(layer1) ## output softmax activation
    
    return layer3

def worker(inputs_queue, outputs_queue, proc_id, run_flag):#, mp_rotation_matrix):
    print(f'Initializing worker {proc_id}')
    while True:
        if not run_flag.value:
            break
        time.sleep(0.01)
        if not inputs_queue.empty(): 
            message = inputs_queue.get()
            if message == 'STOP':
                print(f'[{proc_id}] stopping')
                break

            num1, num2, meta = message
            files_worked = []
            while True:
                if len(num1) == len(files_worked) or len(num1) == 0:
                    print("process finished")
                    break
                for ijk in range(len(num1)):
                    if ijk in files_worked:
                        continue                       
                    if not run_flag.value:
                        num1, files_worked = [], []
                        print(f'[{proc_id}] stopping')
                        break
                    
                    files, cnt, rotation_matrix, strain_matrix, strain_matrixs,\
                    col,colx,coly,match_rate,spots_len,iR_pix,fR_pix,best_match,mat_global,\
                    check,detectorparameters,pixelsize,angbins,\
                    classhkl, hkl_all_class0, hkl_all_class1, emin, emax,\
                    material_, material1_, symmetry, symmetry1,lim_x,lim_y,\
                    strain_calculation, ind_mat, ind_mat1,\
                    model_direc, tolerance , tolerance1,\
                    matricies, ccd_label,\
                    filename_bkg,intensity_threshold,\
                    boxsize,bkg_treatment,\
                    filenameDirec, experimental_prefix,\
                    blacklist_file, text_file, \
                    files_treated,try_previous1,\
                    wb, temp_key, cor_file_directory, mode_spotCycle1,\
                    softmax_threshold_global123,mr_threshold_global123,\
                    cap_matchrate123, tolerance_strain123, tolerance_strain1231,\
                    NumberMaxofFits123,fit_peaks_gaussian_global123,\
                    FitPixelDev_global123,coeff123,coeff_overlap,\
                    material0_limit, material1_limit, use_previous_UBmatrix_name1,\
                        material_phase_always_present1, crystal, crystal1, strain_free_parameters = num1[ijk]
                    
                    if np.all(check[cnt,:]) == 1:
                        continue
                    
                    if os.path.isfile(files):
                        # try:                        
                        strain_matrix12, strain_matrixs12, \
                            rotation_matrix12, col12, \
                                colx12, coly12,\
                        match_rate12, mat_global12, cnt12,\
                            files_treated12, spots_len12, \
                                iR_pix12, fR_pix12, check12, \
                                    best_match12, pred_hkl = predict_preprocessMP(files, cnt, 
                                                                   rotation_matrix,strain_matrix,strain_matrixs,
                                                                   col,colx,coly,match_rate,spots_len,iR_pix,fR_pix,best_match,
                                                                   mat_global,
                                                                   check,detectorparameters,pixelsize,angbins,
                                                                   classhkl, hkl_all_class0, hkl_all_class1, emin, emax,
                                                                   material_, material1_, symmetry, symmetry1,lim_x,lim_y,
                                                                   strain_calculation, ind_mat, ind_mat1,
                                                                   model_direc, tolerance, tolerance1,
                                                                   matricies, ccd_label,
                                                                   filename_bkg,intensity_threshold,
                                                                   boxsize,bkg_treatment,
                                                                   filenameDirec, experimental_prefix,
                                                                   blacklist_file, text_file, 
                                                                   files_treated,try_previous1,
                                                                   wb, temp_key, cor_file_directory, mode_spotCycle1,
                                                                   softmax_threshold_global123,mr_threshold_global123,
                                                                   cap_matchrate123, tolerance_strain123,
                                                                   tolerance_strain1231,NumberMaxofFits123,
                                                                   fit_peaks_gaussian_global123,
                                                                   FitPixelDev_global123, coeff123,coeff_overlap,
                                                                   material0_limit,material1_limit,
                                                                   use_previous_UBmatrix_name1,
                                                                   material_phase_always_present1,
                                                                   crystal, crystal1, strain_free_parameters)
                        files_worked.append(ijk)
                        meta['proc_id'] = proc_id
                        r_message = (strain_matrix12, strain_matrixs12, rotation_matrix12, col12, \
                                     colx12, coly12, match_rate12, mat_global12, cnt12, meta, \
                                     files_treated12, spots_len12, iR_pix12, fR_pix12, best_match12, check12)
                        outputs_queue.put(r_message)
                        # except Exception as e:
                        #     print(e)
                        #     continue
    print("broke the worker while loop")

def predict_preprocessMP_vsingle(files, cnt, 
                         rotation_matrix,strain_matrix,strain_matrixs,
                        col,colx,coly,match_rate,spots_len,iR_pix,fR_pix,best_match,mat_global,
                        check,detectorparameters,pixelsize,angbins,
                        classhkl, hkl_all_class0, hkl_all_class1, emin, emax,
                        material_, material1_, symmetry, symmetry1,lim_x,lim_y,
                        strain_calculation, ind_mat, ind_mat1,
                        model_direc=None, tolerance =None, tolerance1 =None,
                       matricies=None, ccd_label=None,               
                       filenameDirec=None, experimental_prefix=None,
                       files_treated=None,try_previous1=False,
                       wb=None, temp_key=None, cor_file_directory=None, mode_spotCycle1=None,
                       softmax_threshold_global123=None,mr_threshold_global123=None,
                       cap_matchrate123=None,tolerance_strain123=None,tolerance_strain1231=None,\
                       coeff123=None, coeff_overlap=None,
                       material0_limit=None, material1_limit=None, use_previous_UBmatrix_name=None,
                       material_phase_always_present=None, crystal=None, crystal1=None, peak_XY=None,
                       strain_free_parameters=None):
    
    if files in files_treated:
        return strain_matrix, strain_matrixs, rotation_matrix, col, colx, coly, \
            match_rate, mat_global, cnt, files_treated,spots_len,iR_pix,fR_pix, check, best_match
            
    print("# Predicting for "+ files)
    call_global()
    
    CCDLabel=ccd_label
    seednumber = "Experimental "+CCDLabel+" file" 
    s_ix = np.argsort(peak_XY[:, 2])[::-1]
    peak_XY = peak_XY[s_ix]

    framedim = dictLT.dict_CCD[CCDLabel][0]
    twicetheta, chi = Lgeo.calc_uflab(peak_XY[:,0], peak_XY[:,1], detectorparameters,
                                        returnAngles=1,
                                        pixelsize=pixelsize,
                                        kf_direction='Z>0')
    data_theta, data_chi = twicetheta/2., chi
    
    framedim = dictLT.dict_CCD[CCDLabel][0]
    dict_dp={}
    dict_dp['kf_direction']='Z>0'
    dict_dp['detectorparameters']=detectorparameters
    dict_dp['detectordistance']=detectorparameters[0]
    dict_dp['detectordiameter']=pixelsize*framedim[0]#TODO*2
    dict_dp['pixelsize']=pixelsize
    dict_dp['dim']=framedim
    dict_dp['peakX']=peak_XY[:,0]
    dict_dp['peakY']=peak_XY[:,1]
    dict_dp['intensity']=peak_XY[:,2]
    CCDcalib = {"CCDLabel":CCDLabel,
                "dd":detectorparameters[0], 
                "xcen":detectorparameters[1], 
                "ycen":detectorparameters[2], 
                "xbet":detectorparameters[3], 
                "xgam":detectorparameters[4],
                "pixelsize": pixelsize}
    path = os.path.normpath(files)
    IOLT.writefile_cor(cor_file_directory+"//"+path.split(os.sep)[-1].split(".")[0], twicetheta, 
                       chi, peak_XY[:,0], peak_XY[:,1], peak_XY[:,2],
                       param=CCDcalib, sortedexit=0)
    
    sorted_data = np.transpose(np.array([data_theta, data_chi]))
    tabledistancerandom = np.transpose(GT.calculdist_from_thetachi(sorted_data, sorted_data))

    codebars_all = []

    spots_in_center = np.arange(0,len(data_theta))
    spots_in_center = spots_in_center[:nb_spots_consider]
    
    for i in spots_in_center:
        spotangles = tabledistancerandom[i]
        spotangles = np.delete(spotangles, i)# removing the self distance
        codebars = np.histogram(spotangles, bins=angbins)[0]
        # codebars = histogram1d(spotangles, range=[min(angbins),max(angbins)], bins=len(angbins)-1)
        ## normalize the same way as training data
        max_codebars = np.max(codebars)
        codebars = codebars/ max_codebars
        codebars_all.append(codebars)
    ## reshape for the model to predict all spots at once
    codebars = np.array(codebars_all)
    ## Do prediction of all spots at once
    prediction = predict(codebars, wb, temp_key)
    max_pred = np.max(prediction, axis = 1)
    class_predicted = np.argmax(prediction, axis = 1)
    predicted_hkl123 = classhkl[class_predicted]
    predicted_hkl123 = predicted_hkl123.astype(int)

    s_tth = data_theta * 2.
    s_chi = data_chi

    rotation_matrix1, mr_highest, mat_highest, \
        strain_crystal, strain_sample, iR_pix1, \
                    fR_pix1, spots_len1,\
                    best_match1, check12 = predict_ubmatrix(seednumber, spots_in_center, classhkl, 
                                                hkl_all_class0, 
                                                hkl_all_class1, files,
                                                  s_tth1=s_tth,s_chi1=s_chi,
                                                  predicted_hkl1=predicted_hkl123,
                                                  class_predicted1=class_predicted,
                                                  max_pred1=max_pred,
                                                  emin=emin,emax=emax,
                                                  material_=material_, 
                                                  material1_=material1_, 
                                                  lim_y=lim_y, lim_x=lim_x, 
                                                  cnt=cnt,
                                                  dict_dp=dict_dp,
                                                  rotation_matrix=rotation_matrix,
                                                  mat_global=mat_global,
                                                  strain_calculation=strain_calculation,
                                                  ind_mat=ind_mat, 
                                                  ind_mat1=ind_mat1,
                                                  tolerance=tolerance, 
                                                  tolerance1 =tolerance1,
                                                  matricies=matricies,
                                                  tabledistancerandom=tabledistancerandom,
                                                  text_file = None,
                                                  try_previous1=True,
                                                  mode_spotCycle=mode_spotCycle1,
                                                  softmax_threshold_global123 = softmax_threshold_global123,
                                                  mr_threshold_global123=mr_threshold_global123,
                                                  cap_matchrate123=cap_matchrate123,
                                                  tolerance_strain123=tolerance_strain123,
                                                  tolerance_strain1231=tolerance_strain1231,
                                                  coeff123=coeff123,
                                                  coeff_overlap=coeff_overlap,
                                                  material0_limit=material0_limit, 
                                                  material1_limit=material1_limit,
                                                  model_direc=model_direc,
                                                  use_previous_UBmatrix_name=use_previous_UBmatrix_name,
                                                  material_phase_always_present=material_phase_always_present,
                                                  match_rate=match_rate,
                                                  check=check[cnt,:],
                                                  crystal=crystal,
                                                  crystal1=crystal1, angbins=angbins,
                                                  wb=wb, temp_key=temp_key,
                                                  strain_free_parameters=strain_free_parameters)
    
    for intmat in range(matricies):
        if len(rotation_matrix1[intmat]) == 0:
            col[intmat][0][cnt,:] = 0,0,0
            colx[intmat][0][cnt,:] = 0,0,0
            coly[intmat][0][cnt,:] = 0,0,0
        else:
            mat_global[intmat][0][cnt] = mat_highest[intmat][0]
            
            final_symm =symmetry
            final_crystal = crystal
            if mat_highest[intmat][0] == 1:
                final_symm = symmetry
                final_crystal = crystal
            elif mat_highest[intmat][0] == 2:
                final_symm = symmetry1
                final_crystal = crystal1
            symm_operator = final_crystal._hklsym
            strain_matrix[intmat][0][cnt,:,:] = strain_crystal[intmat][0]
            strain_matrixs[intmat][0][cnt,:,:] = strain_sample[intmat][0]
            rotation_matrix[intmat][0][cnt,:,:] = rotation_matrix1[intmat][0]
            col_temp = get_ipf_colour(rotation_matrix1[intmat][0], np.array([0., 0., 1.]), final_symm, symm_operator)
            col[intmat][0][cnt,:] = col_temp
            col_tempx = get_ipf_colour(rotation_matrix1[intmat][0], np.array([1., 0., 0.]), final_symm, symm_operator)
            colx[intmat][0][cnt,:] = col_tempx
            col_tempy = get_ipf_colour(rotation_matrix1[intmat][0], np.array([0., 1., 0.]), final_symm, symm_operator)
            coly[intmat][0][cnt,:] = col_tempy
            match_rate[intmat][0][cnt] = mr_highest[intmat][0]
            spots_len[intmat][0][cnt] = spots_len1[intmat][0]
            iR_pix[intmat][0][cnt] = iR_pix1[intmat][0]
            fR_pix[intmat][0][cnt] = fR_pix1[intmat][0]
            best_match[intmat][0][cnt] = best_match1[intmat][0]
            check[cnt,intmat] = check12[intmat]

    files_treated.append(files)
    return strain_matrix, strain_matrixs, rotation_matrix, col, colx, coly, match_rate, \
            mat_global, cnt, files_treated, spots_len, iR_pix, fR_pix, check, best_match, predicted_hkl123

def predict_preprocessMP(files, cnt, 
                         rotation_matrix,strain_matrix,strain_matrixs,
                        col,colx,coly,match_rate,spots_len,iR_pix,fR_pix,best_match,mat_global,
                        check,detectorparameters,pixelsize,angbins,
                        classhkl, hkl_all_class0, hkl_all_class1, emin, emax,
                        material_, material1_, symmetry, symmetry1,lim_x,lim_y,
                        strain_calculation, ind_mat, ind_mat1,
                        model_direc=None, tolerance =None, tolerance1 =None,
                       matricies=None, ccd_label=None,
                       filename_bkg=None,intensity_threshold=None,
                       boxsize=None,bkg_treatment=None,
                       filenameDirec=None, experimental_prefix=None,
                       blacklist_file =None, text_file=None, 
                       files_treated=None,try_previous1=False,
                       wb=None, temp_key=None, cor_file_directory=None, mode_spotCycle1=None,
                       softmax_threshold_global123=None,mr_threshold_global123=None,
                       cap_matchrate123=None,tolerance_strain123=None,tolerance_strain1231=None,\
                       NumberMaxofFits123=None,fit_peaks_gaussian_global123=None,
                       FitPixelDev_global123=None,coeff123=None, coeff_overlap=None,
                       material0_limit=None, material1_limit=None, use_previous_UBmatrix_name=None,
                       material_phase_always_present=None, crystal=None, crystal1=None, strain_free_parameters=None):
    
    if files in files_treated:
        return strain_matrix, strain_matrixs, rotation_matrix, col, colx, coly, \
            match_rate, mat_global, cnt, files_treated,spots_len,iR_pix,fR_pix, check, best_match, None
            
    print("# Predicting for "+ files)
    
    call_global()

    if files.split(".")[-1] != "cor":
        CCDLabel=ccd_label
        seednumber = "Experimental "+CCDLabel+" file"    
        
        try:
            out_name = blacklist_file
        except:
            out_name = None  
            
        if bkg_treatment == None:
            bkg_treatment = "A-B"

        try:
            ### Max space = space between pixles
            peak_XY = RMCCD.PeakSearch(
                                        files,
                                        stackimageindex = -1,
                                        CCDLabel=CCDLabel,
                                        NumberMaxofFits=NumberMaxofFits123,
                                        PixelNearRadius=10,
                                        removeedge=2,
                                        IntensityThreshold=intensity_threshold,
                                        local_maxima_search_method=0,
                                        boxsize=boxsize,
                                        position_definition=1,
                                        verbose=0,
                                        fit_peaks_gaussian=fit_peaks_gaussian_global123,
                                        xtol=0.001,                
                                        FitPixelDev=FitPixelDev_global123,
                                        return_histo=0,
                                        # Saturation_value=1e10,  # to be merged in CCDLabel
                                        # Saturation_value_flatpeak=1e10,
                                        MinIntensity=0,
                                        PeakSizeRange=(0.65,200),
                                        write_execution_time=1,
                                        Data_for_localMaxima = "auto_background",
                                        formulaexpression=bkg_treatment,
                                        Remove_BlackListedPeaks_fromfile=out_name,
                                        reject_negative_baseline=True,
                                        Fit_with_Data_for_localMaxima=False,
                                        maxPixelDistanceRejection=15.0,
                                        )
            peak_XY = peak_XY[0]#[:,:2] ##[2] Integer peak lists
        except:
            print("Error in Peak detection for "+ files)
            for intmat in range(matricies):
                rotation_matrix[intmat][0][cnt,:,:] = np.zeros((3,3))
                strain_matrix[intmat][0][cnt,:,:] = np.zeros((3,3))
                strain_matrixs[intmat][0][cnt,:,:] = np.zeros((3,3))
                col[intmat][0][cnt,:] = 0,0,0
                colx[intmat][0][cnt,:] = 0,0,0
                coly[intmat][0][cnt,:] = 0,0,0
                match_rate[intmat][0][cnt] = 0
                mat_global[intmat][0][cnt] = 0
                spots_len[intmat][0][cnt] = 0
                iR_pix[intmat][0][cnt] = 0
                fR_pix[intmat][0][cnt] = 0
                check[cnt,intmat] = 0
            files_treated.append(files)
            return strain_matrix, strain_matrixs, rotation_matrix, col, colx, coly, \
                match_rate, mat_global, cnt, files_treated,spots_len,iR_pix,fR_pix, check, best_match, None

        try:
            s_ix = np.argsort(peak_XY[:, 2])[::-1]
            peak_XY = peak_XY[s_ix]
        except:
            print("Error in Peak detection (argsort routine) for "+ files)
            for intmat in range(matricies):
                rotation_matrix[intmat][0][cnt,:,:] = np.zeros((3,3))
                strain_matrix[intmat][0][cnt,:,:] = np.zeros((3,3))
                strain_matrixs[intmat][0][cnt,:,:] = np.zeros((3,3))
                col[intmat][0][cnt,:] = 0,0,0
                colx[intmat][0][cnt,:] = 0,0,0
                coly[intmat][0][cnt,:] = 0,0,0
                match_rate[intmat][0][cnt] = 0
                mat_global[intmat][0][cnt] = 0
                spots_len[intmat][0][cnt] = 0
                iR_pix[intmat][0][cnt] = 0
                fR_pix[intmat][0][cnt] = 0
                check[cnt,intmat] = 0
            files_treated.append(files)
            return strain_matrix, strain_matrixs, rotation_matrix, col, colx, coly, \
                match_rate, mat_global, cnt, files_treated,spots_len,iR_pix,fR_pix, check, best_match, None

        framedim = dictLT.dict_CCD[CCDLabel][0]
        twicetheta, chi = Lgeo.calc_uflab(peak_XY[:,0], peak_XY[:,1], detectorparameters,
                                            returnAngles=1,
                                            pixelsize=pixelsize,
                                            kf_direction='Z>0')
        data_theta, data_chi = twicetheta/2., chi
        
        framedim = dictLT.dict_CCD[CCDLabel][0]
        dict_dp={}
        dict_dp['kf_direction']='Z>0'
        dict_dp['detectorparameters']=detectorparameters
        dict_dp['detectordistance']=detectorparameters[0]
        dict_dp['detectordiameter']=pixelsize*framedim[0]#TODO*2
        dict_dp['pixelsize']=pixelsize
        dict_dp['dim']=framedim
        dict_dp['peakX']=peak_XY[:,0]
        dict_dp['peakY']=peak_XY[:,1]
        dict_dp['intensity']=peak_XY[:,2]
        CCDcalib = {"CCDLabel":CCDLabel,
                    "dd":detectorparameters[0], 
                    "xcen":detectorparameters[1], 
                    "ycen":detectorparameters[2], 
                    "xbet":detectorparameters[3], 
                    "xgam":detectorparameters[4],
                    "pixelsize": pixelsize}
        path = os.path.normpath(files)
        IOLT.writefile_cor(cor_file_directory+"//"+path.split(os.sep)[-1].split(".")[0], twicetheta, 
                           chi, peak_XY[:,0], peak_XY[:,1], peak_XY[:,2],
                           param=CCDcalib, sortedexit=0)
        
    elif files.split(".")[-1] == "cor":
        seednumber = "Experimental COR file"
        allres = IOLT.readfile_cor(files, True)
        data_theta, data_chi, peakx, peaky, intensity = allres[1:6]
        CCDcalib = allres[-1]
        detectorparameters = allres[-2]
        pixelsize = CCDcalib['pixelsize']
        CCDLabel = CCDcalib['CCDLabel']
        framedim = dictLT.dict_CCD[CCDLabel][0]
        dict_dp={}
        dict_dp['kf_direction']='Z>0'
        dict_dp['detectorparameters']=detectorparameters
        dict_dp['detectordistance']=detectorparameters[0]
        dict_dp['detectordiameter']=pixelsize*framedim[0]#TODO*2
        dict_dp['pixelsize']=pixelsize
        dict_dp['dim']=framedim
        dict_dp['peakX']=peakx
        dict_dp['peakY']=peaky
        dict_dp['intensity']=intensity

    sorted_data = np.transpose(np.array([data_theta, data_chi]))
    tabledistancerandom = np.transpose(GT.calculdist_from_thetachi(sorted_data, sorted_data))

    codebars_all = []
    
    if len(data_theta) == 0:
        print("No peaks Found for : " + files)
        for intmat in range(matricies):
            rotation_matrix[intmat][0][cnt,:,:] = np.zeros((3,3))
            strain_matrix[intmat][0][cnt,:,:] = np.zeros((3,3))
            strain_matrixs[intmat][0][cnt,:,:] = np.zeros((3,3))
            col[intmat][0][cnt,:] = 0,0,0
            colx[intmat][0][cnt,:] = 0,0,0
            coly[intmat][0][cnt,:] = 0,0,0
            match_rate[intmat][0][cnt] = 0
            mat_global[intmat][0][cnt] = 0
            spots_len[intmat][0][cnt] = 0
            iR_pix[intmat][0][cnt] = 0
            fR_pix[intmat][0][cnt] = 0
            check[cnt,intmat] = 0
        files_treated.append(files)
        return strain_matrix, strain_matrixs, rotation_matrix, col, colx, coly, \
                match_rate, mat_global, cnt, files_treated,spots_len,iR_pix,fR_pix, check, best_match, None
    
    if not use_om_user:
        spots_in_center = np.arange(0,len(data_theta))
        spots_in_center = spots_in_center[:nb_spots_consider]
        
        for i in spots_in_center:
            spotangles = tabledistancerandom[i]
            spotangles = np.delete(spotangles, i)# removing the self distance
            codebars = np.histogram(spotangles, bins=angbins)[0]
            # codebars = histogram1d(spotangles, range=[min(angbins),max(angbins)], bins=len(angbins)-1)
            ## normalize the same way as training data
            max_codebars = np.max(codebars)
            codebars = codebars/ max_codebars
            codebars_all.append(codebars)
        ## reshape for the model to predict all spots at once
        codebars = np.array(codebars_all)
        ## Do prediction of all spots at once
        prediction = predict(codebars, wb, temp_key)
        max_pred = np.max(prediction, axis = 1)
        class_predicted = np.argmax(prediction, axis = 1)
        predicted_hkl123 = classhkl[class_predicted]
        predicted_hkl123 = predicted_hkl123.astype(int)
    else:
        max_pred = None
        class_predicted = None
        predicted_hkl123 = None
        spots_in_center = None

    s_tth = data_theta * 2.
    s_chi = data_chi

    rotation_matrix1, mr_highest, mat_highest, \
        strain_crystal, strain_sample, iR_pix1, \
                    fR_pix1, spots_len1,\
                    best_match1, check12 = predict_ubmatrix(seednumber, spots_in_center, classhkl, 
                                                hkl_all_class0, 
                                                hkl_all_class1, files,
                                                  s_tth1=s_tth,s_chi1=s_chi,
                                                  predicted_hkl1=predicted_hkl123,
                                                  class_predicted1=class_predicted,
                                                  max_pred1=max_pred,
                                                  emin=emin,emax=emax,
                                                  material_=material_, 
                                                  material1_=material1_, 
                                                  lim_y=lim_y, lim_x=lim_x, 
                                                  cnt=cnt,
                                                  dict_dp=dict_dp,
                                                  rotation_matrix=rotation_matrix,
                                                  mat_global=mat_global,
                                                  strain_calculation=strain_calculation,
                                                  ind_mat=ind_mat, 
                                                  ind_mat1=ind_mat1,
                                                  tolerance=tolerance, 
                                                  tolerance1 =tolerance1,
                                                  matricies=matricies,
                                                  tabledistancerandom=tabledistancerandom,
                                                  text_file = text_file,
                                                  try_previous1=try_previous1,
                                                  mode_spotCycle=mode_spotCycle1,
                                                  softmax_threshold_global123 = softmax_threshold_global123,
                                                  mr_threshold_global123=mr_threshold_global123,
                                                  cap_matchrate123=cap_matchrate123,
                                                  tolerance_strain123=tolerance_strain123,
                                                  tolerance_strain1231=tolerance_strain1231,
                                                  coeff123=coeff123,
                                                  coeff_overlap=coeff_overlap,
                                                  material0_limit=material0_limit, 
                                                  material1_limit=material1_limit,
                                                  model_direc=model_direc,
                                                  use_previous_UBmatrix_name=use_previous_UBmatrix_name,
                                                  material_phase_always_present=material_phase_always_present,
                                                  match_rate=match_rate,
                                                  check=check[cnt,:],
                                                  crystal=crystal,
                                                  crystal1=crystal1, angbins=angbins,
                                                  wb=wb, temp_key=temp_key,
                                                  strain_free_parameters=strain_free_parameters)
    
    for intmat in range(matricies):
        if len(rotation_matrix1[intmat]) == 0:
            col[intmat][0][cnt,:] = 0,0,0
            colx[intmat][0][cnt,:] = 0,0,0
            coly[intmat][0][cnt,:] = 0,0,0
        else:
            mat_global[intmat][0][cnt] = mat_highest[intmat][0]
            
            final_symm =symmetry
            final_crystal = crystal
            if mat_highest[intmat][0] == 1:
                final_symm = symmetry
                final_crystal = crystal
            elif mat_highest[intmat][0] == 2:
                final_symm = symmetry1
                final_crystal = crystal1
            symm_operator = final_crystal._hklsym
            strain_matrix[intmat][0][cnt,:,:] = strain_crystal[intmat][0]
            strain_matrixs[intmat][0][cnt,:,:] = strain_sample[intmat][0]
            rotation_matrix[intmat][0][cnt,:,:] = rotation_matrix1[intmat][0]
            col_temp = get_ipf_colour(rotation_matrix1[intmat][0], np.array([0., 0., 1.]), final_symm, symm_operator)
            col[intmat][0][cnt,:] = col_temp
            col_tempx = get_ipf_colour(rotation_matrix1[intmat][0], np.array([1., 0., 0.]), final_symm, symm_operator)
            colx[intmat][0][cnt,:] = col_tempx
            col_tempy = get_ipf_colour(rotation_matrix1[intmat][0], np.array([0., 1., 0.]), final_symm, symm_operator)
            coly[intmat][0][cnt,:] = col_tempy
            match_rate[intmat][0][cnt] = mr_highest[intmat][0]
            spots_len[intmat][0][cnt] = spots_len1[intmat][0]
            iR_pix[intmat][0][cnt] = iR_pix1[intmat][0]
            fR_pix[intmat][0][cnt] = fR_pix1[intmat][0]
            best_match[intmat][0][cnt] = best_match1[intmat][0]
            check[cnt,intmat] = check12[intmat]

    files_treated.append(files)
    return strain_matrix, strain_matrixs, rotation_matrix, col, colx, coly, match_rate, \
            mat_global, cnt, files_treated, spots_len, iR_pix, fR_pix, check, best_match, predicted_hkl123


def predict_ubmatrix(seednumber, spots_in_center, classhkl, hkl_all_class0, 
                     hkl_all_class1, filename, 
                     s_tth1,s_chi1,predicted_hkl1,class_predicted1,max_pred1,
                     emin, emax, material_, material1_, lim_y, lim_x, cnt,
                     dict_dp,rotation_matrix,mat_global,strain_calculation,
                     ind_mat, ind_mat1,
                     tolerance=None,  tolerance1 =None, matricies=None, tabledistancerandom=None,
                     text_file=None,try_previous1=False, mode_spotCycle=None,
                     softmax_threshold_global123=None,mr_threshold_global123=None,
                     cap_matchrate123=None, tolerance_strain123=None,tolerance_strain1231=None, coeff123=None,
                     coeff_overlap=None, material0_limit=None, material1_limit=None, model_direc=None,
                     use_previous_UBmatrix_name=None, material_phase_always_present=None, match_rate=None,
                     check = None, crystal=None, crystal1=None, angbins=None, wb=None, temp_key=None,
                     strain_free_parameters=None):
    
    input_params = {"tolerance": tolerance,
                    "tolerance1":tolerance1,
                    "tolerancestrain": tolerance_strain123, ## For strain calculations
                    "tolerancestrain1": tolerance_strain1231,
                    "emin": emin,
                    "emax": emax,
                    "mat":0}
    call_global()
    
    strain_matrix = [[] for i in range(matricies)]
    strain_matrixs = [[] for i in range(matricies)]
    best_matrix = [[] for i in range(matricies)]
    mr_highest = [[] for i in range(matricies)]
    ir_pixels = [[] for i in range(matricies)]
    fr_pixels = [[] for i in range(matricies)]
    spots_len = [[] for i in range(matricies)]
    mat_highest = [[] for i in range(matricies)]
    best_match = [[] for i in range(matricies)]
    spots1 = []
    spots1_global = [[] for i in range(matricies)]
    
    if not use_om_user: 
        dist = tabledistancerandom        
        ## one time calculations
        lattice_params0 = dictLT.dict_Materials[material_][1]
        B0 = CP.calc_B_RR(lattice_params0)
        Gstar_metric0 = CP.Gstar_from_directlatticeparams(lattice_params0[0],lattice_params0[1],\
                                                         lattice_params0[2],lattice_params0[3],\
                                                             lattice_params0[4],lattice_params0[5])
        tab_distance_classhkl_data0 = get_material_dataP(Gstar_metric0, predicted_hkl1[:nb_spots_consider,:])
        
        if material_ != material1_:
            lattice_params1 = dictLT.dict_Materials[material1_][1]
            B1 = CP.calc_B_RR(lattice_params1)
            Gstar_metric1 = CP.Gstar_from_directlatticeparams(lattice_params1[0],lattice_params1[1],\
                                                             lattice_params1[2],lattice_params1[3],\
                                                                 lattice_params1[4],lattice_params1[5])
            tab_distance_classhkl_data1 = get_material_dataP(Gstar_metric1, predicted_hkl1[:nb_spots_consider,:])
        else:
            tab_distance_classhkl_data1 = None
            Gstar_metric1 = None
            B1 = None
    else:
        dist = tabledistancerandom 
        tab_distance_classhkl_data0 = None
        tab_distance_classhkl_data1 = None
        ## one time calculations
        lattice_params0 = dictLT.dict_Materials[material_][1]
        B0 = CP.calc_B_RR(lattice_params0)
        Gstar_metric0 = CP.Gstar_from_directlatticeparams(lattice_params0[0],lattice_params0[1],\
                                                         lattice_params0[2],lattice_params0[3],\
                                                             lattice_params0[4],lattice_params0[5])        
        if material_ != material1_:
            lattice_params1 = dictLT.dict_Materials[material1_][1]
            B1 = CP.calc_B_RR(lattice_params1)
            Gstar_metric1 = CP.Gstar_from_directlatticeparams(lattice_params1[0],lattice_params1[1],\
                                                             lattice_params1[2],lattice_params1[3],\
                                                                 lattice_params1[4],lattice_params1[5])
        else:
            Gstar_metric1 = None
            B1 = None
        
    spots = []
    first_match = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.zeros((3,3))]
    max_mr = 0
    mat = 0
    iR = 0
    fR = 0
    strain_crystal = np.zeros((3,3))
    strain_sample = np.zeros((3,3))
    material0_count = 0
    material1_count = 0
    calcul_done = False
    objective_function1 = None
    
    for igrain in range(matricies):
        # if check[igrain] == 1: # or len(spots1_global[igrain]) != 0:
        #     continue
        
        try_previous = try_previous1
        max_mr, min_mr = 0, 0
        iR, fR= 0, 0
        case = "None"
        
        if use_om_user:
            use_previous_UBmatrix_name = False
            try_previous = False

            temp_qsd = np.loadtxt(path_user_OM, delimiter=",")
            temp_qsd = temp_qsd.reshape((len(temp_qsd),3,3))
            rotationmatrix_indexed = temp_qsd[igrain,:,:]
            
            mat = 1

            if mat == 1:
                Keymaterial_ = material_
                case = material_
                Bkey = B0
                input_params["mat"] = 1
                input_params["Bmat"] = Bkey
            elif mat == 2:
                Keymaterial_ = material1_
                case = material1_
                Bkey = B1
                input_params["mat"] = 2
                input_params["Bmat"] = Bkey
                
            spots_prev, theo_spots_prev = remove_spots(s_tth1, s_chi1, rotationmatrix_indexed, 
                                                         Keymaterial_, input_params, dict_dp['detectorparameters'],
                                                         dict_dp)
            newmatchrate = 100*len(spots_prev)/theo_spots_prev
            
            ## Filter indexation by matching rate
            if newmatchrate < cap_matchrate123:
                strain_crystal, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
                first_match = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                    0, 0, 0, 0, 0, np.zeros((3,3))]
                spots = []
                max_mr, min_mr = 0, 0
            else:
                if strain_calculation:
                    strain_crystal, strain_sample, iR, fR, rot_mat_UB = calculate_strains_fromUB(s_tth1, s_chi1, 
                                                                                  rotationmatrix_indexed,
                                                                                  Keymaterial_, 
                                                                                 input_params, dict_dp['detectorparameters'], 
                                                                                 dict_dp, spots1, Bkey,
                                                                                 strain_free_parameters)
                else:
                    strain_crystal, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
                    rot_mat_UB = np.copy(rotationmatrix_indexed)
                spots = spots_prev
                expected = theo_spots_prev
                max_mr, min_mr = 100*(len(spots)/expected), 100*(len(spots)/expected)
                first_match = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                0, len(spots), expected, max_mr, 0, rot_mat_UB]
                    
            try_previous = False
            calcul_done = True

        
        elif use_previous_UBmatrix_name:
            try:
                try_previous = False
                ### try already indexed UB matricies
                # xy = np.load('xy.npz')
                # xy.zip.fp.close()
                # xy.close()
                with np.load(model_direc+"//rotation_matrix_indexed_1.npz") as load_objectind:
                    # load_objectind = np.load(model_direc+"//rotation_matrix_indexed.npz")
                    rotationmatrix_indexed = load_objectind["arr_0"]
                    mat_global_indexed = load_objectind["arr_1"]
                    match_rate_indexed = load_objectind["arr_2"]
                    avg_match_rate_indexed = load_objectind["arr_3"]
                calcul_done = False
                for ind_mat_UBmat in range(len(rotationmatrix_indexed[igrain][0])):
                    if calcul_done:
                        continue
                    
                    if np.all(rotationmatrix_indexed[igrain][0][ind_mat_UBmat,:,:]) == 0:
                        continue

                    if match_rate_indexed[igrain][0][ind_mat_UBmat] < 0.8*avg_match_rate_indexed[igrain]:
                        continue
                    
                    mat = mat_global_indexed[igrain][0][ind_mat_UBmat]
                    if mat == 1:
                        Keymaterial_ = material_
                        case = material_
                        Bkey = B0
                        input_params["mat"] = 1
                        input_params["Bmat"] = Bkey
                    elif mat == 2:
                        Keymaterial_ = material1_
                        case = material1_
                        Bkey = B1
                        input_params["mat"] = 2
                        input_params["Bmat"] = Bkey
                    else:
                        Keymaterial_ = None
                        Bkey = None
                        input_params["mat"] = 0
                        input_params["Bmat"] = None
                        continue
                    
                    spots_prev, theo_spots_prev = remove_spots(s_tth1, s_chi1, rotationmatrix_indexed[igrain][0][ind_mat_UBmat,:,:], 
                                                                 Keymaterial_, input_params, dict_dp['detectorparameters'],
                                                                 dict_dp)
                    newmatchrate = 100*len(spots_prev)/theo_spots_prev
                    condition_prev = newmatchrate < 0.8*(match_rate_indexed[igrain][0][ind_mat_UBmat])
                    current_spots = [len(list(set(spots_prev) & set(spots1_global[igr]))) > coeff_overlap*len(spots1_global[igr]) for igr in range(len(spots1_global))]
                    
                    if condition_prev or (newmatchrate <= cap_matchrate123) or np.any(current_spots):# or overlap:
                        try_previous = try_previous1
                    else:
                        try_previous = False
                        calcul_done = True
                        if strain_calculation:
                            strain_crystal, strain_sample, iR, fR, rot_mat_UB = calculate_strains_fromUB(s_tth1, s_chi1, 
                                                                                          rotationmatrix_indexed[igrain][0][ind_mat_UBmat,:,:],
                                                                                          Keymaterial_, 
                                                                                         input_params, dict_dp['detectorparameters'], 
                                                                                         dict_dp, spots1, Bkey,
                                                                                         strain_free_parameters)
                        else:
                            strain_crystal, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
                            rot_mat_UB = np.copy(rotationmatrix_indexed[igrain][0][ind_mat_UBmat,:,:])
                        spots = spots_prev
                        expected = theo_spots_prev
                        max_mr, min_mr = 100*(len(spots)/expected), 100*(len(spots)/expected)
                        first_match = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                        0, len(spots), expected, max_mr, 0, rot_mat_UB]
                        break
            except:
                try_previous = False
                calcul_done = False
                
        if try_previous and (cnt % lim_y == 0) and cnt != 0:
            if np.all(rotation_matrix[igrain][0][cnt-lim_y,:,:]) == 0:
                try_previous = False
            else:
                mat = mat_global[igrain][0][cnt-lim_y]
                if mat == 1:
                    Keymaterial_ = material_
                    case = material_
                    Bkey = B0
                    input_params["mat"] = 1
                    input_params["Bmat"] = Bkey
                elif mat == 2:
                    Keymaterial_ = material1_
                    case = material1_
                    Bkey = B1
                    input_params["mat"] = 2
                    input_params["Bmat"] = Bkey
                else:
                    Keymaterial_ = None
                    Bkey = None
                    input_params["mat"] = 0
                    input_params["Bmat"] = None
                    continue
                
                spots_lr, theo_spots_lr = remove_spots(s_tth1, s_chi1, 
                                                            rotation_matrix[igrain][0][cnt-lim_y,:,:], 
                                                         Keymaterial_, input_params, dict_dp['detectorparameters'],
                                                         dict_dp)
                # last_row = len(spots_lr) <= coeff123*theo_spots_lr
                newmatchrate = 100*(len(spots_lr)/theo_spots_lr)
                condition_prev = newmatchrate < 0.9*(match_rate[igrain][0][cnt-lim_y])
                last_row = condition_prev
                if last_row or condition_prev: ## new spots less than 8 count, not good match SKIP
                    try_previous = False
                else:
                    try_previous = True
                    current_spots = [len(list(set(spots_lr) & set(spots1_global[igr]))) > coeff_overlap*len(spots1_global[igr]) for igr in range(len(spots1_global))]
                    if np.any(current_spots):
                        try_previous = False
                        continue
                    
                    if strain_calculation:
                        strain_crystal, strain_sample, iR, fR, rot_mat_UB = calculate_strains_fromUB(s_tth1, s_chi1, 
                                                                                      rotation_matrix[igrain][0][cnt-lim_y,:,:], 
                                                                                      Keymaterial_, 
                                                                                     input_params, dict_dp['detectorparameters'], 
                                                                                     dict_dp, spots1, Bkey,
                                                                                     strain_free_parameters)
                    else:
                        strain_crystal, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
                        rot_mat_UB = np.copy(rotation_matrix[igrain][0][cnt-lim_y,:,:])
                    spots = spots_lr
                    expected = theo_spots_lr
                    max_mr, min_mr = 100*(len(spots_lr)/theo_spots_lr), 100*(len(spots_lr)/theo_spots_lr)
                    first_match = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                    0, len(spots), expected, max_mr, 0, rot_mat_UB]   
                
        elif try_previous and (cnt % lim_y != 0):
            last_row = True
            left_row = True
            condition_prev = True
            condition_prev1 = True
            if np.all(rotation_matrix[igrain][0][cnt-1,:,:]) == 0:
                left_row = True
            else:
                mat = mat_global[igrain][0][cnt-1]
                if mat == 1:
                    Keymaterial_ = material_
                    case = material_
                    Bkey = B0
                    input_params["mat"] = 1
                    input_params["Bmat"] = Bkey
                elif mat == 2:
                    Keymaterial_ = material1_
                    case = material1_
                    Bkey = B1
                    input_params["mat"] = 2
                    input_params["Bmat"] = Bkey
                else:
                    Keymaterial_ = None
                    Bkey = None
                    input_params["mat"] = 0
                    input_params["Bmat"] = None
                    continue
                ## new row start when % == 0
                ## use left index pixels matrix values
                spots_left, theo_spots_left = remove_spots(s_tth1, s_chi1, rotation_matrix[igrain][0][cnt-1,:,:], 
                                                         Keymaterial_, input_params, dict_dp['detectorparameters'],
                                                         dict_dp)
                # left_row = len(spots_left) <= coeff123*theo_spots_left 
                newmatchrate = 100*(len(spots_left)/theo_spots_left)
                condition_prev = newmatchrate < 0.9*(match_rate[igrain][0][cnt-1])
                left_row = condition_prev
            if cnt >= lim_y:
                if np.all(rotation_matrix[igrain][0][cnt-lim_y,:,:]) == 0:
                    last_row = True   
                else:
                    mat = mat_global[igrain][0][cnt-lim_y]
                    if mat == 1:
                        Keymaterial_ = material_
                        case = material_
                        Bkey = B0
                        input_params["mat"] = 1
                        input_params["Bmat"] = Bkey
                    elif mat == 2:
                        Keymaterial_ = material1_
                        case = material1_
                        Bkey = B1
                        input_params["mat"] = 2
                        input_params["Bmat"] = Bkey
                    else:
                        Keymaterial_ = None
                        Bkey = None
                        input_params["mat"] = 0
                        input_params["Bmat"] = None
                        continue
                    ## use bottom index pixels matrix values
                    spots_lr, theo_spots_lr = remove_spots(s_tth1, s_chi1, rotation_matrix[igrain][0][cnt-lim_y,:,:], 
                                                             Keymaterial_, input_params, dict_dp['detectorparameters'],
                                                             dict_dp)
                    
                    # last_row = len(spots_lr) <= coeff123*theo_spots_lr 
                    newmatchrate1 = 100*(len(spots_lr)/theo_spots_lr)
                    condition_prev1 = newmatchrate1 < 0.9*(match_rate[igrain][0][cnt-lim_y])
                    last_row = condition_prev1
            if (left_row and last_row): 
                try_previous = False
            elif condition_prev and condition_prev1:
                try_previous = False
            elif not left_row and not last_row:
                try_previous = True
                
                if len(spots_lr) > len(spots_left):
                    current_spots = [len(list(set(spots_lr) & set(spots1_global[igr]))) > coeff_overlap*len(spots1_global[igr]) for igr in range(len(spots1_global))]
                    if np.any(current_spots):
                        try_previous = False
                        continue
                    
                    mat = mat_global[igrain][0][cnt-lim_y]
                    if mat == 1:
                        Keymaterial_ = material_
                        case = material_
                        Bkey = B0
                        input_params["mat"] = 1
                        input_params["Bmat"] = Bkey
                    elif mat == 2:
                        Keymaterial_ = material1_
                        case = material1_
                        Bkey = B1
                        input_params["mat"] = 2
                        input_params["Bmat"] = Bkey
                    else:
                        Keymaterial_ = None
                        Bkey = None
                        input_params["mat"] = 0
                        input_params["Bmat"] = None
                        continue
                    
                    if strain_calculation:
                        strain_crystal, strain_sample, iR, fR, rot_mat_UB = calculate_strains_fromUB(s_tth1, s_chi1, 
                                                                                      rotation_matrix[igrain][0][cnt-lim_y,:,:], 
                                                                                      Keymaterial_, 
                                                                                     input_params, dict_dp['detectorparameters'], 
                                                                                     dict_dp, spots1, Bkey,
                                                                                     strain_free_parameters)
                    else:
                        strain_crystal, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
                        rot_mat_UB = np.copy(rotation_matrix[igrain][0][cnt-lim_y,:,:])
                    spots = spots_lr
                    expected = theo_spots_lr
                    max_mr, min_mr = 100*(len(spots_lr)/theo_spots_lr), 100*(len(spots_lr)/theo_spots_lr)
                    first_match = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                0, len(spots), expected, max_mr, 0, rot_mat_UB]
                else:
                    current_spots = [len(list(set(spots_left) & set(spots1_global[igr]))) > coeff_overlap*len(spots1_global[igr]) for igr in range(len(spots1_global))]
                    if np.any(current_spots):
                        try_previous = False
                        continue

                    mat = mat_global[igrain][0][cnt-1]
                    if mat == 1:
                        Keymaterial_ = material_
                        case = material_
                        Bkey = B0
                        input_params["mat"] = 1
                        input_params["Bmat"] = Bkey
                    elif mat == 2:
                        Keymaterial_ = material1_
                        case = material1_
                        Bkey = B1
                        input_params["mat"] = 2
                        input_params["Bmat"] = Bkey
                    else:
                        Keymaterial_ = None
                        Bkey = None
                        input_params["mat"] = 0
                        input_params["Bmat"] = None
                        continue
                    
                    if strain_calculation:
                        strain_crystal, strain_sample, iR, fR, rot_mat_UB = calculate_strains_fromUB(s_tth1, s_chi1, 
                                                                                      rotation_matrix[igrain][0][cnt-1,:,:], 
                                                                                      Keymaterial_, 
                                                                                     input_params, dict_dp['detectorparameters'], 
                                                                                     dict_dp, spots1, Bkey,
                                                                                     strain_free_parameters)
                    else:
                        strain_crystal, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
                        rot_mat_UB = np.copy(rotation_matrix[igrain][0][cnt-1,:,:])
                    spots = spots_left
                    expected = theo_spots_left
                    max_mr, min_mr = 100*(len(spots_left)/theo_spots_left), 100*(len(spots_left)/theo_spots_left)
                    first_match = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                0, len(spots), expected, max_mr, 0, rot_mat_UB]    
            
            elif not left_row and last_row:
                try_previous = True
                current_spots = [len(list(set(spots_left) & set(spots1_global[igr]))) > coeff_overlap*len(spots1_global[igr]) for igr in range(len(spots1_global))]
                if np.any(current_spots):
                    try_previous = False
                    continue
                
                mat = mat_global[igrain][0][cnt-1]
                if mat == 1:
                    Keymaterial_ = material_
                    case = material_
                    Bkey = B0
                    input_params["mat"] = 1
                    input_params["Bmat"] = Bkey
                elif mat == 2:
                    Keymaterial_ = material1_
                    case = material1_
                    Bkey = B1
                    input_params["mat"] = 2
                    input_params["Bmat"] = Bkey
                else:
                    Keymaterial_ = None
                    Bkey = None
                    input_params["mat"] = 0
                    input_params["Bmat"] = None
                    continue
                
                if strain_calculation:
                    strain_crystal, strain_sample, iR, fR, rot_mat_UB = calculate_strains_fromUB(s_tth1, s_chi1, 
                                                                                  rotation_matrix[igrain][0][cnt-1,:,:], 
                                                                                  Keymaterial_, 
                                                                                 input_params, dict_dp['detectorparameters'], 
                                                                                 dict_dp, spots1, Bkey,
                                                                                 strain_free_parameters)
                else:
                    strain_crystal, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
                    rot_mat_UB = np.copy(rotation_matrix[igrain][0][cnt-1,:,:])
                spots = spots_left
                expected = theo_spots_left
                max_mr, min_mr = 100*(len(spots_left)/theo_spots_left), 100*(len(spots_left)/theo_spots_left)
                first_match = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                0, len(spots), expected, max_mr, 0, rot_mat_UB]  
                    
            elif left_row and not last_row:
                try_previous = True
                current_spots = [len(list(set(spots_lr) & set(spots1_global[igr]))) > coeff_overlap*len(spots1_global[igr]) for igr in range(len(spots1_global))]
                if np.any(current_spots):
                    try_previous = False
                    continue
                
                mat = mat_global[igrain][0][cnt-lim_y]
                if mat == 1:
                    Keymaterial_ = material_
                    case = material_
                    Bkey = B0
                    input_params["mat"] = 1
                    input_params["Bmat"] = Bkey
                elif mat == 2:
                    Keymaterial_ = material1_
                    case = material1_
                    Bkey = B1
                    input_params["mat"] = 2
                    input_params["Bmat"] = Bkey
                else:
                    Keymaterial_ = None
                    Bkey = None
                    input_params["mat"] = 0
                    input_params["Bmat"] = None
                    continue
                
                if strain_calculation:
                    strain_crystal, strain_sample, iR, fR, rot_mat_UB = calculate_strains_fromUB(s_tth1, s_chi1, 
                                                                                  rotation_matrix[igrain][0][cnt-lim_y,:,:], 
                                                                                  Keymaterial_, 
                                                                                 input_params, dict_dp['detectorparameters'], 
                                                                                 dict_dp, spots1, Bkey,
                                                                                 strain_free_parameters)
                else:
                    strain_crystal, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
                    rot_mat_UB = np.copy(rotation_matrix[igrain][0][cnt-lim_y,:,:])
                    
                spots = spots_lr
                expected = theo_spots_lr    
                max_mr, min_mr = 100*(len(spots_lr)/theo_spots_lr), 100*(len(spots_lr)/theo_spots_lr)
                first_match = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                0, len(spots), expected, max_mr, 0, rot_mat_UB]  

        else:
            try_previous = False
        
        if not try_previous and not calcul_done:
            ### old version
            if mode_spotCycle == "slow":
                # print("Slow mode of analysis")
                first_match, max_mr, min_mr, spots, \
                        case, mat, strain_crystal, \
                            strain_sample, iR, fR  = get_orient_mat(s_tth1, s_chi1,
                                                                        material_, material1_, classhkl,
                                                                        class_predicted1, predicted_hkl1,
                                                                        input_params, hkl_all_class0, hkl_all_class1,
                                                                        max_pred1, dict_dp, 
                                                                        spots1, dist, 
                                                                        Gstar_metric0, Gstar_metric1, B0, B1,
                                                                        softmax_threshold=softmax_threshold_global123,
                                                                        mr_threshold=mr_threshold_global123,
                                                                        tab_distance_classhkl_data0=tab_distance_classhkl_data0,
                                                                        tab_distance_classhkl_data1=tab_distance_classhkl_data1,
                                                                        spots1_global = spots1_global,
                                                                        coeff_overlap = coeff_overlap,
                                                                        ind_mat=ind_mat, ind_mat1=ind_mat1, 
                                                                        strain_calculation=strain_calculation,
                                                                        cap_matchrate123=cap_matchrate123,
                                                                        material0_count=material0_count,
                                                                        material1_count=material1_count,
                                                                        material0_limit=material0_limit,
                                                                        material1_limit=material1_limit,
                                                                        igrain=igrain,
                                                                        material_phase_always_present=material_phase_always_present,
                                                                        strain_free_parameters=strain_free_parameters)
            elif mode_spotCycle == "houghmode":
                # print("Slow mode of analysis")
                first_match, max_mr, min_mr, spots, \
                        case, mat, strain_crystal, \
                            strain_sample, iR, fR  = get_orient_mat_HM(s_tth1, s_chi1,
                                                                        material_, material1_, classhkl,
                                                                        class_predicted1, predicted_hkl1,
                                                                        input_params, hkl_all_class0, hkl_all_class1,
                                                                        max_pred1, dict_dp, 
                                                                        spots1, dist, 
                                                                        Gstar_metric0, Gstar_metric1, B0, B1,
                                                                        softmax_threshold=softmax_threshold_global123,
                                                                        mr_threshold=mr_threshold_global123,
                                                                        tab_distance_classhkl_data0=tab_distance_classhkl_data0,
                                                                        tab_distance_classhkl_data1=tab_distance_classhkl_data1,
                                                                        spots1_global = spots1_global,
                                                                        coeff_overlap = coeff_overlap,
                                                                        ind_mat=ind_mat, ind_mat1=ind_mat1, 
                                                                        strain_calculation=strain_calculation,
                                                                        cap_matchrate123=cap_matchrate123,
                                                                        material0_count=material0_count,
                                                                        material1_count=material1_count,
                                                                        material0_limit=material0_limit,
                                                                        material1_limit=material1_limit,
                                                                        igrain=igrain,
                                                                        material_phase_always_present=material_phase_always_present,
                                                                        strain_free_parameters=strain_free_parameters)
            elif mode_spotCycle == "houghgraphmode":
                # print("Fast mode of analysis")
                first_match, max_mr, min_mr, spots, \
                        case, mat, strain_crystal, \
                            strain_sample, iR, fR,\
                            objective_function1  = get_orient_mat_graphv1HM(s_tth1, s_chi1,
                                                                        material_, material1_, classhkl,
                                                                        class_predicted1, predicted_hkl1,
                                                                        input_params, hkl_all_class0, hkl_all_class1,
                                                                        max_pred1, dict_dp, 
                                                                        spots1, dist, 
                                                                        Gstar_metric0, Gstar_metric1, B0, B1,
                                                                        softmax_threshold=softmax_threshold_global123,
                                                                        mr_threshold=mr_threshold_global123,
                                                                        tab_distance_classhkl_data0=tab_distance_classhkl_data0,
                                                                        tab_distance_classhkl_data1=tab_distance_classhkl_data1,
                                                                        spots1_global = spots1_global,
                                                                        coeff_overlap = coeff_overlap,
                                                                        ind_mat=ind_mat, ind_mat1=ind_mat1, 
                                                                        strain_calculation=strain_calculation,
                                                                        cap_matchrate123=cap_matchrate123,
                                                                        material0_count=material0_count,
                                                                        material1_count=material1_count,
                                                                        material0_limit=material0_limit,
                                                                        material1_limit=material1_limit,
                                                                        igrain=igrain,
                                                                        material_phase_always_present=material_phase_always_present,
                                                                        objective_function= objective_function1,
                                                                        crystal=crystal,
                                                                        crystal1=crystal1,
                                                                        strain_free_parameters=strain_free_parameters)

            elif mode_spotCycle == "graphmode":
                # print("Fast mode of analysis")
                first_match, max_mr, min_mr, spots, \
                        case, mat, strain_crystal, \
                            strain_sample, iR, fR,\
                            objective_function1  = get_orient_mat_graphv1(s_tth1, s_chi1,
                                                                        material_, material1_, classhkl,
                                                                        class_predicted1, predicted_hkl1,
                                                                        input_params, hkl_all_class0, hkl_all_class1,
                                                                        max_pred1, dict_dp, 
                                                                        spots1, dist, 
                                                                        Gstar_metric0, Gstar_metric1, B0, B1,
                                                                        softmax_threshold=softmax_threshold_global123,
                                                                        mr_threshold=mr_threshold_global123,
                                                                        tab_distance_classhkl_data0=tab_distance_classhkl_data0,
                                                                        tab_distance_classhkl_data1=tab_distance_classhkl_data1,
                                                                        spots1_global = spots1_global,
                                                                        coeff_overlap = coeff_overlap,
                                                                        ind_mat=ind_mat, ind_mat1=ind_mat1, 
                                                                        strain_calculation=strain_calculation,
                                                                        cap_matchrate123=cap_matchrate123,
                                                                        material0_count=material0_count,
                                                                        material1_count=material1_count,
                                                                        material0_limit=material0_limit,
                                                                        material1_limit=material1_limit,
                                                                        igrain=igrain,
                                                                        material_phase_always_present=material_phase_always_present,
                                                                        objective_function= objective_function1,
                                                                        crystal=crystal,
                                                                        crystal1=crystal1,
                                                                        strain_free_parameters=strain_free_parameters)
            elif mode_spotCycle == "update_reupdate":
                # print("Fast mode of analysis")
                first_match, max_mr, min_mr, spots, \
                    case, mat, strain_crystal, \
                    strain_sample, iR, fR, objective_function1,\
                    s_tth1, s_chi1, class_predicted1, \
                    predicted_hkl1, max_pred1, dist = get_orient_mat_repredict(s_tth1, s_chi1,
                                                                        material_, material1_, classhkl,
                                                                        class_predicted1, predicted_hkl1,
                                                                        input_params, hkl_all_class0, hkl_all_class1,
                                                                        max_pred1, dict_dp, 
                                                                        spots1, dist, 
                                                                        Gstar_metric0, Gstar_metric1, B0, B1,
                                                                        softmax_threshold=softmax_threshold_global123,
                                                                        mr_threshold=mr_threshold_global123,
                                                                        tab_distance_classhkl_data0=tab_distance_classhkl_data0,
                                                                        tab_distance_classhkl_data1=tab_distance_classhkl_data1,
                                                                        spots1_global = spots1_global,
                                                                        coeff_overlap = coeff_overlap,
                                                                        ind_mat=ind_mat, ind_mat1=ind_mat1, 
                                                                        strain_calculation=strain_calculation,
                                                                        cap_matchrate123=cap_matchrate123,
                                                                        material0_count=material0_count,
                                                                        material1_count=material1_count,
                                                                        material0_limit=material0_limit,
                                                                        material1_limit=material1_limit,
                                                                        igrain=igrain,
                                                                        material_phase_always_present=material_phase_always_present,
                                                                        objective_function= objective_function1,
                                                                        crystal=crystal,
                                                                        crystal1=crystal1,
                                                                        angbins=angbins,
                                                                        wb=wb, temp_key=temp_key,
                                                                        strain_free_parameters=strain_free_parameters)
            else:
                print("selected mode of treating spots is not ready")
                
        for ispot in spots:
            spots1.append(ispot)
            spots1_global[igrain].append(ispot)

        ## make copy of best rotation matrix
        best_match[igrain].append(np.copy(first_match))
        best_matrix[igrain].append(np.copy(first_match[14]))
        mr_highest[igrain].append(np.copy(max_mr))
        mat_highest[igrain].append(np.copy(mat))
        ir_pixels[igrain].append(np.copy(iR))
        fr_pixels[igrain].append(np.copy(fR))
        spots_len[igrain].append(np.copy(len(spots)))
        strain_matrix[igrain].append(np.copy(strain_crystal))
        strain_matrixs[igrain].append(np.copy(strain_sample))
        
        if np.all(first_match[14] != 0):
            check[igrain] = 1
        
        if mat == 1:
            material0_count += 1
        if mat == 2:
            material1_count += 1

    return best_matrix, mr_highest, mat_highest, strain_matrix, strain_matrixs, ir_pixels, fr_pixels, spots_len, best_match, check


def get_material_dataP(Gstar, classhkl = None):
    hkl2 = np.copy(classhkl)
    hkl1 = np.copy(classhkl)
    # compute square matrix containing angles
    metrics = Gstar
    H1 = hkl1
    n1 = hkl1.shape[0]
    H2 = hkl2
    n2 = hkl2.shape[0]
    dstar_square_1 = np.diag(np.inner(np.inner(H1, metrics), H1))
    dstar_square_2 = np.diag(np.inner(np.inner(H2, metrics), H2))
    scalar_product = np.inner(np.inner(H1, metrics), H2) * 1.0
    d1 = np.sqrt(dstar_square_1.reshape((n1, 1))) * 1.0
    d2 = np.sqrt(dstar_square_2.reshape((n2, 1))) * 1.0
    outy = np.outer(d1, d2)
    ratio = scalar_product / outy
    ratio = np.round(ratio, decimals=7)
    tab_angulardist = np.arccos(ratio) / (np.pi / 180.0)
    np.putmask(tab_angulardist, np.abs(tab_angulardist) < 0.001, 400)
    return tab_angulardist

def get_orient_mat_repredict(s_tth, s_chi, material0_, material1_, classhkl, class_predicted, predicted_hkl,
                       input_params, hkl_all_class0, hkl_all_class1, max_pred, dict_dp, spots, 
                       dist, Gstar_metric0, Gstar_metric1, B0, B1, softmax_threshold=0.85, mr_threshold=0.85, 
                       tab_distance_classhkl_data0=None, tab_distance_classhkl_data1=None, spots1_global=None,
                       coeff_overlap = None, ind_mat=None, ind_mat1=None, strain_calculation=None, cap_matchrate123=None,
                       material0_count=None, material1_count=None, material0_limit=None, material1_limit=None,
                       igrain=None, material_phase_always_present=None, objective_function=None, crystal=None,
                       crystal1=None, angbins=None, wb=None, temp_key=None, strain_free_parameters=None):    
    if objective_function == None:
        call_global()
        
        init_mr = 0
        init_mat = 0
        init_material = "None"
        init_case = "None"
        init_B = None
        final_match_rate = 0
        match_rate_mma = []
        final_rmv_ind = []

        if material0_ == material1_:
            list_of_sets = []
            for ii in range(0, min(nb_spots_consider, len(dist))):
                if max_pred[ii] < softmax_threshold:
                    continue 
                a1 = np.round(dist[ii],3)
                
                for i in range(0, min(nb_spots_consider, len(dist))):
                    if ii==i:
                        continue
                    if (ii,i) in list_of_sets or (i,ii) in list_of_sets:
                        continue
                    
                    if max_pred[i] < softmax_threshold:
                        continue
                    hkl1 = hkl_all_class0[str(predicted_hkl[ii])]
                    hkl1_list = np.array(hkl1)
                    hkl2 = hkl_all_class0[str(predicted_hkl[i])]
                    hkl2_list = np.array(hkl2)
                    Gstar_metric = Gstar_metric0
                    
                    tab_angulardist_temp = CP.AngleBetweenNormals(hkl1_list, hkl2_list, Gstar_metric)
                    np.putmask(tab_angulardist_temp, np.abs(tab_angulardist_temp) < 0.001, 400)
                    
                    list_ = np.where(np.abs(tab_angulardist_temp-a1[i]) < input_params["tolerance"])
                    if len(list_[0]) != 0:
                        list_of_sets.append((ii,i))

        else:
            list_of_sets = []
            for ii in range(0, min(nb_spots_consider, len(dist))):
                if max_pred[ii] < softmax_threshold:
                    continue 
                
                a1 = np.round(dist[ii],3)

                for i in range(0, min(nb_spots_consider, len(dist))):
                    if ii==i:
                        continue
                    if (ii,i) in list_of_sets or (i,ii) in list_of_sets:
                        continue
                    
                    if max_pred[i] < softmax_threshold:
                        continue
                    if class_predicted[ii] < ind_mat and class_predicted[i] < ind_mat:
                        tab_distance_classhkl_data = tab_distance_classhkl_data0
                        tolerance_new = input_params["tolerance"]
                        hkl1 = hkl_all_class0[str(predicted_hkl[ii])]
                        hkl1_list = np.array(hkl1)
                        hkl2 = hkl_all_class0[str(predicted_hkl[i])]
                        hkl2_list = np.array(hkl2)
                        Gstar_metric = Gstar_metric0
                        
                    elif (ind_mat <= class_predicted[ii] < (ind_mat+ind_mat1)) and \
                                        (ind_mat <= class_predicted[i] < (ind_mat+ind_mat1)):
                        tab_distance_classhkl_data = tab_distance_classhkl_data1
                        tolerance_new = input_params["tolerance1"]
                        hkl1 = hkl_all_class1[str(predicted_hkl[ii])]
                        hkl1_list = np.array(hkl1)
                        hkl2 = hkl_all_class1[str(predicted_hkl[i])]
                        hkl2_list = np.array(hkl2)
                        Gstar_metric = Gstar_metric1
                        
                    else:
                        continue
                    
                    tab_angulardist_temp = CP.AngleBetweenNormals(hkl1_list, hkl2_list, Gstar_metric)
                    np.putmask(tab_angulardist_temp, np.abs(tab_angulardist_temp) < 0.001, 400)
                    list_ = np.where(np.abs(tab_angulardist_temp-a1[i]) < tolerance_new)
                    if len(list_[0]) != 0:
                        list_of_sets.append((ii,i))

        ## build a direct connection graph object
        graph_obj = nx.DiGraph(list_of_sets)
        connected_nodes_length = []
        connected_nodes = [[] for i in range(len(graph_obj))]
        for i,line in enumerate(nx.generate_adjlist(graph_obj)):
            connected_nodes_length.append(len(line.split(" ")))
            connected_nodes[i].append([int(jj) for jj in line.split(" ")])
        
        ## sort by maximum node occurance
        connected_nodes_length = np.array(connected_nodes_length)
        connected_nodes_length_sort_ind = np.argsort(connected_nodes_length)[::-1]
  
        mat = 0
        case = "None"
        tried_spots = []
        
        objective_function = []
        for toplist in range(len(graph_obj)):
            # ## continue if less than 3 connections are found for a graph
            # if connected_nodes_length[connected_nodes_length_sort_ind[toplist]] < 2:
            #     continue
            
            for j in connected_nodes[connected_nodes_length_sort_ind[toplist]][0]:
                init_mr = 0
                final_match_rate = 0
                final_rmv_ind = []
                all_stats = []
                for i in connected_nodes[connected_nodes_length_sort_ind[toplist]][0]:
                    if j == i:
                        continue
                    
                    if j in tried_spots and i in tried_spots:
                        continue
                    
                    if material0_ == material1_:
                        tab_distance_classhkl_data = tab_distance_classhkl_data0
                        hkl_all_class = hkl_all_class0
                        material_ = material0_
                        B = B0
                        Gstar_metric = Gstar_metric0
                        case = material_
                        mat = 1
                        input_params["mat"] = mat
                        input_params["Bmat"] = B
                    else:
                        if class_predicted[i] < ind_mat and class_predicted[j] < ind_mat:
                            tab_distance_classhkl_data = tab_distance_classhkl_data0
                            hkl_all_class = hkl_all_class0
                            material_ = material0_
                            B = B0
                            Gstar_metric = Gstar_metric0
                            case = material_
                            mat = 1
                            input_params["mat"] = mat
                            input_params["Bmat"] = B
                        elif (ind_mat <= class_predicted[i] < (ind_mat+ind_mat1)) and \
                                            (ind_mat <= class_predicted[j] < (ind_mat+ind_mat1)):
                            tab_distance_classhkl_data = tab_distance_classhkl_data1
                            hkl_all_class = hkl_all_class1
                            material_ = material1_
                            B = B1
                            Gstar_metric = Gstar_metric1
                            case = material_  
                            mat = 2
                            input_params["mat"] = mat
                            input_params["Bmat"] = B
                        else:
                            mat = 0
                            case = "None"
                            input_params["mat"] = mat
                            input_params["Bmat"] = None
                    
                    if mat == 0:
                        continue                    
                    
                    tth_chi_spot1 = np.array([s_tth[i], s_chi[i]])
                    tth_chi_spot2 = np.array([s_tth[j], s_chi[j]])         
        
                    hkl1 = hkl_all_class[str(predicted_hkl[i])]
                    hkl1_list = np.array(hkl1)
                    hkl2 = hkl_all_class[str(predicted_hkl[j])]
                    hkl2_list = np.array(hkl2)
                    actual_mat, flagAM, \
                    spot1_hkl, spot2_hkl = propose_UB_matrix(hkl1_list, hkl2_list, 
                                                            Gstar_metric, input_params, 
                                                            dist[i,j],
                                                            tth_chi_spot1, tth_chi_spot2, 
                                                            B, method=0, crystal=crystal,
                                                            crystal1=crystal1)
                    
                    if flagAM:
                        continue
                    
                    for iind in range(len(actual_mat)):
                        rot_mat123 = actual_mat[iind]

                        rmv_ind, theospots = remove_spots(s_tth, s_chi, rot_mat123, 
                                                                material_, input_params, 
                                                                dict_dp['detectorparameters'], dict_dp)
                        
                        match_rate = np.round(100 * len(rmv_ind)/theospots, 3)
                        
                        match_rate_mma.append(match_rate)

                        if match_rate > init_mr:
                            final_rmv_ind = rmv_ind                    
                            init_mat = np.copy(mat)
                            input_params["mat"] = init_mat
                            init_material = np.copy(material_)
                            init_case = np.copy(case)
                            init_B = np.copy(B)  
                            input_params["Bmat"] = init_B                                     
                            final_match_rate = np.copy(match_rate)
                            init_mr = np.copy(match_rate)                   
                            all_stats = [i, j, \
                                         spot1_hkl[iind], spot2_hkl[iind], \
                                        tth_chi_spot1, tth_chi_spot2, \
                                        dist[i,j], tab_distance_classhkl_data[i,j], np.round(max_pred[i]*100,3), \
                                        np.round(max_pred[j]*100,3), len(rmv_ind), theospots,\
                                        match_rate, 0.0, rot_mat123, init_mat, init_material, init_B, init_case]
                    tried_spots.append(i)                 
                    
                if (final_match_rate <= cap_matchrate123): ## Nothing found!! 
                    ## Either peaks are not well defined or not found within tolerance and prediction accuracy
                    all_stats = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                        0, 0, 0, 0, 0, np.zeros((3,3))]
                    max_mr, min_mr = 0, 0
                    spot_ind = []
                    mat = 0
                    input_params["mat"] = 0
                    case = "None"
                    objective_function.append([0, [], []])
                else:
                    objective_function.append([final_match_rate, final_rmv_ind, all_stats])     
                tried_spots.append(j)

    sort_ind = []
    for ijk in objective_function:
        sort_ind.append(ijk[0])
    sort_ind = np.array(sort_ind)
    sort_ind = np.argsort(sort_ind)[::-1]
    
    for gr_count123 in range(len(sort_ind)):           
        max_mr = objective_function[sort_ind[gr_count123]][0]
        rmv_ind = objective_function[sort_ind[gr_count123]][1]
        all_stats = objective_function[sort_ind[gr_count123]][2]
        
        if len(rmv_ind) == 0 or max_mr==0:
            continue
        
        mat = all_stats[15]
        if mat == 1:
            if igrain==0 and material_phase_always_present ==2:
                mat = 0
                case="None"
            if material0_count >= material0_limit:
                mat = 0
                case="None"
        elif mat == 2:
            if igrain==0 and material_phase_always_present ==1:
                mat = 0
                case="None"
            if material1_count >= material1_limit:
                mat = 0
                case="None"
        
        if mat == 0:
            continue

        current_spots = [len(list(set(rmv_ind) & set(spots1_global[igr])))> coeff_overlap*len(spots1_global[igr]) for igr in range(len(spots1_global))]
        
        if np.any(current_spots):
            continue
                  
        input_params["mat"] = all_stats[15]
        if strain_calculation:
            dev_strain, strain_sample, iR, fR, rot_mat_UB = calculate_strains_fromUB(s_tth, s_chi, all_stats[14], str(all_stats[16]), 
                                                                 input_params, dict_dp['detectorparameters'], 
                                                                 dict_dp, spots, all_stats[17],
                                                                 strain_free_parameters)
        else:
            dev_strain, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
            rot_mat_UB = np.copy(all_stats[14])
        all_stats[14] = rot_mat_UB     
        
        ## delete the indexed spots and repredict the spots HKL in the absence of indexed spots
        ## maybe it makes it easier to detect some grains
        ##update list
        # s_tth = np.delete(s_tth, rmv_ind, axis=0)
        # s_chi = np.delete(s_chi, rmv_ind, axis=0)
        s_tth[rmv_ind] = np.nan
        s_chi[rmv_ind] = np.nan
        
        sorted_data = np.transpose(np.array([s_tth/2., s_chi]))
        tabledistancerandom = np.transpose(GT.calculdist_from_thetachi(sorted_data, sorted_data))
        
        spots_in_center = np.arange(0,len(s_tth))
        spots_in_center = spots_in_center[:nb_spots_consider]
        codebars_all = []
        for i in spots_in_center:
            spotangles = tabledistancerandom[i]
            spotangles = np.delete(spotangles, i)# removing the self distance
            codebars = np.histogram(spotangles, bins=angbins)[0]
            # codebars = histogram1d(spotangles, range=[min(angbins),max(angbins)], bins=len(angbins)-1)
            ## normalize the same way as training data
            max_codebars = np.max(codebars)
            codebars = codebars/ max_codebars
            codebars_all.append(codebars)
        ## reshape for the model to predict all spots at once
        codebars = np.array(codebars_all)
        ## Do prediction of all spots at once
        prediction = predict(codebars, wb, temp_key)
        max_pred = np.max(prediction, axis = 1)
        class_predicted = np.argmax(prediction, axis = 1)
        predicted_hkl123 = classhkl[class_predicted]
        predicted_hkl123 = predicted_hkl123.astype(int)

        
        return all_stats, np.max(max_mr), np.min(max_mr), \
                rmv_ind, str(all_stats[18]), all_stats[15], dev_strain, strain_sample, iR, fR, objective_function,\
                    s_tth, s_chi, class_predicted, predicted_hkl123, max_pred, tabledistancerandom
    
    all_stats = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                        0, 0, 0, 0, 0, np.zeros((3,3))]
    max_mr, min_mr = 0, 0
    spot_ind = []
    mat = 0
    input_params["mat"] = 0
    case = "None"
    return all_stats, max_mr, min_mr, spot_ind, case, mat, np.zeros((3,3)), np.zeros((3,3)), 0, 0, objective_function,\
            s_tth, s_chi, class_predicted, predicted_hkl, max_pred, dist


def get_orient_mat_graphv1HM(s_tth, s_chi, material0_, material1_, classhkl, class_predicted, predicted_hkl,
                       input_params, hkl_all_class0, hkl_all_class1, max_pred, dict_dp, spots, 
                       dist, Gstar_metric0, Gstar_metric1, B0, B1, softmax_threshold=0.85, mr_threshold=0.85, 
                       tab_distance_classhkl_data0=None, tab_distance_classhkl_data1=None, spots1_global=None,
                       coeff_overlap = None, ind_mat=None, ind_mat1=None, strain_calculation=None, cap_matchrate123=None,
                       material0_count=None, material1_count=None, material0_limit=None, material1_limit=None,
                       igrain=None, material_phase_always_present=None, objective_function=None, crystal=None,
                       crystal1=None, strain_free_parameters=None):
    
    if objective_function == None:
        call_global()
        
        init_mr = 0
        init_mat = 0
        init_material = "None"
        init_case = "None"
        init_B = None
        final_match_rate = 0
        match_rate_mma = []
        final_rmv_ind = []
        
        #calculate the gnemonic projection space
        imageGNO, nbpeaks, halfdiagonal = computeGnomonicImage(s_tth, s_chi)
        hough, theta_h, d_h = hough_line(imageGNO)
        
        if material0_ == material1_:
            list_of_sets = []
            for ii in range(0, min(nb_spots_consider, len(dist))):
                if max_pred[ii] < softmax_threshold:
                    continue 
                a1 = np.round(dist[ii],3)
                
                for i in range(0, min(nb_spots_consider, len(dist))):
                    if ii==i:
                        continue
                    if (ii,i) in list_of_sets or (i,ii) in list_of_sets:
                        continue
                    
                    if max_pred[i] < softmax_threshold:
                        continue
                    hkl1 = hkl_all_class0[str(predicted_hkl[ii])]
                    hkl1_list = np.array(hkl1)
                    hkl2 = hkl_all_class0[str(predicted_hkl[i])]
                    hkl2_list = np.array(hkl2)
                    Gstar_metric = Gstar_metric0
                    
                    tab_angulardist_temp = CP.AngleBetweenNormals(hkl1_list, hkl2_list, Gstar_metric)
                    np.putmask(tab_angulardist_temp, np.abs(tab_angulardist_temp) < 0.001, 400)
                    
                    list_ = np.where(np.abs(tab_angulardist_temp-a1[i]) < input_params["tolerance"])
                    if len(list_[0]) != 0:
                        list_of_sets.append((ii,i))

        else:
            list_of_sets = []
            for ii in range(0, min(nb_spots_consider, len(dist))):
                if max_pred[ii] < softmax_threshold:
                    continue 
                
                a1 = np.round(dist[ii],3)

                for i in range(0, min(nb_spots_consider, len(dist))):
                    if ii==i:
                        continue
                    if (ii,i) in list_of_sets or (i,ii) in list_of_sets:
                        continue
                    
                    if max_pred[i] < softmax_threshold:
                        continue
                    if class_predicted[ii] < ind_mat and class_predicted[i] < ind_mat:
                        tab_distance_classhkl_data = tab_distance_classhkl_data0
                        tolerance_new = input_params["tolerance"]
                        hkl1 = hkl_all_class0[str(predicted_hkl[ii])]
                        hkl1_list = np.array(hkl1)
                        hkl2 = hkl_all_class0[str(predicted_hkl[i])]
                        hkl2_list = np.array(hkl2)
                        Gstar_metric = Gstar_metric0
                        
                    elif (ind_mat <= class_predicted[ii] < (ind_mat+ind_mat1)) and \
                                        (ind_mat <= class_predicted[i] < (ind_mat+ind_mat1)):
                        tab_distance_classhkl_data = tab_distance_classhkl_data1
                        tolerance_new = input_params["tolerance1"]
                        hkl1 = hkl_all_class1[str(predicted_hkl[ii])]
                        hkl1_list = np.array(hkl1)
                        hkl2 = hkl_all_class1[str(predicted_hkl[i])]
                        hkl2_list = np.array(hkl2)
                        Gstar_metric = Gstar_metric1
                        
                    else:
                        continue
                    
                    tab_angulardist_temp = CP.AngleBetweenNormals(hkl1_list, hkl2_list, Gstar_metric)
                    np.putmask(tab_angulardist_temp, np.abs(tab_angulardist_temp) < 0.001, 400)
                    list_ = np.where(np.abs(tab_angulardist_temp-a1[i]) < tolerance_new)
                    if len(list_[0]) != 0:
                        list_of_sets.append((ii,i))

        ## build a direct connection graph object
        graph_obj = nx.DiGraph(list_of_sets)
        connected_nodes_length = []
        connected_nodes = [[] for i in range(len(graph_obj))]
        for i,line in enumerate(nx.generate_adjlist(graph_obj)):
            connected_nodes_length.append(len(line.split(" ")))
            connected_nodes[i].append([int(jj) for jj in line.split(" ")])
        
        ## sort by maximum node occurance
        connected_nodes_length = np.array(connected_nodes_length)
        connected_nodes_length_sort_ind = np.argsort(connected_nodes_length)[::-1]
  
        mat = 0
        case = "None"
        tried_spots = []
        
        objective_function = []
        for toplist in range(len(graph_obj)):
            # ## continue if less than 3 connections are found for a graph
            # if connected_nodes_length[connected_nodes_length_sort_ind[toplist]] < 2:
            #     continue
            
            for j in connected_nodes[connected_nodes_length_sort_ind[toplist]][0]:
                init_mr = 0
                final_match_rate = 0
                final_rmv_ind = []
                all_stats = []
                for i in connected_nodes[connected_nodes_length_sort_ind[toplist]][0]:
                    if j == i:
                        continue
                    
                    if j in tried_spots and i in tried_spots:
                        continue
                    
                    ## condition to check if spots lie on the same line
                    in_hough_line = False
                    for _, anglehs, disths in zip(*hough_line_peaks(hough, theta_h, d_h)):
                        y0 = (disths - 0 * np.cos(anglehs)) / np.sin(anglehs)
                        y1 = (disths - imageGNO.shape[1] * np.cos(anglehs)) / np.sin(anglehs)
                        p1 = np.array((0,y0))
                        p2 = np.array((imageGNO.shape[1], y1))
                        
                        p3_0 = ComputeGnomon_singledata(s_tth[i], s_chi[i])
                        p3_1 = ComputeGnomon_singledata(s_tth[j], s_chi[j])
                
                        distance_0 = np.abs(np.cross(p2-p1, p3_0-p1)) / np.linalg.norm(p2-p1)
                        distance_1 = np.abs(np.cross(p2-p1, p3_1-p1)) / np.linalg.norm(p2-p1)
                        
                        if distance_0 < dist_threshold and distance_1 < dist_threshold:
                            # print(distance_0, distance_1)
                            in_hough_line = True
                            
                        if in_hough_line:
                            break
                
                    if not in_hough_line:
                        continue
                    
                    if material0_ == material1_:
                        tab_distance_classhkl_data = tab_distance_classhkl_data0
                        hkl_all_class = hkl_all_class0
                        material_ = material0_
                        B = B0
                        Gstar_metric = Gstar_metric0
                        case = material_
                        mat = 1
                        input_params["mat"] = mat
                        input_params["Bmat"] = B
                    else:
                        if class_predicted[i] < ind_mat and class_predicted[j] < ind_mat:
                            tab_distance_classhkl_data = tab_distance_classhkl_data0
                            hkl_all_class = hkl_all_class0
                            material_ = material0_
                            B = B0
                            Gstar_metric = Gstar_metric0
                            case = material_
                            mat = 1
                            input_params["mat"] = mat
                            input_params["Bmat"] = B
                        elif (ind_mat <= class_predicted[i] < (ind_mat+ind_mat1)) and \
                                            (ind_mat <= class_predicted[j] < (ind_mat+ind_mat1)):
                            tab_distance_classhkl_data = tab_distance_classhkl_data1
                            hkl_all_class = hkl_all_class1
                            material_ = material1_
                            B = B1
                            Gstar_metric = Gstar_metric1
                            case = material_  
                            mat = 2
                            input_params["mat"] = mat
                            input_params["Bmat"] = B
                        else:
                            mat = 0
                            case = "None"
                            input_params["mat"] = mat
                            input_params["Bmat"] = None
                    
                    if mat == 0:
                        continue                    
                    
                    tth_chi_spot1 = np.array([s_tth[i], s_chi[i]])
                    tth_chi_spot2 = np.array([s_tth[j], s_chi[j]])         
        
                    hkl1 = hkl_all_class[str(predicted_hkl[i])]
                    hkl1_list = np.array(hkl1)
                    hkl2 = hkl_all_class[str(predicted_hkl[j])]
                    hkl2_list = np.array(hkl2)
                    actual_mat, flagAM, \
                    spot1_hkl, spot2_hkl = propose_UB_matrix(hkl1_list, hkl2_list, 
                                                            Gstar_metric, input_params, 
                                                            dist[i,j],
                                                            tth_chi_spot1, tth_chi_spot2, 
                                                            B, method=0, crystal=crystal,
                                                            crystal1=crystal1)
                    
                    if flagAM:
                        continue
                    
                    for iind in range(len(actual_mat)):
                        rot_mat123 = actual_mat[iind]

                        rmv_ind, theospots = remove_spots(s_tth, s_chi, rot_mat123, 
                                                                material_, input_params, 
                                                                dict_dp['detectorparameters'], dict_dp)
                        
                        match_rate = np.round(100 * len(rmv_ind)/theospots, 3)
                        
                        match_rate_mma.append(match_rate)

                        if match_rate > init_mr:
                            final_rmv_ind = rmv_ind                    
                            init_mat = np.copy(mat)
                            input_params["mat"] = init_mat
                            init_material = np.copy(material_)
                            init_case = np.copy(case)
                            init_B = np.copy(B)  
                            input_params["Bmat"] = init_B                                     
                            final_match_rate = np.copy(match_rate)
                            init_mr = np.copy(match_rate)                   
                            all_stats = [i, j, \
                                         spot1_hkl[iind], spot2_hkl[iind], \
                                        tth_chi_spot1, tth_chi_spot2, \
                                        dist[i,j], tab_distance_classhkl_data[i,j], np.round(max_pred[i]*100,3), \
                                        np.round(max_pred[j]*100,3), len(rmv_ind), theospots,\
                                        match_rate, 0.0, rot_mat123, init_mat, init_material, init_B, init_case]
                    tried_spots.append(i)                 
                    
                if (final_match_rate <= cap_matchrate123): ## Nothing found!! 
                    ## Either peaks are not well defined or not found within tolerance and prediction accuracy
                    all_stats = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                        0, 0, 0, 0, 0, np.zeros((3,3))]
                    max_mr, min_mr = 0, 0
                    spot_ind = []
                    mat = 0
                    input_params["mat"] = 0
                    case = "None"
                    objective_function.append([0, [], []])
                else:
                    objective_function.append([final_match_rate, final_rmv_ind, all_stats])     
                tried_spots.append(j)
 
    sort_ind = []
    for ijk in objective_function:
        sort_ind.append(ijk[0])
    sort_ind = np.array(sort_ind)
    sort_ind = np.argsort(sort_ind)[::-1]
    
    for gr_count123 in range(len(sort_ind)):           
        max_mr = objective_function[sort_ind[gr_count123]][0]
        rmv_ind = objective_function[sort_ind[gr_count123]][1]
        all_stats = objective_function[sort_ind[gr_count123]][2]
        
        if len(rmv_ind) == 0 or max_mr==0:
            continue
        
        mat = all_stats[15]
        if mat == 1:
            if igrain==0 and material_phase_always_present ==2:
                mat = 0
                case="None"
            if material0_count >= material0_limit:
                mat = 0
                case="None"
        elif mat == 2:
            if igrain==0 and material_phase_always_present ==1:
                mat = 0
                case="None"
            if material1_count >= material1_limit:
                mat = 0
                case="None"
        
        if mat == 0:
            continue

        current_spots = [len(list(set(rmv_ind) & set(spots1_global[igr])))> coeff_overlap*len(spots1_global[igr]) for igr in range(len(spots1_global))]
        
        if np.any(current_spots):
            continue
                  
        input_params["mat"] = all_stats[15]
        if strain_calculation:
            dev_strain, strain_sample, iR, fR, rot_mat_UB = calculate_strains_fromUB(s_tth, s_chi, all_stats[14], str(all_stats[16]), 
                                                                 input_params, dict_dp['detectorparameters'], 
                                                                 dict_dp, spots, all_stats[17],
                                                                 strain_free_parameters)
        else:
            dev_strain, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
            rot_mat_UB = np.copy(all_stats[14])
        all_stats[14] = rot_mat_UB     
        
        return all_stats, np.max(max_mr), np.min(max_mr), \
                rmv_ind, str(all_stats[18]), all_stats[15], dev_strain, strain_sample, iR, fR, objective_function
    
    all_stats = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                        0, 0, 0, 0, 0, np.zeros((3,3))]
    max_mr, min_mr = 0, 0
    spot_ind = []
    mat = 0
    input_params["mat"] = 0
    case = "None"
    return all_stats, max_mr, min_mr, spot_ind, case, mat, np.zeros((3,3)), np.zeros((3,3)), 0, 0, objective_function

def get_orient_mat_graphv1(s_tth, s_chi, material0_, material1_, classhkl, class_predicted, predicted_hkl,
                       input_params, hkl_all_class0, hkl_all_class1, max_pred, dict_dp, spots, 
                       dist, Gstar_metric0, Gstar_metric1, B0, B1, softmax_threshold=0.85, mr_threshold=0.85, 
                       tab_distance_classhkl_data0=None, tab_distance_classhkl_data1=None, spots1_global=None,
                       coeff_overlap = None, ind_mat=None, ind_mat1=None, strain_calculation=None, cap_matchrate123=None,
                       material0_count=None, material1_count=None, material0_limit=None, material1_limit=None,
                       igrain=None, material_phase_always_present=None, objective_function=None, crystal=None,
                       crystal1=None, strain_free_parameters=None):
    
    if objective_function == None:
        call_global()
        
        init_mr = 0
        init_mat = 0
        init_material = "None"
        init_case = "None"
        init_B = None
        final_match_rate = 0
        match_rate_mma = []
        final_rmv_ind = []
        
        if material0_ == material1_:
            list_of_sets = []
            for ii in range(0, min(nb_spots_consider, len(dist))):
                if max_pred[ii] < softmax_threshold:
                    continue 
                a1 = np.round(dist[ii],3)
                
                for i in range(0, min(nb_spots_consider, len(dist))):
                    if ii==i:
                        continue
                    if (ii,i) in list_of_sets or (i,ii) in list_of_sets:
                        continue
                    
                    if max_pred[i] < softmax_threshold:
                        continue
                    
                    hkl1 = hkl_all_class0[str(predicted_hkl[ii])]
                    hkl1_list = np.array(hkl1)
                    hkl2 = hkl_all_class0[str(predicted_hkl[i])]
                    hkl2_list = np.array(hkl2)
                    Gstar_metric = Gstar_metric0
                    
                    tab_angulardist_temp = CP.AngleBetweenNormals(hkl1_list, hkl2_list, Gstar_metric)
                    np.putmask(tab_angulardist_temp, np.abs(tab_angulardist_temp) < 0.001, 400)
                    
                    list_ = np.where(np.abs(tab_angulardist_temp-a1[i]) < input_params["tolerance"])
                    if len(list_[0]) != 0:
                        list_of_sets.append((ii,i))

        else:
            list_of_sets = []
            for ii in range(0, min(nb_spots_consider, len(dist))):
                if max_pred[ii] < softmax_threshold:
                    continue 
                
                a1 = np.round(dist[ii],3)

                for i in range(0, min(nb_spots_consider, len(dist))):
                    if ii==i:
                        continue
                    if (ii,i) in list_of_sets or (i,ii) in list_of_sets:
                        continue
                    
                    if max_pred[i] < softmax_threshold:
                        continue
                    if class_predicted[ii] < ind_mat and class_predicted[i] < ind_mat:
                        tab_distance_classhkl_data = tab_distance_classhkl_data0
                        tolerance_new = input_params["tolerance"]
                        hkl1 = hkl_all_class0[str(predicted_hkl[ii])]
                        hkl1_list = np.array(hkl1)
                        hkl2 = hkl_all_class0[str(predicted_hkl[i])]
                        hkl2_list = np.array(hkl2)
                        Gstar_metric = Gstar_metric0
                        
                    elif (ind_mat <= class_predicted[ii] < (ind_mat+ind_mat1)) and \
                                        (ind_mat <= class_predicted[i] < (ind_mat+ind_mat1)):
                        tab_distance_classhkl_data = tab_distance_classhkl_data1
                        tolerance_new = input_params["tolerance1"]
                        hkl1 = hkl_all_class1[str(predicted_hkl[ii])]
                        hkl1_list = np.array(hkl1)
                        hkl2 = hkl_all_class1[str(predicted_hkl[i])]
                        hkl2_list = np.array(hkl2)
                        Gstar_metric = Gstar_metric1
                        
                    else:
                        continue
                    
                    tab_angulardist_temp = CP.AngleBetweenNormals(hkl1_list, hkl2_list, Gstar_metric)
                    np.putmask(tab_angulardist_temp, np.abs(tab_angulardist_temp) < 0.001, 400)
                    list_ = np.where(np.abs(tab_angulardist_temp-a1[i]) < tolerance_new)
                    if len(list_[0]) != 0:
                        list_of_sets.append((ii,i))

        ## build a direct connection graph object
        graph_obj = nx.DiGraph(list_of_sets)
        connected_nodes_length = []
        connected_nodes = [[] for i in range(len(graph_obj))]
        for i,line in enumerate(nx.generate_adjlist(graph_obj)):
            connected_nodes_length.append(len(line.split(" ")))
            connected_nodes[i].append([int(jj) for jj in line.split(" ")])
        
        ## sort by maximum node occurance
        connected_nodes_length = np.array(connected_nodes_length)
        connected_nodes_length_sort_ind = np.argsort(connected_nodes_length)[::-1]
  
        mat = 0
        case = "None"
        tried_spots = []
        
        objective_function = []
        for toplist in range(len(graph_obj)):
            # ## continue if less than 3 connections are found for a graph
            # if connected_nodes_length[connected_nodes_length_sort_ind[toplist]] < 2:
            #     continue
            
            for j in connected_nodes[connected_nodes_length_sort_ind[toplist]][0]:
                init_mr = 0
                final_match_rate = 0
                final_rmv_ind = []
                all_stats = []
                for i in connected_nodes[connected_nodes_length_sort_ind[toplist]][0]:
                    if j == i:
                        continue
                    
                    if j in tried_spots and i in tried_spots:
                        continue
                    
                    if material0_ == material1_:
                        tab_distance_classhkl_data = tab_distance_classhkl_data0
                        hkl_all_class = hkl_all_class0
                        material_ = material0_
                        B = B0
                        Gstar_metric = Gstar_metric0
                        case = material_
                        mat = 1
                        input_params["mat"] = mat
                        input_params["Bmat"] = B
                    else:
                        if class_predicted[i] < ind_mat and class_predicted[j] < ind_mat:
                            tab_distance_classhkl_data = tab_distance_classhkl_data0
                            hkl_all_class = hkl_all_class0
                            material_ = material0_
                            B = B0
                            Gstar_metric = Gstar_metric0
                            case = material_
                            mat = 1
                            input_params["mat"] = mat
                            input_params["Bmat"] = B
                        elif (ind_mat <= class_predicted[i] < (ind_mat+ind_mat1)) and \
                                            (ind_mat <= class_predicted[j] < (ind_mat+ind_mat1)):
                            tab_distance_classhkl_data = tab_distance_classhkl_data1
                            hkl_all_class = hkl_all_class1
                            material_ = material1_
                            B = B1
                            Gstar_metric = Gstar_metric1
                            case = material_  
                            mat = 2
                            input_params["mat"] = mat
                            input_params["Bmat"] = B
                        else:
                            mat = 0
                            case = "None"
                            input_params["mat"] = mat
                            input_params["Bmat"] = None
                    
                    if mat == 0:
                        continue                    
                    
                    tth_chi_spot1 = np.array([s_tth[i], s_chi[i]])
                    tth_chi_spot2 = np.array([s_tth[j], s_chi[j]])         
        
                    hkl1 = hkl_all_class[str(predicted_hkl[i])]
                    hkl1_list = np.array(hkl1)
                    hkl2 = hkl_all_class[str(predicted_hkl[j])]
                    hkl2_list = np.array(hkl2)
                    
                    actual_mat, flagAM, \
                    spot1_hkl, spot2_hkl = propose_UB_matrix(hkl1_list, hkl2_list, 
                                                            Gstar_metric, input_params, 
                                                            dist[i,j],
                                                            tth_chi_spot1, tth_chi_spot2, 
                                                            B, method=0, crystal=crystal,
                                                            crystal1=crystal1)
                    
                    if flagAM:
                        continue
                    
                    for iind in range(len(actual_mat)):
                        rot_mat123 = actual_mat[iind]

                        rmv_ind, theospots = remove_spots(s_tth, s_chi, rot_mat123, 
                                                                material_, input_params, 
                                                                dict_dp['detectorparameters'], dict_dp)
                        
                        match_rate = np.round(100 * len(rmv_ind)/theospots, 3)
                        
                        match_rate_mma.append(match_rate)

                        if match_rate > init_mr:
                            final_rmv_ind = rmv_ind                    
                            init_mat = np.copy(mat)
                            input_params["mat"] = init_mat
                            init_material = np.copy(material_)
                            init_case = np.copy(case)
                            init_B = np.copy(B)  
                            input_params["Bmat"] = init_B                                     
                            final_match_rate = np.copy(match_rate)
                            init_mr = np.copy(match_rate)                   
                            all_stats = [i, j, \
                                         spot1_hkl[iind], spot2_hkl[iind], \
                                        tth_chi_spot1, tth_chi_spot2, \
                                        dist[i,j], tab_distance_classhkl_data[i,j], np.round(max_pred[i]*100,3), \
                                        np.round(max_pred[j]*100,3), len(rmv_ind), theospots,\
                                        match_rate, 0.0, rot_mat123, init_mat, init_material, init_B, init_case]
                    tried_spots.append(i)                 
                    
                if (final_match_rate <= cap_matchrate123): ## Nothing found!! 
                    ## Either peaks are not well defined or not found within tolerance and prediction accuracy
                    all_stats = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                        0, 0, 0, 0, 0, np.zeros((3,3))]
                    max_mr, min_mr = 0, 0
                    spot_ind = []
                    mat = 0
                    input_params["mat"] = 0
                    case = "None"
                    objective_function.append([0, [], []])
                else:
                    objective_function.append([final_match_rate, final_rmv_ind, all_stats])     
                tried_spots.append(j)
 
    sort_ind = []
    for ijk in objective_function:
        sort_ind.append(ijk[0])
    sort_ind = np.array(sort_ind)
    sort_ind = np.argsort(sort_ind)[::-1]
    
    for gr_count123 in range(len(sort_ind)):           
        max_mr = objective_function[sort_ind[gr_count123]][0]
        rmv_ind = objective_function[sort_ind[gr_count123]][1]
        all_stats = objective_function[sort_ind[gr_count123]][2]
        
        if len(rmv_ind) == 0 or max_mr==0:
            continue
        
        mat = all_stats[15]
        if mat == 1:
            if igrain==0 and material_phase_always_present ==2:
                mat = 0
                case="None"
            if material0_count >= material0_limit:
                mat = 0
                case="None"
        elif mat == 2:
            if igrain==0 and material_phase_always_present ==1:
                mat = 0
                case="None"
            if material1_count >= material1_limit:
                mat = 0
                case="None"
        
        if mat == 0:
            continue

        current_spots = [len(list(set(rmv_ind) & set(spots1_global[igr])))> coeff_overlap*len(spots1_global[igr]) for igr in range(len(spots1_global))]
        
        if np.any(current_spots):
            continue
                  
        input_params["mat"] = all_stats[15]
        if strain_calculation:
            dev_strain, strain_sample, iR, fR, rot_mat_UB = calculate_strains_fromUB(s_tth, s_chi, all_stats[14], str(all_stats[16]), 
                                                                 input_params, dict_dp['detectorparameters'], 
                                                                 dict_dp, spots, all_stats[17],
                                                                 strain_free_parameters)
        else:
            dev_strain, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
            rot_mat_UB = np.copy(all_stats[14])
        all_stats[14] = rot_mat_UB     
        
        return all_stats, np.max(max_mr), np.min(max_mr), \
                rmv_ind, str(all_stats[18]), all_stats[15], dev_strain, strain_sample, iR, fR, objective_function
    
    all_stats = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                        0, 0, 0, 0, 0, np.zeros((3,3))]
    max_mr, min_mr = 0, 0
    spot_ind = []
    mat = 0
    input_params["mat"] = 0
    case = "None"
    return all_stats, max_mr, min_mr, spot_ind, case, mat, np.zeros((3,3)), np.zeros((3,3)), 0, 0, objective_function

def get_orient_mat_HM(s_tth, s_chi, material0_, material1_, classhkl, class_predicted, predicted_hkl,
                   input_params, hkl_all_class0, hkl_all_class1, max_pred, dict_dp, spots, 
                   dist, Gstar_metric0, Gstar_metric1, B0, B1, softmax_threshold=0.85, mr_threshold=0.85, 
                   tab_distance_classhkl_data0=None, tab_distance_classhkl_data1=None, spots1_global=None,
                   coeff_overlap = None, ind_mat=None, ind_mat1=None, strain_calculation=None,cap_matchrate123=None,
                   material0_count=None, material1_count=None, material0_limit=None, material1_limit=None,
                   igrain=None, material_phase_always_present=None, strain_free_parameters=None):
    call_global()
    
    init_mr = 0
    init_mat = 0
    init_material = "None"
    init_case = "None"
    init_B = None
    final_match_rate = 0
    match_rate_mma = []
    final_rmv_ind = []
    current_spots1 = [0 for igr in range(len(spots1_global))]
    mat = 0
    case = "None"
    all_stats = []
    
    
    #calculate the gnemonic projection space
    imageGNO, nbpeaks, halfdiagonal = computeGnomonicImage(s_tth, s_chi)
    hough, theta_h, d_h = hough_line(imageGNO)

    for i in range(0, min(nb_spots_consider, len(s_tth))):
        for j in range(i+1, min(nb_spots_consider, len(s_tth))):
            overlap = False
            
            ## condition to check if spots lie on the same line
            in_hough_line = False
            for _, anglehs, disths in zip(*hough_line_peaks(hough, theta_h, d_h)):
                y0 = (disths - 0 * np.cos(anglehs)) / np.sin(anglehs)
                y1 = (disths - imageGNO.shape[1] * np.cos(anglehs)) / np.sin(anglehs)
                p1 = np.array((0,y0))
                p2 = np.array((imageGNO.shape[1], y1))
                
                p3_0 = ComputeGnomon_singledata(s_tth[i], s_chi[i])
                p3_1 = ComputeGnomon_singledata(s_tth[j], s_chi[j])

                distance_0 = np.abs(np.cross(p2-p1, p3_0-p1)) / np.linalg.norm(p2-p1)
                distance_1 = np.abs(np.cross(p2-p1, p3_1-p1)) / np.linalg.norm(p2-p1)
                
                if distance_0 < dist_threshold and distance_1 < dist_threshold:
                    # print(distance_0, distance_1)
                    in_hough_line = True
                    
                if in_hough_line:
                    break
            
            if not in_hough_line:
                continue
            
            if (max_pred[j] < softmax_threshold) or (j in spots) or \
                (max_pred[i] < softmax_threshold) or (i in spots):
                continue
            
            if material0_ == material1_:
                tab_distance_classhkl_data = tab_distance_classhkl_data0
                hkl_all_class = hkl_all_class0
                material_ = material0_
                B = B0
                Gstar_metric = Gstar_metric0
                case = material_
                mat = 1
                input_params["mat"] = mat
                input_params["Bmat"] = B
            else:
                if class_predicted[i] < ind_mat and class_predicted[j] < ind_mat:
                    tab_distance_classhkl_data = tab_distance_classhkl_data0
                    hkl_all_class = hkl_all_class0
                    material_ = material0_
                    B = B0
                    Gstar_metric = Gstar_metric0
                    case = material_
                    mat = 1
                    if igrain==0 and material_phase_always_present == 2:
                        mat = 0
                        case="None"
                    if material0_count >= material0_limit:
                        mat = 0
                        case="None"
                    input_params["mat"] = mat
                    input_params["Bmat"] = B
                elif (ind_mat <= class_predicted[i] < (ind_mat+ind_mat1)) and \
                                    (ind_mat <= class_predicted[j] < (ind_mat+ind_mat1)):
                    tab_distance_classhkl_data = tab_distance_classhkl_data1
                    hkl_all_class = hkl_all_class1
                    material_ = material1_
                    B = B1
                    Gstar_metric = Gstar_metric1
                    case = material_  
                    mat = 2
                    if igrain==0 and material_phase_always_present == 1:
                        mat = 0
                        case="None"
                    if material1_count >= material1_limit:
                        mat = 0
                        case="None"
                    input_params["mat"] = mat
                    input_params["Bmat"] = B
                else:
                    mat = 0
                    case = "None"
                    input_params["mat"] = mat
                    input_params["Bmat"] = None
            
            if mat == 0:
                continue
            
            tth_chi_spot1 = np.array([s_tth[i], s_chi[i]])
            tth_chi_spot2 = np.array([s_tth[j], s_chi[j]])

            hkl1 = hkl_all_class[str(predicted_hkl[i])]
            hkl1_list = np.array(hkl1)
            hkl2 = hkl_all_class[str(predicted_hkl[j])]
            hkl2_list = np.array(hkl2)
            
            actual_mat, flagAM, \
            spot1_hkl, spot2_hkl = propose_UB_matrix(hkl1_list, hkl2_list, 
                                                    Gstar_metric, input_params, 
                                                    dist[i,j],
                                                    tth_chi_spot1, tth_chi_spot2, 
                                                    B, method=0)
            
            if flagAM:
                continue

            for iind in range(len(actual_mat)): 
                rot_mat123 = actual_mat[iind]

                rmv_ind, theospots = remove_spots(s_tth, s_chi, rot_mat123, 
                                                    material_, input_params, 
                                                    dict_dp['detectorparameters'], dict_dp)
                
                overlap = False
                current_spots = [len(list(set(rmv_ind) & set(spots1_global[igr]))) for igr in range(len(spots1_global))]
                for igr in range(len(spots1_global)):
                    if current_spots[igr] > coeff_overlap*len(spots1_global[igr]):
                        overlap = True
                        break
                
                if overlap:
                    continue
    
                match_rate = np.round(100 * len(rmv_ind)/theospots,3)
                
                match_rate_mma.append(match_rate)
                if match_rate > init_mr:
                    current_spots1 = current_spots                       
                    init_mat = np.copy(mat)
                    input_params["mat"] = init_mat
                    init_material = np.copy(material_)
                    init_case = np.copy(case)
                    init_B = np.copy(B)
                    input_params["Bmat"] = init_B  
                    final_rmv_ind = rmv_ind                            
                    final_match_rate = np.copy(match_rate)
                    init_mr = np.copy(match_rate)
                    all_stats = [i, j, \
                                 spot1_hkl[iind], spot2_hkl[iind], \
                                tth_chi_spot1, tth_chi_spot2, \
                                dist[i,j], tab_distance_classhkl_data[i,j], np.round(max_pred[i]*100,3), \
                                np.round(max_pred[j]*100,3), len(rmv_ind), theospots,\
                                match_rate, 0.0, rot_mat123]
    
                if (final_match_rate >= mr_threshold*100.) and not overlap:
                    if strain_calculation:
                        dev_strain, strain_sample, iR, fR, rot_mat_UB = calculate_strains_fromUB(s_tth, s_chi, all_stats[14], str(init_material), 
                                                                             input_params, dict_dp['detectorparameters'], 
                                                                             dict_dp, spots, init_B,
                                                                             strain_free_parameters)
                    else:
                        dev_strain, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
                        rot_mat_UB = np.copy(all_stats[14])
                    
                    all_stats[14] = rot_mat_UB
                    return all_stats, np.max(match_rate_mma), np.min(match_rate_mma), \
                            final_rmv_ind, str(init_case), init_mat, dev_strain, strain_sample, iR, fR

    overlap = False
    for igr in range(len(spots1_global)):
        if current_spots1[igr] > coeff_overlap*len(spots1_global[igr]):
            overlap = True
            
    if (final_match_rate <= cap_matchrate123) or overlap: ## Nothing found!! 
        ## Either peaks are not well defined or not found within tolerance and prediction accuracy
        all_stats = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                            0, 0, 0, 0, 0, np.zeros((3,3))]
        max_mr, min_mr = 0, 0
        spot_ind = []
        mat = 0
        input_params["mat"] = 0
        case = "None"
        return all_stats, max_mr, min_mr, spot_ind, case, mat, np.zeros((3,3)), np.zeros((3,3)), 0, 0

    input_params["mat"] = init_mat
    if strain_calculation:
        dev_strain, strain_sample, iR, fR, rot_mat_UB = calculate_strains_fromUB(s_tth, s_chi, all_stats[14], str(init_material), 
                                                             input_params, dict_dp['detectorparameters'], 
                                                             dict_dp, spots, init_B,
                                                             strain_free_parameters)
    else:
        dev_strain, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
        rot_mat_UB = np.copy(all_stats[14])
    all_stats[14] = rot_mat_UB  
    return all_stats, np.max(match_rate_mma), np.min(match_rate_mma), \
            final_rmv_ind, str(init_case), init_mat, dev_strain, strain_sample, iR, fR

def get_orient_mat(s_tth, s_chi, material0_, material1_, classhkl, class_predicted, predicted_hkl,
                   input_params, hkl_all_class0, hkl_all_class1, max_pred, dict_dp, spots, 
                   dist, Gstar_metric0, Gstar_metric1, B0, B1, softmax_threshold=0.85, mr_threshold=0.85, 
                   tab_distance_classhkl_data0=None, tab_distance_classhkl_data1=None, spots1_global=None,
                   coeff_overlap = None, ind_mat=None, ind_mat1=None, strain_calculation=None,cap_matchrate123=None,
                   material0_count=None, material1_count=None, material0_limit=None, material1_limit=None,
                   igrain=None, material_phase_always_present=None, strain_free_parameters=None):
    call_global()
    
    init_mr = 0
    init_mat = 0
    init_material = "None"
    init_case = "None"
    init_B = None
    final_match_rate = 0
    match_rate_mma = []
    final_rmv_ind = []
    current_spots1 = [0 for igr in range(len(spots1_global))]
    mat = 0
    case = "None"
    all_stats = []
    
    for i in range(0, min(nb_spots_consider, len(s_tth))):
        for j in range(i+1, min(nb_spots_consider, len(s_tth))):
            overlap = False

            if (max_pred[j] < softmax_threshold) or (j in spots) or \
                (max_pred[i] < softmax_threshold) or (i in spots):
                continue
            
            if material0_ == material1_:
                tab_distance_classhkl_data = tab_distance_classhkl_data0
                hkl_all_class = hkl_all_class0
                material_ = material0_
                B = B0
                Gstar_metric = Gstar_metric0
                case = material_
                mat = 1
                input_params["mat"] = mat
                input_params["Bmat"] = B
            else:
                if class_predicted[i] < ind_mat and class_predicted[j] < ind_mat:
                    tab_distance_classhkl_data = tab_distance_classhkl_data0
                    hkl_all_class = hkl_all_class0
                    material_ = material0_
                    B = B0
                    Gstar_metric = Gstar_metric0
                    case = material_
                    mat = 1
                    if igrain==0 and material_phase_always_present == 2:
                        mat = 0
                        case="None"
                    if material0_count >= material0_limit:
                        mat = 0
                        case="None"
                    input_params["mat"] = mat
                    input_params["Bmat"] = B
                elif (ind_mat <= class_predicted[i] < (ind_mat+ind_mat1)) and \
                                    (ind_mat <= class_predicted[j] < (ind_mat+ind_mat1)):
                    tab_distance_classhkl_data = tab_distance_classhkl_data1
                    hkl_all_class = hkl_all_class1
                    material_ = material1_
                    B = B1
                    Gstar_metric = Gstar_metric1
                    case = material_  
                    mat = 2
                    if igrain==0 and material_phase_always_present == 1:
                        mat = 0
                        case="None"
                    if material1_count >= material1_limit:
                        mat = 0
                        case="None"
                    input_params["mat"] = mat
                    input_params["Bmat"] = B
                else:
                    mat = 0
                    case = "None"
                    input_params["mat"] = mat
                    input_params["Bmat"] = None
            
            if mat == 0:
                continue
            
            tth_chi_spot1 = np.array([s_tth[i], s_chi[i]])
            tth_chi_spot2 = np.array([s_tth[j], s_chi[j]])

            hkl1 = hkl_all_class[str(predicted_hkl[i])]
            hkl1_list = np.array(hkl1)
            hkl2 = hkl_all_class[str(predicted_hkl[j])]
            hkl2_list = np.array(hkl2)
            
            actual_mat, flagAM, \
            spot1_hkl, spot2_hkl = propose_UB_matrix(hkl1_list, hkl2_list, 
                                                    Gstar_metric, input_params, 
                                                    dist[i,j],
                                                    tth_chi_spot1, tth_chi_spot2, 
                                                    B, method=0)
            
            if flagAM:
                continue

            for iind in range(len(actual_mat)): 
                rot_mat123 = actual_mat[iind]

                rmv_ind, theospots = remove_spots(s_tth, s_chi, rot_mat123, 
                                                    material_, input_params, 
                                                    dict_dp['detectorparameters'], dict_dp)
                
                overlap = False
                current_spots = [len(list(set(rmv_ind) & set(spots1_global[igr]))) for igr in range(len(spots1_global))]
                for igr in range(len(spots1_global)):
                    if current_spots[igr] > coeff_overlap*len(spots1_global[igr]):
                        overlap = True
                        break
                
                if overlap:
                    continue
    
                match_rate = np.round(100 * len(rmv_ind)/theospots,3)
                
                match_rate_mma.append(match_rate)
                if match_rate > init_mr:
                    current_spots1 = current_spots                       
                    init_mat = np.copy(mat)
                    input_params["mat"] = init_mat
                    init_material = np.copy(material_)
                    init_case = np.copy(case)
                    init_B = np.copy(B)
                    input_params["Bmat"] = init_B  
                    final_rmv_ind = rmv_ind                            
                    final_match_rate = np.copy(match_rate)
                    init_mr = np.copy(match_rate)
                    all_stats = [i, j, \
                                 spot1_hkl[iind], spot2_hkl[iind], \
                                tth_chi_spot1, tth_chi_spot2, \
                                dist[i,j], tab_distance_classhkl_data[i,j], np.round(max_pred[i]*100,3), \
                                np.round(max_pred[j]*100,3), len(rmv_ind), theospots,\
                                match_rate, 0.0, rot_mat123]
    
                if (final_match_rate >= mr_threshold*100.) and not overlap:
                    if strain_calculation:
                        dev_strain, strain_sample, iR, fR, rot_mat_UB = calculate_strains_fromUB(s_tth, s_chi, all_stats[14], str(init_material), 
                                                                             input_params, dict_dp['detectorparameters'], 
                                                                             dict_dp, spots, init_B,
                                                                             strain_free_parameters)
                    else:
                        dev_strain, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
                        rot_mat_UB = np.copy(all_stats[14])
                    
                    all_stats[14] = rot_mat_UB
                    return all_stats, np.max(match_rate_mma), np.min(match_rate_mma), \
                            final_rmv_ind, str(init_case), init_mat, dev_strain, strain_sample, iR, fR

    overlap = False
    for igr in range(len(spots1_global)):
        if current_spots1[igr] > coeff_overlap*len(spots1_global[igr]):
            overlap = True
            
    if (final_match_rate <= cap_matchrate123) or overlap: ## Nothing found!! 
        ## Either peaks are not well defined or not found within tolerance and prediction accuracy
        all_stats = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                            0, 0, 0, 0, 0, np.zeros((3,3))]
        max_mr, min_mr = 0, 0
        spot_ind = []
        mat = 0
        input_params["mat"] = 0
        case = "None"
        return all_stats, max_mr, min_mr, spot_ind, case, mat, np.zeros((3,3)), np.zeros((3,3)), 0, 0

    input_params["mat"] = init_mat
    if strain_calculation:
        dev_strain, strain_sample, iR, fR, rot_mat_UB = calculate_strains_fromUB(s_tth, s_chi, all_stats[14], str(init_material), 
                                                             input_params, dict_dp['detectorparameters'], 
                                                             dict_dp, spots, init_B,
                                                             strain_free_parameters)
    else:
        dev_strain, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
        rot_mat_UB = np.copy(all_stats[14])
    all_stats[14] = rot_mat_UB  
    return all_stats, np.max(match_rate_mma), np.min(match_rate_mma), \
            final_rmv_ind, str(init_case), init_mat, dev_strain, strain_sample, iR, fR

def propose_UB_matrix(hkl1_list, hkl2_list, Gstar_metric, input_params, dist123,
                      tth_chi_spot1, tth_chi_spot2, B, method=0, crystal=None,
                      crystal1=None):
    
    if method == 0:
        tab_angulardist_temp = CP.AngleBetweenNormals(hkl1_list, hkl2_list, Gstar_metric)
        
        if input_params["mat"] == 1:
            list_ = np.where(np.abs(tab_angulardist_temp-dist123) < input_params["tolerance"])
            final_crystal=crystal
            
        elif input_params["mat"] == 2:
            list_ = np.where(np.abs(tab_angulardist_temp-dist123) < input_params["tolerance1"])
            final_crystal=crystal1
        
        if final_crystal != None:
            symm_operator = final_crystal._hklsym
        else:
            symm_operator = np.eye(3)
        
        if len(list_[0]) == 0:
            return None, True, 0, 0

        rot_mat_abs = []
        actual_mat = []
        spot1_hkl = []
        spot2_hkl = []
        
        triedspots = []
        for ii, jj in zip(list_[0], list_[1]):
            if ii in triedspots and jj in triedspots:
                continue

            conti_ = False
            
            try:
                rot_mat1 = FindO.OrientMatrix_from_2hkl(hkl1_list[ii], tth_chi_spot1, \
                                                        hkl2_list[jj], tth_chi_spot2,
                                                        B)
                # rot_mat1 = find_uniq_u(rot_mat1, symm_operator)
            except:
                continue                    
            
            copy_rm = np.copy(rot_mat1)
            copy_rm = np.round(np.abs(copy_rm),5)
            copy_rm.sort(axis=1)
            for iji in rot_mat_abs:
                iji.sort(axis=1)                        
                if np.all(iji==copy_rm):
                    conti_ = True
                    break
            if conti_:
                continue
            rot_mat_abs.append(np.round(np.abs(rot_mat1),5))
            actual_mat.append(rot_mat1)
            spot1_hkl.append(hkl1_list[ii])
            spot2_hkl.append(hkl2_list[jj])
            triedspots.append(ii)
            triedspots.append(jj)
    else:  
        # method 2
        hkl_all = np.vstack((hkl1_list, hkl2_list))
        LUT = FindO.GenerateLookUpTable(hkl_all, Gstar_metric)
        if input_params["mat"] == 1:
            hkls = FindO.PlanePairs_2(dist123, input_params["tolerance"], LUT, onlyclosest=1)
        elif input_params["mat"] == 2:
            hkls = FindO.PlanePairs_2(dist123, input_params["tolerance1"], LUT, onlyclosest=1)            
         
        if np.all(hkls == None):
            return None, True, 0, 0
                
        rot_mat_abs = []
        actual_mat = []
        spot1_hkl = []
        spot2_hkl = []
        
        for ii in range(len(hkls)):
            if np.all(hkls[ii][0] == hkls[ii][1]):
                continue
            conti_ = False
            
            try:
                rot_mat1 = FindO.OrientMatrix_from_2hkl(hkls[ii][0], tth_chi_spot1, \
                                                        hkls[ii][1], tth_chi_spot2,
                                                        B)
                # rot_mat1 = find_uniq_u(rot_mat1, symm_operator)
            except:
                continue                    
            
            copy_rm = np.copy(rot_mat1)
            copy_rm = np.round(np.abs(copy_rm),5)
            copy_rm.sort(axis=1)
            for iji in rot_mat_abs:
                iji.sort(axis=1)
                if np.all(iji==copy_rm):
                    conti_ = True
                    break

            if conti_:
                continue
            rot_mat_abs.append(np.round(np.abs(rot_mat1),5))
            actual_mat.append(rot_mat1)
            spot1_hkl.append(hkls[ii][0])
            spot2_hkl.append(hkls[ii][1])
    
    ## just fixing a* to x seems ok; if not think of aligning b* to xy plane
    sum_sign = []
    for nkl in range(len(actual_mat)):
        temp_mat = np.dot(actual_mat[nkl], B)
        ## fix could be to choose a matrix that aligns best the b* vector to Y axis or a* to X axis
        # if np.argmax(np.abs(temp_mat[:2,0])) == 0 and \
        #         np.argmax(np.abs(temp_mat[:2,1])) == 1: ##a* along x, b*along y
        if np.argmax(np.abs(temp_mat[:2,0])) == 0: ##a* along x
            sum_sign.append(2)
        elif np.argmax(np.abs(temp_mat[:2,0])) ==  np.argmax(np.abs(temp_mat[:2,1])):
            sum_sign.append(0)
        else:
            sum_sign.append(1)
    ind_sort = np.argsort(sum_sign)[::-1]
    ## re-arrange
    actual_mat1 = []
    spot1_hkl1, spot2_hkl1 = [], []
    for inin in ind_sort:
        actual_mat1.append(actual_mat[inin])
        spot1_hkl1.append(spot1_hkl[inin])
        spot2_hkl1.append(spot2_hkl[inin])
    actual_mat, spot1_hkl, spot2_hkl = actual_mat1, spot1_hkl1, spot2_hkl1
    return actual_mat, False, spot1_hkl, spot2_hkl

def find_uniq_u(u, syms):
    """
    Unique representation of rotation matrix:
        apply this function before strain 
        as distorted unit cell may produce undesireable matrix
    """
    uniq = u
    tmax = np.trace(uniq)
    for sym in syms:
        cand = np.dot(sym, uniq)
        t = np.trace(cand)
        if np.trace(cand) > tmax:
            uniq = cand
            tmax = t
    return np.array(uniq)

def remove_spots(s_tth, s_chi, first_match123, material_, input_params, detectorparameters, dict_dp):
    try:
        grain = CP.Prepare_Grain(material_, first_match123, dictmaterials=dictLT.dict_Materials)
        ### initialize global variables to be used later
        call_global()
    except:
        return [], 100
    #### Perhaps better than SimulateResult function
    kf_direction = dict_dp["kf_direction"]
    detectordistance = dict_dp["detectorparameters"][0]
    detectordiameter = dict_dp["detectordiameter"]
    pixelsize = dict_dp["pixelsize"]
    dim = dict_dp["dim"]
           
    spots2pi = LT.getLaueSpots(CST_ENERGYKEV / input_params["emax"], 
                               CST_ENERGYKEV / input_params["emin"],
                                    [grain],
                                    fastcompute=1,
                                    verbose=0,
                                    kf_direction=kf_direction,
                                    ResolutionAngstrom=False,
                                    dictmaterials=dictLT.dict_Materials)

    TwicethetaChi = LT.filterLaueSpots_full_np(spots2pi[0][0], None, onlyXYZ=False,
                                                    HarmonicsRemoval=0,
                                                    fastcompute=1,
                                                    kf_direction=kf_direction,
                                                    detectordistance=detectordistance,
                                                    detectordiameter=detectordiameter,
                                                    pixelsize=pixelsize,
                                                    dim=dim)
    ## get proximity for exp and theo spots
    if input_params["mat"] == 1:
        angtol = input_params["tolerance"]
    elif input_params["mat"] == 2:
        angtol = input_params["tolerance1"]
    else:
        return [], 100
    
    if option_global =="v1":
        # print("entering v1")
        List_Exp_spot_close, residues_link, _ = getProximityv1(np.array([TwicethetaChi[0], TwicethetaChi[1]]),  # warning array(2theta, chi)
                                                  s_tth/2.0, s_chi,  # warning theta, chi for exp
                                                  angtol=angtol)
    elif option_global =="v2":
        List_Exp_spot_close, residues_link, _ = getProximityv1_ambigious(np.array([TwicethetaChi[0], TwicethetaChi[1]]),  # warning array(2theta, chi)
                                                  s_tth/2.0, s_chi,  # warning theta, chi for exp
                                                  angtol=angtol)
    else:
        List_Exp_spot_close, residues_link, _ = getProximityv1_ambigious(np.array([TwicethetaChi[0], TwicethetaChi[1]]),  # warning array(2theta, chi)
                                                  s_tth/2.0, s_chi,  # warning theta, chi for exp
                                                  angtol=angtol)
        List_Exp_spot_close, ind_uniq = np.unique(List_Exp_spot_close, return_index=True)
        residues_link = np.take(residues_link, ind_uniq)

    if np.average(residues_link) > residues_threshold:
        return [], 100
    
    if len(np.unique(List_Exp_spot_close)) < nb_spots_global_threshold:
        return [], 100
    
    return List_Exp_spot_close, len(TwicethetaChi[0])

def simulate_spots(rot_mat, material_, emax, emin, detectorparameters, dict_dp, angtol,
                   s_tth, s_chi):
    try:
        grain = CP.Prepare_Grain(material_, rot_mat, dictmaterials=dictLT.dict_Materials)
        ### initialize global variables to be used later
        call_global()
    except:
        return [], [], [], [], []
    
    #### Perhaps better than SimulateResult function
    kf_direction = dict_dp["kf_direction"]
    detectordistance = dict_dp["detectorparameters"][0]
    detectordiameter = dict_dp["detectordiameter"]
    pixelsize = dict_dp["pixelsize"]
    dim = dict_dp["dim"]
           
    spots2pi = LT.getLaueSpots(CST_ENERGYKEV / emax, CST_ENERGYKEV / emin,
                                    [grain],
                                    fastcompute=0,
                                    verbose=0,
                                    kf_direction=kf_direction,
                                    ResolutionAngstrom=False,
                                    dictmaterials=dictLT.dict_Materials)
    TwicethetaChi = LT.filterLaueSpots_full_np(spots2pi[0][0], spots2pi[1][0], onlyXYZ=False,
                                                    HarmonicsRemoval=0,
                                                    fastcompute=0,
                                                    kf_direction=kf_direction,
                                                    detectordistance=detectordistance,
                                                    detectordiameter=detectordiameter,
                                                    pixelsize=pixelsize,
                                                    dim=dim)
    if option_global =="v1":
        List_Exp_spot_close, residues_link, theo_index = getProximityv1(np.array([TwicethetaChi[0], TwicethetaChi[1]]),  # warning array(2theta, chi)
                                                  s_tth/2.0, s_chi,  # warning theta, chi for exp
                                                  angtol=angtol)
    elif option_global =="v2":
        List_Exp_spot_close, residues_link, theo_index = getProximityv1_ambigious(np.array([TwicethetaChi[0], TwicethetaChi[1]]),  # warning array(2theta, chi)
                                                  s_tth/2.0, s_chi,  # warning theta, chi for exp
                                                  angtol=angtol)
    else:
        List_Exp_spot_close, residues_link, theo_index = getProximityv1_ambigious(np.array([TwicethetaChi[0], TwicethetaChi[1]]),  # warning array(2theta, chi)
                                                  s_tth/2.0, s_chi,  # warning theta, chi for exp
                                                  angtol=angtol)
        List_Exp_spot_close, ind_uniq = np.unique(List_Exp_spot_close, return_index=True)
        residues_link = np.take(residues_link, ind_uniq)
        theo_index = np.take(theo_index, ind_uniq)
    return TwicethetaChi[0], TwicethetaChi[1], TwicethetaChi[2], TwicethetaChi[3], List_Exp_spot_close, residues_link, theo_index

def getProximityv1_ambigious(TwicethetaChi, data_theta, data_chi, angtol=0.5):
    # theo simul data
    theodata = np.array([TwicethetaChi[0] / 2.0, TwicethetaChi[1]]).T
    # exp data
    sorted_data = np.array([data_theta, data_chi]).T
    table_dist = GT.calculdist_from_thetachi(sorted_data, theodata)
    prox_table = np.argmin(table_dist, axis=1)
    allresidues = np.amin(table_dist, axis=1)
    very_close_ind = np.where(allresidues < angtol)[0]
    List_Exp_spot_close = []
    theo_index = []
    if len(very_close_ind) > 0:
        for theospot_ind in very_close_ind:  # loop over theo spots index
            List_Exp_spot_close.append(prox_table[theospot_ind])
            theo_index.append(theospot_ind)
    return List_Exp_spot_close, allresidues[very_close_ind], theo_index

def getProximityv1( TwicethetaChi, data_theta, data_chi, angtol=0.5):
    theodata = np.array([TwicethetaChi[0] / 2.0, TwicethetaChi[1]]).T
    # exp data
    sorted_data = np.array([data_theta, data_chi]).T
    table_dist = GT.calculdist_from_thetachi(sorted_data, theodata)

    prox_table = np.argmin(table_dist, axis=1)
    allresidues = np.amin(table_dist, axis=1)
    very_close_ind = np.where(allresidues < angtol)[0]
    List_Exp_spot_close = []
    Miller_Exp_spot = []
    if len(very_close_ind) > 0:
        for theospot_ind in very_close_ind:  # loop over theo spots index
            List_Exp_spot_close.append(prox_table[theospot_ind])
            Miller_Exp_spot.append(1)
    else:
        return [], [], []
    # removing exp spot which appears many times(close to several simulated spots of one grain)--------------
    arrayLESC = np.array(List_Exp_spot_close, dtype=float)
    sorted_LESC = np.sort(arrayLESC)
    diff_index = sorted_LESC - np.array(list(sorted_LESC[1:]) + [sorted_LESC[0]])
    toremoveindex = np.where(diff_index == 0)[0]
    if len(toremoveindex) > 0:
        # index of exp spot in arrayLESC that are duplicated
        ambiguous_exp_ind = GT.find_closest(np.array(sorted_LESC[toremoveindex], dtype=float), arrayLESC, 0.1)[1]
        for ind in ambiguous_exp_ind:
            Miller_Exp_spot[ind] = None
    
    ProxTablecopy = np.copy(prox_table)

    for theo_ind, exp_ind in enumerate(prox_table):
        where_th_ind = np.where(ProxTablecopy == exp_ind)[0]
        if len(where_th_ind) > 1:
            for indy in where_th_ind:
                ProxTablecopy[indy] = -prox_table[indy]
            closest = np.argmin(allresidues[where_th_ind])
            ProxTablecopy[where_th_ind[closest]] = -ProxTablecopy[where_th_ind[closest]]
    
    singleindices = []
    refine_indexed_spots = {}
    # loop over close exp. spots
    for k in range(len(List_Exp_spot_close)):
        exp_index = List_Exp_spot_close[k]
        if not singleindices.count(exp_index):
            singleindices.append(exp_index)
            theo_index = np.where(ProxTablecopy == exp_index)[0]
            if (len(theo_index) == 1):  # only one theo spot close to the current exp. spot
                refine_indexed_spots[exp_index] = [exp_index, theo_index, Miller_Exp_spot[k]]
            else:  # recent PATCH:
                closest_theo_ind = np.argmin(allresidues[theo_index])
                if allresidues[theo_index][closest_theo_ind] < angtol:
                    refine_indexed_spots[exp_index] = [exp_index, theo_index[closest_theo_ind], Miller_Exp_spot[k]]       
    listofpairs = []
    theo_index = []
    linkResidues = []        
    selectedAbsoluteSpotIndices = np.arange(len(data_theta))
    for val in list(refine_indexed_spots.values()):
        if val[2] is not None:
            localspotindex = val[0]
            if not isinstance(val[1], (list, np.ndarray)):
                closetheoindex = val[1]
            else:
                closetheoindex = val[1][0]
            absolute_spot_index = selectedAbsoluteSpotIndices[localspotindex]
            listofpairs.append(absolute_spot_index)  # Exp, Theo,  where -1 for specifying that it came from automatic linking
            theo_index.append(closetheoindex)
            linkResidues.append(allresidues[closetheoindex])
    return listofpairs, linkResidues, theo_index

def refineonce_fromUB(s_tth, s_chi, UBmat, grain, input_params, 
                             detectorparameters, dict_dp, B_matrix):
    # starting B0matrix corresponding to the unit cell   -----
    B0matrix = np.copy(B_matrix)
    if input_params["mat"] == 1:
        AngTol = input_params["tolerance"]
    elif input_params["mat"] == 2:
        AngTol = input_params["tolerance1"]
    #### Spots in first match (no refining, just simple auto links to filter spots)
    Twicetheta, Chi, Miller_ind, posx, posy, _ = LT.SimulateLaue(grain,
                                                             input_params["emin"], 
                                                             input_params["emax"], 
                                                             detectorparameters,
                                                             kf_direction=dict_dp['kf_direction'],
                                                             removeharmonics=1,
                                                             pixelsize=dict_dp['pixelsize'],
                                                             dim=dict_dp['dim'],
                                                             ResolutionAngstrom=False,
                                                             detectordiameter=dict_dp['detectordiameter'],
                                                             dictmaterials=dictLT.dict_Materials)
    ## get proximity for exp and theo spots
    linkedspots_link, linkExpMiller_link, \
        linkResidues_link = getProximityv0(np.array([Twicetheta, Chi]),  # warning array(2theta, chi)
                                                                            s_tth/2.0, s_chi, Miller_ind,  # warning theta, chi for exp
                                                                            angtol=float(AngTol))
    if len(linkedspots_link) < 8:
        return UBmat
    
    linkedspots_fit = linkedspots_link
    linkExpMiller_fit = linkExpMiller_link
    
    arraycouples = np.array(linkedspots_fit)
    exp_indices = np.array(arraycouples[:, 0], dtype=np.int)
    sim_indices = np.array(arraycouples[:, 1], dtype=np.int)

    nb_pairs = len(exp_indices)
    Data_Q = np.array(linkExpMiller_fit)[:, 1:]
    sim_indices = np.arange(nb_pairs)  # for fitting function this must be an arange...

    pixX = np.take(dict_dp['peakX'], exp_indices)
    pixY = np.take(dict_dp['peakY'], exp_indices)
    weights = None #np.take(dict_dp['intensity'], exp_indices)
    
    starting_orientmatrix = np.copy(UBmat)

    results = None
    # ----------------------------------
    #  refinement model
    # ----------------------------------
    # -------------------------------------------------------
    allparameters = np.array(detectorparameters + [1, 1, 0, 0, 0] + [0, 0, 0])
    # strain & orient
    initial_values = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0])
    arr_indexvaryingparameters = np.arange(5, 13)

    results = FitO.fit_on_demand_strain(initial_values,
                                            Data_Q,
                                            allparameters,
                                            FitO.error_function_on_demand_strain,
                                            arr_indexvaryingparameters,
                                            sim_indices,
                                            pixX,
                                            pixY,
                                            initrot=starting_orientmatrix,
                                            Bmat=B0matrix,
                                            pixelsize=dict_dp['pixelsize'],
                                            dim=dict_dp['dim'],
                                            verbose=0,
                                            weights=weights,
                                            kf_direction=dict_dp['kf_direction'])

    if results is None:
        return UBmat
    residues, deltamat, newmatrix = FitO.error_function_on_demand_strain(
                                                                        results,
                                                                        Data_Q,
                                                                        allparameters,
                                                                        arr_indexvaryingparameters,
                                                                        sim_indices,
                                                                        pixX,
                                                                        pixY,
                                                                        initrot=starting_orientmatrix,
                                                                        Bmat=B0matrix,
                                                                        pureRotation=0,
                                                                        verbose=1,
                                                                        pixelsize=dict_dp['pixelsize'],
                                                                        dim=dict_dp['dim'],
                                                                        weights=weights,
                                                                        kf_direction=dict_dp['kf_direction'])
    UBmat = np.copy(newmatrix) 
    return UBmat

def calculate_strains_fromUB(s_tth, s_chi, UBmat, material_, input_params, 
                             detectorparameters, dict_dp, spots, B_matrix, strain_free_parameters):
    ## for the moment strain_free_parameters is a trial implementation 
    if ("a" not in strain_free_parameters) and len(strain_free_parameters)>=5:
        if additional_expression[0] != "none":
            print("Note: additional_expression is not applied for the current set of strain free parameters")
        # starting B0matrix corresponding to the unit cell   -----
        B0matrix = np.copy(B_matrix)
        latticeparams = dictLT.dict_Materials[material_][1]
        ## Included simple multi level refinement of strains
        init_residues = -0.1
        final_residues = -0.1
        
        if input_params["mat"] == 1:
            straintolerance = input_params["tolerancestrain"]
        elif input_params["mat"] == 2:
            straintolerance = input_params["tolerancestrain1"]
        
        devstrain, deviatoricstrain_sampleframe = np.zeros((3,3)), np.zeros((3,3))
        for ijk, AngTol in enumerate(straintolerance):
            #### Spots in first match (no refining, just simple auto links to filter spots)        
            grain = CP.Prepare_Grain(material_, UBmat, dictmaterials=dictLT.dict_Materials)

            Twicetheta, Chi, Miller_ind, posx, posy, _ = LT.SimulateLaue(grain,
                                                                     input_params["emin"], 
                                                                     input_params["emax"], 
                                                                     detectorparameters,
                                                                     kf_direction=dict_dp['kf_direction'],
                                                                     removeharmonics=1,
                                                                     pixelsize=dict_dp['pixelsize'],
                                                                     dim=dict_dp['dim'],
                                                                     ResolutionAngstrom=False,
                                                                     detectordiameter=dict_dp['detectordiameter'],
                                                                     dictmaterials=dictLT.dict_Materials)
            ## get proximity for exp and theo spots
            linkedspots_link, linkExpMiller_link, \
                linkResidues_link = getProximityv0(np.array([Twicetheta, Chi]),  # warning array(2theta, chi)
                                                                                    s_tth/2.0, s_chi, Miller_ind,  # warning theta, chi for exp
                                                                                    angtol=float(AngTol))
            
            if len(linkedspots_link) < 8:
                return np.zeros((3,3)), np.zeros((3,3)), init_residues, final_residues, UBmat
            
            linkedspots_fit = linkedspots_link
            linkExpMiller_fit = linkExpMiller_link
            
            arraycouples = np.array(linkedspots_fit)
            exp_indices = np.array(arraycouples[:, 0], dtype=np.int)
            sim_indices = np.array(arraycouples[:, 1], dtype=np.int)
        
            nb_pairs = len(exp_indices)
            Data_Q = np.array(linkExpMiller_fit)[:, 1:]
            sim_indices = np.arange(nb_pairs)  # for fitting function this must be an arange...
        
            pixX = np.take(dict_dp['peakX'], exp_indices)
            pixY = np.take(dict_dp['peakY'], exp_indices)
            weights = None #np.take(dict_dp['intensity'], exp_indices)
            
            starting_orientmatrix = np.copy(UBmat)
        
            results = None
            # ----------------------------------
            #  refinement model
            # ----------------------------------
            # -------------------------------------------------------
            allparameters = np.array(detectorparameters + [1, 1, 0, 0, 0] + [0, 0, 0])
            # strain & orient
            initial_values = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0])
            arr_indexvaryingparameters = np.arange(5, 13)
        
            residues, deltamat, newmatrix = FitO.error_function_on_demand_strain(
                                                                                initial_values,
                                                                                Data_Q,
                                                                                allparameters,
                                                                                arr_indexvaryingparameters,
                                                                                sim_indices,
                                                                                pixX,
                                                                                pixY,
                                                                                initrot=starting_orientmatrix,
                                                                                Bmat=B0matrix,
                                                                                pureRotation=0,
                                                                                verbose=1,
                                                                                pixelsize=dict_dp['pixelsize'],
                                                                                dim=dict_dp['dim'],
                                                                                weights=weights,
                                                                                kf_direction=dict_dp['kf_direction'])
            init_mean_residues = np.copy(np.mean(residues))
            
            if ijk == 0:
                init_residues = np.copy(init_mean_residues)
            
            results = FitO.fit_on_demand_strain(initial_values,
                                                    Data_Q,
                                                    allparameters,
                                                    FitO.error_function_on_demand_strain,
                                                    arr_indexvaryingparameters,
                                                    sim_indices,
                                                    pixX,
                                                    pixY,
                                                    initrot=starting_orientmatrix,
                                                    Bmat=B0matrix,
                                                    pixelsize=dict_dp['pixelsize'],
                                                    dim=dict_dp['dim'],
                                                    verbose=0,
                                                    weights=weights,
                                                    kf_direction=dict_dp['kf_direction'])
        
            if results is None:
                return np.zeros((3,3)), np.zeros((3,3)), init_residues, final_residues, UBmat
        
            residues, deltamat, newmatrix = FitO.error_function_on_demand_strain(
                                                                                results,
                                                                                Data_Q,
                                                                                allparameters,
                                                                                arr_indexvaryingparameters,
                                                                                sim_indices,
                                                                                pixX,
                                                                                pixY,
                                                                                initrot=starting_orientmatrix,
                                                                                Bmat=B0matrix,
                                                                                pureRotation=0,
                                                                                verbose=1,
                                                                                pixelsize=dict_dp['pixelsize'],
                                                                                dim=dict_dp['dim'],
                                                                                weights=weights,
                                                                                kf_direction=dict_dp['kf_direction'])
            # if np.mean(residues) > final_residues:
            #     return devstrain, deviatoricstrain_sampleframe, init_residues, final_residues, UBmat
            final_mean_residues = np.copy(np.mean(residues))
            final_residues = np.copy(final_mean_residues)
            # building B mat
            # param_strain_sol = results
            # varyingstrain = np.array([[1.0, param_strain_sol[2], param_strain_sol[3]],
            #                                 [0, param_strain_sol[0], param_strain_sol[4]],
            #                                 [0, 0, param_strain_sol[1]]])
            # newUmat = np.dot(deltamat, starting_orientmatrix)
            # newUBmat = np.dot(newUmat, varyingstrain)
            newUBmat = np.copy(newmatrix) 
            # Bstar_s = np.dot(newUBmat, B0matrix)
            # ---------------------------------------------------------------
            # postprocessing of unit cell orientation and strain refinement
            # ---------------------------------------------------------------
            UBmat = np.copy(newmatrix) 
            (devstrain, lattice_parameter_direct_strain) = CP.compute_deviatoricstrain(newUBmat, B0matrix, latticeparams)
            # overwrite and rescale possibly lattice lengthes
            # constantlength = "a"
            # lattice_parameter_direct_strain = CP.computeLatticeParameters_from_UB(newUBmat, material_, constantlength, dictmaterials=dictLT.dict_Materials)
            # print(lattice_parameter_direct_strain)
            deviatoricstrain_sampleframe = CP.strain_from_crystal_to_sample_frame2(devstrain, newUBmat)
            # in % already
            devstrain = np.round(devstrain * 100, decimals=3)
            deviatoricstrain_sampleframe = np.round(deviatoricstrain_sampleframe * 100, decimals=3)
    else:
        # starting B0matrix corresponding to the unit cell   -----
        B0matrix = np.copy(B_matrix)
        latticeparams = dictLT.dict_Materials[material_][1]
        ## Included simple multi level refinement of strains
        init_residues = -0.1
        final_residues = -0.1
        
        if input_params["mat"] == 1:
            straintolerance = input_params["tolerancestrain"]
        elif input_params["mat"] == 2:
            straintolerance = input_params["tolerancestrain1"]
        
        devstrain, deviatoricstrain_sampleframe = np.zeros((3,3)), np.zeros((3,3))
        for ijk, AngTol in enumerate(straintolerance):
            #### Spots in first match (no refining, just simple auto links to filter spots)        
            grain = CP.Prepare_Grain(material_, UBmat, dictmaterials=dictLT.dict_Materials)
            Twicetheta, Chi, Miller_ind, posx, posy, _ = LT.SimulateLaue(grain,
                                                                     input_params["emin"], 
                                                                     input_params["emax"], 
                                                                     detectorparameters,
                                                                     kf_direction=dict_dp['kf_direction'],
                                                                     removeharmonics=1,
                                                                     pixelsize=dict_dp['pixelsize'],
                                                                     dim=dict_dp['dim'],
                                                                     ResolutionAngstrom=False,
                                                                     detectordiameter=dict_dp['detectordiameter'],
                                                                     dictmaterials=dictLT.dict_Materials)
            ## get proximity for exp and theo spots
            linkedspots_link, linkExpMiller_link, \
                linkResidues_link = getProximityv0(np.array([Twicetheta, Chi]),  # warning array(2theta, chi)
                                                            s_tth/2.0, s_chi, Miller_ind,  # warning theta, chi for exp
                                                            angtol=float(AngTol))
            
            if len(linkedspots_link) < 8:
                return np.zeros((3,3)), np.zeros((3,3)), init_residues, final_residues, UBmat
            
            linkedspots_fit = linkedspots_link
            linkExpMiller_fit = linkExpMiller_link
            
            arraycouples = np.array(linkedspots_fit)
            exp_indices = np.array(arraycouples[:, 0], dtype=np.int)
            sim_indices = np.array(arraycouples[:, 1], dtype=np.int)
        
            nb_pairs = len(exp_indices)
            Data_Q = np.array(linkExpMiller_fit)[:, 1:]
            sim_indices = np.arange(nb_pairs)  # for fitting function this must be an arange...
        
            pixX = np.take(dict_dp['peakX'], exp_indices)
            pixY = np.take(dict_dp['peakY'], exp_indices)
            weights = None #np.take(dict_dp['intensity'], exp_indices)
            
            starting_orientmatrix = np.copy(UBmat)
        
            results = None
            # ----------------------------------
            #  refinement model
            # ----------------------------------
            # -------------------------------------------------------
            allparameters = np.array(detectorparameters + [0, 0, 0] + latticeparams)
            
            fitting_parameters_keys = ["anglex", "angley", "anglez"]
            fitting_parameters_values =  [0, 0, 0]
            constantlength = "a"
            if ("a" in strain_free_parameters) and ("b" in strain_free_parameters) and ("c" in strain_free_parameters):
                constantlength = "a"                    
            elif ("b" not in strain_free_parameters) and additional_expression[0]=="none" and\
                "b" not in additional_expression[0]:
                constantlength = "b"
            elif ("c" not in strain_free_parameters):
                constantlength = "c"
            
            for jjkk in strain_free_parameters:
                if jjkk == "a" and constantlength != "a":
                    fitting_parameters_keys.append("a")
                    fitting_parameters_values.append(latticeparams[0])
                if jjkk == "b" and constantlength != "b":
                    fitting_parameters_keys.append("b")
                    fitting_parameters_values.append(latticeparams[1])
                if jjkk == "c" and constantlength != "c":
                    fitting_parameters_keys.append("c")
                    fitting_parameters_values.append(latticeparams[2])
                if jjkk == "alpha":
                    fitting_parameters_keys.append("alpha")
                    fitting_parameters_values.append(latticeparams[3])
                if jjkk == "beta":
                    fitting_parameters_keys.append("beta")
                    fitting_parameters_values.append(latticeparams[4])
                if jjkk == "gamma":
                    fitting_parameters_keys.append("gamma")
                    fitting_parameters_values.append(latticeparams[5])
                    
            pureUmatrix, _ = GT.UBdecomposition_RRPP(starting_orientmatrix)
            absolutespotsindices = np.arange(len(pixX))
            
            (residues, _, _,
                _,  _, ) = FitO.error_function_latticeparameters(fitting_parameters_values,
                                                                fitting_parameters_keys,
                                                                Data_Q,
                                                                allparameters,
                                                                absolutespotsindices,
                                                                pixX,
                                                                pixY,
                                                                initrot=pureUmatrix,
                                                                pureRotation=0,
                                                                verbose=0,
                                                                pixelsize=dict_dp['pixelsize'],
                                                                dim=dict_dp['dim'],
                                                                weights=weights,
                                                                kf_direction=dict_dp['kf_direction'],
                                                                returnalldata=True,
                                                                additional_expression = additional_expression[0])
            init_mean_residues = np.copy(np.mean(residues))
            if ijk == 0:
                init_residues = np.copy(init_mean_residues)
                
            results = FitO.fit_function_latticeparameters(fitting_parameters_values,
                                                            fitting_parameters_keys,
                                                            Data_Q,
                                                            allparameters,
                                                            absolutespotsindices,
                                                            pixX,
                                                            pixY,
                                                            UBmatrix_start=pureUmatrix,
                                                            nb_grains=1,
                                                            pureRotation=0,
                                                            verbose=0,
                                                            pixelsize=dict_dp['pixelsize'],
                                                            dim=dict_dp['dim'],
                                                            weights=weights,
                                                            kf_direction=dict_dp['kf_direction'],
                                                            additional_expression = additional_expression[0])
            if results is None:
                return np.zeros((3,3)), np.zeros((3,3)), init_residues, final_residues, UBmat
            
            (residues, Uxyz, newUmat,
                newB0matrix,  _, ) = FitO.error_function_latticeparameters(results,
                                                                fitting_parameters_keys,
                                                                Data_Q,
                                                                allparameters,
                                                                absolutespotsindices,
                                                                pixX,
                                                                pixY,
                                                                initrot=pureUmatrix,
                                                                pureRotation=0,
                                                                verbose=0,
                                                                pixelsize=dict_dp['pixelsize'],
                                                                dim=dict_dp['dim'],
                                                                weights=weights,
                                                                kf_direction=dict_dp['kf_direction'],
                                                                returnalldata=True,
                                                                additional_expression = additional_expression[0])
            final_mean_residues = np.copy(np.mean(residues))
            final_residues = np.copy(final_mean_residues)
            newUBmat = np.dot(np.dot(newUmat, newB0matrix), np.linalg.inv(B0matrix))
            UBmat = np.copy(newUBmat) 
            # ---------------------------------------------------------------
            # postprocessing of unit cell orientation and strain refinement
            # ---------------------------------------------------------------
            (devstrain, lattice_parameter_direct_strain) = CP.compute_deviatoricstrain(newUBmat, B0matrix, latticeparams)
            deviatoricstrain_sampleframe = CP.strain_from_crystal_to_sample_frame2(devstrain, newUBmat)
            # in % already
            devstrain = np.round(devstrain * 100, decimals=3)
            deviatoricstrain_sampleframe = np.round(deviatoricstrain_sampleframe * 100, decimals=3)
    return devstrain, deviatoricstrain_sampleframe, init_residues, final_residues, UBmat

def getProximityv0(TwicethetaChi, data_theta, data_chi, data_hkl, angtol=0.5):
    # theo simul data
    theodata = np.array([TwicethetaChi[0] / 2.0, TwicethetaChi[1]]).T
    # exp data
    sorted_data = np.array([data_theta, data_chi]).T
    table_dist = GT.calculdist_from_thetachi(sorted_data, theodata)

    prox_table = np.argmin(table_dist, axis=1)
    allresidues = np.amin(table_dist, axis=1)
    very_close_ind = np.where(allresidues < angtol)[0]
    List_Exp_spot_close = []
    Miller_Exp_spot = []
    if len(very_close_ind) > 0:
        for theospot_ind in very_close_ind:  # loop over theo spots index
            List_Exp_spot_close.append(prox_table[theospot_ind])
            Miller_Exp_spot.append(data_hkl[theospot_ind])
    else:
        return [],[],[]
    # removing exp spot which appears many times(close to several simulated spots of one grain)--------------
    arrayLESC = np.array(List_Exp_spot_close, dtype=float)
    sorted_LESC = np.sort(arrayLESC)
    diff_index = sorted_LESC - np.array(list(sorted_LESC[1:]) + [sorted_LESC[0]])
    toremoveindex = np.where(diff_index == 0)[0]
    if len(toremoveindex) > 0:
        # index of exp spot in arrayLESC that are duplicated
        ambiguous_exp_ind = GT.find_closest(np.array(sorted_LESC[toremoveindex], dtype=float), arrayLESC, 0.1)[1]
        for ind in ambiguous_exp_ind:
            Miller_Exp_spot[ind] = None
    
    ProxTablecopy = np.copy(prox_table)

    for theo_ind, exp_ind in enumerate(prox_table):
        where_th_ind = np.where(ProxTablecopy == exp_ind)[0]
        if len(where_th_ind) > 1:
            for indy in where_th_ind:
                ProxTablecopy[indy] = -prox_table[indy]
            closest = np.argmin(allresidues[where_th_ind])
            ProxTablecopy[where_th_ind[closest]] = -ProxTablecopy[where_th_ind[closest]]
    
    singleindices = []
    refine_indexed_spots = {}
    # loop over close exp. spots
    for k in range(len(List_Exp_spot_close)):
        exp_index = List_Exp_spot_close[k]
        if not singleindices.count(exp_index):
            singleindices.append(exp_index)
            theo_index = np.where(ProxTablecopy == exp_index)[0]
            if (len(theo_index) == 1):  # only one theo spot close to the current exp. spot
                refine_indexed_spots[exp_index] = [exp_index, theo_index, Miller_Exp_spot[k]]
            else:  # recent PATCH:
                closest_theo_ind = np.argmin(allresidues[theo_index])
                if allresidues[theo_index][closest_theo_ind] < angtol:
                    refine_indexed_spots[exp_index] = [exp_index, theo_index[closest_theo_ind], Miller_Exp_spot[k]]
    
    listofpairs = []
    linkExpMiller = []
    linkResidues = []
    
    selectedAbsoluteSpotIndices = np.arange(len(data_theta))
    for val in list(refine_indexed_spots.values()):
        if val[2] is not None:
            localspotindex = val[0]
            if not isinstance(val[1], (list, np.ndarray)):
                closetheoindex = val[1]
            else:
                closetheoindex = val[1][0]
            absolute_spot_index = selectedAbsoluteSpotIndices[localspotindex]
            listofpairs.append([absolute_spot_index, closetheoindex])  # Exp, Theo,  where -1 for specifying that it came from automatic linking
            linkExpMiller.append([float(absolute_spot_index)] + [float(elem) for elem in val[2]])  # float(val) for further handling as floats array
            linkResidues.append([absolute_spot_index, closetheoindex, allresidues[closetheoindex]])

    linkedspots_link = np.array(listofpairs)
    linkExpMiller_link = linkExpMiller
    linkResidues_link = linkResidues
    return linkedspots_link, linkExpMiller_link, linkResidues_link

def get_ipf_colour(orientation_matrix1, axis=np.array([0., 0., 1.]), symmetry=None, symm_operator=None):
    """Compute the IPF (inverse pole figure) colour for this orientation.
    Given a particular axis expressed in the laboratory coordinate system,
    one can compute the so called IPF colour based on that direction
    expressed in the crystal coordinate system as :math:`[x_c,y_c,z_c]`.
    There is only one tuple (u,v,w) such that:
    .. math::
      [x_c,y_c,z_c]=u.[0,0,1]+v.[0,1,1]+w.[1,1,1]
    and it is used to assign the RGB colour.
    :param ndarray axis: the direction to use to compute the IPF colour.
    :param Symmetry symmetry: the symmetry operator to use.
    :return tuple: a tuple contining the RGB values.
    """
    if not np.all(orientation_matrix1==0):
        orientation_matrix = orientation_matrix1
    else:
        return 0,0,0
    # ## rotate orientation by 40degrees to bring in Sample RF
    omega = np.deg2rad(-40.0)
    # rotation de -omega autour de l'axe x (or Y?) pour repasser dans Rsample
    cw = np.cos(omega)
    sw = np.sin(omega)
    mat_from_lab_to_sample_frame = np.array([[cw, 0.0, sw], [0.0, 1.0, 0.0], [-sw, 0, cw]])
    orientation_matrix = np.dot(mat_from_lab_to_sample_frame.T, orientation_matrix)
    if np.linalg.det(orientation_matrix) < 0:
        orientation_matrix = -orientation_matrix
    axis /= np.linalg.norm(axis)
    
    # rgb = get_field_color(orientation_matrix, axis, symmetry=symmetry, syms=syms)            
    # return rgb

    Vc = np.dot(orientation_matrix, axis)
    # get the symmetry operators
    syms = np.array(symm_operator) #symmetry.symmetry_operators()
    syms = np.concatenate((syms, -syms))
    syms = np.unique(syms, axis=0)
    
    if symmetry == symmetry.cubic:
        rgb = get_field_color(orientation_matrix, axis, symmetry, syms)            
        return rgb
        # angleR = 45 - Vc_chi  # red color proportional to (45 - chi)
        # minAngleR = 0
        # maxAngleR = 45
        # angleB = Vc_phi  # blue color proportional to phi
        # minAngleB = 0
        # maxAngleB = 45
    elif symmetry == symmetry.hexagonal:
        Vc_syms = np.dot(syms, Vc)
        # phi: rotation around 001 axis, from 100 axis to Vc vector, projected on (100,010) plane
        Vc_phi = np.arctan2(Vc_syms[:, 1], Vc_syms[:, 0]) * 180 / np.pi
        # chi: rotation around 010 axis, from 001 axis to Vc vector, projected on (100,001) plane
        # Vc_chi = np.arctan2(Vc_syms[:, 0], Vc_syms[:, 2]) * 180 / np.pi
        # psi : angle from 001 axis to Vc vector
        Vc_psi = np.arccos(Vc_syms[:, 2]) * 180 / np.pi
        
        angleR = 90 - Vc_psi  # red color proportional to (90 - psi)
        minAngleR = 0
        maxAngleR = 90
        angleB = Vc_phi  # blue color proportional to phi
        minAngleB = 0
        maxAngleB = 30
    else:
        rgb = get_field_color(orientation_matrix, axis, symmetry, syms)            
        return rgb
    # find the axis lying in the fundamental zone
    fz_list = ((angleR >= minAngleR) & (angleR < maxAngleR) &
                (angleB >= minAngleB) & (angleB < maxAngleB)).tolist()
    if not fz_list.count(True) == 1:
        # print("funda problem")
        rgb = get_field_color(orientation_matrix, axis, symmetry, syms)            
        return rgb
    i_SST = fz_list.index(True)
    r = angleR[i_SST] / maxAngleR
    g = (maxAngleR - angleR[i_SST]) / maxAngleR * (maxAngleB - angleB[i_SST]) / maxAngleB
    b = (maxAngleR - angleR[i_SST]) / maxAngleR * angleB[i_SST] / maxAngleB
    rgb = np.array([r, g, b])
    rgb = rgb / rgb.max()
    return rgb 

def get_field_color(orientation_matrix, axis=np.array([0., 0., 1.]), symmetry=None, syms=None):
    """Compute the IPF (inverse pole figure) colour for this orientation.
    Given a particular axis expressed in the laboratory coordinate system,
    one can compute the so called IPF colour based on that direction
    expressed in the crystal coordinate system as :math:`[x_c,y_c,z_c]`.
    There is only one tuple (u,v,w) such that:
    .. math::
      [x_c,y_c,z_c]=u.[0,0,1]+v.[0,1,1]+w.[1,1,1]
    and it is used to assign the RGB colour.
    :param ndarray axis: the direction to use to compute the IPF colour.
    :param Symmetry symmetry: the symmetry operator to use.
    :return tuple: a tuple contining the RGB values.
    """
    for sym in syms:
        Osym = np.dot(sym, orientation_matrix)
        Vc = np.dot(Osym, axis)
        if Vc[2] < 0:
            Vc *= -1.  # using the upward direction
        uvw = np.array([Vc[2] - Vc[1], Vc[1] - Vc[0], Vc[0]])
        uvw /= np.linalg.norm(uvw)
        uvw /= max(uvw)
        if (uvw[0] >= 0. and uvw[0] <= 1.0) and (uvw[1] >= 0. and uvw[1] <= 1.0) and (
                uvw[2] >= 0. and uvw[2] <= 1.0):
            break
    uvw = uvw / uvw.max()
    return uvw

class Symmetry(enum.Enum):
    """
    Class to describe crystal symmetry defined by its Laue class symbol.
    # Laue Groups
    #group 1 -- triclinic: '-1'
    #group 2 -- monoclinic: '2/m'
    #group 3 -- orthorhombic: 'mmm'
    #group 4 -- tetragonal: '4/m'
    #group 5 -- tetragonal: '4/mmm'
    #group 6 -- trigonal: '-3'
    #group 7 -- trigonal: '-3m'
    #group 8 -- hexagonal: '6/m'
    #group 9 -- hexagonal: '6/mmm'
    #group 10 -- cubic: 'm3'
    #group 11 -- cubic: 'm3m'
    """
    cubic = 'm3m'
    hexagonal = '6/mmm'
    orthorhombic = 'mmm'
    tetragonal = '4/mmm'
    trigonal = 'bar3m'
    monoclinic = '2/m'
    triclinic = 'bar1'
    # operation_rotation = None
    
    def symmetry_operators(self, use_miller_bravais=False):
        """Define the equivalent crystal symmetries.
        Those come from Randle & Engler, 2000. For instance in the cubic
        crystal struture, for instance there are 24 equivalent cube orientations.
        :returns array: A numpy array of shape (n, 3, 3) where n is the \
        number of symmetries of the given crystal structure.
        """
        if self is Symmetry.cubic:
            #m-3 only 24 component
            #m-3m 48 component
            sym = np.zeros((48, 3, 3), dtype=np.float)
            sym[0] = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
            sym[1] = np.array([[0., 0., -1.], [0., -1., 0.], [-1., 0., 0.]])
            sym[2] = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
            sym[3] = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
            sym[4] = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
            sym[5] = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
            sym[6] = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
            sym[7] = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
            sym[8] = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
            sym[9] = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
            sym[10] = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
            sym[11] = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
            sym[12] = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
            sym[13] = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
            sym[14] = np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])
            sym[15] = np.array([[0., 1., 0.], [0., 0., -1.], [-1., 0., 0.]])
            sym[16] = np.array([[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]])
            sym[17] = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
            sym[18] = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
            sym[19] = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
            sym[20] = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
            sym[21] = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])
            sym[22] = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
            sym[23] = np.array([[-1., 0., 0.], [0., 0., -1.], [0., -1., 0.]])
        elif self is Symmetry.hexagonal:
            # using the Miller-Bravais representation here
            if use_miller_bravais:
                sym = np.zeros((12, 4, 4), dtype=np.int)
                sym[0] = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                sym[1] = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
                sym[2] = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
                sym[3] = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
                sym[4] = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1]])
                sym[5] = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, -1]])
                sym[6] = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                sym[7] = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
                sym[8] = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
                sym[9] = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
                sym[10] = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, -1]])
                sym[11] = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, -1]])
            else:
                sym = np.zeros((12, 3, 3), dtype=np.float)
                s60 = np.sin(60 * np.pi / 180)
                sym[0] = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
                sym[1] = np.array([[0.5, s60, 0.], [-s60, 0.5, 0.], [0., 0., 1.]])
                sym[2] = np.array([[-0.5, s60, 0.], [-s60, -0.5, 0.], [0., 0., 1.]])
                sym[3] = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
                sym[4] = np.array([[-0.5, -s60, 0.], [s60, -0.5, 0.], [0., 0., 1.]])
                sym[5] = np.array([[0.5, -s60, 0.], [s60, 0.5, 0.], [0., 0., 1.]])
                sym[6] = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
                sym[7] = np.array([[0.5, s60, 0.], [s60, -0.5, 0.], [0., 0., -1.]])
                sym[8] = np.array([[-0.5, s60, 0.], [s60, 0.5, 0.], [0., 0., -1.]])
                sym[9] = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
                sym[10] = np.array([[-0.5, -s60, 0.], [-s60, 0.5, 0.], [0., 0., -1.]])
                sym[11] = np.array([[0.5, -s60, 0.], [-s60, -0.5, 0.], [0., 0., -1.]])
        elif self is Symmetry.orthorhombic:
            sym = np.zeros((8, 3, 3), dtype=np.float)
            sym[0] = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
            sym[1] = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
            sym[2] = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
            sym[3] = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
            sym[4] = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
            sym[5] = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
            sym[6] = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
            sym[7] = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
        elif self is Symmetry.tetragonal:
            sym = np.zeros((8, 3, 3), dtype=np.float)
            sym[0] = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
            sym[1] = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
            sym[2] = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
            sym[3] = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
            sym[4] = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
            sym[5] = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
            sym[6] = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
            sym[7] = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
        elif self is Symmetry.monoclinic:
            sym = np.zeros((4, 3, 3), dtype=np.float)
            sym[0] = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
            sym[1] = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
            sym[2] = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
            sym[3] = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
        elif self is Symmetry.triclinic:
            sym = np.zeros((2, 3, 3), dtype=np.float)
            sym[0] = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
            sym[1] = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
        else:
            raise ValueError('warning, symmetry not supported: %s' % self)
        return sym
    
class Lattice:
    '''
    The Lattice class to create one of the 14 Bravais lattices.
    This particular class has been partly inspired from the pymatgen
    project at https://github.com/materialsproject/pymatgen
    Any of the 7 lattice systems (each corresponding to one point group)
    can be easily created and manipulated.
    The lattice centering can be specified to form any of the 14 Bravais
    lattices:
     * Primitive (P): lattice points on the cell corners only (default);
     * Body (I): one additional lattice point at the center of the cell;
     * Face (F): one additional lattice point at the center of each of
       the faces of the cell;
     * Base (A, B or C): one additional lattice point at the center of
       each of one pair of the cell faces.
    ::
      a = 0.352 # FCC Nickel
      l = Lattice.face_centered_cubic(a)
      print(l.volume())
    Addditionnally the point-basis can be controlled to address non
    Bravais lattice cells. It is set to a single atoms at (0, 0, 0) by
    default so that each cell is a Bravais lattice but may be changed to
    something more complex to achieve HCP structure or Diamond structure
    for instance.
    '''

    def __init__(self, matrix, centering='P', symmetry=None):
        '''Create a crystal lattice (unit cell).
        Create a lattice from a 3x3 matrix.
        Each row in the matrix represents one lattice vector.
        '''
        m = np.array(matrix, dtype=np.float64).reshape((3, 3))
        lengths = np.sqrt(np.sum(m ** 2, axis=1))
        angles = np.zeros(3)
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            angles[i] = dot(m[j], m[k]) / (lengths[j] * lengths[k])
        angles = np.arccos(angles) * 180. / pi
        self._angles = angles
        self._lengths = lengths
        self._matrix = m
        self._centering = centering
        self._symmetry = symmetry
        
    def __eq__(self, other):
        """Override the default Equals behavior.
        The equality of two Lattice objects is based on the equality of their angles, lengths, and centering.
        """
        if not isinstance(other, self.__class__):
            return False
        for i in range(3):
            if self._angles[i] != other._angles[i]:
                return False
            elif self._lengths[i] != other._lengths[i]:
                return False
        if self._centering != other._centering:
            return False
        if self._symmetry != other._symmetry:
            return False
        return True

    def reciprocal_lattice(self):
        '''Compute the reciprocal lattice.
        The reciprocal lattice defines a crystal in terms of vectors that
        are normal to a plane and whose lengths are the inverse of the
        interplanar spacing. This method computes the three reciprocal
        lattice vectors defined by:
        .. math::
         * a.a^* = 1
         * b.b^* = 1
         * c.c^* = 1
        '''
        [a, b, c] = self._matrix
        V = self.volume()
        astar = np.cross(b, c) / V
        bstar = np.cross(c, a) / V
        cstar = np.cross(a, b) / V
        return [astar, bstar, cstar]

    @property
    def matrix(self):
        """Returns a copy of matrix representing the Lattice."""
        return np.copy(self._matrix)

    def get_symmetry(self):
        """Returns the type of `Symmetry` of the Lattice."""
        return self._symmetry

    
    def symmetry(crystal_structure=Symmetry.cubic, use_miller_bravais=False):
        """Define the equivalent crystal symmetries.
        Those come from Randle & Engler, 2000. For instance in the cubic
        crystal struture, for instance there are 24 equivalent cube orientations.
        :param crystal_structure: an instance of the `Symmetry` class describing the crystal symmetry.
        :raise ValueError: if the given symmetry is not supported.
        :returns array: A numpy array of shape (n, 3, 3) where n is the \
        number of symmetries of the given crystal structure.
        """
        return crystal_structure.symmetry_operators(use_miller_bravais=use_miller_bravais)

    @staticmethod
    def cubic(a):
        '''
        Create a cubic Lattice unit cell.
        *Parameters*
        **a**: first lattice length parameter (a = b = c here)
        *Returns*
        A `Lattice` instance corresponding to a primitice cubic lattice.
        '''
        return Lattice([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]], symmetry=Symmetry.cubic)

    @staticmethod
    def body_centered_cubic(a):
        '''
        Create a body centered cubic Lattice unit cell.
        *Parameters*
        **a**: first lattice length parameter (a = b = c here)
        *Returns*
        A `Lattice` instance corresponding to a body centered cubic
        lattice.
        '''
        return Lattice.from_parameters(a, a, a, 90, 90, 90, centering='I', symmetry=Symmetry.cubic)

    @staticmethod
    def face_centered_cubic(a):
        '''
        Create a face centered cubic Lattice unit cell.
        *Parameters*
        **a**: first lattice length parameter (a = b = c here)
        *Returns*
        A `Lattice` instance corresponding to a face centered cubic
        lattice.
        '''
        return Lattice.from_parameters(a, a, a, 90, 90, 90, centering='F', symmetry=Symmetry.cubic)

    @staticmethod
    def tetragonal(a, c):
        '''
        Create a tetragonal Lattice unit cell.
        *Parameters*
        **a**: first lattice length parameter
        **c**: third lattice length parameter (b = a here)
        *Returns*
        A `Lattice` instance corresponding to a primitive tetragonal
        lattice.
        '''
        return Lattice.from_parameters(a, a, c, 90, 90, 90, symmetry=Symmetry.tetragonal)

    @staticmethod
    def body_centered_tetragonal(a, c):
        '''
        Create a body centered tetragonal Lattice unit cell.
        *Parameters*
        **a**: first lattice length parameter
        **c**: third lattice length parameter (b = a here)
        *Returns*
        A `Lattice` instance corresponding to a body centered tetragonal
        lattice.
        '''
        return Lattice.from_parameters(a, a, c, 90, 90, 90, centering='I', symmetry=Symmetry.tetragonal)

    @staticmethod
    def orthorhombic(a, b, c):
        '''
        Create a tetragonal Lattice unit cell with 3 different length
        parameters a, b and c.
        '''
        return Lattice.from_parameters(a, b, c, 90, 90, 90, symmetry=Symmetry.orthorhombic)

    @staticmethod
    def base_centered_orthorhombic(a, b, c):
        '''
        Create a based centered orthorombic Lattice unit cell.
        *Parameters*
        **a**: first lattice length parameter
        **b**: second lattice length parameter
        **c**: third lattice length parameter
        *Returns*
        A `Lattice` instance corresponding to a based centered orthorombic
        lattice.
        '''
        return Lattice.from_parameters(a, b, c, 90, 90, 90, centering='C', symmetry=Symmetry.orthorhombic)

    @staticmethod
    def body_centered_orthorhombic(a, b, c):
        '''
        Create a body centered orthorombic Lattice unit cell.
        *Parameters*
        **a**: first lattice length parameter
        **b**: second lattice length parameter
        **c**: third lattice length parameter
        *Returns*
        A `Lattice` instance corresponding to a body centered orthorombic
        lattice.
        '''
        return Lattice.from_parameters(a, b, c, 90, 90, 90, centering='I', symmetry=Symmetry.orthorhombic)

    @staticmethod
    def face_centered_orthorhombic(a, b, c):
        '''
        Create a face centered orthorombic Lattice unit cell.
        *Parameters*
        **a**: first lattice length parameter
        **b**: second lattice length parameter
        **c**: third lattice length parameter
        *Returns*
        A `Lattice` instance corresponding to a face centered orthorombic
        lattice.
        '''
        return Lattice.from_parameters(a, b, c, 90, 90, 90, centering='F', symmetry=Symmetry.orthorhombic)

    @staticmethod
    def hexagonal(a, c):
        '''
        Create a hexagonal Lattice unit cell with length parameters a and c.
        '''
        return Lattice.from_parameters(a, a, c, 90, 90, 120, symmetry=Symmetry.hexagonal)

    @staticmethod
    def rhombohedral(a, alpha):
        '''
        Create a rhombohedral Lattice unit cell with one length
        parameter a and the angle alpha.
        '''
        return Lattice.from_parameters(a, a, a, alpha, alpha, alpha, symmetry=Symmetry.trigonal)

    @staticmethod
    def monoclinic(a, b, c, alpha):
        '''
        Create a monoclinic Lattice unit cell with 3 different length
        parameters a, b and c. The cell angle is given by alpha.
        The lattice centering id primitive ie. 'P'
        '''
        return Lattice.from_parameters(a, b, c, alpha, 90, 90, symmetry=Symmetry.monoclinic)

    @staticmethod
    def base_centered_monoclinic(a, b, c, alpha):
        '''
        Create a based centered monoclinic Lattice unit cell.
        *Parameters*
        **a**: first lattice length parameter
        **b**: second lattice length parameter
        **c**: third lattice length parameter
        **alpha**: first lattice angle parameter
        *Returns*
        A `Lattice` instance corresponding to a based centered monoclinic
        lattice.
        '''
        return Lattice.from_parameters(a, b, c, alpha, 90, 90, centering='C', symmetry=Symmetry.monoclinic)

    @staticmethod
    def triclinic(a, b, c, alpha, beta, gamma):
        '''
        Create a triclinic Lattice unit cell with 3 different length
        parameters a, b, c and three different cell angles alpha, beta
        and gamma.
        ..note::
           This method is here for the sake of completeness since one can
           create the triclinic cell directly using the `from_parameters`
           method.
        '''
        return Lattice.from_parameters(a, b, c, alpha, beta, gamma, symmetry=Symmetry.triclinic)
    
    @staticmethod
    def from_parameters(a, b, c, alpha, beta, gamma, x_aligned_with_a=False, centering='P', symmetry=Symmetry.triclinic):
        """
        Create a Lattice using unit cell lengths and angles (in degrees).
        The lattice centering can also be specified (among 'P', 'I', 'F',
        'A', 'B' or 'C').
        :param float a: first lattice length parameter.
        :param float b: second lattice length parameter.
        :param float c: third lattice length parameter.
        :param float alpha: first lattice angle parameter.
        :param float beta: second lattice angle parameter.
        :param float gamma: third lattice angle parameter.
        :param bool x_aligned_with_a: flag to control the convention used to define the Cartesian frame.
        :param str centering: lattice centering ('P' by default) passed to the `Lattice` class.
        :param symmetry: a `Symmetry` instance to be passed to the lattice.
        :return: A `Lattice` instance with the specified lattice parameters and centering.
        """
        alpha_r = radians(alpha)
        beta_r = radians(beta)
        gamma_r = radians(gamma)
        if x_aligned_with_a:  # first lattice vector (a) is aligned with X
            vector_a = a * np.array([1, 0, 0])
            vector_b = b * np.array([np.cos(gamma_r), np.sin(gamma_r), 0])
            c1 = c * np.cos(beta_r)
            c2 = c * (np.cos(alpha_r) - np.cos(gamma_r) * np.cos(beta_r)) / np.sin(gamma_r)
            vector_c = np.array([c1, c2, np.sqrt(c ** 2 - c1 ** 2 - c2 ** 2)])
        else:  # third lattice vector (c) is aligned with Z
            cos_gamma_star = (np.cos(alpha_r) * np.cos(beta_r) - np.cos(gamma_r)) / (np.sin(alpha_r) * np.sin(beta_r))
            sin_gamma_star = np.sqrt(1 - cos_gamma_star ** 2)
            vector_a = [a * np.sin(beta_r), 0.0, a * np.cos(beta_r)]
            vector_b = [-b * np.sin(alpha_r) * cos_gamma_star, b * np.sin(alpha_r) * sin_gamma_star, b * np.cos(alpha_r)]
            vector_c = [0.0, 0.0, float(c)]
        return Lattice([vector_a, vector_b, vector_c], centering=centering, symmetry=symmetry)

    def volume(self):
        """Compute the volume of the unit cell."""
        m = self._matrix
        return abs(np.dot(np.cross(m[0], m[1]), m[2]))

    def get_hkl_family(self, hkl):
        """Get a list of the hkl planes composing the given family for
        this crystal lattice.
        *Parameters*
        **hkl**: miller indices of the requested family
        *Returns*
        A list of the hkl planes in the given family.
        """
        planes = HklPlane.get_family(hkl, lattice=self, crystal_structure=self._symmetry)
        return planes

class HklObject:
    def __init__(self, h, k, l, lattice=None):
        '''Create a new hkl object with the given Miller indices and
           crystal lattice.
        '''
        if lattice == None:
            lattice = Lattice.cubic(1.0)
        self._lattice = lattice
        self._h = h
        self._k = k
        self._l = l

    @property
    def lattice(self):
        return self._lattice

    def set_lattice(self, lattice):
        """Assign a new `Lattice` to this instance.
        :param lattice: the new crystal lattice.
        """
        self._lattice = lattice

    @property
    def h(self):
        return self._h

    @property
    def k(self):
        return self._k

    @property
    def l(self):
        return self._l

    def miller_indices(self):
        '''
        Returns an immutable tuple of the plane Miller indices.
        '''
        return (self._h, self._k, self._l)

class HklDirection(HklObject):
    def direction(self):
        '''Returns a normalized vector, expressed in the cartesian
        coordinate system, corresponding to this crystallographic direction.
        '''
        (h, k, l) = self.miller_indices()
        M = self._lattice.matrix.T  # the columns of M are the a, b, c vector in the cartesian coordinate system
        l_vect = M.dot(np.array([h, k, l]))
        return l_vect / np.linalg.norm(l_vect)

    def angle_with_direction(self, hkl):
        '''Computes the angle between this crystallographic direction and
        the given direction (in radian).'''
        return np.arccos(np.dot(self.direction(), hkl.direction()))

    @staticmethod
    def angle_between_directions(hkl1, hkl2, lattice=None):
        '''Computes the angle between two crystallographic directions (in radian).
        :param tuple hkl1: The triplet of the miller indices of the first direction.
        :param tuple hkl2: The triplet of the miller indices of the second direction.
        :param Lattice lattice: The crystal lattice, will default to cubic if not specified.
        :returns float: The angle in radian.
        '''
        d1 = HklDirection(*hkl1, lattice=lattice)
        d2 = HklDirection(*hkl2, lattice=lattice)
        return d1.angle_with_direction(d2)

    @staticmethod
    def three_to_four_indices(u, v, w):
        """Convert from Miller indices to Miller-Bravais indices. this is used for hexagonal crystal lattice."""
        return (2 * u - v) / 3., (2 * v - u) / 3., -(u + v) / 3., w

    @staticmethod
    def four_to_three_indices(U, V, T, W):
        """Convert from Miller-Bravais indices to Miller indices. this is used for hexagonal crystal lattice."""
        u, v, w = U - T, V - T, W
        gcd = functools.reduce(math.gcd, (u, v, w))
        return u / gcd, v / gcd, w / gcd

    @staticmethod
    def angle_between_4indices_directions(hkil1, hkil2, ac):
        """Computes the angle between two crystallographic directions in a hexagonal lattice.
        The solution was derived by F. Frank in:
        On Miller - Bravais indices and four dimensional vectors. Acta Cryst. 18, 862-866 (1965)
        :param tuple hkil1: The quartet of the indices of the first direction.
        :param tuple hkil2: The quartet of the indices of the second direction.
        :param tuple ac: the lattice parameters of the hexagonal structure in the form (a, c).
        :returns float: The angle in radian.
        """
        h1, k1, i1, l1 = hkil1
        h2, k2, i2, l20 = hkil2
        a, c = ac
        lambda_square = 2. / 3 * (c / a) ** 2
        value = (h1 * h2 + k1 * k2 + i1 * i2 + lambda_square * l1 * l20) / \
                (np.sqrt(h1 ** 2 + k1 ** 2 + i1 ** 2 + lambda_square * l1 ** 2) *
                 np.sqrt(h2 ** 2 + k2 ** 2 + i2 ** 2 + lambda_square * l20 ** 2))
        return np.arccos(value)

class HklPlane(HklObject):
    '''
    This class define crystallographic planes using Miller indices.
    A plane can be create by speficying its Miller indices and the
    crystal lattice (default is cubic with lattice parameter of 1.0)
    ::
      a = 0.405 # FCC Aluminium
      l = Lattice.cubic(a)
      p = HklPlane(1, 1, 1, lattice=l)
      print(p)
      print(p.scattering_vector())
      print(p.interplanar_spacing())
    .. note::
      Miller indices are defined in terms of the inverse of the intercept
      of the plane on the three crystal axes a, b, and c.
    '''

    def __eq__(self, other):
        """Override the default Equals behavior.
        The equality of two HklObjects is based on the equality of their miller indices.
        """
        if isinstance(other, self.__class__):
            return self._h == other._h and self._k == other._k and \
                   self._l == other._l and self._lattice == other._lattice
        return False

    def __ne__(self, other):
        """Define a non-equality test"""
        return not self.__eq__(other)

    def normal(self):
        '''Returns the unit vector normal to the plane.
        We use of the repiprocal lattice to compute the normal to the plane
        and return a normalised vector.
        '''
        n = self.scattering_vector()
        return n / np.linalg.norm(n)

    def scattering_vector(self):
        '''Calculate the scattering vector of this `HklPlane`.
        The scattering vector (or reciprocal lattice vector) is normal to
        this `HklPlane` and its length is equal to the inverse of the
        interplanar spacing. In the cartesian coordinate system of the
        crystal, it is given by:
        ..math
          G_c = h.a^* + k.b^* + l.c^*
        :returns: a numpy vector expressed in the cartesian coordinate system of the crystal.
        '''
        [astar, bstar, cstar] = self._lattice.reciprocal_lattice()
        (h, k, l) = self.miller_indices()
        # express (h, k, l) in the cartesian crystal CS
        Gc = h * astar + k * bstar + l * cstar
        return Gc

    def friedel_pair(self):
        """Create the Friedel pair of the HklPlane."""
        (h, k, l) = self.miller_indices()
        pair = HklPlane(-h, -k, -l, self._lattice)
        return pair

    def interplanar_spacing(self):
        '''
        Compute the interplanar spacing.
        For cubic lattice, it is:
        .. math::
           d = a / \sqrt{h^2 + k^2 + l^2}
        The general formula comes from 'Introduction to Crystallography'
        p. 68 by Donald E. Sands.
        '''
        (a, b, c) = self._lattice._lengths
        (h, k, l) = self.miller_indices()
        (alpha, beta, gamma) = radians(self._lattice._angles)
        # d = a / np.sqrt(h**2 + k**2 + l**2) # for cubic structure only
        d = self._lattice.volume() / np.sqrt(h ** 2 * b ** 2 * c ** 2 * np.sin(alpha) ** 2 + \
                                             k ** 2 * a ** 2 * c ** 2 * np.sin(
                                                 beta) ** 2 + l ** 2 * a ** 2 * b ** 2 * np.sin(gamma) ** 2 + \
                                             2 * h * l * a * b ** 2 * c * (
                                                 np.cos(alpha) * np.cos(gamma) - np.cos(beta)) + \
                                             2 * h * k * a * b * c ** 2 * (
                                                 np.cos(alpha) * np.cos(beta) - np.cos(gamma)) + \
                                             2 * k * l * a ** 2 * b * c * (
                                                 np.cos(beta) * np.cos(gamma) - np.cos(alpha)))
        return d

    @staticmethod
    def four_to_three_indices(U, V, T, W):
        """Convert four to three index representation of a slip plane (used for hexagonal crystal lattice)."""
        return U, V, W

    @staticmethod
    def three_to_four_indices(u, v, w):
        """Convert three to four index representation of a slip plane (used for hexagonal crystal lattice)."""
        return u, v, -(u + v), w

    def is_in_list(self, hkl_planes, friedel_pair=False):
        """Check if the hkl plane is in the given list.
        By default this relies on the built in in test from the list type which in turn calls in the __eq__ method.
        This means it will return True if a plane with the exact same miller indices (and same lattice) is in the list.
        Turning on the friedel_pair flag will allow to test also the Friedel pair (-h, -k, -l) and return True if it is
        in the list.
        For instance (0,0,1) and (0,0,-1) are in general considered as the same lattice plane.
        """
        if not friedel_pair:
            return self in hkl_planes
        else:
            return self in hkl_planes or self.friedel_pair() in hkl_planes

    @staticmethod
    def is_same_family(hkl1, hkl2, crystal_structure=Symmetry.cubic):
        """Static mtd to test if both lattice planes belongs to same family.
        A family {hkl} is composed by all planes that are equivalent to (hkl)
        using the symmetry of the lattice. The lattice assoiated with `hkl2`
        is not taken into account here.
        """
        return hkl1.is_in_list(HklPlane.get_family(hkl2.miller_indices(), lattice=hkl1._lattice,
                                                   crystal_structure=crystal_structure))

    @staticmethod
    def get_family(hkl, lattice=None, include_friedel_pairs=False, crystal_structure=Symmetry.cubic):
        """Static method to obtain a list of the different crystallographic
        planes in a particular family.
        :param str hkl: a sequence of 3 (4 for hexagonal) numbers corresponding to the miller indices.
        :param Lattice lattice: The reference crystal lattice (default None).
        :param bool include_friedel_pairs: Flag to include the Friedel pairs in the list (False by default).
        :param str crystal_structure: A string descibing the crystal structure (cubic by default).
        :raise ValueError: if the given string does not correspond to a supported family.
        :returns list: a list of the :py:class:`~HklPlane` in the given hkl family.
        .. note::
          The method account for the lattice symmetry to create a list of equivalent lattice plane from the point
          of view of the point group symmetry. A flag can be used to include or not the Friedel pairs. If not, the
          family is contstructed using the miller indices limited the number of minus signs. For instance  (1,0,0)
          will be in the list and not (-1,0,0).
        """
        if not (len(hkl) == 3 or (len(hkl) == 4 and crystal_structure == Symmetry.hexagonal)):
            raise ValueError('warning, family not supported: {}'.format(hkl))
        # handle hexagonal case
        if len(hkl) == 4:
            h = int(hkl[0])
            k = int(hkl[1])
            i = int(hkl[2])
            l = int(hkl[3])
            (h, k, l) = HklPlane.four_to_three_indices(h, k, i, l)  # useless as it just drops i
        else:  # 3 indices
            h = int(hkl[0])
            k = int(hkl[1])
            l = int(hkl[2])
            if crystal_structure == Symmetry.hexagonal:
                i = -(h + k)
        family = []
        # construct lattice plane family from the symmetry operators
        if crystal_structure == Symmetry.hexagonal:
          syms = Lattice.symmetry(crystal_structure, use_miller_bravais=True)
        else:
          syms = Lattice.symmetry(crystal_structure)
        for sym in syms:
            if crystal_structure == Symmetry.hexagonal:
                n_sym = np.dot(sym, np.array([h, k, i, l]))
                n_sym = HklPlane.four_to_three_indices(*n_sym)
            else:  # 3 indices
                n_sym = np.dot(sym, np.array([h, k, l]))
            hkl_sym = HklPlane(*n_sym, lattice=lattice)
            if not hkl_sym.is_in_list(family, friedel_pair=True):
                family.append(hkl_sym)
            if include_friedel_pairs:
                hkl_sym = HklPlane(-n_sym[0], -n_sym[1], -n_sym[2], lattice=lattice)
                if not hkl_sym.is_in_list(family, friedel_pair=False):
                    family.append(hkl_sym)
        if not include_friedel_pairs:
            # for each hkl plane chose between (h, k, l) and (-h, -k, -l) to have the less minus signs
            for i in range(len(family)):
                hkl = family[i]
                (h, k, l) = hkl.miller_indices()
                if np.where(np.array([h, k, l]) < 0)[0].size > 0 and np.where(np.array([h, k, l]) <= 0)[0].size >= 2:
                    family[i] = hkl.friedel_pair()
                    #print('replacing plane (%d%d%d) by its pair: (%d%d%d)' % (h, k, l, -h, -k, -l))
        return family

    def multiplicity(self, symmetry=Symmetry.cubic):
        """compute the general multiplicity for this `HklPlane` and the given `Symmetry`.
        :param Symmetry symmetry: The crystal symmetry to take into account.
        :return: the number of equivalent planes in the family.
        """
        return len(HklPlane.get_family(self.miller_indices(), include_friedel_pairs=True, crystal_structure=symmetry))        

class PoleFigure:
    """A class to handle pole figures.

    A pole figure is a popular tool to plot multiple crystal orientations,
    either in the sample coordinate system (direct pole figure) or
    alternatively plotting a particular direction in the crystal
    coordinate system (inverse pole figure).
    """
    def __init__(self, lattice=None, axis='Z', hkl='111', proj='stereo'):
        """
        Create an empty PoleFigure object associated with an empty Microstructure.
        :param microstructure: the :py:class:`~pymicro.crystal.microstructure.Microstructure` containing the collection of orientations to plot (None by default).
        :param lattice: the crystal :py:class:`~pymicro.crystal.lattice.Lattice`.
        :param str axis: the pole figure axis ('Z' by default), vertical axis in the direct pole figure and direction plotted on the inverse pole figure.
        .. warning::
           Any crystal structure is now supported (you have to set the proper
           crystal lattice) but it has only really be tested for cubic.
        :param str hkl: slip plane family ('111' by default)
        :param str proj: projection type, can be either 'stereo' (default) or 'flat'
        """
        self.proj = proj
        self.axis = axis
        
        if self.axis == 'Z':
            self.axis_crystal = np.array([0, 0, 1])
        elif self.axis == 'Y':
            self.axis_crystal = np.array([0, 1, 0])
        else:
            self.axis_crystal = np.array([1, 0, 0])

        if lattice:
            self.lattice = lattice
        else:
            self.lattice = Lattice.cubic(1.0)
        self.family = None
        self.poles = []
        self.set_hkl_poles(hkl)
        self.mksize = 50
        self.x = np.array([1., 0., 0.])
        self.y = np.array([0., 1., 0.])
        self.z = np.array([0., 0., 1.])

    def set_hkl_poles(self, hkl='111'):
        """Set the pole (aka hkl planes) list to to use in the `PoleFigure`.

        The list of poles can be given by the family type or directly by a list of `HklPlanes` objects.

        :params str/list hkl: slip plane family ('111' by default)
        """
        if type(hkl) is str:
            self.family = hkl  # keep a record of this
            hkl_planes = self.lattice.get_hkl_family(self.family)
        elif type(hkl) is list:
            self.family = None
            hkl_planes = hkl
        self.poles = hkl_planes  #[p.normal() for p in hkl_planes]

    def plot_line_between_crystal_dir(self, c1, c2, ax=None, steps=25, col='k'):
        '''Plot a curve between two crystal directions.

        The curve is actually composed of several straight lines segments to
        draw from direction 1 to direction 2.

        :param c1: vector describing crystal direction 1
        :param c2: vector describing crystal direction 2
        :param ax: a reference to a pyplot ax to draw the line
        :param int steps: number of straight lines composing the curve (11 by default)
        :param col: line color (black by default)
        '''
        path = np.zeros((steps, 2), dtype=float)
        for j, i in enumerate(np.linspace(0., 1., steps)):
            ci = i * c1 + (1 - i) * c2
            ci /= np.linalg.norm(ci)
            if self.proj == 'stereo':
                ci += self.z
                ci /= ci[2]
            path[j, 0] = ci[0]
            path[j, 1] = ci[1]
        ax.plot(path[:, 0], path[:, 1], color=col, markersize=self.mksize, linewidth=0.5, zorder=0)
        plt.axis("off")
        
    def plot_pf_background(self, ax, labels=True):
        '''Function to plot the background of the pole figure.
        :param ax: a reference to a pyplot ax to draw the backgroud.
        :param bool labels: add lables to axes (True by default).
        '''
        an = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(an), np.sin(an), 'k-', zorder=0)
        ax.plot([-1, 1], [0, 0], 'k-', zorder=0)
        ax.plot([0, 0], [-1, 1], 'k-', zorder=0)
        axe_labels = ['X', 'Y', 'Z']
        if self.axis == 'Z':
            (h, v, _) = (0, 1, 2)
        elif self.axis == 'Y':
            (h, v, _) = (0, 2, 1)
        else:
            (h, v, _) = (1, 2, 0)
        if labels:
            ax.annotate(axe_labels[h], (1.01, 0.0), xycoords='data', fontsize=8,
                        horizontalalignment='left', verticalalignment='center')
            ax.annotate(axe_labels[v], (0.0, 1.01), xycoords='data', fontsize=8,
                        horizontalalignment='center', verticalalignment='bottom')

    def sst_symmetry(self, v, symms):
        """Transform a given vector according to the lattice symmetry associated
        with the pole figure.

        This function transform a vector so that it lies in the smallest
        symmetry equivalent zone.

        :param v: the vector to transform.
        :return: the transformed vector.
        """
        # get the symmetry from the lattice associated with the pole figure
        symmetry = self.lattice._symmetry
        if symmetry == symmetry.cubic:
            return PoleFigure.sst_symmetry_cubic(v)
        elif symmetry == symmetry.hexagonal:
            #syms = symmetry.symmetry_operators()
            # syms = np.concatenate((symms, -symms))
            syms = np.unique(symms, axis=0)
            for i in range(len(syms)):
                sym = syms[i]
                v_sym = np.dot(sym, v)
                # look at vectors pointing up
                if v_sym[2] < 0:
                    v_sym *= -1
                # now evaluate if projection is in the sst
                if v_sym[1] < 0 or v_sym[0] < 0:
                    continue
                elif v_sym[1] / v_sym[0] > np.tan(np.pi / 6):
                    continue
                else:
                    break
            return v_sym
        else:
            print('unsupported symmetry: %s' % symmetry)
            return None

    @staticmethod
    def sst_symmetry_cubic(z_rot):
        '''Transform a given vector according to the cubic symmetry.

        This function transform a vector so that it lies in the unit SST triangle.

        :param z_rot: vector to transform.
        :return: the transformed vector.
        '''
        if z_rot[0] < 0: z_rot[0] = -z_rot[0]
        if z_rot[1] < 0: z_rot[1] = -z_rot[1]
        if z_rot[2] < 0: z_rot[2] = -z_rot[2]
        if (z_rot[2] > z_rot[1]):
            z_rot[1], z_rot[2] = z_rot[2], z_rot[1]
        if (z_rot[1] > z_rot[0]):
            z_rot[0], z_rot[1] = z_rot[1], z_rot[0]
        if (z_rot[2] > z_rot[1]):
            z_rot[1], z_rot[2] = z_rot[2], z_rot[1]
        return np.array([z_rot[1], z_rot[2], z_rot[0]])
        
    def plot_pf(self, col, orient_data, ax=None, mk='o', ann=False, ftsize=6):
        """Create the direct pole figure.

        :param ax: a reference to a pyplot ax to draw the poles.
        :param mk: marker used to plot the poles (disc by default).
        :param bool ann: Annotate the pole with the coordinates of the vector
            if True (False by default).
            
        """
        self.plot_pf_background(ax)
        cp_0, cp_1 = [], []
        colors = []
        for igr, g in enumerate(orient_data):
            if np.isnan(g).all() or np.all(g==0):
                continue
            
            gt = g.transpose()
            for i, hkl_plane in enumerate(self.poles):
                c = hkl_plane.normal()
                c_rot = gt.dot(c)
                color = col[igr]
                
                if self.axis == 'Z':
                    (h, v, u) = (0, 1, 2)
                elif self.axis == 'Y':
                    (h, v, u) = (0, 2, 1)
                else:
                    (h, v, u) = (1, 2, 0)
                    
                axis_rot = c_rot[[h, v, u]]
                # the direction to plot is given by c_dir[h,v,u]
                
                if axis_rot[2] < 0:
                    axis_rot *= -1  # make unit vector have z>0
                if self.proj == 'flat':
                    cp = axis_rot
                elif self.proj == 'stereo':
                    c = axis_rot + self.z
                    c /= c[2]  # SP'/SP = r/z with r=1
                    cp = c
                    # cp = np.cross(c, self.z)
                else:
                    raise ValueError('Error, unsupported projection type', self.proj)
                
                cp_0.append(cp[0])
                cp_1.append(cp[1])
                colors.append(color)
                # Next 3 lines are necessary in case c_dir[2]=0, as for Euler angles [45, 45, 0]
                if axis_rot[2] < 0.000001:
                    cp_0.append(-cp[0])
                    cp_1.append(-cp[1])
                    colors.append(color)
                    # ax.scatter(-cp[0], -cp[1], linewidth=0, c=color, marker='o', s=axis_rot)
        ax.scatter(cp_0, cp_1, c=colors, s=self.mksize, zorder=2)
                
        ax.axis([-1.1, 1.1, -1.1, 1.1])
        ax.axis('off')
        ax.set_title('{%s} direct %s projection' % (self.family, self.proj), fontsize = ftsize)
        
    def plot_sst_color(self, col, orient_data, ax=None, mk='s', \
                          ann=False, ftsize=6, phase = 0, symms=None):
        """ Create the inverse pole figure in the unit standard triangle.
        :param ax: a reference to a pyplot ax to draw the poles.
        :param mk: marker used to plot the poles (square by default).
        :param bool ann: Annotate the pole with the coordinates of the vector if True (False by default).
        """
        system = None
        symmetry = self.lattice._symmetry
        if phase==0:
            sst_poles = [(0, 0, 1), (1, 0, 1), (1, 1, 1)]
            ax.axis([-0.05, 0.45, -0.05, 0.40])
            system = 'cubic'
        elif phase==1:
            sst_poles = [(0, 0, 1), (2, -1, 0), (1, 0, 0)]
            ax.axis([-0.05, 1.05, -0.05, 0.6])
            system = 'hexa'
        else:
            print('unssuported symmetry: %s' % symmetry)
        A = HklPlane(*sst_poles[0], lattice=self.lattice)
        B = HklPlane(*sst_poles[1], lattice=self.lattice)
        C = HklPlane(*sst_poles[2], lattice=self.lattice)
        if system == 'cubic':
            self.plot_line_between_crystal_dir(A.normal(), B.normal(), ax=ax, steps=int(1+(45/5)), col='k')
            self.plot_line_between_crystal_dir(B.normal(), C.normal(), ax=ax, steps=int(1+(35/5)), col='k')
            self.plot_line_between_crystal_dir(C.normal(), A.normal(), ax=ax, steps=int(1+(55/5)), col='k')
        elif system == 'hexa':
            self.plot_line_between_crystal_dir(A.normal(), B.normal(), ax=ax, steps=int(1+(90/5)), col='k')
            self.plot_line_between_crystal_dir(B.normal(), C.normal(), ax=ax, steps=int(1+(30/5)), col='k')
            self.plot_line_between_crystal_dir(C.normal(), A.normal(), ax=ax, steps=int(1+(90/5)), col='k')
        else:
            self.plot_line_between_crystal_dir(A.normal(), B.normal(), ax=ax, col='k')
            self.plot_line_between_crystal_dir(B.normal(), C.normal(), ax=ax, col='k')
            self.plot_line_between_crystal_dir(C.normal(), A.normal(), ax=ax, col='k')
        # display the 3 crystal axes
        poles = [A, B, C]
        v_align = ['top', 'top', 'bottom']
        for i in range(3):
            hkl = poles[i]
            c_dir = hkl.normal()
            c = c_dir + self.z
            c /= c[2]  # SP'/SP = r/z with r=1
            pole_str = '%d%d%d' % hkl.miller_indices()
            if phase==1:
                pole_str = '%d%d%d%d' % HklPlane.three_to_four_indices(*hkl.miller_indices())
            ax.annotate(pole_str, (c[0], c[1] - (2 * (i < 2) - 1) * 0.01), xycoords='data',
                        fontsize=8, horizontalalignment='center', verticalalignment=v_align[i])
        # now plot the sample axis
        cp_0, cp_1 = [], []
        colors = []
        for igr, g in enumerate(orient_data):
            if np.isnan(g).all() or np.all(g==0):
                continue
            # compute axis and apply SST symmetry
            if self.axis == 'Z':
                axis = self.z
            elif self.axis == 'Y':
                axis = self.y
            else:
                axis = self.x
                
            axis_rot = self.sst_symmetry(g.dot(axis), symms)
            color = np.round(col[igr],5)
            if axis_rot[2] < 0:
                axis_rot *= -1  # make unit vector have z>0
            if self.proj == 'flat':
                cp = axis_rot
            elif self.proj == 'stereo':
                c = axis_rot + self.z
                c /= c[2]  # SP'/SP = r/z with r=1
                cp = c
                # cp = np.cross(c, self.z)
            else:
                raise ValueError('Error, unsupported projection type', self.proj)
            
            cp_0.append(cp[0])
            cp_1.append(cp[1])
            colors.append(color)
            # Next 3 lines are necessary in case c_dir[2]=0, as for Euler angles [45, 45, 0]
            if axis_rot[2] < 0.000001:
                cp_0.append(-cp[0])
                cp_1.append(-cp[1])
                colors.append(color)
                # ax.scatter(-cp[0], -cp[1], linewidth=0, c=color, marker='o', s=axis_rot)
        ax.scatter(cp_0, cp_1, c=colors, s=self.mksize, zorder=2)        
        ax.set_title('%s-axis SST inverse %s projection' % (self.axis, self.proj), fontsize = ftsize)
        plt.axis("off")


# =============================================================================
# Plot functions
# =============================================================================
# def rot_mat_to_euler(rot_mat): 
#     r = R.from_matrix(rot_mat)
#     return r.as_euler('zxz')* 180/np.pi

def OrientationMatrix2Euler(g):
    """
    Compute the Euler angles from the orientation matrix.
    This conversion follows the paper of Rowenhorst et al. :cite:`Rowenhorst2015`.
    In particular when :math:`g_{33} = 1` within the machine precision,
    there is no way to determine the values of :math:`\phi_1` and :math:`\phi_2`
    (only their sum is defined). The convention is to attribute
    the entire angle to :math:`\phi_1` and set :math:`\phi_2` to zero.
    :param g: The 3x3 orientation matrix
    :return: The 3 euler angles in degrees.
    """
    eps = np.finfo('float').eps
    (phi1, Phi, phi2) = (0.0, 0.0, 0.0)
    # treat special case where g[2, 2] = 1
    if np.abs(g[2, 2]) >= 1 - eps:
        if g[2, 2] > 0.0:
            phi1 = np.arctan2(g[0][1], g[0][0])
        else:
            phi1 = -np.arctan2(-g[0][1], g[0][0])
            Phi = np.pi
    else:
        Phi = np.arccos(g[2][2])
        zeta = 1.0 / np.sqrt(1.0 - g[2][2] ** 2)
        phi1 = np.arctan2(g[2][0] * zeta, -g[2][1] * zeta)
        phi2 = np.arctan2(g[0][2] * zeta, g[1][2] * zeta)
    # ensure angles are in the range [0, 2*pi]
    if phi1 < 0.0:
        phi1 += 2 * np.pi
    if Phi < 0.0:
        Phi += 2 * np.pi
    if phi2 < 0.0:
        phi2 += 2 * np.pi
    return np.degrees([phi2, Phi, phi1])

def simple_plots(lim_x, lim_y, strain_matrix, strain_matrixs, col, colx, coly,
                 match_rate, mat_global, spots_len, iR_pix, fR_pix,
                 model_direc, material_, material1_, match_rate_threshold=5, bins=30):    
    if material_ == material1_:
        matid = 0
        for index in range(len(strain_matrix)):
            nan_index = np.where(match_rate[index][0] <= match_rate_threshold)[0]            
            col_plot = np.copy(col[index][0])
            col_plot[nan_index,:] = np.nan,np.nan,np.nan
            col_plot = col_plot.reshape((lim_x, lim_y, 3))
        
            mr_plot = np.copy(match_rate[index][0])
            mr_plot[nan_index,:] = np.nan
            mr_plot = mr_plot.reshape((lim_x, lim_y))
            
            mat_glob = np.copy(mat_global[index][0])
            mat_glob[nan_index,:] = np.nan
            mat_glob = mat_glob.reshape((lim_x, lim_y))
            
            fig = plt.figure(figsize=(11.69,8.27), dpi=100)
            bottom, top = 0.1, 0.9
            left, right = 0.1, 0.8
            fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
        
            axs = fig.subplots(1, 3)
            axs[0].set_title(r"IPF Z map", loc='center', fontsize=8)
            axs[0].imshow(col_plot, origin='lower')
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            
            axs[1].set_title(r"Material Index", loc='center', fontsize=8)
            im = axs[1].imshow(mat_glob, origin='lower', vmin=0, vmax=1)
            axs[1].set_xticks([])
            axs[1].set_yticks([])
            
            divider = make_axes_locatable(axs[1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
            
            axs[2].set_title(r"Matching rate", loc='center', fontsize=8)
            im = axs[2].imshow(mr_plot, origin='lower', cmap=plt.cm.jet, vmin=0, vmax=100)
            axs[2].set_xticks([])
            axs[2].set_yticks([])
            
            divider = make_axes_locatable(axs[2])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
        
            for ax in axs.flat:
                ax.label_outer()
        
            plt.savefig(model_direc+ "//figure_global_UB"+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
            plt.close(fig)
            
            spots_len_plot = np.copy(spots_len[index][0])
            spots_len_plot[nan_index,:] = np.nan
            spots_len_plot = spots_len_plot.reshape((lim_x, lim_y))
            
            iR_pix_plot = np.copy(iR_pix[index][0])
            iR_pix_plot[nan_index,:] = np.nan
            iR_pix_plot = iR_pix_plot.reshape((lim_x, lim_y))
            
            fR_pix_plot = np.copy(fR_pix[index][0])
            fR_pix_plot[nan_index,:] = np.nan
            fR_pix_plot = fR_pix_plot.reshape((lim_x, lim_y))
            
            fig = plt.figure(figsize=(11.69,8.27), dpi=100)
            bottom, top = 0.1, 0.9
            left, right = 0.1, 0.8
            fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
        
            axs = fig.subplots(1, 3)
            axs[0].set_title(r"Number of spots detected", loc='center', fontsize=8)
            im = axs[0].imshow(spots_len_plot, origin='lower', cmap=plt.cm.jet)
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            
            divider = make_axes_locatable(axs[0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
        
            axs[1].set_title(r"Initial pixel residues", loc='center', fontsize=8)
            im = axs[1].imshow(iR_pix_plot, origin='lower', cmap=plt.cm.jet)
            axs[1].set_xticks([])
            axs[1].set_yticks([])
            
            divider = make_axes_locatable(axs[1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
        
            axs[2].set_title(r"Final pixel residues", loc='center', fontsize=8)
            im = axs[2].imshow(fR_pix_plot, origin='lower', cmap=plt.cm.jet)
            axs[2].set_xticks([])
            axs[2].set_yticks([])
            
            divider = make_axes_locatable(axs[2])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
        
            for ax in axs.flat:
                ax.label_outer()
        
            plt.savefig(model_direc+'//figure_mr_ir_fr_UB'+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
            plt.close(fig)
    else:    
    
        for matid in range(2):
            for index in range(len(strain_matrix)):
                nan_index1 = np.where(match_rate[index][0] <= match_rate_threshold)[0]
                mat_id_index = np.where(mat_global[index][0] != matid+1)[0]
                nan_index = np.hstack((mat_id_index,nan_index1))
                nan_index = np.unique(nan_index)
                
                try:
                    col_plot = np.copy(col[index][0])
                    col_plot[nan_index,:] = np.nan,np.nan,np.nan
                    col_plot = col_plot.reshape((lim_x, lim_y, 3))
                
                    mr_plot = np.copy(match_rate[index][0])
                    mr_plot[nan_index,:] = np.nan
                    mr_plot = mr_plot.reshape((lim_x, lim_y))
                    
                    mat_glob = np.copy(mat_global[index][0])
                    mat_glob[nan_index,:] = np.nan
                    mat_glob = mat_glob.reshape((lim_x, lim_y))
                    
                    fig = plt.figure(figsize=(11.69,8.27), dpi=100)
                    bottom, top = 0.1, 0.9
                    left, right = 0.1, 0.8
                    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
                
                    axs = fig.subplots(1, 3)
                    axs[0].set_title(r"IPF Z map", loc='center', fontsize=8)
                    axs[0].imshow(col_plot, origin='lower')
                    axs[0].set_xticks([])
                    axs[0].set_yticks([])
                    
                    axs[1].set_title(r"Material Index", loc='center', fontsize=8)
                    im = axs[1].imshow(mat_glob, origin='lower', vmin=0, vmax=2)
                    axs[1].set_xticks([])
                    axs[1].set_yticks([])
                    
                    divider = make_axes_locatable(axs[1])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im, cax=cax, orientation='vertical')
                    
                    axs[2].set_title(r"Matching rate", loc='center', fontsize=8)
                    im = axs[2].imshow(mr_plot, origin='lower', cmap=plt.cm.jet, vmin=0, vmax=100)
                    axs[2].set_xticks([])
                    axs[2].set_yticks([])
                    
                    divider = make_axes_locatable(axs[2])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im, cax=cax, orientation='vertical')
                
                    for ax in axs.flat:
                        ax.label_outer()
                
                    plt.savefig(model_direc+ "//figure_global_mat"+str(matid)+"_UB"+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
                    plt.close(fig)
                except:
                    print("Error in plots")
                    
                spots_len_plot = np.copy(spots_len[index][0])
                spots_len_plot[nan_index,:] = np.nan
                spots_len_plot = spots_len_plot.reshape((lim_x, lim_y))
                
                iR_pix_plot = np.copy(iR_pix[index][0])
                iR_pix_plot[nan_index,:] = np.nan
                iR_pix_plot = iR_pix_plot.reshape((lim_x, lim_y))
                
                fR_pix_plot = np.copy(fR_pix[index][0])
                fR_pix_plot[nan_index,:] = np.nan
                fR_pix_plot = fR_pix_plot.reshape((lim_x, lim_y))
                
                try:
                    fig = plt.figure(figsize=(11.69,8.27), dpi=100)
                    bottom, top = 0.1, 0.9
                    left, right = 0.1, 0.8
                    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
                
                    axs = fig.subplots(1, 3)
                    axs[0].set_title(r"Number of spots detected", loc='center', fontsize=8)
                    im = axs[0].imshow(spots_len_plot, origin='lower', cmap=plt.cm.jet)
                    axs[0].set_xticks([])
                    axs[0].set_yticks([])
                    
                    divider = make_axes_locatable(axs[0])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im, cax=cax, orientation='vertical')
                
                    axs[1].set_title(r"Initial pixel residues", loc='center', fontsize=8)
                    im = axs[1].imshow(iR_pix_plot, origin='lower', cmap=plt.cm.jet)
                    axs[1].set_xticks([])
                    axs[1].set_yticks([])
                    
                    divider = make_axes_locatable(axs[1])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im, cax=cax, orientation='vertical')
                
                    axs[2].set_title(r"Final pixel residues", loc='center', fontsize=8)
                    im = axs[2].imshow(fR_pix_plot, origin='lower', cmap=plt.cm.jet)
                    axs[2].set_xticks([])
                    axs[2].set_yticks([])
                    
                    divider = make_axes_locatable(axs[2])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im, cax=cax, orientation='vertical')
                
                    for ax in axs.flat:
                        ax.label_outer()
                
                    plt.savefig(model_direc+'//figure_mr_ir_fr_mat'+str(matid)+"_UB"+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
                    plt.close(fig)
                except:
                    print("Error in plots")
      
def global_plots(lim_x, lim_y, rotation_matrix1, strain_matrix, strain_matrixs, col, colx, coly,
                 match_rate, mat_global, spots_len, iR_pix, fR_pix,
                 model_direc, material_, material1_, match_rate_threshold=5, bins=30, constantlength="a"):
    call_global()
    if material_ == material1_:
        mu_sd = []
        mu_sdc = []
        for index in range(len(spots_len)):
            ### index for nans
            nan_index = np.where(match_rate[index][0] <= match_rate_threshold)[0]
            if index == 0:
                spots_len_plot = np.copy(spots_len[index][0])
                mr_plot = np.copy(match_rate[index][0])
                iR_pix_plot = np.copy(iR_pix[index][0])
                fR_pix_plot = np.copy(fR_pix[index][0])
                strain_matrix_plot = np.copy(strain_matrix[index][0])
                e11c = strain_matrix_plot[:,0,0]#.reshape((lim_x, lim_y))
                e22c = strain_matrix_plot[:,1,1]#.reshape((lim_x, lim_y))
                e33c = strain_matrix_plot[:,2,2]#.reshape((lim_x, lim_y))
                e12c = strain_matrix_plot[:,0,1]#.reshape((lim_x, lim_y))
                e13c = strain_matrix_plot[:,0,2]#.reshape((lim_x, lim_y))
                e23c = strain_matrix_plot[:,1,2]#.reshape((lim_x, lim_y))
                strain_matrixs_plot = np.copy(strain_matrixs[index][0])
                e11s = strain_matrixs_plot[:,0,0]#.reshape((lim_x, lim_y))
                e22s = strain_matrixs_plot[:,1,1]#.reshape((lim_x, lim_y))
                e33s = strain_matrixs_plot[:,2,2]#.reshape((lim_x, lim_y))
                e12s = strain_matrixs_plot[:,0,1]#.reshape((lim_x, lim_y))
                e13s = strain_matrixs_plot[:,0,2]#.reshape((lim_x, lim_y))
                e23s = strain_matrixs_plot[:,1,2]#.reshape((lim_x, lim_y))
                spots_len_plot[nan_index] = np.nan 
                mr_plot[nan_index] = np.nan 
                iR_pix_plot[nan_index] = np.nan 
                fR_pix_plot[nan_index] = np.nan 
                e11c[nan_index] = np.nan 
                e22c[nan_index] = np.nan 
                e33c[nan_index] = np.nan 
                e12c[nan_index] = np.nan 
                e13c[nan_index] = np.nan 
                e23c[nan_index] = np.nan 
                e11s[nan_index] = np.nan 
                e22s[nan_index] = np.nan 
                e33s[nan_index] = np.nan 
                e12s[nan_index] = np.nan 
                e13s[nan_index] = np.nan 
                e23s[nan_index] = np.nan 
                
            else:
                temp = np.copy(spots_len[index][0])
                temp[nan_index] = np.nan
                spots_len_plot = np.vstack((spots_len_plot,temp))
                
                temp = np.copy(match_rate[index][0])
                temp[nan_index] = np.nan
                mr_plot = np.vstack((mr_plot,temp))
                
                temp = np.copy(iR_pix[index][0])
                temp[nan_index] = np.nan
                iR_pix_plot = np.vstack((iR_pix_plot,temp))
        
                temp = np.copy(fR_pix[index][0])
                temp[nan_index] = np.nan
                fR_pix_plot = np.vstack((fR_pix_plot,temp))
                
                strain_matrix_plot = np.copy(strain_matrix[index][0])
                temp = np.copy(strain_matrix_plot[:,0,0])
                temp[nan_index] = np.nan
                e11c = np.vstack((e11c,temp))
                temp = np.copy(strain_matrix_plot[:,1,1])
                temp[nan_index] = np.nan
                e22c = np.vstack((e22c,temp))
                temp = np.copy(strain_matrix_plot[:,2,2])
                temp[nan_index] = np.nan
                e33c = np.vstack((e33c,temp))
                temp = np.copy(strain_matrix_plot[:,0,1])
                temp[nan_index] = np.nan
                e12c = np.vstack((e12c,temp))
                temp = np.copy(strain_matrix_plot[:,0,2])
                temp[nan_index] = np.nan
                e13c = np.vstack((e13c,temp))
                temp = np.copy(strain_matrix_plot[:,1,2])
                temp[nan_index] = np.nan
                e23c = np.vstack((e23c,temp))
                ##
                strain_matrixs_plot = np.copy(strain_matrixs[index][0])
                temp = np.copy(strain_matrixs_plot[:,0,0])
                temp[nan_index] = np.nan
                e11s = np.vstack((e11s,temp))
                temp = np.copy(strain_matrixs_plot[:,1,1])
                temp[nan_index] = np.nan
                e22s = np.vstack((e22s,temp))
                temp = np.copy(strain_matrixs_plot[:,2,2])
                temp[nan_index] = np.nan
                e33s = np.vstack((e33s,temp))
                temp = np.copy(strain_matrixs_plot[:,0,1])
                temp[nan_index] = np.nan
                e12s = np.vstack((e12s,temp))
                temp = np.copy(strain_matrixs_plot[:,0,2])
                temp[nan_index] = np.nan
                e13s = np.vstack((e13s,temp))
                temp = np.copy(strain_matrixs_plot[:,1,2])
                temp[nan_index] = np.nan
                e23s = np.vstack((e23s,temp))
        
        spots_len_plot = spots_len_plot.flatten()
        mr_plot = mr_plot.flatten()
        iR_pix_plot = iR_pix_plot.flatten()
        fR_pix_plot = fR_pix_plot.flatten() 
        e11c = e11c.flatten()
        e22c = e22c.flatten()
        e33c = e33c.flatten()
        e12c = e12c.flatten()
        e13c = e13c.flatten()
        e23c = e23c.flatten()
        e11s = e11s.flatten()
        e22s = e22s.flatten()
        e33s = e33s.flatten()
        e12s = e12s.flatten()
        e13s = e13s.flatten()
        e23s = e23s.flatten()
        
        spots_len_plot = spots_len_plot[~np.isnan(spots_len_plot)]
        mr_plot = mr_plot[~np.isnan(mr_plot)]
        iR_pix_plot = iR_pix_plot[~np.isnan(iR_pix_plot)]
        fR_pix_plot = fR_pix_plot[~np.isnan(fR_pix_plot)]
        e11c = e11c[~np.isnan(e11c)]
        e22c = e22c[~np.isnan(e22c)]
        e33c = e33c[~np.isnan(e33c)]
        e12c = e12c[~np.isnan(e12c)]
        e13c = e13c[~np.isnan(e13c)]
        e23c = e23c[~np.isnan(e23c)]
        e11s = e11s[~np.isnan(e11s)]
        e22s = e22s[~np.isnan(e22s)]
        e33s = e33s[~np.isnan(e33s)]
        e12s = e12s[~np.isnan(e12s)]
        e13s = e13s[~np.isnan(e13s)]
        e23s = e23s[~np.isnan(e23s)]
        
        try:
            title = "Number of spots and matching rate"
            fig = plt.figure()
            axs = fig.subplots(1, 2)
            axs[0].set_title("Number of spots", loc='center', fontsize=8)
            axs[0].hist(spots_len_plot, bins=bins)
            axs[0].set_ylabel('Frequency', fontsize=8)
            axs[0].tick_params(axis='both', which='major', labelsize=8)
            axs[0].tick_params(axis='both', which='minor', labelsize=8)
            axs[1].set_title("matching rate", loc='center', fontsize=8)
            axs[1].hist(mr_plot, bins=bins)
            axs[1].set_ylabel('Frequency', fontsize=8)
            axs[1].tick_params(axis='both', which='major', labelsize=8)
            axs[1].tick_params(axis='both', which='minor', labelsize=8)
            plt.tight_layout()
            plt.savefig(model_direc+ "//"+title+'.png', format='png', dpi=1000) 
            plt.close(fig)
        except:
            pass
        try:
            title = "Initial and Final residues"
            fig = plt.figure()
            axs = fig.subplots(1, 2)
            axs[0].set_title("Initial residues", loc='center', fontsize=8)
            axs[0].hist(iR_pix_plot, bins=bins)
            axs[0].set_ylabel('Frequency', fontsize=8)
            axs[0].tick_params(axis='both', which='major', labelsize=8)
            axs[0].tick_params(axis='both', which='minor', labelsize=8)
            axs[1].set_title("Final residues", loc='center', fontsize=8)
            axs[1].hist(fR_pix_plot, bins=bins)
            axs[1].set_ylabel('Frequency', fontsize=8)
            axs[1].tick_params(axis='both', which='major', labelsize=8)
            axs[1].tick_params(axis='both', which='minor', labelsize=8)
            plt.tight_layout()
            plt.savefig(model_direc+ "//"+title+'.png',format='png', dpi=1000) 
            plt.close(fig)
        except:
            pass
        try:
            title = "strain Crystal reference"
            fig = plt.figure()
            fig.suptitle(title, fontsize=10)
            axs = fig.subplots(2, 3)
            axs[0, 0].set_title(r"$\epsilon_{11}$ (%)", loc='center', fontsize=8)
            logdata = e11c #np.log(e11c)
            xmin = logdata.min()
            xmax = logdata.max()
            x1 = np.linspace(xmin, xmax, 1000)
            estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
            pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
            axs[0, 0].axvline(x=estimated_mu, c="k")
            axs[0, 0].plot(x1, pdf, 'r')
            axs[0, 0].hist(logdata, bins=bins, density=True, alpha=0.8)
            mu_sdc.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
            axs[0, 0].set_ylabel('Frequency', fontsize=8)
            axs[0, 0].tick_params(axis='both', which='major', labelsize=8)
            axs[0, 0].tick_params(axis='both', which='minor', labelsize=8)
            
            axs[0, 1].set_title(r"$\epsilon_{22}$ (%)", loc='center', fontsize=8)
            logdata = e22c #np.log(e22c)
            xmin = logdata.min()
            xmax = logdata.max()
            x1 = np.linspace(xmin, xmax, 1000)
            estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
            pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
            axs[0, 1].axvline(x=estimated_mu, c="k")
            axs[0, 1].plot(x1, pdf, 'r')
            axs[0, 1].hist(logdata, bins=bins, density=True, alpha=0.8)
            mu_sdc.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
            # axs[0, 1].hist(e22c, bins=bins)
            axs[0, 1].set_ylabel('Frequency', fontsize=8)
            axs[0, 1].tick_params(axis='both', which='major', labelsize=8)
            axs[0, 1].tick_params(axis='both', which='minor', labelsize=8)
            
            axs[0, 2].set_title(r"$\epsilon_{33}$ (%)", loc='center', fontsize=8)
            logdata = e33c #np.log(e33c)
            xmin = logdata.min()
            xmax = logdata.max()
            x1 = np.linspace(xmin, xmax, 1000)
            estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
            pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
            axs[0, 2].axvline(x=estimated_mu, c="k")
            axs[0, 2].plot(x1, pdf, 'r')
            axs[0, 2].hist(logdata, bins=bins, density=True, alpha=0.8)
            mu_sdc.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
            # axs[0, 2].hist(e33c, bins=bins)
            axs[0, 2].set_ylabel('Frequency', fontsize=8)
            axs[0, 2].tick_params(axis='both', which='major', labelsize=8)
            axs[0, 2].tick_params(axis='both', which='minor', labelsize=8)
            
            axs[1, 0].set_title(r"$\epsilon_{12}$ (%)", loc='center', fontsize=8)
            logdata = e12c#np.log(e12c)
            xmin = logdata.min()
            xmax = logdata.max()
            x1 = np.linspace(xmin, xmax, 1000)
            estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
            pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
            axs[1, 0].axvline(x=estimated_mu, c="k")
            axs[1, 0].plot(x1, pdf, 'r')
            axs[1, 0].hist(logdata, bins=bins, density=True, alpha=0.8)
            mu_sdc.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
            # axs[1, 0].hist(e12c, bins=bins)
            axs[1, 0].set_ylabel('Frequency', fontsize=8)
            axs[1, 0].tick_params(axis='both', which='major', labelsize=8)
            axs[1, 0].tick_params(axis='both', which='minor', labelsize=8)
            
            axs[1, 1].set_title(r"$\epsilon_{13}$ (%)", loc='center', fontsize=8)
            logdata = e13c#np.log(e13c)
            xmin = logdata.min()
            xmax = logdata.max()
            x1 = np.linspace(xmin, xmax, 1000)
            estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
            pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
            axs[1, 1].axvline(x=estimated_mu, c="k")
            axs[1, 1].plot(x1, pdf, 'r')
            axs[1, 1].hist(logdata, bins=bins, density=True, alpha=0.8)
            mu_sdc.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
            # axs[1, 1].hist(e13c, bins=bins)
            axs[1, 1].set_ylabel('Frequency', fontsize=8)
            axs[1, 1].tick_params(axis='both', which='major', labelsize=8)
            axs[1, 1].tick_params(axis='both', which='minor', labelsize=8)
            
            axs[1, 2].set_title(r"$\epsilon_{23}$ (%)", loc='center', fontsize=8)
            logdata = e23c#np.log(e23c)
            xmin = logdata.min()
            xmax = logdata.max()
            x1 = np.linspace(xmin, xmax, 1000)
            estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
            pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
            axs[1, 2].axvline(x=estimated_mu, c="k")
            axs[1, 2].plot(x1, pdf, 'r')
            axs[1, 2].hist(logdata, bins=bins, density=True, alpha=0.8)
            mu_sdc.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
            # axs[1, 2].hist(e23c, bins=bins)
            axs[1, 2].set_ylabel('Frequency', fontsize=8)
            axs[1, 2].tick_params(axis='both', which='major', labelsize=8)
            axs[1, 2].tick_params(axis='both', which='minor', labelsize=8)
            plt.tight_layout()
            plt.savefig(model_direc+ "//"+title+'.png', format='png', dpi=1000) 
            plt.close(fig)
        except:
            pass
    
        try:
            title = "strain Sample reference"
            fig = plt.figure()
            fig.suptitle(title, fontsize=10)
            axs = fig.subplots(2, 3)
            axs[0, 0].set_title(r"$\epsilon_{11}$ (%)", loc='center', fontsize=8)
            logdata = e11s #np.log(e11c)
            xmin = logdata.min()
            xmax = logdata.max()
            x1 = np.linspace(xmin, xmax, 1000)
            estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
            pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
            axs[0, 0].axvline(x=estimated_mu, c="k")
            axs[0, 0].plot(x1, pdf, 'r')
            axs[0, 0].hist(logdata, bins=bins, density=True, alpha=0.8)
            # axs[0, 0].hist(e11s, bins=bins)
            axs[0, 0].set_ylabel('Frequency', fontsize=8)
            axs[0, 0].tick_params(axis='both', which='major', labelsize=8)
            axs[0, 0].tick_params(axis='both', which='minor', labelsize=8)
            
            mu_sd.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
            
            axs[0, 1].set_title(r"$\epsilon_{22}$ (%)", loc='center', fontsize=8)
            logdata = e22s #np.log(e22c)
            xmin = logdata.min()
            xmax = logdata.max()
            x1 = np.linspace(xmin, xmax, 1000)
            estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
            pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
            axs[0, 1].axvline(x=estimated_mu, c="k")
            axs[0, 1].plot(x1, pdf, 'r')
            axs[0, 1].hist(logdata, bins=bins, density=True, alpha=0.8)
            # axs[0, 1].hist(e22s, bins=bins)
            axs[0, 1].set_ylabel('Frequency', fontsize=8)
            axs[0, 1].tick_params(axis='both', which='major', labelsize=8)
            axs[0, 1].tick_params(axis='both', which='minor', labelsize=8)
            
            mu_sd.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
            
            axs[0, 2].set_title(r"$\epsilon_{33}$ (%)", loc='center', fontsize=8)
            logdata = e33s #np.log(e33c)
            xmin = logdata.min()
            xmax = logdata.max()
            x1 = np.linspace(xmin, xmax, 1000)
            estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
            pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
            axs[0, 2].axvline(x=estimated_mu, c="k")
            axs[0, 2].plot(x1, pdf, 'r')
            axs[0, 2].hist(logdata, bins=bins, density=True, alpha=0.8)
            # axs[0, 2].hist(e33s, bins=bins)
            axs[0, 2].set_ylabel('Frequency', fontsize=8)
            axs[0, 2].tick_params(axis='both', which='major', labelsize=8)
            axs[0, 2].tick_params(axis='both', which='minor', labelsize=8)
            
            mu_sd.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
            
            axs[1, 0].set_title(r"$\epsilon_{12}$ (%)", loc='center', fontsize=8)
            logdata = e12s#np.log(e12c)
            xmin = logdata.min()
            xmax = logdata.max()
            x1 = np.linspace(xmin, xmax, 1000)
            estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
            pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
            axs[1, 0].axvline(x=estimated_mu, c="k")
            axs[1, 0].plot(x1, pdf, 'r')
            axs[1, 0].hist(logdata, bins=bins, density=True, alpha=0.8)
            # axs[1, 0].hist(e12s, bins=bins)
            axs[1, 0].set_ylabel('Frequency', fontsize=8)
            axs[1, 0].tick_params(axis='both', which='major', labelsize=8)
            axs[1, 0].tick_params(axis='both', which='minor', labelsize=8)
            
            mu_sd.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
            
            axs[1, 1].set_title(r"$\epsilon_{13}$ (%)", loc='center', fontsize=8)
            logdata = e13s#np.log(e13c)
            xmin = logdata.min()
            xmax = logdata.max()
            x1 = np.linspace(xmin, xmax, 1000)
            estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
            pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
            axs[1, 1].axvline(x=estimated_mu, c="k")
            axs[1, 1].plot(x1, pdf, 'r')
            axs[1, 1].hist(logdata, bins=bins, density=True, alpha=0.8)
            # axs[1, 1].hist(e13s, bins=bins)
            axs[1, 1].set_ylabel('Frequency', fontsize=8)
            axs[1, 1].tick_params(axis='both', which='major', labelsize=8)
            axs[1, 1].tick_params(axis='both', which='minor', labelsize=8)
            
            mu_sd.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
            
            axs[1, 2].set_title(r"$\epsilon_{23}$ (%)", loc='center', fontsize=8)
            logdata = e23s#np.log(e23c)
            xmin = logdata.min()
            xmax = logdata.max()
            x1 = np.linspace(xmin, xmax, 1000)
            estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
            pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
            axs[1, 2].axvline(x=estimated_mu, c="k")
            axs[1, 2].plot(x1, pdf, 'r')
            axs[1, 2].hist(logdata, bins=bins, density=True, alpha=0.8)
            # axs[1, 2].hist(e23s, bins=bins)
            axs[1, 2].set_ylabel('Frequency', fontsize=8)
            axs[1, 2].tick_params(axis='both', which='major', labelsize=8)
            axs[1, 2].tick_params(axis='both', which='minor', labelsize=8)
            
            mu_sd.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
            
            plt.tight_layout()
            plt.savefig(model_direc+ "//"+title+'.png', format='png', dpi=1000) 
            plt.close(fig)  
        except:
            pass

    else:
        mu_sd = []
        mu_sdc = []
        material_id = [material_, material1_]
        for matid in range(2):
            for index in range(len(spots_len)):
                ### index for nans
                nan_index1 = np.where(match_rate[index][0] <= match_rate_threshold)[0]
                mat_id_index = np.where(mat_global[index][0] != matid+1)[0]
                nan_index = np.hstack((mat_id_index,nan_index1))
                nan_index = np.unique(nan_index)
                
                if index == 0:
                    spots_len_plot = np.copy(spots_len[index][0])
                    mr_plot = np.copy(match_rate[index][0])
                    iR_pix_plot = np.copy(iR_pix[index][0])
                    fR_pix_plot = np.copy(fR_pix[index][0])
                    strain_matrix_plot = np.copy(strain_matrix[index][0])
                    e11c = strain_matrix_plot[:,0,0]#.reshape((lim_x, lim_y))
                    e22c = strain_matrix_plot[:,1,1]#.reshape((lim_x, lim_y))
                    e33c = strain_matrix_plot[:,2,2]#.reshape((lim_x, lim_y))
                    e12c = strain_matrix_plot[:,0,1]#.reshape((lim_x, lim_y))
                    e13c = strain_matrix_plot[:,0,2]#.reshape((lim_x, lim_y))
                    e23c = strain_matrix_plot[:,1,2]#.reshape((lim_x, lim_y))
                    strain_matrixs_plot = np.copy(strain_matrixs[index][0])
                    e11s = strain_matrixs_plot[:,0,0]#.reshape((lim_x, lim_y))
                    e22s = strain_matrixs_plot[:,1,1]#.reshape((lim_x, lim_y))
                    e33s = strain_matrixs_plot[:,2,2]#.reshape((lim_x, lim_y))
                    e12s = strain_matrixs_plot[:,0,1]#.reshape((lim_x, lim_y))
                    e13s = strain_matrixs_plot[:,0,2]#.reshape((lim_x, lim_y))
                    e23s = strain_matrixs_plot[:,1,2]#.reshape((lim_x, lim_y))
                    spots_len_plot[nan_index] = np.nan 
                    mr_plot[nan_index] = np.nan 
                    iR_pix_plot[nan_index] = np.nan 
                    fR_pix_plot[nan_index] = np.nan 
                    e11c[nan_index] = np.nan 
                    e22c[nan_index] = np.nan 
                    e33c[nan_index] = np.nan 
                    e12c[nan_index] = np.nan 
                    e13c[nan_index] = np.nan 
                    e23c[nan_index] = np.nan 
                    e11s[nan_index] = np.nan 
                    e22s[nan_index] = np.nan 
                    e33s[nan_index] = np.nan 
                    e12s[nan_index] = np.nan 
                    e13s[nan_index] = np.nan 
                    e23s[nan_index] = np.nan 
                    
                else:
                    temp = np.copy(spots_len[index][0])
                    temp[nan_index] = np.nan
                    spots_len_plot = np.vstack((spots_len_plot,temp))
                    
                    temp = np.copy(match_rate[index][0])
                    temp[nan_index] = np.nan
                    mr_plot = np.vstack((mr_plot,temp))
                    
                    temp = np.copy(iR_pix[index][0])
                    temp[nan_index] = np.nan
                    iR_pix_plot = np.vstack((iR_pix_plot,temp))
            
                    temp = np.copy(fR_pix[index][0])
                    temp[nan_index] = np.nan
                    fR_pix_plot = np.vstack((fR_pix_plot,temp))
                    
                    strain_matrix_plot = np.copy(strain_matrix[index][0])
                    temp = np.copy(strain_matrix_plot[:,0,0])
                    temp[nan_index] = np.nan
                    e11c = np.vstack((e11c,temp))
                    temp = np.copy(strain_matrix_plot[:,1,1])
                    temp[nan_index] = np.nan
                    e22c = np.vstack((e22c,temp))
                    temp = np.copy(strain_matrix_plot[:,2,2])
                    temp[nan_index] = np.nan
                    e33c = np.vstack((e33c,temp))
                    temp = np.copy(strain_matrix_plot[:,0,1])
                    temp[nan_index] = np.nan
                    e12c = np.vstack((e12c,temp))
                    temp = np.copy(strain_matrix_plot[:,0,2])
                    temp[nan_index] = np.nan
                    e13c = np.vstack((e13c,temp))
                    temp = np.copy(strain_matrix_plot[:,1,2])
                    temp[nan_index] = np.nan
                    e23c = np.vstack((e23c,temp))
                    ##
                    strain_matrixs_plot = np.copy(strain_matrixs[index][0])
                    temp = np.copy(strain_matrixs_plot[:,0,0])
                    temp[nan_index] = np.nan
                    e11s = np.vstack((e11s,temp))
                    temp = np.copy(strain_matrixs_plot[:,1,1])
                    temp[nan_index] = np.nan
                    e22s = np.vstack((e22s,temp))
                    temp = np.copy(strain_matrixs_plot[:,2,2])
                    temp[nan_index] = np.nan
                    e33s = np.vstack((e33s,temp))
                    temp = np.copy(strain_matrixs_plot[:,0,1])
                    temp[nan_index] = np.nan
                    e12s = np.vstack((e12s,temp))
                    temp = np.copy(strain_matrixs_plot[:,0,2])
                    temp[nan_index] = np.nan
                    e13s = np.vstack((e13s,temp))
                    temp = np.copy(strain_matrixs_plot[:,1,2])
                    temp[nan_index] = np.nan
                    e23s = np.vstack((e23s,temp))
            
            spots_len_plot = spots_len_plot.flatten()
            mr_plot = mr_plot.flatten()
            iR_pix_plot = iR_pix_plot.flatten()
            fR_pix_plot = fR_pix_plot.flatten() 
            e11c = e11c.flatten()
            e22c = e22c.flatten()
            e33c = e33c.flatten()
            e12c = e12c.flatten()
            e13c = e13c.flatten()
            e23c = e23c.flatten()
            e11s = e11s.flatten()
            e22s = e22s.flatten()
            e33s = e33s.flatten()
            e12s = e12s.flatten()
            e13s = e13s.flatten()
            e23s = e23s.flatten()
            
            spots_len_plot = spots_len_plot[~np.isnan(spots_len_plot)]
            mr_plot = mr_plot[~np.isnan(mr_plot)]
            iR_pix_plot = iR_pix_plot[~np.isnan(iR_pix_plot)]
            fR_pix_plot = fR_pix_plot[~np.isnan(fR_pix_plot)]
            e11c = e11c[~np.isnan(e11c)]
            e22c = e22c[~np.isnan(e22c)]
            e33c = e33c[~np.isnan(e33c)]
            e12c = e12c[~np.isnan(e12c)]
            e13c = e13c[~np.isnan(e13c)]
            e23c = e23c[~np.isnan(e23c)]
            e11s = e11s[~np.isnan(e11s)]
            e22s = e22s[~np.isnan(e22s)]
            e33s = e33s[~np.isnan(e33s)]
            e12s = e12s[~np.isnan(e12s)]
            e13s = e13s[~np.isnan(e13s)]
            e23s = e23s[~np.isnan(e23s)]
            
            try:
                title = "Number of spots and matching rate"
                fig = plt.figure()
                axs = fig.subplots(1, 2)
                axs[0].set_title("Number of spots", loc='center', fontsize=8)
                axs[0].hist(spots_len_plot, bins=bins)
                axs[0].set_ylabel('Frequency', fontsize=8)
                axs[0].tick_params(axis='both', which='major', labelsize=8)
                axs[0].tick_params(axis='both', which='minor', labelsize=8)
                axs[1].set_title("matching rate", loc='center', fontsize=8)
                axs[1].hist(mr_plot, bins=bins)
                axs[1].set_ylabel('Frequency', fontsize=8)
                axs[1].tick_params(axis='both', which='major', labelsize=8)
                axs[1].tick_params(axis='both', which='minor', labelsize=8)
                plt.tight_layout()
                plt.savefig(model_direc+ "//"+title+"_"+material_id[matid]+'.png', format='png', dpi=1000) 
                plt.close(fig)
            except:
                pass
            
            try:
                title = "Initial and Final residues"
                fig = plt.figure()
                axs = fig.subplots(1, 2)
                axs[0].set_title("Initial residues", loc='center', fontsize=8)
                axs[0].hist(iR_pix_plot, bins=bins)
                axs[0].set_ylabel('Frequency', fontsize=8)
                axs[0].tick_params(axis='both', which='major', labelsize=8)
                axs[0].tick_params(axis='both', which='minor', labelsize=8)
                axs[1].set_title("Final residues", loc='center', fontsize=8)
                axs[1].hist(fR_pix_plot, bins=bins)
                axs[1].set_ylabel('Frequency', fontsize=8)
                axs[1].tick_params(axis='both', which='major', labelsize=8)
                axs[1].tick_params(axis='both', which='minor', labelsize=8)
                plt.tight_layout()
                plt.savefig(model_direc+ "//"+title+"_"+material_id[matid]+'.png',format='png', dpi=1000) 
                plt.close(fig)
            except:
                pass            
            
            try:
                title = "strain Crystal reference"+" "+material_id[matid]
                fig = plt.figure()
                fig.suptitle(title, fontsize=10)
                axs = fig.subplots(2, 3)
                axs[0, 0].set_title(r"$\epsilon_{11}$ (%)", loc='center', fontsize=8)
                logdata = e11c #np.log(e11c)
                xmin = logdata.min()
                xmax = logdata.max()
                x1 = np.linspace(xmin, xmax, 1000)
                estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
                pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
                axs[0, 0].axvline(x=estimated_mu, c="k")
                axs[0, 0].plot(x1, pdf, 'r')
                axs[0, 0].hist(logdata, bins=bins, density=True, alpha=0.8)
                mu_sdc.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
                axs[0, 0].set_ylabel('Frequency', fontsize=8)
                axs[0, 0].tick_params(axis='both', which='major', labelsize=8)
                axs[0, 0].tick_params(axis='both', which='minor', labelsize=8)
                
                axs[0, 1].set_title(r"$\epsilon_{22}$ (%)", loc='center', fontsize=8)
                logdata = e22c #np.log(e22c)
                xmin = logdata.min()
                xmax = logdata.max()
                x1 = np.linspace(xmin, xmax, 1000)
                estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
                pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
                axs[0, 1].axvline(x=estimated_mu, c="k")
                axs[0, 1].plot(x1, pdf, 'r')
                axs[0, 1].hist(logdata, bins=bins, density=True, alpha=0.8)
                mu_sdc.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
                # axs[0, 1].hist(e22c, bins=bins)
                axs[0, 1].set_ylabel('Frequency', fontsize=8)
                axs[0, 1].tick_params(axis='both', which='major', labelsize=8)
                axs[0, 1].tick_params(axis='both', which='minor', labelsize=8)
                
                axs[0, 2].set_title(r"$\epsilon_{33}$ (%)", loc='center', fontsize=8)
                logdata = e33c #np.log(e33c)
                xmin = logdata.min()
                xmax = logdata.max()
                x1 = np.linspace(xmin, xmax, 1000)
                estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
                pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
                axs[0, 2].axvline(x=estimated_mu, c="k")
                axs[0, 2].plot(x1, pdf, 'r')
                axs[0, 2].hist(logdata, bins=bins, density=True, alpha=0.8)
                mu_sdc.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
                # axs[0, 2].hist(e33c, bins=bins)
                axs[0, 2].set_ylabel('Frequency', fontsize=8)
                axs[0, 2].tick_params(axis='both', which='major', labelsize=8)
                axs[0, 2].tick_params(axis='both', which='minor', labelsize=8)
                
                axs[1, 0].set_title(r"$\epsilon_{12}$ (%)", loc='center', fontsize=8)
                logdata = e12c#np.log(e12c)
                xmin = logdata.min()
                xmax = logdata.max()
                x1 = np.linspace(xmin, xmax, 1000)
                estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
                pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
                axs[1, 0].axvline(x=estimated_mu, c="k")
                axs[1, 0].plot(x1, pdf, 'r')
                axs[1, 0].hist(logdata, bins=bins, density=True, alpha=0.8)
                mu_sdc.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
                # axs[1, 0].hist(e12c, bins=bins)
                axs[1, 0].set_ylabel('Frequency', fontsize=8)
                axs[1, 0].tick_params(axis='both', which='major', labelsize=8)
                axs[1, 0].tick_params(axis='both', which='minor', labelsize=8)
                
                axs[1, 1].set_title(r"$\epsilon_{13}$ (%)", loc='center', fontsize=8)
                logdata = e13c#np.log(e13c)
                xmin = logdata.min()
                xmax = logdata.max()
                x1 = np.linspace(xmin, xmax, 1000)
                estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
                pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
                axs[1, 1].axvline(x=estimated_mu, c="k")
                axs[1, 1].plot(x1, pdf, 'r')
                axs[1, 1].hist(logdata, bins=bins, density=True, alpha=0.8)
                mu_sdc.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
                # axs[1, 1].hist(e13c, bins=bins)
                axs[1, 1].set_ylabel('Frequency', fontsize=8)
                axs[1, 1].tick_params(axis='both', which='major', labelsize=8)
                axs[1, 1].tick_params(axis='both', which='minor', labelsize=8)
                
                axs[1, 2].set_title(r"$\epsilon_{23}$ (%)", loc='center', fontsize=8)
                logdata = e23c#np.log(e23c)
                xmin = logdata.min()
                xmax = logdata.max()
                x1 = np.linspace(xmin, xmax, 1000)
                estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
                pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
                axs[1, 2].axvline(x=estimated_mu, c="k")
                axs[1, 2].plot(x1, pdf, 'r')
                axs[1, 2].hist(logdata, bins=bins, density=True, alpha=0.8)
                # axs[1, 2].hist(e23c, bins=bins)
                axs[1, 2].set_ylabel('Frequency', fontsize=8)
                axs[1, 2].tick_params(axis='both', which='major', labelsize=8)
                axs[1, 2].tick_params(axis='both', which='minor', labelsize=8)
                mu_sdc.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
                plt.tight_layout()
                plt.savefig(model_direc+ "//"+title+'.png', format='png', dpi=1000) 
                plt.close(fig)
            except:
                pass
        
            try:
                title = "strain Sample reference"+" "+material_id[matid]
                fig = plt.figure()
                fig.suptitle(title, fontsize=10)
                axs = fig.subplots(2, 3)
                axs[0, 0].set_title(r"$\epsilon_{11}$ (%)", loc='center', fontsize=8)
                logdata = e11s #np.log(e11c)
                xmin = logdata.min()
                xmax = logdata.max()
                x1 = np.linspace(xmin, xmax, 1000)
                estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
                pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
                axs[0, 0].axvline(x=estimated_mu, c="k")
                axs[0, 0].plot(x1, pdf, 'r')
                axs[0, 0].hist(logdata, bins=bins, density=True, alpha=0.8)
                # axs[0, 0].hist(e11s, bins=bins)
                axs[0, 0].set_ylabel('Frequency', fontsize=8)
                axs[0, 0].tick_params(axis='both', which='major', labelsize=8)
                axs[0, 0].tick_params(axis='both', which='minor', labelsize=8)
                
                mu_sd.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
                
                axs[0, 1].set_title(r"$\epsilon_{22}$ (%)", loc='center', fontsize=8)
                logdata = e22s #np.log(e22c)
                xmin = logdata.min()
                xmax = logdata.max()
                x1 = np.linspace(xmin, xmax, 1000)
                estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
                pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
                axs[0, 1].axvline(x=estimated_mu, c="k")
                axs[0, 1].plot(x1, pdf, 'r')
                axs[0, 1].hist(logdata, bins=bins, density=True, alpha=0.8)
                # axs[0, 1].hist(e22s, bins=bins)
                axs[0, 1].set_ylabel('Frequency', fontsize=8)
                axs[0, 1].tick_params(axis='both', which='major', labelsize=8)
                axs[0, 1].tick_params(axis='both', which='minor', labelsize=8)
                
                mu_sd.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
                
                axs[0, 2].set_title(r"$\epsilon_{33}$ (%)", loc='center', fontsize=8)
                logdata = e33s #np.log(e33c)
                xmin = logdata.min()
                xmax = logdata.max()
                x1 = np.linspace(xmin, xmax, 1000)
                estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
                pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
                axs[0, 2].axvline(x=estimated_mu, c="k")
                axs[0, 2].plot(x1, pdf, 'r')
                axs[0, 2].hist(logdata, bins=bins, density=True, alpha=0.8)
                # axs[0, 2].hist(e33s, bins=bins)
                axs[0, 2].set_ylabel('Frequency', fontsize=8)
                axs[0, 2].tick_params(axis='both', which='major', labelsize=8)
                axs[0, 2].tick_params(axis='both', which='minor', labelsize=8)
                
                mu_sd.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
                
                axs[1, 0].set_title(r"$\epsilon_{12}$ (%)", loc='center', fontsize=8)
                logdata = e12s#np.log(e12c)
                xmin = logdata.min()
                xmax = logdata.max()
                x1 = np.linspace(xmin, xmax, 1000)
                estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
                pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
                axs[1, 0].axvline(x=estimated_mu, c="k")
                axs[1, 0].plot(x1, pdf, 'r')
                axs[1, 0].hist(logdata, bins=bins, density=True, alpha=0.8)
                # axs[1, 0].hist(e12s, bins=bins)
                axs[1, 0].set_ylabel('Frequency', fontsize=8)
                axs[1, 0].tick_params(axis='both', which='major', labelsize=8)
                axs[1, 0].tick_params(axis='both', which='minor', labelsize=8)
                
                mu_sd.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
                
                axs[1, 1].set_title(r"$\epsilon_{13}$ (%)", loc='center', fontsize=8)
                logdata = e13s#np.log(e13c)
                xmin = logdata.min()
                xmax = logdata.max()
                x1 = np.linspace(xmin, xmax, 1000)
                estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
                pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
                axs[1, 1].axvline(x=estimated_mu, c="k")
                axs[1, 1].plot(x1, pdf, 'r')
                axs[1, 1].hist(logdata, bins=bins, density=True, alpha=0.8)
                # axs[1, 1].hist(e13s, bins=bins)
                axs[1, 1].set_ylabel('Frequency', fontsize=8)
                axs[1, 1].tick_params(axis='both', which='major', labelsize=8)
                axs[1, 1].tick_params(axis='both', which='minor', labelsize=8)
                
                mu_sd.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
                
                axs[1, 2].set_title(r"$\epsilon_{23}$ (%)", loc='center', fontsize=8)
                logdata = e23s#np.log(e23c)
                xmin = logdata.min()
                xmax = logdata.max()
                x1 = np.linspace(xmin, xmax, 1000)
                estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
                pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
                axs[1, 2].axvline(x=estimated_mu, c="k")
                axs[1, 2].plot(x1, pdf, 'r')
                axs[1, 2].hist(logdata, bins=bins, density=True, alpha=0.8)
                # axs[1, 2].hist(e23s, bins=bins)
                axs[1, 2].set_ylabel('Frequency', fontsize=8)
                axs[1, 2].tick_params(axis='both', which='major', labelsize=8)
                axs[1, 2].tick_params(axis='both', which='minor', labelsize=8)
                
                mu_sd.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
                
                plt.tight_layout()
                plt.savefig(model_direc+ "//"+title+'.png', format='png', dpi=1000) 
                plt.close(fig)  
            except:
                pass

    if material_ == material1_:
        matid = 0
        for index in range(len(strain_matrix)):
            nan_index = np.where(match_rate[index][0] <= match_rate_threshold)[0]
        
            strain_matrix_plot = np.copy(strain_matrixs[index][0])
            strain_matrix_plot[nan_index,:,:] = np.nan             
        
            fig = plt.figure(figsize=(11.69,8.27), dpi=100)
            bottom, top = 0.1, 0.9
            left, right = 0.1, 0.8
            fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
            
            vmin, vmax = mu_sd[matid*6]
            axs = fig.subplots(2, 3)
            axs[0, 0].set_title(r"$\epsilon_{11}$ (%)", loc='center', fontsize=8)
            im=axs[0, 0].imshow(strain_matrix_plot[:,0,0].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
            axs[0, 0].set_xticks([])
            axs[0, 0].set_yticks([])
            divider = make_axes_locatable(axs[0,0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.ax.tick_params(labelsize=8) 
            
            vmin, vmax = mu_sd[matid*6+1]
            axs[0, 1].set_title(r"$\epsilon_{22}$ (%)", loc='center', fontsize=8)
            im=axs[0, 1].imshow(strain_matrix_plot[:,1,1].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
            divider = make_axes_locatable(axs[0,1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.ax.tick_params(labelsize=8) 
            
            vmin, vmax = mu_sd[matid*6+2]
            axs[0, 2].set_title(r"$\epsilon_{33}$ (%)", loc='center', fontsize=8)
            im=axs[0, 2].imshow(strain_matrix_plot[:,2,2].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
            divider = make_axes_locatable(axs[0,2])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.ax.tick_params(labelsize=8) 
            
            vmin, vmax = mu_sd[matid*6+3]
            axs[1, 0].set_title(r"$\epsilon_{12}$ (%)", loc='center', fontsize=8)
            im=axs[1, 0].imshow(strain_matrix_plot[:,0,1].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
            axs[1, 0].set_xticks([])
            axs[1, 0].set_yticks([])
            divider = make_axes_locatable(axs[1,0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.ax.tick_params(labelsize=8) 
            
            vmin, vmax = mu_sd[matid*6+4]
            axs[1, 1].set_title(r"$\epsilon_{13}$ (%)", loc='center', fontsize=8)
            im=axs[1, 1].imshow(strain_matrix_plot[:,0,2].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
            axs[1, 1].set_xticks([])
            divider = make_axes_locatable(axs[1,1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.ax.tick_params(labelsize=8) 
            
            vmin, vmax = mu_sd[matid*6+5]
            axs[1, 2].set_title(r"$\epsilon_{23}$ (%)", loc='center', fontsize=8)
            im = axs[1, 2].imshow(strain_matrix_plot[:,1,2].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
            axs[1, 2].set_xticks([]) 
            divider = make_axes_locatable(axs[1,2])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.ax.tick_params(labelsize=8) 
        
            for ax in axs.flat:
                ax.label_outer()
        
            plt.savefig(model_direc+ '//figure_strain_UBsample_UB'+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
            plt.close(fig)
               
            strain_matrix_plot = np.copy(strain_matrix[index][0])
            strain_matrix_plot[nan_index,:,:] = np.nan             
        
            fig = plt.figure(figsize=(11.69,8.27), dpi=100)
            bottom, top = 0.1, 0.9
            left, right = 0.1, 0.8
            fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
            
            vmin, vmax = mu_sdc[matid*6]
            axs = fig.subplots(2, 3)
            axs[0, 0].set_title(r"$\epsilon_{11}$ (%)", loc='center', fontsize=8)
            im=axs[0, 0].imshow(strain_matrix_plot[:,0,0].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
            axs[0, 0].set_xticks([])
            axs[0, 0].set_yticks([])
            divider = make_axes_locatable(axs[0,0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.ax.tick_params(labelsize=8) 
            
            vmin, vmax = mu_sdc[matid*6+1]
            axs[0, 1].set_title(r"$\epsilon_{22}$ (%)", loc='center', fontsize=8)
            im=axs[0, 1].imshow(strain_matrix_plot[:,1,1].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
            divider = make_axes_locatable(axs[0,1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.ax.tick_params(labelsize=8) 
            
            vmin, vmax = mu_sdc[matid*6+2]
            axs[0, 2].set_title(r"$\epsilon_{33}$ (%)", loc='center', fontsize=8)
            im=axs[0, 2].imshow(strain_matrix_plot[:,2,2].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
            divider = make_axes_locatable(axs[0,2])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.ax.tick_params(labelsize=8) 
            
            vmin, vmax = mu_sdc[matid*6+3]
            axs[1, 0].set_title(r"$\epsilon_{12}$ (%)", loc='center', fontsize=8)
            im=axs[1, 0].imshow(strain_matrix_plot[:,0,1].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
            axs[1, 0].set_xticks([])
            axs[1, 0].set_yticks([])
            divider = make_axes_locatable(axs[1,0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.ax.tick_params(labelsize=8) 
            
            vmin, vmax = mu_sdc[matid*6+4]
            axs[1, 1].set_title(r"$\epsilon_{13}$ (%)", loc='center', fontsize=8)
            im=axs[1, 1].imshow(strain_matrix_plot[:,0,2].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
            axs[1, 1].set_xticks([])
            divider = make_axes_locatable(axs[1,1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.ax.tick_params(labelsize=8) 
            
            vmin, vmax = mu_sdc[matid*6+5]
            axs[1, 2].set_title(r"$\epsilon_{23}$ (%)", loc='center', fontsize=8)
            im = axs[1, 2].imshow(strain_matrix_plot[:,1,2].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
            axs[1, 2].set_xticks([]) 
            divider = make_axes_locatable(axs[1,2])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.ax.tick_params(labelsize=8) 
        
            for ax in axs.flat:
                ax.label_outer()
        
            plt.savefig(model_direc+ '//figure_strain_UBcrystal_UB'+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
            plt.close(fig)
            
            col_plot = np.copy(col[index][0])
            col_plot[nan_index,:] = np.nan,np.nan,np.nan
            col_plot = col_plot.reshape((lim_x, lim_y, 3))
        
            colx_plot = np.copy(colx[index][0])
            colx_plot[nan_index,:] = np.nan,np.nan,np.nan
            colx_plot = colx_plot.reshape((lim_x, lim_y,3))
            
            coly_plot = np.copy(coly[index][0])
            coly_plot[nan_index,:] = np.nan,np.nan,np.nan
            coly_plot = coly_plot.reshape((lim_x, lim_y,3))
            
            fig = plt.figure(figsize=(11.69,8.27), dpi=100)
            bottom, top = 0.1, 0.9
            left, right = 0.1, 0.8
            fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
        
            axs = fig.subplots(1, 3)
            axs[0].set_title(r"IPF Z map", loc='center', fontsize=8)
            axs[0].imshow(col_plot, origin='lower')
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            
            axs[1].set_title(r"IPF Y map", loc='center', fontsize=8)
            axs[1].imshow(coly_plot, origin='lower')
            axs[1].set_xticks([])
            axs[1].set_yticks([])
            
            axs[2].set_title(r"IPF X map", loc='center', fontsize=8)
            im = axs[2].imshow(colx_plot, origin='lower')
            axs[2].set_xticks([])
            axs[2].set_yticks([])
        
            for ax in axs.flat:
                ax.label_outer()
        
            plt.savefig(model_direc+ '//IPF_map_UB'+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
            plt.close(fig)
            
            
            col_plot = np.copy(col[index][0])
            col_plot[nan_index,:] = np.nan,np.nan,np.nan
            col_plot = col_plot.reshape((lim_x, lim_y, 3))
        
            mr_plot = np.copy(match_rate[index][0])
            mr_plot[nan_index,:] = np.nan
            mr_plot = mr_plot.reshape((lim_x, lim_y))
            
            mat_glob = np.copy(mat_global[index][0])
            mat_glob[nan_index,:] = np.nan
            mat_glob = mat_glob.reshape((lim_x, lim_y))
            
            fig = plt.figure(figsize=(11.69,8.27), dpi=100)
            bottom, top = 0.1, 0.9
            left, right = 0.1, 0.8
            fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
        
            axs = fig.subplots(1, 3)
            axs[0].set_title(r"IPF Z map", loc='center', fontsize=8)
            axs[0].imshow(col_plot, origin='lower')
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            
            axs[1].set_title(r"Material Index", loc='center', fontsize=8)
            im = axs[1].imshow(mat_glob, origin='lower', vmin=0, vmax=1)
            axs[1].set_xticks([])
            axs[1].set_yticks([])
            
            divider = make_axes_locatable(axs[1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
            
            axs[2].set_title(r"Matching rate", loc='center', fontsize=8)
            im = axs[2].imshow(mr_plot, origin='lower', cmap=plt.cm.jet, vmin=0, vmax=100)
            axs[2].set_xticks([])
            axs[2].set_yticks([])
            
            divider = make_axes_locatable(axs[2])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
        
            for ax in axs.flat:
                ax.label_outer()
        
            plt.savefig(model_direc+ "//figure_global_UB"+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
            plt.close(fig)
            
            spots_len_plot = np.copy(spots_len[index][0])
            spots_len_plot[nan_index,:] = np.nan
            spots_len_plot = spots_len_plot.reshape((lim_x, lim_y))
            
            iR_pix_plot = np.copy(iR_pix[index][0])
            iR_pix_plot[nan_index,:] = np.nan
            iR_pix_plot = iR_pix_plot.reshape((lim_x, lim_y))
            
            fR_pix_plot = np.copy(fR_pix[index][0])
            fR_pix_plot[nan_index,:] = np.nan
            fR_pix_plot = fR_pix_plot.reshape((lim_x, lim_y))
            
            fig = plt.figure(figsize=(11.69,8.27), dpi=100)
            bottom, top = 0.1, 0.9
            left, right = 0.1, 0.8
            fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
        
            axs = fig.subplots(1, 3)
            axs[0].set_title(r"Number of spots detected", loc='center', fontsize=8)
            im = axs[0].imshow(spots_len_plot, origin='lower', cmap=plt.cm.jet)
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            
            divider = make_axes_locatable(axs[0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
        
            axs[1].set_title(r"Initial pixel residues", loc='center', fontsize=8)
            im = axs[1].imshow(iR_pix_plot, origin='lower', cmap=plt.cm.jet)
            axs[1].set_xticks([])
            axs[1].set_yticks([])
            
            divider = make_axes_locatable(axs[1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
        
            axs[2].set_title(r"Final pixel residues", loc='center', fontsize=8)
            im = axs[2].imshow(fR_pix_plot, origin='lower', cmap=plt.cm.jet)
            axs[2].set_xticks([])
            axs[2].set_yticks([])
            
            divider = make_axes_locatable(axs[2])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
        
            for ax in axs.flat:
                ax.label_outer()
        
            plt.savefig(model_direc+'//figure_mr_ir_fr_UB'+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
            plt.close(fig)
            
            try:
                a,b,c,alp,bet,gam = [],[],[],[],[],[]
                
                constantlength = "a"
                if ("a" in strain_free_parameters) and ("b" in strain_free_parameters) and ("c" in strain_free_parameters):
                    constantlength = "a"                    
                elif ("b" not in strain_free_parameters) and additional_expression[0]=="none" and\
                    "b" not in additional_expression[0]:
                    constantlength = "b"
                elif ("c" not in strain_free_parameters):
                    constantlength = "c"
                    
                for irot in range(len(rotation_matrix1[index][0])):
                    lattice_parameter_direct_strain = CP.computeLatticeParameters_from_UB(rotation_matrix1[index][0][irot,:,:], 
                                                                                          material_, 
                                                                                          constantlength, 
                                                                                          dictmaterials=dictLT.dict_Materials)
                    a.append(lattice_parameter_direct_strain[0])
                    b.append(lattice_parameter_direct_strain[1])
                    c.append(lattice_parameter_direct_strain[2])
                    alp.append(lattice_parameter_direct_strain[3])
                    bet.append(lattice_parameter_direct_strain[4])
                    gam.append(lattice_parameter_direct_strain[5])
                
                logdata = np.array(a)
                logdata = logdata[~np.isnan(logdata)]
                rangemina, rangemaxa = np.min(logdata)-0.01, np.max(logdata)+0.01
                logdata = np.array(b)
                logdata = logdata[~np.isnan(logdata)]
                rangeminb, rangemaxb = np.min(logdata)-0.01, np.max(logdata)+0.01
                logdata = np.array(c)
                logdata = logdata[~np.isnan(logdata)]
                rangeminc, rangemaxc = np.min(logdata)-0.01, np.max(logdata)+0.01
                logdata = np.array(alp)
                logdata = logdata[~np.isnan(logdata)]
                rangeminal, rangemaxal = np.min(logdata)-0.01, np.max(logdata)+0.01
                logdata = np.array(bet)
                logdata = logdata[~np.isnan(logdata)]
                rangeminbe, rangemaxbe = np.min(logdata)-0.01, np.max(logdata)+0.01
                logdata = np.array(gam)
                logdata = logdata[~np.isnan(logdata)]
                rangeminga, rangemaxga = np.min(logdata)-0.01, np.max(logdata)+0.01
        
                fig = plt.figure(figsize=(11.69,8.27), dpi=100)
                bottom, top = 0.1, 0.9
                left, right = 0.1, 0.8
                fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
                
                vmin = rangemina
                vmax = rangemaxa
                axs = fig.subplots(2, 3)
                axs[0, 0].set_title(r"$a$", loc='center', fontsize=8)
                strain_matrix_plot = np.array(a)
                im=axs[0, 0].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                axs[0, 0].set_xticks([])
                axs[0, 0].set_yticks([])
                divider = make_axes_locatable(axs[0,0])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
        
                vmin = rangeminb
                vmax = rangemaxb
                axs[0, 1].set_title(r"$b$", loc='center', fontsize=8)
                strain_matrix_plot = np.array(b)
                im=axs[0, 1].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                divider = make_axes_locatable(axs[0,1])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
        
                vmin = rangeminc
                vmax = rangemaxc
                axs[0, 2].set_title(r"$c$", loc='center', fontsize=8)
                strain_matrix_plot = np.array(c)
                im=axs[0, 2].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                divider = make_axes_locatable(axs[0,2])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
        
                vmin = rangeminal
                vmax = rangemaxal
                axs[1, 0].set_title(r"$\alpha$", loc='center', fontsize=8)
                strain_matrix_plot = np.array(alp)
                im=axs[1, 0].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                axs[1, 0].set_xticks([])
                axs[1, 0].set_yticks([])
                divider = make_axes_locatable(axs[1,0])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
        
                vmin = rangeminbe
                vmax = rangemaxbe
                axs[1, 1].set_title(r"$\beta$", loc='center', fontsize=8)
                strain_matrix_plot = np.array(bet)
                im=axs[1, 1].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                axs[1, 1].set_xticks([])
                divider = make_axes_locatable(axs[1,1])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
        
                vmin = rangeminga
                vmax = rangemaxga
                axs[1, 2].set_title(r"$\gamma$", loc='center', fontsize=8)
                strain_matrix_plot = np.array(gam)
                im = axs[1, 2].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                axs[1, 2].set_xticks([]) 
                divider = make_axes_locatable(axs[1,2])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.formatter.set_useOffset(False)
                cbar.ax.tick_params(labelsize=8) 
                
                for ax in axs.flat:
                    ax.label_outer()
                plt.savefig(model_direc+ "//"+'figure_unitcell_'+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
                plt.close(fig)
            except:
                pass
            
            try:
                latticeparams = dictLT.dict_Materials[material_][1]

                a,b,c,alp,bet,gam = [],[],[],[],[],[]
        
                constantlength = "a"
                if ("a" in strain_free_parameters) and ("b" in strain_free_parameters) and ("c" in strain_free_parameters):
                    constantlength = "a"    
                elif ("b" not in strain_free_parameters) and additional_expression[0]=="none" and \
                    "b" not in additional_expression[0]:
                    constantlength = "b"
                elif ("c" not in strain_free_parameters):
                    constantlength = "c"
                    
                for irot in range(len(rotation_matrix1[index][0])):
                    lattice_parameter_direct_strain = CP.computeLatticeParameters_from_UB(rotation_matrix1[index][0][irot,:,:], 
                                                                                          material_, 
                                                                                          constantlength, 
                                                                                          dictmaterials=dictLT.dict_Materials)
                    a.append(lattice_parameter_direct_strain[0])
                    b.append(lattice_parameter_direct_strain[1])
                    c.append(lattice_parameter_direct_strain[2])
                    alp.append(lattice_parameter_direct_strain[3])
                    bet.append(lattice_parameter_direct_strain[4])
                    gam.append(lattice_parameter_direct_strain[5])
        
                logdata = np.array(a) - latticeparams[0]
                logdata = logdata[~np.isnan(logdata)]
                rangemina, rangemaxa = np.min(logdata) - 0.01e-2, np.max(logdata) + 0.01e-2
                logdata = np.array(b) - latticeparams[1]
                logdata = logdata[~np.isnan(logdata)]
                rangeminb, rangemaxb = np.min(logdata) - 0.01e-2, np.max(logdata) + 0.01e-2
                logdata = np.array(c) - latticeparams[2]
                logdata = logdata[~np.isnan(logdata)]
                rangeminc, rangemaxc = np.min(logdata) - 0.01e-2, np.max(logdata) + 0.01e-2
                logdata = np.array(alp) - latticeparams[3]
                logdata = logdata[~np.isnan(logdata)]
                rangeminal, rangemaxal = np.min(logdata) - 0.01, np.max(logdata) + 0.01
                logdata = np.array(bet) - latticeparams[4]
                logdata = logdata[~np.isnan(logdata)]
                rangeminbe, rangemaxbe = np.min(logdata) - 0.01, np.max(logdata) + 0.01
                logdata = np.array(gam) - latticeparams[5]
                logdata = logdata[~np.isnan(logdata)]
                rangeminga, rangemaxga = np.min(logdata) - 0.01, np.max(logdata) + 0.01
        
                fig = plt.figure(figsize=(11.69,8.27), dpi=100)
                bottom, top = 0.1, 0.9
                left, right = 0.1, 0.8
                fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
        
                vmin = rangemina
                vmax = rangemaxa
                axs = fig.subplots(2, 3)
                axs[0, 0].set_title(r"$a$", loc='center', fontsize=8)
                strain_matrix_plot = np.array(a) - latticeparams[0]
                im=axs[0, 0].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                axs[0, 0].set_xticks([])
                axs[0, 0].set_yticks([])
                divider = make_axes_locatable(axs[0,0])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
        
                vmin = rangeminb
                vmax = rangemaxb
                axs[0, 1].set_title(r"$b$", loc='center', fontsize=8)
                strain_matrix_plot = np.array(b) - latticeparams[1]
                im=axs[0, 1].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                divider = make_axes_locatable(axs[0,1])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
        
                vmin = rangeminc
                vmax = rangemaxc
                axs[0, 2].set_title(r"$c$", loc='center', fontsize=8)
                strain_matrix_plot = np.array(c) - latticeparams[2]
                im=axs[0, 2].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                divider = make_axes_locatable(axs[0,2])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
        
                vmin = rangeminal
                vmax = rangemaxal
                axs[1, 0].set_title(r"$\alpha$", loc='center', fontsize=8)
                strain_matrix_plot = np.array(alp) - latticeparams[3]
                im=axs[1, 0].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                axs[1, 0].set_xticks([])
                axs[1, 0].set_yticks([])
                divider = make_axes_locatable(axs[1,0])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
        
                vmin = rangeminbe
                vmax = rangemaxbe
                axs[1, 1].set_title(r"$\beta$", loc='center', fontsize=8)
                strain_matrix_plot = np.array(bet) - latticeparams[4]
                im=axs[1, 1].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                axs[1, 1].set_xticks([])
                divider = make_axes_locatable(axs[1,1])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
        
                vmin = rangeminga
                vmax = rangemaxga
                axs[1, 2].set_title(r"$\gamma$", loc='center', fontsize=8)
                strain_matrix_plot = np.array(gam) - latticeparams[5]
                im = axs[1, 2].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                axs[1, 2].set_xticks([]) 
                divider = make_axes_locatable(axs[1,2])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.formatter.set_useOffset(False)
                cbar.ax.tick_params(labelsize=8) 
        
                for ax in axs.flat:
                    ax.label_outer()
                plt.savefig(model_direc + "//" + 'figure_unitcell_relative_'+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
                plt.close(fig)
            except:
                pass
    else:    
    
        for matid in range(2):
            for index in range(len(strain_matrix)):
                nan_index1 = np.where(match_rate[index][0] <= match_rate_threshold)[0]
                mat_id_index = np.where(mat_global[index][0] != matid+1)[0]
                nan_index = np.hstack((mat_id_index,nan_index1))
                nan_index = np.unique(nan_index)
            
                strain_matrix_plot = np.copy(strain_matrixs[index][0])
                strain_matrix_plot[nan_index,:,:] = np.nan             
            
                fig = plt.figure(figsize=(11.69,8.27), dpi=100)
                bottom, top = 0.1, 0.9
                left, right = 0.1, 0.8
                fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
                
                try:
                    vmin, vmax = mu_sd[matid*6]
                    axs = fig.subplots(2, 3)
                    axs[0, 0].set_title(r"$\epsilon_{11}$ (%)", loc='center', fontsize=8)
                    im=axs[0, 0].imshow(strain_matrix_plot[:,0,0].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                    axs[0, 0].set_xticks([])
                    axs[0, 0].set_yticks([])
                    divider = make_axes_locatable(axs[0,0])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.ax.tick_params(labelsize=8) 
                    
                    vmin, vmax = mu_sd[matid*6+1]
                    axs[0, 1].set_title(r"$\epsilon_{22}$ (%)", loc='center', fontsize=8)
                    im=axs[0, 1].imshow(strain_matrix_plot[:,1,1].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                    divider = make_axes_locatable(axs[0,1])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.ax.tick_params(labelsize=8) 
                    
                    vmin, vmax = mu_sd[matid*6+2]
                    axs[0, 2].set_title(r"$\epsilon_{33}$ (%)", loc='center', fontsize=8)
                    im=axs[0, 2].imshow(strain_matrix_plot[:,2,2].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                    divider = make_axes_locatable(axs[0,2])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.ax.tick_params(labelsize=8) 
                    
                    vmin, vmax = mu_sd[matid*6+3]
                    axs[1, 0].set_title(r"$\epsilon_{12}$ (%)", loc='center', fontsize=8)
                    im=axs[1, 0].imshow(strain_matrix_plot[:,0,1].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                    axs[1, 0].set_xticks([])
                    axs[1, 0].set_yticks([])
                    divider = make_axes_locatable(axs[1,0])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.ax.tick_params(labelsize=8) 
                    
                    vmin, vmax = mu_sd[matid*6+4]
                    axs[1, 1].set_title(r"$\epsilon_{13}$ (%)", loc='center', fontsize=8)
                    im=axs[1, 1].imshow(strain_matrix_plot[:,0,2].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                    axs[1, 1].set_xticks([])
                    divider = make_axes_locatable(axs[1,1])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.ax.tick_params(labelsize=8) 
                    
                    vmin, vmax = mu_sd[matid*6+5]
                    axs[1, 2].set_title(r"$\epsilon_{23}$ (%)", loc='center', fontsize=8)
                    im = axs[1, 2].imshow(strain_matrix_plot[:,1,2].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                    axs[1, 2].set_xticks([]) 
                    divider = make_axes_locatable(axs[1,2])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.ax.tick_params(labelsize=8) 
                
                    for ax in axs.flat:
                        ax.label_outer()
                
                    plt.savefig(model_direc+ '//figure_strain_UBsample_mat'+str(matid)+"_UB"+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
                    plt.close(fig)
                except:
                    print("Error in strain plot")
                
                    
                strain_matrix_plot = np.copy(strain_matrix[index][0])
                strain_matrix_plot[nan_index,:,:] = np.nan             
                
                try:
                    fig = plt.figure(figsize=(11.69,8.27), dpi=100)
                    bottom, top = 0.1, 0.9
                    left, right = 0.1, 0.8
                    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
                    
                    vmin, vmax = mu_sdc[matid*6]
                    axs = fig.subplots(2, 3)
                    axs[0, 0].set_title(r"$\epsilon_{11}$ (%)", loc='center', fontsize=8)
                    im=axs[0, 0].imshow(strain_matrix_plot[:,0,0].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                    axs[0, 0].set_xticks([])
                    axs[0, 0].set_yticks([])
                    divider = make_axes_locatable(axs[0,0])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.ax.tick_params(labelsize=8) 
                    
                    vmin, vmax = mu_sdc[matid*6+1]
                    axs[0, 1].set_title(r"$\epsilon_{22}$ (%)", loc='center', fontsize=8)
                    im=axs[0, 1].imshow(strain_matrix_plot[:,1,1].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                    divider = make_axes_locatable(axs[0,1])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.ax.tick_params(labelsize=8) 
                    
                    vmin, vmax = mu_sdc[matid*6+2]
                    axs[0, 2].set_title(r"$\epsilon_{33}$ (%)", loc='center', fontsize=8)
                    im=axs[0, 2].imshow(strain_matrix_plot[:,2,2].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                    divider = make_axes_locatable(axs[0,2])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.ax.tick_params(labelsize=8) 
                    
                    vmin, vmax = mu_sdc[matid*6+3]
                    axs[1, 0].set_title(r"$\epsilon_{12}$ (%)", loc='center', fontsize=8)
                    im=axs[1, 0].imshow(strain_matrix_plot[:,0,1].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                    axs[1, 0].set_xticks([])
                    axs[1, 0].set_yticks([])
                    divider = make_axes_locatable(axs[1,0])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.ax.tick_params(labelsize=8) 
                    
                    vmin, vmax = mu_sdc[matid*6+4]
                    axs[1, 1].set_title(r"$\epsilon_{13}$ (%)", loc='center', fontsize=8)
                    im=axs[1, 1].imshow(strain_matrix_plot[:,0,2].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                    axs[1, 1].set_xticks([])
                    divider = make_axes_locatable(axs[1,1])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.ax.tick_params(labelsize=8) 
                    
                    vmin, vmax = mu_sdc[matid*6+5]
                    axs[1, 2].set_title(r"$\epsilon_{23}$ (%)", loc='center', fontsize=8)
                    im = axs[1, 2].imshow(strain_matrix_plot[:,1,2].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                    axs[1, 2].set_xticks([]) 
                    divider = make_axes_locatable(axs[1,2])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.ax.tick_params(labelsize=8) 
                
                    for ax in axs.flat:
                        ax.label_outer()
                
                    plt.savefig(model_direc+ '//figure_strain_UBcrystal_mat'+str(matid)+"_UB"+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
                    plt.close(fig)
                except:
                    print("Error in strain plots")
                    
                col_plot = np.copy(col[index][0])
                col_plot[nan_index,:] = np.nan,np.nan,np.nan
                col_plot = col_plot.reshape((lim_x, lim_y, 3))
            
                colx_plot = np.copy(colx[index][0])
                colx_plot[nan_index,:] = np.nan,np.nan,np.nan
                colx_plot = colx_plot.reshape((lim_x, lim_y,3))
                
                coly_plot = np.copy(coly[index][0])
                coly_plot[nan_index,:] = np.nan,np.nan,np.nan
                coly_plot = coly_plot.reshape((lim_x, lim_y,3))
                
                try:
                    fig = plt.figure(figsize=(11.69,8.27), dpi=100)
                    bottom, top = 0.1, 0.9
                    left, right = 0.1, 0.8
                    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
                    
                    axs = fig.subplots(1, 3)
                    axs[0].set_title(r"IPF Z map", loc='center', fontsize=8)
                    axs[0].imshow(col_plot, origin='lower')
                    axs[0].set_xticks([])
                    axs[0].set_yticks([])
                    
                    axs[1].set_title(r"IPF Y map", loc='center', fontsize=8)
                    axs[1].imshow(coly_plot, origin='lower')
                    axs[1].set_xticks([])
                    axs[1].set_yticks([])
                    
                    axs[2].set_title(r"IPF X map", loc='center', fontsize=8)
                    im = axs[2].imshow(colx_plot, origin='lower')
                    axs[2].set_xticks([])
                    axs[2].set_yticks([])
                
                    for ax in axs.flat:
                        ax.label_outer()
                
                    plt.savefig(model_direc+ '//IPF_map_mat'+str(matid)+"_UB"+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
                    plt.close(fig)
    
                    col_plot = np.copy(col[index][0])
                    col_plot[nan_index,:] = np.nan,np.nan,np.nan
                    col_plot = col_plot.reshape((lim_x, lim_y, 3))
                
                    mr_plot = np.copy(match_rate[index][0])
                    mr_plot[nan_index,:] = np.nan
                    mr_plot = mr_plot.reshape((lim_x, lim_y))
                    
                    mat_glob = np.copy(mat_global[index][0])
                    mat_glob[nan_index,:] = np.nan
                    mat_glob = mat_glob.reshape((lim_x, lim_y))
                    
                    fig = plt.figure(figsize=(11.69,8.27), dpi=100)
                    bottom, top = 0.1, 0.9
                    left, right = 0.1, 0.8
                    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
                
                    axs = fig.subplots(1, 3)
                    axs[0].set_title(r"IPF Z map", loc='center', fontsize=8)
                    axs[0].imshow(col_plot, origin='lower')
                    axs[0].set_xticks([])
                    axs[0].set_yticks([])
                    
                    axs[1].set_title(r"Material Index", loc='center', fontsize=8)
                    im = axs[1].imshow(mat_glob, origin='lower', vmin=0, vmax=2)
                    axs[1].set_xticks([])
                    axs[1].set_yticks([])
                    
                    divider = make_axes_locatable(axs[1])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im, cax=cax, orientation='vertical')
                    
                    axs[2].set_title(r"Matching rate", loc='center', fontsize=8)
                    im = axs[2].imshow(mr_plot, origin='lower', cmap=plt.cm.jet, vmin=0, vmax=100)
                    axs[2].set_xticks([])
                    axs[2].set_yticks([])
                    
                    divider = make_axes_locatable(axs[2])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im, cax=cax, orientation='vertical')
                
                    for ax in axs.flat:
                        ax.label_outer()
                
                    plt.savefig(model_direc+ "//figure_global_mat"+str(matid)+"_UB"+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
                    plt.close(fig)
                except:
                    print("Error in plots")
                spots_len_plot = np.copy(spots_len[index][0])
                spots_len_plot[nan_index,:] = np.nan
                spots_len_plot = spots_len_plot.reshape((lim_x, lim_y))
                
                iR_pix_plot = np.copy(iR_pix[index][0])
                iR_pix_plot[nan_index,:] = np.nan
                iR_pix_plot = iR_pix_plot.reshape((lim_x, lim_y))
                
                fR_pix_plot = np.copy(fR_pix[index][0])
                fR_pix_plot[nan_index,:] = np.nan
                fR_pix_plot = fR_pix_plot.reshape((lim_x, lim_y))
                
                try:
                    fig = plt.figure(figsize=(11.69,8.27), dpi=100)
                    bottom, top = 0.1, 0.9
                    left, right = 0.1, 0.8
                    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
                
                    axs = fig.subplots(1, 3)
                    axs[0].set_title(r"Number of spots detected", loc='center', fontsize=8)
                    im = axs[0].imshow(spots_len_plot, origin='lower', cmap=plt.cm.jet)
                    axs[0].set_xticks([])
                    axs[0].set_yticks([])
                    
                    divider = make_axes_locatable(axs[0])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im, cax=cax, orientation='vertical')
                
                    axs[1].set_title(r"Initial pixel residues", loc='center', fontsize=8)
                    im = axs[1].imshow(iR_pix_plot, origin='lower', cmap=plt.cm.jet)
                    axs[1].set_xticks([])
                    axs[1].set_yticks([])
                    
                    divider = make_axes_locatable(axs[1])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im, cax=cax, orientation='vertical')
                
                    axs[2].set_title(r"Final pixel residues", loc='center', fontsize=8)
                    im = axs[2].imshow(fR_pix_plot, origin='lower', cmap=plt.cm.jet)
                    axs[2].set_xticks([])
                    axs[2].set_yticks([])
                    
                    divider = make_axes_locatable(axs[2])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im, cax=cax, orientation='vertical')
                
                    for ax in axs.flat:
                        ax.label_outer()
                
                    plt.savefig(model_direc+'//figure_mr_ir_fr_mat'+str(matid)+"_UB"+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
                    plt.close(fig)
                except:
                    print("Error in plots")
                    
                try:
                    a,b,c,alp,bet,gam = [],[],[],[],[],[]
                    
                    constantlength = "a"
                    if ("a" in strain_free_parameters) and ("b" in strain_free_parameters) and ("c" in strain_free_parameters):
                        constantlength = "a"                    
                    elif ("b" not in strain_free_parameters) and additional_expression[0]=="none" and\
                        "b" not in additional_expression[0]:
                        constantlength = "b"
                    elif ("c" not in strain_free_parameters):
                        constantlength = "c"
                        
                    for irot in range(len(rotation_matrix1[index][0])):
                        lattice_parameter_direct_strain = CP.computeLatticeParameters_from_UB(rotation_matrix1[index][0][irot,:,:], 
                                                                                              material_, 
                                                                                              constantlength, 
                                                                                              dictmaterials=dictLT.dict_Materials)
                        a.append(lattice_parameter_direct_strain[0])
                        b.append(lattice_parameter_direct_strain[1])
                        c.append(lattice_parameter_direct_strain[2])
                        alp.append(lattice_parameter_direct_strain[3])
                        bet.append(lattice_parameter_direct_strain[4])
                        gam.append(lattice_parameter_direct_strain[5])
                    
                    logdata = np.array(a)
                    logdata = logdata[~np.isnan(logdata)]
                    rangemina, rangemaxa = np.min(logdata)-0.01, np.max(logdata)+0.01
                    logdata = np.array(b)
                    logdata = logdata[~np.isnan(logdata)]
                    rangeminb, rangemaxb = np.min(logdata)-0.01, np.max(logdata)+0.01
                    logdata = np.array(c)
                    logdata = logdata[~np.isnan(logdata)]
                    rangeminc, rangemaxc = np.min(logdata)-0.01, np.max(logdata)+0.01
                    logdata = np.array(alp)
                    logdata = logdata[~np.isnan(logdata)]
                    rangeminal, rangemaxal = np.min(logdata)-0.01, np.max(logdata)+0.01
                    logdata = np.array(bet)
                    logdata = logdata[~np.isnan(logdata)]
                    rangeminbe, rangemaxbe = np.min(logdata)-0.01, np.max(logdata)+0.01
                    logdata = np.array(gam)
                    logdata = logdata[~np.isnan(logdata)]
                    rangeminga, rangemaxga = np.min(logdata)-0.01, np.max(logdata)+0.01
            
                    fig = plt.figure(figsize=(11.69,8.27), dpi=100)
                    bottom, top = 0.1, 0.9
                    left, right = 0.1, 0.8
                    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
                    
                    vmin = rangemina
                    vmax = rangemaxa
                    axs = fig.subplots(2, 3)
                    axs[0, 0].set_title(r"$a$", loc='center', fontsize=8)
                    strain_matrix_plot = np.array(a)
                    im=axs[0, 0].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                    axs[0, 0].set_xticks([])
                    axs[0, 0].set_yticks([])
                    divider = make_axes_locatable(axs[0,0])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.ax.tick_params(labelsize=8) 
            
                    vmin = rangeminb
                    vmax = rangemaxb
                    axs[0, 1].set_title(r"$b$", loc='center', fontsize=8)
                    strain_matrix_plot = np.array(b)
                    im=axs[0, 1].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                    divider = make_axes_locatable(axs[0,1])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.ax.tick_params(labelsize=8) 
            
                    vmin = rangeminc
                    vmax = rangemaxc
                    axs[0, 2].set_title(r"$c$", loc='center', fontsize=8)
                    strain_matrix_plot = np.array(c)
                    im=axs[0, 2].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                    divider = make_axes_locatable(axs[0,2])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.ax.tick_params(labelsize=8) 
            
                    vmin = rangeminal
                    vmax = rangemaxal
                    axs[1, 0].set_title(r"$\alpha$", loc='center', fontsize=8)
                    strain_matrix_plot = np.array(alp)
                    im=axs[1, 0].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                    axs[1, 0].set_xticks([])
                    axs[1, 0].set_yticks([])
                    divider = make_axes_locatable(axs[1,0])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.ax.tick_params(labelsize=8) 
            
                    vmin = rangeminbe
                    vmax = rangemaxbe
                    axs[1, 1].set_title(r"$\beta$", loc='center', fontsize=8)
                    strain_matrix_plot = np.array(bet)
                    im=axs[1, 1].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                    axs[1, 1].set_xticks([])
                    divider = make_axes_locatable(axs[1,1])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.ax.tick_params(labelsize=8) 
            
                    vmin = rangeminga
                    vmax = rangemaxga
                    axs[1, 2].set_title(r"$\gamma$", loc='center', fontsize=8)
                    strain_matrix_plot = np.array(gam)
                    im = axs[1, 2].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                    axs[1, 2].set_xticks([]) 
                    divider = make_axes_locatable(axs[1,2])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.formatter.set_useOffset(False)
                    cbar.ax.tick_params(labelsize=8) 
                    
                    for ax in axs.flat:
                        ax.label_outer()
                    plt.savefig(model_direc+ "//"+'figure_unitcell_'+str(matid)+'_'+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
                    plt.close(fig)
                except:
                    pass
                
                try:
                    latticeparams = dictLT.dict_Materials[material_][1]

                    a,b,c,alp,bet,gam = [],[],[],[],[],[]
            
                    constantlength = "a"
                    if ("a" in strain_free_parameters) and ("b" in strain_free_parameters) and ("c" in strain_free_parameters):
                        constantlength = "a"    
                    elif ("b" not in strain_free_parameters) and additional_expression[0]=="none" and \
                        "b" not in additional_expression[0]:
                        constantlength = "b"
                    elif ("c" not in strain_free_parameters):
                        constantlength = "c"
                        
                    for irot in range(len(rotation_matrix1[index][0])):
                        lattice_parameter_direct_strain = CP.computeLatticeParameters_from_UB(rotation_matrix1[index][0][irot,:,:], 
                                                                                              material_, 
                                                                                              constantlength, 
                                                                                              dictmaterials=dictLT.dict_Materials)
                        a.append(lattice_parameter_direct_strain[0])
                        b.append(lattice_parameter_direct_strain[1])
                        c.append(lattice_parameter_direct_strain[2])
                        alp.append(lattice_parameter_direct_strain[3])
                        bet.append(lattice_parameter_direct_strain[4])
                        gam.append(lattice_parameter_direct_strain[5])
            
                    logdata = np.array(a) - latticeparams[0]
                    logdata = logdata[~np.isnan(logdata)]
                    rangemina, rangemaxa = np.min(logdata) - 0.01e-2, np.max(logdata) + 0.01e-2
                    logdata = np.array(b) - latticeparams[1]
                    logdata = logdata[~np.isnan(logdata)]
                    rangeminb, rangemaxb = np.min(logdata) - 0.01e-2, np.max(logdata) + 0.01e-2
                    logdata = np.array(c) - latticeparams[2]
                    logdata = logdata[~np.isnan(logdata)]
                    rangeminc, rangemaxc = np.min(logdata) - 0.01e-2, np.max(logdata) + 0.01e-2
                    logdata = np.array(alp) - latticeparams[3]
                    logdata = logdata[~np.isnan(logdata)]
                    rangeminal, rangemaxal = np.min(logdata) - 0.01, np.max(logdata) + 0.01
                    logdata = np.array(bet) - latticeparams[4]
                    logdata = logdata[~np.isnan(logdata)]
                    rangeminbe, rangemaxbe = np.min(logdata) - 0.01, np.max(logdata) + 0.01
                    logdata = np.array(gam) - latticeparams[5]
                    logdata = logdata[~np.isnan(logdata)]
                    rangeminga, rangemaxga = np.min(logdata) - 0.01, np.max(logdata) + 0.01
            
                    fig = plt.figure(figsize=(11.69,8.27), dpi=100)
                    bottom, top = 0.1, 0.9
                    left, right = 0.1, 0.8
                    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
            
                    vmin = rangemina
                    vmax = rangemaxa
                    axs = fig.subplots(2, 3)
                    axs[0, 0].set_title(r"$a$", loc='center', fontsize=8)
                    strain_matrix_plot = np.array(a) - latticeparams[0]
                    im=axs[0, 0].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                    axs[0, 0].set_xticks([])
                    axs[0, 0].set_yticks([])
                    divider = make_axes_locatable(axs[0,0])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.ax.tick_params(labelsize=8) 
            
                    vmin = rangeminb
                    vmax = rangemaxb
                    axs[0, 1].set_title(r"$b$", loc='center', fontsize=8)
                    strain_matrix_plot = np.array(b) - latticeparams[1]
                    im=axs[0, 1].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                    divider = make_axes_locatable(axs[0,1])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.ax.tick_params(labelsize=8) 
            
                    vmin = rangeminc
                    vmax = rangemaxc
                    axs[0, 2].set_title(r"$c$", loc='center', fontsize=8)
                    strain_matrix_plot = np.array(c) - latticeparams[2]
                    im=axs[0, 2].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                    divider = make_axes_locatable(axs[0,2])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.ax.tick_params(labelsize=8) 
            
                    vmin = rangeminal
                    vmax = rangemaxal
                    axs[1, 0].set_title(r"$\alpha$", loc='center', fontsize=8)
                    strain_matrix_plot = np.array(alp) - latticeparams[3]
                    im=axs[1, 0].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                    axs[1, 0].set_xticks([])
                    axs[1, 0].set_yticks([])
                    divider = make_axes_locatable(axs[1,0])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.ax.tick_params(labelsize=8) 
            
                    vmin = rangeminbe
                    vmax = rangemaxbe
                    axs[1, 1].set_title(r"$\beta$", loc='center', fontsize=8)
                    strain_matrix_plot = np.array(bet) - latticeparams[4]
                    im=axs[1, 1].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                    axs[1, 1].set_xticks([])
                    divider = make_axes_locatable(axs[1,1])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.ax.tick_params(labelsize=8) 
            
                    vmin = rangeminga
                    vmax = rangemaxga
                    axs[1, 2].set_title(r"$\gamma$", loc='center', fontsize=8)
                    strain_matrix_plot = np.array(gam) - latticeparams[5]
                    im = axs[1, 2].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                    axs[1, 2].set_xticks([]) 
                    divider = make_axes_locatable(axs[1,2])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.formatter.set_useOffset(False)
                    cbar.ax.tick_params(labelsize=8) 
            
                    for ax in axs.flat:
                        ax.label_outer()
                    plt.savefig(model_direc + "//" + 'figure_unitcell_relative_'+str(matid)+'_'+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
                    plt.close(fig)
                except:
                    pass
                
def sst_texture(orient_data=None, col_array=None, direc="", symmetry=None, symmetry_name=None, lattice=None,
                axis="Z", fn="", symms=None):
    
    print("symmetry of the current phase is : "+symmetry_name)
    
    if np.max(col_array) > 1:
        col_array[np.where(col_array>1)]=1
        
    fig = plt.figure(1)
    if symmetry_name == "cubic":
        pole_hkls = ['111','110','100']            
        ax1 = fig.add_subplot(221, aspect='equal')
        ax2 = fig.add_subplot(222, aspect='equal')
        ax3 = fig.add_subplot(223, aspect='equal')
        ax4 = fig.add_subplot(224, aspect='equal')
    elif symmetry_name == "hexagonal":
        pole_hkls = ['001','100','101','102','110']
        ax1 = fig.add_subplot(231, aspect='equal')
        ax2 = fig.add_subplot(232, aspect='equal')
        ax3 = fig.add_subplot(233, aspect='equal')
        ax4 = fig.add_subplot(234, aspect='equal')
        ax5 = fig.add_subplot(235, aspect='equal')
        ax6 = fig.add_subplot(236, aspect='equal')
    else:
        print("PF and IPF plots are only supported for Cubic and Hexagonal systems for now")
        return
    
    for pfs in range(len(pole_hkls)):
        pf1 = PoleFigure(hkl=pole_hkls[pfs], proj='stereo', lattice=lattice, axis=axis)     
        pf1.mksize = 1.
        if pfs == 0:
            pf1.plot_pf(col_array, orient_data, ax=ax1, ftsize=6)
        elif pfs == 1:
            pf1.plot_pf(col_array, orient_data, ax=ax2, ftsize=6)
        elif pfs == 2:
            pf1.plot_pf(col_array, orient_data, ax=ax3, ftsize=6)                    
        elif pfs == 3:
            pf1.plot_pf(col_array, orient_data, ax=ax4, ftsize=6)
        elif pfs == 4:
            pf1.plot_pf(col_array, orient_data, ax=ax5, ftsize=6)                    
    if symmetry_name == "cubic":
        pf1.plot_sst_color(col_array, orient_data, ax=ax4, ftsize=6, phase=0, symms=symms)
    elif symmetry_name == "hexagonal":
        pf1.plot_sst_color(col_array, orient_data, ax=ax6, ftsize=6, phase=1, symms=symms)
    plt.savefig(direc+"//PF_IPF_"+fn+".png", bbox_inches='tight',format='png', dpi=1000)
    plt.close() 
    
def save_sst(lim_x, lim_y, strain_matrix, strain_matrixs, col, colx, coly,
                      match_rate, mat_global, spots_len, iR_pix, fR_pix,
                      model_direc, material_, material1_, lattice_, lattice1_, 
                      symmetry_, symmetry1_, crystal, crystal1, rotation_matrix1, symmetry_name, symmetry1_name,
                      mac_axis = [0., 0., 1.],axis_text="Z",match_rate_threshold = 5):

    rotation_matrix_sst = [[] for i in range(len(rotation_matrix1))]
    for i in range(len(rotation_matrix1)):
        rotation_matrix_sst[i].append(np.zeros((lim_x*lim_y,3,3)))
        
    for i in range(len(rotation_matrix1)):
        temp_mat = rotation_matrix1[i][0]
        for j in range(len(temp_mat)):
            orientation_matrix123 = temp_mat[j,:,:]
            # ## rotate orientation by 40degrees to bring in Sample RF
            omega = np.deg2rad(-40.0)
            # rotation de -omega autour de l'axe x (or Y?) pour repasser dans Rsample
            cw = np.cos(omega)
            sw = np.sin(omega)
            mat_from_lab_to_sample_frame = np.array([[cw, 0.0, sw], [0.0, 1.0, 0.0], [-sw, 0, cw]])
            orientation_matrix123 = np.dot(mat_from_lab_to_sample_frame.T, orientation_matrix123)
            if np.linalg.det(orientation_matrix123) < 0:
                orientation_matrix123 = -orientation_matrix123
            rotation_matrix_sst[i][0][j,:,:] = orientation_matrix123
    
    rangeval = len(match_rate)
    if material_ == material1_:
        for index in range(rangeval):
            ### index for nans
            nan_index = np.where(match_rate[index][0] <= match_rate_threshold)[0]
            if index == 0:
                rotation_matrix_plot = np.copy(rotation_matrix_sst[index][0])
                col_plot = np.copy(col[index][0])
                col_plot[nan_index,:] = np.nan 
                rotation_matrix_plot[nan_index,:,:] = np.nan 
                
                sst_texture(orient_data=rotation_matrix_plot, 
                            col_array=col_plot, 
                            direc=model_direc, 
                            symmetry=symmetry_, 
                            symmetry_name = symmetry_name,
                            lattice=lattice_, axis=axis_text, fn="UB_"+str(index),
                            symms=crystal._hklsym)
            else:
                tempori = np.copy(rotation_matrix_sst[index][0])
                tempori[nan_index,:,:] = np.nan
                rotation_matrix_plot = np.vstack((rotation_matrix_plot,tempori))
                tempcol = np.copy(col[index][0])
                tempcol[nan_index,:] = np.nan
                col_plot = np.vstack((col_plot,tempcol))   
                
                sst_texture(orient_data=tempori, 
                            col_array=tempcol, 
                            direc=model_direc, 
                            symmetry=symmetry_, 
                            symmetry_name = symmetry_name,
                            lattice=lattice_, axis=axis_text, fn="UB_"+str(index),
                            symms=crystal._hklsym)
        ### Plot pole figures and IPF (cubic and hexagonal are supported for now)
        sst_texture(orient_data=rotation_matrix_plot, 
                    col_array=col_plot, 
                    direc=model_direc, 
                    symmetry=symmetry_, 
                    symmetry_name = symmetry_name,
                    lattice=lattice_, axis=axis_text, fn="all_UBs",
                    symms=crystal._hklsym)
    else:
        for matid in range(2):
            if matid == 0:
                symmetry_name_plot = symmetry_name
                symmetry_plot = symmetry_
                lattice_plot = lattice_
                symms = crystal._hklsym
            else:
                symmetry_name_plot = symmetry1_name
                symmetry_plot = symmetry1_
                lattice_plot = lattice1_
                symms = crystal1._hklsym
            
            for index in range(rangeval):
                ### index for nans
                nan_index1 = np.where(match_rate[index][0] <= match_rate_threshold)[0]
                mat_id_index = np.where(mat_global[index][0] != matid+1)[0]
                nan_index = np.hstack((mat_id_index,nan_index1))
                nan_index = np.unique(nan_index)
                if index == 0:
                    rotation_matrix_plot = np.copy(rotation_matrix_sst[index][0])
                    rotation_matrix_plot[nan_index,:,:] = np.nan 
                    col_plot = np.copy(col[index][0])
                    col_plot[nan_index,:] = np.nan
                    
                    sst_texture(orient_data=rotation_matrix_plot, 
                                col_array=col_plot, 
                                direc=model_direc, 
                                symmetry=symmetry_plot, 
                                symmetry_name = symmetry_name_plot,
                                lattice=lattice_plot, axis=axis_text, fn="mat_"+str(matid)+"_UB_"+str(index),
                                symms=symms)
                else:
                    tempori = np.copy(rotation_matrix_sst[index][0])
                    tempori[nan_index,:,:] = np.nan
                    rotation_matrix_plot = np.vstack((rotation_matrix_plot,tempori))
                    tempcol = np.copy(col[index][0])
                    tempcol[nan_index,:] = np.nan
                    col_plot = np.vstack((col_plot,tempcol))
                    
                    sst_texture(orient_data=tempori, 
                                col_array=tempcol, 
                                direc=model_direc, 
                                symmetry=symmetry_plot, 
                                symmetry_name = symmetry_name_plot,
                                lattice=lattice_plot, axis=axis_text, fn="mat_"+str(matid)+"_UB_"+str(index),
                                symms=symms)
                    
            sst_texture(orient_data=rotation_matrix_plot, 
                            col_array=col_plot, 
                            direc=model_direc, 
                            symmetry=symmetry_plot, 
                            symmetry_name = symmetry_name_plot,
                            lattice=lattice_plot, axis=axis_text, fn="mat_"+str(matid)+"_all_UBs",
                            symms=symms)


texttstr1 = "\n\
### config file for LaueNeuralNetwork \n\
[CPU]\n\
n_cpu = 8\n\
\n\
[GLOBAL_DIRECTORY]\n\
prefix = \n\
## directory where all training related data and results will be saved \n\
main_directory = C:\\Users\\purushot\\Desktop\\pattern_matching\\experimental\\GUIv0\\latest_version\n\
\n\
[MATERIAL]\n\
## same material key as lauetools (see dictlauetools.py for complete key)\n\
## as of now symmetry can be cubic, hexagonal, orthorhombic, tetragonal, trigonal, monoclinic, triclinic\n\
\n\
material = In2Bi\n\
symmetry = hexagonal\n\
space_group = between 1 and 230\n\
general_diffraction_rules = true\n\
\n\
## if second phase is present, else none\n\
material1 = In_epsilon\n\
symmetry1 = tetragonal\n\
space_group1 = between 1 and 230\n\
general_diffraction_rules1 = true\n\
\n\
[DETECTOR]\n\
## path to detector calibration file (.det)\n\
detectorfile = C:\\Users\\purushot\\Desktop\\In_JSM\\calib.det\n\
## Max and Min energy to be used for generating training dataset, as well as for calcualting matching rate\n\
emax = 21\n\
emin = 5\n\
\n\
[TRAINING]\n\
## classes_with_frequency_to_remove: HKL class with less appearance than specified will be ignored in output\n\
## desired_classes_output : can be all or an integer: to limit the number of output classes\n\
## max_HKL_index : can be auto or integer: Maximum index of HKL to build output classes\n\
## max_nb_grains : Maximum number of grains to simulate per lauepattern\n\
####### Material 0\n\
classes_with_frequency_to_remove = 500\n\
desired_classes_output = all\n\
max_HKL_index = 5\n\
max_nb_grains = 1\n\
####### Material 1\n\
## HKL class with less appearance than specified will be ignored in output\n\
classes_with_frequency_to_remove1 = 500\n\
desired_classes_output1 = all\n\
max_HKL_index1 = 5\n\
max_nb_grains1 = 1\n\
\n\
## Max number of simulations per number of grains\n\
## Include single crystal misorientation (1 deg) data in training\n\
## Maximum angular distance to probe (in deg)\n\
## step size in angular distribution to discretize (in deg)\n\
## batch size and epochs for training\n\
max_simulations = 1000\n\
include_small_misorientation = false\n\
misorientation_angle = 30\n\
angular_distance = 90\n\
step_size = 0.1\n\
batch_size = 50\n\
epochs = 5\n\
\n\
[PREDICTION]\n\
# model_weight_file: if none, it will select by default the latest H5 weight file, else provide a specific model\n\
# softmax_threshold_global: thresholding to limit the predicted spots search zone\n\
# mr_threshold_global: thresholding to ignore all matricies less than the MR threshold\n\
# cap_matchrate: any UB matrix providing MR less than this will be ignored\n\
# coeff: should be same as cap_matchrate or no? (this is for try previous UB matrix)\n\
# coeff_overlap: coefficient to limit the overlapping between spots; if more than this, new solution will be computed\n\
# mode_spotCycle: How to cycle through predicted spots (slow or graphmode )\n\
UB_matrix_to_detect = 1\n\
\n\
matrix_tolerance = 0.9\n\
matrix_tolerance1 = 0.9\n\
\n\
material0_limit = 1\n\
material1_limit = 1\n\
\n\
model_weight_file = none\n\
softmax_threshold_global = 0.85\n\
mr_threshold_global = 0.80\n\
cap_matchrate = 0.01\n\
coeff = 0.3\n\
coeff_overlap = 0.3\n\
mode_spotCycle = slow\n\
##true for few crystal and prefered texture case, otherwise time consuming; advised for single phase alone\n\
use_previous = true\n\
\n\
[EXPERIMENT]\n\
experiment_directory = C:\\Users\\purushot\\Desktop\\In_JSM\\ech875_ROI01\n\
experiment_file_prefix = ech875_ROI01_\n\
image_grid_x = 51\n\
image_grid_y = 51\n\
\n\
[PEAKSEARCH]\n\
intensity_threshold = 90\n\
boxsize = 15\n\
fit_peaks_gaussian = 1\n\
FitPixelDev = 15\n\
NumberMaxofFits = 3000\n\
\n\
[STRAINCALCULATION]\n\
strain_compute = true\n\
tolerance_strain_refinement = 0.7,0.6,0.5,0.4,0.3,0.2\n\
tolerance_strain_refinement1 = 0.7,0.6,0.5,0.4,0.3,0.2\n\
free_parameters = b,c,alpha,beta,gamma\n\
\n\
[POSTPROCESS]\n\
hkls_subsets = [1,1,0],[1,0,0],[1,1,1]\n\
\n\
\n\
[CALLER]\n\
residues_threshold=0.15\n\
nb_spots_global_threshold=10\n\
option_global = v1\n\
use_om_user = true\n\
nb_spots_consider = 100\n\
# User defined orientation matrix supplied in a file\n\
use_om_user = false\n\
path_user_OM = ""\n\
[DEVELOPMENT]\n\
# could be 1 or 2 / none in case of single phase\n\
material_phase_always_present = 1\n\
matrix_phase_always_present = 0.5673,0.5334,-0.6264,-0.6814,0.7330,0.00604,0.4625,0.4245,0.7805;Si\n\
generate_additional_data=false\n\
write_MTEX_file = true\n\
\n\
# Laue Groups\n\
# space group 1 -- triclinic: '-1'\n\
# space group 2 -- monoclinic: '2/m'\n\
# space group 3 -- orthorhombic: 'mmm'\n\
# space group 4 -- tetragonal: '4/m'\n\
# space group 5 -- tetragonal: '4/mmm'\n\
# space group 6 -- trigonal: '-3'\n\
# space group 7 -- trigonal: '-3m'\n\
# space group 8 -- hexagonal: '6/m'\n\
# space group 9 -- hexagonal: '6/mmm'\n\
# space group 10 -- cubic: 'm3'\n\
# space group 11 -- cubic: 'm3m'"

class Transform(object):
    def __init__(self, matrix):
        self.matrix = matrix
        self._imatrix = None

    @property
    def imatrix(self):
        if self._imatrix is None:
            try:
                self._imatrix = np.linalg.inv(self.matrix)
            except np.linalg.LinAlgError:
                raise Exception("XU.math.Transform: matrix cannot be inverted"
                                " - seems to be singular")
        return self._imatrix

    def inverse(self, args, rank=1):
        """
        performs inverse transformation a vector, matrix or tensor of rank 4

        Parameters
        ----------
        args :      list or array-like
            object to transform, list or np array of shape (..., n)
            (..., n, n), (..., n, n, n, n) where n is the size of the
            transformation matrix.
        rank :      int
            rank of the supplied object. allowed values are 1, 2, and 4
        """
        it = Transform(self.imatrix)
        return it(args, rank)

    def __call__(self, args, rank=1):
        """
        transforms a vector, matrix or tensor of rank 4
        (e.g. elasticity tensor)

        Parameters
        ----------
        args :      list or array-like
            object to transform, list or np array of shape (..., n)
            (..., n, n), (..., n, n, n, n) where n is the size of the
            transformation matrix.
        rank :      int
            rank of the supplied object. allowed values are 1, 2, and 4
        """

        m = self.matrix
        if rank == 1:  # argument is a vector
            # out_i = m_ij * args_j
            out = np.einsum('ij,...j', m, args)
        elif rank == 2:  # argument is a matrix
            # out_ij = m_ik * m_jl * args_kl
            out = np.einsum('ik, jl,...kl', m, m, args)
        elif rank == 4:
            # cp_ijkl = m_in * m_jo * m_kp * m_lq * args_nopq
            out = np.einsum('in, jo, kp, lq,...nopq', m, m, m, m, args)

        return out

    def __str__(self):
        ostr = "Transformation matrix:\n"
        ostr += str(self.matrix)
        return ostr

def VecCross(v1, v2, out=None):
    """
    Calculate the vector cross product.

    Parameters
    ----------
    v1, v2 :    list or array-like
        input vector(s), either one vector or an array of vectors with shape
        (n, 3)
    out :       list or array-like, optional
        output vector

    Returns
    -------
    ndarray
        cross product either of shape (3, ) or (n, 3)
    """
    if isinstance(v1, np.ndarray):
        if len(v1.shape) >= 2 or len(v2.shape) >= 2:
            return np.cross(v1, v2)
    if len(v1) != 3 or len(v2) != 3:
        raise ValueError("Vectors must be of size 3! (len(v1)=%d len(v2)=%d)"
                         % (len(v1), len(v2)))
    if out is None:
        out = np.empty(3)
    out[0] = v1[1] * v2[2] - v1[2] * v2[1]
    out[1] = v1[2] * v2[0] - v1[0] * v2[2]
    out[2] = v1[0] * v2[1] - v1[1] * v2[0]
    return out


def get_possible_sgrp_suf(sgrp_nr):
    """
    determine possible space group suffix. Multiple suffixes might be possible
    for one space group due to different origin choice, unique axis, or choice
    of the unit cell shape.

    Parameters
    ----------
    sgrp_nr :   int
        space group number

    Returns
    -------
    str or list
        either an empty string or a list of possible valid suffix strings
    """
    sgrp_suf = ''
    if sgrp_nr in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        sgrp_suf = [':b', ':c']
    elif sgrp_nr in [48, 50, 59, 68, 70, 85, 86, 88, 125, 126,
                     129, 130, 133, 134, 137, 138, 141, 142,
                     201, 203, 222, 224, 227, 228]:
        sgrp_suf = [':1', ':2']
    elif sgrp_nr in [146, 148, 155, 160, 161, 166, 167]:
        sgrp_suf = [':H', ':R']
    return sgrp_suf


def get_default_sgrp_suf(sgrp_nr):
    """
    determine default space group suffix
    """
    possibilities = get_possible_sgrp_suf(sgrp_nr)
    if possibilities:
        return possibilities[0]
    else:
        return ''

class SGLattice(object):
    """
    lattice object created from the space group number and corresponding unit
    cell parameters.
    """
    def __init__(self, sgrp, *args):
        """
        initialize class with space group number and atom list
        Parameters
        ----------
        sgrp :  int or str
            Space group number
        *args : float
            space group parameters. depending on the space group number this
            are 1 (cubic) to 6 (triclinic) parameters.
            cubic : a (lattice parameter).
            hexagonal : a, c.
            trigonal : a, c.
            tetragonal : a, c.
            orthorhombic : a, b, c.
            monoclinic : a, b, c, beta (in degree).
            triclinic : a, b, c, alpha, beta, gamma (in degree).
        """
        self.space_groupstr = str(sgrp)
        self.space_group = str(sgrp)
        self.space_group_nr = int(self.space_group.split(':')[0])
        try:
            self.space_group_suf = ':' + self.space_group.split(':')[1]
        except IndexError:
            self.space_group_suf = get_default_sgrp_suf(self.space_group_nr)

        if self.space_group_suf != '':
            self.space_group = str(self.space_group_nr) + self.space_group_suf
        self.name = sgrp_name[str(self.space_group_nr)] + self.space_group_suf
        self.crystal_system, nargs = sgrp_sym[self.space_group_nr]
        self.crystal_system += self.space_group_suf
        if len(args) != nargs:
            raise ValueError('XU: number of parameters (%d) does not match the'
                              ' crystal symmetry (%s:%d)' % (len(args), self.crystal_system, nargs))
        self.free_parameters = OrderedDict()
        for a, par in zip(args, sgrp_params[self.crystal_system][0]):
            self.free_parameters[par] = a

        self._parameters = OrderedDict()
        for i, p in enumerate(('a', 'b', 'c', 'alpha', 'beta', 'gamma')):
            key = sgrp_params[self.crystal_system][1][i]
            if isinstance(key, str):
                self._parameters[p] = self.free_parameters[key]
            else:
                self._parameters[p] = key
        # define lattice vectors
        self._ai = np.zeros((3, 3))
        self._bi = np.empty((3, 3))
        a, b, c, alpha, beta, gamma = self._parameters.values()
        ra = radians(alpha)
        self._paramhelp = [cos(ra), cos(radians(beta)),
                            cos(radians(gamma)), sin(ra), 0]
        self._setlat()
        # save general Wyckoff position
        self._gplabel = sorted(wp[self.space_group],
                               key=lambda s: int(s[:-1]))[-1]
        self._gp = wp[self.space_group][self._gplabel]

        # symmetry operations and reflection conditions placeholder
        self._hklmat = []
        self._symops = []
        self._hklcond = []
        self._hklcond_wp = []
        self._iscentrosymmetric = None

    @property
    def symops(self):
        """
        return the set of symmetry operations from the general Wyckoff
        position of the space group.
        """
        if self._symops == []:
            for p in self._gp[1]:
                self._symops.append(SymOp.from_xyz(p))
        return self._symops

    @property
    def _hklsym(self):
        if self._hklmat == []:
            for s in self.symops:
                self._hklmat.append(np.round(self._qtransform.imatrix @
                                                self._transform.matrix @ s.D @
                                                self._transform.imatrix @
                                                self._qtransform.matrix,
                                                DIGITS))
        return self._hklmat

    def _setlat(self):
        a, b, c, alpha, beta, gamma = self._parameters.values()
        ca, cb, cg, sa, vh = self._paramhelp
        vh = sqrt(1 - ca**2-cb**2-cg**2 + 2*ca*cb*cg)
        self._paramhelp[4] = vh
        self._ai[0, 0] = a * vh / sa
        self._ai[0, 1] = a * (cg-cb*ca) / sa
        self._ai[0, 2] = a * cb
        self._ai[1, 1] = b * sa
        self._ai[1, 2] = b * ca
        self._ai[2, 2] = c
        self._transform = Transform(self._ai.T)
        self._setb()

    def _setb(self):
        V = self.UnitCellVolume()
        p = 2. * np.pi / V
        VecCross(p*self._ai[1, :], self._ai[2, :], out=self._bi[0, :])
        VecCross(p*self._ai[2, :], self._ai[0, :], out=self._bi[1, :])
        VecCross(p*self._ai[0, :], self._ai[1, :], out=self._bi[2, :])
        self._qtransform = Transform(self._bi.T)

    def _set_params_from_sym(self):
        for i, p in enumerate(('a', 'b', 'c', 'alpha', 'beta', 'gamma')):
            key = sgrp_params[self.crystal_system][1][i]
            if isinstance(key, str):
                if p not in self.free_parameters:
                    self._parameters[p] = self.free_parameters[key]

    @property
    def a(self):
        return self._parameters['a']

    @a.setter
    def a(self, value):
        if 'a' not in self.free_parameters:
            raise RuntimeError("a can not be set, its not a free parameter!")
        self._parameters['a'] = value
        self.free_parameters['a'] = value
        self._set_params_from_sym()
        self._setlat()

    @property
    def b(self):
        return self._parameters['b']

    @b.setter
    def b(self, value):
        if 'b' not in self.free_parameters:
            raise RuntimeError("b can not be set, its not a free parameter!")
        self._parameters['b'] = value
        self.free_parameters['b'] = value
        self._set_params_from_sym()
        self._setlat()

    @property
    def c(self):
        return self._parameters['c']

    @c.setter
    def c(self, value):
        if 'c' not in self.free_parameters:
            raise RuntimeError("c can not be set, its not a free parameter!")
        self._parameters['c'] = value
        self.free_parameters['c'] = value
        self._set_params_from_sym()
        self._setlat()

    @property
    def alpha(self):
        return self._parameters['alpha']

    @alpha.setter
    def alpha(self, value):
        if 'alpha' not in self.free_parameters:
            raise RuntimeError("alpha can not be set for this space group!")
        self._parameters['alpha'] = value
        self.free_parameters['alpha'] = value
        self._set_params_from_sym()
        ra = radians(value)
        self._paramhelp[0] = cos(ra)
        self._paramhelp[3] = sin(ra)
        self._setlat()

    @property
    def beta(self):
        return self._parameters['beta']

    @beta.setter
    def beta(self, value):
        if 'beta' not in self.free_parameters:
            raise RuntimeError("beta can not be set for this space group!")
        self._parameters['beta'] = value
        self.free_parameters['beta'] = value
        self._set_params_from_sym()
        self._paramhelp[1] = cos(radians(value))
        self._setlat()

    @property
    def gamma(self):
        return self._parameters['gamma']

    @gamma.setter
    def gamma(self, value):
        if 'gamma' not in self.free_parameters:
            raise RuntimeError("gamma can not be set for this space group!")
        self._parameters['gamma'] = value
        self.free_parameters['gamma'] = value
        self._set_params_from_sym()
        self._paramhelp[2] = cos(radians(value))
        self._setlat()

    def UnitCellVolume(self):
        """
        function to calculate the unit cell volume of a lattice (angstrom^3)
        """
        a, b, c, alpha, beta, gamma = self._parameters.values()
        return a * b * c * self._paramhelp[4]

    @property
    def iscentrosymmetric(self):
        """
        returns a boolean to determine if the lattice has centrosymmetry.
        """
        if self._iscentrosymmetric is None:
            self._iscentrosymmetric = False
            for s in self.symops:
                if np.all(-np.identity(3) == s.D):
                    self._iscentrosymmetric = True
                    break
        return self._iscentrosymmetric

    def isequivalent(self, hkl1, hkl2):
        """
        determining if hkl1 and hkl2 are two crystallographical equivalent
        pairs of Miller indices. Note that this function considers the effect
        of non-centrosymmetry!

        Parameters
        ----------
        hkl1, hkl2 :    list
            Miller indices to be checked for equivalence

        Returns
        -------
        bool
        """
        return tuple(hkl2) in self.equivalent_hkls(hkl1)

    def equivalent_hkls(self, hkl):
        """
        returns a list of equivalent hkl peaks depending on the crystal system
        """
        suf = self.space_group_suf
        nr = self.space_group_nr
        if suf == get_default_sgrp_suf(nr):
            ehkl = set(eqhkl_default[nr](hkl[0], hkl[1], hkl[2]))
        elif suf in get_possible_sgrp_suf(nr):
            ehkl = set(eqhkl_custom[nr](hkl[0], hkl[1], hkl[2]))
        else:  # fallback calculation with symmetry operations
            ehkl = np.unique(np.einsum('...ij,j', self._hklsym, hkl),
                                axis=0)
            ehkl = set(tuple(e) for e in ehkl)
        return ehkl
    
    def hkl_allowed(self, hkl, returnequivalents=False):
        """
        check if Bragg reflection with Miller indices hkl can exist according
        to the reflection conditions. If no reflection conditions are available
        this function returns True for all hkl values!
        Parameters
        ----------
        hkl : tuple or list
         Miller indices of the reflection to check
        returnequivalents : bool, optional
         If True all the equivalent Miller indices of hkl are returned in a
         set as second return argument.
        Returns
        -------
        allowed : bool
         True if reflection can have non-zero structure factor, false otherwise
        equivalents : set, optional
         set of equivalent Miller indices if returnequivalents is True
        """
        # generate all equivalent hkl values which also need to be checked:
        hkls = self.equivalent_hkls(hkl)

        def build_return(allowed, requi=returnequivalents):
            if requi:
                return allowed, hkls
            else:
                return allowed

        # load reflection conditions if needed
        if self._gp[2] == 'n/a':
            return build_return(True)

        if self._hklcond == [] and self._gp[2] is not None:
            self._hklcond = hklcond_group.findall(self._gp[2])

        ret = testhklcond(hkls, self._hklcond)
        return build_return(ret)
    
    def hkl_allowed_array(self, hkl):
        rethkl = testhklcond_generalrules_array(self.space_groupstr, hkl)
        return rethkl
    
def check2n(h):
    if (h % 2 == 0):
        return 1
    else:
        return 0

def check2np1(h):
    if ((h-1) % 2 == 0):
        return 1
    else:
        return 0

def check3n(h):
    if (h % 3 == 0):
        return 1
    else:
        return 0

def check3np1(h):
    if ((h-1) % 3 == 0):
        return 1
    else:
        return 0

def check3np2(h):
    if ((h-2) % 3 == 0):
        return 1
    else:
        return 0

def check4n(h):
    if (h % 4 == 0):
        return 1
    else:
        return 0

def check4np2(h):
    if ((h-2) % 4 == 0):
        return 1
    else:
        return 0


def check6n(h):
    if (h % 6 == 0):
        return 1
    else:
        return 0

def check8n(h):
    if (h % 8 == 0):
        return 1
    else:
        return 0

def check8np1(h):
    if ((h-1) % 8 == 0):
        return 1
    else:
        return 0

def check8nm1(h):
    if ((h+1) % 8 == 0):
        return 1
    else:
        return 0

def check8np3(h):
    if ((h-3) % 8 == 0):
        return 1
    else:
        return 0

def check8nm3(h):
    if ((h+3) % 8 == 0):
        return 1
    else:
        return 0

def check8np4(h):
    if ((h-4) % 8 == 0):
        return 1
    else:
        return 0
 
def check8np5(h):
    if ((h-5) % 8 == 0):
        return 1
    else:
        return 0

def check8np7(h):
    if ((h-7) % 8 == 0):
        return 1
    else:
        return 0

def testhklcond(hkls, condition, verbose=False):
    """
     * test if a Bragg peak is allowed according to reflection conditions
     *
     * Parameters
     * ----------
     *  hkl :           Miller indices of the peak to test (integer array)
     *  condgeneral :   General reflection conditions (list of tuples)
     *  condwp :        Reflection conditions for Wyckoff positions
     *                  (list of list of tuples)
     *
     * Returns
     * -------
     * bool : True if peak is allowed, False otherwise
     """
    # /* test general reflection conditions
    #  * if they are violated the peak is forbidden
    #  */
    pattern_applied = 0
    condition_met = 2

    for hkl in hkls:
        for i in condition:
            hklpattern = i[0]
            cond = i[1]
            if hklpattern_applies(hkl, hklpattern):
                pattern_applied = 1
                if verbose:
                    print(hkl, hklpattern, cond)
                r = reflection_condition_met(hkl, cond)
                if r == 1:
                    condition_met = 1
                else:
                    condition_met = 0
                if verbose:
                    print(condition_met, pattern_applied)
                    
            if condition_met == 0:
                break
        if condition_met == 0:
            break
            
    if (condition_met == 1 or pattern_applied == 0):
        return True
    else:
        if pattern_applied == 1:
            return False
        else:
            return True
        
def hklpattern_applies(hkl, condhkl):
    """/*
     * helper function to determine if Miller indices fit a certain pattern
     *
     * Parameters
     * ----------
     *  hkl : array of three integers Miller indices
     *  condhkl : condition string similar to 'hkl', 'hh0', or '0k0'
     *
     * Returns
     * -------
     *  1 if hkl fulfills the pattern, 0 otherwise
    */"""
    n=0
    if (condhkl[n] == '0' and hkl[0] != 0):
        return 0
    n = n + 1
    if (condhkl[n] == '-'):
        n = n + 1
        if (condhkl[n] == 'h' and hkl[1] != -hkl[0]):
            return 0
    elif (condhkl[n] == '0' and hkl[1] != 0):
        return 0
    elif (condhkl[n] == 'h' and hkl[1] != hkl[0]):
        return 0
    if (condhkl[len(condhkl)-1] == '0' and hkl[2] != 0):
        return 0
    return 1

def strcmp(expa, expb):
    if expa == expb:
        return 1
    else:
        return 0
    
def reflection_condition_met(hkl, cond):
    """/*
     * helper function to determine allowed Miller indices
     *
     * Parameters
     * ----------
     *  hkl: list or tuple
     *   Miller indices of the reflection
     *  cond: str
     *   condition string similar to 'h+k=2n, h+l,k+l=2n'
     *
     * Returns
     * -------
     *  1 if condition is met, 0 otherwise
    */"""
    fulfilled = 1
    condi = cond.split("=")
    if len(condi) > 2:
        condi = cond.split(", ")
        if len(condi) >2:
            fulfilled = 0
            print("right hand expression error")

        for kun in condi:
            condi1 = kun.split("=")
            rexpr = condi1[1]
            lexpr_global = condi1[0]
            
            if strcmp(rexpr, "2n"):
                checkfunc = check2n
            elif strcmp(rexpr, "2n+1"):
                checkfunc = check2np1
            elif strcmp(rexpr, "3n"):
                checkfunc = check3n
            elif strcmp(rexpr, "3n+1"):
                checkfunc = check3np1
            elif strcmp(rexpr, "3n+2"):
                checkfunc = check3np2
            elif strcmp(rexpr, "4n"):
                checkfunc = check4n
            elif strcmp(rexpr, "4n+2"):
                checkfunc = check4np2
            elif strcmp(rexpr, "6n"):
                checkfunc = check6n
            elif strcmp(rexpr, "8n"):
                checkfunc = check8n
            elif strcmp(rexpr, "8n+1"):
                checkfunc = check8np1
            elif strcmp(rexpr, "8n-1"):
                checkfunc = check8nm1
            elif strcmp(rexpr, "8n+3"):
                checkfunc = check8np3
            elif strcmp(rexpr, "8n-3"):
                checkfunc = check8nm3
            elif strcmp(rexpr, "8n+4"):
                checkfunc = check8np4
            elif strcmp(rexpr, "8n+5"):
                checkfunc = check8np5
            elif strcmp(rexpr, "8n+7"):
                checkfunc = check8np7
            else:
                print("Right hand side of reflection condition (%s) not implemented" %(rexpr))
                return -1
            
            for lexpr in lexpr_global.split(','):
                if strcmp(lexpr, "h"):
                    if (checkfunc(hkl[0]) == 0):
                        fulfilled = 0
                elif strcmp(lexpr, "k"):
                    if (checkfunc(hkl[1]) == 0):
                        fulfilled = 0
                elif strcmp(lexpr, "l"):
                    if (checkfunc(hkl[2]) == 0):
                        fulfilled = 0
                elif strcmp(lexpr, "h+k"):
                    if (checkfunc(hkl[0] + hkl[1]) == 0):
                        fulfilled = 0
                elif strcmp(lexpr, "h-k"):
                    if (checkfunc(hkl[0] - hkl[1]) == 0):
                        fulfilled = 0
                elif strcmp(lexpr, "-h+k"):
                    if (checkfunc(-hkl[0] + hkl[1]) == 0):
                        fulfilled = 0
                elif strcmp(lexpr, "h+l"):
                    if (checkfunc(hkl[0] + hkl[2]) == 0):
                        fulfilled = 0
                elif strcmp(lexpr, "k+l"):
                    if (checkfunc(hkl[1] + hkl[2]) == 0):
                        fulfilled = 0
                elif strcmp(lexpr, "h+k+l"):
                    if (checkfunc(hkl[0] + hkl[1] + hkl[2]) == 0):
                        fulfilled = 0
                elif strcmp(lexpr, "-h+k+l"):
                    if (checkfunc(-hkl[0] + hkl[1] + hkl[2]) == 0):
                        fulfilled = 0
                elif strcmp(lexpr, "2h+l"):
                    if (checkfunc(2*hkl[0] + hkl[2]) == 0):
                        fulfilled = 0;
                elif strcmp(lexpr, "2k+l"):
                    if (checkfunc(2*hkl[1] + hkl[2]) == 0):
                        fulfilled = 0
            
    else:
        rexpr = condi[1]
        lexpr_global = condi[0]
    
        if strcmp(rexpr, "2n"):
            checkfunc = check2n
        elif strcmp(rexpr, "2n+1"):
            checkfunc = check2np1
        elif strcmp(rexpr, "3n"):
            checkfunc = check3n
        elif strcmp(rexpr, "3n+1"):
            checkfunc = check3np1
        elif strcmp(rexpr, "3n+2"):
            checkfunc = check3np2
        elif strcmp(rexpr, "4n"):
            checkfunc = check4n
        elif strcmp(rexpr, "4n+2"):
            checkfunc = check4np2
        elif strcmp(rexpr, "6n"):
            checkfunc = check6n
        elif strcmp(rexpr, "8n"):
            checkfunc = check8n
        elif strcmp(rexpr, "8n+1"):
            checkfunc = check8np1
        elif strcmp(rexpr, "8n-1"):
            checkfunc = check8nm1
        elif strcmp(rexpr, "8n+3"):
            checkfunc = check8np3
        elif strcmp(rexpr, "8n-3"):
            checkfunc = check8nm3
        elif strcmp(rexpr, "8n+4"):
            checkfunc = check8np4
        elif strcmp(rexpr, "8n+5"):
            checkfunc = check8np5
        elif strcmp(rexpr, "8n+7"):
            checkfunc = check8np7
        else:
            print("Right hand side of reflection condition (%s) not implemented" %(rexpr))
            return -1
        
        for lexpr in lexpr_global.split(','):
            if strcmp(lexpr, "h"):
                if (checkfunc(hkl[0]) == 0):
                    fulfilled = 0
            elif strcmp(lexpr, "k"):
                if (checkfunc(hkl[1]) == 0):
                    fulfilled = 0
            elif strcmp(lexpr, "l"):
                if (checkfunc(hkl[2]) == 0):
                    fulfilled = 0
            elif strcmp(lexpr, "h+k"):
                if (checkfunc(hkl[0] + hkl[1]) == 0):
                    fulfilled = 0
            elif strcmp(lexpr, "h-k"):
                if (checkfunc(hkl[0] - hkl[1]) == 0):
                    fulfilled = 0
            elif strcmp(lexpr, "-h+k"):
                if (checkfunc(-hkl[0] + hkl[1]) == 0):
                    fulfilled = 0
            elif strcmp(lexpr, "h+l"):
                if (checkfunc(hkl[0] + hkl[2]) == 0):
                    fulfilled = 0
            elif strcmp(lexpr, "k+l"):
                if (checkfunc(hkl[1] + hkl[2]) == 0):
                    fulfilled = 0
            elif strcmp(lexpr, "h+k+l"):
                if (checkfunc(hkl[0] + hkl[1] + hkl[2]) == 0):
                    fulfilled = 0
            elif strcmp(lexpr, "-h+k+l"):
                if (checkfunc(-hkl[0] + hkl[1] + hkl[2]) == 0):
                    fulfilled = 0
            elif strcmp(lexpr, "2h+l"):
                if (checkfunc(2*hkl[0] + hkl[2]) == 0):
                    fulfilled = 0;
            elif strcmp(lexpr, "2k+l"):
                if (checkfunc(2*hkl[1] + hkl[2]) == 0):
                    fulfilled = 0

    if (fulfilled == 1):
        return 1
    else:
        return 0

class SymOp(object):
    """
    Class descriping a symmetry operation in a crystal. The symmetry operation
    is characterized by a 3x3 transformation matrix as well as a 3-vector
    describing a translation. For magnetic symmetry operations also the time
    reversal symmetry can be specified (not used in xrayutilities)
    """

    def __init__(self, D, t, m=1):
        """
        Initialize the symmetry operation

        Parameters
        ----------
        D : array-like
            transformation matrix (3x3)
        t : array-like
            translation vector (3)
        m : int, optional
            indicates time reversal in magnetic groups. +1 (default, no time
            reveral) or -1
        """
        self._W = np.zeros((4, 4))
        self._W[:3, :3] = np.asarray(D)
        self._W[:3, 3] = np.asarray(t)
        self._W[3, 3] = 1
        self._m = m

    @classmethod
    def from_xyz(cls, xyz):
        """
        create a SymOp from the xyz notation typically used in CIF files.

        Parameters
        ----------
         xyz :   str
            string describing the symmetry operation (e.g. '-y, -x, z')
        """
        D = np.zeros((3, 3))
        t = np.array(eval(xyz, {'x': 0, 'y': 0, 'z': 0})[:3])
        m = 1
        for i, expr in enumerate(xyz.strip('()').split(',')):
            if i == 3:  # time reversal property
                m = int(expr)
                continue
            if 'x' in expr:
                D[i, 0] = -1 if '-x' in expr else 1
            if 'y' in expr:
                D[i, 1] = -1 if '-y' in expr else 1
            if 'z' in expr:
                D[i, 2] = -1 if '-z' in expr else 1
        return SymOp(D, t, m)

    def xyz(self, showtimerev=False):
        """
        return the symmetry operation in xyz notation
        """
        ret = ''
        t = self.t
        for i in range(3):
            expr = ''
            if abs(self._W[i, 0]) == 1:
                expr += '+x' if self._W[i, 0] == 1 else '-x'
            if abs(self._W[i, 1]) == 1:
                expr += '+y' if self._W[i, 1] == 1 else '-y'
            if abs(self._W[i, 2]) == 1:
                expr += '+z' if self._W[i, 2] == 1 else '-z'
            if t[i] != 0:
                expr += '+' if t[i] > 0 else ''
                expr += str(fractions.Fraction(t[i]).limit_denominator(100))
            expr = expr.strip('+')
            ret += expr + ', '
        if showtimerev:
            ret += '{:+d}'.format(self._m)
        return ret.strip(', ')

    @property
    def D(self):
        """transformation matrix of the symmetry operation"""
        return self._W[:3, :3]

    @property
    def t(self):
        """translation vector of the symmetry operation"""
        return self._W[:3, 3]

    def __eq__(self, other):
        if not isinstance(other, SymOp):
            return NotImplemented
        return self._m == other._m and np.all(self._W == other._W)

    @staticmethod
    def foldback(v):
        return v - np.round(v, DIGITS) // 1

    def apply_rotation(self, vec):
        return self.D @ vec

    def apply(self, vec, foldback=True):
        lv = np.asarray(list(vec) + [1, ])
        result = (self._W @ lv)[:3]
        if foldback:
            return self.foldback(result)
        return result

    def apply_axial(self, vec):
        return self._m * np.linalg.det(self.D) * self.D @ vec

    def combine(self, other):
        if not isinstance(other, SymOp):
            return NotImplemented
        W = self._W @ other._W
        return SymOp(W[:3, :3], self.foldback(W[:3, 3]), self._m*other._m)

    def __str__(self):
        return '({})'.format(self.xyz(showtimerev=True))

    def __repr__(self):
        return self.__str__()

def _round_indices(indices, max_index=12):
    """Round a set of index triplet (Miller) or quartet (Miller-Bravais)
    to the *closest* smallest integers.

    Adopted from MTEX's Miller.round function.

    Parameters
    ----------
    indices : list, tuple, or np.ndarray
        Set of index triplet(s) or quartet(s) to round.
    max_index : int, optional
        Maximum integer index to round to, by default 12.

    Return
    ------
    new_indices : np.ndarray
        Integer array of rounded set of index triplet(s) or quartet(s).
    """
    # Allow list and tuple input (and don't overwrite `indices`)
    idx = np.asarray(indices)

    # Flatten and remove redundant third index if Miller-Bravais
    n_idx = idx.shape[-1]  # 3 or 4
    idx_flat = np.reshape(idx, (-1, n_idx))
    if n_idx == 4:
        idx_flat = idx_flat[..., [0, 1, 3]]

    # Get number of sets, max. index per set, and all possible integer
    # multipliers between 1 and `max_index`
    n_sets = idx_flat.size // 3
    max_per_set = np.max(np.abs(idx_flat), axis=-1)
    multipliers = np.arange(1, max_index + 1)

    # Divide by highest index, repeat array `max_index` number of times,
    # and multiply with all multipliers
    idx_scaled = (
        np.broadcast_to(idx_flat / max_per_set[..., np.newaxis], (max_index, n_sets, 3))
        * multipliers[..., np.newaxis, np.newaxis]
    )

    # Find the most suitable multiplier per set, which gives the
    # smallest error between the initial set and the scaled and rounded
    # set
    error = 1e-7 * np.round(
        1e7
        * np.sum((idx_scaled - np.round(idx_scaled)) ** 2, axis=-1)
        / np.sum(idx_scaled ** 2, axis=-1)
    )
    idx_min_error = np.argmin(error, axis=0)
    multiplier = (idx_min_error + 1) / max_per_set

    # Reshape `multiplier` to match indices shape
    multiplier = multiplier.reshape(idx.shape[:-1])[..., np.newaxis]

    # Finally, multiply each set with their most suitable multiplier,
    # and round
    new_indices = np.round(multiplier * idx).astype(int)

    return new_indices

# =============================================================================
# PYMICRO FUNCTION IMPORTS
# =============================================================================

def move_rotation_to_FZ(g, symmetry_operators = None):
    """Compute the rotation matrix in the Fundamental Zone of a given
    `Symmetry` instance.

    :param g: a 3x3 matrix representing the rotation.
    :param verbose: flag for verbose mode.
    :return: a new 3x3 matrix for the rotation in the fundamental zone.
    """
    omegas = []  # list to store all the rotation angles
    syms = symmetry_operators
    for sym in syms:
        # apply the symmetry operator
        om = np.dot(sym, g)
        cw = 0.5 * (om.trace() - 1)
        omega = np.arccos(cw)
        omegas.append(omega)
    index = np.argmin(omegas)
    return np.dot(syms[index], g)

def misorientation_axis_from_delta(delta):
    """Compute the misorientation axis from the misorientation matrix.

    :param delta: The 3x3 misorientation matrix.
    :returns: the misorientation axis (normalised vector).
    """
    n = np.array([delta[1, 2] - delta[2, 1], delta[2, 0] -
                  delta[0, 2], delta[0, 1] - delta[1, 0]])
    n /= np.sqrt((delta[1, 2] - delta[2, 1]) ** 2 +
                 (delta[2, 0] - delta[0, 2]) ** 2 +
                 (delta[0, 1] - delta[1, 0]) ** 2)
    return n

def misorientation_angle_from_delta(delta):
    """Compute the misorientation angle from the misorientation matrix.

    Compute the angle associated with this misorientation matrix :math:`\\Delta g`.
    It is defined as :math:`\\omega = \\arccos(\\text{trace}(\\Delta g)/2-1)`.
    To avoid float rounding error, the argument is rounded to 1.0 if it is
    within 1 and 1 plus 32 bits floating point precison.

    .. note::

      This does not account for the crystal symmetries. If you want to
      find the disorientation between two orientations, use the
      :py:meth:`~pymicro.crystal.microstructure.Orientation.disorientation`
      method.

    :param delta: The 3x3 misorientation matrix.
    :returns float: the misorientation angle in radians.
    """
    cw = 0.5 * (delta.trace() - 1)
    if cw > 1. and cw - 1. < 10 * np.finfo('float32').eps:
        cw = 1.
    omega = np.arccos(cw)
    return omega

def disorientation(orientation_matrix, orientation_matrix1, crystal_structure=None):
    """Compute the disorientation another crystal orientation.

    Considering all the possible crystal symmetries, the disorientation
    is defined as the combination of the minimum misorientation angle
    and the misorientation axis lying in the fundamental zone, which
    can be used to bring the two lattices into coincidence.

    .. note::

     Both orientations are supposed to have the same symmetry. This is not
     necessarily the case in multi-phase materials.

    :param orientation: an instance of
        :py:class:`~pymicro.crystal.microstructure.Orientation` class
        describing the other crystal orientation from which to compute the
        angle.
    :param crystal_structure: an instance of the `Symmetry` class
        describing the crystal symmetry, triclinic (no symmetry) by
        default.
    :returns tuple: the misorientation angle in radians, the axis as a
        numpy vector (crystal coordinates), the axis as a numpy vector
        (sample coordinates).
    """
    the_angle = np.pi
    symmetries = crystal_structure.symmetry_operators()
    (gA, gB) = (orientation_matrix, orientation_matrix1)  # nicknames
    for (g1, g2) in [(gA, gB), (gB, gA)]:
        for j in range(symmetries.shape[0]):
            sym_j = symmetries[j]
            oj = np.dot(sym_j, g1)  # the crystal symmetry operator is left applied
            for i in range(symmetries.shape[0]):
                sym_i = symmetries[i]
                oi = np.dot(sym_i, g2)
                delta = np.dot(oi, oj.T)
                mis_angle = misorientation_angle_from_delta(delta)
                if mis_angle < the_angle:
                    # now compute the misorientation axis, should check if it lies in the fundamental zone
                    mis_axis = misorientation_axis_from_delta(delta)
                    the_angle = mis_angle
                    the_axis = mis_axis
                    the_axis_xyz = np.dot(oi.T, the_axis)
    return the_angle, the_axis, the_axis_xyz

# =============================================================================
# Notebook functions
# =============================================================================

def generate_dataset(material_="Cu", material1_="Cu", ang_maxx=18.,step=0.1, mode=0, 
                         nb_grains=1, nb_grains1=1, grains_nb_simulate=100, data_realism = False, 
                         detectorparameters=None, pixelsize=None, type_="training",
                         var0 = 0, dim1=2048, dim2=2048, removeharmonics=1, save_directory="",
                         write_to_console=None, emin=5, emax=22, modelp = "random",
                         misorientation_angle = None, general_diff_rules = False, 
                         crystal = None, crystal1 = None, include_scm=False, 
                         matrix_phase_always_present=None): 
    """
    works for all symmetries now.
    """
    from multiprocessing import Process, Queue, cpu_count
    ncpu = cpu_count()
    
    ## make sure directory exists
    save_directory_ = save_directory+"//"+type_
    if not os.path.exists(save_directory_):
        os.makedirs(save_directory_)

    try:
        with open(save_directory+"//classhkl_data_"+material_+".pickle", "rb") as input_file:
            classhkl, _, _, n, _, \
                hkl_all_class, _, lattice_material, symmetry = cPickle.load(input_file)
        max_millerindex = int(n)
        max_millerindex1 = int(n)
        if material_ != material1_:
            with open(save_directory+"//classhkl_data_"+material1_+".pickle", "rb") as input_file:
                classhkl1, _, _, n1, _, \
                    hkl_all_class1, _, lattice_material1, symmetry1 = cPickle.load(input_file)
            max_millerindex1 = int(n1)
    except:
        write_to_console("Class HKL library data not found, please run it first")
        return None

    if var0==1:
        codebars, angbins = get_material_data(material_ = material_, ang_maxx = ang_maxx, step = step,
                                                   hkl_ref=n, classhkl=classhkl)
        loc = np.array([ij for ij in range(len(classhkl))])

        write_to_console("Verifying if two different HKL class have same angular distribution (can be very time consuming depending on the symmetry)")
        index = []
        list_appended = []
        count_cbs = 0
        for i, j in enumerate(codebars):
            for k, l in enumerate(codebars):
                # if i in list_appended and k in list_appended:
                #     continue
                if i != k and np.all(j == l):
                    index.append((i,k))
                    string0 = "HKL's "+ str(classhkl[i])+" and "+str(classhkl[k])+" have exactly the same angular distribution."
                    write_to_console(string0)
                list_appended.append(i)
                list_appended.append(k)
            count_cbs += 1

        if len(index) == 0:
            write_to_console("Great! No two HKL class have same angular distribution")
            #np.savez_compressed(save_directory_+'//grain_init.npz', codebars, loc)
        else:
            write_to_console("Some HKL's have similar angular distribution; this will likely reduce the accuracy of the neural network; verify if symmetry matrix and other parameters are properly configured; this is just for the dictionary; keep eye on the dataset being generated for training")
            write_to_console("This is likely the result of the symmetry operation available in a user_defined space group; this shouldn't affect the general accuracy of the model")
            np.savez_compressed(save_directory+'//conflict_angular_distribution_debug.npz', codebars, index)           
        np.savez_compressed(save_directory+'//grain_classhkl_angbin.npz', classhkl, angbins)
             
        if material_ != material1_:
            codebars, angbins = get_material_data(material_ = material1_, ang_maxx = ang_maxx, step = step,
                                                   hkl_ref=n1, classhkl=classhkl1)
            ind_offset = loc[-1] + 1
            loc = np.array([ind_offset + ij for ij in range(len(classhkl1))])
            write_to_console("Verifying if two different HKL class have same angular distribution (can be very time consuming depending on the symmetry)")
            index = []
            list_appended = []
            count_cbs = 0
            for i, j in enumerate(codebars):
                for k, l in enumerate(codebars):
                    # if i in list_appended and k in list_appended:
                    #     continue
                    if i != k and np.all(j == l):
                        index.append((i,k))
                        string0 = "HKL's "+ str(classhkl1[i])+" and "+str(classhkl1[k])+" have exactly the same angular distribution."
                        write_to_console(string0)
                    list_appended.append(i)
                    list_appended.append(k)
                count_cbs += 1

            if len(index) == 0:
                write_to_console("Great! No two HKL class have same angular distribution")
                #np.savez_compressed(save_directory_+'//grain_init1.npz', codebars, loc)
            else:
                write_to_console("Some HKL's have similar angular distribution; this will likely reduce the accuracy of the neural network; verify if symmetry matrix and other parameters are properly configured; this is just for the dictionary; keep eye on the dataset being generated for training")
                write_to_console("This is likely the result of the symmetry operation available in a user_defined space group; this shouldn't affect the general accuracy of the model")
                np.savez_compressed(save_directory+'//conflict_angular_distribution1_debug.npz', codebars, index)                
            np.savez_compressed(save_directory+'//grain_classhkl_angbin1.npz', classhkl1, angbins)
    
    ## make comprehensive list of dictionary    
    normal_hkl_ = np.zeros((1,3))
    for j in hkl_all_class.keys():
        normal_hkl_ = np.vstack((normal_hkl_, hkl_all_class[j]["family"]))
    normal_hkl = np.delete(normal_hkl_, 0, axis =0)
    
    if material_ != material1_:
        normal_hkl1_ = np.zeros((1,3))
        for j in hkl_all_class1.keys():
            normal_hkl1_ = np.vstack((normal_hkl1_, hkl_all_class1[j]["family"]))
        normal_hkl1 = np.delete(normal_hkl1_, 0, axis =0)
    
    index_hkl = [j for j,k in enumerate(hkl_all_class.keys()) for i in range(len(hkl_all_class[k]["family"]))]
    
    if material_ != material1_:
        ind_offset = index_hkl[-1] + 1
        index_hkl1 = [ind_offset+j for j,k in enumerate(hkl_all_class1.keys()) for i in range(len(hkl_all_class1[k]["family"]))]

    if material_ == material1_:
        index_hkl1 = None
        normal_hkl1 = None
        classhkl1 = None
        hkl_all_class1 = None
        lattice_material1 = None
        symmetry1 = None
    
    write_to_console("Generating "+type_+" and saving them")
    
    if material_ != material1_:
        nb_grains_list = list(range(nb_grains+1))
        nb_grains1_list = list(range(nb_grains1+1))
        list_permute = list(itertools.product(nb_grains_list, nb_grains1_list))
        list_permute.pop(0)
        max_progress = len(list_permute)*grains_nb_simulate
        
        if matrix_phase_always_present != None and type_ != "testing_data":
            dummy_, key_material_new = matrix_phase_always_present.split(';')
            if key_material_new == material_:
                max_progress = len(list_permute)*grains_nb_simulate + (len(nb_grains1_list)-1)*grains_nb_simulate
            else:
                max_progress = len(list_permute)*grains_nb_simulate + (len(nb_grains_list)-1)*grains_nb_simulate
    else:
        max_progress = nb_grains*grains_nb_simulate
        if matrix_phase_always_present != None and type_ != "testing_data":
            max_progress = nb_grains*grains_nb_simulate*2
            
    if include_scm:
        max_progress = max_progress + grains_nb_simulate
        if material_ != material1_:
             max_progress = max_progress + 2*grains_nb_simulate
                 
    _inputs_queue = Queue()
    _outputs_queue = Queue()
    _worker_process = {}
    for i in range(ncpu):
        _worker_process[i]= Process(target=worker_generation, args=(_inputs_queue, 
                                                                          _outputs_queue, 
                                                                          i+1),)
    for i in range(ncpu):
        _worker_process[i].start()            
    time.sleep(0.1)    
    
    if material_ != material1_:
        if modelp == "uniform":
            
            if type_ =="training_data":
                xlim, ylim = 0, int(0.8*2000)
            else:
                xlim, ylim = int(0.8*2000), 2000-1
            path_array = resource_path("uniform_orientations_2000.npz")
            arr = np.load(path_array)
            
            if symmetry == symmetry.cubic:
                odf_data = arr["arr_6"][xlim:ylim]
                # print("Laue group 11")
            elif symmetry == symmetry.hexagonal:
                odf_data = arr["arr_5"][xlim:ylim]
                # print("Laue group 9")
            elif symmetry == symmetry.trigonal:
                odf_data = arr["arr_4"][xlim:ylim]
                # print("Laue group 7")
            elif symmetry == symmetry.tetragonal:
                odf_data = arr["arr_3"][xlim:ylim]
                # print("Laue group 5")
            elif symmetry == symmetry.orthorhombic:
                odf_data = arr["arr_2"][xlim:ylim]
                # print("Laue group 3")
            elif symmetry == symmetry.monoclinic:
                odf_data = arr["arr_1"][xlim:ylim]
                # print("Laue group 2")
            elif symmetry == symmetry.triclinic:
                odf_data = arr["arr_0"][xlim:ylim]
                # print("Laue group 1")
                                
            if symmetry1 == symmetry.cubic:
                odf_data1 = arr["arr_6"][xlim:ylim]
                # print("Laue group 11")
            elif symmetry1 == symmetry.hexagonal:
                odf_data1 = arr["arr_5"][xlim:ylim]
                # print("Laue group 9")
            elif symmetry1 == symmetry.trigonal:
                odf_data1 = arr["arr_4"][xlim:ylim]
                # print("Laue group 7")
            elif symmetry1 == symmetry.tetragonal:
                odf_data1 = arr["arr_3"][xlim:ylim]
                # print("Laue group 5")
            elif symmetry1 == symmetry.orthorhombic:
                odf_data1 = arr["arr_2"][xlim:ylim]
                # print("Laue group 3")
            elif symmetry1 == symmetry.monoclinic:
                odf_data1 = arr["arr_1"][xlim:ylim]
                # print("Laue group 2")
            elif symmetry1 == symmetry.triclinic:
                odf_data1 = arr["arr_0"][xlim:ylim]
                # print("Laue group 1")
        ## list of combination of training dataset
        ## to be seen if this improves the prediction quality
        ## increases time significantly to generate the data 
        nb_grains_list = list(range(nb_grains+1))
        nb_grains1_list = list(range(nb_grains1+1))
        list_permute = list(itertools.product(nb_grains_list, nb_grains1_list))
        list_permute.pop(0) ## removing the 0,0 index

        # Idea 2 Or generate a database upto n grain LP
        values = []
        for i in range(len(list_permute)):
            ii, jj = list_permute[i]
            
            for j in range(grains_nb_simulate):
                if data_realism:
                    ## three types of data augmentation to mimic reality ?
                    if j < grains_nb_simulate*0.25:
                        noisy_data = False
                        remove_peaks = False
                    elif (j >= grains_nb_simulate*0.25) and (j < grains_nb_simulate*0.5):
                        noisy_data = True
                        remove_peaks = False
                    elif (j >= grains_nb_simulate*0.5) and (j < grains_nb_simulate*0.75):
                        noisy_data = False
                        remove_peaks = True
                    elif (j >= grains_nb_simulate*0.75):
                        noisy_data = True
                        remove_peaks = True
                else:
                    noisy_data = False
                    remove_peaks = False
                
                if modelp == "uniform":
                    rand_choice = np.random.choice(len(odf_data), ii, replace=False)
                    rand_choice1 = np.random.choice(len(odf_data1), jj, replace=False)
                    data_odf_data = odf_data[rand_choice,:,:]
                    data_odf_data1 = odf_data1[rand_choice1,:,:]
                else:
                    data_odf_data = None
                    data_odf_data1 = None

                seednumber = np.random.randint(1e6)
                values.append([ii, jj, material_,material1_,
                                emin, emax, detectorparameters,
                                pixelsize,True,
                                ang_maxx, step,
                                classhkl, classhkl1,
                                noisy_data, 
                                remove_peaks,
                                seednumber,
                                hkl_all_class,
                                lattice_material,
                                None,
                                normal_hkl,
                                index_hkl, 
                                hkl_all_class1,
                                lattice_material1,
                                None,
                                normal_hkl1,
                                index_hkl1, 
                                dim1, dim2,
                                removeharmonics,
                                0, i, j, save_directory_, 
                                data_odf_data,
                                data_odf_data1,
                                modelp,
                                misorientation_angle,
                                max_millerindex,max_millerindex1,
                                general_diff_rules,
                                crystal, 
                                crystal1,
                                None])
                
                if matrix_phase_always_present != None and \
                    type_ != "testing_data":
                    
                    dummy_, key_material_new = matrix_phase_always_present.split(';')
                    
                    if key_material_new == material_ and ii == 0:
                        values.append([0, jj, material_,material1_,
                                        emin, emax, detectorparameters,
                                        pixelsize,True,
                                        ang_maxx, step,
                                        classhkl, classhkl1,
                                        noisy_data, 
                                        remove_peaks,
                                        seednumber,
                                        hkl_all_class,
                                        lattice_material,
                                        None,
                                        normal_hkl,
                                        index_hkl, 
                                        hkl_all_class1,
                                        lattice_material1,
                                        None,
                                        normal_hkl1,
                                        index_hkl1, 
                                        dim1, dim2,
                                        removeharmonics,
                                        0, i, j, save_directory_, 
                                        data_odf_data,
                                        data_odf_data1,
                                        modelp,
                                        misorientation_angle,
                                        max_millerindex,max_millerindex1,
                                        general_diff_rules,
                                        crystal, 
                                        crystal1,
                                        matrix_phase_always_present])

                    elif key_material_new == material1_ and jj == 0:
                        values.append([ii, 0, material_,material1_,
                                        emin, emax, detectorparameters,
                                        pixelsize,True,
                                        ang_maxx, step,
                                        classhkl, classhkl1,
                                        noisy_data, 
                                        remove_peaks,
                                        seednumber,
                                        hkl_all_class,
                                        lattice_material,
                                        None,
                                        normal_hkl,
                                        index_hkl, 
                                        hkl_all_class1,
                                        lattice_material1,
                                        None,
                                        normal_hkl1,
                                        index_hkl1, 
                                        dim1, dim2,
                                        removeharmonics,
                                        0, i, j, save_directory_, 
                                        data_odf_data,
                                        data_odf_data1,
                                        modelp,
                                        misorientation_angle,
                                        max_millerindex,max_millerindex1,
                                        general_diff_rules,
                                        crystal, 
                                        crystal1,
                                        matrix_phase_always_present])
                
        chunks = chunker_list(values, ncpu)
        chunks_mp = list(chunks)

        if include_scm:
            meta = {'t1':time.time(),
                    'flag':0}
        else:
            meta = {'t1':time.time(),
                    'flag':1}
        for ijk in range(int(ncpu)):
            _inputs_queue.put((chunks_mp[ijk], ncpu, meta))

    else:
        # Idea 2 Or generate a database upto n grain LP
        if modelp == "uniform":
            ## training split                
            if type_ =="training_data":
                xlim, ylim = 0, int(0.8*2000)
            else:
                xlim, ylim = int(0.8*2000), 2000-1
            path_array = resource_path("uniform_orientations_2000.npz")
            arr = np.load(path_array)
            
            if symmetry == symmetry.cubic:
                odf_data = arr["arr_6"][xlim:ylim]
                print("Laue group 11")
            elif symmetry == symmetry.hexagonal:
                odf_data = arr["arr_5"][xlim:ylim]
                print("Laue group 9")
            elif symmetry == symmetry.trigonal:
                odf_data = arr["arr_4"][xlim:ylim]
                print("Laue group 7")
            elif symmetry == symmetry.tetragonal:
                odf_data = arr["arr_3"][xlim:ylim]
                print("Laue group 5")
            elif symmetry == symmetry.orthorhombic:
                odf_data = arr["arr_2"][xlim:ylim]
                print("Laue group 3")
            elif symmetry == symmetry.monoclinic:
                odf_data = arr["arr_1"][xlim:ylim]
                print("Laue group 2")
            elif symmetry == symmetry.triclinic:
                odf_data = arr["arr_0"][xlim:ylim]
                print("Laue group 1")

        values = []
        for i in range(nb_grains):
            for j in range(grains_nb_simulate):
                if data_realism:
                    ## three types of data augmentation to mimic reality ?
                    if j < grains_nb_simulate*0.25:
                        noisy_data = False
                        remove_peaks = False
                    elif (j >= grains_nb_simulate*0.25) and (j < grains_nb_simulate*0.5):
                        noisy_data = True
                        remove_peaks = False
                    elif (j >= grains_nb_simulate*0.5) and (j < grains_nb_simulate*0.75):
                        noisy_data = False
                        remove_peaks = True
                    elif (j >= grains_nb_simulate*0.75):
                        noisy_data = True
                        remove_peaks = True
                else:
                    noisy_data = False
                    remove_peaks = False
                
                if modelp == "uniform":
                    rand_choice = np.random.choice(len(odf_data), i+1, replace=False)
                    data_odf_data = odf_data[rand_choice,:,:]
                    data_odf_data1 = None
                else:
                    data_odf_data = None
                    data_odf_data1 = None
                    
                seednumber = np.random.randint(1e6)
                values.append([i+1, 0, material_,material1_,
                                emin, emax, detectorparameters,
                                pixelsize,True,
                                ang_maxx, step,
                                classhkl, classhkl1,
                                noisy_data, 
                                remove_peaks,
                                seednumber,
                                hkl_all_class,
                                lattice_material,
                                None,
                                normal_hkl,
                                index_hkl, 
                                hkl_all_class1,
                                lattice_material1,
                                None,
                                normal_hkl1,
                                index_hkl1, 
                                dim1, dim2,
                                removeharmonics,
                                0, i, j, save_directory_, 
                                data_odf_data,
                                data_odf_data1,
                                modelp,
                                misorientation_angle,
                                max_millerindex,max_millerindex1,
                                general_diff_rules,
                                crystal, 
                                crystal1,
                                None])
                
                if matrix_phase_always_present != None and \
                    type_ != "testing_data":
                    values.append([i+1, 0, material_,material1_,
                                    emin, emax, detectorparameters,
                                    pixelsize,True,
                                    ang_maxx, step,
                                    classhkl, classhkl1,
                                    noisy_data, 
                                    remove_peaks,
                                    seednumber,
                                    hkl_all_class,
                                    lattice_material,
                                    None,
                                    normal_hkl,
                                    index_hkl, 
                                    hkl_all_class1,
                                    lattice_material1,
                                    None,
                                    normal_hkl1,
                                    index_hkl1, 
                                    dim1, dim2,
                                    removeharmonics,
                                    0, i, j, save_directory_, 
                                    data_odf_data,
                                    data_odf_data1,
                                    modelp,
                                    misorientation_angle,
                                    max_millerindex,max_millerindex1,
                                    general_diff_rules,
                                    crystal, 
                                    crystal1,
                                    matrix_phase_always_present])
                
        chunks = chunker_list(values, ncpu)
        chunks_mp = list(chunks)
        
        if include_scm:
            meta = {'t1':time.time(),
                    'flag':0}
        else:
            meta = {'t1':time.time(),
                    'flag':1}
        for ijk in range(int(ncpu)):
            _inputs_queue.put((chunks_mp[ijk], ncpu, meta))

    if include_scm:
        write_to_console("Generating small angle misorientation single crystals")  
        values = []
        for i in range(grains_nb_simulate):
            if data_realism:
                ## three types of data augmentation to mimic reality ?
                if i < grains_nb_simulate*0.25:
                    noisy_data = False
                    remove_peaks = False
                elif (i >= grains_nb_simulate*0.25) and (i < grains_nb_simulate*0.5):
                    noisy_data = True
                    remove_peaks = False
                elif (i >= grains_nb_simulate*0.5) and (i < grains_nb_simulate*0.75):
                    noisy_data = False
                    remove_peaks = True
                elif (i >= grains_nb_simulate*0.75):
                    noisy_data = True
                    remove_peaks = True
            else:
                noisy_data = False
                remove_peaks = False
            seednumber = np.random.randint(1e6)
            values.append([1, 0, material_,material1_,
                                    emin, emax, detectorparameters,
                                    pixelsize,True,
                                    ang_maxx, step,
                                    classhkl, classhkl1,
                                    noisy_data, 
                                    remove_peaks,
                                    seednumber,
                                    hkl_all_class,
                                    lattice_material,
                                    None,
                                    normal_hkl,
                                    index_hkl, 
                                    hkl_all_class1,
                                    lattice_material1,
                                    None,
                                    normal_hkl1,
                                    index_hkl1, 
                                    dim1, dim2,
                                    removeharmonics,
                                    1, i, i, save_directory_,
                                    None, None, modelp,
                                    misorientation_angle,
                                    max_millerindex,max_millerindex1,
                                    general_diff_rules,
                                    crystal, 
                                    crystal1,
                                    None])
            
            if material_ != material1_:
                seednumber = np.random.randint(1e6)
                values.append([0, 1, material_,material1_,
                                    emin, emax, detectorparameters,
                                    pixelsize,True,
                                    ang_maxx, step,
                                    classhkl, classhkl1,
                                    noisy_data, 
                                    remove_peaks,
                                    seednumber,
                                    hkl_all_class,
                                    lattice_material,
                                    None,
                                    normal_hkl,
                                    index_hkl, 
                                    hkl_all_class1,
                                    lattice_material1,
                                    None,
                                    normal_hkl1,
                                    index_hkl1, 
                                    dim1, dim2,
                                    removeharmonics,
                                    2, i, i, save_directory_,
                                    None, None, modelp,
                                    misorientation_angle,
                                    max_millerindex,max_millerindex1,
                                    general_diff_rules,
                                    crystal, 
                                    crystal1,
                                    None])
                
                ### include slightly misoriented two crystals of different materails
                seednumber = np.random.randint(1e6)
                values.append([1, 1, material_,material1_,
                                    emin, emax, detectorparameters,
                                    pixelsize,True,
                                    ang_maxx, step,
                                    classhkl, classhkl1,
                                    noisy_data, 
                                    remove_peaks,
                                    seednumber,
                                    hkl_all_class,
                                    lattice_material,
                                    None,
                                    normal_hkl,
                                    index_hkl, 
                                    hkl_all_class1,
                                    lattice_material1,
                                    None,
                                    normal_hkl1,
                                    index_hkl1, 
                                    dim1, dim2,
                                    removeharmonics,
                                    3, i, i, save_directory_,
                                    None, None, modelp,
                                    misorientation_angle,
                                    max_millerindex,max_millerindex1,
                                    general_diff_rules,
                                    crystal, 
                                    crystal1,
                                    None])
                
        chunks = chunker_list(values, ncpu)
        chunks_mp = list(chunks)

        meta = {'t1':time.time(),
                'flag':1}
        for ijk in range(int(ncpu)):
            _inputs_queue.put((chunks_mp[ijk], ncpu, meta))
            
    max_progress = max_progress
    while True:
        count = 0
        for i in range(ncpu):
            if not _worker_process[i].is_alive():
                _worker_process[i].join()
                count += 1
            else:
                time.sleep(0.1)
                
        if count == ncpu:
            return

def get_material_detail(material_=None, SG=None, symm_=None,
                        material1_=None, SG1=None, symm1_=None):
    """
        Returns material details

    """
    a, b, c, alpha, beta, gamma = dictLT.dict_Materials[material_][1]
    # Gstar = CP.Gstar_from_directlatticeparams(a, b, c, alpha, beta, gamma)
    rules = dictLT.dict_Materials[material_][-1]
    
    if symm_ =="cubic":
        symmetry = Symmetry.cubic
        lattice_material = Lattice.cubic(a)
        if SG == None:
            SG = 230
        crystal = SGLattice(int(SG), a)
    elif symm_ =="monoclinic":
        symmetry = Symmetry.monoclinic
        lattice_material = Lattice.monoclinic(a, b, c, beta)
        if SG == None:
            SG = 10
        crystal = SGLattice(int(SG),a, b, c, beta)
    elif symm_ == "hexagonal":
        symmetry = Symmetry.hexagonal
        lattice_material = Lattice.hexagonal(a, c)
        if SG == None:
            SG = 191
        crystal = SGLattice(int(SG),a, c)
    elif symm_ == "orthorhombic":
        symmetry = Symmetry.orthorhombic
        lattice_material = Lattice.orthorhombic(a, b, c)
        if SG == None:
            SG = 47
        crystal = SGLattice(int(SG),a, b, c)
    elif symm_ == "tetragonal":
        symmetry = Symmetry.tetragonal
        lattice_material = Lattice.tetragonal(a, c)
        if SG == None:
            SG = 123
        crystal = SGLattice(int(SG),a, c)
    elif symm_ == "trigonal":
        symmetry = Symmetry.trigonal
        lattice_material = Lattice.rhombohedral(a, alpha)
        if SG == None:
            SG = 162
        crystal = SGLattice(int(SG),a, alpha)
    elif symm_ == "triclinic":
        symmetry = Symmetry.triclinic
        lattice_material = Lattice.triclinic(a, b, c, alpha, beta, gamma)
        if SG == None:
            SG = 2
        crystal = SGLattice(int(SG),a, b, c, alpha, beta, gamma)
    
    
    if material_ != material1_:
        a1, b1, c1, alpha1, beta1, gamma1 = dictLT.dict_Materials[material1_][1]
        # Gstar1 = CP.Gstar_from_directlatticeparams(a1, b1, c1, alpha1, beta1, gamma1)
        rules1 = dictLT.dict_Materials[material1_][-1]
        # =============================================================================
        # Symmetry input
        # =============================================================================
        if symm1_ =="cubic":
            symmetry1 = Symmetry.cubic
            lattice_material1 = Lattice.cubic(a1)
            if SG1 == None:
                SG1 = 230
            crystal1 = SGLattice(int(SG1), a1)
        elif symm1_ =="monoclinic":
            symmetry1 = Symmetry.monoclinic
            lattice_material1 = Lattice.monoclinic(a1, b1, c1, beta1)
            if SG1 == None:
                SG1 = 10
            crystal1 = SGLattice(int(SG1),a1, b1, c1, beta1)
        elif symm1_ == "hexagonal":
            symmetry1 = Symmetry.hexagonal
            lattice_material1 = Lattice.hexagonal(a1, c1)
            if SG1 == None:
                SG1 = 191
            crystal1 = SGLattice(int(SG1),a1, c1)
        elif symm1_ == "orthorhombic":
            symmetry1 = Symmetry.orthorhombic
            lattice_material1 = Lattice.orthorhombic(a1, b1, c1)
            if SG1 == None:
                SG1 = 47
            crystal1 = SGLattice(int(SG1),a1, b1, c1)
        elif symm1_ == "tetragonal":
            symmetry1 = Symmetry.tetragonal
            lattice_material1 = Lattice.tetragonal(a1, c1)
            if SG1 == None:
                SG1 = 123
            crystal1 = SGLattice(int(SG1),a1, c1)
        elif symm1_ == "trigonal":
            symmetry1 = Symmetry.trigonal
            lattice_material1 = Lattice.rhombohedral(a1, alpha1)
            if SG1 == None:
                SG1 = 162
            crystal1 = SGLattice(int(SG1),a1, alpha1)
        elif symm1_ == "triclinic":
            symmetry1 = Symmetry.triclinic
            lattice_material1 = Lattice.triclinic(a1, b1, c1, alpha1, beta1, gamma1)
            if SG1 == None:
                SG1 = 2
            crystal1 = SGLattice(int(SG1),a1, b1, c1, alpha1, beta1, gamma1)
    else:
        rules1 = None
        symmetry1 = None
        lattice_material1 = None
        crystal1 = None
        
    return rules, symmetry, lattice_material, crystal, SG, rules1, symmetry1, lattice_material1, crystal1, SG1


def predict_preprocessMultiProcess(files, cnt, 
                                     rotation_matrix,strain_matrix,strain_matrixs,
                                    col,colx,coly,match_rate,spots_len,iR_pix,fR_pix,best_match,mat_global,
                                    check,detectorparameters,pixelsize,angbins,
                                    classhkl, hkl_all_class0, hkl_all_class1, emin, emax,
                                    material_, material1_, symmetry, symmetry1,lim_x,lim_y,
                                    strain_calculation, ind_mat, ind_mat1,
                                    model_direc=None, tolerance =None, tolerance1 =None,
                                   matricies=None, ccd_label=None,
                                   filename_bkg=None,intensity_threshold=None,
                                   boxsize=None,bkg_treatment=None,
                                   filenameDirec=None, experimental_prefix=None,
                                   blacklist_file =None, text_file=None, 
                                   files_treated=None,try_previous1=False,
                                   wb=None, temp_key=None, cor_file_directory=None, mode_spotCycle1=None,
                                   softmax_threshold_global123=None,mr_threshold_global123=None,
                                   cap_matchrate123=None,tolerance_strain123=None,tolerance_strain1231=None,\
                                   NumberMaxofFits123=None,fit_peaks_gaussian_global123=None,
                                   FitPixelDev_global123=None,coeff123=None, coeff_overlap=None,
                                   material0_limit=None, material1_limit=None, use_previous_UBmatrix_name=None,
                                   material_phase_always_present=None, crystal=None, crystal1=None, strain_free_parameters=None):
    
    if files in files_treated:
        return strain_matrix, strain_matrixs, rotation_matrix, col, colx, coly, \
            match_rate, mat_global, cnt, files_treated,spots_len,iR_pix,fR_pix, check, best_match
    
    call_global()
    # print("Predicting for "+files)    
    if files.split(".")[-1] != "cor":
        CCDLabel=ccd_label
        seednumber = "Experimental "+CCDLabel+" file"    
        
        try:
            out_name = blacklist_file
        except:
            out_name = None  
            
        if bkg_treatment == None:
            bkg_treatment = "A-B"
            
        try:
            ### Max space = space betzeen pixles
            peak_XY = RMCCD.PeakSearch(
                                        files,
                                        stackimageindex = -1,
                                        CCDLabel=CCDLabel,
                                        NumberMaxofFits=NumberMaxofFits123,
                                        PixelNearRadius=10,
                                        removeedge=2,
                                        IntensityThreshold=intensity_threshold,
                                        local_maxima_search_method=0,
                                        boxsize=boxsize,
                                        position_definition=1,
                                        verbose=0,
                                        fit_peaks_gaussian=fit_peaks_gaussian_global123,
                                        xtol=0.001,                
                                        FitPixelDev=FitPixelDev_global123,
                                        return_histo=0,
                                        # Saturation_value=1e10,  # to be merged in CCDLabel
                                        # Saturation_value_flatpeak=1e10,
                                        MinIntensity=0,
                                        PeakSizeRange=(0.65,200),
                                        write_execution_time=1,
                                        Data_for_localMaxima = "auto_background",
                                        formulaexpression=bkg_treatment,
                                        Remove_BlackListedPeaks_fromfile=out_name,
                                        reject_negative_baseline=True,
                                        Fit_with_Data_for_localMaxima=False,
                                        maxPixelDistanceRejection=15.0,
                                        )
            peak_XY = peak_XY[0]#[:,:2] ##[2] Integer peak lists
        except:
            print("Error in Peak detection for "+ files)
            for intmat in range(matricies):
                rotation_matrix[intmat][0][cnt,:,:] = np.zeros((3,3))
                strain_matrix[intmat][0][cnt,:,:] = np.zeros((3,3))
                strain_matrixs[intmat][0][cnt,:,:] = np.zeros((3,3))
                col[intmat][0][cnt,:] = 0,0,0
                colx[intmat][0][cnt,:] = 0,0,0
                coly[intmat][0][cnt,:] = 0,0,0
                match_rate[intmat][0][cnt] = 0
                mat_global[intmat][0][cnt] = 0
                spots_len[intmat][0][cnt] = 0
                iR_pix[intmat][0][cnt] = 0
                fR_pix[intmat][0][cnt] = 0
                check[cnt,intmat] = 0
            # files_treated.append(files)
            return strain_matrix, strain_matrixs, rotation_matrix, col, colx, coly, \
                match_rate, mat_global, cnt, files_treated,spots_len,iR_pix,fR_pix, check, best_match
        
        try:
            s_ix = np.argsort(peak_XY[:, 2])[::-1]
            peak_XY = peak_XY[s_ix]
        except:
            print("Error in Peak detection for "+ files)
            for intmat in range(matricies):
                rotation_matrix[intmat][0][cnt,:,:] = np.zeros((3,3))
                strain_matrix[intmat][0][cnt,:,:] = np.zeros((3,3))
                strain_matrixs[intmat][0][cnt,:,:] = np.zeros((3,3))
                col[intmat][0][cnt,:] = 0,0,0
                colx[intmat][0][cnt,:] = 0,0,0
                coly[intmat][0][cnt,:] = 0,0,0
                match_rate[intmat][0][cnt] = 0
                mat_global[intmat][0][cnt] = 0
                spots_len[intmat][0][cnt] = 0
                iR_pix[intmat][0][cnt] = 0
                fR_pix[intmat][0][cnt] = 0
                check[cnt,intmat] = 0
            # files_treated.append(files)
            return strain_matrix, strain_matrixs, rotation_matrix, col, colx, coly, \
                match_rate, mat_global, cnt, files_treated,spots_len,iR_pix,fR_pix, check, best_match
        
        
        framedim = dictLT.dict_CCD[CCDLabel][0]
        twicetheta, chi = Lgeo.calc_uflab(peak_XY[:,0], peak_XY[:,1], detectorparameters,
                                            returnAngles=1,
                                            pixelsize=pixelsize,
                                            kf_direction='Z>0')
        data_theta, data_chi = twicetheta/2., chi
        
        framedim = dictLT.dict_CCD[CCDLabel][0]
        dict_dp={}
        dict_dp['kf_direction']='Z>0'
        dict_dp['detectorparameters']=detectorparameters
        dict_dp['detectordistance']=detectorparameters[0]
        dict_dp['detectordiameter']=pixelsize*framedim[0]
        dict_dp['pixelsize']=pixelsize
        dict_dp['dim']=framedim
        dict_dp['peakX']=peak_XY[:,0]
        dict_dp['peakY']=peak_XY[:,1]
        dict_dp['intensity']=peak_XY[:,2]
        
        CCDcalib = {"CCDLabel":CCDLabel,
                    "dd":detectorparameters[0], 
                    "xcen":detectorparameters[1], 
                    "ycen":detectorparameters[2], 
                    "xbet":detectorparameters[3], 
                    "xgam":detectorparameters[4],
                    "pixelsize": pixelsize}
        
        path = os.path.normpath(files)
        IOLT.writefile_cor(cor_file_directory+"//"+path.split(os.sep)[-1].split(".")[0], twicetheta, 
                           chi, peak_XY[:,0], peak_XY[:,1], peak_XY[:,2],
                           param=CCDcalib, sortedexit=0)
        
    elif files.split(".")[-1] == "cor":
        # print("Entering Cor file read section")
        seednumber = "Experimental COR file"
        allres = IOLT.readfile_cor(files, True)
        data_theta, data_chi, peakx, peaky, intensity = allres[1:6]
        CCDcalib = allres[-1]
        detectorparameters = allres[-2]
        # print('detectorparameters from file are: '+ str(detectorparameters))
        pixelsize = CCDcalib['pixelsize']
        CCDLabel = CCDcalib['CCDLabel']
        framedim = dictLT.dict_CCD[CCDLabel][0]
        dict_dp={}
        dict_dp['kf_direction']='Z>0'
        dict_dp['detectorparameters']=detectorparameters
        dict_dp['detectordistance']=detectorparameters[0]
        dict_dp['detectordiameter']=pixelsize*framedim[0]
        dict_dp['pixelsize']=pixelsize
        dict_dp['dim']=framedim
        dict_dp['peakX']=peakx
        dict_dp['peakY']=peaky
        dict_dp['intensity']=intensity

    sorted_data = np.transpose(np.array([data_theta, data_chi]))
    tabledistancerandom = np.transpose(GT.calculdist_from_thetachi(sorted_data, sorted_data))

    codebars_all = []
    
    if len(data_theta) == 0:
        print("No peaks Found for : " + files)
        for intmat in range(matricies):
            rotation_matrix[intmat][0][cnt,:,:] = np.zeros((3,3))
            strain_matrix[intmat][0][cnt,:,:] = np.zeros((3,3))
            strain_matrixs[intmat][0][cnt,:,:] = np.zeros((3,3))
            col[intmat][0][cnt,:] = 0,0,0
            colx[intmat][0][cnt,:] = 0,0,0
            coly[intmat][0][cnt,:] = 0,0,0
            match_rate[intmat][0][cnt] = 0
            mat_global[intmat][0][cnt] = 0
            spots_len[intmat][0][cnt] = 0
            iR_pix[intmat][0][cnt] = 0
            fR_pix[intmat][0][cnt] = 0
            check[cnt,intmat] = 0
        return strain_matrix, strain_matrixs, rotation_matrix, col, colx, coly, \
                match_rate, mat_global, cnt, files_treated,spots_len,iR_pix,fR_pix, check, best_match
    
    if not use_om_user:
        # print("Entering GOOD section")
        spots_in_center = np.arange(0,len(data_theta))
        spots_in_center = spots_in_center[:nb_spots_consider]
        
        for i in spots_in_center:
            spotangles = tabledistancerandom[i]
            spotangles = np.delete(spotangles, i)# removing the self distance
            codebars = np.histogram(spotangles, bins=angbins)[0]
            # codebars = histogram1d(spotangles, range=[min(angbins),max(angbins)], bins=len(angbins)-1)
            ## normalize the same way as training data
            max_codebars = np.max(codebars)
            codebars = codebars/ max_codebars
            codebars_all.append(codebars)
            
        ## reshape for the model to predict all spots at once
        codebars = np.array(codebars_all)
        ## Do prediction of all spots at once
        prediction = predict(codebars, wb, temp_key)
        
        # prediction = model.predict(codebars)
        max_pred = np.max(prediction, axis = 1)
        class_predicted = np.argmax(prediction, axis = 1)
        
        predicted_hkl123 = classhkl[class_predicted]
        predicted_hkl123 = predicted_hkl123.astype(int)
    else:
        max_pred = None
        class_predicted = None
        predicted_hkl123 = None
        spots_in_center = None
        
    s_tth = data_theta * 2.
    s_chi = data_chi
    
    # print("Computing UB")
    rotation_matrix1, mr_highest, mat_highest, \
        strain_crystal, strain_sample, iR_pix1, \
                    fR_pix1, spots_len1,\
                    best_match1, check12 = predict_ubmatrix(seednumber, spots_in_center, classhkl, 
                                                hkl_all_class0, 
                                                hkl_all_class1, files,
                                                  s_tth1=s_tth,s_chi1=s_chi,
                                                  predicted_hkl1=predicted_hkl123,
                                                  class_predicted1=class_predicted,
                                                  max_pred1=max_pred,
                                                  emin=emin,emax=emax,
                                                  material_=material_, 
                                                  material1_=material1_, 
                                                  lim_y=lim_y, lim_x=lim_x, 
                                                  cnt=cnt,
                                                  dict_dp=dict_dp,
                                                  rotation_matrix=rotation_matrix,
                                                  mat_global=mat_global,
                                                  strain_calculation=strain_calculation,
                                                  ind_mat=ind_mat, 
                                                  ind_mat1=ind_mat1,
                                                  tolerance=tolerance, 
                                                  tolerance1 =tolerance1,
                                                  matricies=matricies,
                                                  tabledistancerandom=tabledistancerandom,
                                                  text_file = text_file,
                                                  try_previous1=try_previous1,
                                                  mode_spotCycle=mode_spotCycle1,
                                                  softmax_threshold_global123 = softmax_threshold_global123,
                                                  mr_threshold_global123=mr_threshold_global123,
                                                  cap_matchrate123=cap_matchrate123,
                                                  tolerance_strain123=tolerance_strain123,
                                                  tolerance_strain1231=tolerance_strain1231,
                                                  coeff123=coeff123,
                                                  coeff_overlap=coeff_overlap,
                                                  material0_limit=material0_limit, 
                                                  material1_limit=material1_limit,
                                                  model_direc=model_direc,
                                                  use_previous_UBmatrix_name=use_previous_UBmatrix_name,
                                                  material_phase_always_present=material_phase_always_present,
                                                  match_rate=match_rate,
                                                  check=check[cnt,:],
                                                  crystal=crystal,
                                                  crystal1=crystal1, angbins=angbins,
                                                  wb=wb, temp_key=temp_key,
                                                  strain_free_parameters=strain_free_parameters)
    for intmat in range(matricies):
        if len(rotation_matrix1[intmat]) == 0:
            col[intmat][0][cnt,:] = 0,0,0
            colx[intmat][0][cnt,:] = 0,0,0
            coly[intmat][0][cnt,:] = 0,0,0
        else:
            mat_global[intmat][0][cnt] = mat_highest[intmat][0]
            
            final_symm =symmetry
            final_crystal = crystal
            if mat_highest[intmat][0] == 1:
                final_symm = symmetry
                final_crystal = crystal
            elif mat_highest[intmat][0] == 2:
                final_symm = symmetry1
                final_crystal = crystal1
            symm_operator = final_crystal._hklsym
            strain_matrix[intmat][0][cnt,:,:] = strain_crystal[intmat][0]
            strain_matrixs[intmat][0][cnt,:,:] = strain_sample[intmat][0]
            rotation_matrix[intmat][0][cnt,:,:] = rotation_matrix1[intmat][0]
            col_temp = get_ipf_colour(rotation_matrix1[intmat][0], np.array([0., 0., 1.]), final_symm, symm_operator)
            col[intmat][0][cnt,:] = col_temp
            col_tempx = get_ipf_colour(rotation_matrix1[intmat][0], np.array([1., 0., 0.]), final_symm, symm_operator)
            colx[intmat][0][cnt,:] = col_tempx
            col_tempy = get_ipf_colour(rotation_matrix1[intmat][0], np.array([0., 1., 0.]), final_symm, symm_operator)
            coly[intmat][0][cnt,:] = col_tempy
            match_rate[intmat][0][cnt] = mr_highest[intmat][0]
            spots_len[intmat][0][cnt] = spots_len1[intmat][0]
            iR_pix[intmat][0][cnt] = iR_pix1[intmat][0]
            fR_pix[intmat][0][cnt] = fR_pix1[intmat][0]
            best_match[intmat][0][cnt] = best_match1[intmat][0]
            check[cnt,intmat] = check12[intmat]

    files_treated.append(files)
    return strain_matrix, strain_matrixs, rotation_matrix, col, colx, coly, match_rate, \
            mat_global, cnt, files_treated, spots_len, iR_pix, fR_pix, check, best_match

def new_MP_function(argu):
    
    files, cnt, rotation_matrix, strain_matrix, strain_matrixs,\
            col,colx,coly,match_rate,spots_len,iR_pix,fR_pix,best_match,mat_global,\
            check,detectorparameters,pixelsize,angbins,\
            classhkl, hkl_all_class0, hkl_all_class1, emin, emax,\
            material_, material1_, symmetry, symmetry1,lim_x,lim_y,\
            strain_calculation, ind_mat, ind_mat1,\
            model_direc, tolerance , tolerance1,\
            matricies, ccd_label,\
            filename_bkg,intensity_threshold,\
            boxsize,bkg_treatment,\
            filenameDirec, experimental_prefix,\
            blacklist_file, text_file, \
            files_treated,try_previous1,\
            wb, temp_key, cor_file_directory, mode_spotCycle1,\
            softmax_threshold_global123,mr_threshold_global123,\
            cap_matchrate123, tolerance_strain123, tolerance_strain1231,\
            NumberMaxofFits123,fit_peaks_gaussian_global123,\
            FitPixelDev_global123,coeff123,coeff_overlap,\
            material0_limit, material1_limit, use_previous_UBmatrix_name1,\
            material_phase_always_present1, crystal, crystal1, strain_free_parameters = argu
                        
    strain_matrix12, strain_matrixs12, \
        rotation_matrix12, col12, \
            colx12, coly12,\
    match_rate12, mat_global12, cnt12,\
        files_treated12, spots_len12, \
            iR_pix12, fR_pix12, check12, best_match12 = predict_preprocessMultiProcess(files, cnt, 
                                               rotation_matrix,strain_matrix,strain_matrixs,
                                               col,colx,coly,match_rate,spots_len,iR_pix,fR_pix,best_match,
                                               mat_global,
                                               check,detectorparameters,pixelsize,angbins,
                                               classhkl, hkl_all_class0, hkl_all_class1, emin, emax,
                                               material_, material1_, symmetry, symmetry1,lim_x,lim_y,
                                               strain_calculation, ind_mat, ind_mat1,
                                               model_direc, tolerance, tolerance1,
                                               matricies, ccd_label,
                                               filename_bkg,intensity_threshold,
                                               boxsize,bkg_treatment,
                                               filenameDirec, experimental_prefix,
                                               blacklist_file, text_file, 
                                               files_treated,try_previous1,
                                               wb, temp_key, cor_file_directory, mode_spotCycle1,
                                               softmax_threshold_global123,mr_threshold_global123,
                                               cap_matchrate123, tolerance_strain123,
                                               tolerance_strain1231,NumberMaxofFits123,
                                               fit_peaks_gaussian_global123,
                                               FitPixelDev_global123, coeff123,coeff_overlap,
                                               material0_limit,material1_limit,
                                               use_previous_UBmatrix_name1,
                                               material_phase_always_present1,
                                               crystal, crystal1, strain_free_parameters)
    meta = {}
    return strain_matrix12, strain_matrixs12, rotation_matrix12, col12, \
                 colx12, coly12, match_rate12, mat_global12, cnt12, meta, \
                 files_treated12, spots_len12, iR_pix12, fR_pix12, best_match12, check12
                 
                 
def prepare_LP_NB(nbgrains, nbgrains1, material_, verbose, material1_=None, seed=None, sortintensity=False,
               detectorparameters=None, pixelsize=None, dim1=2048,dim2=2048, emin=5, emax=23, flag=0, noisy_data=False,
               remove_peaks = False):
    if flag == 10:
        s_tth, s_chi, s_miller_ind, s_posx, s_posy, \
                                    s_intensity, g, g1  = simulatemultiplepatterns_NB(nbgrains, nbgrains1, seed=seed, 
                                                                                key_material=material_,
                                                                                key_material1=material1_,
                                                                                emin=emin, emax=emax,
                                                                                detectorparameters=detectorparameters,
                                                                                pixelsize=pixelsize,
                                                                                sortintensity=sortintensity, 
                                                                                dim1=dim1,dim2=dim2, flag=flag)
    else:
        s_tth, s_chi, s_miller_ind, s_posx, s_posy, \
                                    s_intensity  = simulatemultiplepatterns_NB(nbgrains, nbgrains1, seed=seed, 
                                                                                key_material=material_,
                                                                                key_material1=material1_,
                                                                                emin=emin, emax=emax,
                                                                                detectorparameters=detectorparameters,
                                                                                pixelsize=pixelsize,
                                                                                sortintensity=sortintensity, 
                                                                                dim1=dim1,dim2=dim2, flag=flag)

    
    if noisy_data:
        ## apply random gaussian type noise to the data (tth and chi)
        ## So adding noise to the angular distances
        ## Instead of adding noise to all HKL's ... Add to few selected HKLs
        ## Adding noise to randomly 30% of the HKLs
        indices_noise = np.random.choice(len(s_tth), int(len(s_tth)*0.2), replace=False)
        noise_ = np.random.normal(0,0.1,len(indices_noise))
        s_tth[indices_noise] = s_tth[indices_noise] + noise_
        s_chi[indices_noise] = s_chi[indices_noise] + noise_
            
    if remove_peaks:
        len_mi = np.array([iq for iq in range(len(s_miller_ind))])
        len_mi = len_mi[int(0.5*len(s_miller_ind)):]
        indices_remove = np.random.choice(len_mi, int(len(len_mi)*0.2), replace=False)
        ## delete randomly selected less intense peaks
        ## to simulate real peak detection, where some peaks may not be
        ## well detected
        s_tth = np.delete(s_tth, indices_remove)
        s_chi = np.delete(s_chi, indices_remove)
        s_posx = np.delete(s_posx, indices_remove)
        s_posy = np.delete(s_posy, indices_remove)
        s_intensity = np.delete(s_intensity, indices_remove)
        s_miller_ind = np.delete(s_miller_ind, indices_remove, axis=0)
        
    # considering all spots
    allspots_the_chi = np.transpose(np.array([s_tth/2., s_chi]))
    tabledistancerandom = np.transpose(GT.calculdist_from_thetachi(allspots_the_chi, allspots_the_chi))
    # ground truth
    hkl_sol = s_miller_ind
    if flag == 10:
        return tabledistancerandom, hkl_sol, s_posx, s_posy, s_intensity, s_tth, s_chi, g, g1
    return tabledistancerandom, hkl_sol, s_posx, s_posy, s_intensity, s_tth, s_chi

def simulatemultiplepatterns_NB(nbUBs, nbUBs1, seed=123, key_material=None, key_material1=None, emin=5, emax=23,
                             detectorparameters=None, pixelsize=None,
                             sortintensity = False, dim1=2048, dim2=2048, flag=0):
    detectordiameter = pixelsize * dim1
    g = np.zeros((nbUBs, 3, 3))
    g1 = np.zeros((nbUBs1, 3, 3))
    for igr in range(nbUBs):
        phi1 = np.random.rand() * 360.
        phi = 180. * acos(2 * np.random.rand() - 1) / np.pi
        phi2 = np.random.rand() * 360.
        g[igr] = Euler2OrientationMatrix((phi1, phi, phi2))
    if key_material != key_material1:
        for igr in range(nbUBs1):    
            phi1 = np.random.rand() * 360.
            phi = 180. * acos(2 * np.random.rand() - 1) / np.pi
            phi2 = np.random.rand() * 360.
            g1[igr] = Euler2OrientationMatrix((phi1, phi, phi2))
    l_tth, l_chi, l_miller_ind, l_posx, l_posy, l_E, l_intensity = [],[],[],[],[],[],[]
    for grainind in range(nbUBs):
        UBmatrix = g[grainind]
        grain = CP.Prepare_Grain(key_material, UBmatrix)
        s_tth, s_chi, s_miller_ind, s_posx, s_posy, s_E= LT.SimulateLaue_full_np(grain, emin, emax,
                                                                                 detectorparameters,
                                                                                 pixelsize=pixelsize,
                                                                                 dim=(dim1, dim2),
                                                                                 detectordiameter=detectordiameter,
                                                                                 removeharmonics=1)
        s_miller_ind = np.c_[ s_miller_ind, np.zeros(len(s_miller_ind)) ]
        s_intensity = 1./s_E
        l_tth.append(s_tth)
        l_chi.append(s_chi)
        l_miller_ind.append(s_miller_ind)
        l_posx.append(s_posx)
        l_posy.append(s_posy)
        l_E.append(s_E)
        l_intensity.append(s_intensity)
  
    if key_material != key_material1:
        for grainind in range(nbUBs1):
            if key_material1 != None:
                UBmatrix = g1[grainind]
                grain = CP.Prepare_Grain(key_material1, UBmatrix)
                s_tth, s_chi, s_miller_ind, s_posx, s_posy, s_E= LT.SimulateLaue_full_np(grain, emin, emax,
                                                                                         detectorparameters,
                                                                                         pixelsize=pixelsize,
                                                                                         dim=(dim1, dim2),
                                                                                         detectordiameter=detectordiameter,
                                                                                         removeharmonics=1)
                s_miller_ind = np.c_[ s_miller_ind, np.ones(len(s_miller_ind)) ]
                s_intensity = 1./s_E
                l_tth.append(s_tth)
                l_chi.append(s_chi)
                l_miller_ind.append(s_miller_ind)
                l_posx.append(s_posx)
                l_posy.append(s_posy)
                l_E.append(s_E)
                l_intensity.append(s_intensity)
    #flat_list = [item for sublist in l for item in sublist]
    s_tth = np.array([item for sublist in l_tth for item in sublist])
    s_chi = np.array([item for sublist in l_chi for item in sublist])
    s_miller_ind = np.array([item for sublist in l_miller_ind for item in sublist])
    s_posx = np.array([item for sublist in l_posx for item in sublist])
    s_posy = np.array([item for sublist in l_posy for item in sublist])
    s_E = np.array([item for sublist in l_E for item in sublist])
    s_intensity=np.array([item for sublist in l_intensity for item in sublist])
    if sortintensity:
        indsort = np.argsort(s_intensity)[::-1]
        s_tth=np.take(s_tth, indsort)
        s_chi=np.take(s_chi, indsort)
        s_miller_ind=np.take(s_miller_ind, indsort, axis=0)
        s_posx=np.take(s_posx, indsort)
        s_posy=np.take(s_posy, indsort)
        s_E=np.take(s_E, indsort)
        s_intensity=np.take(s_intensity, indsort)
    if flag == 10:
        return s_tth, s_chi, s_miller_ind, s_posx, s_posy, s_intensity, g, g1
    return s_tth, s_chi, s_miller_ind, s_posx, s_posy, s_intensity

# =============================================================================
# Multi material functions
# =============================================================================

def generate_multimat_dataset(  material_=["Cu"], 
                                ang_maxx=18.,
                                step=0.1, 
                                nb_grains=[1], 
                                grains_nb_simulate=100, 
                                data_realism = False, 
                                detectorparameters=None, 
                                pixelsize=None, 
                                type_="training",
                                var0 = 0, 
                                dim1=2048, 
                                dim2=2048, 
                                removeharmonics=1, 
                                save_directory="",
                                write_to_console=None, 
                                emin=5, 
                                emax=22,
                                modelp = "random",
                                general_diff_rules = False, 
                                crystal = [None]): 
    """
    works for n phases now.
    """
    ncpu = cpu_count()
    ## make sure directory exists
    save_directory_ = save_directory+"//"+type_
    if not os.path.exists(save_directory_):
        os.makedirs(save_directory_)
    
    classhkl, n, hkl_all_class, lattice_material, symmetry = [], [], [], [],[]
    max_millerindex = []
    try:
        for imat in material_:
            with open(save_directory+"//classhkl_data_"+imat+".pickle", "rb") as input_file:
                classhkl_mat, _, _, n_mat, _, \
                    hkl_all_class_mat, _, \
                        lattice_material_mat, symmetry_mat = cPickle.load(input_file)
                        
            classhkl.append(classhkl_mat)
            n.append(n_mat)
            hkl_all_class.append(hkl_all_class_mat)
            lattice_material.append(lattice_material_mat)
            symmetry.append(symmetry_mat)
            max_millerindex.append(int(n_mat))
            
            if var0==1:
                codebars, angbins = get_material_data(material_ = imat, 
                                                      ang_maxx = ang_maxx, 
                                                      step = step,
                                                      hkl_ref=n_mat, 
                                                      classhkl=classhkl_mat) 
                np.savez_compressed(save_directory+'//grain_classhkl_angbin_'+imat+'.npz',\
                                    classhkl_mat, angbins)
    except:
        write_to_console("Class HKL library data not found, please run it first")
        return None
             
    ## make comprehensive list of dictionary
    normal_hkl_multimat = []
    index_hkl_mutimat = []
    for ino, imat in enumerate(material_):
        normal_hkl_ = np.zeros((1,3))
        for j in hkl_all_class[ino].keys():
            normal_hkl_ = np.vstack((normal_hkl_, hkl_all_class[ino][j]["family"]))
        normal_hkl = np.delete(normal_hkl_, 0, axis =0)
        normal_hkl_multimat.append(normal_hkl)
    
        if ino > 0:
            ind_offset = index_hkl_mutimat[ino-1][-1] + 1
            index_hkl = [ind_offset+j for j,k in enumerate(hkl_all_class[ino].keys()) for i in range(len(hkl_all_class[ino][k]["family"]))]
        else:
            index_hkl = [j for j,k in enumerate(hkl_all_class[ino].keys()) for i in range(len(hkl_all_class[ino][k]["family"]))]
        
        index_hkl_mutimat.append(index_hkl)

    write_to_console("Generating "+type_+" and saving them")
    
    _inputs_queue = Queue()
    _outputs_queue = Queue()
    _worker_process = {}
    for i in range(ncpu):
        _worker_process[i]= Process(target=worker_generation_multimat, args=(_inputs_queue, 
                                                                          _outputs_queue, 
                                                                          i+1),)
    for i in range(ncpu):
        _worker_process[i].start()     
        
    time.sleep(0.1)  
    
    ## list of combination of training dataset
    ## to be seen if this improves the prediction quality
    ## increases time significantly to generate the data 
    nb_grains_list = []
    for ino, imat in enumerate(material_):
        nb_grains_list.append(list(range(nb_grains[ino]+1)))
    list_permute = list(itertools.product(*nb_grains_list))
    list_permute.pop(0)
    max_progress = len(list_permute)*(grains_nb_simulate)

    # generate a database upto n grain LP
    values = []
    for i in range(len(list_permute)):
        
        for j in range(grains_nb_simulate):
            
            if data_realism:
                ## three types of data augmentation to mimic reality ?
                if j < grains_nb_simulate*0.25:
                    noisy_data = False
                    remove_peaks = False
                elif (j >= grains_nb_simulate*0.25) and (j < grains_nb_simulate*0.5):
                    noisy_data = True
                    remove_peaks = False
                elif (j >= grains_nb_simulate*0.5) and (j < grains_nb_simulate*0.75):
                    noisy_data = False
                    remove_peaks = True
                elif (j >= grains_nb_simulate*0.75):
                    noisy_data = True
                    remove_peaks = True
            else:
                noisy_data = False
                remove_peaks = False

            seednumber = np.random.randint(1e6)
            values.append([ list_permute[i], 
                            material_,
                            emin, emax, detectorparameters,
                            pixelsize,True,
                            ang_maxx, step,
                            classhkl,
                            noisy_data, 
                            remove_peaks,
                            seednumber,
                            hkl_all_class,
                            lattice_material,
                            None,
                            normal_hkl_multimat,
                            index_hkl_mutimat,
                            dim1, dim2,
                            removeharmonics,
                            0, i, j, save_directory_,
                            modelp,
                            max_millerindex,
                            general_diff_rules,
                            crystal,])                
    chunks = chunker_list(values, ncpu)
    chunks_mp = list(chunks)

    meta = {'t1':time.time(),
                'flag':1}
    
    for ijk in range(int(ncpu)):
        _inputs_queue.put((chunks_mp[ijk], ncpu, meta))

    max_progress = len(values)
    update_progress = 0
    pbar = tqdm(total=max_progress)
    while True:
        time.sleep(2)
        if not _outputs_queue.empty():
            r_message = _outputs_queue.get()
            print(r_message)
            update_progress = update_progress + r_message
            pbar.update(update_progress)
            
        count = 0
        for i in range(ncpu):
            if not _worker_process[i].is_alive():
                _worker_process[i].join()
                count += 1
            else:
                time.sleep(0.1)
        if count == ncpu:
            pbar.close()
            return
        
def getMMpatterns_(nb, material_=None, emin=5, emax=23, detectorparameters=None, pixelsize=None, 
                 sortintensity = False, ang_maxx = 45, step = 0.5, classhkl = None, noisy_data=False, 
                 remove_peaks=False, seed = None,hkl_all=None, lattice_material=None, family_hkl=None,
                 normal_hkl=None, index_hkl=None, dim1=2048, dim2=2048, removeharmonics=1, flag = 0,
                 img_i=None, img_j=None, save_directory_=None, modelp=None,
                 max_millerindex=0, general_diff_cond=False, crystal=None,
                 ):
    if np.all(np.array(nb)==0):
        print("Skipping a simulation file: "+save_directory_+'//grain_'+\
                            str(img_i)+"_"+str(img_j)+'.npz'+"; Due to zero grains")
        return

    ori_mat, ori_mat1 = [], []
    s_tth, s_chi, s_miller_ind, _, _, _ = simulatemultimatpatterns(nb, seed=seed, 
                                                                   key_material=material_, 
                                                                emin=emin, emax=emax,
                                                                 detectorparameters=detectorparameters,
                                                                 pixelsize=pixelsize,
                                                                 sortintensity = sortintensity, 
                                                                 dim1=dim1, dim2=dim2, 
                                                                 removeharmonics=removeharmonics,
                                                                 flag=flag, mode=modelp,
                                                                 )
    if noisy_data:
        ## apply random gaussian type noise to the data (tth and chi)
        ## So adding noise to the angular distances
        ## Instead of adding noise to all HKL's ... Add to few selected HKLs
        ## Adding noise to randomly 30% of the HKLs
        ## Realistic way of introducting strains is through Pixels and not 2theta
        noisy_pixel = 0.15
        indices_noise = np.random.choice(len(s_tth), int(len(s_tth)*0.3), replace=False)
        noise_ = np.random.normal(0,noisy_pixel,len(indices_noise))
        s_tth[indices_noise] = s_tth[indices_noise] + noise_
        noise_ = np.random.normal(0,noisy_pixel,len(indices_noise)) 
        s_chi[indices_noise] = s_chi[indices_noise] + noise_
        
    if remove_peaks:
        len_mi = np.array([iq for iq in range(len(s_miller_ind))])
        len_mi = len_mi[int(0.6*len(s_miller_ind)):]
        indices_remove = np.random.choice(len_mi, int(len(len_mi)*0.3), replace=False)
        ## delete randomly selected less intense peaks
        ## to simulate real peak detection, where some peaks may not be
        ## well detected
        ## Include maybe Intensity approach: Delete peaks based on their SF and position in detector
        if len(indices_remove) !=0:
            s_tth = np.delete(s_tth, indices_remove)
            s_chi = np.delete(s_chi, indices_remove)
            s_miller_ind = np.delete(s_miller_ind, indices_remove, axis=0)
    
    add_random_noise = False
    if add_random_noise:
        nb_random_spots = 500
        ## add random two theta and chi spots to the dataset
        # to simulate noise in the patterns (in reality these 
        # are additional peaks from partial Laue patterns).
        
        # we can do 2D sampling of 2theta and chi from one Cor file;
        # but apparantly the chi is uniform from (-40 to +40)
        # while 2theta has a distribution
        pass
        
        
    # replace all hkl class with relevant hkls
    ## skip HKLS that dont follow the general diffraction rules
    location = []
    skip_hkl = []
    delete_spots = []
    for j, i in enumerate(s_miller_ind):
        if np.all(i[:3] == 0):
            skip_hkl.append(j)
            continue
        
        new_hkl = _round_indices(i[:3])
        
        mat_index = int(i[3])
            
        if general_diff_cond:
            cond_proceed = crystal[mat_index].hkl_allowed(i[:3], returnequivalents=False)
        else:
            cond_proceed = True
        
        if not cond_proceed:
            delete_spots.append(j)
            continue
        
        if np.any(np.abs(new_hkl)>max_millerindex[mat_index]):
            skip_hkl.append(j)
            continue
        
        temp_ = np.all(new_hkl == normal_hkl[mat_index], axis=1)
        if len(np.where(temp_)[0]) == 1:
            ind_ = np.where(temp_)[0][0]
            location.append(index_hkl[mat_index][ind_])
        elif len(np.where(temp_)[0]) == 0:
            # print("Entering -100 for "+ str(i) + "\n")
            skip_hkl.append(j)
        elif len(np.where(temp_)[0]) > 1:
            ## first check if they both are same class or not
            class_output = []
            for ij in range(len(np.where(temp_)[0])):
                indc = index_hkl[mat_index][np.where(temp_)[0][ij]]
                class_output.append(indc)
            if len(set(class_output)) <= 1:
                location.append(class_output[0])
            else:
                skip_hkl.append(j)
                print(i)
                print(np.where(temp_)[0])
                for ij in range(len(np.where(temp_)[0])):
                    indc = index_hkl[mat_index][np.where(temp_)[0][ij]]
                    print(classhkl[mat_index][indc])
                print("Entering -500: Skipping HKL as something is not proper with equivalent HKL module")

    allspots_the_chi = np.transpose(np.array([s_tth/2., s_chi]))
    tabledistancerandom = np.transpose(GT.calculdist_from_thetachi(allspots_the_chi, allspots_the_chi))
    
    codebars = []
    angbins = np.arange(0,ang_maxx+step,step)
    for i in range(len(tabledistancerandom)):
        if i in skip_hkl or i in delete_spots: ## not saving skipped HKL
            continue
        angles = tabledistancerandom[i]
        spots_delete = [i]
        for del_spts in delete_spots:
            spots_delete.append(del_spts)
        angles = np.delete(angles, spots_delete)
        fingerprint = np.histogram(angles, bins=angbins)[0]
        # fingerprint = histogram1d(angles, range=[min(angbins),max(angbins)], bins=len(angbins)-1)
        ## same normalization as before
        max_codebars = np.max(fingerprint)
        fingerprint = fingerprint/ max_codebars
        codebars.append(fingerprint)
    
    suffix_ = ""
    if flag == 0:
        if len(codebars) != 0:
            
            mat_prefix = ""
            for no, i in enumerate(nb):
                if i != 0:
                    mat_prefix = mat_prefix + material_[no]

            np.savez_compressed(save_directory_+'//'+mat_prefix+'_grain_'+str(img_i)+"_"+\
                                str(img_j)+suffix_+'.npz', codebars, location, ori_mat, ori_mat1, flag,\
                                s_tth, s_chi, s_miller_ind)
        else:
            print("Skipping a simulation file: "+save_directory_+'//grain_'+\
                                str(img_i)+"_"+str(img_j)+suffix_+'.npz'+"; Due to no data conforming user settings")

def simulatemultimatpatterns(nbUBs, seed=123, key_material=None, 
                             emin=5, emax=23, detectorparameters=None, pixelsize=None,
                             sortintensity = False, dim1=2048, dim2=2048, removeharmonics=1, flag = 0,
                             mode="random"):
    l_tth, l_chi, l_miller_ind, l_posx, l_posy, l_E, l_intensity = [],[],[],[],[],[],[]
    detectordiameter = pixelsize * dim1
    if flag == 0:
        if mode == "random":
            for no, i in enumerate(nbUBs):
                if i != 0:                    
                    for igr in range(i):
                        phi1 = rand1() * 360.
                        phi = 180. * acos(2 * rand1() - 1) / np.pi
                        phi2 = rand1() * 360.
                        UBmatrix = Euler2OrientationMatrix((phi1, phi, phi2))
                        
                        grain = CP.Prepare_Grain(key_material[no], UBmatrix)
                        s_tth, s_chi, s_miller_ind, \
                            s_posx, s_posy, s_E= LT.SimulateLaue_full_np(grain, emin, emax,
                                                                        detectorparameters,
                                                                        pixelsize=pixelsize,
                                                                        dim=(dim1, dim2),
                                                                        detectordiameter=detectordiameter,
                                                                        removeharmonics=removeharmonics)
                        s_miller_ind = np.c_[s_miller_ind, np.ones(len(s_miller_ind))*no]
                        s_intensity = 1./s_E
                        l_tth.append(s_tth)
                        l_chi.append(s_chi)
                        l_miller_ind.append(s_miller_ind)
                        l_posx.append(s_posx)
                        l_posy.append(s_posy)
                        l_E.append(s_E)
                        l_intensity.append(s_intensity)
    
    #flat_list = [item for sublist in l for item in sublist]
    s_tth = np.array([item for sublist in l_tth for item in sublist])
    s_chi = np.array([item for sublist in l_chi for item in sublist])
    s_miller_ind = np.array([item for sublist in l_miller_ind for item in sublist])
    s_posx = np.array([item for sublist in l_posx for item in sublist])
    s_posy = np.array([item for sublist in l_posy for item in sublist])
    s_E = np.array([item for sublist in l_E for item in sublist])
    s_intensity=np.array([item for sublist in l_intensity for item in sublist])
    
    if sortintensity:
        indsort = np.argsort(s_intensity)[::-1]
        s_tth=np.take(s_tth, indsort)
        s_chi=np.take(s_chi, indsort)
        s_miller_ind=np.take(s_miller_ind, indsort, axis=0)
        s_posx=np.take(s_posx, indsort)
        s_posy=np.take(s_posy, indsort)
        s_E=np.take(s_E, indsort)
        s_intensity=np.take(s_intensity, indsort)
    return s_tth, s_chi, s_miller_ind, s_posx, s_posy, s_intensity


def worker_generation_multimat(inputs_queue, outputs_queue, proc_id):
    while True:
        time.sleep(0.01)
        if not inputs_queue.empty():
            message = inputs_queue.get()
            num1, _, meta = message
            flag1 = meta['flag']
            for ijk in range(len(num1)):
                nb, material_, emin, emax, detectorparameters, pixelsize, \
                 sortintensity, ang_maxx, step, classhkl, noisy_data, \
                 remove_peaks, seed,hkl_all, lattice_material, family_hkl,\
                 normal_hkl, index_hkl, dim1, dim2, removeharmonics, flag,\
                 img_i, img_j, save_directory_, modelp, max_millerindex,\
                         general_diff_cond, crystal = num1[ijk]


                getMMpatterns_(nb, material_, emin, emax, detectorparameters, pixelsize, \
                                         sortintensity, ang_maxx, step, classhkl, noisy_data, \
                                         remove_peaks, seed,hkl_all, lattice_material, family_hkl,\
                                         normal_hkl, index_hkl, dim1, dim2, removeharmonics, flag,\
                                         img_i, img_j, save_directory_, modelp, \
                                         max_millerindex, general_diff_cond, crystal)
                    
                if ijk%10 == 0 and ijk!=0:
                    outputs_queue.put(11)
            if flag1 == 1:
                break

def get_multimaterial_detail(material_=None, SG_mat=None, symm_mat=None):
    """
        Returns material details
    """
    rules, symmetry, lattice_material, crystal, SG = [],[],[],[],[]
    for ino, imat in enumerate(material_):
        a, b, c, alpha, beta, gamma = dictLT.dict_Materials[imat][1]
        rules.append(dictLT.dict_Materials[imat][-1])
        
        symm_ = symm_mat[ino]
        
        if symm_ =="cubic":
            symmetry.append(Symmetry.cubic)
            lattice_material.append(Lattice.cubic(a))
            if SG_mat[ino] == None:
                SG.append(230)
                SG_mat[ino] = 230
            else:
                SG.append(SG_mat[ino])
            crystal.append(SGLattice(int(SG_mat[ino]), a))
        elif symm_ =="monoclinic":
            symmetry.append(Symmetry.monoclinic)
            lattice_material.append(Lattice.monoclinic(a, b, c, beta))
            if SG_mat[ino] == None:
                SG.append(10)
                SG_mat[ino] = 10
            else:
                SG.append(SG_mat[ino])
            crystal.append(SGLattice(int(SG_mat[ino]),a, b, c, beta))
        elif symm_ == "hexagonal":
            symmetry.append(Symmetry.hexagonal)
            lattice_material.append(Lattice.hexagonal(a, c))
            if SG_mat[ino] == None:
                SG.append(191)
                SG_mat[ino] = 191
            else:
                SG.append(SG_mat[ino])
            crystal.append(SGLattice(int(SG_mat[ino]),a, c))
        elif symm_ == "orthorhombic":
            symmetry.append(Symmetry.orthorhombic)
            lattice_material.append(Lattice.orthorhombic(a, b, c))
            if SG_mat[ino] == None:
                SG.append(47)
                SG_mat[ino] = 47
            else:
                SG.append(SG_mat[ino])
            crystal.append(SGLattice(int(SG_mat[ino]),a, b, c))
        elif symm_ == "tetragonal":
            symmetry.append(Symmetry.tetragonal)
            lattice_material.append(Lattice.tetragonal(a, c))
            if SG_mat[ino] == None:
                SG.append(123)
                SG_mat[ino] = 123
            else:
                SG.append(SG_mat[ino])
            crystal.append(SGLattice(int(SG_mat[ino]),a, c))
        elif symm_ == "trigonal":
            symmetry.append(Symmetry.trigonal)
            lattice_material.append(Lattice.rhombohedral(a, alpha))
            if SG_mat[ino] == None:
                SG.append(162)
                SG_mat[ino] = 162
            else:
                SG.append(SG_mat[ino])
            crystal.append(SGLattice(int(SG_mat[ino]),a, alpha))
        elif symm_ == "triclinic":
            symmetry.append(Symmetry.triclinic)
            lattice_material.append(Lattice.triclinic(a, b, c, alpha, beta, gamma))
            if SG_mat[ino] == None:
                SG.append(2)
                SG_mat[ino] = 2
            else:
                SG.append(SG_mat[ino])
            crystal.append(SGLattice(int(SG_mat[ino]),a, b, c, alpha, beta, gamma))
        
    return rules, symmetry, lattice_material, crystal, SG

def rmv_freq_class_MM(freq_rmv = [0], elements=["all"],
                       save_directory="", material_=None, 
                       write_to_console=None,
                       progress=None, qapp=None, list_hkl_keep=None):
    
    classhkl_mm = []
    ind_mat_mm = []
    for ino, imat in enumerate(material_):
        classhkl0 = np.load(save_directory+"//grain_classhkl_angbin_"+imat+".npz")["arr_0"]
        angbins = np.load(save_directory+"//grain_classhkl_angbin_"+imat+".npz")["arr_1"]
        if write_to_console != None:
            write_to_console(imat +" material index length: " + str(len(classhkl0)))
            
        if ino == 0:
            ind_mat = np.array([ij for ij in range(len(classhkl0))])
        else:
            pre_ind = ind_mat_mm[ino-1][-1] + 1
            ind_mat = np.array([pre_ind+ij for ij in range(len(classhkl0))])
        classhkl_mm.append(classhkl0)
        ind_mat_mm.append(ind_mat)
            
    for ino, classhkl0 in enumerate(classhkl_mm):
        if ino == 0:
            classhkl = classhkl0
        else:
            classhkl = np.vstack((classhkl, classhkl0))
    
    loc = np.array([ij for ij in range(len(classhkl))])
    trainy_ = array_generatorV2(save_directory+"//training_data", 0, progress, qapp)
    
    ## split trainy_ for multi materials index
    trainy_mat_MM = [[] for _ in range(len(material_))]
    for ino, imat in enumerate(material_):
        for ijnode in trainy_:
            if ijnode in ind_mat_mm[ino]:
                trainy_mat_MM[ino].append(ijnode)

    if write_to_console != None:            
        write_to_console("Class ID and frequency; check for data imbalance and select "+\
                         "appropriate LOSS function for training the model")
    
    ## lets extract the least common occuring classes to simplify the training dataset
    for ino, imat in enumerate(material_):
        if elements[ino] == "all":
            most_common0 = collections.Counter(np.array(trainy_mat_MM[ino])).most_common()
        else:
            most_common0 = collections.Counter(np.array(trainy_mat_MM[ino])).most_common()[:elements[ino]]
        print("Most common classhkl elements in "+imat+" are:")
        print(most_common0)
        ##N simpler approach
        ## include hkl family search also #TODO
        if list_hkl_keep != None:
            all_common = np.array(collections.Counter(np.array(trainy_mat_MM[ino])).most_common())
            list_keep = list_hkl_keep[ino]
            for ih, ik, il in list_keep:
                conda= classhkl[:,0]==ih
                condb= classhkl[:,1]==ik
                condc= classhkl[:,2]==il
                indd123 = np.where(conda*condb*condc == True)[0]
                if len(indd123) == 0:
                    print("Demanded HKL not found;",ih,ik,il)
                    continue
                for ihkl in indd123:
                    if ihkl in all_common[:,0]:
                        inind = np.where(all_common[:,0]==ihkl)[0]
                        print("Demanded hkl",ih,ik,il," has the following occurance",all_common[inind,1])
                        to_add = (all_common[inind,0][0], all_common[inind,1][0])
                        if to_add not in most_common0:
                            most_common0.append(to_add)     
        if ino == 0:
            most_common = most_common0
        else:
            most_common = most_common + most_common0

    class_present = [most_common[i][0] for i in range(len(most_common))]
    rmv_indices = []

    for i in loc:
        if i not in class_present:
            rmv_indices.append(i)
        elif i in class_present:
            ind_ = np.where(np.array(class_present)==i)[0]
            ij = most_common[ind_[0]]
            for ino, imat in enumerate(material_):
                if (ij[0] in ind_mat_mm[ino]) and (ij[1] <= freq_rmv[ino]):
                    rmv_indices.append(int(ij[0]))
        else:
            if write_to_console != None:
                write_to_console("Something Fishy in Remove Freq Class module")
    
    
    for ino, imat in enumerate(material_):
        for i in rmv_indices:
            if i in ind_mat_mm[ino]:
                indd = np.where(ind_mat_mm[ino] == i)[0]
                ind_mat_mm[ino] = np.delete(ind_mat_mm[ino], indd, axis=0)
                
    loc_new = np.delete(loc, rmv_indices)

    occurances = [most_common[i][1] for i in range(len(most_common)) if int(most_common[i][0]) in loc_new]
    occurances = np.array(occurances)
    
    class_weight = {}
    class_weight_temp = {}
    count = 0
    for i in loc_new:
        for ij in most_common:
            if int(ij[0]) == i:
                class_weight[count] = int(np.max(occurances)/ij[1])
                class_weight_temp[int(ij[0])] = int(np.max(occurances)/ij[1])
                count += 1
    
    for occ in range(len(most_common)):
        if int(most_common[occ][0]) in loc_new:
            if write_to_console != None:
                suffix_string = ""
                for ino, imat in enumerate(material_):
                    if int(most_common[occ][0]) in ind_mat_mm[ino]:
                        suffix_string = "; material: "+imat
                        
                if int(most_common[occ][0]) == -100:
                    write_to_console("Unclassified HKL (-100); occurance : "+str(most_common[occ][1])+\
                                        "; NN_weights : 0.0 "+suffix_string)
                else:
                    write_to_console("HKL : " +str(classhkl[int(most_common[occ][0])])+"; occurance : "+\
                                     str(most_common[occ][1])+\
                                    "; NN_weights : "+ \
                                    str(class_weight_temp[int(most_common[occ][0])])+suffix_string)
    if write_to_console != None:
        write_to_console(str(len(rmv_indices))+ " classes removed from the classHKL object [removal frequency: "+\
                            str(freq_rmv)+"] (before:"+str(len(classhkl))+", now:"+str(len(classhkl)-len(rmv_indices))+")")
    print(str(len(rmv_indices))+ " classes removed from the classHKL object [removal frequency: "+\
                        str(freq_rmv)+"] (before:"+str(len(classhkl))+", now:"+str(len(classhkl)-len(rmv_indices))+")")
    
    if len(rmv_indices) == len(classhkl):
        if write_to_console != None:
            write_to_console("Error; no classes left in the classhkl array; please reduce frequency to Keep some classes")
        else:
            print("Error; no classes left in the classhkl array; please reduce frequency to Keep some classes")
        return None
    classhkl = np.delete(classhkl, rmv_indices, axis=0)
    ## save the altered classHKL object
    np.savez_compressed(save_directory+'//MOD_grain_classhkl_angbin.npz', classhkl, angbins, loc_new, 
                            rmv_indices, freq_rmv, ind_mat_mm)

    with open(save_directory + "//class_weights.pickle", "wb") as output_file:
        cPickle.dump([class_weight], output_file)
    if write_to_console != None:
        write_to_console("Saved class weights data")
        
        
def predict_preprocessMultiMatProcess(files, cnt, 
                                     rotation_matrix,strain_matrix,strain_matrixs,
                                    col,colx,coly,match_rate,spots_len,iR_pix,fR_pix,best_match,mat_global,
                                    check,detectorparameters,pixelsize,angbins,
                                    classhkl, hkl_all_class0, emin, emax,
                                    material_, symmetry, lim_x, lim_y,
                                    strain_calculation, ind_mat,
                                    model_direc=None, tolerance =None,
                                   matricies=None, ccd_label=None,
                                   filename_bkg=None,intensity_threshold=None,
                                   boxsize=None,bkg_treatment=None,
                                   filenameDirec=None, experimental_prefix=None,
                                   blacklist_file =None, text_file=None, 
                                   files_treated=None,try_previous1=False,
                                   wb=None, temp_key=None, cor_file_directory=None, mode_spotCycle1=None,
                                   softmax_threshold_global123=None,mr_threshold_global123=None,
                                   cap_matchrate123=None,tolerance_strain123=None,\
                                   NumberMaxofFits123=None,fit_peaks_gaussian_global123=None,
                                   FitPixelDev_global123=None,coeff123=None, coeff_overlap=None,
                                   material0_limit=None, use_previous_UBmatrix_name=None,
                                   material_phase_always_present=None, crystal=None, strain_free_parameters=None):
    
    if files in files_treated:
        return strain_matrix, strain_matrixs, rotation_matrix, col, colx, coly, \
            match_rate, mat_global, cnt, files_treated,spots_len,iR_pix,fR_pix, check, best_match, None
    
    call_global()
    print("Predicting for "+files)    
    
    if files.split(".")[-1] != "cor":
        CCDLabel=ccd_label
        seednumber = "Experimental "+CCDLabel+" file"    
        
        try:
            out_name = blacklist_file
        except:
            out_name = None  
            
        if bkg_treatment == None:
            bkg_treatment = "A-B"
            
        try:
            ### Max space = space betzeen pixles
            peak_XY = RMCCD.PeakSearch(
                                        files,
                                        stackimageindex = -1,
                                        CCDLabel=CCDLabel,
                                        NumberMaxofFits=NumberMaxofFits123,
                                        PixelNearRadius=10,
                                        removeedge=2,
                                        IntensityThreshold=intensity_threshold,
                                        local_maxima_search_method=0,
                                        boxsize=boxsize,
                                        position_definition=1,
                                        verbose=0,
                                        fit_peaks_gaussian=fit_peaks_gaussian_global123,
                                        xtol=0.001,                
                                        FitPixelDev=FitPixelDev_global123,
                                        return_histo=0,
                                        # Saturation_value=1e10,  # to be merged in CCDLabel
                                        # Saturation_value_flatpeak=1e10,
                                        MinIntensity=0,
                                        PeakSizeRange=(0.65,200),
                                        write_execution_time=1,
                                        Data_for_localMaxima = "auto_background",
                                        formulaexpression=bkg_treatment,
                                        Remove_BlackListedPeaks_fromfile=out_name,
                                        reject_negative_baseline=True,
                                        Fit_with_Data_for_localMaxima=False,
                                        maxPixelDistanceRejection=15.0,
                                        )
            peak_XY = peak_XY[0]#[:,:2] ##[2] Integer peak lists
        except:
            print("Error in Peak detection for "+ files)
            for intmat in range(matricies):
                rotation_matrix[intmat][0][cnt,:,:] = np.zeros((3,3))
                strain_matrix[intmat][0][cnt,:,:] = np.zeros((3,3))
                strain_matrixs[intmat][0][cnt,:,:] = np.zeros((3,3))
                col[intmat][0][cnt,:] = 0,0,0
                colx[intmat][0][cnt,:] = 0,0,0
                coly[intmat][0][cnt,:] = 0,0,0
                match_rate[intmat][0][cnt] = 0
                mat_global[intmat][0][cnt] = 0
                spots_len[intmat][0][cnt] = 0
                iR_pix[intmat][0][cnt] = 0
                fR_pix[intmat][0][cnt] = 0
                check[cnt,intmat] = 0
            # files_treated.append(files)
            return strain_matrix, strain_matrixs, rotation_matrix, col, colx, coly, \
                match_rate, mat_global, cnt, files_treated,spots_len,iR_pix,fR_pix, check, best_match, None
        
        try:
            s_ix = np.argsort(peak_XY[:, 2])[::-1]
            peak_XY = peak_XY[s_ix]
        except:
            print("Error in Peak detection for "+ files)
            for intmat in range(matricies):
                rotation_matrix[intmat][0][cnt,:,:] = np.zeros((3,3))
                strain_matrix[intmat][0][cnt,:,:] = np.zeros((3,3))
                strain_matrixs[intmat][0][cnt,:,:] = np.zeros((3,3))
                col[intmat][0][cnt,:] = 0,0,0
                colx[intmat][0][cnt,:] = 0,0,0
                coly[intmat][0][cnt,:] = 0,0,0
                match_rate[intmat][0][cnt] = 0
                mat_global[intmat][0][cnt] = 0
                spots_len[intmat][0][cnt] = 0
                iR_pix[intmat][0][cnt] = 0
                fR_pix[intmat][0][cnt] = 0
                check[cnt,intmat] = 0
            # files_treated.append(files)
            return strain_matrix, strain_matrixs, rotation_matrix, col, colx, coly, \
                match_rate, mat_global, cnt, files_treated,spots_len,iR_pix,fR_pix, check, best_match
        
        
        framedim = dictLT.dict_CCD[CCDLabel][0]
        twicetheta, chi = Lgeo.calc_uflab(peak_XY[:,0], peak_XY[:,1], detectorparameters,
                                            returnAngles=1,
                                            pixelsize=pixelsize,
                                            kf_direction='Z>0')
        data_theta, data_chi = twicetheta/2., chi
        
        framedim = dictLT.dict_CCD[CCDLabel][0]
        dict_dp={}
        dict_dp['kf_direction']='Z>0'
        dict_dp['detectorparameters']=detectorparameters
        dict_dp['detectordistance']=detectorparameters[0]
        dict_dp['detectordiameter']=pixelsize*framedim[0]
        dict_dp['pixelsize']=pixelsize
        dict_dp['dim']=framedim
        dict_dp['peakX']=peak_XY[:,0]
        dict_dp['peakY']=peak_XY[:,1]
        dict_dp['intensity']=peak_XY[:,2]
        
        CCDcalib = {"CCDLabel":CCDLabel,
                    "dd":detectorparameters[0], 
                    "xcen":detectorparameters[1], 
                    "ycen":detectorparameters[2], 
                    "xbet":detectorparameters[3], 
                    "xgam":detectorparameters[4],
                    "pixelsize": pixelsize}
        
        path = os.path.normpath(files)
        IOLT.writefile_cor(cor_file_directory+"//"+path.split(os.sep)[-1].split(".")[0], twicetheta, 
                           chi, peak_XY[:,0], peak_XY[:,1], peak_XY[:,2],
                           param=CCDcalib, sortedexit=0)
        
    elif files.split(".")[-1] == "cor":
        # print("Entering Cor file read section")
        seednumber = "Experimental COR file"
        allres = IOLT.readfile_cor(files, True)
        data_theta, data_chi, peakx, peaky, intensity = allres[1:6]
        CCDcalib = allres[-1]
        detectorparameters = allres[-2]
        # print('detectorparameters from file are: '+ str(detectorparameters))
        pixelsize = CCDcalib['pixelsize']
        CCDLabel = CCDcalib['CCDLabel']
        framedim = dictLT.dict_CCD[CCDLabel][0]
        dict_dp={}
        dict_dp['kf_direction']='Z>0'
        dict_dp['detectorparameters']=detectorparameters
        dict_dp['detectordistance']=detectorparameters[0]
        dict_dp['detectordiameter']=pixelsize*framedim[0]
        dict_dp['pixelsize']=pixelsize
        dict_dp['dim']=framedim
        dict_dp['peakX']=peakx
        dict_dp['peakY']=peaky
        dict_dp['intensity']=intensity

    sorted_data = np.transpose(np.array([data_theta, data_chi]))
    tabledistancerandom = np.transpose(GT.calculdist_from_thetachi(sorted_data, sorted_data))

    codebars_all = []
    
    if len(data_theta) == 0:
        print("No peaks Found for : " + files)
        for intmat in range(matricies):
            rotation_matrix[intmat][0][cnt,:,:] = np.zeros((3,3))
            strain_matrix[intmat][0][cnt,:,:] = np.zeros((3,3))
            strain_matrixs[intmat][0][cnt,:,:] = np.zeros((3,3))
            col[intmat][0][cnt,:] = 0,0,0
            colx[intmat][0][cnt,:] = 0,0,0
            coly[intmat][0][cnt,:] = 0,0,0
            match_rate[intmat][0][cnt] = 0
            mat_global[intmat][0][cnt] = 0
            spots_len[intmat][0][cnt] = 0
            iR_pix[intmat][0][cnt] = 0
            fR_pix[intmat][0][cnt] = 0
            check[cnt,intmat] = 0
        return strain_matrix, strain_matrixs, rotation_matrix, col, colx, coly, \
                match_rate, mat_global, cnt, files_treated,spots_len,iR_pix,fR_pix, check, best_match, None
    
    # print("Entering GOOD section")
    spots_in_center = np.arange(0,len(data_theta))
    spots_in_center = spots_in_center[:nb_spots_consider]
    
    for i in spots_in_center:
        spotangles = tabledistancerandom[i]
        spotangles = np.delete(spotangles, i)# removing the self distance
        codebars = np.histogram(spotangles, bins=angbins)[0]
        # codebars = histogram1d(spotangles, range=[min(angbins),max(angbins)], bins=len(angbins)-1)
        ## normalize the same way as training data
        max_codebars = np.max(codebars)
        codebars = codebars/ max_codebars
        codebars_all.append(codebars)
        
    codebars = np.array(codebars_all)
    ## Do prediction of all spots at once
    try:
        prediction = predict(codebars, wb, temp_key)
    except:
        if len(material_) > 1:
            prefix_mat = material_[0]
            for ino, imat in enumerate(material_):
                if ino == 0:
                    continue
                prefix_mat = prefix_mat + "_" + imat
        else:
            prefix_mat = material_
        json_file = open(model_direc+"//model_"+prefix_mat+".json", 'r')
        load_weights = model_direc + "//model_"+prefix_mat+".h5"
        # # load json and create model
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(load_weights)
        prediction = model.predict(codebars)
    
    max_pred = np.max(prediction, axis = 1)
    class_predicted = np.argmax(prediction, axis = 1)
    predicted_hkl123 = classhkl[class_predicted]
    predicted_hkl123 = predicted_hkl123.astype(int)
        
    s_tth = data_theta * 2.
    s_chi = data_chi
    
    # print("Computing UB")
    rotation_matrix1, mr_highest, mat_highest, \
        strain_crystal, strain_sample, iR_pix1, \
                    fR_pix1, spots_len1,\
                    best_match1, check12 = predict_ub_MM(seednumber, spots_in_center, classhkl, 
                                                  hkl_all_class0, 
                                                  files,
                                                  s_tth1=s_tth,s_chi1=s_chi,
                                                  predicted_hkl1=predicted_hkl123,
                                                  class_predicted1=class_predicted,
                                                  max_pred1=max_pred,
                                                  emin=emin,emax=emax,
                                                  material_=material_, 
                                                  lim_y=lim_y, lim_x=lim_x, 
                                                  cnt=cnt,
                                                  dict_dp=dict_dp,
                                                  rotation_matrix=rotation_matrix,
                                                  mat_global=mat_global,
                                                  strain_calculation=strain_calculation,
                                                  ind_mat=ind_mat, 
                                                  tolerance=tolerance, 
                                                  matricies=matricies,
                                                  tabledistancerandom=tabledistancerandom,
                                                  text_file = text_file,
                                                  try_previous1=try_previous1,
                                                  mode_spotCycle=mode_spotCycle1,
                                                  softmax_threshold_global123 = softmax_threshold_global123,
                                                  mr_threshold_global123=mr_threshold_global123,
                                                  cap_matchrate123=cap_matchrate123,
                                                  tolerance_strain123=tolerance_strain123,
                                                  coeff123=coeff123,
                                                  coeff_overlap=coeff_overlap,
                                                  material0_limit=material0_limit, 
                                                  model_direc=model_direc,
                                                  use_previous_UBmatrix_name=use_previous_UBmatrix_name,
                                                  material_phase_always_present=material_phase_always_present,
                                                  match_rate=match_rate,
                                                  check=check[cnt,:],
                                                  crystal=crystal,
                                                  angbins=angbins,
                                                  wb=wb, temp_key=temp_key,
                                                  strain_free_parameters=strain_free_parameters)
    for intmat in range(matricies):
        if len(rotation_matrix1[intmat]) == 0:
            col[intmat][0][cnt,:] = 0,0,0
            colx[intmat][0][cnt,:] = 0,0,0
            coly[intmat][0][cnt,:] = 0,0,0
        else:
            mat_global[intmat][0][cnt] = mat_highest[intmat][0]

            final_symm = symmetry[int(mat_highest[intmat][0])-1]
            final_crystal = crystal[int(mat_highest[intmat][0])-1]

            symm_operator = final_crystal._hklsym
            strain_matrix[intmat][0][cnt,:,:] = strain_crystal[intmat][0]
            strain_matrixs[intmat][0][cnt,:,:] = strain_sample[intmat][0]
            rotation_matrix[intmat][0][cnt,:,:] = rotation_matrix1[intmat][0]
            col_temp = get_ipf_colour(rotation_matrix1[intmat][0], np.array([0., 0., 1.]), final_symm, symm_operator)
            col[intmat][0][cnt,:] = col_temp
            col_tempx = get_ipf_colour(rotation_matrix1[intmat][0], np.array([1., 0., 0.]), final_symm, symm_operator)
            colx[intmat][0][cnt,:] = col_tempx
            col_tempy = get_ipf_colour(rotation_matrix1[intmat][0], np.array([0., 1., 0.]), final_symm, symm_operator)
            coly[intmat][0][cnt,:] = col_tempy
            match_rate[intmat][0][cnt] = mr_highest[intmat][0]
            spots_len[intmat][0][cnt] = spots_len1[intmat][0]
            iR_pix[intmat][0][cnt] = iR_pix1[intmat][0]
            fR_pix[intmat][0][cnt] = fR_pix1[intmat][0]
            best_match[intmat][0][cnt] = best_match1[intmat][0]
            check[cnt,intmat] = check12[intmat]
    files_treated.append(files)
    return strain_matrix, strain_matrixs, rotation_matrix, col, colx, coly, match_rate, \
            mat_global, cnt, files_treated, spots_len, iR_pix, fR_pix, check, best_match, predicted_hkl123

def new_MP_multimat_function(argu):
    files, cnt, rotation_matrix, strain_matrix, strain_matrixs,\
            col,colx,coly,match_rate,spots_len,iR_pix,fR_pix,best_match,mat_global,\
            check,detectorparameters,pixelsize,angbins,\
            classhkl, hkl_all_class0, emin, emax,\
            material_, symmetry, lim_x, lim_y,\
            strain_calculation, ind_mat,\
            model_direc, tolerance,\
            matricies, ccd_label,\
            filename_bkg,intensity_threshold,\
            boxsize,bkg_treatment,\
            filenameDirec, experimental_prefix,\
            blacklist_file, text_file, \
            files_treated,try_previous1,\
            wb, temp_key, cor_file_directory, mode_spotCycle1,\
            softmax_threshold_global123,mr_threshold_global123,\
            cap_matchrate123, tolerance_strain123,\
            NumberMaxofFits123,fit_peaks_gaussian_global123,\
            FitPixelDev_global123,coeff123,coeff_overlap,\
            material0_limit, use_previous_UBmatrix_name1,\
            material_phase_always_present1, crystal, strain_free_parameters = argu
                        
    strain_matrix12, strain_matrixs12, \
        rotation_matrix12, col12, \
            colx12, coly12,\
    match_rate12, mat_global12, cnt12,\
        files_treated12, spots_len12, \
            iR_pix12, fR_pix12, check12,\
                best_match12, _ = predict_preprocessMultiMatProcess(files, cnt, 
                                               rotation_matrix,strain_matrix,strain_matrixs,
                                               col,colx,coly,match_rate,spots_len,iR_pix,fR_pix,best_match,
                                               mat_global,
                                               check,detectorparameters,pixelsize,angbins,
                                               classhkl, hkl_all_class0, emin, emax,
                                               material_, symmetry,lim_x, lim_y,
                                               strain_calculation, ind_mat,
                                               model_direc, tolerance,
                                               matricies, ccd_label,
                                               filename_bkg,intensity_threshold,
                                               boxsize,bkg_treatment,
                                               filenameDirec, experimental_prefix,
                                               blacklist_file, text_file, 
                                               files_treated,try_previous1,
                                               wb, temp_key, cor_file_directory, mode_spotCycle1,
                                               softmax_threshold_global123,mr_threshold_global123,
                                               cap_matchrate123, tolerance_strain123,
                                               NumberMaxofFits123,
                                               fit_peaks_gaussian_global123,
                                               FitPixelDev_global123, coeff123,coeff_overlap,
                                               material0_limit,
                                               use_previous_UBmatrix_name1,
                                               material_phase_always_present1,
                                               crystal, strain_free_parameters)
    meta = {}
    return strain_matrix12, strain_matrixs12, rotation_matrix12, col12, \
                 colx12, coly12, match_rate12, mat_global12, cnt12, meta, \
                 files_treated12, spots_len12, iR_pix12, fR_pix12, best_match12, check12
                 
def new_MP_multimat_functionGUI(inputs_queue, outputs_queue, proc_id, run_flag):
    print(f'Initializing worker {proc_id}')
    while True:
        if not run_flag.value:
            break
        time.sleep(0.01)
        if not inputs_queue.empty(): 
            message = inputs_queue.get()
            if message == 'STOP':
                print(f'[{proc_id}] stopping')
                break

            num1, num2, meta = message
            files_worked = []
            while True:
                if len(num1) == len(files_worked) or len(num1) == 0:
                    print("process finished")
                    break
                for ijk in range(len(num1)):
                    if ijk in files_worked:
                        continue                       
                    if not run_flag.value:
                        num1, files_worked = [], []
                        print(f'[{proc_id}] stopping')
                        break
                    
                    files, cnt, rotation_matrix, strain_matrix, strain_matrixs,\
                    col,colx,coly,match_rate,spots_len,iR_pix,fR_pix,best_match,mat_global,\
                    check,detectorparameters,pixelsize,angbins,\
                    classhkl, hkl_all_class0, emin, emax,\
                    material_, symmetry, lim_x, lim_y,\
                    strain_calculation, ind_mat,\
                    model_direc, tolerance,\
                    matricies, ccd_label,\
                    filename_bkg,intensity_threshold,\
                    boxsize,bkg_treatment,\
                    filenameDirec, experimental_prefix,\
                    blacklist_file, text_file, \
                    files_treated,try_previous1,\
                    wb, temp_key, cor_file_directory, mode_spotCycle1,\
                    softmax_threshold_global123,mr_threshold_global123,\
                    cap_matchrate123, tolerance_strain123,\
                    NumberMaxofFits123,fit_peaks_gaussian_global123,\
                    FitPixelDev_global123,coeff123,coeff_overlap,\
                    material0_limit, use_previous_UBmatrix_name1,\
                    material_phase_always_present1, crystal, strain_free_parameters = num1[ijk]
                    
                    if np.all(check[cnt,:]) == 1:
                        continue
                    
                    strain_matrix12, strain_matrixs12, \
                    rotation_matrix12, col12, colx12, coly12,\
                    match_rate12, mat_global12, cnt12,\
                    files_treated12, spots_len12, \
                    iR_pix12, fR_pix12, check12,\
                    best_match12, _ = predict_preprocessMultiMatProcess(files, cnt, 
                                                               rotation_matrix,strain_matrix,strain_matrixs,
                                                               col,colx,coly,match_rate,spots_len,iR_pix,fR_pix,best_match,
                                                               mat_global,
                                                               check,detectorparameters,pixelsize,angbins,
                                                               classhkl, hkl_all_class0, emin, emax,
                                                               material_, symmetry,lim_x, lim_y,
                                                               strain_calculation, ind_mat,
                                                               model_direc, tolerance,
                                                               matricies, ccd_label,
                                                               filename_bkg,intensity_threshold,
                                                               boxsize,bkg_treatment,
                                                               filenameDirec, experimental_prefix,
                                                               blacklist_file, text_file, 
                                                               files_treated,try_previous1,
                                                               wb, temp_key, cor_file_directory, mode_spotCycle1,
                                                               softmax_threshold_global123,mr_threshold_global123,
                                                               cap_matchrate123, tolerance_strain123,
                                                               NumberMaxofFits123,
                                                               fit_peaks_gaussian_global123,
                                                               FitPixelDev_global123, coeff123,coeff_overlap,
                                                               material0_limit,
                                                               use_previous_UBmatrix_name1,
                                                               material_phase_always_present1,
                                                               crystal, strain_free_parameters)
                    files_worked.append(ijk)
                    meta['proc_id'] = proc_id
                    r_message = (strain_matrix12, strain_matrixs12, rotation_matrix12, col12, \
                                  colx12, coly12, match_rate12, mat_global12, cnt12, meta, \
                                  files_treated12, spots_len12, iR_pix12, fR_pix12, best_match12, check12)
                    outputs_queue.put(r_message)
    print("broke the worker while loop")  

def predict_ub_MM(seednumber, spots_in_center, classhkl, hkl_all_class0, 
                     filename, 
                     s_tth1,s_chi1,predicted_hkl1,class_predicted1,max_pred1,
                     emin, emax, material_, lim_y, lim_x, cnt,
                     dict_dp,rotation_matrix,mat_global,strain_calculation,
                     ind_mat,
                     tolerance=None, matricies=None, tabledistancerandom=None,
                     text_file=None, try_previous1=False, mode_spotCycle=None,
                     softmax_threshold_global123=None,mr_threshold_global123=None,
                     cap_matchrate123=None, tolerance_strain123=None, coeff123=None,
                     coeff_overlap=None, material0_limit=None, model_direc=None,
                     use_previous_UBmatrix_name=None, material_phase_always_present=None, match_rate=None,
                     check = None, crystal=None, angbins=None, wb=None, temp_key=None,
                     strain_free_parameters=None):
    
    input_params = {"tolerance": tolerance,
                    "tolerancestrain": tolerance_strain123, ## For strain calculations
                    "emin": emin,
                    "emax": emax,
                    "mat":0}
    call_global()
    
    strain_matrix = [[] for i in range(matricies)]
    strain_matrixs = [[] for i in range(matricies)]
    best_matrix = [[] for i in range(matricies)]
    mr_highest = [[] for i in range(matricies)]
    ir_pixels = [[] for i in range(matricies)]
    fr_pixels = [[] for i in range(matricies)]
    spots_len = [[] for i in range(matricies)]
    mat_highest = [[] for i in range(matricies)]
    best_match = [[] for i in range(matricies)]
    spots1 = []
    spots1_global = [[] for i in range(matricies)]
    
    dist = tabledistancerandom        
    ## one time calculations
    B0mat, Gstar_metric0mat, tab_distance_classhkl_data0mat = [], [], []
    for ino, imat in enumerate(material_):
        lattice_params00 = dictLT.dict_Materials[imat][1]
        B00 = CP.calc_B_RR(lattice_params00)
        Gstar_metric00 = CP.Gstar_from_directlatticeparams(lattice_params00[0],lattice_params00[1],\
                                                         lattice_params00[2],lattice_params00[3],\
                                                             lattice_params00[4],lattice_params00[5])
        tab_distance_classhkl_data00 = get_material_dataP(Gstar_metric00, predicted_hkl1[:nb_spots_consider,:])
        
        B0mat.append(B00)
        Gstar_metric0mat.append(Gstar_metric00)
        tab_distance_classhkl_data0mat.append(tab_distance_classhkl_data00)
        
    spots = []
    first_match = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.zeros((3,3))]
    max_mr = 0
    mat = 0
    iR = 0
    fR = 0
    strain_crystal = np.zeros((3,3))
    strain_sample = np.zeros((3,3))
    material0_count = [0 for _ in range(len(material_))]
    calcul_done = False
    objective_function1 = None
    
    for igrain in range(matricies):
        try_previous = try_previous1
        max_mr, min_mr = 0, 0
        iR, fR= 0, 0
        case = "None"
        
        if use_om_user:
            use_previous_UBmatrix_name = False
            try_previous = False

            temp_qsd = np.loadtxt(path_user_OM, delimiter=",")
            # TODO include mat in the text file and read here
            temp_qsd = temp_qsd.reshape((len(temp_qsd),3,3))
            rotationmatrix_indexed = temp_qsd[igrain,:,:]
            
            mat = 1

            if mat != 0:
                Keymaterial_ = material_[int(mat)-1]
                case = material_[int(mat)-1]
                Bkey = B0mat[int(mat)-1]
                input_params["mat"] = int(mat)
                input_params["Bmat"] = Bkey
            else:
                Keymaterial_ = None
                Bkey = None
                input_params["mat"] = 0
                input_params["Bmat"] = None
                continue
                
            spots_prev, theo_spots_prev = remove_spotsMM(s_tth1, s_chi1, 
                                                        rotationmatrix_indexed, 
                                                        Keymaterial_, 
                                                        input_params, 
                                                        dict_dp['detectorparameters'], 
                                                        dict_dp)

            newmatchrate = 100*len(spots_prev)/theo_spots_prev
            
            ## Filter indexation by matching rate
            if newmatchrate < cap_matchrate123:
                strain_crystal, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
                first_match = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                    0, 0, 0, 0, 0, np.zeros((3,3))]
                spots = []
                max_mr, min_mr = 0, 0
            else:
                if strain_calculation:
                    strain_crystal, strain_sample, \
                        iR, fR, rot_mat_UB = calculate_strains_fromUBMM(s_tth1, s_chi1, 
                                                                        rotationmatrix_indexed, 
                                                                        Keymaterial_, 
                                                                        input_params, 
                                                                        dict_dp['detectorparameters'], 
                                                                        dict_dp, 
                                                                        spots1,
                                                                        Bkey,
                                                                        strain_free_parameters)
                else:
                    strain_crystal, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
                    rot_mat_UB = np.copy(rotationmatrix_indexed)
                spots = spots_prev
                expected = theo_spots_prev
                max_mr, min_mr = 100*(len(spots)/expected), 100*(len(spots)/expected)
                first_match = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                0, len(spots), expected, max_mr, 0, rot_mat_UB]
                    
            try_previous = False
            calcul_done = True

        
        elif use_previous_UBmatrix_name:
            try:
                try_previous = False
                ### try already indexed UB matricies
                # xy = np.load('xy.npz')
                # xy.zip.fp.close()
                # xy.close()
                with np.load(model_direc+"//rotation_matrix_indexed_1.npz") as load_objectind:
                    # load_objectind = np.load(model_direc+"//rotation_matrix_indexed.npz")
                    rotationmatrix_indexed = load_objectind["arr_0"]
                    mat_global_indexed = load_objectind["arr_1"]
                    match_rate_indexed = load_objectind["arr_2"]
                    avg_match_rate_indexed = load_objectind["arr_3"]
                calcul_done = False
                for ind_mat_UBmat in range(len(rotationmatrix_indexed[igrain][0])):
                    if calcul_done:
                        continue
                    
                    if np.all(rotationmatrix_indexed[igrain][0][ind_mat_UBmat,:,:]) == 0:
                        continue

                    if match_rate_indexed[igrain][0][ind_mat_UBmat] < 0.8*avg_match_rate_indexed[igrain]:
                        continue
                    
                    mat = mat_global_indexed[igrain][0][ind_mat_UBmat]
                    if mat != 0:
                        Keymaterial_ = material_[int(mat)-1]
                        case = material_[int(mat)-1]
                        Bkey = B0mat[int(mat)-1]
                        input_params["mat"] = int(mat)
                        input_params["Bmat"] = Bkey
                    else:
                        Keymaterial_ = None
                        Bkey = None
                        input_params["mat"] = 0
                        input_params["Bmat"] = None
                        continue
                    
                    spots_prev, theo_spots_prev = remove_spotsMM(s_tth1, s_chi1, 
                                                             rotationmatrix_indexed[igrain][0][ind_mat_UBmat,:,:], 
                                                             Keymaterial_, 
                                                             input_params, 
                                                             dict_dp['detectorparameters'], 
                                                             dict_dp)

                    newmatchrate = 100*len(spots_prev)/theo_spots_prev
                    condition_prev = newmatchrate < 0.8*(match_rate_indexed[igrain][0][ind_mat_UBmat])
                    current_spots = [len(list(set(spots_prev) & set(spots1_global[igr]))) > coeff_overlap*len(spots1_global[igr]) for igr in range(len(spots1_global))]
                    
                    if condition_prev or (newmatchrate <= cap_matchrate123) or np.any(current_spots):# or overlap:
                        try_previous = try_previous1
                    else:
                        try_previous = False
                        calcul_done = True
                        if strain_calculation:
                            strain_crystal, strain_sample, \
                                iR, fR, rot_mat_UB = calculate_strains_fromUBMM(s_tth1, s_chi1, 
                                                                                rotationmatrix_indexed[igrain][0][ind_mat_UBmat,:,:], 
                                                                                Keymaterial_, 
                                                                                input_params, 
                                                                                dict_dp['detectorparameters'], 
                                                                                dict_dp, 
                                                                                spots1,
                                                                                Bkey,
                                                                                strain_free_parameters)
                        else:
                            strain_crystal, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
                            rot_mat_UB = np.copy(rotationmatrix_indexed[igrain][0][ind_mat_UBmat,:,:])
                        spots = spots_prev
                        expected = theo_spots_prev
                        max_mr, min_mr = 100*(len(spots)/expected), 100*(len(spots)/expected)
                        first_match = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                        0, len(spots), expected, max_mr, 0, rot_mat_UB]
                        break
            except:
                try_previous = False
                calcul_done = False
                
        if try_previous and (cnt % lim_y == 0) and cnt != 0:
            if np.all(rotation_matrix[igrain][0][cnt-lim_y,:,:]) == 0:
                try_previous = False
            else:
                mat = mat_global[igrain][0][cnt-lim_y]
                if mat != 0:
                    Keymaterial_ = material_[int(mat)-1]
                    case = material_[int(mat)-1]
                    Bkey = B0mat[int(mat)-1]
                    input_params["mat"] = int(mat)
                    input_params["Bmat"] = Bkey
                else:
                    Keymaterial_ = None
                    Bkey = None
                    input_params["mat"] = 0
                    input_params["Bmat"] = None
                    continue
                
                spots_lr, theo_spots_lr = remove_spotsMM(s_tth1, s_chi1, 
                                                         rotation_matrix[igrain][0][cnt-lim_y,:,:], 
                                                         Keymaterial_, 
                                                         input_params, 
                                                         dict_dp['detectorparameters'], 
                                                         dict_dp)

                # last_row = len(spots_lr) <= coeff123*theo_spots_lr
                newmatchrate = 100*(len(spots_lr)/theo_spots_lr)
                condition_prev = newmatchrate < 0.9*(match_rate[igrain][0][cnt-lim_y])
                last_row = condition_prev
                if last_row or condition_prev: ## new spots less than 8 count, not good match SKIP
                    try_previous = False
                else:
                    try_previous = True
                    current_spots = [len(list(set(spots_lr) & set(spots1_global[igr]))) > coeff_overlap*len(spots1_global[igr]) for igr in range(len(spots1_global))]
                    if np.any(current_spots):
                        try_previous = False
                        continue
                    
                    if strain_calculation:
                        strain_crystal, strain_sample, \
                            iR, fR, rot_mat_UB = calculate_strains_fromUBMM(s_tth1, s_chi1, 
                                                                            rotation_matrix[igrain][0][cnt-lim_y,:,:], 
                                                                            Keymaterial_, 
                                                                            input_params, 
                                                                            dict_dp['detectorparameters'], 
                                                                            dict_dp, 
                                                                            spots1,
                                                                            Bkey,
                                                                            strain_free_parameters)
                    else:
                        strain_crystal, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
                        rot_mat_UB = np.copy(rotation_matrix[igrain][0][cnt-lim_y,:,:])
                    spots = spots_lr
                    expected = theo_spots_lr
                    max_mr, min_mr = 100*(len(spots_lr)/theo_spots_lr), 100*(len(spots_lr)/theo_spots_lr)
                    first_match = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                    0, len(spots), expected, max_mr, 0, rot_mat_UB]   
                
        elif try_previous and (cnt % lim_y != 0):
            last_row = True
            left_row = True
            condition_prev = True
            condition_prev1 = True
            if np.all(rotation_matrix[igrain][0][cnt-1,:,:]) == 0:
                left_row = True
            else:
                mat = mat_global[igrain][0][cnt-1]
                if mat != 0:
                    Keymaterial_ = material_[int(mat)-1]
                    case = material_[int(mat)-1]
                    Bkey = B0mat[int(mat)-1]
                    input_params["mat"] = int(mat)
                    input_params["Bmat"] = Bkey
                else:
                    Keymaterial_ = None
                    Bkey = None
                    input_params["mat"] = 0
                    input_params["Bmat"] = None
                    continue
                ## new row start when % == 0
                ## use left index pixels matrix values
                spots_left, theo_spots_left = remove_spotsMM(s_tth1, s_chi1, 
                                                         rotation_matrix[igrain][0][cnt-1,:,:], 
                                                         Keymaterial_, 
                                                         input_params, 
                                                         dict_dp['detectorparameters'], 
                                                         dict_dp)

                # left_row = len(spots_left) <= coeff123*theo_spots_left 
                newmatchrate = 100*(len(spots_left)/theo_spots_left)
                condition_prev = newmatchrate < 0.9*(match_rate[igrain][0][cnt-1])
                left_row = condition_prev
            if cnt >= lim_y:
                if np.all(rotation_matrix[igrain][0][cnt-lim_y,:,:]) == 0:
                    last_row = True   
                else:
                    mat = mat_global[igrain][0][cnt-lim_y]
                    if mat != 0:
                        Keymaterial_ = material_[int(mat)-1]
                        case = material_[int(mat)-1]
                        Bkey = B0mat[int(mat)-1]
                        input_params["mat"] = int(mat)
                        input_params["Bmat"] = Bkey
                    else:
                        Keymaterial_ = None
                        Bkey = None
                        input_params["mat"] = 0
                        input_params["Bmat"] = None
                        continue
                    ## use bottom index pixels matrix values
                    spots_lr, theo_spots_lr = remove_spotsMM(s_tth1, s_chi1, 
                                                             rotation_matrix[igrain][0][cnt-lim_y,:,:], 
                                                             Keymaterial_, 
                                                             input_params, 
                                                             dict_dp['detectorparameters'], 
                                                             dict_dp)
                    
                    # last_row = len(spots_lr) <= coeff123*theo_spots_lr 
                    newmatchrate1 = 100*(len(spots_lr)/theo_spots_lr)
                    condition_prev1 = newmatchrate1 < 0.9*(match_rate[igrain][0][cnt-lim_y])
                    last_row = condition_prev1
            if (left_row and last_row): 
                try_previous = False
            elif condition_prev and condition_prev1:
                try_previous = False
            elif not left_row and not last_row:
                try_previous = True
                
                if len(spots_lr) > len(spots_left):
                    current_spots = [len(list(set(spots_lr) & set(spots1_global[igr]))) > coeff_overlap*len(spots1_global[igr]) for igr in range(len(spots1_global))]
                    if np.any(current_spots):
                        try_previous = False
                        continue
                    
                    mat = mat_global[igrain][0][cnt-lim_y]
                    if mat != 0:
                        Keymaterial_ = material_[int(mat)-1]
                        case = material_[int(mat)-1]
                        Bkey = B0mat[int(mat)-1]
                        input_params["mat"] = int(mat)
                        input_params["Bmat"] = Bkey
                    else:
                        Keymaterial_ = None
                        Bkey = None
                        input_params["mat"] = 0
                        input_params["Bmat"] = None
                        continue
                    
                    if strain_calculation:
                        strain_crystal, strain_sample, \
                            iR, fR, rot_mat_UB = calculate_strains_fromUBMM(s_tth1, s_chi1, 
                                                                            rotation_matrix[igrain][0][cnt-lim_y,:,:], 
                                                                            Keymaterial_, 
                                                                            input_params, 
                                                                            dict_dp['detectorparameters'], 
                                                                            dict_dp, 
                                                                            spots1,
                                                                            Bkey,
                                                                            strain_free_parameters)
                    else:
                        strain_crystal, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
                        rot_mat_UB = np.copy(rotation_matrix[igrain][0][cnt-lim_y,:,:])
                    spots = spots_lr
                    expected = theo_spots_lr
                    max_mr, min_mr = 100*(len(spots_lr)/theo_spots_lr), 100*(len(spots_lr)/theo_spots_lr)
                    first_match = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                0, len(spots), expected, max_mr, 0, rot_mat_UB]
                else:
                    current_spots = [len(list(set(spots_left) & set(spots1_global[igr]))) > coeff_overlap*len(spots1_global[igr]) for igr in range(len(spots1_global))]
                    if np.any(current_spots):
                        try_previous = False
                        continue

                    mat = mat_global[igrain][0][cnt-1]
                    if mat != 0:
                        Keymaterial_ = material_[int(mat)-1]
                        case = material_[int(mat)-1]
                        Bkey = B0mat[int(mat)-1]
                        input_params["mat"] = int(mat)
                        input_params["Bmat"] = Bkey
                    else:
                        Keymaterial_ = None
                        Bkey = None
                        input_params["mat"] = 0
                        input_params["Bmat"] = None
                        continue
                    
                    if strain_calculation:
                        strain_crystal, strain_sample, \
                            iR, fR, rot_mat_UB = calculate_strains_fromUBMM(s_tth1, s_chi1, 
                                                                            rotation_matrix[igrain][0][cnt-1,:,:], 
                                                                            Keymaterial_, 
                                                                            input_params, 
                                                                            dict_dp['detectorparameters'], 
                                                                            dict_dp, 
                                                                            spots1,
                                                                            Bkey,
                                                                            strain_free_parameters)
                    else:
                        strain_crystal, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
                        rot_mat_UB = np.copy(rotation_matrix[igrain][0][cnt-1,:,:])
                    spots = spots_left
                    expected = theo_spots_left
                    max_mr, min_mr = 100*(len(spots_left)/theo_spots_left), 100*(len(spots_left)/theo_spots_left)
                    first_match = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                0, len(spots), expected, max_mr, 0, rot_mat_UB]    
            
            elif not left_row and last_row:
                try_previous = True
                current_spots = [len(list(set(spots_left) & set(spots1_global[igr]))) > coeff_overlap*len(spots1_global[igr]) for igr in range(len(spots1_global))]
                if np.any(current_spots):
                    try_previous = False
                    continue
                
                mat = mat_global[igrain][0][cnt-1]
                if mat != 0:
                    Keymaterial_ = material_[int(mat)-1]
                    case = material_[int(mat)-1]
                    Bkey = B0mat[int(mat)-1]
                    input_params["mat"] = int(mat)
                    input_params["Bmat"] = Bkey
                else:
                    Keymaterial_ = None
                    Bkey = None
                    input_params["mat"] = 0
                    input_params["Bmat"] = None
                    continue
                
                if strain_calculation:
                    strain_crystal, strain_sample, \
                        iR, fR, rot_mat_UB = calculate_strains_fromUBMM(s_tth1, s_chi1, 
                                                                        rotation_matrix[igrain][0][cnt-1,:,:], 
                                                                        Keymaterial_, 
                                                                        input_params, 
                                                                        dict_dp['detectorparameters'], 
                                                                        dict_dp, 
                                                                        spots1,
                                                                        Bkey,
                                                                        strain_free_parameters)
                else:
                    strain_crystal, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
                    rot_mat_UB = np.copy(rotation_matrix[igrain][0][cnt-1,:,:])
                spots = spots_left
                expected = theo_spots_left
                max_mr, min_mr = 100*(len(spots_left)/theo_spots_left), 100*(len(spots_left)/theo_spots_left)
                first_match = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                0, len(spots), expected, max_mr, 0, rot_mat_UB]  
                    
            elif left_row and not last_row:
                try_previous = True
                current_spots = [len(list(set(spots_lr) & set(spots1_global[igr]))) > coeff_overlap*len(spots1_global[igr]) for igr in range(len(spots1_global))]
                if np.any(current_spots):
                    try_previous = False
                    continue
                
                mat = mat_global[igrain][0][cnt-lim_y]
                if mat != 0:
                    Keymaterial_ = material_[int(mat)-1]
                    case = material_[int(mat)-1]
                    Bkey = B0mat[int(mat)-1]
                    input_params["mat"] = int(mat)
                    input_params["Bmat"] = Bkey
                else:
                    Keymaterial_ = None
                    Bkey = None
                    input_params["mat"] = 0
                    input_params["Bmat"] = None
                    continue

                if strain_calculation:
                    strain_crystal, strain_sample, \
                        iR, fR, rot_mat_UB = calculate_strains_fromUBMM(s_tth1, s_chi1, 
                                                                        rotation_matrix[igrain][0][cnt-lim_y,:,:], 
                                                                        Keymaterial_, 
                                                                        input_params, 
                                                                        dict_dp['detectorparameters'], 
                                                                        dict_dp, 
                                                                        spots1,
                                                                        Bkey,
                                                                        strain_free_parameters)
                else:
                    strain_crystal, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
                    rot_mat_UB = np.copy(rotation_matrix[igrain][0][cnt-lim_y,:,:])
                    
                spots = spots_lr
                expected = theo_spots_lr    
                max_mr, min_mr = 100*(len(spots_lr)/theo_spots_lr), 100*(len(spots_lr)/theo_spots_lr)
                first_match = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                0, len(spots), expected, max_mr, 0, rot_mat_UB]  
        else:
            try_previous = False
        
        if not try_previous and not calcul_done:
            ### old version
            if mode_spotCycle == "slow":
                # print("Slow mode of analysis")
                first_match, max_mr, min_mr, spots, \
                        case, mat, strain_crystal, \
                            strain_sample, iR, fR  = get_orient_matMM(s_tth1, 
                                                                      s_chi1,
                                                                      material_, 
                                                                      classhkl,
                                                                      class_predicted1, 
                                                                      predicted_hkl1,
                                                                      input_params, 
                                                                      hkl_all_class0,
                                                                      max_pred1, 
                                                                      dict_dp, 
                                                                      spots1, 
                                                                      dist, 
                                                                      Gstar_metric0mat, 
                                                                      B0mat,
                                                                      softmax_threshold=softmax_threshold_global123,
                                                                      mr_threshold=mr_threshold_global123,
                                                                      tab_distance_classhkl_data0=tab_distance_classhkl_data0mat,
                                                                      spots1_global = spots1_global,
                                                                      coeff_overlap = coeff_overlap,
                                                                      ind_mat=ind_mat,
                                                                      strain_calculation=strain_calculation,
                                                                      cap_matchrate123=cap_matchrate123,
                                                                      material0_count=material0_count,
                                                                      material0_limit=material0_limit,
                                                                      igrain=igrain,
                                                                      material_phase_always_present=material_phase_always_present,
                                                                      strain_free_parameters=strain_free_parameters)
            elif mode_spotCycle == "graphmode":
                # print("Fast mode of analysis")
                first_match, max_mr, min_mr, spots, \
                case, mat, strain_crystal, \
                    strain_sample, iR, fR,\
                    objective_function1  = get_orient_mat_graphv1_MM(s_tth1, 
                                                                     s_chi1,
                                                                     material_, 
                                                                     classhkl,
                                                                     class_predicted1, 
                                                                     predicted_hkl1,
                                                                     input_params, 
                                                                     hkl_all_class0,
                                                                     max_pred1, 
                                                                     dict_dp, 
                                                                     spots1, 
                                                                     dist, 
                                                                     Gstar_metric0mat, 
                                                                     B0mat,
                                                                     softmax_threshold=softmax_threshold_global123,
                                                                     mr_threshold=mr_threshold_global123,
                                                                     tab_distance_classhkl_data0=tab_distance_classhkl_data0mat,
                                                                     spots1_global = spots1_global,
                                                                     coeff_overlap = coeff_overlap,
                                                                     ind_mat=ind_mat, 
                                                                     strain_calculation=strain_calculation,
                                                                     cap_matchrate123=cap_matchrate123,
                                                                     material0_count=material0_count,
                                                                     material0_limit=material0_limit,
                                                                     igrain=igrain,
                                                                     material_phase_always_present=material_phase_always_present,
                                                                     objective_function= objective_function1,
                                                                     strain_free_parameters=strain_free_parameters)
            elif mode_spotCycle == "update_reupdate":
                # print("Fast mode of analysis")
                first_match, max_mr, min_mr, spots, \
                    case, mat, strain_crystal, \
                    strain_sample, iR, fR, objective_function1,\
                    s_tth1, s_chi1, class_predicted1, \
                    predicted_hkl1, max_pred1, dist = get_orient_mat_repredict_MM(
                                                                        s_tth1, 
                                                                        s_chi1,
                                                                        material_, 
                                                                        classhkl,
                                                                        class_predicted1, 
                                                                        predicted_hkl1,
                                                                        input_params, 
                                                                        hkl_all_class0, 
                                                                        max_pred1, 
                                                                        dict_dp, 
                                                                        spots1, 
                                                                        dist, 
                                                                        Gstar_metric0mat, 
                                                                        B0mat, 
                                                                        softmax_threshold=softmax_threshold_global123,
                                                                        mr_threshold=mr_threshold_global123,
                                                                        tab_distance_classhkl_data0=tab_distance_classhkl_data0mat,
                                                                        spots1_global = spots1_global,
                                                                        coeff_overlap = coeff_overlap,
                                                                        ind_mat=ind_mat,
                                                                        strain_calculation=strain_calculation,
                                                                        cap_matchrate123=cap_matchrate123,
                                                                        material0_count=material0_count,
                                                                        material0_limit=material0_limit,
                                                                        igrain=igrain,
                                                                        material_phase_always_present=material_phase_always_present,
                                                                        objective_function= objective_function1,
                                                                        crystal=crystal,
                                                                        angbins=angbins,
                                                                        wb=wb, temp_key=temp_key,
                                                                        strain_free_parameters=strain_free_parameters,
                                                                        model_direc=model_direc)
            else:
                print("selected mode of treating spots is not ready")
                
        for ispot in spots:
            spots1.append(ispot)
            spots1_global[igrain].append(ispot)

        ## make copy of best rotation matrix
        best_match[igrain].append(np.copy(first_match))
        best_matrix[igrain].append(np.copy(first_match[14]))
        mr_highest[igrain].append(np.copy(max_mr))
        mat_highest[igrain].append(np.copy(mat))
        ir_pixels[igrain].append(np.copy(iR))
        fr_pixels[igrain].append(np.copy(fR))
        spots_len[igrain].append(np.copy(len(spots)))
        strain_matrix[igrain].append(np.copy(strain_crystal))
        strain_matrixs[igrain].append(np.copy(strain_sample))
        
        if np.all(first_match[14] != 0):
            check[igrain] = 1
        
        material0_count[int(mat)-1] = material0_count[int(mat)-1]+1

    return best_matrix, mr_highest, mat_highest, strain_matrix, strain_matrixs, ir_pixels, fr_pixels, spots_len, best_match, check


def get_orient_mat_repredict_MM(s_tth, s_chi, material0_, classhkl, class_predicted, predicted_hkl,
                       input_params, hkl_all_class0, max_pred, dict_dp, spots, 
                       dist, Gstar_metric0, B0, softmax_threshold=0.85, mr_threshold=0.85, 
                       tab_distance_classhkl_data0=None, spots1_global=None,
                       coeff_overlap = None, ind_mat=None, strain_calculation=None, cap_matchrate123=None,
                       material0_count=None, material0_limit=None,
                       igrain=None, material_phase_always_present=None, objective_function=None, crystal=None,
                       angbins=None, wb=None, temp_key=None, strain_free_parameters=None, model_direc=None):    
    if objective_function == None:
        call_global()
        
        init_mr = 0
        init_mat = 0
        init_material = "None"
        init_case = "None"
        init_B = None
        final_match_rate = 0
        match_rate_mma = []
        final_rmv_ind = []
        
        list_of_sets = []
        for ii in range(0, min(nb_spots_consider, len(dist))):
            if max_pred[ii] < softmax_threshold:
                continue 
            
            a1 = np.round(dist[ii],3)

            for i in range(0, min(nb_spots_consider, len(dist))):
                if ii==i:
                    continue
                if (ii,i) in list_of_sets or (i,ii) in list_of_sets:
                    continue
                if max_pred[i] < softmax_threshold:
                    continue
                
                belong_same_mat = False
                for ino, imat in enumerate(material0_):
                    if ino == 0:
                        if class_predicted[ii] < ind_mat[ino] and class_predicted[i] < ind_mat[ino] :
                            tab_distance_classhkl_data = tab_distance_classhkl_data0[ino] 
                            tolerance_new = input_params["tolerance"][ino]
                            hkl1 = hkl_all_class0[ino][str(predicted_hkl[ii])]
                            hkl1_list = np.array(hkl1)
                            hkl2 = hkl_all_class0[ino][str(predicted_hkl[i])]
                            hkl2_list = np.array(hkl2)
                            Gstar_metric = Gstar_metric0[ino] 
                            belong_same_mat = True
                    else:
                        if (ind_mat[ino-1] <= class_predicted[ii] < ind_mat[ino]) and \
                                            (ind_mat[ino-1] <= class_predicted[i] < ind_mat[ino]):
                            tab_distance_classhkl_data = tab_distance_classhkl_data0[ino]
                            tolerance_new = input_params["tolerance"][ino]
                            hkl1 = hkl_all_class0[ino][str(predicted_hkl[ii])]
                            hkl1_list = np.array(hkl1)
                            hkl2 = hkl_all_class0[ino][str(predicted_hkl[i])]
                            hkl2_list = np.array(hkl2)
                            Gstar_metric = Gstar_metric0[ino]
                            belong_same_mat = True
                if not belong_same_mat:
                    continue
                tab_angulardist_temp = CP.AngleBetweenNormals(hkl1_list, hkl2_list, Gstar_metric)
                np.putmask(tab_angulardist_temp, np.abs(tab_angulardist_temp) < 0.001, 400)
                list_ = np.where(np.abs(tab_angulardist_temp-a1[i]) < tolerance_new)
                if len(list_[0]) != 0:
                    list_of_sets.append((ii,i))

        ## build a direct connection graph object
        graph_obj = nx.DiGraph(list_of_sets)
        connected_nodes_length = []
        connected_nodes = [[] for i in range(len(graph_obj))]
        for i,line in enumerate(nx.generate_adjlist(graph_obj)):
            connected_nodes_length.append(len(line.split(" ")))
            connected_nodes[i].append([int(jj) for jj in line.split(" ")])
        
        ## sort by maximum node occurance
        connected_nodes_length = np.array(connected_nodes_length)
        connected_nodes_length_sort_ind = np.argsort(connected_nodes_length)[::-1]
  
        mat = 0
        case = "None"
        tried_spots = []
        
        if len(graph_obj) == 0:
            print("no object in graph network")
            
        objective_function = []
        for toplist in range(len(graph_obj)):
            # ## continue if less than 3 connections are found for a graph
            # if connected_nodes_length[connected_nodes_length_sort_ind[toplist]] < 2:
            #     continue
            for j in connected_nodes[connected_nodes_length_sort_ind[toplist]][0]:
                init_mr = 0
                final_match_rate = 0
                final_rmv_ind = []
                all_stats = []
                for i in connected_nodes[connected_nodes_length_sort_ind[toplist]][0]:
                    if j == i:
                        continue
                    
                    if j in tried_spots and i in tried_spots:
                        continue
                    
                    #TODO replace by simpler step
                    mat = 0
                    case = "None"
                    input_params["mat"] = mat
                    input_params["Bmat"] = None
                    for ino, imat in enumerate(material0_):
                        if ino == 0:
                            if class_predicted[i] < ind_mat[ino] and class_predicted[j] < ind_mat[ino] :
                                tab_distance_classhkl_data = tab_distance_classhkl_data0[ino] 
                                hkl_all_class = hkl_all_class0[ino] 
                                material_ = imat
                                B = B0[ino] 
                                Gstar_metric = Gstar_metric0[ino] 
                                case = imat
                                mat = ino + 1
                                if material0_count[ino] >= material0_limit[ino]:
                                    mat = 0
                                    case="None"
                                input_params["mat"] = mat
                                input_params["Bmat"] = B
                        else:
                            if (ind_mat[ino-1] <= class_predicted[i] < ind_mat[ino]) and \
                                                (ind_mat[ino-1] <= class_predicted[j] < ind_mat[ino]):
                                tab_distance_classhkl_data = tab_distance_classhkl_data0[ino]
                                hkl_all_class = hkl_all_class0[ino] 
                                material_ = imat
                                B = B0[ino]
                                Gstar_metric = Gstar_metric0[ino]
                                case = imat  
                                mat = ino + 1
                                if material0_count[ino] >= material0_limit[ino]:
                                    mat = 0
                                    case="None"
                                input_params["mat"] = mat
                                input_params["Bmat"] = B
                    
                    if mat == 0:
                        continue   
                    
                    tth_chi_spot1 = np.array([s_tth[i], s_chi[i]])
                    tth_chi_spot2 = np.array([s_tth[j], s_chi[j]])         
        
                    hkl1 = hkl_all_class[str(predicted_hkl[i])]
                    hkl1_list = np.array(hkl1)
                    hkl2 = hkl_all_class[str(predicted_hkl[j])]
                    hkl2_list = np.array(hkl2)
                    
                    actual_mat, flagAM, \
                    spot1_hkl, spot2_hkl = propose_UB_matrixMM(hkl1_list, hkl2_list, 
                                                            Gstar_metric, input_params, 
                                                            dist[i,j],
                                                            tth_chi_spot1, tth_chi_spot2, 
                                                            B, method=0)

                    if flagAM:
                        continue
                    
                    for iind in range(len(actual_mat)):
                        rot_mat123 = actual_mat[iind]
                        rmv_ind, theospots = remove_spotsMM(s_tth, s_chi, rot_mat123, 
                                                            material_, input_params, 
                                                            dict_dp['detectorparameters'], dict_dp)
                        match_rate = np.round(100 * len(rmv_ind)/theospots, 3)                        
                        match_rate_mma.append(match_rate)

                        if match_rate > init_mr:
                            final_rmv_ind = rmv_ind                    
                            init_mat = np.copy(mat)
                            input_params["mat"] = init_mat
                            init_material = np.copy(material_)
                            init_case = np.copy(case)
                            init_B = np.copy(B)  
                            input_params["Bmat"] = init_B                                     
                            final_match_rate = np.copy(match_rate)
                            init_mr = np.copy(match_rate)                   
                            all_stats = [i, j, \
                                         spot1_hkl[iind], spot2_hkl[iind], \
                                        tth_chi_spot1, tth_chi_spot2, \
                                        dist[i,j], tab_distance_classhkl_data[i,j], np.round(max_pred[i]*100,3), \
                                        np.round(max_pred[j]*100,3), len(rmv_ind), theospots,\
                                        match_rate, 0.0, rot_mat123, init_mat, init_material, init_B, init_case]
                    tried_spots.append(i)                 
                    
                if (final_match_rate <= cap_matchrate123): ## Nothing found!! 
                    ## Either peaks are not well defined or not found within tolerance and prediction accuracy
                    all_stats = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                        0, 0, 0, 0, 0, np.zeros((3,3))]
                    max_mr, min_mr = 0, 0
                    spot_ind = []
                    mat = 0
                    input_params["mat"] = 0
                    case = "None"
                    objective_function.append([0, [], []])
                else:
                    objective_function.append([final_match_rate, final_rmv_ind, all_stats])     
                tried_spots.append(j)
    
    sort_ind = []
    for ijk in objective_function:
        sort_ind.append(ijk[0])
    sort_ind = np.array(sort_ind)
    sort_ind = np.argsort(sort_ind)[::-1]
    
    for gr_count123 in range(len(sort_ind)):           
        max_mr = objective_function[sort_ind[gr_count123]][0]
        rmv_ind = objective_function[sort_ind[gr_count123]][1]
        all_stats = objective_function[sort_ind[gr_count123]][2]
        
        if len(rmv_ind) == 0 or max_mr==0:
            continue
        
        mat = all_stats[15]
        if material_phase_always_present != None:
            mat1 = 0
            for igr, iii in enumerate(material_phase_always_present):
                if igrain==igr and iii == mat:
                    mat1 = np.copy(mat)
        else:
            mat1 = np.copy(mat)
            
        if mat1 == 0:
            continue                    
        
        if material0_count[int(mat)-1] >= material0_limit[int(mat)-1]:
            mat = 0
            continue            

        current_spots = [len(list(set(rmv_ind) & set(spots1_global[igr])))> coeff_overlap*len(spots1_global[igr]) for igr in range(len(spots1_global))]
        
        if np.any(current_spots):
            continue

        input_params["mat"] = all_stats[15]
        if strain_calculation:
            dev_strain, strain_sample, iR, fR, rot_mat_UB = calculate_strains_fromUBMM(s_tth, s_chi, all_stats[14], str(all_stats[16]), 
                                                                 input_params, dict_dp['detectorparameters'], 
                                                                 dict_dp, spots, all_stats[17],
                                                                 strain_free_parameters)
        else:
            dev_strain, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
            rot_mat_UB = np.copy(all_stats[14])
        all_stats[14] = rot_mat_UB
        
        ## delete the indexed spots and repredict the spots HKL in the absence of indexed spots
        ## maybe it makes it easier to detect some grains
        ##update list
        # TODO
        # s_tth = np.delete(s_tth, rmv_ind, axis=0)
        # s_chi = np.delete(s_chi, rmv_ind, axis=0)
        # s_tth[rmv_ind] = np.nan
        # s_chi[rmv_ind] = np.nan
        sorted_data = np.transpose(np.array([s_tth/2., s_chi]))
        tabledistancerandom = np.transpose(GT.calculdist_from_thetachi(sorted_data, sorted_data))
        spots_in_center = np.arange(0, len(s_tth))
        # spots_in_center = spots_in_center[:nb_spots_consider]
        codebars_all = []
        for i in spots_in_center:
            spotangles = tabledistancerandom[i]
            #TODO
            spots_rmv = [i]
            for irmv in rmv_ind:
                spots_rmv.append(irmv)
            spotangles = np.delete(spotangles, spots_rmv)# removing the self distance
            
            codebars = np.histogram(spotangles, bins=angbins)[0]
            # codebars = histogram1d(spotangles, range=[min(angbins),max(angbins)], bins=len(angbins)-1)
            ## normalize the same way as training data
            max_codebars = np.max(codebars)
            codebars = codebars/ max_codebars
            codebars_all.append(codebars)
        ## reshape for the model to predict all spots at once
        codebars = np.array(codebars_all)
        
        ## Do prediction of all spots at once
        try:
            prediction = predict(codebars, wb, temp_key)
        except:
            if len(material_) > 1:
                prefix_mat = material_[0]
                for ino, imat in enumerate(material_):
                    if ino == 0:
                        continue
                    prefix_mat = prefix_mat + "_" + imat
            else:
                prefix_mat = material_
            json_file = open(model_direc+"//model_"+prefix_mat+".json", 'r')
            load_weights = model_direc + "//model_"+prefix_mat+".h5"
            # # load json and create model
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights(load_weights)
            prediction = model.predict(codebars)

        max_pred = np.max(prediction, axis = 1)
        class_predicted = np.argmax(prediction, axis = 1)
        predicted_hkl123 = classhkl[class_predicted]
        predicted_hkl123 = predicted_hkl123.astype(int)

        objective_function = None # to recalculate
        return all_stats, np.max(max_mr), np.min(max_mr), \
                rmv_ind, str(all_stats[18]), all_stats[15], dev_strain, strain_sample, iR, fR, objective_function,\
                    s_tth, s_chi, class_predicted, predicted_hkl123, max_pred, tabledistancerandom
    
    all_stats = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                        0, 0, 0, 0, 0, np.zeros((3,3))]
    max_mr, min_mr = 0, 0
    spot_ind = []
    mat = 0
    input_params["mat"] = 0
    case = "None"
    objective_function = None # to recalculate
    return all_stats, max_mr, min_mr, spot_ind, case, mat, np.zeros((3,3)), np.zeros((3,3)), 0, 0, objective_function,\
            s_tth, s_chi, class_predicted, predicted_hkl, max_pred, dist

def get_orient_mat_graphv1_MM(s_tth, s_chi, material0_, classhkl, class_predicted, predicted_hkl,
                               input_params, hkl_all_class0, max_pred, dict_dp, spots, 
                               dist, Gstar_metric0, B0, softmax_threshold=0.85, mr_threshold=0.85, 
                               tab_distance_classhkl_data0=None, spots1_global=None,
                               coeff_overlap = None, ind_mat=None, strain_calculation=None, cap_matchrate123=None,
                               material0_count=None, material0_limit=None, igrain=None, material_phase_always_present=None,
                               strain_free_parameters=None, objective_function=None):
    
    if objective_function == None:
        call_global()
        
        init_mr = 0
        init_mat = 0
        init_material = "None"
        init_case = "None"
        init_B = None
        final_match_rate = 0
        match_rate_mma = []
        final_rmv_ind = []
        
        list_of_sets = []
        for ii in range(0, min(nb_spots_consider, len(dist))):
            if max_pred[ii] < softmax_threshold:
                continue 
            
            a1 = np.round(dist[ii],3)

            for i in range(0, min(nb_spots_consider, len(dist))):
                if ii==i:
                    continue
                if (ii,i) in list_of_sets or (i,ii) in list_of_sets:
                    continue
                if max_pred[i] < softmax_threshold:
                    continue
                
                belong_same_mat = False
                for ino, imat in enumerate(material0_):
                    if ino == 0:
                        if class_predicted[ii] < ind_mat[ino] and class_predicted[i] < ind_mat[ino] :
                            tab_distance_classhkl_data = tab_distance_classhkl_data0[ino] 
                            tolerance_new = input_params["tolerance"][ino]
                            hkl1 = hkl_all_class0[ino][str(predicted_hkl[ii])]
                            hkl1_list = np.array(hkl1)
                            hkl2 = hkl_all_class0[ino][str(predicted_hkl[i])]
                            hkl2_list = np.array(hkl2)
                            Gstar_metric = Gstar_metric0[ino] 
                            belong_same_mat = True
                    else:
                        if (ind_mat[ino-1] <= class_predicted[ii] < ind_mat[ino]) and \
                                            (ind_mat[ino-1] <= class_predicted[i] < ind_mat[ino]):
                            tab_distance_classhkl_data = tab_distance_classhkl_data0[ino]
                            tolerance_new = input_params["tolerance"][ino]
                            hkl1 = hkl_all_class0[ino][str(predicted_hkl[ii])]
                            hkl1_list = np.array(hkl1)
                            hkl2 = hkl_all_class0[ino][str(predicted_hkl[i])]
                            hkl2_list = np.array(hkl2)
                            Gstar_metric = Gstar_metric0[ino]
                            belong_same_mat = True
                if not belong_same_mat:
                    continue
                tab_angulardist_temp = CP.AngleBetweenNormals(hkl1_list, hkl2_list, Gstar_metric)
                np.putmask(tab_angulardist_temp, np.abs(tab_angulardist_temp) < 0.001, 400)
                list_ = np.where(np.abs(tab_angulardist_temp-a1[i]) < tolerance_new)
                if len(list_[0]) != 0:
                    list_of_sets.append((ii,i))

        ## build a direct connection graph object
        graph_obj = nx.DiGraph(list_of_sets)
        connected_nodes_length = []
        connected_nodes = [[] for i in range(len(graph_obj))]
        for i,line in enumerate(nx.generate_adjlist(graph_obj)):
            connected_nodes_length.append(len(line.split(" ")))
            connected_nodes[i].append([int(jj) for jj in line.split(" ")])
        
        ## sort by maximum node occurance
        connected_nodes_length = np.array(connected_nodes_length)
        connected_nodes_length_sort_ind = np.argsort(connected_nodes_length)[::-1]
  
        mat = 0
        case = "None"
        tried_spots = []
        
        if len(graph_obj) == 0:
            print("no object in graph network")
            
        objective_function = []
        for toplist in range(len(graph_obj)):
            # ## continue if less than 3 connections are found for a graph
            # if connected_nodes_length[connected_nodes_length_sort_ind[toplist]] < 2:
            #     continue
            for j in connected_nodes[connected_nodes_length_sort_ind[toplist]][0]:
                init_mr = 0
                final_match_rate = 0
                final_rmv_ind = []
                all_stats = []
                for i in connected_nodes[connected_nodes_length_sort_ind[toplist]][0]:
                    if j == i:
                        continue
                    
                    if j in tried_spots and i in tried_spots:
                        continue
                    
                    #TODO replace by simpler step
                    mat = 0
                    case = "None"
                    input_params["mat"] = mat
                    input_params["Bmat"] = None
                    for ino, imat in enumerate(material0_):
                        if ino == 0:
                            if class_predicted[i] < ind_mat[ino] and class_predicted[j] < ind_mat[ino] :
                                tab_distance_classhkl_data = tab_distance_classhkl_data0[ino] 
                                hkl_all_class = hkl_all_class0[ino] 
                                material_ = imat
                                B = B0[ino] 
                                Gstar_metric = Gstar_metric0[ino] 
                                case = imat
                                mat = ino + 1
                                if material0_count[ino] >= material0_limit[ino]:
                                    mat = 0
                                    case="None"
                                input_params["mat"] = mat
                                input_params["Bmat"] = B
                        else:
                            if (ind_mat[ino-1] <= class_predicted[i] < ind_mat[ino]) and \
                                                (ind_mat[ino-1] <= class_predicted[j] < ind_mat[ino]):
                                tab_distance_classhkl_data = tab_distance_classhkl_data0[ino]
                                hkl_all_class = hkl_all_class0[ino] 
                                material_ = imat
                                B = B0[ino]
                                Gstar_metric = Gstar_metric0[ino]
                                case = imat  
                                mat = ino + 1
                                if material0_count[ino] >= material0_limit[ino]:
                                    mat = 0
                                    case="None"
                                input_params["mat"] = mat
                                input_params["Bmat"] = B
                    
                    if mat == 0:
                        continue   
                    
                    tth_chi_spot1 = np.array([s_tth[i], s_chi[i]])
                    tth_chi_spot2 = np.array([s_tth[j], s_chi[j]])         
        
                    hkl1 = hkl_all_class[str(predicted_hkl[i])]
                    hkl1_list = np.array(hkl1)
                    hkl2 = hkl_all_class[str(predicted_hkl[j])]
                    hkl2_list = np.array(hkl2)
                    
                    actual_mat, flagAM, \
                    spot1_hkl, spot2_hkl = propose_UB_matrixMM(hkl1_list, hkl2_list, 
                                                            Gstar_metric, input_params, 
                                                            dist[i,j],
                                                            tth_chi_spot1, tth_chi_spot2, 
                                                            B, method=0)

                    if flagAM:
                        continue
                    
                    for iind in range(len(actual_mat)):
                        rot_mat123 = actual_mat[iind]
                        rmv_ind, theospots = remove_spotsMM(s_tth, s_chi, rot_mat123, 
                                                            material_, input_params, 
                                                            dict_dp['detectorparameters'], dict_dp)
                        match_rate = np.round(100 * len(rmv_ind)/theospots, 3)                        
                        match_rate_mma.append(match_rate)

                        if match_rate > init_mr:
                            final_rmv_ind = rmv_ind                    
                            init_mat = np.copy(mat)
                            input_params["mat"] = init_mat
                            init_material = np.copy(material_)
                            init_case = np.copy(case)
                            init_B = np.copy(B)  
                            input_params["Bmat"] = init_B                                     
                            final_match_rate = np.copy(match_rate)
                            init_mr = np.copy(match_rate)                   
                            all_stats = [i, j, \
                                         spot1_hkl[iind], spot2_hkl[iind], \
                                        tth_chi_spot1, tth_chi_spot2, \
                                        dist[i,j], tab_distance_classhkl_data[i,j], np.round(max_pred[i]*100,3), \
                                        np.round(max_pred[j]*100,3), len(rmv_ind), theospots,\
                                        match_rate, 0.0, rot_mat123, init_mat, init_material, init_B, init_case]
                    tried_spots.append(i)                 
                    
                if (final_match_rate <= cap_matchrate123): ## Nothing found!! 
                    ## Either peaks are not well defined or not found within tolerance and prediction accuracy
                    all_stats = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                        0, 0, 0, 0, 0, np.zeros((3,3))]
                    max_mr, min_mr = 0, 0
                    spot_ind = []
                    mat = 0
                    input_params["mat"] = 0
                    case = "None"
                    objective_function.append([0, [], []])
                else:
                    objective_function.append([final_match_rate, final_rmv_ind, all_stats])     
                tried_spots.append(j)
 
    sort_ind = []
    for ijk in objective_function:
        sort_ind.append(ijk[0])
    sort_ind = np.array(sort_ind)
    sort_ind = np.argsort(sort_ind)[::-1]
    
    for gr_count123 in range(len(sort_ind)):           
        max_mr = objective_function[sort_ind[gr_count123]][0]
        rmv_ind = objective_function[sort_ind[gr_count123]][1]
        all_stats = objective_function[sort_ind[gr_count123]][2]
        
        if len(rmv_ind) == 0 or max_mr==0:
            continue
        
        mat = all_stats[15]
        if material_phase_always_present != None:
            mat1 = 0
            for igr, iii in enumerate(material_phase_always_present):
                if igrain==igr and iii == mat:
                    mat1 = np.copy(mat)
        else:
            mat1 = np.copy(mat)
            
        if mat1 == 0:
            continue                    
        
        if material0_count[int(mat)-1] >= material0_limit[int(mat)-1]:
            mat = 0
            continue            

        current_spots = [len(list(set(rmv_ind) & set(spots1_global[igr])))> coeff_overlap*len(spots1_global[igr]) for igr in range(len(spots1_global))]
        
        if np.any(current_spots):
            continue

        input_params["mat"] = all_stats[15]
        if strain_calculation:
            dev_strain, strain_sample, iR, fR, rot_mat_UB = calculate_strains_fromUBMM(s_tth, s_chi, all_stats[14], str(all_stats[16]), 
                                                                 input_params, dict_dp['detectorparameters'], 
                                                                 dict_dp, spots, all_stats[17],
                                                                 strain_free_parameters)
        else:
            dev_strain, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
            rot_mat_UB = np.copy(all_stats[14])
        all_stats[14] = rot_mat_UB     
        
        return all_stats, np.max(max_mr), np.min(max_mr), \
                rmv_ind, str(all_stats[18]), all_stats[15], dev_strain, strain_sample, iR, fR, objective_function
    
    all_stats = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                        0, 0, 0, 0, 0, np.zeros((3,3))]
    max_mr, min_mr = 0, 0
    spot_ind = []
    mat = 0
    input_params["mat"] = 0
    case = "None"
    return all_stats, max_mr, min_mr, spot_ind, case, mat, np.zeros((3,3)), np.zeros((3,3)), 0, 0, objective_function

def get_orient_matMM(s_tth, s_chi, material0_, classhkl, class_predicted, predicted_hkl,
                   input_params, hkl_all_class0, max_pred, dict_dp, spots, 
                   dist, Gstar_metric0, B0, softmax_threshold=0.85, mr_threshold=0.85, 
                   tab_distance_classhkl_data0=None, spots1_global=None,
                   coeff_overlap = None, ind_mat=None, strain_calculation=None,cap_matchrate123=None,
                   material0_count=None, material0_limit=None,
                   igrain=None, material_phase_always_present=None, strain_free_parameters=None):
    call_global()
    
    init_mr = 0
    init_mat = 0
    init_material = "None"
    init_case = "None"
    init_B = None
    final_match_rate = 0
    match_rate_mma = []
    final_rmv_ind = []
    current_spots1 = [0 for igr in range(len(spots1_global))]
    mat = 0
    case = "None"
    all_stats = []
    
    for i in range(0, min(nb_spots_consider, len(s_tth))):
        for j in range(i+1, min(nb_spots_consider, len(s_tth))):
            overlap = False

            if (max_pred[j] < softmax_threshold) or (j in spots) or \
                (max_pred[i] < softmax_threshold) or (i in spots):
                continue
            
            mat = 0
            case = "None"
            input_params["mat"] = mat
            input_params["Bmat"] = None
            mat1 = 0
            for ino, imat in enumerate(material0_):
                if ino == 0:
                    if class_predicted[i] < ind_mat[ino] and class_predicted[j] < ind_mat[ino] :
                        tab_distance_classhkl_data = tab_distance_classhkl_data0[ino] 
                        hkl_all_class = hkl_all_class0[ino] 
                        material_ = imat
                        B = B0[ino] 
                        Gstar_metric = Gstar_metric0[ino] 
                        case = imat
                        mat = ino + 1
                        if material0_count[ino] >= material0_limit[ino]:
                            mat = 0
                            case="None"
                        input_params["mat"] = mat
                        input_params["Bmat"] = B
                else:
                    if (ind_mat[ino-1] <= class_predicted[i] < ind_mat[ino]) and \
                                        (ind_mat[ino-1] <= class_predicted[j] < ind_mat[ino]):
                        tab_distance_classhkl_data = tab_distance_classhkl_data0[ino]
                        hkl_all_class = hkl_all_class0[ino] 
                        material_ = imat
                        B = B0[ino]
                        Gstar_metric = Gstar_metric0[ino]
                        case = imat  
                        mat = ino + 1
                        if material0_count[ino] >= material0_limit[ino]:
                            mat = 0
                            case="None"
                        input_params["mat"] = mat
                        input_params["Bmat"] = B
                        
                if material_phase_always_present != None:
                    mat1 = 0
                    for igr, iii in enumerate(material_phase_always_present):
                        if igrain==igr and iii == mat:
                            mat1 = np.copy(mat)
                else:
                    mat1 = np.copy(mat)
            
            if mat1 == 0:
                continue
            
            tth_chi_spot1 = np.array([s_tth[i], s_chi[i]])
            tth_chi_spot2 = np.array([s_tth[j], s_chi[j]])

            hkl1 = hkl_all_class[str(predicted_hkl[i])]
            hkl1_list = np.array(hkl1)
            hkl2 = hkl_all_class[str(predicted_hkl[j])]
            hkl2_list = np.array(hkl2)
            
            actual_mat, flagAM, \
            spot1_hkl, spot2_hkl = propose_UB_matrixMM(hkl1_list, hkl2_list, 
                                                    Gstar_metric, input_params, 
                                                    dist[i,j],
                                                    tth_chi_spot1, tth_chi_spot2, 
                                                    B, method=0)
            
            if flagAM:
                continue

            for iind in range(len(actual_mat)): 
                rot_mat123 = actual_mat[iind]
                rmv_ind, theospots = remove_spotsMM(s_tth, s_chi, rot_mat123, 
                                                    material_, input_params, 
                                                    dict_dp['detectorparameters'], dict_dp)
                
                overlap = False
                current_spots = [len(list(set(rmv_ind) & set(spots1_global[igr]))) for igr in range(len(spots1_global))]
                for igr in range(len(spots1_global)):
                    if current_spots[igr] > coeff_overlap*len(spots1_global[igr]):
                        overlap = True
                        break
                
                if overlap:
                    continue
    
                match_rate = np.round(100 * len(rmv_ind)/theospots,3)
                
                match_rate_mma.append(match_rate)
                if match_rate > init_mr:
                    current_spots1 = current_spots                       
                    init_mat = np.copy(mat)
                    input_params["mat"] = init_mat
                    init_material = np.copy(material_)
                    init_case = np.copy(case)
                    init_B = np.copy(B)
                    input_params["Bmat"] = init_B  
                    final_rmv_ind = rmv_ind                            
                    final_match_rate = np.copy(match_rate)
                    init_mr = np.copy(match_rate)
                    all_stats = [i, j, \
                                 spot1_hkl[iind], spot2_hkl[iind], \
                                tth_chi_spot1, tth_chi_spot2, \
                                dist[i,j], tab_distance_classhkl_data[i,j], np.round(max_pred[i]*100,3), \
                                np.round(max_pred[j]*100,3), len(rmv_ind), theospots,\
                                match_rate, 0.0, rot_mat123]
    
                if (final_match_rate >= mr_threshold*100.) and not overlap:
                    if strain_calculation:
                        dev_strain, strain_sample, iR, fR, rot_mat_UB = calculate_strains_fromUBMM(s_tth, s_chi, all_stats[14], str(init_material), 
                                                                             input_params, dict_dp['detectorparameters'], 
                                                                             dict_dp, spots, init_B,
                                                                             strain_free_parameters)
                    else:
                        dev_strain, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
                        rot_mat_UB = np.copy(all_stats[14])
                    
                    all_stats[14] = rot_mat_UB
                    return all_stats, np.max(match_rate_mma), np.min(match_rate_mma), \
                            final_rmv_ind, str(init_case), init_mat, dev_strain, strain_sample, iR, fR

    overlap = False
    for igr in range(len(spots1_global)):
        if current_spots1[igr] > coeff_overlap*len(spots1_global[igr]):
            overlap = True
            
    if (final_match_rate <= cap_matchrate123) or overlap: ## Nothing found!! 
        ## Either peaks are not well defined or not found within tolerance and prediction accuracy
        all_stats = [0, 0, 0, 0, 0, 0, 0, 0, 0, \
                            0, 0, 0, 0, 0, np.zeros((3,3))]
        max_mr, min_mr = 0, 0
        spot_ind = []
        mat = 0
        input_params["mat"] = 0
        case = "None"
        return all_stats, max_mr, min_mr, spot_ind, case, mat, np.zeros((3,3)), np.zeros((3,3)), 0, 0

    input_params["mat"] = init_mat
    if strain_calculation:
        dev_strain, strain_sample, iR, fR, rot_mat_UB = calculate_strains_fromUBMM(s_tth, s_chi, all_stats[14], str(init_material), 
                                                             input_params, dict_dp['detectorparameters'], 
                                                             dict_dp, spots, init_B,
                                                             strain_free_parameters)
    else:
        dev_strain, strain_sample, iR, fR = np.zeros((3,3)), np.zeros((3,3)), 0, 0
        rot_mat_UB = np.copy(all_stats[14])
    all_stats[14] = rot_mat_UB  
    return all_stats, np.max(match_rate_mma), np.min(match_rate_mma), \
            final_rmv_ind, str(init_case), init_mat, dev_strain, strain_sample, iR, fR
            

def propose_UB_matrixMM(hkl1_list, hkl2_list, Gstar_metric, input_params, dist123,
                      tth_chi_spot1, tth_chi_spot2, B, method=0, crystal=None):
    
    if method == 0:
        tab_angulardist_temp = CP.AngleBetweenNormals(hkl1_list, hkl2_list, Gstar_metric)
        list_ = np.where(np.abs(tab_angulardist_temp-dist123) < input_params["tolerance"][input_params["mat"]-1])
        
        if crystal != None:
            final_crystal=crystal[input_params["mat"]-1]
            symm_operator = final_crystal._hklsym
        else:
            symm_operator = np.eye(3)
        
        if len(list_[0]) == 0:
            return None, True, 0, 0

        rot_mat_abs = []
        actual_mat = []
        spot1_hkl = []
        spot2_hkl = []
        
        triedspots = []
        for ii, jj in zip(list_[0], list_[1]):
            if ii in triedspots and jj in triedspots:
                continue

            conti_ = False
            
            try:
                rot_mat1 = FindO.OrientMatrix_from_2hkl(hkl1_list[ii], tth_chi_spot1, \
                                                        hkl2_list[jj], tth_chi_spot2,
                                                        B)
                # rot_mat1 = find_uniq_u(rot_mat1, symm_operator)
            except:
                continue                    
            
            copy_rm = np.copy(rot_mat1)
            copy_rm = np.round(np.abs(copy_rm),5)
            copy_rm.sort(axis=1)
            for iji in rot_mat_abs:
                iji.sort(axis=1)                        
                if np.all(iji==copy_rm):
                    conti_ = True
                    break
            if conti_:
                continue
            rot_mat_abs.append(np.round(np.abs(rot_mat1),5))
            actual_mat.append(rot_mat1)
            spot1_hkl.append(hkl1_list[ii])
            spot2_hkl.append(hkl2_list[jj])
            triedspots.append(ii)
            triedspots.append(jj)
    else:  
        # method 2
        hkl_all = np.vstack((hkl1_list, hkl2_list))
        LUT = FindO.GenerateLookUpTable(hkl_all, Gstar_metric)
        hkls = FindO.PlanePairs_2(dist123, input_params["tolerance"][input_params["mat"]-1], LUT, onlyclosest=1)

        if np.all(hkls == None):
            return None, True, 0, 0
                
        rot_mat_abs = []
        actual_mat = []
        spot1_hkl = []
        spot2_hkl = []
        
        for ii in range(len(hkls)):
            if np.all(hkls[ii][0] == hkls[ii][1]):
                continue
            conti_ = False
            
            try:
                rot_mat1 = FindO.OrientMatrix_from_2hkl(hkls[ii][0], tth_chi_spot1, \
                                                        hkls[ii][1], tth_chi_spot2,
                                                        B)
                # rot_mat1 = find_uniq_u(rot_mat1, symm_operator)
            except:
                continue                    
            
            copy_rm = np.copy(rot_mat1)
            copy_rm = np.round(np.abs(copy_rm),5)
            copy_rm.sort(axis=1)
            for iji in rot_mat_abs:
                iji.sort(axis=1)
                if np.all(iji==copy_rm):
                    conti_ = True
                    break

            if conti_:
                continue
            rot_mat_abs.append(np.round(np.abs(rot_mat1),5))
            actual_mat.append(rot_mat1)
            spot1_hkl.append(hkls[ii][0])
            spot2_hkl.append(hkls[ii][1])
    
    ## just fixing a* to x seems ok; if not think of aligning b* to xy plane
    sum_sign = []
    for nkl in range(len(actual_mat)):
        temp_mat = np.dot(actual_mat[nkl], B)
        ## fix could be to choose a matrix that aligns best the b* vector to Y axis or a* to X axis
        # if np.argmax(np.abs(temp_mat[:2,0])) == 0 and \
        #         np.argmax(np.abs(temp_mat[:2,1])) == 1: ##a* along x, b*along y
        if np.argmax(np.abs(temp_mat[:2,0])) == 0: ##a* along x
            sum_sign.append(2)
        elif np.argmax(np.abs(temp_mat[:2,0])) ==  np.argmax(np.abs(temp_mat[:2,1])):
            sum_sign.append(0)
        else:
            sum_sign.append(1)
    ind_sort = np.argsort(sum_sign)[::-1]
    ## re-arrange
    actual_mat1 = []
    spot1_hkl1, spot2_hkl1 = [], []
    for inin in ind_sort:
        actual_mat1.append(actual_mat[inin])
        spot1_hkl1.append(spot1_hkl[inin])
        spot2_hkl1.append(spot2_hkl[inin])
    actual_mat, spot1_hkl, spot2_hkl = actual_mat1, spot1_hkl1, spot2_hkl1
    return actual_mat, False, spot1_hkl, spot2_hkl


def remove_spotsMM(s_tth, s_chi, first_match123, material_, input_params, detectorparameters, dict_dp):
    try:
        grain = CP.Prepare_Grain(material_, first_match123, dictmaterials=dictLT.dict_Materials)
        ### initialize global variables to be used later
        call_global()
    except:
        return [], 100
    #### Perhaps better than SimulateResult function
    kf_direction = dict_dp["kf_direction"]
    detectordistance = dict_dp["detectorparameters"][0]
    detectordiameter = dict_dp["detectordiameter"]
    pixelsize = dict_dp["pixelsize"]
    dim = dict_dp["dim"]
           
    spots2pi = LT.getLaueSpots(CST_ENERGYKEV / input_params["emax"], 
                               CST_ENERGYKEV / input_params["emin"],
                                    [grain],
                                    fastcompute=1,
                                    verbose=0,
                                    kf_direction=kf_direction,
                                    ResolutionAngstrom=False,
                                    dictmaterials=dictLT.dict_Materials)

    TwicethetaChi = LT.filterLaueSpots_full_np(spots2pi[0][0], None, onlyXYZ=False,
                                                    HarmonicsRemoval=0,
                                                    fastcompute=1,
                                                    kf_direction=kf_direction,
                                                    detectordistance=detectordistance,
                                                    detectordiameter=detectordiameter,
                                                    pixelsize=pixelsize,
                                                    dim=dim)
    ## get proximity for exp and theo spots
    if input_params["mat"] == 0:
        return [], 100
    angtol = input_params["tolerance"][input_params["mat"] -1]
    
    if option_global =="v1":
        # print("entering v1")
        List_Exp_spot_close, residues_link, _ = getProximityv1(np.array([TwicethetaChi[0], TwicethetaChi[1]]),  # warning array(2theta, chi)
                                                  s_tth/2.0, s_chi,  # warning theta, chi for exp
                                                  angtol=angtol)
    elif option_global =="v2":
        List_Exp_spot_close, residues_link, _ = getProximityv1_ambigious(np.array([TwicethetaChi[0], TwicethetaChi[1]]),  # warning array(2theta, chi)
                                                  s_tth/2.0, s_chi,  # warning theta, chi for exp
                                                  angtol=angtol)
    else:
        List_Exp_spot_close, residues_link, _ = getProximityv1_ambigious(np.array([TwicethetaChi[0], TwicethetaChi[1]]),  # warning array(2theta, chi)
                                                  s_tth/2.0, s_chi,  # warning theta, chi for exp
                                                  angtol=angtol)
        List_Exp_spot_close, ind_uniq = np.unique(List_Exp_spot_close, return_index=True)
        residues_link = np.take(residues_link, ind_uniq)

    if np.average(residues_link) > residues_threshold:
        return [], 100
    
    if len(np.unique(List_Exp_spot_close)) < nb_spots_global_threshold:
        return [], 100
    
    return List_Exp_spot_close, len(TwicethetaChi[0])


def calculate_strains_fromUBMM(s_tth, s_chi, UBmat, material_, input_params, 
                             detectorparameters, dict_dp, spots, B_matrix, strain_free_parameters):
    ## for the moment strain_free_parameters is a trial implementation 
    if ("a" not in strain_free_parameters) and len(strain_free_parameters)>=5:
        if additional_expression[0] != "none":
            print("Note: additional_expression is not applied for the current set of strain free parameters")
        # starting B0matrix corresponding to the unit cell   -----
        B0matrix = np.copy(B_matrix)
        latticeparams = dictLT.dict_Materials[material_][1]
        ## Included simple multi level refinement of strains
        init_residues = -0.1
        final_residues = -0.1
        
        straintolerance = input_params["tolerancestrain"][input_params["mat"]-1]
        
        devstrain, deviatoricstrain_sampleframe = np.zeros((3,3)), np.zeros((3,3))
        for ijk, AngTol in enumerate(straintolerance):
            #### Spots in first match (no refining, just simple auto links to filter spots)        
            grain = CP.Prepare_Grain(material_, UBmat, dictmaterials=dictLT.dict_Materials)

            Twicetheta, Chi, Miller_ind, posx, posy, _ = LT.SimulateLaue(grain,
                                                                         input_params["emin"], 
                                                                         input_params["emax"], 
                                                                         detectorparameters,
                                                                         kf_direction=dict_dp['kf_direction'],
                                                                         removeharmonics=1,
                                                                         pixelsize=dict_dp['pixelsize'],
                                                                         dim=dict_dp['dim'],
                                                                         ResolutionAngstrom=False,
                                                                         detectordiameter=dict_dp['detectordiameter'],
                                                                         dictmaterials=dictLT.dict_Materials)
            ## get proximity for exp and theo spots
            linkedspots_link, linkExpMiller_link, \
                linkResidues_link = getProximityv0(np.array([Twicetheta, Chi]),  # warning array(2theta, chi)
                                                                                    s_tth/2.0, s_chi, Miller_ind,  # warning theta, chi for exp
                                                                                    angtol=float(AngTol))
            
            if len(linkedspots_link) < 8:
                return np.zeros((3,3)), np.zeros((3,3)), init_residues, final_residues, UBmat
            
            linkedspots_fit = linkedspots_link
            linkExpMiller_fit = linkExpMiller_link
            
            arraycouples = np.array(linkedspots_fit)
            exp_indices = np.array(arraycouples[:, 0], dtype=np.int)
            sim_indices = np.array(arraycouples[:, 1], dtype=np.int)
        
            nb_pairs = len(exp_indices)
            Data_Q = np.array(linkExpMiller_fit)[:, 1:]
            sim_indices = np.arange(nb_pairs)  # for fitting function this must be an arange...
        
            pixX = np.take(dict_dp['peakX'], exp_indices)
            pixY = np.take(dict_dp['peakY'], exp_indices)
            weights = None #np.take(dict_dp['intensity'], exp_indices)
            
            starting_orientmatrix = np.copy(UBmat)
        
            results = None
            # ----------------------------------
            #  refinement model
            # ----------------------------------
            # -------------------------------------------------------
            allparameters = np.array(detectorparameters + [1, 1, 0, 0, 0] + [0, 0, 0])
            # strain & orient
            initial_values = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0])
            arr_indexvaryingparameters = np.arange(5, 13)
        
            residues, deltamat, newmatrix = FitO.error_function_on_demand_strain(
                                                                                initial_values,
                                                                                Data_Q,
                                                                                allparameters,
                                                                                arr_indexvaryingparameters,
                                                                                sim_indices,
                                                                                pixX,
                                                                                pixY,
                                                                                initrot=starting_orientmatrix,
                                                                                Bmat=B0matrix,
                                                                                pureRotation=0,
                                                                                verbose=1,
                                                                                pixelsize=dict_dp['pixelsize'],
                                                                                dim=dict_dp['dim'],
                                                                                weights=weights,
                                                                                kf_direction=dict_dp['kf_direction'])
            init_mean_residues = np.copy(np.mean(residues))
            
            if ijk == 0:
                init_residues = np.copy(init_mean_residues)
            
            results = FitO.fit_on_demand_strain(initial_values,
                                                    Data_Q,
                                                    allparameters,
                                                    FitO.error_function_on_demand_strain,
                                                    arr_indexvaryingparameters,
                                                    sim_indices,
                                                    pixX,
                                                    pixY,
                                                    initrot=starting_orientmatrix,
                                                    Bmat=B0matrix,
                                                    pixelsize=dict_dp['pixelsize'],
                                                    dim=dict_dp['dim'],
                                                    verbose=0,
                                                    weights=weights,
                                                    kf_direction=dict_dp['kf_direction'])
        
            if results is None:
                return np.zeros((3,3)), np.zeros((3,3)), init_residues, final_residues, UBmat
        
            residues, deltamat, newmatrix = FitO.error_function_on_demand_strain(
                                                                                results,
                                                                                Data_Q,
                                                                                allparameters,
                                                                                arr_indexvaryingparameters,
                                                                                sim_indices,
                                                                                pixX,
                                                                                pixY,
                                                                                initrot=starting_orientmatrix,
                                                                                Bmat=B0matrix,
                                                                                pureRotation=0,
                                                                                verbose=1,
                                                                                pixelsize=dict_dp['pixelsize'],
                                                                                dim=dict_dp['dim'],
                                                                                weights=weights,
                                                                                kf_direction=dict_dp['kf_direction'])
            # if np.mean(residues) > final_residues:
            #     return devstrain, deviatoricstrain_sampleframe, init_residues, final_residues, UBmat
            final_mean_residues = np.copy(np.mean(residues))
            final_residues = np.copy(final_mean_residues)
            # building B mat
            # param_strain_sol = results
            # varyingstrain = np.array([[1.0, param_strain_sol[2], param_strain_sol[3]],
            #                                 [0, param_strain_sol[0], param_strain_sol[4]],
            #                                 [0, 0, param_strain_sol[1]]])
            # newUmat = np.dot(deltamat, starting_orientmatrix)
            # newUBmat = np.dot(newUmat, varyingstrain)
            newUBmat = np.copy(newmatrix) 
            # Bstar_s = np.dot(newUBmat, B0matrix)
            # ---------------------------------------------------------------
            # postprocessing of unit cell orientation and strain refinement
            # ---------------------------------------------------------------
            UBmat = np.copy(newmatrix) 
            (devstrain, lattice_parameter_direct_strain) = CP.compute_deviatoricstrain(newUBmat, B0matrix, latticeparams)
            # overwrite and rescale possibly lattice lengthes
            # constantlength = "a"
            # lattice_parameter_direct_strain = CP.computeLatticeParameters_from_UB(newUBmat, material_, constantlength, dictmaterials=dictLT.dict_Materials)
            # print(lattice_parameter_direct_strain)
            deviatoricstrain_sampleframe = CP.strain_from_crystal_to_sample_frame2(devstrain, newUBmat)
            # in % already
            devstrain = np.round(devstrain * 100, decimals=3)
            deviatoricstrain_sampleframe = np.round(deviatoricstrain_sampleframe * 100, decimals=3)
    else:
        # starting B0matrix corresponding to the unit cell   -----
        B0matrix = np.copy(B_matrix)
        latticeparams = dictLT.dict_Materials[material_][1]
        ## Included simple multi level refinement of strains
        init_residues = -0.1
        final_residues = -0.1
        
        straintolerance = input_params["tolerancestrain"][input_params["mat"]-1]
        
        devstrain, deviatoricstrain_sampleframe = np.zeros((3,3)), np.zeros((3,3))
        for ijk, AngTol in enumerate(straintolerance):
            #### Spots in first match (no refining, just simple auto links to filter spots)        
            grain = CP.Prepare_Grain(material_, UBmat, dictmaterials=dictLT.dict_Materials)
            Twicetheta, Chi, Miller_ind, posx, posy, _ = LT.SimulateLaue(grain,
                                                                     input_params["emin"], 
                                                                     input_params["emax"], 
                                                                     detectorparameters,
                                                                     kf_direction=dict_dp['kf_direction'],
                                                                     removeharmonics=1,
                                                                     pixelsize=dict_dp['pixelsize'],
                                                                     dim=dict_dp['dim'],
                                                                     ResolutionAngstrom=False,
                                                                     detectordiameter=dict_dp['detectordiameter'],
                                                                     dictmaterials=dictLT.dict_Materials)
            ## get proximity for exp and theo spots
            linkedspots_link, linkExpMiller_link, \
                linkResidues_link = getProximityv0(np.array([Twicetheta, Chi]),  # warning array(2theta, chi)
                                                            s_tth/2.0, s_chi, Miller_ind,  # warning theta, chi for exp
                                                            angtol=float(AngTol))
            
            if len(linkedspots_link) < 8:
                return np.zeros((3,3)), np.zeros((3,3)), init_residues, final_residues, UBmat
            
            linkedspots_fit = linkedspots_link
            linkExpMiller_fit = linkExpMiller_link
            
            arraycouples = np.array(linkedspots_fit)
            exp_indices = np.array(arraycouples[:, 0], dtype=np.int)
            sim_indices = np.array(arraycouples[:, 1], dtype=np.int)
        
            nb_pairs = len(exp_indices)
            Data_Q = np.array(linkExpMiller_fit)[:, 1:]
            sim_indices = np.arange(nb_pairs)  # for fitting function this must be an arange...
        
            pixX = np.take(dict_dp['peakX'], exp_indices)
            pixY = np.take(dict_dp['peakY'], exp_indices)
            weights = None #np.take(dict_dp['intensity'], exp_indices)
            
            starting_orientmatrix = np.copy(UBmat)
        
            results = None
            # ----------------------------------
            #  refinement model
            # ----------------------------------
            # -------------------------------------------------------
            allparameters = np.array(detectorparameters + [0, 0, 0] + latticeparams)
            
            fitting_parameters_keys = ["anglex", "angley", "anglez"]
            fitting_parameters_values =  [0, 0, 0]
            constantlength = "a"
            if ("a" in strain_free_parameters) and ("b" in strain_free_parameters) and ("c" in strain_free_parameters):
                constantlength = "a"                    
            elif ("b" not in strain_free_parameters) and additional_expression[0]=="none" and\
                "b" not in additional_expression[0]:
                constantlength = "b"
            elif ("c" not in strain_free_parameters):
                constantlength = "c"
            
            for jjkk in strain_free_parameters:
                if jjkk == "a" and constantlength != "a":
                    fitting_parameters_keys.append("a")
                    fitting_parameters_values.append(latticeparams[0])
                if jjkk == "b" and constantlength != "b":
                    fitting_parameters_keys.append("b")
                    fitting_parameters_values.append(latticeparams[1])
                if jjkk == "c" and constantlength != "c":
                    fitting_parameters_keys.append("c")
                    fitting_parameters_values.append(latticeparams[2])
                if jjkk == "alpha":
                    fitting_parameters_keys.append("alpha")
                    fitting_parameters_values.append(latticeparams[3])
                if jjkk == "beta":
                    fitting_parameters_keys.append("beta")
                    fitting_parameters_values.append(latticeparams[4])
                if jjkk == "gamma":
                    fitting_parameters_keys.append("gamma")
                    fitting_parameters_values.append(latticeparams[5])
                    
            pureUmatrix, _ = GT.UBdecomposition_RRPP(starting_orientmatrix)
            absolutespotsindices = np.arange(len(pixX))
            
            (residues, _, _,
                _,  _, ) = FitO.error_function_latticeparameters(fitting_parameters_values,
                                                                fitting_parameters_keys,
                                                                Data_Q,
                                                                allparameters,
                                                                absolutespotsindices,
                                                                pixX,
                                                                pixY,
                                                                initrot=pureUmatrix,
                                                                pureRotation=0,
                                                                verbose=0,
                                                                pixelsize=dict_dp['pixelsize'],
                                                                dim=dict_dp['dim'],
                                                                weights=weights,
                                                                kf_direction=dict_dp['kf_direction'],
                                                                returnalldata=True,
                                                                additional_expression = additional_expression[0])
            init_mean_residues = np.copy(np.mean(residues))
            if ijk == 0:
                init_residues = np.copy(init_mean_residues)
                
            results = FitO.fit_function_latticeparameters(fitting_parameters_values,
                                                            fitting_parameters_keys,
                                                            Data_Q,
                                                            allparameters,
                                                            absolutespotsindices,
                                                            pixX,
                                                            pixY,
                                                            UBmatrix_start=pureUmatrix,
                                                            nb_grains=1,
                                                            pureRotation=0,
                                                            verbose=0,
                                                            pixelsize=dict_dp['pixelsize'],
                                                            dim=dict_dp['dim'],
                                                            weights=weights,
                                                            kf_direction=dict_dp['kf_direction'],
                                                            additional_expression = additional_expression[0])
            if results is None:
                return np.zeros((3,3)), np.zeros((3,3)), init_residues, final_residues, UBmat
            
            (residues, Uxyz, newUmat,
                newB0matrix,  _, ) = FitO.error_function_latticeparameters(results,
                                                                fitting_parameters_keys,
                                                                Data_Q,
                                                                allparameters,
                                                                absolutespotsindices,
                                                                pixX,
                                                                pixY,
                                                                initrot=pureUmatrix,
                                                                pureRotation=0,
                                                                verbose=0,
                                                                pixelsize=dict_dp['pixelsize'],
                                                                dim=dict_dp['dim'],
                                                                weights=weights,
                                                                kf_direction=dict_dp['kf_direction'],
                                                                returnalldata=True,
                                                                additional_expression = additional_expression[0])
            final_mean_residues = np.copy(np.mean(residues))
            final_residues = np.copy(final_mean_residues)
            newUBmat = np.dot(np.dot(newUmat, newB0matrix), np.linalg.inv(B0matrix))
            UBmat = np.copy(newUBmat) 
            # ---------------------------------------------------------------
            # postprocessing of unit cell orientation and strain refinement
            # ---------------------------------------------------------------
            (devstrain, lattice_parameter_direct_strain) = CP.compute_deviatoricstrain(newUBmat, B0matrix, latticeparams)
            deviatoricstrain_sampleframe = CP.strain_from_crystal_to_sample_frame2(devstrain, newUBmat)
            # in % already
            devstrain = np.round(devstrain * 100, decimals=3)
            deviatoricstrain_sampleframe = np.round(deviatoricstrain_sampleframe * 100, decimals=3)
    return devstrain, deviatoricstrain_sampleframe, init_residues, final_residues, UBmat


      
def global_plots_MM(lim_x, lim_y, rotation_matrix1, strain_matrix, strain_matrixs, col, colx, coly,
                 match_rate, mat_global, spots_len, iR_pix, fR_pix,
                 model_direc, material_, match_rate_threshold=5, bins=30, constantlength="a"):
    call_global()

    mu_sd = []
    mu_sdc = []
    material_id = material_
    for matid in range(len(material_)):
        for index in range(len(spots_len)):
            ### index for nans
            nan_index1 = np.where(match_rate[index][0] <= match_rate_threshold)[0]
            mat_id_index = np.where(mat_global[index][0] != matid+1)[0]
            nan_index = np.hstack((mat_id_index,nan_index1))
            nan_index = np.unique(nan_index)
            
            if index == 0:
                spots_len_plot = np.copy(spots_len[index][0])
                mr_plot = np.copy(match_rate[index][0])
                iR_pix_plot = np.copy(iR_pix[index][0])
                fR_pix_plot = np.copy(fR_pix[index][0])
                strain_matrix_plot = np.copy(strain_matrix[index][0])
                e11c = strain_matrix_plot[:,0,0]#.reshape((lim_x, lim_y))
                e22c = strain_matrix_plot[:,1,1]#.reshape((lim_x, lim_y))
                e33c = strain_matrix_plot[:,2,2]#.reshape((lim_x, lim_y))
                e12c = strain_matrix_plot[:,0,1]#.reshape((lim_x, lim_y))
                e13c = strain_matrix_plot[:,0,2]#.reshape((lim_x, lim_y))
                e23c = strain_matrix_plot[:,1,2]#.reshape((lim_x, lim_y))
                strain_matrixs_plot = np.copy(strain_matrixs[index][0])
                e11s = strain_matrixs_plot[:,0,0]#.reshape((lim_x, lim_y))
                e22s = strain_matrixs_plot[:,1,1]#.reshape((lim_x, lim_y))
                e33s = strain_matrixs_plot[:,2,2]#.reshape((lim_x, lim_y))
                e12s = strain_matrixs_plot[:,0,1]#.reshape((lim_x, lim_y))
                e13s = strain_matrixs_plot[:,0,2]#.reshape((lim_x, lim_y))
                e23s = strain_matrixs_plot[:,1,2]#.reshape((lim_x, lim_y))
                spots_len_plot[nan_index] = np.nan 
                mr_plot[nan_index] = np.nan 
                iR_pix_plot[nan_index] = np.nan 
                fR_pix_plot[nan_index] = np.nan 
                e11c[nan_index] = np.nan 
                e22c[nan_index] = np.nan 
                e33c[nan_index] = np.nan 
                e12c[nan_index] = np.nan 
                e13c[nan_index] = np.nan 
                e23c[nan_index] = np.nan 
                e11s[nan_index] = np.nan 
                e22s[nan_index] = np.nan 
                e33s[nan_index] = np.nan 
                e12s[nan_index] = np.nan 
                e13s[nan_index] = np.nan 
                e23s[nan_index] = np.nan 
                
            else:
                temp = np.copy(spots_len[index][0])
                temp[nan_index] = np.nan
                spots_len_plot = np.vstack((spots_len_plot,temp))
                
                temp = np.copy(match_rate[index][0])
                temp[nan_index] = np.nan
                mr_plot = np.vstack((mr_plot,temp))
                
                temp = np.copy(iR_pix[index][0])
                temp[nan_index] = np.nan
                iR_pix_plot = np.vstack((iR_pix_plot,temp))
        
                temp = np.copy(fR_pix[index][0])
                temp[nan_index] = np.nan
                fR_pix_plot = np.vstack((fR_pix_plot,temp))
                
                strain_matrix_plot = np.copy(strain_matrix[index][0])
                temp = np.copy(strain_matrix_plot[:,0,0])
                temp[nan_index] = np.nan
                e11c = np.vstack((e11c,temp))
                temp = np.copy(strain_matrix_plot[:,1,1])
                temp[nan_index] = np.nan
                e22c = np.vstack((e22c,temp))
                temp = np.copy(strain_matrix_plot[:,2,2])
                temp[nan_index] = np.nan
                e33c = np.vstack((e33c,temp))
                temp = np.copy(strain_matrix_plot[:,0,1])
                temp[nan_index] = np.nan
                e12c = np.vstack((e12c,temp))
                temp = np.copy(strain_matrix_plot[:,0,2])
                temp[nan_index] = np.nan
                e13c = np.vstack((e13c,temp))
                temp = np.copy(strain_matrix_plot[:,1,2])
                temp[nan_index] = np.nan
                e23c = np.vstack((e23c,temp))
                ##
                strain_matrixs_plot = np.copy(strain_matrixs[index][0])
                temp = np.copy(strain_matrixs_plot[:,0,0])
                temp[nan_index] = np.nan
                e11s = np.vstack((e11s,temp))
                temp = np.copy(strain_matrixs_plot[:,1,1])
                temp[nan_index] = np.nan
                e22s = np.vstack((e22s,temp))
                temp = np.copy(strain_matrixs_plot[:,2,2])
                temp[nan_index] = np.nan
                e33s = np.vstack((e33s,temp))
                temp = np.copy(strain_matrixs_plot[:,0,1])
                temp[nan_index] = np.nan
                e12s = np.vstack((e12s,temp))
                temp = np.copy(strain_matrixs_plot[:,0,2])
                temp[nan_index] = np.nan
                e13s = np.vstack((e13s,temp))
                temp = np.copy(strain_matrixs_plot[:,1,2])
                temp[nan_index] = np.nan
                e23s = np.vstack((e23s,temp))
        
        spots_len_plot = spots_len_plot.flatten()
        mr_plot = mr_plot.flatten()
        iR_pix_plot = iR_pix_plot.flatten()
        fR_pix_plot = fR_pix_plot.flatten() 
        e11c = e11c.flatten()
        e22c = e22c.flatten()
        e33c = e33c.flatten()
        e12c = e12c.flatten()
        e13c = e13c.flatten()
        e23c = e23c.flatten()
        e11s = e11s.flatten()
        e22s = e22s.flatten()
        e33s = e33s.flatten()
        e12s = e12s.flatten()
        e13s = e13s.flatten()
        e23s = e23s.flatten()
        
        spots_len_plot = spots_len_plot[~np.isnan(spots_len_plot)]
        mr_plot = mr_plot[~np.isnan(mr_plot)]
        iR_pix_plot = iR_pix_plot[~np.isnan(iR_pix_plot)]
        fR_pix_plot = fR_pix_plot[~np.isnan(fR_pix_plot)]
        e11c = e11c[~np.isnan(e11c)]
        e22c = e22c[~np.isnan(e22c)]
        e33c = e33c[~np.isnan(e33c)]
        e12c = e12c[~np.isnan(e12c)]
        e13c = e13c[~np.isnan(e13c)]
        e23c = e23c[~np.isnan(e23c)]
        e11s = e11s[~np.isnan(e11s)]
        e22s = e22s[~np.isnan(e22s)]
        e33s = e33s[~np.isnan(e33s)]
        e12s = e12s[~np.isnan(e12s)]
        e13s = e13s[~np.isnan(e13s)]
        e23s = e23s[~np.isnan(e23s)]
        
        try:
            title = "Number of spots and matching rate"
            fig = plt.figure()
            axs = fig.subplots(1, 2)
            axs[0].set_title("Number of spots", loc='center', fontsize=8)
            axs[0].hist(spots_len_plot, bins=bins)
            axs[0].set_ylabel('Frequency', fontsize=8)
            axs[0].tick_params(axis='both', which='major', labelsize=8)
            axs[0].tick_params(axis='both', which='minor', labelsize=8)
            axs[1].set_title("matching rate", loc='center', fontsize=8)
            axs[1].hist(mr_plot, bins=bins)
            axs[1].set_ylabel('Frequency', fontsize=8)
            axs[1].tick_params(axis='both', which='major', labelsize=8)
            axs[1].tick_params(axis='both', which='minor', labelsize=8)
            plt.tight_layout()
            plt.savefig(model_direc+ "//"+title+"_"+material_id[matid]+'.png', format='png', dpi=1000) 
            plt.close(fig)
        except:
            pass
        
        try:
            title = "Initial and Final residues"
            fig = plt.figure()
            axs = fig.subplots(1, 2)
            axs[0].set_title("Initial residues", loc='center', fontsize=8)
            axs[0].hist(iR_pix_plot, bins=bins)
            axs[0].set_ylabel('Frequency', fontsize=8)
            axs[0].tick_params(axis='both', which='major', labelsize=8)
            axs[0].tick_params(axis='both', which='minor', labelsize=8)
            axs[1].set_title("Final residues", loc='center', fontsize=8)
            axs[1].hist(fR_pix_plot, bins=bins)
            axs[1].set_ylabel('Frequency', fontsize=8)
            axs[1].tick_params(axis='both', which='major', labelsize=8)
            axs[1].tick_params(axis='both', which='minor', labelsize=8)
            plt.tight_layout()
            plt.savefig(model_direc+ "//"+title+"_"+material_id[matid]+'.png',format='png', dpi=1000) 
            plt.close(fig)
        except:
            pass            
        
        try:
            title = "strain Crystal reference"+" "+material_id[matid]
            fig = plt.figure()
            fig.suptitle(title, fontsize=10)
            axs = fig.subplots(2, 3)
            axs[0, 0].set_title(r"$\epsilon_{11}$ (%)", loc='center', fontsize=8)
            logdata = e11c #np.log(e11c)
            xmin = logdata.min()
            xmax = logdata.max()
            x1 = np.linspace(xmin, xmax, 1000)
            estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
            pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
            axs[0, 0].axvline(x=estimated_mu, c="k")
            axs[0, 0].plot(x1, pdf, 'r')
            axs[0, 0].hist(logdata, bins=bins, density=True, alpha=0.8)
            mu_sdc.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
            axs[0, 0].set_ylabel('Frequency', fontsize=8)
            axs[0, 0].tick_params(axis='both', which='major', labelsize=8)
            axs[0, 0].tick_params(axis='both', which='minor', labelsize=8)
            
            axs[0, 1].set_title(r"$\epsilon_{22}$ (%)", loc='center', fontsize=8)
            logdata = e22c #np.log(e22c)
            xmin = logdata.min()
            xmax = logdata.max()
            x1 = np.linspace(xmin, xmax, 1000)
            estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
            pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
            axs[0, 1].axvline(x=estimated_mu, c="k")
            axs[0, 1].plot(x1, pdf, 'r')
            axs[0, 1].hist(logdata, bins=bins, density=True, alpha=0.8)
            mu_sdc.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
            # axs[0, 1].hist(e22c, bins=bins)
            axs[0, 1].set_ylabel('Frequency', fontsize=8)
            axs[0, 1].tick_params(axis='both', which='major', labelsize=8)
            axs[0, 1].tick_params(axis='both', which='minor', labelsize=8)
            
            axs[0, 2].set_title(r"$\epsilon_{33}$ (%)", loc='center', fontsize=8)
            logdata = e33c #np.log(e33c)
            xmin = logdata.min()
            xmax = logdata.max()
            x1 = np.linspace(xmin, xmax, 1000)
            estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
            pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
            axs[0, 2].axvline(x=estimated_mu, c="k")
            axs[0, 2].plot(x1, pdf, 'r')
            axs[0, 2].hist(logdata, bins=bins, density=True, alpha=0.8)
            mu_sdc.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
            # axs[0, 2].hist(e33c, bins=bins)
            axs[0, 2].set_ylabel('Frequency', fontsize=8)
            axs[0, 2].tick_params(axis='both', which='major', labelsize=8)
            axs[0, 2].tick_params(axis='both', which='minor', labelsize=8)
            
            axs[1, 0].set_title(r"$\epsilon_{12}$ (%)", loc='center', fontsize=8)
            logdata = e12c#np.log(e12c)
            xmin = logdata.min()
            xmax = logdata.max()
            x1 = np.linspace(xmin, xmax, 1000)
            estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
            pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
            axs[1, 0].axvline(x=estimated_mu, c="k")
            axs[1, 0].plot(x1, pdf, 'r')
            axs[1, 0].hist(logdata, bins=bins, density=True, alpha=0.8)
            mu_sdc.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
            # axs[1, 0].hist(e12c, bins=bins)
            axs[1, 0].set_ylabel('Frequency', fontsize=8)
            axs[1, 0].tick_params(axis='both', which='major', labelsize=8)
            axs[1, 0].tick_params(axis='both', which='minor', labelsize=8)
            
            axs[1, 1].set_title(r"$\epsilon_{13}$ (%)", loc='center', fontsize=8)
            logdata = e13c#np.log(e13c)
            xmin = logdata.min()
            xmax = logdata.max()
            x1 = np.linspace(xmin, xmax, 1000)
            estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
            pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
            axs[1, 1].axvline(x=estimated_mu, c="k")
            axs[1, 1].plot(x1, pdf, 'r')
            axs[1, 1].hist(logdata, bins=bins, density=True, alpha=0.8)
            mu_sdc.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
            # axs[1, 1].hist(e13c, bins=bins)
            axs[1, 1].set_ylabel('Frequency', fontsize=8)
            axs[1, 1].tick_params(axis='both', which='major', labelsize=8)
            axs[1, 1].tick_params(axis='both', which='minor', labelsize=8)
            
            axs[1, 2].set_title(r"$\epsilon_{23}$ (%)", loc='center', fontsize=8)
            logdata = e23c#np.log(e23c)
            xmin = logdata.min()
            xmax = logdata.max()
            x1 = np.linspace(xmin, xmax, 1000)
            estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
            pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
            axs[1, 2].axvline(x=estimated_mu, c="k")
            axs[1, 2].plot(x1, pdf, 'r')
            axs[1, 2].hist(logdata, bins=bins, density=True, alpha=0.8)
            # axs[1, 2].hist(e23c, bins=bins)
            axs[1, 2].set_ylabel('Frequency', fontsize=8)
            axs[1, 2].tick_params(axis='both', which='major', labelsize=8)
            axs[1, 2].tick_params(axis='both', which='minor', labelsize=8)
            mu_sdc.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
            plt.tight_layout()
            plt.savefig(model_direc+ "//"+title+'.png', format='png', dpi=1000) 
            plt.close(fig)
        except:
            pass
    
        try:
            title = "strain Sample reference"+" "+material_id[matid]
            fig = plt.figure()
            fig.suptitle(title, fontsize=10)
            axs = fig.subplots(2, 3)
            axs[0, 0].set_title(r"$\epsilon_{11}$ (%)", loc='center', fontsize=8)
            logdata = e11s #np.log(e11c)
            xmin = logdata.min()
            xmax = logdata.max()
            x1 = np.linspace(xmin, xmax, 1000)
            estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
            pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
            axs[0, 0].axvline(x=estimated_mu, c="k")
            axs[0, 0].plot(x1, pdf, 'r')
            axs[0, 0].hist(logdata, bins=bins, density=True, alpha=0.8)
            # axs[0, 0].hist(e11s, bins=bins)
            axs[0, 0].set_ylabel('Frequency', fontsize=8)
            axs[0, 0].tick_params(axis='both', which='major', labelsize=8)
            axs[0, 0].tick_params(axis='both', which='minor', labelsize=8)
            
            mu_sd.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
            
            axs[0, 1].set_title(r"$\epsilon_{22}$ (%)", loc='center', fontsize=8)
            logdata = e22s #np.log(e22c)
            xmin = logdata.min()
            xmax = logdata.max()
            x1 = np.linspace(xmin, xmax, 1000)
            estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
            pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
            axs[0, 1].axvline(x=estimated_mu, c="k")
            axs[0, 1].plot(x1, pdf, 'r')
            axs[0, 1].hist(logdata, bins=bins, density=True, alpha=0.8)
            # axs[0, 1].hist(e22s, bins=bins)
            axs[0, 1].set_ylabel('Frequency', fontsize=8)
            axs[0, 1].tick_params(axis='both', which='major', labelsize=8)
            axs[0, 1].tick_params(axis='both', which='minor', labelsize=8)
            
            mu_sd.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
            
            axs[0, 2].set_title(r"$\epsilon_{33}$ (%)", loc='center', fontsize=8)
            logdata = e33s #np.log(e33c)
            xmin = logdata.min()
            xmax = logdata.max()
            x1 = np.linspace(xmin, xmax, 1000)
            estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
            pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
            axs[0, 2].axvline(x=estimated_mu, c="k")
            axs[0, 2].plot(x1, pdf, 'r')
            axs[0, 2].hist(logdata, bins=bins, density=True, alpha=0.8)
            # axs[0, 2].hist(e33s, bins=bins)
            axs[0, 2].set_ylabel('Frequency', fontsize=8)
            axs[0, 2].tick_params(axis='both', which='major', labelsize=8)
            axs[0, 2].tick_params(axis='both', which='minor', labelsize=8)
            
            mu_sd.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
            
            axs[1, 0].set_title(r"$\epsilon_{12}$ (%)", loc='center', fontsize=8)
            logdata = e12s#np.log(e12c)
            xmin = logdata.min()
            xmax = logdata.max()
            x1 = np.linspace(xmin, xmax, 1000)
            estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
            pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
            axs[1, 0].axvline(x=estimated_mu, c="k")
            axs[1, 0].plot(x1, pdf, 'r')
            axs[1, 0].hist(logdata, bins=bins, density=True, alpha=0.8)
            # axs[1, 0].hist(e12s, bins=bins)
            axs[1, 0].set_ylabel('Frequency', fontsize=8)
            axs[1, 0].tick_params(axis='both', which='major', labelsize=8)
            axs[1, 0].tick_params(axis='both', which='minor', labelsize=8)
            
            mu_sd.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
            
            axs[1, 1].set_title(r"$\epsilon_{13}$ (%)", loc='center', fontsize=8)
            logdata = e13s#np.log(e13c)
            xmin = logdata.min()
            xmax = logdata.max()
            x1 = np.linspace(xmin, xmax, 1000)
            estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
            pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
            axs[1, 1].axvline(x=estimated_mu, c="k")
            axs[1, 1].plot(x1, pdf, 'r')
            axs[1, 1].hist(logdata, bins=bins, density=True, alpha=0.8)
            # axs[1, 1].hist(e13s, bins=bins)
            axs[1, 1].set_ylabel('Frequency', fontsize=8)
            axs[1, 1].tick_params(axis='both', which='major', labelsize=8)
            axs[1, 1].tick_params(axis='both', which='minor', labelsize=8)
            
            mu_sd.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
            
            axs[1, 2].set_title(r"$\epsilon_{23}$ (%)", loc='center', fontsize=8)
            logdata = e23s#np.log(e23c)
            xmin = logdata.min()
            xmax = logdata.max()
            x1 = np.linspace(xmin, xmax, 1000)
            estimated_mu, estimated_sigma = scipy.stats.norm.fit(logdata)
            pdf = scipy.stats.norm.pdf(x1, loc=estimated_mu, scale=estimated_sigma)
            axs[1, 2].axvline(x=estimated_mu, c="k")
            axs[1, 2].plot(x1, pdf, 'r')
            axs[1, 2].hist(logdata, bins=bins, density=True, alpha=0.8)
            # axs[1, 2].hist(e23s, bins=bins)
            axs[1, 2].set_ylabel('Frequency', fontsize=8)
            axs[1, 2].tick_params(axis='both', which='major', labelsize=8)
            axs[1, 2].tick_params(axis='both', which='minor', labelsize=8)
            
            mu_sd.append((estimated_mu-estimated_sigma, estimated_mu+estimated_sigma))
            
            plt.tight_layout()
            plt.savefig(model_direc+ "//"+title+'.png', format='png', dpi=1000) 
            plt.close(fig)  
        except:
            pass

    for matid in range(len(material_)):
        for index in range(len(strain_matrix)):
            nan_index1 = np.where(match_rate[index][0] <= match_rate_threshold)[0]
            mat_id_index = np.where(mat_global[index][0] != matid+1)[0]
            nan_index = np.hstack((mat_id_index,nan_index1))
            nan_index = np.unique(nan_index)
        
            strain_matrix_plot = np.copy(strain_matrixs[index][0])
            strain_matrix_plot[nan_index,:,:] = np.nan             
        
            fig = plt.figure(figsize=(11.69,8.27), dpi=100)
            bottom, top = 0.1, 0.9
            left, right = 0.1, 0.8
            fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
            
            try:
                vmin, vmax = mu_sd[matid*6]
                axs = fig.subplots(2, 3)
                axs[0, 0].set_title(r"$\epsilon_{11}$ (%)", loc='center', fontsize=8)
                im=axs[0, 0].imshow(strain_matrix_plot[:,0,0].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                axs[0, 0].set_xticks([])
                axs[0, 0].set_yticks([])
                divider = make_axes_locatable(axs[0,0])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
                
                vmin, vmax = mu_sd[matid*6+1]
                axs[0, 1].set_title(r"$\epsilon_{22}$ (%)", loc='center', fontsize=8)
                im=axs[0, 1].imshow(strain_matrix_plot[:,1,1].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                divider = make_axes_locatable(axs[0,1])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
                
                vmin, vmax = mu_sd[matid*6+2]
                axs[0, 2].set_title(r"$\epsilon_{33}$ (%)", loc='center', fontsize=8)
                im=axs[0, 2].imshow(strain_matrix_plot[:,2,2].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                divider = make_axes_locatable(axs[0,2])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
                
                vmin, vmax = mu_sd[matid*6+3]
                axs[1, 0].set_title(r"$\epsilon_{12}$ (%)", loc='center', fontsize=8)
                im=axs[1, 0].imshow(strain_matrix_plot[:,0,1].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                axs[1, 0].set_xticks([])
                axs[1, 0].set_yticks([])
                divider = make_axes_locatable(axs[1,0])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
                
                vmin, vmax = mu_sd[matid*6+4]
                axs[1, 1].set_title(r"$\epsilon_{13}$ (%)", loc='center', fontsize=8)
                im=axs[1, 1].imshow(strain_matrix_plot[:,0,2].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                axs[1, 1].set_xticks([])
                divider = make_axes_locatable(axs[1,1])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
                
                vmin, vmax = mu_sd[matid*6+5]
                axs[1, 2].set_title(r"$\epsilon_{23}$ (%)", loc='center', fontsize=8)
                im = axs[1, 2].imshow(strain_matrix_plot[:,1,2].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                axs[1, 2].set_xticks([]) 
                divider = make_axes_locatable(axs[1,2])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
            
                for ax in axs.flat:
                    ax.label_outer()
            
                plt.savefig(model_direc+ '//figure_strain_UBsample_mat'+str(matid)+"_UB"+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
                plt.close(fig)
            except:
                print("Error in strain plot")
            
                
            strain_matrix_plot = np.copy(strain_matrix[index][0])
            strain_matrix_plot[nan_index,:,:] = np.nan             
            
            try:
                fig = plt.figure(figsize=(11.69,8.27), dpi=100)
                bottom, top = 0.1, 0.9
                left, right = 0.1, 0.8
                fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
                
                vmin, vmax = mu_sdc[matid*6]
                axs = fig.subplots(2, 3)
                axs[0, 0].set_title(r"$\epsilon_{11}$ (%)", loc='center', fontsize=8)
                im=axs[0, 0].imshow(strain_matrix_plot[:,0,0].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                axs[0, 0].set_xticks([])
                axs[0, 0].set_yticks([])
                divider = make_axes_locatable(axs[0,0])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
                
                vmin, vmax = mu_sdc[matid*6+1]
                axs[0, 1].set_title(r"$\epsilon_{22}$ (%)", loc='center', fontsize=8)
                im=axs[0, 1].imshow(strain_matrix_plot[:,1,1].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                divider = make_axes_locatable(axs[0,1])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
                
                vmin, vmax = mu_sdc[matid*6+2]
                axs[0, 2].set_title(r"$\epsilon_{33}$ (%)", loc='center', fontsize=8)
                im=axs[0, 2].imshow(strain_matrix_plot[:,2,2].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                divider = make_axes_locatable(axs[0,2])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
                
                vmin, vmax = mu_sdc[matid*6+3]
                axs[1, 0].set_title(r"$\epsilon_{12}$ (%)", loc='center', fontsize=8)
                im=axs[1, 0].imshow(strain_matrix_plot[:,0,1].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                axs[1, 0].set_xticks([])
                axs[1, 0].set_yticks([])
                divider = make_axes_locatable(axs[1,0])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
                
                vmin, vmax = mu_sdc[matid*6+4]
                axs[1, 1].set_title(r"$\epsilon_{13}$ (%)", loc='center', fontsize=8)
                im=axs[1, 1].imshow(strain_matrix_plot[:,0,2].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                axs[1, 1].set_xticks([])
                divider = make_axes_locatable(axs[1,1])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
                
                vmin, vmax = mu_sdc[matid*6+5]
                axs[1, 2].set_title(r"$\epsilon_{23}$ (%)", loc='center', fontsize=8)
                im = axs[1, 2].imshow(strain_matrix_plot[:,1,2].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                axs[1, 2].set_xticks([]) 
                divider = make_axes_locatable(axs[1,2])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
            
                for ax in axs.flat:
                    ax.label_outer()
            
                plt.savefig(model_direc+ '//figure_strain_UBcrystal_mat'+str(matid)+"_UB"+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
                plt.close(fig)
            except:
                print("Error in strain plots")
                
            col_plot = np.copy(col[index][0])
            col_plot[nan_index,:] = np.nan,np.nan,np.nan
            col_plot = col_plot.reshape((lim_x, lim_y, 3))
        
            colx_plot = np.copy(colx[index][0])
            colx_plot[nan_index,:] = np.nan,np.nan,np.nan
            colx_plot = colx_plot.reshape((lim_x, lim_y,3))
            
            coly_plot = np.copy(coly[index][0])
            coly_plot[nan_index,:] = np.nan,np.nan,np.nan
            coly_plot = coly_plot.reshape((lim_x, lim_y,3))
            
            try:
                fig = plt.figure(figsize=(11.69,8.27), dpi=100)
                bottom, top = 0.1, 0.9
                left, right = 0.1, 0.8
                fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
                
                axs = fig.subplots(1, 3)
                axs[0].set_title(r"IPF Z map", loc='center', fontsize=8)
                axs[0].imshow(col_plot, origin='lower')
                axs[0].set_xticks([])
                axs[0].set_yticks([])
                
                axs[1].set_title(r"IPF Y map", loc='center', fontsize=8)
                axs[1].imshow(coly_plot, origin='lower')
                axs[1].set_xticks([])
                axs[1].set_yticks([])
                
                axs[2].set_title(r"IPF X map", loc='center', fontsize=8)
                im = axs[2].imshow(colx_plot, origin='lower')
                axs[2].set_xticks([])
                axs[2].set_yticks([])
            
                for ax in axs.flat:
                    ax.label_outer()
            
                plt.savefig(model_direc+ '//IPF_map_mat'+str(matid)+"_UB"+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
                plt.close(fig)

                col_plot = np.copy(col[index][0])
                col_plot[nan_index,:] = np.nan,np.nan,np.nan
                col_plot = col_plot.reshape((lim_x, lim_y, 3))
            
                mr_plot = np.copy(match_rate[index][0])
                mr_plot[nan_index,:] = np.nan
                mr_plot = mr_plot.reshape((lim_x, lim_y))
                
                mat_glob = np.copy(mat_global[index][0])
                mat_glob[nan_index,:] = np.nan
                mat_glob = mat_glob.reshape((lim_x, lim_y))
                
                fig = plt.figure(figsize=(11.69,8.27), dpi=100)
                bottom, top = 0.1, 0.9
                left, right = 0.1, 0.8
                fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
            
                axs = fig.subplots(1, 3)
                axs[0].set_title(r"IPF Z map", loc='center', fontsize=8)
                axs[0].imshow(col_plot, origin='lower')
                axs[0].set_xticks([])
                axs[0].set_yticks([])
                
                axs[1].set_title(r"Material Index", loc='center', fontsize=8)
                im = axs[1].imshow(mat_glob, origin='lower', vmin=0, vmax=2)
                axs[1].set_xticks([])
                axs[1].set_yticks([])
                
                divider = make_axes_locatable(axs[1])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')
                
                axs[2].set_title(r"Matching rate", loc='center', fontsize=8)
                im = axs[2].imshow(mr_plot, origin='lower', cmap=plt.cm.jet, vmin=0, vmax=100)
                axs[2].set_xticks([])
                axs[2].set_yticks([])
                
                divider = make_axes_locatable(axs[2])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')
            
                for ax in axs.flat:
                    ax.label_outer()
            
                plt.savefig(model_direc+ "//figure_global_mat"+str(matid)+"_UB"+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
                plt.close(fig)
            except:
                print("Error in plots")
            spots_len_plot = np.copy(spots_len[index][0])
            spots_len_plot[nan_index,:] = np.nan
            spots_len_plot = spots_len_plot.reshape((lim_x, lim_y))
            
            iR_pix_plot = np.copy(iR_pix[index][0])
            iR_pix_plot[nan_index,:] = np.nan
            iR_pix_plot = iR_pix_plot.reshape((lim_x, lim_y))
            
            fR_pix_plot = np.copy(fR_pix[index][0])
            fR_pix_plot[nan_index,:] = np.nan
            fR_pix_plot = fR_pix_plot.reshape((lim_x, lim_y))
            
            try:
                fig = plt.figure(figsize=(11.69,8.27), dpi=100)
                bottom, top = 0.1, 0.9
                left, right = 0.1, 0.8
                fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
            
                axs = fig.subplots(1, 3)
                axs[0].set_title(r"Number of spots detected", loc='center', fontsize=8)
                im = axs[0].imshow(spots_len_plot, origin='lower', cmap=plt.cm.jet)
                axs[0].set_xticks([])
                axs[0].set_yticks([])
                
                divider = make_axes_locatable(axs[0])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')
            
                axs[1].set_title(r"Initial pixel residues", loc='center', fontsize=8)
                im = axs[1].imshow(iR_pix_plot, origin='lower', cmap=plt.cm.jet)
                axs[1].set_xticks([])
                axs[1].set_yticks([])
                
                divider = make_axes_locatable(axs[1])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')
            
                axs[2].set_title(r"Final pixel residues", loc='center', fontsize=8)
                im = axs[2].imshow(fR_pix_plot, origin='lower', cmap=plt.cm.jet)
                axs[2].set_xticks([])
                axs[2].set_yticks([])
                
                divider = make_axes_locatable(axs[2])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')
            
                for ax in axs.flat:
                    ax.label_outer()
            
                plt.savefig(model_direc+'//figure_mr_ir_fr_mat'+str(matid)+"_UB"+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
                plt.close(fig)
            except:
                print("Error in plots")
                
            try:
                a,b,c,alp,bet,gam = [],[],[],[],[],[]
                
                constantlength = "a"
                if ("a" in strain_free_parameters) and ("b" in strain_free_parameters) and ("c" in strain_free_parameters):
                    constantlength = "a"                    
                elif ("b" not in strain_free_parameters) and additional_expression[0]=="none" and\
                    "b" not in additional_expression[0]:
                    constantlength = "b"
                elif ("c" not in strain_free_parameters):
                    constantlength = "c"
                    
                for irot in range(len(rotation_matrix1[index][0])):
                    lattice_parameter_direct_strain = CP.computeLatticeParameters_from_UB(rotation_matrix1[index][0][irot,:,:], 
                                                                                          material_, 
                                                                                          constantlength, 
                                                                                          dictmaterials=dictLT.dict_Materials)
                    a.append(lattice_parameter_direct_strain[0])
                    b.append(lattice_parameter_direct_strain[1])
                    c.append(lattice_parameter_direct_strain[2])
                    alp.append(lattice_parameter_direct_strain[3])
                    bet.append(lattice_parameter_direct_strain[4])
                    gam.append(lattice_parameter_direct_strain[5])
                
                logdata = np.array(a)
                logdata = logdata[~np.isnan(logdata)]
                rangemina, rangemaxa = np.min(logdata)-0.01, np.max(logdata)+0.01
                logdata = np.array(b)
                logdata = logdata[~np.isnan(logdata)]
                rangeminb, rangemaxb = np.min(logdata)-0.01, np.max(logdata)+0.01
                logdata = np.array(c)
                logdata = logdata[~np.isnan(logdata)]
                rangeminc, rangemaxc = np.min(logdata)-0.01, np.max(logdata)+0.01
                logdata = np.array(alp)
                logdata = logdata[~np.isnan(logdata)]
                rangeminal, rangemaxal = np.min(logdata)-0.01, np.max(logdata)+0.01
                logdata = np.array(bet)
                logdata = logdata[~np.isnan(logdata)]
                rangeminbe, rangemaxbe = np.min(logdata)-0.01, np.max(logdata)+0.01
                logdata = np.array(gam)
                logdata = logdata[~np.isnan(logdata)]
                rangeminga, rangemaxga = np.min(logdata)-0.01, np.max(logdata)+0.01
        
                fig = plt.figure(figsize=(11.69,8.27), dpi=100)
                bottom, top = 0.1, 0.9
                left, right = 0.1, 0.8
                fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
                
                vmin = rangemina
                vmax = rangemaxa
                axs = fig.subplots(2, 3)
                axs[0, 0].set_title(r"$a$", loc='center', fontsize=8)
                strain_matrix_plot = np.array(a)
                im=axs[0, 0].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                axs[0, 0].set_xticks([])
                axs[0, 0].set_yticks([])
                divider = make_axes_locatable(axs[0,0])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
        
                vmin = rangeminb
                vmax = rangemaxb
                axs[0, 1].set_title(r"$b$", loc='center', fontsize=8)
                strain_matrix_plot = np.array(b)
                im=axs[0, 1].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                divider = make_axes_locatable(axs[0,1])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
        
                vmin = rangeminc
                vmax = rangemaxc
                axs[0, 2].set_title(r"$c$", loc='center', fontsize=8)
                strain_matrix_plot = np.array(c)
                im=axs[0, 2].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                divider = make_axes_locatable(axs[0,2])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
        
                vmin = rangeminal
                vmax = rangemaxal
                axs[1, 0].set_title(r"$\alpha$", loc='center', fontsize=8)
                strain_matrix_plot = np.array(alp)
                im=axs[1, 0].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                axs[1, 0].set_xticks([])
                axs[1, 0].set_yticks([])
                divider = make_axes_locatable(axs[1,0])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
        
                vmin = rangeminbe
                vmax = rangemaxbe
                axs[1, 1].set_title(r"$\beta$", loc='center', fontsize=8)
                strain_matrix_plot = np.array(bet)
                im=axs[1, 1].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                axs[1, 1].set_xticks([])
                divider = make_axes_locatable(axs[1,1])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
        
                vmin = rangeminga
                vmax = rangemaxga
                axs[1, 2].set_title(r"$\gamma$", loc='center', fontsize=8)
                strain_matrix_plot = np.array(gam)
                im = axs[1, 2].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                axs[1, 2].set_xticks([]) 
                divider = make_axes_locatable(axs[1,2])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.formatter.set_useOffset(False)
                cbar.ax.tick_params(labelsize=8) 
                
                for ax in axs.flat:
                    ax.label_outer()
                plt.savefig(model_direc+ "//"+'figure_unitcell_'+str(matid)+'_'+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
                plt.close(fig)
            except:
                pass
            
            try:
                latticeparams = dictLT.dict_Materials[material_][1]

                a,b,c,alp,bet,gam = [],[],[],[],[],[]
        
                constantlength = "a"
                if ("a" in strain_free_parameters) and ("b" in strain_free_parameters) and ("c" in strain_free_parameters):
                    constantlength = "a"    
                elif ("b" not in strain_free_parameters) and additional_expression[0]=="none" and \
                    "b" not in additional_expression[0]:
                    constantlength = "b"
                elif ("c" not in strain_free_parameters):
                    constantlength = "c"
                    
                for irot in range(len(rotation_matrix1[index][0])):
                    lattice_parameter_direct_strain = CP.computeLatticeParameters_from_UB(rotation_matrix1[index][0][irot,:,:], 
                                                                                          material_, 
                                                                                          constantlength, 
                                                                                          dictmaterials=dictLT.dict_Materials)
                    a.append(lattice_parameter_direct_strain[0])
                    b.append(lattice_parameter_direct_strain[1])
                    c.append(lattice_parameter_direct_strain[2])
                    alp.append(lattice_parameter_direct_strain[3])
                    bet.append(lattice_parameter_direct_strain[4])
                    gam.append(lattice_parameter_direct_strain[5])
        
                logdata = np.array(a) - latticeparams[0]
                logdata = logdata[~np.isnan(logdata)]
                rangemina, rangemaxa = np.min(logdata) - 0.01e-2, np.max(logdata) + 0.01e-2
                logdata = np.array(b) - latticeparams[1]
                logdata = logdata[~np.isnan(logdata)]
                rangeminb, rangemaxb = np.min(logdata) - 0.01e-2, np.max(logdata) + 0.01e-2
                logdata = np.array(c) - latticeparams[2]
                logdata = logdata[~np.isnan(logdata)]
                rangeminc, rangemaxc = np.min(logdata) - 0.01e-2, np.max(logdata) + 0.01e-2
                logdata = np.array(alp) - latticeparams[3]
                logdata = logdata[~np.isnan(logdata)]
                rangeminal, rangemaxal = np.min(logdata) - 0.01, np.max(logdata) + 0.01
                logdata = np.array(bet) - latticeparams[4]
                logdata = logdata[~np.isnan(logdata)]
                rangeminbe, rangemaxbe = np.min(logdata) - 0.01, np.max(logdata) + 0.01
                logdata = np.array(gam) - latticeparams[5]
                logdata = logdata[~np.isnan(logdata)]
                rangeminga, rangemaxga = np.min(logdata) - 0.01, np.max(logdata) + 0.01
        
                fig = plt.figure(figsize=(11.69,8.27), dpi=100)
                bottom, top = 0.1, 0.9
                left, right = 0.1, 0.8
                fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
        
                vmin = rangemina
                vmax = rangemaxa
                axs = fig.subplots(2, 3)
                axs[0, 0].set_title(r"$a$", loc='center', fontsize=8)
                strain_matrix_plot = np.array(a) - latticeparams[0]
                im=axs[0, 0].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                axs[0, 0].set_xticks([])
                axs[0, 0].set_yticks([])
                divider = make_axes_locatable(axs[0,0])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
        
                vmin = rangeminb
                vmax = rangemaxb
                axs[0, 1].set_title(r"$b$", loc='center', fontsize=8)
                strain_matrix_plot = np.array(b) - latticeparams[1]
                im=axs[0, 1].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                divider = make_axes_locatable(axs[0,1])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
        
                vmin = rangeminc
                vmax = rangemaxc
                axs[0, 2].set_title(r"$c$", loc='center', fontsize=8)
                strain_matrix_plot = np.array(c) - latticeparams[2]
                im=axs[0, 2].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                divider = make_axes_locatable(axs[0,2])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
        
                vmin = rangeminal
                vmax = rangemaxal
                axs[1, 0].set_title(r"$\alpha$", loc='center', fontsize=8)
                strain_matrix_plot = np.array(alp) - latticeparams[3]
                im=axs[1, 0].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                axs[1, 0].set_xticks([])
                axs[1, 0].set_yticks([])
                divider = make_axes_locatable(axs[1,0])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
        
                vmin = rangeminbe
                vmax = rangemaxbe
                axs[1, 1].set_title(r"$\beta$", loc='center', fontsize=8)
                strain_matrix_plot = np.array(bet) - latticeparams[4]
                im=axs[1, 1].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                axs[1, 1].set_xticks([])
                divider = make_axes_locatable(axs[1,1])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8) 
        
                vmin = rangeminga
                vmax = rangemaxga
                axs[1, 2].set_title(r"$\gamma$", loc='center', fontsize=8)
                strain_matrix_plot = np.array(gam) - latticeparams[5]
                im = axs[1, 2].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                axs[1, 2].set_xticks([]) 
                divider = make_axes_locatable(axs[1,2])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.formatter.set_useOffset(False)
                cbar.ax.tick_params(labelsize=8) 
        
                for ax in axs.flat:
                    ax.label_outer()
                plt.savefig(model_direc + "//" + 'figure_unitcell_relative_'+str(matid)+'_'+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
                plt.close(fig)
            except:
                pass

# =============================================================================
# For 2D histogram sampling 
# =============================================================================
class WalkerRandomSampling(object):
    """Walker's alias method for random objects with different probablities.
    Based on the implementation of Denis Bzowy at the following URL:
    http://code.activestate.com/recipes/576564-walkers-alias-method-for-random-objects-with-diffe/
    """
    def __init__(self, weights, keys=None):
        """Builds the Walker tables ``prob`` and ``inx`` for calls to `random()`.
        The weights (a list or tuple or iterable) can be in any order and they
        do not even have to sum to 1."""
        n = self.n = len(weights)
        if keys is None:
            self.keys = keys
        else:
            self.keys = np.array(keys)

        if isinstance(weights, (list, tuple)):
            weights = np.array(weights, dtype=float)
        elif isinstance(weights, np.ndarray):
            if weights.dtype != float:
                weights = weights.astype(float)
        else:
            weights = np.array(list(weights), dtype=float)

        if weights.ndim != 1:
            raise ValueError("weights must be a vector")

        weights = weights * n / weights.sum()

        inx = -np.ones(n, dtype=int)
        short = np.where(weights < 1)[0].tolist()
        long = np.where(weights > 1)[0].tolist()
        while short and long:
            j = short.pop()
            k = long[-1]

            inx[j] = k
            weights[k] -= (1 - weights[j])
            if weights[k] < 1:
                short.append( k )
                long.pop()

        self.prob = weights
        self.inx = inx

    def random(self, count=None):
        """Returns a given number of random integers or keys, with probabilities
        being proportional to the weights supplied in the constructor.
        When `count` is ``None``, returns a single integer or key, otherwise
        returns a NumPy array with a length given in `count`.
        """
        if count is None:
            u = np.random.random()
            j = np.random.randint(self.n)
            k = j if u <= self.prob[j] else self.inx[j]
            return self.keys[k] if self.keys is not None else k

        u = np.random.random(count)
        j = np.random.randint(self.n, size=count)
        k = np.where(u <= self.prob[j], j, self.inx[j])
        return self.keys[k] if self.keys is not None else k
