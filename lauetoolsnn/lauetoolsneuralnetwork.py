# -*- coding: utf-8 -*-
"""
Created on June 18 06:54:04 2021
GUI routine for Laue neural network training and prediction

@author: Ravi raj purohit PURUSHOTTAM RAJ PUROHIT (purushot@esrf.fr)
@guide: jean-Sebastien MICHA (micha@esrf.fr)

Credits:
Lattice and symmetry routines are extracted and adapted from the PYMICRO and Xrayutilities repository

TODO:
    1. Dynamic Multi processing variables ? COMPLICATED
    2. Structure factor calculation with xrayutilities
    
    # Optional TODO
    # Write a function that looks for pixels with no indexation having atleast 6 neighbors indexed
    # idea is to index with their rotation matrix ?
    # Also write a function to rearrange matricies of each pixel to have complete grain representation
    # Auto save data
    # extract average UB from the list of UBs for a given MR threshold and run again the analysis
    # calculate similarity between top 100 intese peaks in similarity algorithm
    # zoom without changing the image reshape
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

try:
    import pkg_resources  # part of setuptools
    version_package = pkg_resources.require("lauetoolsnn")[0].version
except:
    version_package = "3.0.0"

frame_title = "Laue Neural-Network model- v3 @Ravi @Jean-Sebastien \n@author: Ravi raj purohit PURUSHOTTAM RAJ PUROHIT (purushot@esrf.fr) \n@guide: Jean-Sebastien MICHA (micha@esrf.fr)"

import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams.update({'font.size': 14})
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from matplotlib import cm

import numpy as np
import itertools
import re
import glob
import _pickle as cPickle
import time, datetime
import sys
import inspect
import threading
import multiprocessing as multip
from multiprocessing import Process, Queue, cpu_count
import ast, configparser
from sklearn.metrics import classification_report
from skimage.transform import (hough_line, hough_line_peaks)

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QSettings, QTimer
from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtWidgets import QApplication, QMainWindow,\
                            QPushButton, QWidget, QFormLayout, \
                            QToolBar, QStatusBar, QSlider, \
                            QVBoxLayout, QTextEdit, QProgressBar, \
                            QComboBox, QLineEdit, QFileDialog, QMenuBar,QScrollArea, QSplashScreen
from PyQt5.Qt import QUrl, QDesktopServices
## for faster binning of histogram
## C version of hist (Relace by Numpy histogram)
# from fast_histogram import histogram1d
## Keras import
tensorflow_keras = True
try:
    import tensorflow as tf
    from keras.models import model_from_json
    from keras.callbacks import EarlyStopping, ModelCheckpoint
except:
    tensorflow_keras = False

## util library with MP function
try:
    from lauetoolsnn.utils_lauenn import Symmetry,Lattice,\
        simulatemultiplepatterns, worker_generation, chunker_list,call_global,\
        get_ipf_colour,predict_ubmatrix,\
        predict_preprocessMP, global_plots, texttstr1, get_material_data,\
        write_training_testing_dataMTEX, SGLattice, simulate_spots, mse_images, \
        generate_classHKL, rmv_freq_class, array_generator, vali_array, array_generator_verify,\
        worker, predict_preprocessMP_vsingle,\
        computeGnomonicImage, OrientationMatrix2Euler #save_sst
except:
    from utils_lauenn import Symmetry,Lattice,\
        simulatemultiplepatterns, worker_generation, chunker_list,call_global,\
        get_ipf_colour,predict_ubmatrix,\
        predict_preprocessMP, global_plots, texttstr1, get_material_data,\
        write_training_testing_dataMTEX, SGLattice, simulate_spots, mse_images, \
        generate_classHKL, rmv_freq_class, array_generator, vali_array, array_generator_verify,\
        worker, predict_preprocessMP_vsingle,\
        computeGnomonicImage, OrientationMatrix2Euler #save_sst

try:
    from lauetoolsnn.NNmodels import read_hdf5, model_arch_general, \
                                    model_arch_CNN_DNN_optimized, user_defined_model, \
                                    LoggingCallback, predict_DNN    
except:
    from NNmodels import read_hdf5, model_arch_general, \
                            model_arch_CNN_DNN_optimized, user_defined_model, \
                            LoggingCallback, predict_DNN

try:
    import lauetoolsnn.lauetools.dict_LaueTools as dictLT
    import lauetoolsnn.lauetools.IOLaueTools as IOLT
    import lauetoolsnn.lauetools.generaltools as GT
    import lauetoolsnn.lauetools.LaueGeometry as Lgeo
    import lauetoolsnn.lauetools.readmccd as RMCCD
    import lauetoolsnn.lauetools.IOimagefile as IOimage
    import lauetoolsnn.lauetools.imageprocessing as ImProc
except:
    from lauetools import dict_LaueTools as dictLT
    from lauetools import IOLaueTools as IOLT
    from lauetools import generaltools as GT
    from lauetools import LaueGeometry as Lgeo
    from lauetools import readmccd as RMCCD
    from lauetools import IOimagefile as IOimage
    from lauetools import imageprocessing as ImProc
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

Logo = resource_path("lauetoolsnn_logo_bXM_2.png",  verbose=0)
Logo_splash = resource_path("lauetoolsnn_splash_bXM_2.png",  verbose=0)
gif_path = resource_path("frames_medres.gif",  verbose=0)

default_initialization = True
if default_initialization:
    material_global = "GaN" ## same key as used in LaueTools
    symmetry_global = "hexagonal"
    material1_global = "Si" ## same key as used in LaueTools
    symmetry1_global = "cubic"
    prefix_global = ""
    detectorparameters_global = [79.50900, 977.9000, 931.8900, 0.3570000, 0.4370000]
    pixelsize_global = 0.0734 # 0.079142 #
    ccd_label_global = "sCMOS" #"MARCCD165" #"Cor"#
    dim1_global = 2018 #2048 #
    dim2_global = 2016 #2048 #
    emax_global = 22
    emin_global = 5
    UB_matrix_global = 1
    image_grid_globalx = 10
    image_grid_globaly = 10
    intensity_threshold_global = 5 #75 800
    boxsize_global = 10
    fit_peaks_gaussian_global = 1
    FitPixelDev_global = 15
    strain_label_global = "YES" ## compute and plot strains
    tolerance_strain = [0.5,0.4,0.3,0.2]   ## reduced tolerance for strain calculations
    tolerance_strain1 = [0.5,0.4,0.3,0.2]
    hkls_list_global = "[1,1,0],[1,0,0],[1,1,1]"#,[3,1,0],[5,2,9],[7,5,7],[7,5,9]"
    ##exp directory
    if material_global == material1_global:
        fn1 = material_global + prefix_global
    else:
        fn1 = material_global + "_" + material1_global + prefix_global
    expfile_global = None #r"C:\Users\purushot\Desktop\Tungsten_olivier_data\d0-300MPa"
    exp_prefix_global = None #"Wmap_WB_13sep_d0_300MPa_" #"nw2_" #None #"roi3_" #
    modelfile_global = resource_path("models",  verbose=0) + "//" + fn1
    if material_global == material1_global:
        fn1 = material_global
        if exp_prefix_global == None:
            exp_prefix_global = material_global + "_"
        weightfile_global = modelfile_global + "//" + "model_" + material_global + ".h5"
    else:
        fn1  = material_global + "_" + material1_global
        if exp_prefix_global == None:
            exp_prefix_global = material_global + "_"+material1_global + "_"
        weightfile_global = modelfile_global + "//" + "model_" + material_global + "_" + material1_global + ".h5"
    main_directory = resource_path("models",  verbose=0)
    hkl_max_global = "5"
    elements_global = "all"
    freq_rmv_global = 100
    hkl_max1_global = "5"
    elements1_global = "all"
    freq_rmv1_global = 100
    maximum_angle_to_search_global = 120
    step_for_binning_global = 0.1
    nb_grains_per_lp_global = 2
    nb_grains_per_lp1_global = 2
    grains_nb_simulate_global = 1000
    include_scm_global = False
    batch_size_global = 50
    epochs_global = 5
    tolerance_global = 0.5
    tolerance_global1 = 0.5
    model_weight_file = None
    softmax_threshold_global = 0.80 # softmax_threshold
    mr_threshold_global = 0.95 # match rate threshold
    cap_matchrate = 0.05 * 100 ## any UB matrix providing MR less than this will be ignored
    coeff = 0.20 ## should be same as cap_matchrate or no?
    coeff_overlap1212 = 0.2 ##15% spots overlap to avoid bad orientation detection
    NumberMaxofFits = 5000 ### Max peaks per LP
    mode_spotCycle = "graphmode" ## slow: to cycle through all spots else: cycles through smartly selected pair of spots
    material0_limit1212 = 100000
    material1_limit1212 = 100000
    use_previous_UBmatrix = False
    write_mtex_file = True
    misorientation_angle1 = 1
    cpu_count_user = -1
    strain_free_parameters = ["rotx", "roty", "rotz", "alpha", "beta", "gamma", "b", "c"]
    additional_expression = ["none"]
    mode_peaksearch = "skimage"
    
    try:
        if symmetry_global =="cubic":
            material0_lauegroup = "11"
        elif symmetry_global =="monoclinic":
            material0_lauegroup = "2"
        elif symmetry_global == "hexagonal":
            material0_lauegroup = "9"
        elif symmetry_global == "orthorhombic":
            material0_lauegroup = "3"
        elif symmetry_global == "tetragonal":
            material0_lauegroup = "5"
        elif symmetry_global == "trigonal":
            material0_lauegroup = "7"
        elif symmetry_global == "triclinic":
            material0_lauegroup = "1"
    except:
        material0_lauegroup = "11"
        
    try:
        if symmetry1_global =="cubic":
            material1_lauegroup = "11"
        elif symmetry1_global =="monoclinic":
            material1_lauegroup = "2"
        elif symmetry1_global == "hexagonal":
            material1_lauegroup = "9"
        elif symmetry1_global == "orthorhombic":
            material1_lauegroup = "3"
        elif symmetry1_global == "tetragonal":
            material1_lauegroup = "5"
        elif symmetry1_global == "trigonal":
            material1_lauegroup = "7"
        elif symmetry1_global == "triclinic":
            material1_lauegroup = "1"
    except:
        material1_lauegroup = "11"
        
if cpu_count_user == -1:
    cpu_count_user = cpu_count()
GUI_START_TIME = time.time() #in ms
ACCEPTABLE_FORMATS = [".npz"]
gui_state = np.random.randint(1e6)

#%% Main module
class Window(QMainWindow):
    """Main Window."""
    def __init__(self, winx=None, winy=None):
        """Initializer."""
        super(Window, self).__init__()
        # QMainWindow.__init__(self)
        self.flashSplash(Logo_splash)
        time.sleep(2)
        app_icon = QtGui.QIcon()
        app_icon.addFile(Logo, QtCore.QSize(64,64))
        self.setWindowIcon(app_icon)
        
        if winx==None or winy==None:
            self.setFixedSize(16777215,16777215)
        else:
            self.setFixedSize(winx, winy)
        
        self.setWindowTitle("Laue Neural-Network v3")
        self._createMenu()
        self._createToolBar()
        self._createStatusBar()
        
        ## init variables
        self.input_params = {}
        self.factor = 5 ## fixed for 20% validation dataset generation
        self.state = 0
        self.state1 = 0
        self.state2 = 0
        self.model = None
        self.mat0_listHKl = None
        self.mat1_listHKl = None
        self.mode_spotCycleglobal = mode_spotCycle
        self.softmax_threshold_global = softmax_threshold_global
        self.mr_threshold_global = mr_threshold_global
        self.cap_matchrate = cap_matchrate
        self.coeff = coeff
        self.coeff_overlap = coeff_overlap1212
        self.fit_peaks_gaussian_global = fit_peaks_gaussian_global
        self.FitPixelDev_global = FitPixelDev_global
        self.NumberMaxofFits = NumberMaxofFits
        self.tolerance_strain = tolerance_strain
        self.tolerance_strain1 = tolerance_strain1
        self.misorientation_angle = misorientation_angle1
        self.material0_limit = material0_limit1212
        self.material1_limit = material1_limit1212
        self.material_phase_always_present = None
        self.matrix_phase_always_present = None
        self.generate_additional_data=False
        self.use_previous_UBmatrix = use_previous_UBmatrix
        self.crystal = None
        self.SG = None
        self.general_diff_rules = False
        self.crystal1 = None
        self.SG1 = None
        self.general_diff_rules1 = False
        self.strain_free_parameters = strain_free_parameters,
        self.additional_expression = additional_expression
        self.architecture = "FFNN"
        # Add box layout, add table to box layout and add box layout to widget
        self.layout = QVBoxLayout()
        self._centralWidget = QWidget(self)
        self.setCentralWidget(self._centralWidget)
        self._centralWidget.setLayout(self.layout)
        self._createDisplay() ## display screen
        self.setDisplayText("Lauetoolsnn v"+ str(version_package))
        self.setDisplayText(frame_title)
        self.setDisplayText("Uses base libraries of LaueTools (micha@esrf.fr) to simulate Laue patterns for a given detector geometry \nFollows convention of BM32 beamline at ESRF")
        self.setDisplayText("Polefigure and IPF plot modules are taken and modified from PYMICRO repository; HKL multiplicity and conditions are taken from xrayutilities library")
        self.setDisplayText("This version supports multiprocessing \nGUI initialized! \nLog will be printed here \nPlease Train a model first, if not already done.\n")
        self.setDisplayText("New materials and extinction rules can be set in LaueTools DictLP file before launching this module")
        self.setDisplayText("For now the Learning rate of optimizer, Kernel and Bias weight Initializers are already optimized and set in the in-built model (can also be set to different values in the config window)"+\
                            " (TO find another set of parameters please use Hyper parameter optimization routine in GUI)")
        self.setDisplayText("Load a config file first (for example see the example_config tab for template)")
        self.setDisplayText("****Testing****")
        self.setDisplayText("First installation ? In order to test all the functionality of the GUI, please run a example case from the example folder")
        self._formLayout() ## buttons and layout
        self.popups = []
        # self.showMaximized()
        self.setFixedSize(16777215,16777215)
        
        config_setting = configparser.ConfigParser()
        filepath = resource_path('settings.ini')
        config_setting.read(filepath)
        config_setting.set('CALLER', 'residues_threshold',str(0.5))
        config_setting.set('CALLER', 'nb_spots_global_threshold',str(8))
        config_setting.set('CALLER', 'option_global',"v2")
        config_setting.set('CALLER', 'use_om_user',"false")
        config_setting.set('CALLER', 'nb_spots_consider',str(500))
        config_setting.set('CALLER', 'path_user_OM',"none")
        config_setting.set('CALLER', 'intensity', str(200))
        config_setting.set('CALLER', 'boxsize', str(15))
        config_setting.set('CALLER', 'pixdev', str(15))
        config_setting.set('CALLER', 'cap_softmax', str(0.85))
        config_setting.set('CALLER', 'cap_mr', str(0.01))
        config_setting.set('CALLER', 'strain_free_parameters', ",".join(strain_free_parameters))
        config_setting.set('CALLER', 'additional_expression', ",".join(additional_expression))
        config_setting.set('CALLER', 'mode_peaksearch', str(mode_peaksearch))
        with open(filepath, 'w') as configfile:
            config_setting.write(configfile)
    
    def flashSplash(self, logo):
        self.splash = QSplashScreen(QPixmap(logo))
        self.splash.show()
        QTimer.singleShot(2000, self.splash.close)
        
    def closeEvent(self, event):
        try:
            self.text_file_log.close()
        except:
            print("Nothing to close")
        self.close
        QApplication.closeAllWindows()
        super().closeEvent(event)
        
    def _createDisplay(self):
        """Create the display."""
        self.display = QTextEdit()
        self.display.setReadOnly(True)
        self.layout.addWidget(self.display)

    def setDisplayText(self, text):
        self.display.append('%s'%text)
        self.display.moveCursor(QtGui.QTextCursor.End)
        self.display.setFocus()

    def _createMenu(self):
        self.menu = self.menuBar().addMenu("&Menu")
        self.menu.addAction('&Load Config', self.getfileConfig)
        self.menu.addAction('&Exit', self.close)
        
        self.menu1 = self.menuBar().addMenu("&Help")
        self.menu1.addAction('&Documentation', self.documentation)
        self.menu1.addAction('&LaueNN concept', self.concept)
        self.menu1.addAction('&LaueNN terminal commands', self.commands_cmd)
    
    def commands_cmd(self,):
        w21 = sample_commands()
        w21.show()
        self.popups.append(w21)

    def documentation(self):
        url = QUrl("https://lauenn.readthedocs.io/en/latest/")
        QDesktopServices.openUrl(url)
    
    def concept(self):
        w21 = gif_window()
        w21.show()
        self.popups.append(w21)
    
    def getfileConfig(self):
        filters = "Config file (*.lauenn)"
        filenameConfig = QFileDialog.getOpenFileName(self, 'Select a lauenn config file with extension lauenn', 
                                                     resource_path("examples"), filters)
        self.load_config_from_file(filenameConfig[0])
    
    def load_config_from_file(self, configFile):
        global material_global, symmetry_global, material1_global, symmetry1_global
        global prefix_global, main_directory, emin_global, emax_global, ccd_label_global
        global detectorparameters_global, pixelsize_global, dim1_global, dim2_global
        global UB_matrix_global, image_grid_globalx , image_grid_globaly 
        global intensity_threshold_global, boxsize_global, fit_peaks_gaussian_global, FitPixelDev_global
        global strain_label_global, tolerance_strain, tolerance_strain1, hkls_list_global
        global expfile_global, exp_prefix_global, modelfile_global, weightfile_global
        global hkl_max_global, elements_global, freq_rmv_global, hkl_max1_global
        global elements1_global, freq_rmv1_global, maximum_angle_to_search_global
        global step_for_binning_global, nb_grains_per_lp_global, nb_grains_per_lp1_global
        global grains_nb_simulate_global, include_scm_global, batch_size_global, epochs_global
        global tolerance_global, model_weight_file, material0_limit1212, material1_limit1212, tolerance_global1
        global softmax_threshold_global, mr_threshold_global, cap_matchrate, coeff, cpu_count_user
        global coeff_overlap1212, mode_spotCycle, NumberMaxofFits, use_previous_UBmatrix
        global write_mtex_file, material0_lauegroup, material1_lauegroup, misorientation_angle1
        global mode_peaksearch
        
        config = configparser.ConfigParser()
        
        try:
            config.read_file(open(configFile))
        except:
            self.write_to_console("File not selected, nothing to open")
            return
            
        material_global = config.get('MATERIAL', 'material')
        symmetry_global = config.get('MATERIAL', 'symmetry')
        
        try:
            self.SG = int(config.get('MATERIAL', 'space_group'))
        except:
            if symmetry_global =="cubic":
                self.SG = 230
            elif symmetry_global =="monoclinic":
                self.SG = 10
            elif symmetry_global == "hexagonal":
                self.SG = 191
            elif symmetry_global == "orthorhombic":
                self.SG = 47
            elif symmetry_global == "tetragonal":
                self.SG = 123
            elif symmetry_global == "trigonal":
                self.SG = 162
            elif symmetry_global == "triclinic":
                self.SG = 2
            self.write_to_console("Space group is not defined, by default taking the higher order space group number for the specified symmetry")
        
        try:
            self.general_diff_rules = config.get('MATERIAL', 'general_diffraction_rules') == "true"
        except:
            self.general_diff_rules = False
            self.write_to_console("general_diffraction_rules is not defined, by default False")
        
        try:
            cpu_count_user = int(config.get('CPU', 'n_cpu'))
            if cpu_count_user <= 0 or cpu_count_user > cpu_count():
                cpu_count_user = cpu_count()
        except:
            cpu_count_user = cpu_count()
        
        try:
            material1_global = config.get('MATERIAL', 'material1')
            symmetry1_global = config.get('MATERIAL', 'symmetry1')
            
            try:
                self.SG1 = int(config.get('MATERIAL', 'space_group1'))
            except:
                if symmetry1_global =="cubic":
                    self.SG1 = 230
                elif symmetry1_global =="monoclinic":
                    self.SG1 = 10
                elif symmetry1_global == "hexagonal":
                    self.SG1 = 191
                elif symmetry1_global == "orthorhombic":
                    self.SG1 = 47
                elif symmetry1_global == "tetragonal":
                    self.SG1 = 123
                elif symmetry1_global == "trigonal":
                    self.SG1 = 162
                elif symmetry1_global == "triclinic":
                    self.SG1 = 2
                self.write_to_console("Space group 1 is not defined, by default taking the higher order space group number for the specified symmetry")
                
            try:
                self.general_diff_rules1 = config.get('MATERIAL', 'general_diffraction_rules1') == "true"
            except:
                self.general_diff_rules1 = False
                self.write_to_console("general_diffraction_rules1 is not defined, by default False")
        except:
            material1_global = "none"
            symmetry1_global = "none"
            self.SG1 = "none"
            self.general_diff_rules1 = False
            self.write_to_console("Only one material is defined, by default taking the other one as 'none'")
        
        if material1_global == "none" and symmetry1_global =="none":
            material1_global = material_global
            symmetry1_global = symmetry_global         
        
        try:
            prefix_global = str(config.get('GLOBAL_DIRECTORY', 'prefix'))
        except:
            prefix_global = ""
            
        try:
            main_directory = str(config.get('GLOBAL_DIRECTORY', 'main_directory'))
        except:
            main_directory = resource_path("models",  verbose=0)
            
        if main_directory == "default":
            main_directory = resource_path("models",  verbose=0)
        
        detectorfile = config.get('DETECTOR', 'detectorfile')
        
        if detectorfile == "ZnCuOCl":
            detectorfile = resource_path("examples//ZnCUOCl//calib.det",  verbose=0)
        elif detectorfile == "GaN":
            detectorfile = resource_path("examples//GaN_Si//calib.det",  verbose=0)
        elif detectorfile == "user_input":
            detectorfile2 = config.get('DETECTOR', 'params').split(",")
            detectorparameters_global = detectorfile2[:5]
            pixelsize_global = detectorfile2[5]
            dim1_global = detectorfile2[6]
            dim2_global = detectorfile2[7]
            ccd_label_global = detectorfile2[8]
        try:
            emax_global = float(config.get('DETECTOR', 'emax'))
            emin_global = float(config.get('DETECTOR', 'emin'))
        except:
            self.write_to_console("Detector energy range not defined, using default values of 5-23KeV")
        
        if detectorfile != "user_input":
            try:
                _file = open(detectorfile, "r")
                text = _file.readlines()
                _file.close()
                # first line contains parameters
                parameters = [float(elem) for elem in str(text[0]).split(",")]
                detectorparameters_global = parameters[:5]
                pixelsize_global = parameters[5]
                dim1_global = parameters[6]
                dim2_global = parameters[7]
                # others are comments
                comments = text[1:]
                ccd_label_global = ""
                for line in comments:
                    if line.startswith("# CCDLabel"):
                        ccd_label_global = line.split(":")[1].strip()
                if ccd_label_global == "":
                    self.write_to_console("CCD label cannot be read from the calibration file, setting it to latest detector sCMOS")
                    ccd_label_global = "sCMOS"
            except IOError as error:
                self.write_to_console("Error opening file\n" + str(error))
            except UnicodeDecodeError as error:
                self.write_to_console("Error opening file\n" + str(error))
            except:
                self.write_to_console("Error opening file\n")
        
        try:
            UB_matrix_global = int(config.get('PREDICTION', 'UB_matrix_to_detect'))
        except:
            UB_matrix_global = 2
            self.write_to_console("UB matrix to identify not defined, can be set in the Prediction window (setting default to 2)")
        
        try:
            image_grid_globalx = int(config.get('EXPERIMENT', 'image_grid_x'))
            image_grid_globaly = int(config.get('EXPERIMENT', 'image_grid_y'))
        except:
            image_grid_globalx = 10
            image_grid_globaly = 10
            self.write_to_console("Scan grid not defined, can be set in the Prediction window")
        
        try:
            softmax_threshold_global = float(config.get('PREDICTION', 'softmax_threshold_global'))
        except:
            softmax_threshold_global = 0.8
            self.write_to_console("Softmax threshold not defined, using default 80%")
        self.softmax_threshold_global = softmax_threshold_global
        
        try:
            mr_threshold_global = float(config.get('PREDICTION', 'mr_threshold_global'))
        except:
            mr_threshold_global = 0.95
            self.write_to_console("Matching rate threshold not defined, using default 95%")
        self.mr_threshold_global = mr_threshold_global
        
        try:
            coeff = float(config.get('PREDICTION', 'coeff'))
        except:
            coeff = 0.25
            self.write_to_console("Coeff Overlap v0 not defined, using default 25%")
        self.coeff=coeff

        try:
            coeff_overlap1212 = float(config.get('PREDICTION', 'coeff_overlap'))
        except:
            coeff_overlap1212 = 0.25
            self.write_to_console("Coeff Overlap not defined, using default 25%")
        self.coeff_overlap=coeff_overlap1212
        
        try:
            mode_spotCycle = str(config.get('PREDICTION', 'mode_spotCycle'))
        except:
            mode_spotCycle = "graphmode"
            self.write_to_console("Analysis mode not defined, using default graphmode, can be set in Prediction window")
        self.mode_spotCycleglobal = mode_spotCycle
        
        try:
            material0_limit1212 = int(config.get('PREDICTION', 'material0_limit'))
        except:
            self.write_to_console("Max Nb of UB per material 0 not defined, using default maximum")
        self.material0_limit = material0_limit1212
        
        try:
            material1_limit1212 = int(config.get('PREDICTION', 'material1_limit'))
        except:
            self.write_to_console("Max Nb of UB per material 1 not defined, using default maximum")
        self.material1_limit = material1_limit1212
        
        try:
            cap_matchrate = float(config.get('PREDICTION', 'cap_matchrate')) * 100
        except:
            self.write_to_console("Cap_Matching rate not defined, setting default value of 1%")
        self.cap_matchrate=cap_matchrate
        try:
            tolerance_global = float(config.get('PREDICTION', 'matrix_tolerance'))
        except:
            tolerance_global = 0.5
            self.write_to_console("Angle tolerance to detect grains not defined, using default 0.5")
        try:
            tolerance_global1 = float(config.get('PREDICTION', 'matrix_tolerance1'))
        except:
            tolerance_global1 = 0.5
            self.write_to_console("Angle tolerance for Mat 1 to detect grains not defined, using default 0.5")
        try:
            use_previous_UBmatrix = config.get('PREDICTION', 'use_previous') == "true"
        except:
            use_previous_UBmatrix = False
            self.write_to_console("Use previous solutions not defined, using default value False")
        self.use_previous_UBmatrix = use_previous_UBmatrix
        
        try:
            intensity_threshold_global = float(config.get('PEAKSEARCH', 'intensity_threshold'))
        except:
            intensity_threshold_global = 150
            self.write_to_console("intensity_threshold not defined, using default 150cts after BG correction")
            
        try:
            boxsize_global = int(config.get('PEAKSEARCH', 'boxsize'))
        except:
            boxsize_global = 15
            self.write_to_console("boxsize not defined, using default size of 15")
            
        try:
            fit_peaks_gaussian_global = int(config.get('PEAKSEARCH', 'fit_peaks_gaussian'))
        except:
            self.write_to_console("Fitting of peaks not defined, using default Gaussian fitting")
        self.fit_peaks_gaussian_global = fit_peaks_gaussian_global
        
        try:
            FitPixelDev_global = float(config.get('PEAKSEARCH', 'FitPixelDev'))
        except:
            self.write_to_console("Fitting PixelDev of peaks not defined, using default 15 pix")
        self.FitPixelDev_global=FitPixelDev_global
        
        try:
            NumberMaxofFits = float(config.get('PEAKSEARCH', 'NumberMaxofFits'))
        except:
            self.write_to_console("Max fits per LP not defined, using default 3000")
        self.NumberMaxofFits=NumberMaxofFits
        
        try:
            strain_label_global = config.get('STRAINCALCULATION', 'strain_compute') == "true"
            if strain_label_global:
                strain_label_global = "YES"
            else:
                strain_label_global = "NO"
        except:
            strain_label_global = "YES"
            self.write_to_console("Strain computation not defined, default True")
        
        try:
            tolerance_strain_temp = config.get('STRAINCALCULATION', 'tolerance_strain_refinement').split(",")
            tolerance_strain = [float(i) for i in tolerance_strain_temp]
        except:
            tolerance_strain = list(np.linspace(tolerance_global, 0.2, 4))
            self.write_to_console("Strain tolerance material 0 not defined, taking regular space steps")
        self.tolerance_strain = tolerance_strain
        
        try:
            tolerance_strain_temp1 = config.get('STRAINCALCULATION', 'tolerance_strain_refinement1').split(",")
            tolerance_strain1 = [float(i) for i in tolerance_strain_temp1]
        except:
            tolerance_strain1 = list(np.linspace(tolerance_global1, 0.2, 4))
            self.write_to_console("Strain tolerance for material 1 not defined, taking regular space steps")
        self.tolerance_strain1 = tolerance_strain1
        
        try:
            strain_free_parameters = config.get('STRAINCALCULATION', 'free_parameters').split(",")
        except:
            strain_free_parameters = ["rotx", "roty", "rotz", "alpha", "beta", "gamma", "b", "c"]
            self.write_to_console("strain_free_parameters not defined; fixing only 'a' length by default")
        self.strain_free_parameters = strain_free_parameters
        
        try:
            additional_expression = config.get('STRAINCALCULATION', 'additional_expression').split(",")
        except:
            additional_expression = ["none"]
            self.write_to_console("additional_expression not defined; none by default")
        self.additional_expression = additional_expression  
        
        try:
            hkls_list_global = config.get('POSTPROCESS', 'hkls_subsets')
        except:
            self.write_to_console("HKL post processing not defined, currently not used")
        
        expfile_global = config.get('EXPERIMENT', 'experiment_directory')
        exp_prefix_global = config.get('EXPERIMENT', 'experiment_file_prefix')
        
        if expfile_global == "ZnCuOCl":
            expfile_global = resource_path("examples//ZnCUOCl",  verbose=0)
        elif expfile_global == "GaN":
            expfile_global = resource_path("examples//GaN_Si",  verbose=0)
        
        if exp_prefix_global == "ZnCuOCl":
            exp_prefix_global = "HS17O_1_C_"
        elif exp_prefix_global == "GaN":
            exp_prefix_global = "nw1_"
            
        ##exp directory
        if material_global == material1_global:
            fn = material_global + prefix_global
        else:
            fn = material_global + "_" + material1_global + prefix_global
        
        try:
            model_weight_file = config.get('PREDICTION', 'model_weight_file')
        except:
            model_weight_file = "none"
        
        modelfile_global = main_directory + "//" + fn
        if material_global == material1_global:
            if model_weight_file == "none":
                weightfile_global = modelfile_global + "//" + "model_" + material_global + ".h5"
            else:
                weightfile_global = model_weight_file
        else:
            if model_weight_file == "none":
                weightfile_global = modelfile_global + "//" + "model_" + material_global + "_" + material1_global + ".h5"
            else:
                weightfile_global = model_weight_file
        
        try:
            freq_rmv_global = int(config.get('TRAINING', 'classes_with_frequency_to_remove'))
        except:
            self.write_to_console("Frequency removal for HKLs not defined, can be defined in the config window")
        
        try:
            elements_global = config.get('TRAINING', 'desired_classes_output')
        except:
            self.write_to_console("Elements for HKLs not defined, can be defined in the config window")
        try:
            hkl_max_global = config.get('TRAINING', 'max_HKL_index')
        except:
            self.write_to_console("Max HKLs not defined, can be defined in the config window")
        try:
            nb_grains_per_lp_global = int(config.get('TRAINING', 'max_nb_grains'))
        except:
            self.write_to_console("Nb. of grains per LP not defined, can be defined in the config window")
        try:
            freq_rmv1_global = int(config.get('TRAINING', 'classes_with_frequency_to_remove1'))
        except:
            self.write_to_console("Frequency removal for HKLs 1 not defined, can be defined in the config window")
        try:
            elements1_global = config.get('TRAINING', 'desired_classes_output1')
        except:
            self.write_to_console("Elements for HKLs 1 not defined, can be defined in the config window")
        try:
            hkl_max1_global = config.get('TRAINING', 'max_HKL_index1')
        except:
            self.write_to_console("Max HKLs 1 not defined, can be defined in the config window")
        try:
            nb_grains_per_lp1_global = int(config.get('TRAINING', 'max_nb_grains1'))
        except:
            self.write_to_console("Nb. of grains per LP 1 not defined, can be defined in the config window")
        try:
            maximum_angle_to_search_global = float(config.get('TRAINING', 'angular_distance'))
        except:
            self.write_to_console("Histogram angle not defined, can be defined in the config window")
        try:
            step_for_binning_global = float(config.get('TRAINING', 'step_size'))
        except:
            self.write_to_console("steps for histogram binnning not defined, can be defined in the config window")
        try:
            grains_nb_simulate_global = int(config.get('TRAINING', 'max_simulations'))
        except:
            self.write_to_console("Number of simulations per LP not defined, can be defined in the config window")
        try:
            include_scm_global = config.get('TRAINING', 'include_small_misorientation') == "true"
        except:
            include_scm_global = False
            self.write_to_console("Single crystal misorientation not defined, can be defined in the config window")
        try:
            misorientation_angle = float(config.get('TRAINING', 'misorientation_angle'))
        except:
            misorientation_angle = misorientation_angle1
            self.write_to_console("Angle of Single crystal misorientation along Z not defined, can be defined in the config window")
        self.misorientation_angle = misorientation_angle
        try:
            batch_size_global = int(config.get('TRAINING', 'batch_size'))
        except:
            self.write_to_console("Batch size not defined, can be defined in the config window")
        try:
            epochs_global = int(config.get('TRAINING', 'epochs'))
        except:
            self.write_to_console("Epochs not defined, can be defined in the config window")
        
        try:
            material_phase_always_present = config.get('DEVELOPMENT', 'material_phase_always_present')
        except:
            material_phase_always_present = "none"
            self.write_to_console("material_phase_always_present not defined, default is NONE")
        if material_phase_always_present == "none":
            material_phase_always_present = None
        else:
            material_phase_always_present = int(material_phase_always_present)    
        self.material_phase_always_present = material_phase_always_present
        
        ## matrix_phase_always_present; add a matrix to the training dataset to be always present
        try:
            matrix_phase_always_present = config.get('DEVELOPMENT', 'matrix_phase_always_present')
        except:
            matrix_phase_always_present = "none"
            self.write_to_console("matrix_phase_always_present not defined, default is NONE")
        if matrix_phase_always_present == "none":
            matrix_phase_always_present = None
        else:
            matrix_phase_always_present = matrix_phase_always_present
        self.matrix_phase_always_present = matrix_phase_always_present
        try:
            generate_additional_data = config.get('DEVELOPMENT', 'generate_additional_data')=='true'
        except:
            generate_additional_data = False
            self.write_to_console("generate_additional_data not defined, default is False")
        self.generate_additional_data = generate_additional_data
        try:
            write_mtex_file = config.get('DEVELOPMENT', 'write_MTEX_file') == "true"
        except:
            self.write_to_console("Write MTEX texture file not defined, by default True")
            
        try:
            if symmetry_global =="cubic":
                material0_lauegroup = "11"
            elif symmetry_global =="monoclinic":
                material0_lauegroup = "2"
            elif symmetry_global == "hexagonal":
                material0_lauegroup = "9"
            elif symmetry_global == "orthorhombic":
                material0_lauegroup = "3"
            elif symmetry_global == "tetragonal":
                material0_lauegroup = "5"
            elif symmetry_global == "trigonal":
                material0_lauegroup = "7"
            elif symmetry_global == "triclinic":
                material0_lauegroup = "1"
        except:
            material0_lauegroup = "11"
            
        try:
            if symmetry1_global =="cubic":
                material1_lauegroup = "11"
            elif symmetry1_global =="monoclinic":
                material1_lauegroup = "2"
            elif symmetry1_global == "hexagonal":
                material1_lauegroup = "9"
            elif symmetry1_global == "orthorhombic":
                material1_lauegroup = "3"
            elif symmetry1_global == "tetragonal":
                material1_lauegroup = "5"
            elif symmetry1_global == "trigonal":
                material1_lauegroup = "7"
            elif symmetry1_global == "triclinic":
                material1_lauegroup = "1"
        except:
            material1_lauegroup = "11"
            
        ##update config file for Neural network
        try:
            residues_threshold = config.get('CALLER', 'residues_threshold')
        except:
            self.write_to_console("residues_threshold not defined, by default 0.5")
            residues_threshold = 0.5

        try:
            nb_spots_global_threshold = config.get('CALLER', 'nb_spots_global_threshold')
        except:
            self.write_to_console("minimum number of spots for links not defined, by default 8")
            nb_spots_global_threshold = 8
        
        try:
            option_global = config.get('CALLER', 'option_global')
        except:
            self.write_to_console("option_global not defined, by default v2 for calulating the auto-links")
            option_global = "v2"
        
        try:
            use_om_user_global = config.get('CALLER', 'use_om_user')
        except:
            self.write_to_console("use_om_user not defined, by default False")
            use_om_user_global = "false"
            
        try:
            nb_spots_consider_global = int(config.get('CALLER', 'nb_spots_consider'))
        except:
            self.write_to_console("nb_spots_consider not defined, by default first 200")
            nb_spots_consider_global = 200
            
        try:
            path_user_OM_global = config.get('CALLER', 'path_user_OM')
        except:
            self.write_to_console("path_user_OM not defined, by default None")
            path_user_OM_global = ""
        
        try:
            mode_peaksearch = config.get('PEAKSEARCH', 'mode')
        except:
            self.write_to_console("mode_peaksearch not defined, by default skimage method of peak search")
            mode_peaksearch = "skimage"
        
        config_setting = configparser.ConfigParser()
        filepath = resource_path('settings.ini')
        self.write_to_console("Settings path is "+filepath)
        config_setting.read(filepath)
        config_setting.set('CALLER', 'residues_threshold',str(residues_threshold))
        config_setting.set('CALLER', 'nb_spots_global_threshold',str(nb_spots_global_threshold))
        config_setting.set('CALLER', 'option_global',str(option_global))
        config_setting.set('CALLER', 'use_om_user',str(use_om_user_global))
        config_setting.set('CALLER', 'nb_spots_consider',str(nb_spots_consider_global))
        config_setting.set('CALLER', 'path_user_OM',str(path_user_OM_global))
        config_setting.set('CALLER', 'intensity', str(intensity_threshold_global))
        config_setting.set('CALLER', 'boxsize', str(boxsize_global))
        config_setting.set('CALLER', 'pixdev', str(FitPixelDev_global))
        config_setting.set('CALLER', 'cap_softmax', str(softmax_threshold_global))
        config_setting.set('CALLER', 'cap_mr', str(cap_matchrate/100.))
        config_setting.set('CALLER', 'strain_free_parameters', ",".join(strain_free_parameters))
        config_setting.set('CALLER', 'additional_expression', ",".join(additional_expression))
        config_setting.set('CALLER', 'mode_peaksearch', str(mode_peaksearch))
        with open(filepath, 'w') as configfile:
            config_setting.write(configfile)
        self.write_to_console("Config file loaded successfully.")
        # except:
        #     self.write_to_console("Config file Error.")

    def _createToolBar(self):
        self.tools = QToolBar()
        self.addToolBar(self.tools)
        self.trialtoolbar101 = self.tools.addAction('Example_config', self.show_window_config)
        self.trialtoolbar10 = self.tools.addAction('Re-Train saved model', self.show_window_retraining_fromfile)
        self.trialtoolbar1 = self.tools.addAction('Re-Train GUI model', self.show_window_retraining)
        self.trialtoolbar10.setEnabled(False)
        self.trialtoolbar1.setEnabled(False)
        
    def show_window_parameters(self):
        w2 = AnotherWindowParams(self.state, gui_state)
        w2.got_signal.connect(self.postprocesstrain)
        w2.show()
        self.popups.append(w2)
        self.state = self.state +1
    
    def show_window_retraining(self):
        ct = time.time()
        now = datetime.datetime.fromtimestamp(ct)
        c_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        self.train_model(prefix="_"+c_time, tag = 1)
        
    def show_window_retraining_fromfile(self):
        ct = time.time()
        now = datetime.datetime.fromtimestamp(ct)
        c_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        self.train_model(prefix="_"+c_time, tag = 2)
        
    def show_window_config(self):
        w21 = sample_config()
        w21.show()
        self.popups.append(w21)
        
    def show_window_liveprediction(self):
        try:
            if self.material_ != self.material1_:
                with open(self.save_directory+"//classhkl_data_nonpickled_"+self.material_+".pickle", "rb") as input_file:
                    hkl_all_class0 = cPickle.load(input_file)[0]
    
                with open(self.save_directory+"//classhkl_data_nonpickled_"+self.material1_+".pickle", "rb") as input_file:
                    hkl_all_class1 = cPickle.load(input_file)[0]
    
            else:
                hkl_all_class1 = None
                with open(self.save_directory+"//classhkl_data_nonpickled_"+self.material_+".pickle", "rb") as input_file:
                    hkl_all_class0 = cPickle.load(input_file)[0]
        except:
            try:
                if self.material_ != self.material1_:
                    with open(self.save_directory+"//classhkl_data_"+self.material_+".pickle", "rb") as input_file:
                        _, _, _, _, _, hkl_all_class0, _, _, symmetry = cPickle.load(input_file)
    
                    with open(self.save_directory+"//classhkl_data_"+self.material1_+".pickle", "rb") as input_file:
                        _, _, _, _, _, hkl_all_class1, _, _, _ = cPickle.load(input_file)
    
                else:
                    hkl_all_class1 = None
                    with open(self.save_directory+"//classhkl_data_"+self.material_+".pickle", "rb") as input_file:
                        _, _, _, _, _, hkl_all_class0, _, _, _ = cPickle.load(input_file)
            except:
                print("No model could be found for the defined material; please verify")
                return
        
        w2 = AnotherWindowLivePrediction(self.state2, gui_state, 
                                         material_=self.material_, material1_=self.material1_, emin=self.emin, 
                                         emax=self.emax, symmetry=self.symmetry, symmetry1=self.symmetry1,
                                         detectorparameters=self.detectorparameters, pixelsize=self.pixelsize,
                                         lattice_=self.lattice_material, lattice1_ =self.lattice_material1,
                                         hkl_all_class0 = hkl_all_class0, hkl_all_class1=hkl_all_class1,
                                         mode_spotCycleglobal=self.mode_spotCycleglobal,
                                         softmax_threshold_global = self.softmax_threshold_global,
                                         mr_threshold_global =    self.mr_threshold_global,
                                         cap_matchrate =    self.cap_matchrate,
                                         coeff =    self.coeff,
                                         coeff_overlap1212 =    self.coeff_overlap,
                                         fit_peaks_gaussian_global =    self.fit_peaks_gaussian_global,
                                         FitPixelDev_global =    self.FitPixelDev_global,
                                         NumberMaxofFits =    self.NumberMaxofFits,
                                         tolerance_strain =    self.tolerance_strain,
                                         tolerance_strain1 =    self.tolerance_strain1,
                                         material0_limit = self.material0_limit,
                                         material1_limit = self.material1_limit,
                                         symmetry_name = self.symmetry_name, 
                                         symmetry1_name = self.symmetry1_name,
                                         use_previous_UBmatrix_name = self.use_previous_UBmatrix,
                                         material_phase_always_present = self.material_phase_always_present,
                                         crystal=self.crystal, crystal1=self.crystal1,
                                         strain_free_parameters=self.strain_free_parameters,
                                         additional_expression=self.additional_expression)
        w2.show()
        self.popups.append(w2)
        self.state2 += 1
        
    def _createStatusBar(self):
        self.status = QStatusBar()
        self.status.showMessage("status")
        self.setStatusBar(self.status)

    def _formLayout(self):
        self.formLayout = QFormLayout()
        
        self.progress = QProgressBar()
        
        self.configure_nn = QPushButton('Configure parameters')
        self.configure_nn.clicked.connect(self.show_window_parameters)
        self.configure_nn.setEnabled(True)
        
        self.generate_nn = QPushButton('Generate Training dataset')
        self.generate_nn.clicked.connect(self.generate_training_data)
        self.generate_nn.setEnabled(False)
        
        self.train_nn = QPushButton('Train Neural Network')
        self.train_nn.clicked.connect(self.train_neural_network)
        self.train_nn.setEnabled(False)
        
        self.train_nnhp = QPushButton('Hypergrid Params OPT')
        self.train_nnhp.clicked.connect(self.grid_search_hyperparams)
        self.train_nnhp.setEnabled(False)

        self.predict_lnn = QPushButton('Prediction of Laue data')
        self.predict_lnn.clicked.connect(self.show_window_liveprediction)
        self.predict_lnn.setEnabled(False)
        
        self.formLayout.addRow(self.progress)
        self.formLayout.addRow(self.configure_nn)
        self.formLayout.addRow(self.generate_nn)
        self.formLayout.addRow(self.train_nn)
        self.formLayout.addRow(self.train_nnhp)
        self.formLayout.addRow(self.predict_lnn)
        self.layout.addLayout(self.formLayout)
        
    def write_to_console(self, line):
        try:
            self.text_file_log.write(line + "\n")
        except:
            print("Log file not yet created: "+ str(line.encode('utf-8','ignore')))
        self.setDisplayText(str(line.encode('utf-8','ignore'),errors='ignore'))
        QApplication.processEvents() 
    
    def postprocesstrain(self, emit_dict):
        self.input_params = {
                            "material_": emit_dict["material_"], ## same key as used in LaueTools
                            "material1_": emit_dict["material1_"],
                            "prefix": emit_dict["prefix"],
                            "symmetry": emit_dict["symmetry"],
                            "symmetry1": emit_dict["symmetry1"],
                            "hkl_max_identify" : emit_dict["hkl_max_identify"], # can be "auto" or an index i.e 12
                            "hkl_max_identify1" : emit_dict["hkl_max_identify1"],
                            "maximum_angle_to_search" : emit_dict["maximum_angle_to_search"],
                            "step_for_binning" : emit_dict["step_for_binning"],
                            "mode_of_analysis" : emit_dict["mode_of_analysis"],
                            "nb_grains_per_lp" : emit_dict["nb_grains_per_lp"], ## max grains to expect in a LP
                            "nb_grains_per_lp1" : emit_dict["nb_grains_per_lp1"],
                            "grains_nb_simulate" : emit_dict["grains_nb_simulate"],
                            "detectorparameters" : emit_dict["detectorparameters"],
                            "pixelsize" : emit_dict["pixelsize"],
                            "dim1" : emit_dict["dim1"],
                            "dim2" : emit_dict["dim2"],
                            "emin" : emit_dict["emin"],
                            "emax" : emit_dict["emax"],
                            "batch_size" : emit_dict["batch_size"], ## batches of files to use while training
                            "epochs" : emit_dict["epochs"], ## number of epochs for training
                            "texture": emit_dict["texture"],
                            "mode_nn": emit_dict["mode_nn"],
                            "grid_bool": emit_dict["grid_bool"],
                            "directory": emit_dict["directory"],
                            "freq_rmv":  emit_dict["freq_rmv"],
                            "elements":  emit_dict["elements"],
                            "freq_rmv1":  emit_dict["freq_rmv1"],
                            "elements1":  emit_dict["elements1"],
                            "include_scm":  emit_dict["include_scm"],
                            "lr":  emit_dict["lr"],
                            "kc":  emit_dict["kc"],
                            "bc":  emit_dict["bc"],
                            "architecture": emit_dict["architecture"],
                            }
        
        ## get the architecture of the model
        self.architecture = self.input_params["architecture"]
        self.write_to_console("Only FFNN model is optimized for the current indexation problem, other architecture works, but their efficieny is not well tested yet!")
        # self.write_to_console("Prediction routines for other architecture is not well tested, please verify them, falling back to the default FFNN model")
        if self.input_params["architecture"] == "FFNN":
            self.write_to_console("Using the classical inbuilt FFNN Feed Forward Neural Network model, for user defined model, please define a model in the NNmodels.py file (found in LauetoolsNN installation folder)")
        elif self.input_params["architecture"] == "1D_CNN":
            self.write_to_console("Using the 1D_CNN pure Convolution Neural Network model, for user defined model, please define a model in the NNmodels.py file (found in LauetoolsNN installation folder)")
        elif self.input_params["architecture"] == "1D_CNN_DNN":
            self.write_to_console("Using the 1D_CNN_DNN Convolution Network with Fully connected layers at the end, for user defined model, please define a model in the NNmodels.py file (found in LauetoolsNN installation folder)")
        elif self.input_params["architecture"] == "User_defined":
            self.write_to_console("Using the user defined model from the NNmodels.py file (found in LauetoolsNN installation folder)")
        else:
            self.write_to_console("Undefined Neural network model requested, falling back to default FFNN model")
            self.architecture = "FFNN"
            
            
        ## Gray out options based on the mode_nn
        if self.input_params["mode_nn"] == "Generate Data & Train":
            self.write_to_console("Generate and Train the Model")
            self.generate_nn.setEnabled(True)
            
        elif self.input_params["mode_nn"] == "Train":
            self.write_to_console("Data already exists ? Train the Model")
            self.train_nn.setEnabled(True)
            self.trialtoolbar10.setEnabled(True)
            
        elif self.input_params["mode_nn"] == "Predict":
            self.write_to_console("Model already exists? Lets Predict!")
            self.write_to_console("on the fly prediction (fingers crossed)")
            self.predict_lnn.setEnabled(True)

        if self.input_params["grid_bool"] == "True":
            self.train_nnhp.setEnabled(True)
        
        self.include_scm = False
        if self.input_params["include_scm"] == "yes":
            self.include_scm = True  
            
        self.freq_rmv = self.input_params["freq_rmv"]
        self.freq_rmv1 = self.input_params["freq_rmv1"]
        if self.input_params["elements"] == "all":
            self.elements = self.input_params["elements"] #"all"
            self.elements1 = self.input_params["elements1"] #"all"
        else:
            self.elements = int(self.input_params["elements"])
            self.elements1 = int(self.input_params["elements1"])
            
        self.material_ = self.input_params["material_"]
        self.material1_ = self.input_params["material1_"]
        
        self.emin, self.emax = self.input_params["emin"], self.input_params["emax"]
        
        self.learning_rate, self.kernel_coeff, self.bias_coeff = self.input_params["lr"],self.input_params["kc"],self.input_params["bc"]
        
        if self.input_params["directory"] == None: ## default path
            if self.material_ == self.material1_:
                self.save_directory = os.getcwd()+"//"+self.input_params["material_"]+self.input_params["prefix"]
            else:
                self.save_directory = os.getcwd()+"//"+self.input_params["material_"]+"_"+self.input_params["material1_"]+self.input_params["prefix"]
        else:
            if self.material_ == self.material1_:
                self.save_directory = self.input_params["directory"]+"//"+self.input_params["material_"]+self.input_params["prefix"]
            else:
                self.save_directory = self.input_params["directory"]+"//"+self.input_params["material_"]+"_"+self.input_params["material1_"]+self.input_params["prefix"]

        self.n = self.input_params["hkl_max_identify"]
        self.n1 = self.input_params["hkl_max_identify1"]
        self.maximum_angle_to_search = self.input_params["maximum_angle_to_search"]
        self.step_for_binning = self.input_params["step_for_binning"]
        self.mode_of_analysis = self.input_params["mode_of_analysis"]
        self.nb_grains_per_lp = self.input_params["nb_grains_per_lp"]
        self.nb_grains_per_lp1 = self.input_params["nb_grains_per_lp1"]
        self.grains_nb_simulate = self.input_params["grains_nb_simulate"]
        self.detectorparameters = self.input_params["detectorparameters"]
        self.pixelsize = self.input_params["pixelsize"]
        # =============================================================================
        # Symmetry input
        # =============================================================================
        a, b, c, alpha, beta, gamma = dictLT.dict_Materials[self.material_][1]
        # a, b, c = a*0.1, b*0.1, c*0.1
        if self.SG == None:
            if self.input_params["symmetry"] =="cubic":
                self.SG = 230
            elif self.input_params["symmetry"] =="monoclinic":
                self.SG = 10
            elif self.input_params["symmetry"] == "hexagonal":
                self.SG = 191
            elif self.input_params["symmetry"] == "orthorhombic":
                self.SG = 47
            elif self.input_params["symmetry"] == "tetragonal":
                self.SG = 123
            elif self.input_params["symmetry"] == "trigonal":
                self.SG = 162
            elif self.input_params["symmetry"] == "triclinic":
                self.SG = 2
        
        self.rules = dictLT.dict_Materials[self.material_][-1]
        self.symmetry_name = self.input_params["symmetry"]
        if self.input_params["symmetry"] =="cubic":
            self.crystal = SGLattice(int(self.SG), a)
            self.symmetry = Symmetry.cubic
            self.lattice_material = Lattice.cubic(a)
        elif self.input_params["symmetry"] =="monoclinic":
            self.crystal = SGLattice(int(self.SG),a, b, c, beta)
            self.symmetry = Symmetry.monoclinic
            self.lattice_material = Lattice.monoclinic(a, b, c, beta)
        elif self.input_params["symmetry"] == "hexagonal":
            self.crystal = SGLattice(int(self.SG),a, c)
            self.symmetry = Symmetry.hexagonal
            self.lattice_material = Lattice.hexagonal(a, c)
        elif self.input_params["symmetry"] == "orthorhombic":
            self.crystal = SGLattice(int(self.SG),a, b, c)
            self.symmetry = Symmetry.orthorhombic
            self.lattice_material = Lattice.orthorhombic(a, b, c)
        elif self.input_params["symmetry"] == "tetragonal":
            self.crystal = SGLattice(int(self.SG),a, c)
            self.symmetry = Symmetry.tetragonal
            self.lattice_material = Lattice.tetragonal(a, c)
        elif self.input_params["symmetry"] == "trigonal":
            self.crystal = SGLattice(int(self.SG),a, alpha)
            self.symmetry = Symmetry.trigonal
            self.lattice_material = Lattice.rhombohedral(a, alpha)
        elif self.input_params["symmetry"] == "triclinic":
            self.crystal = SGLattice(int(self.SG),a, b, c, alpha, beta, gamma)
            self.symmetry = Symmetry.triclinic
            self.lattice_material = Lattice.triclinic(a, b, c, alpha, beta, gamma)

        if self.material_ != self.material1_:
            
            if self.SG1 == None:
                if self.input_params["symmetry1"] =="cubic":
                    self.SG1 = 230
                elif self.input_params["symmetry1"] =="monoclinic":
                    self.SG1 = 10
                elif self.input_params["symmetry1"] == "hexagonal":
                    self.SG1 = 191
                elif self.input_params["symmetry1"] == "orthorhombic":
                    self.SG1 = 47
                elif self.input_params["symmetry1"] == "tetragonal":
                    self.SG1 = 123
                elif self.input_params["symmetry1"] == "trigonal":
                    self.SG1 = 162
                elif self.input_params["symmetry1"] == "triclinic":
                    self.SG1 = 2
            
            self.symmetry1_name = self.input_params["symmetry1"]
            a1, b1, c1, alpha1, beta1, gamma1 = dictLT.dict_Materials[self.material1_][1]
            self.rules1 = dictLT.dict_Materials[self.material1_][-1]
            if self.input_params["symmetry1"] =="cubic":
                self.crystal1 = SGLattice(int(self.SG1), a1)
                self.symmetry1 = Symmetry.cubic
                self.lattice_material1 = Lattice.cubic(a1)
            elif self.input_params["symmetry1"] =="monoclinic":
                self.crystal1 = SGLattice(int(self.SG1),a1, b1, c1, beta1)
                self.symmetry1 = Symmetry.monoclinic
                self.lattice_material1 = Lattice.monoclinic(a1, b1, c1, beta1)
            elif self.input_params["symmetry1"] == "hexagonal":
                self.crystal1 = SGLattice(int(self.SG1),a1, c1)
                self.symmetry1 = Symmetry.hexagonal
                self.lattice_material1 = Lattice.hexagonal(a1, c1)
            elif self.input_params["symmetry1"] == "orthorhombic":
                self.crystal1 = SGLattice(int(self.SG1),a1, b1, c1)
                self.symmetry1 = Symmetry.orthorhombic
                self.lattice_material1 = Lattice.orthorhombic(a1, b1, c1)
            elif self.input_params["symmetry1"] == "tetragonal":
                self.crystal1 = SGLattice(int(self.SG1),a1, c1)
                self.symmetry1 = Symmetry.tetragonal
                self.lattice_material1 = Lattice.tetragonal(a1, c1)
            elif self.input_params["symmetry1"] == "trigonal":
                self.crystal1 = SGLattice(int(self.SG1),a1, alpha1)
                self.symmetry1 = Symmetry.trigonal
                self.lattice_material1 = Lattice.rhombohedral(a1, alpha1)
            elif self.input_params["symmetry1"] == "triclinic":
                self.crystal1 = SGLattice(int(self.SG1),a1, b1, c1, alpha1, beta1, gamma1)
                self.symmetry1 = Symmetry.triclinic
                self.lattice_material1 = Lattice.triclinic(a1, b1, c1, alpha1, beta1, gamma1)
        else:
            self.rules1 = None
            self.symmetry1 = None
            self.lattice_material1 = None
            self.crystal1 = None
            self.symmetry1_name = self.input_params["symmetry"]
        
        self.modelp = "random" 
        ### Load texture files based on symmetry
        if self.input_params["texture"] == "in-built_Uniform_Distribution":
            self.write_to_console("Using uniform distribution generated with Neper for Training dataset \n") 
            self.modelp = "uniform"
        elif self.input_params["texture"] == "random":
            self.write_to_console("Using random orientation distribution for Training dataset \n") 
            self.modelp = "random"
        else:
            self.modelp = "experimental"
            self.write_to_console("# User defined texture to be used: TODO \n") 
        
        try:
            if not os.path.exists(self.save_directory):
                os.makedirs(self.save_directory)
        except:
            if self.material_ == self.material1_:
                self.save_directory = os.getcwd()+"//"+self.input_params["material_"]+self.input_params["prefix"]
            else:
                self.save_directory = os.getcwd()+"//"+self.input_params["material_"]+"_"+self.input_params["material1_"]+self.input_params["prefix"]
            if not os.path.exists(self.save_directory):
                os.makedirs(self.save_directory)
                
        self.write_to_console("Working directory :"+ self.save_directory)
        
        ### update global parameters 
        global material_global, symmetry_global, material1_global, symmetry1_global
        global prefix_global, emin_global, emax_global
        global detectorparameters_global, pixelsize_global, dim1_global, dim2_global
        global modelfile_global, weightfile_global
        global hkl_max_global, elements_global, freq_rmv_global, hkl_max1_global
        global elements1_global, freq_rmv1_global, maximum_angle_to_search_global
        global step_for_binning_global, nb_grains_per_lp_global, nb_grains_per_lp1_global
        global grains_nb_simulate_global, include_scm_global, batch_size_global, epochs_global
        
        material_global = self.material_ ## same key as used in LaueTools
        symmetry_global = self.input_params["symmetry"]
        material1_global = self.material1_ ## same key as used in LaueTools
        symmetry1_global = self.input_params["symmetry1"]
        prefix_global = self.input_params["prefix"]
        detectorparameters_global = self.input_params["detectorparameters"]
        pixelsize_global = self.input_params["pixelsize"]
        dim1_global = self.input_params["dim1"]
        dim2_global = self.input_params["dim2"]
        emax_global = self.input_params["emax"]
        emin_global = self.input_params["emin"]
        ##exp directory
        modelfile_global = self.save_directory
        if material_global == material1_global:
            weightfile_global = modelfile_global + "//" + "model_" + material_global + ".h5"
        else:
            weightfile_global = modelfile_global + "//" + "model_" + material_global + "_" + material1_global + ".h5"

        hkl_max_global = str(self.n)
        elements_global = str(self.elements)
        freq_rmv_global = self.freq_rmv
        hkl_max1_global = str(self.n1)
        elements1_global = str(self.elements1)
        freq_rmv1_global = self.freq_rmv1
        maximum_angle_to_search_global = self.maximum_angle_to_search
        step_for_binning_global = self.step_for_binning
        nb_grains_per_lp_global = self.nb_grains_per_lp
        nb_grains_per_lp1_global = self.nb_grains_per_lp1
        grains_nb_simulate_global = self.grains_nb_simulate
        include_scm_global = self.include_scm
        batch_size_global = self.input_params["batch_size"]
        epochs_global = self.input_params["epochs"]

        ## Golbal log file
        now = datetime.datetime.fromtimestamp(GUI_START_TIME)
        c_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        
        if self.material_ == self.material1_:
            self.text_file_log = open(self.save_directory+"//log_"+self.material_+".txt", "a")
        else:
            self.text_file_log = open(self.save_directory+"//log_"+self.material_+"_"+self.material1_+".txt", "a")
        self.text_file_log.write("# Log file created at "+ c_time + "\n") 

    def temp_HKL(self, removeharmonics=1):
        material_= self.input_params["material_"]
        nbgrains = self.input_params["nb_grains_per_lp"]
        nbtestspots = 0
        hkl_sol_all = np.zeros((1,4))
        verbose=0
        for _ in range(10):
            seednumber = np.random.randint(1e6)
            tabledistancerandom, hkl_sol, \
                                    _, _, _, _, _ = self.prepare_LP(nbgrains, 0,
                                                                    material_,
                                                                    None,
                                                                    verbose,
                                                                    plotLauePattern=False,
                                                                    seed=seednumber,
                                                                    detectorparameters=self.input_params["detectorparameters"], 
                                                                    pixelsize=self.input_params["pixelsize"],
                                                                    dim1=self.input_params["dim1"],
                                                                    dim2=self.input_params["dim2"],
                                                                    removeharmonics=removeharmonics)
                                    
            spots_in_center = [sp for sp in range(len(tabledistancerandom))] # take all spots in Laue pattern
            hkl_sol_all = np.vstack((hkl_sol_all, hkl_sol))
            nbtestspots = nbtestspots + len(spots_in_center)

        if self.material_ != self.material1_:
            copy1 = np.copy(int(np.max(np.abs(hkl_sol_all))))
            copy1_min = np.copy(int(np.min(hkl_sol_all)))
            material_= self.input_params["material1_"]
            nbgrains = self.input_params["nb_grains_per_lp1"]
            hkl_sol_all = np.zeros((1,4))
            verbose=0
            for _ in range(10):
                seednumber = np.random.randint(1e6)
                tabledistancerandom, hkl_sol, \
                                        _, _, _, _, _ = self.prepare_LP(nbgrains, 0,
                                                                        material_,
                                                                        None,
                                                                        verbose,
                                                                        plotLauePattern=False,
                                                                        seed=seednumber,
                                                                        detectorparameters=self.input_params["detectorparameters"], 
                                                                        pixelsize=self.input_params["pixelsize"],
                                                                        dim1=self.input_params["dim1"],
                                                                        dim2=self.input_params["dim2"],
                                                                        removeharmonics=removeharmonics)
                                        
                spots_in_center = [sp for sp in range(len(tabledistancerandom))] # take all spots in Laue pattern
                hkl_sol_all = np.vstack((hkl_sol_all, hkl_sol))
                nbtestspots = nbtestspots + len(spots_in_center)
            hkl_sol_all = np.delete(hkl_sol_all, 0, axis =0)
            copy_ = np.copy(int(np.max(np.abs(hkl_sol_all))))
            copy_min_ = np.copy(int(np.min(hkl_sol_all)))
            self.write_to_console("Total spots created for calculating HKL bounds:"+str(nbtestspots))
            self.write_to_console("Max HKL index for "+self.material_+" :"+str(copy1))
            self.write_to_console("Min HKL index "+self.material_+" :"+str(copy1_min))
            self.write_to_console("Max HKL index for "+self.material1_+" :"+str(copy_))
            self.write_to_console("Min HKL index "+self.material1_+" :"+str(copy_min_))
            return int(copy1), int(copy_)

        self.write_to_console("Total spots created for calculating HKL bounds:"+str(nbtestspots))
        self.write_to_console("Max HKL index:"+str(np.max(hkl_sol_all)))
        self.write_to_console("Min HKL index:"+str(np.min(hkl_sol_all)))
        return int(np.max(np.abs(hkl_sol_all))), int(np.max(np.abs(hkl_sol_all)))
    
    def prepare_LP(self, nbgrains, nbgrains1, material_, material1_, verbose, plotLauePattern, seed=None, sortintensity=False,
                   detectorparameters=None, pixelsize=None, dim1=2048, dim2=2048, removeharmonics=1):
        s_tth, s_chi, s_miller_ind, s_posx, s_posy, \
                                        s_intensity, _, _ = simulatemultiplepatterns(nbgrains, nbgrains1, seed=seed, 
                                                                                    key_material=material_,
                                                                                    key_material1=material1_,
                                                                                    detectorparameters=detectorparameters,
                                                                                    pixelsize=pixelsize,
                                                                                    emin=self.emin,
                                                                                    emax=self.emax,
                                                                                    sortintensity=sortintensity, 
                                                                                    dim1=dim1,dim2=dim2,
                                                                                    removeharmonics=removeharmonics,
                                                                                    misorientation_angle=1,
                                                                                    phase_always_present=None)
        # considering all spots
        allspots_the_chi = np.transpose(np.array([s_tth/2., s_chi]))
        tabledistancerandom = np.transpose(GT.calculdist_from_thetachi(allspots_the_chi, allspots_the_chi))
        # ground truth
        hkl_sol = s_miller_ind
        return tabledistancerandom, hkl_sol, s_posx, s_posy, s_intensity, s_tth, s_chi
 
    def load_dataset(self, material_="Cu", material1_="Cu", ang_maxx=18.,step=0.1, mode=0, 
                     nb_grains=1, nb_grains1=1, grains_nb_simulate=100, data_realism = False, 
                     detectorparameters=None, pixelsize=None, type_="training",
                     var0 = 0, dim1=2048, dim2=2048, removeharmonics=1,
                     mat0_listHKl=None, mat1_listHKl=None): 
        """
        works for all symmetries now.
        """
        ## make sure directory exists
        save_directory_ = self.save_directory+"//"+type_
        if not os.path.exists(save_directory_):
            os.makedirs(save_directory_)

        try:
            with open(self.save_directory+"//classhkl_data_"+material_+".pickle", "rb") as input_file:
                classhkl, _, _, n, _, \
                    hkl_all_class, _, lattice_material, symmetry = cPickle.load(input_file)
            max_millerindex = int(n)
            max_millerindex1 = int(n)
            if material_ != material1_:
                with open(self.save_directory+"//classhkl_data_"+material1_+".pickle", "rb") as input_file:
                    classhkl1, _, _, n1, _, \
                        hkl_all_class1, _, lattice_material1, symmetry1 = cPickle.load(input_file)
                max_millerindex1 = int(n1)
        except:
            self.write_to_console("Class HKL library data not found, please run it first")
            return None

        if var0==1:
            ## add the list of hkl too in the get_material_data function
            #TODO
            codebars, angbins = get_material_data(material_ = material_, ang_maxx = ang_maxx, step = step,
                                                       hkl_ref=n, classhkl=classhkl)
            loc = np.array([ij for ij in range(len(classhkl))])

            self.write_to_console("Verifying if two different HKL class have same angular distribution (can be very time consuming depending on the symmetry)")
            index = []
            self.progress.setMaximum(len(codebars))
            list_appended = []
            count_cbs = 0
            for i, j in enumerate(codebars):
                for k, l in enumerate(codebars):
                    # if i in list_appended and k in list_appended:
                    #     continue
                    if i != k and np.all(j == l):
                        index.append((i,k))
                        string0 = "HKL's "+ str(classhkl[i])+" and "+str(classhkl[k])+" have exactly the same angular distribution."
                        self.write_to_console(string0)
                    list_appended.append(i)
                    list_appended.append(k)
                count_cbs += 1
                self.progress.setValue(count_cbs)
                QApplication.processEvents()
                  
            if len(index) == 0:
                self.write_to_console("Great! No two HKL class have same angular distribution")
                #np.savez_compressed(save_directory_+'//grain_init.npz', codebars, loc)
            else:
                self.write_to_console("Some HKL's have similar angular distribution; this will likely reduce the accuracy of the neural network; verify if symmetry matrix and other parameters are properly configured; this is just for the dictionary; keep eye on the dataset being generated for training")
                self.write_to_console("This is likely the result of the symmetry operation available in a user_defined space group; this shouldn't affect the general accuracy of the model")
                np.savez_compressed(self.save_directory+'//conflict_angular_distribution_debug.npz', codebars, index)           
            np.savez_compressed(self.save_directory+'//grain_classhkl_angbin.npz', classhkl, angbins)
                 
            if material_ != material1_:
                codebars, angbins = get_material_data(material_ = material1_, ang_maxx = ang_maxx, step = step,
                                                       hkl_ref=n1, classhkl=classhkl1)
                ind_offset = loc[-1] + 1
                loc = np.array([ind_offset + ij for ij in range(len(classhkl1))])
                self.write_to_console("Verifying if two different HKL class have same angular distribution (can be very time consuming depending on the symmetry)")
                index = []
                self.progress.setMaximum(len(codebars))
                list_appended = []
                count_cbs = 0
                for i, j in enumerate(codebars):
                    for k, l in enumerate(codebars):
                        # if i in list_appended and k in list_appended:
                        #     continue
                        if i != k and np.all(j == l):
                            index.append((i,k))
                            string0 = "HKL's "+ str(classhkl1[i])+" and "+str(classhkl1[k])+" have exactly the same angular distribution."
                            self.write_to_console(string0)
                        list_appended.append(i)
                        list_appended.append(k)
                    count_cbs += 1
                    self.progress.setValue(count_cbs)
                    QApplication.processEvents()

                if len(index) == 0:
                    self.write_to_console("Great! No two HKL class have same angular distribution")
                    #np.savez_compressed(save_directory_+'//grain_init1.npz', codebars, loc)
                else:
                    self.write_to_console("Some HKL's have similar angular distribution; this will likely reduce the accuracy of the neural network; verify if symmetry matrix and other parameters are properly configured; this is just for the dictionary; keep eye on the dataset being generated for training")
                    self.write_to_console("This is likely the result of the symmetry operation available in a user_defined space group; this shouldn't affect the general accuracy of the model")
                    np.savez_compressed(self.save_directory+'//conflict_angular_distribution1_debug.npz', codebars, index)                
                np.savez_compressed(self.save_directory+'//grain_classhkl_angbin1.npz', classhkl1, angbins)
        
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
        
        self.write_to_console("Generating "+type_+" and saving them")
        
        if material_ != material1_:
            nb_grains_list = list(range(nb_grains+1))
            nb_grains1_list = list(range(nb_grains1+1))
            list_permute = list(itertools.product(nb_grains_list, nb_grains1_list))
            list_permute.pop(0)
            max_progress = len(list_permute)*grains_nb_simulate
            
            if self.matrix_phase_always_present != None and type_ != "testing_data":
                dummy_, key_material_new = self.matrix_phase_always_present.split(';')
                if key_material_new == material_:
                    max_progress = len(list_permute)*grains_nb_simulate + (len(nb_grains1_list)-1)*grains_nb_simulate
                else:
                    max_progress = len(list_permute)*grains_nb_simulate + (len(nb_grains_list)-1)*grains_nb_simulate
        else:
            max_progress = nb_grains*grains_nb_simulate
            if self.matrix_phase_always_present != None and type_ != "testing_data":
                max_progress = nb_grains*grains_nb_simulate*2
                
        if self.include_scm:
            max_progress = max_progress + grains_nb_simulate
            if material_ != material1_:
                 max_progress = max_progress + 2*grains_nb_simulate
                     
        self.progress.setMaximum(max_progress)

        self._inputs_queue = Queue()
        self._outputs_queue = Queue()
        self._worker_process = {}
        for i in range(self.ncpu):
            self._worker_process[i]= Process(target=worker_generation, args=(self._inputs_queue, 
                                                                              self._outputs_queue, 
                                                                              i+1),)
        for i in range(self.ncpu):
            self._worker_process[i].start()            
        time.sleep(0.1)    
        
        if material_ != material1_:
            if self.modelp == "uniform":
                
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
            # print(list_permute, nb_grains, nb_grains1)
            # Idea 2 Or generate a database upto n grain LP
            values = []
            for i in range(len(list_permute)):
                ii, jj = list_permute[i]
                # print(ii,jj)
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
                    
                    if self.modelp == "uniform":
                        rand_choice = np.random.choice(len(odf_data), ii, replace=False)
                        rand_choice1 = np.random.choice(len(odf_data1), jj, replace=False)
                        data_odf_data = odf_data[rand_choice,:,:]
                        data_odf_data1 = odf_data1[rand_choice1,:,:]
                    else:
                        data_odf_data = None
                        data_odf_data1 = None

                    seednumber = np.random.randint(1e6)
                    values.append([ii, jj, material_,material1_,
                                    self.emin, self.emax, detectorparameters,
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
                                    self.modelp,
                                    self.misorientation_angle,
                                    max_millerindex,max_millerindex1,
                                    self.general_diff_rules,
                                    self.crystal, 
                                    self.crystal1,
                                    None,
                                    mat0_listHKl, 
                                    mat1_listHKl])
                    
                    if self.matrix_phase_always_present != None and \
                        type_ != "testing_data":
                        
                        dummy_, key_material_new = self.matrix_phase_always_present.split(';')
                        
                        if key_material_new == material_ and ii == 0:
                            values.append([0, jj, material_,material1_,
                                            self.emin, self.emax, detectorparameters,
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
                                            self.modelp,
                                            self.misorientation_angle,
                                            max_millerindex,max_millerindex1,
                                            self.general_diff_rules,
                                            self.crystal, 
                                            self.crystal1,
                                            self.matrix_phase_always_present,
                                            mat0_listHKl, 
                                            mat1_listHKl])

                        elif key_material_new == material1_ and jj == 0:
                            values.append([ii, 0, material_,material1_,
                                            self.emin, self.emax, detectorparameters,
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
                                            self.modelp,
                                            self.misorientation_angle,
                                            max_millerindex,max_millerindex1,
                                            self.general_diff_rules,
                                            self.crystal, 
                                            self.crystal1,
                                            self.matrix_phase_always_present,
                                            mat0_listHKl, 
                                            mat1_listHKl])
                # print(ii,jj)
                    
            chunks = chunker_list(values, self.ncpu)
            chunks_mp = list(chunks)

            if self.include_scm:
                meta = {'t1':time.time(),
                        'flag':0}
            else:
                meta = {'t1':time.time(),
                        'flag':1}
            for ijk in range(int(self.ncpu)):
                self._inputs_queue.put((chunks_mp[ijk], self.ncpu, meta))

        else:
            # Idea 2 Or generate a database upto n grain LP
            if self.modelp == "uniform":
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
                    
                    if self.modelp == "uniform":
                        rand_choice = np.random.choice(len(odf_data), i+1, replace=False)
                        data_odf_data = odf_data[rand_choice,:,:]
                        data_odf_data1 = None
                    else:
                        data_odf_data = None
                        data_odf_data1 = None
                        
                    seednumber = np.random.randint(1e6)
                    values.append([i+1, 0, material_,material1_,
                                    self.emin, self.emax, detectorparameters,
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
                                    self.modelp,
                                    self.misorientation_angle,
                                    max_millerindex,max_millerindex1,
                                    self.general_diff_rules,
                                    self.crystal, 
                                    self.crystal1,
                                    None,
                                    mat0_listHKl, 
                                    mat1_listHKl])
                    
                    if self.matrix_phase_always_present != None and \
                        type_ != "testing_data":
                        values.append([i+1, 0, material_,material1_,
                                        self.emin, self.emax, detectorparameters,
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
                                        self.modelp,
                                        self.misorientation_angle,
                                        max_millerindex,max_millerindex1,
                                        self.general_diff_rules,
                                        self.crystal, 
                                        self.crystal1,
                                        self.matrix_phase_always_present,
                                        mat0_listHKl, 
                                        mat1_listHKl])
                    
            chunks = chunker_list(values, self.ncpu)
            chunks_mp = list(chunks)
            
            if self.include_scm:
                meta = {'t1':time.time(),
                        'flag':0}
            else:
                meta = {'t1':time.time(),
                        'flag':1}
            for ijk in range(int(self.ncpu)):
                self._inputs_queue.put((chunks_mp[ijk], self.ncpu, meta))

        if self.include_scm:
            self.write_to_console("Generating small angle misorientation single crystals")  
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
                                        self.emin, self.emax, detectorparameters,
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
                                        None, None, self.modelp,
                                        self.misorientation_angle,
                                        max_millerindex,max_millerindex1,
                                        self.general_diff_rules,
                                        self.crystal, 
                                        self.crystal1,
                                        None,
                                        mat0_listHKl, 
                                        mat1_listHKl])
                
                if material_ != material1_:
                    seednumber = np.random.randint(1e6)
                    values.append([0, 1, material_,material1_,
                                        self.emin, self.emax, detectorparameters,
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
                                        None, None, self.modelp,
                                        self.misorientation_angle,
                                        max_millerindex,max_millerindex1,
                                        self.general_diff_rules,
                                        self.crystal, 
                                        self.crystal1,
                                        None,
                                        mat0_listHKl, 
                                        mat1_listHKl])
                    
                    ### include slightly misoriented two crystals of different materails
                    seednumber = np.random.randint(1e6)
                    values.append([1, 1, material_,material1_,
                                        self.emin, self.emax, detectorparameters,
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
                                        None, None, self.modelp,
                                        self.misorientation_angle,
                                        max_millerindex,max_millerindex1,
                                        self.general_diff_rules,
                                        self.crystal, 
                                        self.crystal1,
                                        None,
                                        mat0_listHKl, 
                                        mat1_listHKl])
                    
            chunks = chunker_list(values, self.ncpu)
            chunks_mp = list(chunks)

            meta = {'t1':time.time(),
                    'flag':1}
            for ijk in range(int(self.ncpu)):
                self._inputs_queue.put((chunks_mp[ijk], self.ncpu, meta))
                
        self.max_progress = max_progress
        while True:
            count = 0
            for i in range(self.ncpu):
                if not self._worker_process[i].is_alive():
                    self._worker_process[i].join()
                    count += 1
                else:
                    time.sleep(0.1)
                    self.progress.setValue(self.update_progress)
                    QApplication.processEvents()
                    
            if count == self.ncpu:
                self.progress.setValue(self.max_progress)
                QApplication.processEvents()
                return
        
    def update_data_mp(self):
        if not self._outputs_queue.empty():
            self.timermp.blockSignals(True)
            r_message = self._outputs_queue.get()
            self.update_progress = self.update_progress + r_message
            self.timermp.blockSignals(False)

    def generate_training_data(self):
        ### using MP libraries
        self.ncpu = cpu_count_user
        self.write_to_console("Using Multiprocessing ("+str(self.ncpu)+" cpus) for generation of simulated Laue patterns for training")
        self._inputs_queue = Queue()
        self._outputs_queue = Queue()
        ## Update data from multiprocessing
        self.update_progress = 0
        self.max_progress = 0
        self.timermp = QtCore.QTimer()
        self.timermp.setInterval(100) ## check every second (update the list of files in folder)
        self.timermp.timeout.connect(self.update_data_mp)
        self.timermp.start()
        
        self.write_to_console("Generating training dataset")
        self.status.showMessage("Training dataset generation in progress!")
        
        if self.input_params["hkl_max_identify"] == "auto" and self.input_params["hkl_max_identify1"] != "auto":
            self.write_to_console("Calculating the HKL bounds for training dataset")
            self.n, _ = self.temp_HKL(removeharmonics=1)
        elif self.input_params["hkl_max_identify"] == "auto" and self.input_params["hkl_max_identify1"] == "auto":
            self.write_to_console("Calculating the HKL bounds for training dataset")
            self.n, self.n1 = self.temp_HKL(removeharmonics=1)
        elif self.input_params["hkl_max_identify"] != "auto" and self.input_params["hkl_max_identify1"] == "auto":
            self.write_to_console("Calculating the HKL bounds for training dataset")
            _, self.n1 = self.temp_HKL(removeharmonics=1)
            
        ## generate reference HKL library      
        self.write_to_console("Directory for training dataset is : "+self.save_directory)
        ## procedure for generation of GROUND TRUTH classes
        # =============================================================================
        # VERY IMPORTANT; TAKES Significant time; verify again for other symmetries
        # =============================================================================
        # mat0_listHKl = [(1,10,1),(11,15,12)]
        self.mat0_listHKl = None
        self.mat1_listHKl = None
        generate_classHKL(self.n, self.rules, self.lattice_material, self.symmetry, self.material_, \
             self.crystal, self.SG, self.general_diff_rules, self.save_directory, \
                 self.write_to_console, self.progress, QApplication, self.maximum_angle_to_search, self.step_for_binning,
                 self.mat0_listHKl)
        if self.material_ != self.material1_:
            # mat1_listHKl = [(1,10,1),(11,15,12)]
            self.mat1_listHKl = None
            generate_classHKL(self.n1, self.rules1, self.lattice_material1, self.symmetry1, self.material1_, \
                 self.crystal1, self.SG1, self.general_diff_rules1, self.save_directory, \
                     self.write_to_console, self.progress, QApplication, self.maximum_angle_to_search, self.step_for_binning,
                     self.mat1_listHKl)
        
        ############ GENERATING TRAINING DATA  
        self.update_progress = 0
        self.max_progress = 0
        self.load_dataset(material_=self.material_, material1_=self.material1_, ang_maxx=self.maximum_angle_to_search,
                          step=self.step_for_binning, mode=self.mode_of_analysis, 
                          nb_grains=self.nb_grains_per_lp,nb_grains1=self.nb_grains_per_lp1,
                          grains_nb_simulate=self.grains_nb_simulate,
                          data_realism = True, detectorparameters=self.detectorparameters, 
                          pixelsize=self.pixelsize, type_="training_data", var0=1,
                          dim1=self.input_params["dim1"], dim2=self.input_params["dim2"], removeharmonics=1,
                          mat0_listHKl=self.mat0_listHKl, mat1_listHKl=self.mat1_listHKl)
        # ############ GENERATING TESTING DATA
        self.update_progress = 0
        self.max_progress = 0
        self.load_dataset(material_=self.material_, material1_=self.material1_, ang_maxx=self.maximum_angle_to_search,
                          step=self.step_for_binning, mode=self.mode_of_analysis, 
                          nb_grains=self.nb_grains_per_lp,nb_grains1=self.nb_grains_per_lp1,
                          grains_nb_simulate=self.grains_nb_simulate//self.factor,
                          data_realism = True, detectorparameters=self.detectorparameters, 
                          pixelsize=self.pixelsize, type_="testing_data", var0=0,
                          dim1=self.input_params["dim1"], dim2=self.input_params["dim2"], removeharmonics=1,
                          mat0_listHKl=self.mat0_listHKl, mat1_listHKl=self.mat1_listHKl)
        
        ## write MTEX data with training orientation
        try:
            write_training_testing_dataMTEX(self.save_directory,self.material_,self.material1_,
                                            self.lattice_material,self.lattice_material1,
                                            material0_lauegroup, material1_lauegroup)
        except:
            print("Error writing the MTEX file of training and testing data")
            self.write_to_console("Error writing the MTEX file of training and testing data")
        
        if self.generate_additional_data:
            print("In development; generating a combination of existing dataset")        
        
        self.status.showMessage("Training dataset generation completed with multi CPUs!")
        
        rmv_freq_class(self.freq_rmv, self.elements, self.freq_rmv1, self.elements1,\
                       self.save_directory, self.material_, self.material1_, self.write_to_console,\
                       self.progress, QApplication, self.mat0_listHKl, self.mat1_listHKl)
        self.write_to_console("See the class occurances above and choose appropriate frequency removal parameter to train quickly the network by having few output classes!, if not continue as it is.")
        self.write_to_console("Press Train network button to Train")
        self.train_nn.setEnabled(True)
        self.timermp.stop()
        self.generate_nn.setEnabled(False)
        
    def train_neural_network(self,):
        self.status.showMessage("Neural network training in progress!")
        self.train_nn.setEnabled(False)
        rmv_freq_class(self.freq_rmv, self.elements, self.freq_rmv1, self.elements1,\
                       self.save_directory, self.material_, self.material1_, self.write_to_console,\
                       self.progress, QApplication, self.mat0_listHKl, self.mat1_listHKl)
        
        self.classhkl = np.load(self.save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_0"]
        self.angbins = np.load(self.save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_1"]
        self.loc_new = np.load(self.save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_2"]
        with open(self.save_directory+"//class_weights.pickle", "rb") as input_file:
            class_weights = cPickle.load(input_file)
        self.class_weights = class_weights[0]
        
        ## load model and train
        if self.architecture == "FFNN":
            self.model = model_arch_general(len(self.angbins)-1, len(self.classhkl),
                                                 kernel_coeff= self.kernel_coeff, bias_coeff=self.bias_coeff, 
                                                 lr=self.learning_rate, write_to_console=self.write_to_console)
        elif self.architecture == "1D_CNN":
            self.model = model_arch_CNN_DNN_optimized(len(self.angbins)-1, 
                                                     layer_activation="relu", 
                                                     output_activation="softmax",
                                                     dropout=0.3,
                                                     stride = [1,1],
                                                     kernel_size = [5,5],
                                                     pool_size=[2,2],
                                                     CNN_layers = 3,
                                                     CNN_filters = [32,64,128],
                                                     DNN_layers = 0,
                                                     DNN_filters = [100],
                                                     output_neurons = len(self.classhkl),
                                                     learning_rate = self.learning_rate,
                                                     output="CNN",
                                                     write_to_console=self.write_to_console, 
                                                     verbose=1)
        elif self.architecture == "1D_CNN_DNN":
            self.model = model_arch_CNN_DNN_optimized(len(self.angbins)-1, 
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
                                                     output_neurons = len(self.classhkl),
                                                     learning_rate = self.learning_rate,
                                                     output="CNN_DNN",
                                                     write_to_console=self.write_to_console, 
                                                     verbose=1)
        elif self.architecture == "User_defined":
            self.model = user_defined_model(len(self.angbins)-1, len(self.classhkl),
                                                 kernel_coeff= self.kernel_coeff, bias_coeff=self.bias_coeff, 
                                                 lr=self.learning_rate, write_to_console=self.write_to_console)
        self.train_model()
        self.trialtoolbar1.setEnabled(True)
        self.predict_lnn.setEnabled(True)
        self.status.showMessage("Neural network training completed!")
      
    def train_model(self, prefix="", tag = 0):
        if tag == 2:
            ## retraining from file
            try:
                self.classhkl = np.load(self.save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_0"]
                self.angbins = np.load(self.save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_1"]
                self.loc_new = np.load(self.save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_2"]
                with open(self.save_directory+"//class_weights.pickle", "rb") as input_file:
                    class_weights = cPickle.load(input_file)
                self.class_weights = class_weights[0]
                ## need to compile again if loaded from file, better to just call the class, if architecture is same
                self.write_to_console("Constructing model")
                # self.model = model_arch_general(len(self.angbins)-1, len(self.classhkl),
                #                                      kernel_coeff= self.kernel_coeff, bias_coeff=self.bias_coeff, 
                #                                      lr=self.learning_rate, write_to_console=self.write_to_console)
                
                if self.architecture == "FFNN":
                    self.model = model_arch_general(len(self.angbins)-1, len(self.classhkl),
                                                         kernel_coeff= self.kernel_coeff, bias_coeff=self.bias_coeff, 
                                                         lr=self.learning_rate, write_to_console=self.write_to_console)
                elif self.architecture == "1D_CNN":
                    self.model = model_arch_CNN_DNN_optimized(len(self.angbins)-1, 
                                                             layer_activation="relu", 
                                                             output_activation="softmax",
                                                             dropout=0.3,
                                                             stride = [1,1],
                                                             kernel_size = [5,5],
                                                             pool_size=[2,2],
                                                             CNN_layers = 3,
                                                             CNN_filters = [32,64,128],
                                                             DNN_layers = 0,
                                                             DNN_filters = [100],
                                                             output_neurons = len(self.classhkl),
                                                             learning_rate = self.learning_rate,
                                                             output="CNN",
                                                             write_to_console=self.write_to_console, 
                                                             verbose=1)
                elif self.architecture == "1D_CNN_DNN":
                    self.model = model_arch_CNN_DNN_optimized(len(self.angbins)-1, 
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
                                                             output_neurons = len(self.classhkl),
                                                             learning_rate = self.learning_rate,
                                                             output="CNN_DNN",
                                                             write_to_console=self.write_to_console, 
                                                             verbose=1)
                elif self.architecture == "User_defined":
                    self.model = user_defined_model(len(self.angbins)-1, len(self.classhkl),
                                                         kernel_coeff= self.kernel_coeff, bias_coeff=self.bias_coeff, 
                                                         lr=self.learning_rate, write_to_console=self.write_to_console)
                
                
                list_of_files = glob.glob(self.save_directory+'//*.h5')
                latest_file = max(list_of_files, key=os.path.getctime)
                self.write_to_console("Taking the latest Weight file from the Folder: " + latest_file)
                load_weights = latest_file
                self.model.load_weights(load_weights)
                self.write_to_console("Uploading weights to model")
                self.write_to_console("All model files found and loaded")
            except:
                self.write_to_console("Model directory is not proper or files are missing. please configure the params")
                return

        ## temp function to quantify the spots and classes present in a batch
        batch_size = self.input_params["batch_size"] 
        trainy_inbatch = array_generator_verify(self.save_directory+"//training_data", batch_size, 
                                                len(self.classhkl), self.loc_new, self.write_to_console)
        self.write_to_console("Number of spots in a batch of %i files : %i" %(batch_size, len(trainy_inbatch)))
        self.write_to_console("Min, Max class ID is %i, %i" %(np.min(trainy_inbatch), np.max(trainy_inbatch)))
        # try varying batch size and epochs
        epochs = self.input_params["epochs"] 
        ## Batch loading for numpy grain files (Keep low value to avoid overcharging the RAM)
        if self.material_ != self.material1_:
            nb_grains_list = list(range(self.nb_grains_per_lp+1))
            nb_grains1_list = list(range(self.nb_grains_per_lp1+1))
            list_permute = list(itertools.product(nb_grains_list, nb_grains1_list))
            list_permute.pop(0)
            steps_per_epoch = (len(list_permute) * self.grains_nb_simulate)//batch_size
        else:
            steps_per_epoch = int((self.nb_grains_per_lp * self.grains_nb_simulate) / batch_size)
            
        val_steps_per_epoch = int(steps_per_epoch / self.factor)
        if steps_per_epoch == 0:
            steps_per_epoch = 1
        if val_steps_per_epoch == 0:
            val_steps_per_epoch = 1   
        ## Load generator objects from filepaths
        training_data_generator = array_generator(self.save_directory+"//training_data", batch_size, \
                                                  len(self.classhkl), self.loc_new, self.write_to_console)
        testing_data_generator = array_generator(self.save_directory+"//testing_data", batch_size, \
                                                 len(self.classhkl), self.loc_new, self.write_to_console)
        ######### TRAIN THE DATA
        self.progress.setMaximum(epochs*steps_per_epoch)
        # from clr_callback import CyclicLR
        # clr = CyclicLR(base_lr=0.0005, max_lr=0.001, step_size=steps_per_epoch*5, mode='triangular')
        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=2)
        # es = EarlyStopping(monitor='categorical_crossentropy', patience=2)
        ms = ModelCheckpoint(self.save_directory+"//best_val_acc_model.h5", monitor='val_accuracy', 
                             mode='max', save_best_only=True)
        
        # model save directory and filename
        if self.material_ != self.material1_:
            model_name = self.save_directory+"//model_"+self.material_+"_"+self.material1_+prefix
        else:
            model_name = self.save_directory+"//model_"+self.material_+prefix
            
        log = LoggingCallback(self.write_to_console, self.progress, QApplication, self.model, model_name)

        stats_model = self.model.fit(
                                    training_data_generator, 
                                    epochs=epochs, 
                                    steps_per_epoch=steps_per_epoch,
                                    validation_data=testing_data_generator,
                                    validation_steps=val_steps_per_epoch,
                                    verbose=1,
                                    class_weight=self.class_weights,
                                    callbacks=[es, ms, log] # es, ms, clr
                                    )
        
        self.progress.setValue(epochs*steps_per_epoch)
        QApplication.processEvents() 
        # Save model config and weightsp
        if tag == 0:
            ## new trained model, save files
            model_json = self.model.to_json()
            with open(model_name+".json", "w") as json_file:
                json_file.write(model_json)            
        # serialize weights to HDF5
        self.model.save_weights(model_name+".h5")
        self.write_to_console("Saved model to disk")

        self.write_to_console( "Training Accuracy: "+str( stats_model.history['accuracy'][-1]))
        self.write_to_console( "Training Loss: "+str( stats_model.history['loss'][-1]))
        self.write_to_console( "Validation Accuracy: "+str( stats_model.history['val_accuracy'][-1]))
        self.write_to_console( "Validation Loss: "+str( stats_model.history['val_loss'][-1]))
        
        epochs = range(1, len(self.model.history.history['loss']) + 1)
        fig, ax = plt.subplots(1,2)
        ax[0].plot(epochs, self.model.history.history['loss'], 'r', label='Training loss')
        ax[0].plot(epochs, self.model.history.history['val_loss'], 'r', ls="dashed", label='Validation loss')
        ax[0].legend()
        ax[1].plot(epochs, self.model.history.history['accuracy'], 'g', label='Training Accuracy')
        ax[1].plot(epochs, self.model.history.history['val_accuracy'], 'g', ls="dashed", label='Validation Accuracy')
        ax[1].legend()
        if self.material_ != self.material1_:
            plt.savefig(self.save_directory+"//loss_accuracy_"+self.material_+"_"+self.material1_+prefix+".png", bbox_inches='tight',format='png', dpi=1000)
        else:
            plt.savefig(self.save_directory+"//loss_accuracy_"+self.material_+prefix+".png", bbox_inches='tight',format='png', dpi=1000)
        plt.close()
        
        if self.material_ != self.material1_:
            text_file = open(self.save_directory+"//loss_accuracy_logger_"+self.material_+"_"+self.material1_+prefix+".txt", "w")
        else:
            text_file = open(self.save_directory+"//loss_accuracy_logger_"+self.material_+prefix+".txt", "w")

        text_file.write("# EPOCH, LOSS, VAL_LOSS, ACCURACY, VAL_ACCURACY" + "\n")
        for inj in range(len(epochs)):
            string1 = str(epochs[inj]) + ","+ str(self.model.history.history['loss'][inj])+\
                    ","+str(self.model.history.history['val_loss'][inj])+","+str(self.model.history.history['accuracy'][inj])+\
                    ","+str(self.model.history.history['val_accuracy'][inj])+" \n"  
            text_file.write(string1)
        text_file.close()
        
        x_test, y_test = vali_array(self.save_directory+"//testing_data", 50, len(self.classhkl), self.loc_new,
                                    self.write_to_console)
        y_test = np.argmax(y_test, axis=-1)
        y_pred = np.argmax(self.model.predict(x_test), axis=-1)
        self.write_to_console(classification_report(y_test, y_pred))
        self.write_to_console( "Training is Completed; You can use the Retrain function to run for more epoch with varied batch size")
        self.write_to_console( "Training is Completed; You can use the Prediction and Live Prediction module now")

    def grid_search_hyperparams(self,): 
        classhkl = np.load(self.save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_0"]
        angbins = np.load(self.save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_1"]
        loc_new = np.load(self.save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_2"]
        with open(self.save_directory+"//class_weights.pickle", "rb") as input_file:
            class_weights = cPickle.load(input_file)
        class_weights = class_weights[0]
        
        batch_size = self.input_params["batch_size"] 
        trainy_inbatch = array_generator_verify(self.save_directory+"//training_data", batch_size, 
                                                len(classhkl), loc_new, self.write_to_console)
        self.write_to_console("Number of spots in a batch of %i files : %i" %(batch_size, len(trainy_inbatch)))
        self.write_to_console("Min, Max class ID is %i, %i" %(np.min(trainy_inbatch), np.max(trainy_inbatch)))
        self.write_to_console("Starting hypergrid optimization: looking in a grid to optimize the learning rate and regularization coefficients.")
        # try varying batch size and epochs
        epochs = 1 #self.input_params["epochs"] 
        ## Batch loading for numpy grain files (Keep low value to avoid overcharging the RAM)
        steps_per_epoch = int((self.nb_grains_per_lp * self.grains_nb_simulate)/batch_size)
        val_steps_per_epoch = int(steps_per_epoch /self.factor)
        if steps_per_epoch == 0:
            steps_per_epoch = 1
        if val_steps_per_epoch == 0:
            val_steps_per_epoch = 1
        ## Load generator objects from filepaths
        training_data_generator = array_generator(self.save_directory+"//training_data", batch_size, \
                                                  len(classhkl), loc_new, self.write_to_console)
        testing_data_generator = array_generator(self.save_directory+"//testing_data", batch_size, \
                                                 len(classhkl), loc_new, self.write_to_console)
        # grid search values
        values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        all_train, all_test = list(), list()
        all_trainL, all_testL = list(), list()
        parameters = list()
        text_file = open(self.save_directory+"//parameter_hypergrid_"+self.material_+".txt", "w")
        text_file.write("# Iter, Learning_Rate, Bias_Coeff, Kernel_Coeff, Train_Acc, Train_Loss, Test_Acc, Test_Loss, LR_index, BC_index, KC_index" + "\n")
        self.progress.setMaximum(len(values)*len(values)*len(values))
        iter_cnt= 0 
        for i, param in enumerate(values):
            for j, param1 in enumerate(values):
                for k, param2 in enumerate(values):
                    # define model
                    iter_cnt += 1
                    model = model_arch_general(len(angbins)-1, len(classhkl), 
                                                       kernel_coeff = param2, 
                                                       bias_coeff = param1,
                                                       lr = param, verbose=0,
                                                       write_to_console=self.write_to_console)
                    # fit model
                    stats_model = model.fit(
                                            training_data_generator, 
                                            epochs=epochs, 
                                            steps_per_epoch=steps_per_epoch,
                                            validation_data=testing_data_generator,
                                            validation_steps=val_steps_per_epoch,
                                            verbose=0,
                                            class_weight=class_weights,
                                            )
                    # evaluate the model
                    train_acc = stats_model.history['accuracy'][-1]
                    test_acc = stats_model.history['val_accuracy'][-1]
                    train_loss = stats_model.history['loss'][-1]
                    test_loss = stats_model.history['val_loss'][-1]
                    all_train.append(train_acc)
                    all_test.append(test_acc)
                    all_trainL.append(train_loss)
                    all_testL.append(test_loss)
                    parameters.append([param,param1,param2])   
                    string1 = str(iter_cnt) +","+ str(param) + ","+ str(param1)+\
                                ","+str(param2)+","+str(train_acc)+\
                                ","+str(train_loss)+ ","+str(test_acc)+","+str(test_loss)+","+ str(i) + ","+ str(j)+\
                                    ","+str(k)+ " \n"  
                    text_file.write(string1)                  
                    self.progress.setValue(iter_cnt)
                    QApplication.processEvents()         
        text_file.close()

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, subplot=1, mat_bool=True):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        
        if mat_bool:
            self.axes = self.fig.add_subplot(131)
            self.axes1 = self.fig.add_subplot(132)
            self.axes2 = self.fig.add_subplot(133)
        else:
            self.axes = self.fig.add_subplot(141)
            self.axes1 = self.fig.add_subplot(142)
            self.axes2 = self.fig.add_subplot(143)
            self.axes3 = self.fig.add_subplot(144)
        super(MplCanvas, self).__init__(self.fig)   
    
class MplCanvas1(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas1, self).__init__(self.fig)

class MplCanvas2(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(231)
        self.axes1 = self.fig.add_subplot(232)
        self.axes2 = self.fig.add_subplot(233)
        self.axes3 = self.fig.add_subplot(234)
        self.axes4 = self.fig.add_subplot(235)
        self.axes5 = self.fig.add_subplot(236)
        super(MplCanvas2, self).__init__(self.fig)   
        
class MyPopup_image_v1(QWidget):
    def __init__(self, ix, iy, file, data, ccd_label, function_predict, image_no, detectorparameters):
        QWidget.__init__(self)
        
        self.layout = QVBoxLayout() # QGridLayout()
        self.canvas = MplCanvas1(self, width=10, height=10, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.iy,self.ix,self.file = iy, ix, file
        self.ccd_label = ccd_label
        self.data = data
        self.pix_x, self.pix_y = [], []
        self.peakXY = []
        self.scatter = None
        self.function_predict = function_predict
        self.image_no = image_no
        self.detectorparameters = detectorparameters
        # set the layout
        self.layout.addWidget(self.toolbar, 0)
        self.layout.addWidget(self.canvas, 100)

        self.ImaxDisplayed = 2*np.average(data)
        self.IminDisplayed = np.average(data) - 0.2*np.average(data)

        self.slider = QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setRange(0, np.max(data))
        self.layout.addWidget(self.slider)
        self.slider.setValue(self.IminDisplayed)
        self.slider.valueChanged[int].connect(self.sliderMin)
        
        self.slider1 = QSlider(QtCore.Qt.Horizontal, self)
        self.slider1.setRange(10, np.max(data))
        self.layout.addWidget(self.slider1)
        self.slider1.setValue(self.ImaxDisplayed)
        self.slider1.valueChanged[int].connect(self.sliderMax)

        self.bkg_treatment = QLineEdit()
        self.bkg_treatment.setText("A-B")
        
        self.peak_params = QLineEdit()
        self.peak_params.setText("500,15,15,lauetools")
        
        self.predict_params = QLineEdit()
        self.predict_params.setText("0.85, 0.20, v2, 8, 0.2, 50")
        
        try:
            config_setting = configparser.ConfigParser()
            filepath = resource_path('settings.ini')
            config_setting.read(filepath)
            residues_threshold = float(config_setting.get('CALLER', 'residues_threshold'))
            nb_spots_global_threshold = int(config_setting.get('CALLER', 'nb_spots_global_threshold'))
            option_global = config_setting.get('CALLER', 'option_global')
            nb_spots_consider = int(config_setting.get('CALLER', 'nb_spots_consider'))
            intensity_threshold = int(float(config_setting.get('CALLER', 'intensity')))
            boxsize = int(float(config_setting.get('CALLER', 'boxsize')))
            FitPixelDev_global123 = int(float(config_setting.get('CALLER', 'pixdev')))
            softmax_threshold_global123 = float(config_setting.get('CALLER', 'cap_softmax'))
            cap_matchrate123 = float(config_setting.get('CALLER', 'cap_mr'))
            strain_free_parameters = config_setting.get('CALLER', 'strain_free_parameters').split(",")
            additional_expression = config_setting.get('CALLER', 'additional_expression').split(",")
            mode_peaksearch = str(config_setting.get('CALLER', 'mode_peaksearch'))
            use_om_user = config_setting.get('CALLER', 'use_om_user')
            path_user_OM = config_setting.get('CALLER', 'path_user_OM')
            if intensity_threshold != None and boxsize != None and FitPixelDev_global123!=None:
                self.peak_params.setText(str(intensity_threshold)+","+str(boxsize)+","+str(FitPixelDev_global123)+","+str(mode_peaksearch))
            self.predict_params.setText(str(float(softmax_threshold_global123))+","+str(float(cap_matchrate123))+
                                        ","+option_global+","+
                                        str(nb_spots_global_threshold)+","+str(residues_threshold)+
                                        ","+str(nb_spots_consider))
            self.use_om_user = use_om_user
            self.path_user_OM = path_user_OM
            self.residues_threshold = residues_threshold
            self.nb_spots_global_threshold = nb_spots_global_threshold
            self.option_global = option_global
            self.nb_spots_consider = nb_spots_consider
            self.intensity_threshold = intensity_threshold
            self.boxsize = boxsize
            self.FitPixelDev_global123 = FitPixelDev_global123
            self.softmax_threshold_global123 = softmax_threshold_global123
            self.cap_matchrate123 = cap_matchrate123
            self.strain_free_parameters = strain_free_parameters
            self.additional_expression = additional_expression
            self.mode_peaksearch = mode_peaksearch
        except:
            self.use_om_user = "false"
            self.path_user_OM = ""
            self.residues_threshold = 0.5
            self.nb_spots_global_threshold = 8
            self.option_global = "v2"
            self.nb_spots_consider = 100
            self.intensity_threshold = 200
            self.boxsize = 15
            self.FitPixelDev_global123 = 15
            self.softmax_threshold_global123 = 0.85
            self.cap_matchrate123 = 0.2
            self.strain_free_parameters = ["rotx", "roty", "rotz", "alpha", "beta", "gamma", "b", "c"]
            self.additional_expression = ["none"]
            self.mode_peaksearch = "LaueTools"
            print("error with setting config file, returning the settings to default values")
            
        self.corrected_data = None
        self.image_mode = QComboBox()
        mode_ = ["raw","bkg_corrected"]
        for s in mode_:
            self.image_mode.addItem(s)
        
        self.peaksearch_mode = QComboBox()
        peaksearchmode_ = ["lauetools","LOG"]
        for s in peaksearchmode_:
            self.peaksearch_mode.addItem(s)
            
        self.btn_peak_search = QPushButton("Peak search")
        self.btn_peak_search.clicked.connect(self.peak_search)
        self.predicthkl = QPushButton("run prediction")
        self.predicthkl.clicked.connect(self.prediction_hkl)
        self.plothough = QPushButton("Show hough transform results")
        self.plothough.clicked.connect(self.plot_houghlines)
        self.refresh_plot = QPushButton("Refresh plot")
        self.refresh_plot.clicked.connect(self.refresh_plots)
        self.propagate_button = QPushButton("Propagate values")
        self.propagate_button.clicked.connect(self.propagate)
        self.btn_peak_search.setEnabled(True)
        self.predicthkl.setEnabled(False)
        self.propagate_button.setEnabled(False)
        self.plothough.setEnabled(False)
        
        ## add some buttons here for peak search and peak options
        ## and then predict again with neural network with its options
        ## send the data back to update the main variables
        formLayout = QFormLayout()
        formLayout.addRow('Background treatment expression', self.bkg_treatment)
        formLayout.addRow('Intensity; box size; pix dev; mode_PeakSearch', self.peak_params)
        formLayout.addRow('Image mode', self.image_mode)
        formLayout.addRow('Peak search mode', self.peaksearch_mode)
        formLayout.addRow('softmax acc.; Mr threshold; 4hyperparams', self.predict_params)
        formLayout.addRow(self.btn_peak_search, self.predicthkl)
        formLayout.addRow(self.refresh_plot, self.plothough)
        formLayout.addRow("Propagate the new parameters", self.propagate_button)
        self.layout.addLayout(formLayout)
        self.setLayout(self.layout)
        # compute background
        self.show_bkg_corrected_img()
        self.draw_something()
        
    def refresh_plots(self):
        self.draw_something()
        
    def show_bkg_corrected_img(self):
        backgroundimage = ImProc.compute_autobackground_image(self.data, boxsizefilter=10)
        self.corrected_data = ImProc.computefilteredimage(self.data, backgroundimage, self.ccd_label, usemask=True,
                                                            formulaexpression=self.bkg_treatment.text())
    
    def propagate(self):
        filepath = resource_path('settings.ini')
        print("Settings path is "+filepath)
        config_setting1 = configparser.ConfigParser()
        config_setting1.read(filepath)
        config_setting1.set('CALLER', 'residues_threshold',str(self.residues_threshold))
        config_setting1.set('CALLER', 'nb_spots_global_threshold',str(self.nb_spots_global_threshold))
        config_setting1.set('CALLER', 'option_global',self.option_global)
        config_setting1.set('CALLER', 'use_om_user',self.use_om_user)
        config_setting1.set('CALLER', 'nb_spots_consider',str(self.nb_spots_consider))
        config_setting1.set('CALLER', 'path_user_OM',self.path_user_OM)
        config_setting1.set('CALLER', 'intensity', str(self.intensity_threshold))
        config_setting1.set('CALLER', 'boxsize', str(self.boxsize))
        config_setting1.set('CALLER', 'pixdev', str(self.FitPixelDev_global123))
        config_setting1.set('CALLER', 'cap_softmax', str(self.softmax_threshold_global123))
        config_setting1.set('CALLER', 'cap_mr', str(self.cap_matchrate123))
        config_setting1.set('CALLER', 'strain_free_parameters', ",".join(self.strain_free_parameters))
        config_setting1.set('CALLER', 'additional_expression', ",".join(self.additional_expression))
        config_setting1.set('CALLER', 'mode_peaksearch', self.mode_peaksearch)
        with open(filepath, 'w') as configfile:
            config_setting1.write(configfile)
        print("Config settings updated")
    
    def neighbor_UB(self):
        #TODO
        print("UB matrix indexation from neighbors in development")
        pass
    
    def plot_houghlines(self):
        print("Plotting Hough lines")
        s_ix = np.argsort(self.peakXY[:, 2])[::-1]
        self.peakXY = self.peakXY[s_ix]
        pixelsize = dictLT.dict_CCD[self.ccd_label][1]
        twicetheta, chi = Lgeo.calc_uflab(self.peakXY[:,0], self.peakXY[:,1], self.detectorparameters,
                                            returnAngles=1,
                                            pixelsize=pixelsize,
                                            kf_direction='Z>0')        
        # Classic straight-line Hough transform
        imageGNO, nbpeaks, halfdiagonal = computeGnomonicImage(twicetheta, chi)
        hough, theta_h, d_h = hough_line(imageGNO)
        
        # Generating figure 1
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        ax = axes.ravel()
        ax[0].imshow(self.data,interpolation="nearest",vmin=self.IminDisplayed, vmax=self.ImaxDisplayed)
        ax[0].set_title('Input image')
        ax[0].set_axis_off()

        ax[1].imshow(np.log(1 + hough),
                     extent=[np.rad2deg(theta_h[-1]), np.rad2deg(theta_h[0]), d_h[-1], d_h[0]],
                     cmap=cm.gray, aspect=1/1.5)
        ax[1].set_title('Hough transform')
        ax[1].set_xlabel('Angles (degrees)')
        ax[1].set_ylabel('Distance (pixels)')
        ax[1].axis('image')

        ax[2].imshow(imageGNO, cmap=cm.gray)
        for _, angle, dist in zip(*hough_line_peaks(hough, theta_h, d_h)):
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - imageGNO.shape[1] * np.cos(angle)) / np.sin(angle)
            ax[2].plot((0, imageGNO.shape[1]), (y0, y1), '-r', lw=0.5)
        ax[2].set_xlim((0, imageGNO.shape[1]))
        ax[2].set_ylim((imageGNO.shape[0], 0))
        ax[2].set_axis_off()
        ax[2].set_title('Detected lines')
        plt.tight_layout()
        plt.show()
    
    
    ## add LOG method of peak search from skimage routine
    def peak_search(self):
        self.propagate_button.setEnabled(False)
        intens = int(float(self.peak_params.text().split(",")[0]))
        bs = int(float(self.peak_params.text().split(",")[1]))
        pixdev = int(float(self.peak_params.text().split(",")[2]))
        mode_peaksearch = str((self.peak_params.text().split(",")[3]))
        bkg_treatment = self.bkg_treatment.text()
        
        if self.peaksearch_mode.currentText() == "lauetools":
            try:
                peak_XY = RMCCD.PeakSearch(
                                            self.file,
                                            stackimageindex = -1,
                                            CCDLabel=self.ccd_label,
                                            NumberMaxofFits=5000,
                                            PixelNearRadius=10,
                                            removeedge=2,
                                            IntensityThreshold=intens,
                                            local_maxima_search_method=0,
                                            boxsize=bs,
                                            position_definition=1,
                                            verbose=0,
                                            fit_peaks_gaussian=1,
                                            xtol=0.001,                
                                            FitPixelDev=pixdev,
                                            return_histo=0,
                                            MinIntensity=0,
                                            PeakSizeRange=(0.65,200),
                                            write_execution_time=1,
                                            Data_for_localMaxima = "auto_background",
                                            formulaexpression=bkg_treatment,
                                            Remove_BlackListedPeaks_fromfile=None,
                                            reject_negative_baseline=True,
                                            Fit_with_Data_for_localMaxima=False,
                                            maxPixelDistanceRejection=15.0,
                                            mode = mode_peaksearch
                                            )
                peak_XY = peak_XY[0]
                self.pix_x, self.pix_y = peak_XY[:,0], peak_XY[:,1]
                self.peakXY = peak_XY
            except:
                print("Error in Peak detection for "+ self.file)
                self.pix_x, self.pix_y = [], []
                self.peakXY = []
                
        elif self.peaksearch_mode.currentText() == "LOG":
            print("fitting peaks with LOG function of skimage")
            from skimage.feature import blob_log
            from skimage import morphology
            try:
                import cv2
            except:
                print("OpenCv2 is not installed, cannot use the LOG peaksearch")
                print("please install opencv2 to use this feature")
                self.pix_x, self.pix_y = [], []
                self.peakXY = []
                
            try:
                data_8bit_raw = plt.imread(self.file)
                backgroundimage = ImProc.compute_autobackground_image(data_8bit_raw, boxsizefilter=10)
                # basic substraction
                data_8bit_raw_bg = ImProc.computefilteredimage(data_8bit_raw, backgroundimage, 
                                                               self.ccd_label, usemask=True, 
                                                               formulaexpression="A-B")
                data_8bit_raw = np.copy(data_8bit_raw_bg)
                ## simple thresholding
                bg_threshold = intens
                data_8bit_raw[data_8bit_raw<bg_threshold] = 0
                data_8bit_raw[data_8bit_raw>0] = 255
                ### Grayscale image (0 to 255)
                data_8bit_raw = data_8bit_raw.astype(np.uint8)
    
                data_8bit_raw_process = morphology.remove_small_objects(data_8bit_raw.astype(bool), 
                                                                     min_size=5, connectivity=2).astype(int)
                
                data_8bit_raw = data_8bit_raw_process.astype(np.uint8)
                data_8bit_raw[data_8bit_raw>0] = 255
    
                kernel = np.ones((3,3),np.uint8)
                thresh = cv2.morphologyEx(data_8bit_raw, cv2.MORPH_OPEN, kernel, iterations = 2)
    
                kernel = np.ones((2,2),np.uint8)
                thresh = cv2.erode(thresh, kernel, iterations = 1)
    
                if np.all(thresh==0):
                    print("threshold of all image is zero")     
    
                minsigma, maxsigma = 2, 10
                threshold_int = 0.01
                #Lapacian of gaussian
                blobs_log = blob_log(thresh, min_sigma=minsigma, max_sigma=maxsigma, 
                                     num_sigma=10, threshold=threshold_int)# Compute radii in the 3rd column.
                blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)
                
                # val_del = np.sqrt(2)+0.1
                # ind_del_log = np.where(blobs_log[:,2]<val_del)[0]            
                # blobs_log = np.delete(blobs_log, ind_del_log, axis=0)
                
                nbpeaks, _ = blobs_log.shape    
                peak_X_nonfit = blobs_log[:,1]
                peak_Y_nonfit = blobs_log[:,0]
                # peak_I_nonfit = blobs_log[:,2]
                # peaklist = np.vstack((peak_X_nonfit, peak_Y_nonfit, peak_I_nonfit)).T
                # peak_XY = peaklist
                
                type_of_function = "gaussian"
                position_start = "max"
                
                peaklist = np.vstack((peak_X_nonfit, peak_Y_nonfit)).T
                peak_dataarray = RMCCD.fitoneimage_manypeaks(self.file,
                                                            peaklist,
                                                            boxsize=bs,
                                                            stackimageindex=-1,
                                                            CCDLabel=self.ccd_label,
                                                            dirname=None,
                                                            position_start=position_start,
                                                            type_of_function=type_of_function,
                                                            xtol=0.00001,
                                                            FitPixelDev=pixdev,
                                                            Ipixmax=None,
                                                            MaxIntensity=65535, #e5,
                                                            MinIntensity=0,
                                                            PeakSizeRange=(0, 200),
                                                            verbose=0,
                                                            position_definition=1,
                                                            NumberMaxofFits=5000,
                                                            ComputeIpixmax=True,
                                                            use_data_corrected=None,
                                                            reject_negative_baseline=True)
                peak_XY = peak_dataarray[0]
                self.pix_x, self.pix_y = peak_XY[:,0], peak_XY[:,1]
                self.peakXY = peak_XY
            except:
                print("Error in Peak detection for "+ self.file)
                self.pix_x, self.pix_y = [], []
                self.peakXY = []
        self.draw_something()
        self.predicthkl.setEnabled(True)
        self.plothough.setEnabled(True)
    
    def prediction_hkl(self):
        try:
            intens = int(float(self.peak_params.text().split(",")[0]))
            bs = int(float(self.peak_params.text().split(",")[1]))
            pixdev = int(float(self.peak_params.text().split(",")[2]))
            mode_peaksearch = str((self.peak_params.text().split(",")[3]))
            smt = float(self.predict_params.text().split(",")[0])
            mrt = float(self.predict_params.text().split(",")[1])
            nb_spots_consider_global123 = int(float(self.predict_params.text().split(",")[5]))
            residues_threshold123 = float(self.predict_params.text().split(",")[4]) 
            option_global123 = self.predict_params.text().split(",")[2]
            nb_spots_global_threshold123 = int(float(self.predict_params.text().split(",")[3]))
            self.residues_threshold = residues_threshold123
            self.nb_spots_global_threshold = nb_spots_global_threshold123
            self.option_global = option_global123
            self.nb_spots_consider = nb_spots_consider_global123
            self.intensity_threshold = intens
            self.boxsize = bs
            self.FitPixelDev_global123 = pixdev
            self.softmax_threshold_global123 = smt
            self.cap_matchrate123 = mrt
            self.mode_peaksearch = mode_peaksearch
            self.function_predict(self.file, intens, bs, pixdev, smt, mrt, self.image_no, nb_spots_consider_global123,
                                  residues_threshold123, option_global123, nb_spots_global_threshold123, self.peakXY,
                                  self.strain_free_parameters,self.additional_expression, self.mode_peaksearch)
            self.propagate_button.setEnabled(True)
        except:
            print("Error during prediction; reinitialize the optimize window again, most probably this will fix it")
        
    def draw_something(self):
        # Drop off the first y element, append a new one.
        self.canvas.axes.cla()
        self.canvas.axes.set_title("Laue pattern of pixel x=%d, y=%d (peaks: %d) (file: %s)"%(self.iy,self.ix,len(self.pix_x),self.file), loc='center', fontsize=8)
        self.canvas.axes.set_ylabel(r'Y pixel',fontsize=8)
        self.canvas.axes.set_xlabel(r'X pixel', fontsize=8)
        if self.image_mode.currentText() == "raw":
            self.canvas.axes.imshow(self.data,interpolation="nearest",vmin=self.IminDisplayed, vmax=self.ImaxDisplayed)
        else:
            self.canvas.axes.imshow(self.corrected_data,interpolation="nearest",vmin=self.IminDisplayed, vmax=self.ImaxDisplayed)
            
        if len(self.pix_x)!=0:
            self.canvas.axes.scatter(self.pix_x, self.pix_y, s=120, facecolor='none', edgecolor='r', label="Peaks")
        self.canvas.draw()
        
    def sliderMin(self, val):
        try:
            #slider control function
            if val > self.ImaxDisplayed:
                print("Min value cannot be greater than Max")
                self.draw_something()
                return
            self.IminDisplayed= val
            self.draw_something()
        except:
            print("Error: value", val)
            
    def sliderMax(self, val):
        try:
            #slider control function
            if val < self.IminDisplayed:
                print("Max value cannot be less than Min")
                self.draw_something()
                return
            self.ImaxDisplayed= val
            self.draw_something()
        except:
            print("Error: value", val)

class MyPopup_image_v2(QWidget):
    def __init__(self, matrix, title, flag=0):
        QWidget.__init__(self)
        app_icon = QtGui.QIcon()
        app_icon.addFile(Logo, QtCore.QSize(16,16))
        self.setWindowIcon(app_icon)
        self.layout = QVBoxLayout() # QGridLayout()
        self.canvas = MplCanvas2(self, width=10, height=10, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        # set the layout
        self.layout.addWidget(self.toolbar, 0)
        self.layout.addWidget(self.canvas, 100)
        self.setLayout(self.layout)
        
        # Drop off the first y element, append a new one.
        self.canvas.axes.cla()
        self.canvas.axes.set_title(title, loc='center', fontsize=10)
        
        if flag == 0:
            matrix = matrix.reshape((matrix.shape[0]*matrix.shape[1],3,3))
            
        e11c = matrix[:,0,0]
        e22c = matrix[:,1,1]
        e33c = matrix[:,2,2]
        e12c = matrix[:,0,1]
        e13c = matrix[:,0,2]
        e23c = matrix[:,1,2]
        
        e11c = e11c.flatten()
        e22c = e22c.flatten()
        e33c = e33c.flatten()
        e12c = e12c.flatten()
        e13c = e13c.flatten()
        e23c = e23c.flatten()

        e11c = e11c[~np.isnan(e11c)]
        e22c = e22c[~np.isnan(e22c)]
        e33c = e33c[~np.isnan(e33c)]
        e12c = e12c[~np.isnan(e12c)]
        e13c = e13c[~np.isnan(e13c)]
        e23c = e23c[~np.isnan(e23c)]
        
        bins = 30
        # try:
        self.canvas.axes.set_title(r"$\epsilon_{11}$ (%)", loc='center', fontsize=10)
        logdata = e11c
        self.canvas.axes.hist(logdata, bins=bins, density=True, alpha=0.8)
        self.canvas.axes.set_ylabel('Frequency', fontsize=8)
        self.canvas.axes.tick_params(axis='both', which='major', labelsize=10)
        self.canvas.axes.tick_params(axis='both', which='minor', labelsize=10)
        self.canvas.axes.grid(True)
        
        self.canvas.axes1.set_title(r"$\epsilon_{22}$ (%)", loc='center', fontsize=10)
        logdata = e22c
        self.canvas.axes1.hist(logdata, bins=bins, density=True, alpha=0.8)
        self.canvas.axes1.set_ylabel('Frequency', fontsize=8)
        self.canvas.axes1.tick_params(axis='both', which='major', labelsize=10)
        self.canvas.axes1.tick_params(axis='both', which='minor', labelsize=10)
        self.canvas.axes1.grid(True)
        
        self.canvas.axes2.set_title(r"$\epsilon_{33}$ (%)", loc='center', fontsize=10)
        logdata = e33c
        self.canvas.axes2.hist(logdata, bins=bins, density=True, alpha=0.8)
        self.canvas.axes2.set_ylabel('Frequency', fontsize=8)
        self.canvas.axes2.tick_params(axis='both', which='major', labelsize=10)
        self.canvas.axes2.tick_params(axis='both', which='minor', labelsize=10)
        self.canvas.axes2.grid(True)
        
        self.canvas.axes3.set_title(r"$\epsilon_{12}$ (%)", loc='center', fontsize=10)
        logdata = e12c
        self.canvas.axes3.hist(logdata, bins=bins, density=True, alpha=0.8)
        self.canvas.axes3.set_ylabel('Frequency', fontsize=8)
        self.canvas.axes3.tick_params(axis='both', which='major', labelsize=10)
        self.canvas.axes3.tick_params(axis='both', which='minor', labelsize=10)
        self.canvas.axes3.grid(True)
        
        self.canvas.axes4.set_title(r"$\epsilon_{13}$ (%)", loc='center', fontsize=10)
        logdata = e13c
        self.canvas.axes4.hist(logdata, bins=bins, density=True, alpha=0.8)
        self.canvas.axes4.set_ylabel('Frequency', fontsize=8)
        self.canvas.axes4.tick_params(axis='both', which='major', labelsize=10)
        self.canvas.axes4.tick_params(axis='both', which='minor', labelsize=10)
        self.canvas.axes4.grid(True)
        
        self.canvas.axes5.set_title(r"$\epsilon_{23}$ (%)", loc='center', fontsize=10)
        logdata = e23c
        self.canvas.axes5.hist(logdata, bins=bins, density=True, alpha=0.8)
        self.canvas.axes5.set_ylabel('Frequency', fontsize=8)
        self.canvas.axes5.tick_params(axis='both', which='major', labelsize=10)
        self.canvas.axes5.tick_params(axis='both', which='minor', labelsize=10)
        self.canvas.axes5.grid(True)
        # except:
        #     pass
        # Trigger the canvas to update and redraw.
        self.canvas.draw()
                
class MyPopup_image(QWidget):
    def __init__(self, th_exp, chi_exp, intensity, tth_sim, chi_sim, sim_energy, sim_hkl, \
                 ix, iy, file, exp_linkspots, residues, theo_index, rotation_matrix):
        QWidget.__init__(self)
        app_icon = QtGui.QIcon()
        app_icon.addFile(Logo, QtCore.QSize(16,16))
        self.setWindowIcon(app_icon)
        self.layout = QVBoxLayout() # QGridLayout()
        self.canvas = MplCanvas1(self, width=10, height=10, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        self.Data_X = th_exp*2.0
        self.Data_Y = chi_exp
        self.Data_index_expspot = list(range(len(th_exp)))
        self.Data_I = intensity
        
        self.data_theo_X = tth_sim
        self.data_theo_Y = chi_sim
        self.data_theo_hkl = sim_hkl
        self.data_theo_energy = sim_energy
        self.Data_index_simspot = list(range(len(tth_sim)))
        self.iy,self.ix,self.file = iy, ix, file
        
        # set the layout
        self.layout.addWidget(self.toolbar, 0)
        self.layout.addWidget(self.canvas, 100)
        self.setLayout(self.layout)
        # Drop off the first y element, append a new one.
        self.intensity = intensity / np.amax(intensity) * 100.0
        self.canvas.axes.cla()
        self.canvas.axes.set_title("Laue pattern of pixel x=%d, y=%d (file: %s)"%(iy,ix,file), loc='center', fontsize=8)
        self.canvas.axes.set_ylabel(r'$\chi$ (in deg)',fontsize=8)
        self.canvas.axes.set_xlabel(r'2$\theta$ (in deg)', fontsize=8)
        self.scatter1 = self.canvas.axes.scatter(th_exp*2.0, chi_exp, c='k', s=self.intensity, cmap="gray", label="Exp spots")
        if len(tth_sim) != 0:
            self.scatter = self.canvas.axes.scatter(tth_sim, chi_sim, s=120, facecolor='none', edgecolor='r', label="Best match spots")
        # Trigger the canvas to update and redraw.
        self.canvas.axes.grid(True)
        self.canvas.axes.legend(fontsize=8)
        self.canvas.draw()
        
        self.annot = self.canvas.axes.annotate("", xy=(0,0),
                                                bbox=dict(boxstyle="round", fc="w"),
                                                arrowprops=dict(arrowstyle="->"))
        self.canvas.mpl_connect('motion_notify_event', self.onmovemouse)
        
        self._createDisplay() ## display screen
        ##create a text string to display
        if len(tth_sim) != 0:
            temp_ = rotation_matrix.flatten()
            string1 = "[["+str(temp_[0])+","+str(temp_[1])+","+str(temp_[2])+"],"+  \
                        "["+str(temp_[3])+","+str(temp_[4])+","+str(temp_[5])+"],"+  \
                            "["+str(temp_[6])+","+str(temp_[7])+","+str(temp_[8])+"]]"+ " \n"  
            texttstr = "Rotation matrix is: "+string1
            self.setDisplayText(texttstr)  
            texttstr = "# Total experimental spots : "+str(len(th_exp))
            self.setDisplayText(texttstr)
            texttstr = "# Average residues : "+str(np.average(residues))
            self.setDisplayText(texttstr)
            texttstr = "# Total linked spots : "+str(len(exp_linkspots))
            self.setDisplayText(texttstr)
            texttstr = "# Matching rate : "+str(len(exp_linkspots)/len(tth_sim))
            self.setDisplayText(texttstr)
            texttstr = "# Simulated_spots\tHKL\tExperimental_spots\tResidues\t "
            self.setDisplayText(texttstr)
            for i in range(len(theo_index)):
                texttstr = str(theo_index[i])+"\t\t"+str(sim_hkl[theo_index[i],:])+"\t"+str(exp_linkspots[i])+"\t\t"+str(residues[i])
                self.setDisplayText(texttstr)
    
    def update_annot(self, ind, vis, cont, ind1, cont1):
        self.canvas.axes.cla()
        self.canvas.axes.set_title("Laue pattern of pixel x=%d, y=%d (file: %s)"%(self.iy,self.ix,self.file), loc='center', fontsize=8)
        self.canvas.axes.set_ylabel(r'$\chi$ (in deg)',fontsize=8)
        self.canvas.axes.set_xlabel(r'2$\theta$ (in deg)', fontsize=8)
        self.scatter1 = self.canvas.axes.scatter(self.Data_X, self.Data_Y, c='k', s=self.intensity, cmap="gray", label="Exp spots")
        if len(self.data_theo_X) != 0:
            self.scatter = self.canvas.axes.scatter(self.data_theo_X, self.data_theo_Y, s=120, facecolor='none', edgecolor='r', label="Best match spots")
        if cont:
            pos = self.scatter.get_offsets()[ind["ind"][0]]
            n = ind["ind"][0]
            if ind1 != None and cont1:
                n1 = ind1["ind"][0]
            else:
                n1=""
            text = "ExpIndex={} \nTheoIndex={} \n2Theta={} \nChi={} \nHKL={} \nEnergy={}".format(" "+str(n1),
                                                                                                      " "+str(n), 
                                                                               " "+str(np.round(self.data_theo_X[n],2)),
                                                                               " "+str(np.round(self.data_theo_Y[n],2)),
                                                                                " "+str(self.data_theo_hkl[n]),
                                                                                " "+str(np.round(self.data_theo_energy[n],2)))
            self.annot = self.canvas.axes.annotate(text, xy=pos,xytext=(20,20),textcoords="offset points",
                                                    bbox=dict(boxstyle="round", fc="gray"),
                                                    arrowprops=dict(arrowstyle="->"))
            self.annot.get_bbox_patch().set_alpha(0.5)
            self.annot.set_visible(True) 
        elif not cont and vis:
            self.annot.set_visible(False) 
        # Trigger the canvas to update and redraw.
        self.canvas.axes.grid(True)
        self.canvas.axes.legend(fontsize=8)
        self.canvas.draw()
        
    def onmovemouse(self,event):
        try:
            vis = self.annot.get_visible()
            if event.inaxes == None:
                return
            cont, ind = self.scatter.contains(event)
            try:
                cont1, ind1 = self.scatter1.contains(event)
            except:
                cont1, ind1 = False, None
            self.update_annot(ind, vis, cont, ind1, cont1)
        except:
            return

    def _createDisplay(self):
        """Create the display."""
        self.display = QTextEdit()
        self.display.setReadOnly(True)
        self.layout.addWidget(self.display)

    def setDisplayText(self, text):
        self.display.append('%s'%text)
        self.display.moveCursor(QtGui.QTextCursor.Start)
        self.display.setFocus()

class Window_allmap(QWidget):
    def __init__(self, limx, limy, filenm, ccd_label, predict_single_file_nodialog, detectorparameters):
        super(Window_allmap, self).__init__()
        QWidget.__init__(self)
        app_icon = QtGui.QIcon()
        app_icon.addFile(Logo, QtCore.QSize(16,16))
        self.setWindowIcon(app_icon)
        self.lim_x, self.lim_y = limx, limy
        self.filenm = filenm
        self.ccd_label = ccd_label
        self.detectorparameters = detectorparameters
        self.diff_data = np.zeros((self.lim_x, self.lim_y))
        self.predict_single_file_nodialog = predict_single_file_nodialog
        self.setWindowTitle("Laue plot module (right click interactive)")
        # self.myQMenuBar = QMenuBar(self) 
        self.layout = QVBoxLayout() # QGridLayout()
        self.canvas = MplCanvas1(self, width=10, height=10, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)        
        self.canvas.mpl_connect('button_press_event', self.onclickImage)
        
        self.refresh_plot = QPushButton("Refresh plot")
        self.refresh_plot.clicked.connect(self.refresh_plots)
        self.img_similarity = QPushButton("Calculate image similarity")
        self.img_similarity.clicked.connect(self.calculate_image_similarity)
     
        # set the layout
        self.popups = []
        self.layout.addWidget(self.toolbar, 0)
        self.layout.addWidget(self.canvas, 100)
        
        self.image_grid = QLineEdit()
        self.image_grid.setText("10,10")
        
        self.path_folder = QLineEdit()
        self.path_folder.setText("")
        
        formLayout = QFormLayout()
        formLayout.addRow(self.refresh_plot, self.img_similarity)
        self.layout.addLayout(formLayout)  
        
        self.setLayout(self.layout)    
        self.draw_something()
        self.setFixedSize(16777215,16777215)
        
    def refresh_plots(self):
        self.draw_something()
        
    def draw_something(self):
        self.canvas.axes.cla()
        self.canvas.axes.set_title("Spatial scan Laue map", loc='center', fontsize=10)
        if np.all(self.diff_data == 0):
            arr = np.random.randint(low = 0, high = 255, size = (self.lim_x, self.lim_y))
        else:
            arr = self.diff_data
        # self.canvas.axes.imshow(arr.astype('uint8'), origin='lower')
        self.canvas.axes.imshow(arr, origin='lower')
        self.canvas.draw()
    
    def calculate_image_similarity(self):
        print("Preparing for image similarity calculation")
        print("By computing the sum of squared difference between each bg. corrected Laue images")
        try:
            values = []
            count = 0 
            total = self.diff_data.shape[0]*self.diff_data.shape[1]        
            for ix in range(self.diff_data.shape[0]):
                for iy in range(self.diff_data.shape[1]):
                    if iy == 0 and ix == 0:
                        continue
                    elif iy != 0 and ix == 0:
                        image_no = iy
                    elif iy == 0 and ix != 0:
                        image_no = ix * self.lim_y
                    elif iy != 0 and ix != 0:
                        image_no = ix * self.lim_y + iy
                        
                    if image_no >= total:
                        continue
                    
                    path = os.path.normpath(self.filenm[image_no].decode())                    
                    
                    if (image_no % self.lim_y == 0) and image_no != 0:
                        old_image = image_no - self.lim_y
                    elif (image_no % self.lim_y != 0):
                        old_image = image_no - 1 
                                    
                    path1 = os.path.normpath(self.filenm[old_image].decode())
                    values.append([path, path1, ix, iy, self.ccd_label, True, count, total])
                    count += 1
            with multip.Pool(cpu_count()) as pool:
                results = [pool.apply_async(mse_images, p) for p in values]
                for r in results:
                    r1 = r.get()
                    self.diff_data[r1[1],r1[2]] = r1[0]
            print("Image similarity computation finished")
        except:
            print("Error in calculation of image similarity module")
        self.draw_something()

    def onclickImage(self, event123):
        if event123.button == 3:
            ix, iy = event123.xdata, event123.ydata
            # try:
            ## read the saved COR file and extract exp spots info.## avoid zero index problem
            ix = int(round(ix))
            iy = int(round(iy))
            try:
                # self.lim_x * self.lim_y
                if iy == 0 and ix == 0:
                    image_no = 0
                elif iy == 0 and ix != 0:
                    image_no = ix
                elif iy != 0 and ix == 0:
                    image_no = iy * self.lim_y
                elif iy != 0 and ix != 0:
                    image_no = iy * self.lim_y + ix
                    
                path = os.path.normpath(self.filenm[image_no].decode())                    
                Data, framedim, fliprot = IOimage.readCCDimage(path,
                                                                stackimageindex=-1,
                                                                CCDLabel=self.ccd_label,
                                                                dirname=None,
                                                                verbose=0)   
            except:
                print(path)
                print('chosen pixel coords are x = %d, y = %d'%(ix, iy))
                print("No IMAGE file could be found for the selected pixel")
                return
            w = MyPopup_image_v1(ix, iy, path, Data, self.ccd_label, 
                                 self.predict_single_file_nodialog, image_no,
                                 self.detectorparameters)
            w.show()       
            self.popups.append(w)
            print('chosen pixel coords are x = %d, y = %d'%(ix, iy))
            # except:
            #     return
        else:
            print("Right click for plotting the pixel values")

class gif_window(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        app_icon = QtGui.QIcon()
        app_icon.addFile(Logo, QtCore.QSize(16,16))
        self.setWindowIcon(app_icon)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        # create label
        self.label = QtWidgets.QLabel()
        self.label.setObjectName("label")
        self.layout.addWidget(self.label)
        # set qmovie as label
        self.movie = QMovie(gif_path)
        self.label.setMovie(self.movie)
        self.movie.start()

class sample_commands(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        app_icon = QtGui.QIcon()
        app_icon.addFile(Logo, QtCore.QSize(16,16))
        self.setWindowIcon(app_icon)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self._createDisplay() ## display screen
        self.setDisplayText("Here are the set of commands that you can use with terminal to execute stuff!")
        self.setDisplayText("----------")
        self.setDisplayText("lauetoolsnn or lauenn : launches the current GUI")
        self.setDisplayText("----------")
        self.setDisplayText("lauenn_addmat : Add user material to laueNN library")
        self.setDisplayText("usage : lauenn_addmat -n Cu -l 3.6 3.6 3.6 90 90 90 -e 227")
        self.setDisplayText("----------")
        self.setDisplayText("lauenn_mat : Query a user material in laueNN library")
        self.setDisplayText("usage : lauenn_mat -n Cu")
        self.setDisplayText("----------")
        self.setDisplayText("lauenn_maxhkl : Calculates the maxHKL available on the Detector with polycrystal Laue Simulation; useful for max_HKL_index parameter estimation in the config file")
        self.setDisplayText("usage : lauenn_maxhkl -n Cu -dd 79.5 -nb 5 -c '3 5 7 10' -s 100")
        self.setDisplayText("----------")
        self.setDisplayText("lauenn_pymatgen : Query the PYMATGEN rest API for lattice parameters and spacegroup")
        self.setDisplayText("usage : lauenn_pymatgen -e Cu Si")
        self.setDisplayText("----------")
        self.setDisplayText("lauenn_example : Copies example script of LaueNN from installation directory to current user directory (terminal cwd)")
        self.setDisplayText("usage : lauenn_example")

    def _createDisplay(self):
        """Create the display."""
        self.display = QTextEdit()
        self.display.setReadOnly(True)
        self.layout.addWidget(self.display)

    def setDisplayText(self, text):
        self.display.append('%s'%text)
        self.display.moveCursor(QtGui.QTextCursor.End)
        self.display.setFocus()
        
class sample_config(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        app_icon = QtGui.QIcon()
        app_icon.addFile(Logo, QtCore.QSize(16,16))
        self.setWindowIcon(app_icon)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self._createDisplay() ## display screen
        self.setDisplayText(texttstr1)

    def _createDisplay(self):
        """Create the display."""
        self.display = QTextEdit()
        self.display.setReadOnly(True)
        self.layout.addWidget(self.display)

    def setDisplayText(self, text):
        self.display.append('%s'%text)
        self.display.moveCursor(QtGui.QTextCursor.End)
        self.display.setFocus()
        
class MyPopup(QWidget):
    def __init__(self, match_rate12, rotation_matrix12, mat_global12, fR_pix12, filename, 
                 straincrystal, strainsample, end_time, mode_analysis, th_exp, chi_exp, intensity, tth_sim, chi_sim,
                 sim_energy, sim_hkl, exp_linkspots, residues, theo_index, hkl_prediction=False):
        QWidget.__init__(self)
        
        app_icon = QtGui.QIcon()
        app_icon.addFile(Logo, QtCore.QSize(16,16))
        self.setWindowIcon(app_icon)
        
        self.layout = QVBoxLayout() # QGridLayout()
        self.canvas = MplCanvas1(self, width=10, height=10, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        self.Data_X = th_exp*2.0
        self.Data_Y = chi_exp
        self.Data_index_expspot = list(range(len(th_exp)))
        self.Data_I = intensity
        self.hkl_prediction = hkl_prediction
        self.data_theo_X = tth_sim
        self.data_theo_Y = chi_sim
        self.data_theo_hkl = sim_hkl
        self.data_theo_energy = sim_energy
        self.Data_index_simspot = list(range(len(tth_sim)))
        self.file = filename
        self.match_rate12987 = match_rate12
        # set the layout
        self.layout.addWidget(self.toolbar, 0)
        self.layout.addWidget(self.canvas, 100)
        self.setLayout(self.layout)
        # Drop off the first y element, append a new one.
        self.intensity = intensity / np.amax(intensity) * 100.0
        self.canvas.axes.cla()
        self.canvas.axes.set_title("Laue pattern (file: %s)"%(filename), loc='center', fontsize=8)
        self.canvas.axes.set_ylabel(r'$\chi$ (in deg)',fontsize=8)
        self.canvas.axes.set_xlabel(r'2$\theta$ (in deg)', fontsize=8)
        self.scatter1 = self.canvas.axes.scatter(th_exp*2.0, chi_exp, c='k', s=self.intensity, cmap="gray", label="Exp spots")
        cycle_color = ["b","c","g","m","r","y","brown","gray","purple","orange","bisque","lime","gold",
                       "b","c","g","m","r","y","brown","gray","purple","orange","bisque","lime","gold",
                       "b","c","g","m","r","y","brown","gray","purple","orange","bisque","lime","gold"]
        self.cycle_color = cycle_color
        ## convert into a single variable
        tth_s, chi_s, color_s, label_s, theohkl_s, theoenergy_s = [], [], [], [], [], []
        for ijk in range(len(match_rate12)):
            if len(tth_sim[ijk]) != 0:
                for okl in range(len(tth_sim[ijk])):
                    tth_s.append(tth_sim[ijk][okl])
                    chi_s.append(chi_sim[ijk][okl])
                    color_s.append(cycle_color[ijk])
                    theohkl_s.append(sim_hkl[ijk][okl])
                    theoenergy_s.append(sim_energy[ijk])
                    label_s.append("Matrix "+str(ijk+1))
                    
        self.scatter = self.canvas.axes.scatter(tth_s, chi_s, s=120, facecolor='none', 
                                  edgecolor=color_s)#, label="Matrix "+str(ijk+1))
        # Trigger the canvas to update and redraw.
        self.canvas.axes.grid(True)
        self.canvas.axes.legend(fontsize=8)
        self.canvas.draw()
        
        self.annot = self.canvas.axes.annotate("", xy=(0,0),
                                                bbox=dict(boxstyle="round", fc="w"),
                                                arrowprops=dict(arrowstyle="->"))
        self.canvas.mpl_connect('motion_notify_event', self.onmovemouse)
                
        self._createDisplay() ## display screen
        ##create a text string to display
        texttstr = "Predicted for File: "+filename+ " \n" 
        self.setDisplayText(texttstr)
        texttstr = "# Total experimental spots : "+str(len(th_exp))
        self.setDisplayText(texttstr)
        self.setDisplayText("################## "+mode_analysis+" MODE ############### \n")
        for ijk in range(len(match_rate12)):
            if len(tth_sim[ijk]) != 0:
                self.setDisplayText("--------------- Matrix "+str(ijk+1)+" \n")
                texttstr = "# Average residues : "+str(np.average(residues[ijk]))
                self.setDisplayText(texttstr)
                texttstr = "# Total linked spots : "+str(len(exp_linkspots[ijk]))
                self.setDisplayText(texttstr)
                texttstr = "# Matching rate : "+str(len(exp_linkspots[ijk])/len(tth_sim[ijk]))
                self.setDisplayText(texttstr)
                texttstr = "Matching rate for the proposed matrix is: "+str(match_rate12[ijk][0])+ " \n" 
                self.setDisplayText(texttstr)
                texttstr = "Identified material index is: "+str(mat_global12[ijk][0])+ " \n" 
                self.setDisplayText(texttstr)
                temp_ = rotation_matrix12[ijk][0].flatten()
                string1 = "[["+str(temp_[0])+","+str(temp_[1])+","+str(temp_[2])+"],"+  \
                            "["+str(temp_[3])+","+str(temp_[4])+","+str(temp_[5])+"],"+  \
                                "["+str(temp_[6])+","+str(temp_[7])+","+str(temp_[8])+"]]"+ " \n"  
                texttstr = "Rotation matrix is: "+string1
                self.setDisplayText(texttstr)
                temp_ = straincrystal[ijk][0].flatten()
                string1 = "[["+str(temp_[0])+","+str(temp_[1])+","+str(temp_[2])+"],"+  \
                            "["+str(temp_[3])+","+str(temp_[4])+","+str(temp_[5])+"],"+  \
                                "["+str(temp_[6])+","+str(temp_[7])+","+str(temp_[8])+"]]"+ " \n"  
                texttstr = "Strain crystal reference frame is: "+string1
                self.setDisplayText(texttstr)
                temp_ = strainsample[ijk][0].flatten()
                string1 = "[["+str(temp_[0])+","+str(temp_[1])+","+str(temp_[2])+"],"+  \
                            "["+str(temp_[3])+","+str(temp_[4])+","+str(temp_[5])+"],"+  \
                                "["+str(temp_[6])+","+str(temp_[7])+","+str(temp_[8])+"]]"+ " \n"  
                texttstr = "Strain sample reference frame is: "+string1
                self.setDisplayText(texttstr)
                texttstr = "Final pixel residues is: "+str(fR_pix12[ijk][0]) + " \n"
                self.setDisplayText(texttstr)
        texttstr = "Total time in seconds (Loading image, peak detection, HKL prediction, Orientation matrix computation, strain computation): "+str(end_time) + " \n"
        self.setDisplayText(texttstr)        
    
    def update_annot(self, ind, vis, cont, ind1, cont1):
        self.canvas.axes.cla()
        self.canvas.axes.set_title("Laue pattern (file: %s)"%(self.file), loc='center', fontsize=8)
        self.canvas.axes.set_ylabel(r'$\chi$ (in deg)',fontsize=8)
        self.canvas.axes.set_xlabel(r'2$\theta$ (in deg)', fontsize=8)
        
        self.scatter1 = self.canvas.axes.scatter(self.Data_X, self.Data_Y, c='k', s=self.intensity, label="Exp spots")
        
        ## convert into a single variable
        tth_s, chi_s, color_s, label_s, theohkl_s, theoenergy_s = [], [], [], [], [], []
        for ijk in range(len(self.match_rate12987)):
            if len(self.data_theo_X[ijk]) != 0:
                for okl in range(len(self.data_theo_X[ijk])):
                    tth_s.append(self.data_theo_X[ijk][okl])
                    chi_s.append(self.data_theo_Y[ijk][okl])
                    color_s.append(self.cycle_color[ijk])
                    theohkl_s.append(self.data_theo_hkl[ijk][okl])
                    theoenergy_s.append(self.data_theo_energy[ijk][okl])
                    label_s.append(ijk+1)
                    
        self.scatter = self.canvas.axes.scatter(tth_s, chi_s, s=120, facecolor='none', 
                                  edgecolor=color_s)
        
        if ind != None and cont:
            pos = self.scatter.get_offsets()[ind["ind"][0]]
            n = ind["ind"][0]
        else:
            n=""
            
        if ind1 != None and cont1:
            pos = self.scatter1.get_offsets()[ind1["ind"][0]]
            n1 = ind1["ind"][0]
        else:
            n1=""
            
        try:
            pp123 = self.hkl_prediction[n1,:]
        except:
            pp123 = "No prediction"
        
        if n=="":
            text = "\nExpIndex={} \nPrediction={}".format(" "+str(n1)," "+str(pp123))
        elif n1 == "":
            text = "\nMatrix={} \nTheoIndex={} \n2Theta={} \nChi={} \nHKL={} \nEnergy={}".format(
                                                                                " "+str(label_s[n]),
                                                                                " "+str(n), 
                                                                                " "+str(np.round(tth_s[n],2)),
                                                                                " "+str(np.round(chi_s[n],2)),
                                                                                " "+str(theohkl_s[n]),
                                                                                " "+str(np.round(theoenergy_s[n],2)))
        else:
            text = "\nMatrix={} \nExpIndex={} \nPrediction={} \nTheoIndex={} \n2Theta={} \nChi={} \nHKL={} \nEnergy={}".format(
                                                                                " "+str(label_s[n]),
                                                                                " "+str(n1),
                                                                                " "+str(pp123),
                                                                                " "+str(n), 
                                                                                " "+str(np.round(tth_s[n],2)),
                                                                                " "+str(np.round(chi_s[n],2)),
                                                                                " "+str(theohkl_s[n]),
                                                                                " "+str(np.round(theoenergy_s[n],2)))
            
        self.annot = self.canvas.axes.annotate(text, xy=pos, xytext=(20,20), textcoords="offset points",
                                                bbox=dict(boxstyle="round", fc="gray"),
                                                arrowprops=dict(arrowstyle="->"))
        
        self.annot.get_bbox_patch().set_alpha(0.5)
        self.annot.set_visible(True)

        # Trigger the canvas to update and redraw.
        self.canvas.axes.grid(True)
        self.canvas.axes.legend(fontsize=8)
        self.canvas.draw()
        
    def onmovemouse(self,event):
        # try:
        vis = self.annot.get_visible()
        if event.inaxes == None:
            return
        try:
            cont, ind = self.scatter.contains(event)
        except:
            cont, ind = False, None
        try:
            cont1, ind1 = self.scatter1.contains(event)
        except:
            cont1, ind1 = False, None
        if cont or cont1:
            self.update_annot(ind, vis, cont, ind1, cont1)
        # except:
        #     return
    
    def _createDisplay(self):
        """Create the display."""
        self.display = QTextEdit()
        self.display.setReadOnly(True)
        self.layout.addWidget(self.display)

    def setDisplayText(self, text):
        self.display.append('%s'%text)
        self.display.moveCursor(QtGui.QTextCursor.Start)
        self.display.setFocus()

class AnotherWindowParams(QWidget):
    got_signal = QtCore.pyqtSignal(dict)
    def __init__(self, state=0, gui_state=0):
        super().__init__()
        
        app_icon = QtGui.QIcon()
        app_icon.addFile(Logo, QtCore.QSize(16,16))
        self.setWindowIcon(app_icon)
        
        self.setFixedSize(500, 500)
        
        self.settings = QSettings("config_data_"+str(gui_state),"ConfigGUI_"+str(gui_state))
        ## Material detail        
        self.dict_LT = QComboBox()
        sortednames = sorted(dictLT.dict_Materials.keys(), key=lambda x:x.lower())
        for s in sortednames:
            self.dict_LT.addItem(s)
        
        self.dict_LT1 = QComboBox()
        sortednames = sorted(dictLT.dict_Materials.keys(), key=lambda x:x.lower())
        for s in sortednames:
            self.dict_LT1.addItem(s)
        
        if main_directory != None:
            self.modelDirecSave = main_directory
        else:
            self.modelDirecSave = None
        self.model_direc_save = QPushButton('Browse')
        self.model_direc_save.clicked.connect(self.getfiles)
        
        self.symmetry = QComboBox()
        symmetry_names = ["cubic","hexagonal","orthorhombic","tetragonal","trigonal","monoclinic","triclinic"]
        for s in symmetry_names:
            self.symmetry.addItem(s)
        
        self.symmetry1 = QComboBox()
        symmetry_names = ["cubic","hexagonal","orthorhombic","tetragonal","trigonal","monoclinic","triclinic"]
        for s in symmetry_names:
            self.symmetry1.addItem(s)
        
        self.prefix = QLineEdit()
        self.prefix.setText("") ## Prefix for folder
        
        self.hkl_max = QLineEdit()
        self.hkl_max.setText("auto") ## auto or some indices of HKL
        
        self.elements = QLineEdit()
        self.elements.setText("200") ## all or some length
        
        self.freq_rmv = QLineEdit()
        self.freq_rmv.setText("1") ## auto or some indices of HKL
        
        self.hkl_max1 = QLineEdit()
        self.hkl_max1.setText("auto") ## auto or some indices of HKL
        
        self.elements1 = QLineEdit()
        self.elements1.setText("200") ## all or some length
        
        self.freq_rmv1 = QLineEdit()
        self.freq_rmv1.setText("1") ## auto or some indices of HKL
        
        self.maximum_angle_to_search = QLineEdit()
        self.maximum_angle_to_search.setText("90")
        
        self.step_for_binning = QLineEdit()
        self.step_for_binning.setText("0.1")
        
        self.mode_of_analysis = QComboBox()
        mode_ = ["1","0"]
        for s in mode_:
            self.mode_of_analysis.addItem(s)
            
        self.nb_grains_per_lp = QLineEdit()
        self.nb_grains_per_lp.setText("5")
        
        self.nb_grains_per_lp1 = QLineEdit()
        self.nb_grains_per_lp1.setText("5")
        
        self.grains_nb_simulate = QLineEdit()
        self.grains_nb_simulate.setText("500")
        
        self.detectordistance = QLineEdit()
        self.detectordistance.setText("79.553")
        
        self.xycenter = QLineEdit()
        self.xycenter.setText("979.32,932.31")
        
        self.bgdetector = QLineEdit()
        self.bgdetector.setText("0.37,0.447")
        
        self.detectordim = QLineEdit()
        self.detectordim.setText("2018,2016")
        
        self.pixelsize = QLineEdit()
        self.pixelsize.setText("0.0734")
        
        self.minmaxE = QLineEdit()
        self.minmaxE.setText("5,18")
        
        self.include_scm = QComboBox()
        modes = ["no", "yes"]
        for s in modes:
            self.include_scm.addItem(s)
            
        self.architecture = QComboBox()
        modes = ["FFNN","1D_CNN","1D_CNN_DNN","User_defined"]
        for s in modes:
            self.architecture.addItem(s)
            
        self.learningrate_rc = QLineEdit()
        self.learningrate_rc.setText("1e-3,1e-5,1e-6")

        self.mode_nn = QComboBox()
        modes = ["Generate Data & Train","Train","Predict"]
        for s in modes:
            self.mode_nn.addItem(s)
        
        self.batch_size = QLineEdit()
        self.batch_size.setText("20")
        
        self.epochs = QLineEdit()
        self.epochs.setText("5")
        
        self.grid_search_hyperparams = QComboBox()
        mode_ = ["False","True"]
        for s in mode_:
            self.grid_search_hyperparams.addItem(s)
            
        self.texture_model = QComboBox()
        mode_ = ["in-built_Uniform_Distribution","random","from file"]
        for s in mode_:
            self.texture_model.addItem(s)
            
        # button to continue training
        self.btn_config = QPushButton('Accept')
        self.btn_config.clicked.connect(self.send_details_mainGUI)
        close_button = QPushButton("Cancel")
        close_button.clicked.connect(self.close)

        ### set some default values
        if freq_rmv_global != None:
            self.freq_rmv.setText(str(freq_rmv_global))
        if elements_global != None:
            self.elements.setText(elements_global)
        if hkl_max_global != None:
            self.hkl_max.setText(hkl_max_global)
        if nb_grains_per_lp_global != None:
            self.nb_grains_per_lp.setText(str(nb_grains_per_lp_global))
        
        if freq_rmv1_global != None:
            self.freq_rmv1.setText(str(freq_rmv1_global))
        if elements1_global != None:
            self.elements1.setText(elements1_global)
        if hkl_max1_global != None:
            self.hkl_max1.setText(hkl_max1_global)
        if nb_grains_per_lp1_global != None:
            self.nb_grains_per_lp1.setText(str(nb_grains_per_lp1_global))
            
        if include_scm_global:
            self.include_scm.setCurrentText("yes")
        else:
            self.include_scm.setCurrentText("no")
            
        if batch_size_global != None:
            self.batch_size.setText(str(batch_size_global))
        if epochs_global != None:
            self.epochs.setText(str(epochs_global))  
            
        if maximum_angle_to_search_global != None:
            self.maximum_angle_to_search.setText(str(maximum_angle_to_search_global))
        if step_for_binning_global != None:
            self.step_for_binning.setText(str(step_for_binning_global))
        if grains_nb_simulate_global != None:
            self.grains_nb_simulate.setText(str(grains_nb_simulate_global))    
            
        if symmetry_global != None:
            self.symmetry.setCurrentText(symmetry_global)
        if symmetry1_global != None:
            self.symmetry1.setCurrentText(symmetry1_global)
        if material_global != None:
            self.dict_LT.setCurrentText(material_global)
        if material1_global != None:
            self.dict_LT1.setCurrentText(material1_global)
        if prefix_global != None:
            self.prefix.setText(prefix_global)
        if detectorparameters_global != None:
            self.detectordistance.setText(str(detectorparameters_global[0]))
            self.xycenter.setText(str(detectorparameters_global[1])+","+str(detectorparameters_global[2]))
            self.bgdetector.setText(str(detectorparameters_global[3])+","+str(detectorparameters_global[4]))
            self.detectordim.setText(str(dim1_global)+","+str(dim2_global))
            self.pixelsize.setText(str(pixelsize_global))
            self.minmaxE.setText(str(emin_global)+","+str(emax_global))

        self.layout = QVBoxLayout() # QGridLayout()
        scroll = QScrollArea(self)
        self.layout.addWidget(scroll)
        scroll.setWidgetResizable(True)
        scrollContent = QWidget(scroll)
        formLayout = QFormLayout(scrollContent)
        # formLayout = QFormLayout()
        # formLayout.setVerticalSpacing(5)
        formLayout.addRow('Training parameters', QLineEdit().setReadOnly(True))
        formLayout.addRow('Directory where \n model files are saved', self.model_direc_save)
        formLayout.addRow('Material details', QLineEdit().setReadOnly(True))
        formLayout.addRow('Prefix for save folder', self.prefix)
        formLayout.addRow('Choose Material and Symmetry \n (incase of 1 material, keep both same)', QLineEdit().setReadOnly(True))
        formLayout.addRow(self.dict_LT, self.dict_LT1)
        formLayout.addRow(self.symmetry, self.symmetry1)
        formLayout.addRow('Class removal frequency', QLineEdit().setReadOnly(True))
        formLayout.addRow(self.freq_rmv, self.freq_rmv1)
        formLayout.addRow('Class length', QLineEdit().setReadOnly(True))
        formLayout.addRow(self.elements, self.elements1)
        formLayout.addRow('HKL max probed', QLineEdit().setReadOnly(True))
        formLayout.addRow(self.hkl_max, self.hkl_max1)
        formLayout.addRow('Histogram parameters', QLineEdit().setReadOnly(True))
        formLayout.addRow('Angular distance to probe (in deg)', self.maximum_angle_to_search)
        formLayout.addRow('Angular bin widths (in deg)', self.step_for_binning)
        formLayout.addRow('Simulation parameters', QLineEdit().setReadOnly(True))
        # formLayout.addRow('Analysis mode', self.mode_of_analysis)
        formLayout.addRow('Max Nb. of grain in a LP', QLineEdit().setReadOnly(True))
        formLayout.addRow(self.nb_grains_per_lp, self.nb_grains_per_lp1)
        formLayout.addRow('Nb. of simulations', self.grains_nb_simulate)
        formLayout.addRow('Include single crystal \n misorientation', self.include_scm)
        formLayout.addRow('Detector parameters', QLineEdit().setReadOnly(True))
        formLayout.addRow('Detector distance', self.detectordistance)
        formLayout.addRow('Detector XY center', self.xycenter)
        formLayout.addRow('Detector Beta Gamma', self.bgdetector)
        formLayout.addRow('Detector Pixel size', self.pixelsize)
        formLayout.addRow('Detector dimensions (dim1,dim2)', self.detectordim)
        formLayout.addRow('Energy (Min, Max)', self.minmaxE)
        formLayout.addRow('Neural Network parameters', QLineEdit().setReadOnly(True))
        formLayout.addRow('Mode of analysis', self.mode_nn)
        formLayout.addRow('Model Architecture', self.architecture)
        formLayout.addRow('LR, Regularization coefficient', self.learningrate_rc)
        formLayout.addRow('Batch size', self.batch_size)
        formLayout.addRow('Epochs', self.epochs)
        formLayout.addRow('Grid search for model Params', self.grid_search_hyperparams)
        formLayout.addRow('Texture for data', self.texture_model)
        # formLayout.setVerticalSpacing(5)
        formLayout.addRow(close_button, self.btn_config)
        
        scrollContent.setLayout(formLayout)
        scroll.setWidget(scrollContent)
        self.setLayout(self.layout)
        
        self.setFixedSize(16777215,16777215)
        #self._gui_save()
        #if state > 0:
        #    self._gui_restore()
    
    def getfiles(self):
        self.modelDirecSave = QFileDialog.getExistingDirectory(self, 'Select Folder in which model files will be saved')
    
    def _gui_save(self):
      # Save geometry
        for name, obj in inspect.getmembers(self):
          # if type(obj) is QComboBox:  # this works similar to isinstance, but missed some field... not sure why?
            if isinstance(obj, QComboBox):
                index = obj.currentIndex()  # get current index from combobox
                text = obj.itemText(index)  # get the text for current index
                self.settings.setValue(name, text)  # save combobox selection to registry
            if isinstance(obj, QLineEdit):
                value = obj.text()
                self.settings.setValue(name, value)  # save ui values, so they can be restored next time
        self.settings.sync()

    def _gui_restore(self):
        # Restore geometry  
        for name, obj in inspect.getmembers(self):
            if isinstance(obj, QComboBox):
                index = obj.currentIndex()  # get current region from combobox
                value = (self.settings.value(name))
                if value == "":
                    continue
                index = obj.findText(value)  # get the corresponding index for specified string in combobox
          
                if index == -1:  # add to list if not found
                    obj.insertItems(0, [value])
                    index = obj.findText(value)
                    obj.setCurrentIndex(index)
                else:
                    obj.setCurrentIndex(index)  # preselect a combobox value by index
            if isinstance(obj, QLineEdit):
                value = (self.settings.value(name))#.decode('utf-8'))  # get stored value from registry
                obj.setText(value)  # restore lineEditFile
        self.settings.sync()
        
    def send_details_mainGUI(self):
        self._gui_save()
        detector_params = [float(self.detectordistance.text()),
                           float(self.xycenter.text().split(",")[0]), 
                           float(self.xycenter.text().split(",")[1]),
                           float(self.bgdetector.text().split(",")[0]), 
                           float(self.bgdetector.text().split(",")[1])]
        
        global prefix_global, weightfile_global, modelfile_global, model_weight_file
        if self.prefix.text() != prefix_global:
            prefix_global = self.prefix.text()
            ##exp directory
            if material_global == material1_global:
                fn = material_global + prefix_global
            else:
                fn = material_global + "_" + material1_global + prefix_global
                        
            modelfile_global = self.modelDirecSave + "//" + fn
            if material_global == material1_global:
                if model_weight_file == "none":
                    weightfile_global = modelfile_global + "//" + "model_" + material_global + ".h5"
                else:
                    weightfile_global = model_weight_file
            else:
                if model_weight_file == "none":
                    weightfile_global = modelfile_global + "//" + "model_" + material_global + "_" + material1_global + ".h5"
                else:
                    weightfile_global = model_weight_file
                    
        # create a dictionary and emit the signal
        emit_dictionary = { "material_": self.dict_LT.currentText(), ## same key as used in LaueTools
                            "material1_": self.dict_LT1.currentText(),
                            "prefix": self.prefix.text(),
                            "symmetry": self.symmetry.currentText(),
                            "symmetry1": self.symmetry1.currentText(),
                            "hkl_max_identify" : self.hkl_max.text(), # can be "auto" or an index i.e 12
                            "hkl_max_identify1" : self.hkl_max1.text(), # can be "auto" or an index i.e 12
                            "maximum_angle_to_search" : float(self.maximum_angle_to_search.text()),
                            "step_for_binning" : float(self.step_for_binning.text()),
                            "mode_of_analysis" : int(self.mode_of_analysis.currentText()),
                            "nb_grains_per_lp" : int(self.nb_grains_per_lp.text()), ## max grains to expect in a LP
                            "nb_grains_per_lp1" : int(self.nb_grains_per_lp1.text()),
                            "grains_nb_simulate" : int(self.grains_nb_simulate.text()),
                            "detectorparameters" : detector_params,
                            "pixelsize" : float(self.pixelsize.text()),
                            "dim1":float(self.detectordim.text().split(",")[0]),
                            "dim2":float(self.detectordim.text().split(",")[1]),
                            "emin":float(self.minmaxE.text().split(",")[0]),
                            "emax" : float(self.minmaxE.text().split(",")[1]),
                            "batch_size": int(self.batch_size.text()), ## batches of files to use while training
                            "epochs": int(self.epochs.text()), ## number of epochs for training
                            "texture": self.texture_model.currentText(),
                            "mode_nn": self.mode_nn.currentText(),
                            "grid_bool": self.grid_search_hyperparams.currentText(),
                            "directory": self.modelDirecSave,
                            "freq_rmv": int(self.freq_rmv.text()),
                            "freq_rmv1": int(self.freq_rmv1.text()),
                            "elements": self.elements.text(),
                            "elements1": self.elements1.text(),
                            "include_scm": self.include_scm.currentText(),
                            "lr":float(self.learningrate_rc.text().split(",")[0]),
                            "kc" : float(self.learningrate_rc.text().split(",")[1]),
                            "bc":float(self.learningrate_rc.text().split(",")[0]),
                            "architecture":self.architecture.currentText()
                            }
        self.got_signal.emit(emit_dictionary)
        self.close() # close the window

class AnotherWindowLivePrediction(QWidget):#QWidget QScrollArea
    def __init__(self, state=0, gui_state=0, material_=None, material1_=None, emin=None, emax=None, 
                 symmetry=None, symmetry1=None, detectorparameters=None, pixelsize=None, lattice_=None, 
                 lattice1_=None, hkl_all_class0=None, hkl_all_class1=None, mode_spotCycleglobal=None,
                 softmax_threshold_global = None, mr_threshold_global =    None, cap_matchrate =    None,
                 coeff =    None, coeff_overlap1212 =    None, fit_peaks_gaussian_global =    None,
                 FitPixelDev_global =    None, NumberMaxofFits =    None, tolerance_strain =    None, tolerance_strain1 =    None,
                 material0_limit = None, material1_limit=None, symmetry_name=None, symmetry1_name=None,
                 use_previous_UBmatrix_name = None, material_phase_always_present=None, crystal=None, crystal1=None,
                 strain_free_parameters=None, additional_expression=None):
        super(AnotherWindowLivePrediction, self).__init__()
        
        app_icon = QtGui.QIcon()
        app_icon.addFile(Logo, QtCore.QSize(16,16))
        self.setWindowIcon(app_icon)
        self.myQMenuBar = QMenuBar(self)
        self._createMenu()
        
        self.material_phase_always_present = material_phase_always_present
        self.symmetry_name = symmetry_name
        self.symmetry1_name = symmetry1_name
        self.material0_limit = material0_limit
        self.material1_limit = material1_limit
        self.softmax_threshold_global = softmax_threshold_global
        self.mr_threshold_global = mr_threshold_global
        self.cap_matchrate = cap_matchrate
        self.coeff = coeff
        self.coeff_overlap = coeff_overlap1212
        self.fit_peaks_gaussian_global = fit_peaks_gaussian_global
        self.FitPixelDev_global = FitPixelDev_global
        self.NumberMaxofFits = NumberMaxofFits        
        self.tolerance_strain = tolerance_strain
        self.tolerance_strain1 = tolerance_strain1
        self.mode_spotCycle = mode_spotCycleglobal
        self.material_ = material_
        self.material1_ = material1_
        self.files_treated = []
        self.cnt = 0
        self.emin = emin
        self.emax= emax
        self.lattice_ = lattice_
        self.lattice1_ = lattice1_
        self.symmetry = symmetry
        self.symmetry1 = symmetry1
        self.crystal = crystal
        self.crystal1 = crystal1
        self.hkl_all_class0 = hkl_all_class0
        self.hkl_all_class1 = hkl_all_class1
        self.col = np.zeros((10,3))
        self.colx = np.zeros((10,3))
        self.coly = np.zeros((10,3))
        self.match_rate = np.zeros((10,1))
        self.spots_len = np.zeros((10,1))
        self.iR_pix = np.zeros((10,1))
        self.fR_pix = np.zeros((10,1))
        self.mat_global = np.zeros((10,1))
        self.rotation_matrix = np.zeros((10,3,3))
        self.strain_matrix = np.zeros((10,3,3))
        self.strain_matrixs = np.zeros((10,3,3))
        self.strain_calculation = False
        self.use_previous_UBmatrix_name = use_previous_UBmatrix_name
        self.strain_free_parameters = strain_free_parameters
        self.additional_expression = additional_expression
        self.detectorparameters = detectorparameters
        self.pixelsize= pixelsize

        if expfile_global != None:
            self.filenameDirec = expfile_global
        else:
            self.filenameDirec = None
        self.experimental = QPushButton('Browse')
        self.experimental.clicked.connect(self.getfiles1)
        
        self.ipf_axis = QComboBox()
        choices = ["Z","Y","X"]
        for s in choices:
            self.ipf_axis.addItem(s)
        
        self.filenamebkg = None
        self.filename_bkg = QPushButton('Browse')
        self.filename_bkg.clicked.connect(self.getfilebkg_file)
        
        self.blacklist_file = None
        self.filename_blst = QPushButton('Browse')
        self.filename_blst.clicked.connect(self.getfileblst_file)
        
        self.tolerance = QLineEdit()
        self.tolerance.setText("0.5")
        
        self.tolerance1 = QLineEdit()
        self.tolerance1.setText("0.5")
        
        self.image_grid = QLineEdit()
        self.image_grid.setText("10,10")
        
        self.ubmat = QLineEdit()
        self.ubmat.setText("1")
        
        self.bkg_treatment = QLineEdit()
        self.bkg_treatment.setText("A-B")

        if modelfile_global != None:
            self.modelDirec = modelfile_global
        else:
            self.modelDirec = None
        self.model_direc = QPushButton('Browse')
        self.model_direc.clicked.connect(self.getfiles)

        if weightfile_global != None:
            self.filenameModel = [weightfile_global]
        else: 
            self.filenameModel = None
        self.model_path = QPushButton('Browse')
        self.model_path.clicked.connect(self.getfileModel)
        
        self.ccd_label = QComboBox()
        self.ccd_label.addItem("Cor")
        choices = dictLT.dict_CCD.keys()
        for s in choices:
            self.ccd_label.addItem(s)
            
        self.intensity_threshold = QLineEdit()
        self.intensity_threshold.setText("500")
        
        self.mode_peaksearch = QLineEdit()
        self.mode_peaksearch.setText("LaueTools")
        
        self.experimental_prefix = QLineEdit()
        self.experimental_prefix.setText("")
        
        self.boxsize = QLineEdit()
        self.boxsize.setText("15")
        
        self.hkl_plot = QLineEdit()
        self.hkl_plot.setText("[1,1,0],[1,1,1],[1,0,0]")
        
        self.matrix_plot = QComboBox()
        choices = ["1"]
        for s in choices:
            self.matrix_plot.addItem(s)
            
        self.strain_plot = QComboBox()
        choices = ["11_sample","22_sample","33_sample","12_sample","13_sample","23_sample",\
                   "11_crystal","22_crystal","33_crystal","12_crystal","13_crystal","23_crystal"]
        for s in choices:
            self.strain_plot.addItem(s)
        
        self.matrix_plot_tech = QComboBox()
        choices = ["Sequential", "MultiProcessing"]
        for s in choices:
            self.matrix_plot_tech.addItem(s)        
        
        self.analysis_plot_tech = QComboBox()
        choices = ["slow", "graphmode", "update_reupdate", "houghmode"]#, "houghgraphmode"]
        for s in choices:
            self.analysis_plot_tech.addItem(s)
        
        self.strain_plot_tech = QComboBox()
        choices = ["NO", "YES"]
        for s in choices:
            self.strain_plot_tech.addItem(s)
        
        ### default values here
        if tolerance_global != None:
            self.tolerance.setText(str(tolerance_global))
        if tolerance_global1 != None:
            self.tolerance1.setText(str(tolerance_global1))
        if image_grid_globalx != None:
            self.image_grid.setText(str(image_grid_globalx)+","+str(image_grid_globaly))
        if exp_prefix_global != None:
            self.experimental_prefix.setText(exp_prefix_global)
        if ccd_label_global != None:
            self.ccd_label.setCurrentText(ccd_label_global)
        if intensity_threshold_global != None:
            self.intensity_threshold.setText(str(intensity_threshold_global))
        if boxsize_global != None:
            self.boxsize.setText(str(boxsize_global))
        if UB_matrix_global != None:
            self.ubmat.setText(str(UB_matrix_global)) 
        if strain_label_global != None:
            self.strain_plot_tech.setCurrentText(strain_label_global)
        if mode_spotCycle != None:
            self.analysis_plot_tech.setCurrentText(mode_spotCycle)
        if hkls_list_global != None:
            self.hkl_plot.setText(hkls_list_global) 
        if mode_peaksearch != None:
            self.mode_peaksearch.setText(mode_peaksearch)
        # button to continue training
        self.btn_config = QPushButton('Predict and Plot')
        self.btn_config.clicked.connect(self.plot_pc)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.plot_btn_stop)
        self.btn_save = QPushButton("Save data and plots")
        self.btn_save.clicked.connect(self.save_btn)
        self.btn_load = QPushButton("Prediction single file")
        self.btn_load.clicked.connect(self.predict_single_file)
        self.btn_loadall = QPushButton("optimize parameters")
        self.btn_loadall.clicked.connect(self.optimize_parameters)
        self.refresh_replot_button = QPushButton("Refresh/ Replot")
        self.refresh_replot_button.clicked.connect(self.refreshplots)
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(False)
        
        mat_bool = False
        if self.material_ == self.material1_:
            mat_bool = True
        
        self.layout = QVBoxLayout() # QGridLayout()
        self.canvas = MplCanvas(self, width=10, height=10, dpi=100, mat_bool=mat_bool)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        self.canvas.mpl_connect('button_press_event', self.onclickImage)
        self.canvas.mpl_connect('key_press_event', toggle_selector)
        self.canvas.mpl_connect('key_press_event', toggle_selector1)
        
        # set the layout
        self.layout.addWidget(self.toolbar, 0)
        self.layout.addWidget(self.canvas, 100)

        formLayout = QFormLayout()
        formLayout.addRow('Image XY grid size',self.image_grid)
        formLayout.addRow('IPF axis (Cubic and HCP system)', self.ipf_axis)
        formLayout.addRow('Matricies to predict (sequential)', self.ubmat)       
        formLayout.addRow('Matrix to plot', self.matrix_plot) 
        formLayout.addRow('Strain component to plot', self.strain_plot) 
        formLayout.addRow('CPU mode', self.matrix_plot_tech) 
        formLayout.addRow('Analysis mode', self.analysis_plot_tech) 
        formLayout.addRow(self.btn_stop, self.btn_config)
        formLayout.addRow(self.btn_load, self.btn_save)
        # formLayout.addRow('Verify parameters of peak/prediction module', self.btn_loadall)
        formLayout.addRow(self.refresh_replot_button, self.btn_loadall)

        self.layout.addLayout(formLayout)
        self.setLayout(self.layout)
        self.file_state=0
        self.timermp1212 = QtCore.QTimer()
        self.popups = []
        self.old_UB_len = 1
        self.initialize_params()
        self.initialize_plot()

    def _createMenu(self):
        self.menu = self.myQMenuBar.addMenu("&Menu")
        self.menu.addAction('&Load results', self.getresults)
        self.menu.addAction('&Refresh plots', self.refreshplots)
        self.menu.addAction('&Reinitialize', self.reinitialize)

    def getfilesManual(self):
        self.manualfiledirec = QFileDialog.getOpenFileName(self, 'Select a single experimental file (prefix and extension will be grabbed from it)')

    def reinitialize(self):
        self.manualfiledirec = None
        
        try:
            self.getfilesManual()
            print(self.manualfiledirec)
        except:
            print("Error: nothing selected")
            return
        
        text_str = self.manualfiledirec[0]
        
        self.filenameDirec = os.path.dirname(text_str)
        
        _, tail = os.path.split(text_str)
        
        if tail.split(".")[-1] == "tif":
            self.ccd_label.setCurrentText("sCMOS")
        elif tail.split(".")[-1] == "cor" or tail.split(".")[-1] == "Cor" or tail.split(".")[-1] == "COR":
            self.ccd_label.setCurrentText("cor")
        else:
            print("Extension of the selected experimental file not recognized or is not added to the list;")
            print("Add the extension in Line 3983 of lauetoolsneuralnetwork.py file (reinitialize function)")
            return
        
        self.experimental_prefix.setText(tail.split(".")[0][:-4])

        
        ## update matrix plot box?
        if self.matrix_plot.count() < int(self.ubmat.text()):
            for intmat in range(int(self.ubmat.text())):
                if intmat == 0 or intmat < self.matrix_plot.count():
                    continue
                self.matrix_plot.addItem(str(intmat+1))
            self.modify_array()
        self.initialize_params()
        self.initialize_plot()
        
    def getresults(self,):
        self.btn_save.setEnabled(True)
        filenameResults = QFileDialog.getOpenFileName(self, 'Select the results pickle file')
        try:
            self.load_results(filenameResults[0])
        except:
            print("No file selected")
    
    def refreshplots(self):
        ## update matrix plot box?
        if self.matrix_plot.count() < int(self.ubmat.text()):
            for intmat in range(int(self.ubmat.text())):
                if intmat == 0 or intmat < self.matrix_plot.count():
                    continue
                self.matrix_plot.addItem(str(intmat+1))
            self.modify_array()
        self.initialize_plot()
        
    def load_results(self, filename):
        self.file_state=0
        self.timermp1212 = QtCore.QTimer()
        self.popups = []
        self.old_UB_len = 1
        self.initialize_params()
        self.initialize_plot()
        try:
            with open(filename, "rb") as input_file:
                self.best_match, \
                self.mat_global, self.rotation_matrix, self.strain_matrix, self.strain_matrixs,\
                    self.col, self.colx, self.coly, self.match_rate, self.files_treated,\
                        self.lim_x, self.lim_y, self.spots_len, self.iR_pix, self.fR_pix, self.material_, \
                            self.material1_, self.lattice, self.lattice1, self.symmetry, self.symmetry1,\
                                self.crystal, self.crystal1 = cPickle.load(input_file)
        except:
            try:
                with open(filename, "rb") as input_file:
                    self.best_match, \
                    self.mat_global, self.rotation_matrix, self.strain_matrix, self.strain_matrixs,\
                        self.col, self.colx, self.coly, self.match_rate, self.files_treated,\
                            self.lim_x, self.lim_y, self.spots_len, self.iR_pix, self.fR_pix, self.material_, \
                                self.material1_, self.lattice, self.lattice1, \
                                    self.symmetry, self.symmetry1 = cPickle.load(input_file)
            except:
                try:
                    with open(filename, "rb") as input_file:
                        self.mat_global, self.rotation_matrix, self.strain_matrix, self.strain_matrixs,\
                            self.col, self.colx, self.coly, self.match_rate, self.files_treated,\
                                self.lim_x, self.lim_y = cPickle.load(input_file)
                except:
                    print("Script version results")
                    ##script version results (only supports the first two phase)
                    with open(filename, "rb") as input_file:
                        self.best_match, \
                        self.mat_global, self.rotation_matrix, self.strain_matrix, self.strain_matrixs,\
                            self.col, self.colx, self.coly, self.match_rate, self.files_treated,\
                                self.lim_x, self.lim_y, self.spots_len, self.iR_pix, self.fR_pix, self.material0_, \
                                    self.lattice0, \
                                        self.symmetry0, self.crystal0 = cPickle.load(input_file)
                                        
                    self.material_, self.material1_ = self.material0_[0], self.material0_[1]
                    self.lattice, self.lattice1 = self.lattice0[0], self.lattice0[1]
                    self.symmetry, self.symmetry1 = self.symmetry0[0], self.symmetry0[1]
                    self.crystal, self.crystal1 = self.crystal0[0], self.crystal0[1]
                                
        self.ubmat.setText(str(len(self.rotation_matrix)))
        ## update matrix plot box?
        if self.matrix_plot.count() < int(self.ubmat.text()):
            for intmat in range(len(self.rotation_matrix)):
                if intmat == 0:
                    continue
                self.matrix_plot.addItem(str(intmat+1))
            
        cond = self.strain_plot_tech.currentText()
        self.strain_calculation = False
        if cond == "YES":
            self.strain_calculation = True
        ## Number of files to generate
        grid_files = np.zeros((self.lim_x,self.lim_y))
        self.filenm = np.chararray((self.lim_x,self.lim_y), itemsize=1000)
        grid_files = grid_files.ravel()
        self.filenm = self.filenm.ravel()
        count_global = self.lim_x * self.lim_y
        
        if self.ccd_label.currentText() == "Cor" or self.ccd_label.currentText() == "cor":
            format_file = "cor"
        else:
            format_file = dictLT.dict_CCD[self.ccd_label.currentText()][7]
        list_of_files = glob.glob(self.filenameDirec+'//'+self.experimental_prefix.text()+'*.'+format_file)
        ## sort files
        ## TypeError: '<' not supported between instances of 'str' and 'int'
        list_of_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
        if len(list_of_files) == count_global:
            for ii in range(len(list_of_files)):
                grid_files[ii] = ii
                self.filenm[ii] = list_of_files[ii]               
        else:
            print("expected "+str(count_global)+" files based on the XY grid ("+str(self.lim_x)+","+str(self.lim_y)+") defined by user")
            print("But found "+str(len(list_of_files))+" files (either all data is not written yet or maybe XY grid definition is not proper)")
            digits = len(str(count_global))
            digits = max(digits,4)
            for ii in range(count_global):
                text = str(ii)
                string = text.zfill(digits)
                file_name_temp = self.filenameDirec+'//'+self.experimental_prefix.text()+string+'.'+format_file
                ## store it in a grid 
                self.filenm[ii] = file_name_temp
        ### Create a COR directory to be loaded in LaueTools
        self.cor_file_directory = self.filenameDirec + "//" + self.experimental_prefix.text()+"CORfiles"
        if format_file in ['cor',"COR","Cor"]:
            self.cor_file_directory = self.filenameDirec
        if not os.path.exists(self.cor_file_directory):
            os.makedirs(self.cor_file_directory)
        self.initialize_plot()
        self.refreshplots()

    def closeEvent(self, event):
        self.close
        super().closeEvent(event)
    
    def getfilebkg_file(self):
        self.filenamebkg = QFileDialog.getOpenFileName(self, 'Select the background image of same detector')
    
    def getfileblst_file(self):
        self.blacklist_file = QFileDialog.getOpenFileName(self, 'Select the list of peaks DAT file to blacklist')
    
    def initialize_params(self):
        self.model_direc = self.modelDirec
        
        self.lim_x, self.lim_y = int(self.image_grid.text().split(",")[0]), int(self.image_grid.text().split(",")[1])
        if self.cnt == 0:
            self.col = [[] for i in range(int(self.ubmat.text()))]
            self.colx = [[] for i in range(int(self.ubmat.text()))]
            self.coly = [[] for i in range(int(self.ubmat.text()))]
            self.rotation_matrix = [[] for i in range(int(self.ubmat.text()))]
            self.strain_matrix = [[] for i in range(int(self.ubmat.text()))]
            self.strain_matrixs = [[] for i in range(int(self.ubmat.text()))]
            self.match_rate = [[] for i in range(int(self.ubmat.text()))]
            self.spots_len = [[] for i in range(int(self.ubmat.text()))]
            self.iR_pix = [[] for i in range(int(self.ubmat.text()))]
            self.fR_pix = [[] for i in range(int(self.ubmat.text()))]
            self.mat_global = [[] for i in range(int(self.ubmat.text()))]
            self.best_match = [[] for i in range(int(self.ubmat.text()))]
            self.spots1_global = [[] for i in range(int(self.ubmat.text()))]
            for i in range(int(self.ubmat.text())):
                self.col[i].append(np.zeros((self.lim_x*self.lim_y,3)))
                self.colx[i].append(np.zeros((self.lim_x*self.lim_y,3)))
                self.coly[i].append(np.zeros((self.lim_x*self.lim_y,3)))
                self.rotation_matrix[i].append(np.zeros((self.lim_x*self.lim_y,3,3)))
                self.strain_matrix[i].append(np.zeros((self.lim_x*self.lim_y,3,3)))
                self.strain_matrixs[i].append(np.zeros((self.lim_x*self.lim_y,3,3)))
                self.match_rate[i].append(np.zeros((self.lim_x*self.lim_y,1)))
                self.spots_len[i].append(np.zeros((self.lim_x*self.lim_y,1)))
                self.iR_pix[i].append(np.zeros((self.lim_x*self.lim_y,1)))
                self.fR_pix[i].append(np.zeros((self.lim_x*self.lim_y,1)))
                self.mat_global[i].append(np.zeros((self.lim_x*self.lim_y,1)))
                self.best_match[i].append([[] for jk in range(self.lim_x*self.lim_y)])
                self.spots1_global[i].append([[] for jk in range(self.lim_x*self.lim_y)])
        count_global = self.lim_x * self.lim_y        
        self.check = np.zeros((count_global,int(self.ubmat.text())))
        self.old_UB_len = int(self.ubmat.text())
        ## load model related files and generate the model
        if self.material_ != self.material1_:
            json_file = open(self.model_direc+"//model_"+self.material_+"_"+self.material1_+".json", 'r')
        else:
            json_file = open(self.model_direc+"//model_"+self.material_+".json", 'r')
                
        self.classhkl = np.load(self.model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_0"]
        self.angbins = np.load(self.model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_1"]
        if self.material_ != self.material1_:
            self.ind_mat = np.load(self.model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_5"]
            self.ind_mat1 = np.load(self.model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_6"]
        else: 
            self.ind_mat = None
            self.ind_mat1 = None  
        load_weights = self.filenameModel[0]
        self.wb = read_hdf5(load_weights)
        self.temp_key = list(self.wb.keys())
        # # load json and create model
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        print("Constructing model")
        load_weights = self.filenameModel[0]
        self.model.load_weights(load_weights)
        print("Uploading weights to model")
        print("All model files found and loaded")
        self.mode_spotCycle = self.analysis_plot_tech.currentText()
        
        if self.use_previous_UBmatrix_name:
            np.savez_compressed(self.model_direc+'//rotation_matrix_indexed_1.npz', self.rotation_matrix, self.mat_global, self.match_rate, 0.0)
        
        cond = self.strain_plot_tech.currentText()
        self.strain_calculation = False
        if cond == "YES":
            self.strain_calculation = True
        self.ncpu = cpu_count_user
        try:
            # =============================================================================
            #         ## Multi-processing routine
            # =============================================================================
            ## Number of files to generate
            grid_files = np.zeros((self.lim_x,self.lim_y))
            self.filenm = np.chararray((self.lim_x,self.lim_y), itemsize=1000)
            grid_files = grid_files.ravel()
            self.filenm = self.filenm.ravel()
            if self.ccd_label.currentText() == "Cor" or self.ccd_label.currentText() == "cor":
                format_file = "cor"
            else:
                format_file = dictLT.dict_CCD[self.ccd_label.currentText()][7]
            list_of_files = glob.glob(self.filenameDirec+'//'+self.experimental_prefix.text()+'*.'+format_file)
            ## sort files
            ## TypeError: '<' not supported between instances of 'str' and 'int'
            list_of_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    
            if len(list_of_files) == count_global:
                for ii in range(len(list_of_files)):
                    grid_files[ii] = ii
                    self.filenm[ii] = list_of_files[ii]               
            else:
                print("expected "+str(count_global)+" files based on the XY grid ("+str(self.lim_x)+","+str(self.lim_y)+") defined by user")
                print("But found "+str(len(list_of_files))+" files (either all data is not written yet or maybe XY grid definition is not proper)")
                digits = len(str(count_global))
                digits = max(digits,4)
    
                for ii in range(count_global):
                    text = str(ii)
                    if ii < 10000:
                        string = text.zfill(4)
                    else:
                        string = text.zfill(5)
                    file_name_temp = self.filenameDirec+'//'+self.experimental_prefix.text()+string+'.'+format_file
                    ## store it in a grid 
                    self.filenm[ii] = file_name_temp
                ## access grid files to process with multi-thread
            self.cor_file_directory = self.filenameDirec + "//" + self.experimental_prefix.text()+"CORfiles"
            if format_file in ['cor',"COR","Cor"]:
                self.cor_file_directory = self.filenameDirec
            if not os.path.exists(self.cor_file_directory):
                os.makedirs(self.cor_file_directory)
        except:
            print("No directory for experimental data is defined; please reinitialize from the menu in the new window")
            self.cor_file_directory = None
            self.filenameDirec = None
            self.filenm = np.chararray((10,10), itemsize=1000)
            self.filenm = self.filenm.ravel()
            
    
    def modify_array(self):
        if self.old_UB_len < int(self.ubmat.text()):            
            differen = abs(self.old_UB_len - int(self.ubmat.text()))
            for iji in range(differen):
                self.col.append([])
                self.colx.append([])
                self.coly.append([])
                self.rotation_matrix.append([])
                self.strain_matrix.append([])
                self.strain_matrixs.append([])
                self.match_rate.append([])
                self.spots_len.append([])
                self.iR_pix.append([])
                self.fR_pix.append([])
                self.mat_global.append([])
                self.best_match.append([])
                self.spots1_global.append([])
                
            for iji in range(differen):
                indd = int(self.old_UB_len + iji)
                self.col[indd].append(np.zeros((self.lim_x*self.lim_y,3)))
                self.colx[indd].append(np.zeros((self.lim_x*self.lim_y,3)))
                self.coly[indd].append(np.zeros((self.lim_x*self.lim_y,3)))
                self.rotation_matrix[indd].append(np.zeros((self.lim_x*self.lim_y,3,3)))
                self.strain_matrix[indd].append(np.zeros((self.lim_x*self.lim_y,3,3)))
                self.strain_matrixs[indd].append(np.zeros((self.lim_x*self.lim_y,3,3)))
                self.match_rate[indd].append(np.zeros((self.lim_x*self.lim_y,1)))
                self.spots_len[indd].append(np.zeros((self.lim_x*self.lim_y,1)))
                self.iR_pix[indd].append(np.zeros((self.lim_x*self.lim_y,1)))
                self.fR_pix[indd].append(np.zeros((self.lim_x*self.lim_y,1)))
                self.mat_global[indd].append(np.zeros((self.lim_x*self.lim_y,1)))
                self.best_match[indd].append([[] for jk in range(self.lim_x*self.lim_y)])
                self.spots1_global[indd].append([[] for jk in range(self.lim_x*self.lim_y)])
                self.check = np.c_[self.check, np.zeros(len(self.check))]
            
            self.old_UB_len = int(self.ubmat.text())
        
    def predict_single_file_nodialog(self, filenameSingleExp, intensity_threshold_global,
                                     boxsize_global, FitPixelDev_global, softmax_threshold_global_123,
                                     cap_matchrate_123, cnt,
                                     nb_spots_consider_global123, residues_threshold123,
                                     option_global123, nb_spots_global_threshold123, peakXY,
                                     strain_free_parameters,additional_expression,mode_peaksearch):
        self.modify_array()
        ## update the main config ini file
        config_setting = configparser.ConfigParser()
        filepath = resource_path('settings.ini')
        config_setting.read(filepath)
        use_om_user = config_setting.get('CALLER', 'use_om_user')
        path_user_OM = config_setting.get('CALLER', 'path_user_OM')
        
        config_setting1 = configparser.ConfigParser()
        print("Settings path is "+filepath)
        config_setting1.read(filepath)
        config_setting1.set('CALLER', 'residues_threshold',str(residues_threshold123))
        config_setting1.set('CALLER', 'nb_spots_global_threshold',str(nb_spots_global_threshold123))
        config_setting1.set('CALLER', 'option_global',str(option_global123))
        config_setting1.set('CALLER', 'use_om_user',str(use_om_user))
        config_setting1.set('CALLER', 'nb_spots_consider',str(nb_spots_consider_global123))
        config_setting1.set('CALLER', 'path_user_OM',path_user_OM)
        config_setting1.set('CALLER', 'intensity', str(intensity_threshold_global))
        config_setting1.set('CALLER', 'boxsize', str(boxsize_global))
        config_setting1.set('CALLER', 'pixdev', str(FitPixelDev_global))
        config_setting1.set('CALLER', 'cap_softmax', str(softmax_threshold_global_123))
        config_setting1.set('CALLER', 'cap_mr', str(cap_matchrate_123))
        config_setting1.set('CALLER', 'strain_free_parameters', ",".join(strain_free_parameters))
        config_setting1.set('CALLER', 'additional_expression', ",".join(additional_expression))
        config_setting1.set('CALLER', 'mode_peaksearch', str(mode_peaksearch))
        
        with open(filepath, 'w') as configfile:
            config_setting1.write(configfile)
            
        ## Provide path to a single tiff or cor file to predict and write a pickle object
        lim_x, lim_y = int(1), int(1)
        
        cond = self.strain_plot_tech.currentText()
        self.strain_calculation = False
        if cond == "YES":
            self.strain_calculation = True

        ## access grid files to process with multi-thread
        check = np.zeros((1,int(self.ubmat.text())))
        
        mode_analysis = self.analysis_plot_tech.currentText()

        start_time = time.time()
        col = [[] for i in range(int(self.ubmat.text()))]
        colx = [[] for i in range(int(self.ubmat.text()))]
        coly = [[] for i in range(int(self.ubmat.text()))]
        rotation_matrix = [[] for i in range(int(self.ubmat.text()))]
        strain_matrix = [[] for i in range(int(self.ubmat.text()))]
        strain_matrixs = [[] for i in range(int(self.ubmat.text()))]
        match_rate = [[] for i in range(int(self.ubmat.text()))]
        spots_len = [[] for i in range(int(self.ubmat.text()))]
        iR_pix = [[] for i in range(int(self.ubmat.text()))]
        fR_pix = [[] for i in range(int(self.ubmat.text()))]
        mat_global = [[] for i in range(int(self.ubmat.text()))]
        best_match = [[] for i in range(int(self.ubmat.text()))]
        for i in range(int(self.ubmat.text())):
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
        
        ##calculate neighbor_UB to be passed as variable directly 
        
        strain_matrix_mpdata, strain_matrixs_mpdata, \
        rotation_matrix_mpdata, col_mpdata, \
        colx_mpdata, coly_mpdata,\
        match_rate_mpdata, mat_global_mpdata, cnt_mpdata,\
        files_treated_mpdata, spots_len_mpdata, \
        iR_pixel_mpdata, fR_pixel_mpdata, check_mpdata, \
            best_match_mpdata, pred_hkl = predict_preprocessMP_vsingle(filenameSingleExp, 0, 
                                                   rotation_matrix,strain_matrix,strain_matrixs,
                                                   col,colx,coly,match_rate,spots_len,iR_pix,fR_pix,best_match,
                                                   mat_global,
                                                   check,self.detectorparameters,self.pixelsize,self.angbins,
                                                   self.classhkl, self.hkl_all_class0, self.hkl_all_class1, self.emin, self.emax,
                                                   self.material_, self.material1_, self.symmetry, self.symmetry1,lim_x,lim_y,
                                                   self.strain_calculation, self.ind_mat, self.ind_mat1,
                                                   self.model_direc, float(self.tolerance.text()), float(self.tolerance1.text()),
                                                   int(self.ubmat.text()), self.ccd_label.currentText(),
                                                   self.filenameDirec, self.experimental_prefix.text(),
                                                   [],False,
                                                   self.wb, self.temp_key, self.cor_file_directory, mode_analysis,
                                                    softmax_threshold_global_123,
                                                    self.mr_threshold_global,
                                                    cap_matchrate_123,
                                                    self.tolerance_strain,
                                                    self.tolerance_strain1,
                                                    self.coeff,
                                                    self.coeff_overlap,
                                                    self.material0_limit,
                                                    self.material1_limit,
                                                    False,
                                                    self.material_phase_always_present,
                                                    self.crystal,
                                                    self.crystal1,peakXY,
                                                    strain_free_parameters)
        end_time = time.time() - start_time
        print("Total time to process one file in "+mode_analysis+" mode (in seconds): "+str(end_time))

        try:  
            path = os.path.normpath(filenameSingleExp)
            files = self.cor_file_directory+"//"+path.split(os.sep)[-1].split(".")[0]+".cor"
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
        except:
            print("No COR/ Exp. file could be found for the selected filename")
            return
        
        sim_twotheta1, sim_chi1, list_spots1, residues1, theo_index1 = [],[],[],[],[]
        sim_energy1 = []
        sim_hkl1 = []
        for ijk in range(len(match_rate_mpdata)):
            mat_global987 = mat_global_mpdata[ijk][0]
            rotation_matrix987 = rotation_matrix_mpdata[ijk][0][0]
            if mat_global987 == 1:
                material_=self.material_
                tolerance_add = float(self.tolerance.text())                
            elif mat_global987 == 2:
                material_=self.material1_
                tolerance_add = float(self.tolerance1.text())
            else:
                print("Matrix "+str(ijk+1)+" is not found")
                material_ = None
                tolerance_add = None
                sim_twotheta = []
                sim_chi = []
                list_spots = []
                residues = []
                theo_index = []
                sim_energy = []
                sim_hkl = []
            
            if np.all(rotation_matrix987==0):
                material_ = None
                sim_twotheta = []
                sim_chi = []
                list_spots = []
                residues = []
                theo_index = []
                sim_energy = []
                sim_hkl = []
                print("No rotation matrix found")
            
            if material_ != None:
                sim_twotheta, sim_chi, sim_energy, sim_hkl,\
                list_spots, residues, theo_index = simulate_spots(rotation_matrix987, 
                                                                    material_, self.emax, self.emin, 
                                                                    dict_dp['detectorparameters'], dict_dp,
                                                                    tolerance_add, data_theta*2.0,
                                                                    data_chi)
                if len(sim_twotheta) == 0:
                    sim_twotheta = []
                    sim_chi = []
                    list_spots = []
                    residues = []
                    theo_index = []
                    sim_energy = []
                    sim_hkl = []
                    print("Nothing simulated")
            sim_twotheta1.append(sim_twotheta)
            sim_chi1.append(sim_chi)
            list_spots1.append(list_spots)
            residues1.append(residues)
            theo_index1.append(theo_index)
            sim_energy1.append(sim_energy)
            sim_hkl1.append(sim_hkl)

        w = MyPopup(match_rate_mpdata, rotation_matrix_mpdata, mat_global_mpdata, fR_pixel_mpdata, \
                    filenameSingleExp, strain_matrix_mpdata, strain_matrixs_mpdata, end_time, mode_analysis,
                    data_theta, data_chi, intensity, sim_twotheta1, sim_chi1, sim_energy1, sim_hkl1,
                    list_spots1, residues1, theo_index1, pred_hkl)
        w.show()       
        self.popups.append(w)
        
        #_ update the count to actual image number
        cnt_mpdata = cnt
        for i_mpdata in files_treated_mpdata:
            self.files_treated.append(i_mpdata)
                        
        for intmat_mpdata in range(int(self.ubmat.text())):
            self.check[cnt_mpdata,intmat_mpdata] = check_mpdata[0,intmat_mpdata]
            self.mat_global[intmat_mpdata][0][cnt_mpdata] = mat_global_mpdata[intmat_mpdata][0][0]
            self.strain_matrix[intmat_mpdata][0][cnt_mpdata,:,:] = strain_matrix_mpdata[intmat_mpdata][0][0,:,:]
            self.strain_matrixs[intmat_mpdata][0][cnt_mpdata,:,:] = strain_matrixs_mpdata[intmat_mpdata][0][0,:,:]
            self.rotation_matrix[intmat_mpdata][0][cnt_mpdata,:,:] = rotation_matrix_mpdata[intmat_mpdata][0][0,:,:]
            self.col[intmat_mpdata][0][cnt_mpdata,:] = col_mpdata[intmat_mpdata][0][0,:]
            self.colx[intmat_mpdata][0][cnt_mpdata,:] = colx_mpdata[intmat_mpdata][0][0,:]
            self.coly[intmat_mpdata][0][cnt_mpdata,:] = coly_mpdata[intmat_mpdata][0][0,:]
            self.match_rate[intmat_mpdata][0][cnt_mpdata] = match_rate_mpdata[intmat_mpdata][0][0]
            self.spots_len[intmat_mpdata][0][cnt_mpdata] = spots_len_mpdata[intmat_mpdata][0][0]
            self.iR_pix[intmat_mpdata][0][cnt_mpdata] = iR_pixel_mpdata[intmat_mpdata][0][0]
            self.fR_pix[intmat_mpdata][0][cnt_mpdata] = fR_pixel_mpdata[intmat_mpdata][0][0]
            self.best_match[intmat_mpdata][0][cnt_mpdata] = best_match_mpdata[intmat_mpdata][0][0]
            
        self.update_plot()
    
    def optimize_parameters(self,):
        self.modify_array()
        # Idea is to open the raster grid here and check the peak search and prediction hyperparameters
        w = Window_allmap(self.lim_x, self.lim_y, self.filenm, 
                          self.ccd_label.currentText(), self.predict_single_file_nodialog,
                          self.detectorparameters)
        w.show()       
        self.popups.append(w)
        
    def predict_single_file(self,):
        
        ## getting settings from settings.ini file
        try:
            config_setting = configparser.ConfigParser()
            filepath = resource_path('settings.ini')
            config_setting.read(filepath)
            residues_threshold = float(config_setting.get('CALLER', 'residues_threshold'))
            nb_spots_global_threshold = int(config_setting.get('CALLER', 'nb_spots_global_threshold'))
            option_global = config_setting.get('CALLER', 'option_global')
            nb_spots_consider = int(config_setting.get('CALLER', 'nb_spots_consider'))
            intensity_threshold = int(float(config_setting.get('CALLER', 'intensity')))
            boxsize = int(float(config_setting.get('CALLER', 'boxsize')))
            FitPixelDev_global123 = int(float(config_setting.get('CALLER', 'pixdev')))
            softmax_threshold_global123 = float(config_setting.get('CALLER', 'cap_softmax'))
            cap_matchrate123 = float(config_setting.get('CALLER', 'cap_mr'))
            strain_free_parameters = config_setting.get('CALLER', 'strain_free_parameters').split(",")
            additional_expression = config_setting.get('CALLER', 'additional_expression').split(",")
            mode_peaksearch = str(config_setting.get('CALLER', 'mode_peaksearch'))
            try_settings_param = True
        except:
            try_settings_param = False
            print("error reading the setting file, sing Default values")
        
        ## Provide path to a single tiff or cor file to predict and write a pickle object
        filenameSingleExp = QFileDialog.getOpenFileName(self, 'Select a single experimental file',
                                                        resource_path("examples"))
        if len(filenameSingleExp[0]) == 0:
            return
        filenameSingleExp = filenameSingleExp[0]
        model_direc = self.modelDirec
        
        lim_x, lim_y = int(1), int(1)
                
        ## load model related files and generate the model
        if self.material_ != self.material1_:
            json_file = open(model_direc+"//model_"+self.material_+"_"+self.material1_+".json", 'r')
        else:
            json_file = open(model_direc+"//model_"+self.material_+".json", 'r')
                
        classhkl = np.load(model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_0"]
        angbins = np.load(model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_1"]
        
        if self.material_ != self.material1_:
            ind_mat = np.load(model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_5"]
            ind_mat1 = np.load(model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_6"]
        else: 
            ind_mat = None
            ind_mat1 = None  
        
        load_weights = self.filenameModel[0]
        wb = read_hdf5(load_weights)
        temp_key = list(wb.keys())
        
        # # load json and create model
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        print("Constructing model")
        load_weights = self.filenameModel[0]
        model.load_weights(load_weights)
        print("Uploading weights to model")
        print("All model files found and loaded")
        
        cond = self.strain_plot_tech.currentText()
        self.strain_calculation = False
        if cond == "YES":
            self.strain_calculation = True

        ## access grid files to process with multi-thread
        check = np.zeros((1,int(self.ubmat.text())))
        # =============================================================================
        try:
            blacklist = self.blacklist_file[0]
        except:
            blacklist = None
        
        ### Create a COR directory to be loaded in LaueTools
        forma = filenameSingleExp.split(".")[1]
        cor_file_directory = self.filenameDirec + "//" + self.experimental_prefix.text()+"CORfiles"
        if forma in ['cor',"COR","Cor"]:
            cor_file_directory = self.filenameDirec
        if not os.path.exists(cor_file_directory):
            os.makedirs(cor_file_directory)
        
        mode_analysis = self.analysis_plot_tech.currentText()

        start_time = time.time()
        col = [[] for i in range(int(self.ubmat.text()))]
        colx = [[] for i in range(int(self.ubmat.text()))]
        coly = [[] for i in range(int(self.ubmat.text()))]
        rotation_matrix = [[] for i in range(int(self.ubmat.text()))]
        strain_matrix = [[] for i in range(int(self.ubmat.text()))]
        strain_matrixs = [[] for i in range(int(self.ubmat.text()))]
        match_rate = [[] for i in range(int(self.ubmat.text()))]
        spots_len = [[] for i in range(int(self.ubmat.text()))]
        iR_pix = [[] for i in range(int(self.ubmat.text()))]
        fR_pix = [[] for i in range(int(self.ubmat.text()))]
        mat_global = [[] for i in range(int(self.ubmat.text()))]
        best_match = [[] for i in range(int(self.ubmat.text()))]
        for i in range(int(self.ubmat.text())):
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
        
        if try_settings_param:
            strain_matrix12, strain_matrixs12, \
            rotation_matrix12, col12, \
            colx12, coly12,\
            match_rate12, mat_global12, cnt12,\
            files_treated12, spots_len12, \
            iR_pix12, fR_pix12, check12, \
                best_match12, pred_hkl = predict_preprocessMP(filenameSingleExp, 0, 
                                                       rotation_matrix,strain_matrix,strain_matrixs,
                                                       col,colx,coly,match_rate,spots_len,iR_pix,fR_pix,best_match,
                                                       mat_global,
                                                       check,self.detectorparameters,self.pixelsize,angbins,
                                                       classhkl, self.hkl_all_class0, self.hkl_all_class1, self.emin, self.emax,
                                                       self.material_, self.material1_, self.symmetry, self.symmetry1,lim_x,lim_y,
                                                       self.strain_calculation, ind_mat, ind_mat1,
                                                       model_direc, float(self.tolerance.text()), float(self.tolerance1.text()),
                                                       int(self.ubmat.text()), self.ccd_label.currentText(),
                                                       None,float(intensity_threshold),
                                                       int(boxsize),self.bkg_treatment.text(),
                                                       self.filenameDirec, self.experimental_prefix.text(),
                                                       blacklist, None, 
                                                       [],False,
                                                       wb, temp_key, cor_file_directory, mode_analysis,
                                                        softmax_threshold_global123,
                                                        self.mr_threshold_global,
                                                        cap_matchrate123,
                                                        self.tolerance_strain,
                                                        self.tolerance_strain1,
                                                        self.NumberMaxofFits,
                                                        self.fit_peaks_gaussian_global,
                                                        FitPixelDev_global123,
                                                        self.coeff,
                                                        self.coeff_overlap,
                                                        self.material0_limit,
                                                        self.material1_limit,
                                                        False,
                                                        self.material_phase_always_present,
                                                        self.crystal,
                                                        self.crystal1,
                                                        strain_free_parameters,
                                                        mode_peaksearch)
        else:
            strain_matrix12, strain_matrixs12, \
            rotation_matrix12, col12, \
            colx12, coly12,\
            match_rate12, mat_global12, cnt12,\
            files_treated12, spots_len12, \
            iR_pix12, fR_pix12, check12, \
                best_match12, pred_hkl = predict_preprocessMP(filenameSingleExp, 0, 
                                                       rotation_matrix,strain_matrix,strain_matrixs,
                                                       col,colx,coly,match_rate,spots_len,iR_pix,fR_pix,best_match,
                                                       mat_global,
                                                       check,self.detectorparameters,self.pixelsize,angbins,
                                                       classhkl, self.hkl_all_class0, self.hkl_all_class1, self.emin, self.emax,
                                                       self.material_, self.material1_, self.symmetry, self.symmetry1,lim_x,lim_y,
                                                       self.strain_calculation, ind_mat, ind_mat1,
                                                       model_direc, float(self.tolerance.text()), float(self.tolerance1.text()),
                                                       int(self.ubmat.text()), self.ccd_label.currentText(),
                                                       None,float(self.intensity_threshold.text()),
                                                       int(self.boxsize.text()),self.bkg_treatment.text(),
                                                       self.filenameDirec, self.experimental_prefix.text(),
                                                       blacklist, None, 
                                                       [],False,
                                                       wb, temp_key, cor_file_directory, mode_analysis,
                                                        self.softmax_threshold_global,
                                                        self.mr_threshold_global,
                                                        self.cap_matchrate,
                                                        self.tolerance_strain,
                                                        self.tolerance_strain1,
                                                        self.NumberMaxofFits,
                                                        self.fit_peaks_gaussian_global,
                                                        self.FitPixelDev_global,
                                                        self.coeff,
                                                        self.coeff_overlap,
                                                        self.material0_limit,
                                                        self.material1_limit,
                                                        False,
                                                        self.material_phase_always_present,
                                                        self.crystal,
                                                        self.crystal1,
                                                        self.strain_free_parameters,
                                                        self.mode_peaksearch.text())
        end_time = time.time() - start_time
        print("Total time to process one file in "+mode_analysis+" mode (in seconds): "+str(end_time))
        
        save_name = filenameSingleExp.split(".")[0].split("/")[-1]
        np.savez_compressed(model_direc+'//'+save_name+"_"+mode_analysis+'.npz', strain_matrix12, strain_matrixs12, \
                            rotation_matrix12, col12, colx12, coly12, match_rate12, mat_global12, cnt12,\
                            files_treated12, spots_len12, iR_pix12, fR_pix12, check12, best_match12)
        
        try:  
            path = os.path.normpath(filenameSingleExp)
            files = cor_file_directory+"//"+path.split(os.sep)[-1].split(".")[0]+".cor"
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
            dict_dp['detectordiameter']=pixelsize*framedim[0]#TODO*2
            dict_dp['pixelsize']=pixelsize
            dict_dp['dim']=framedim
            dict_dp['peakX']=peakx
            dict_dp['peakY']=peaky
            dict_dp['intensity']=intensity
        except:
            try:
                allres = IOLT.readfile_cor(filenameSingleExp, True)
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
                dict_dp['detectordiameter']=pixelsize*framedim[0]#TODO*2
                dict_dp['pixelsize']=pixelsize
                dict_dp['dim']=framedim
                dict_dp['peakX']=peakx
                dict_dp['peakY']=peaky
                dict_dp['intensity']=intensity
            except:
                print("No COR/ Exp. file could be found for the selected filename")
                return
        
        sim_twotheta1, sim_chi1, list_spots1, residues1, theo_index1 = [],[],[],[],[]
        sim_energy1 = []
        sim_hkl1 = []
        for ijk in range(len(match_rate12)):
            mat_global987 = mat_global12[ijk][0]
            rotation_matrix987 = rotation_matrix12[ijk][0][0]
            if mat_global987 == 1:
                material_=self.material_
                tolerance_add = float(self.tolerance.text())                
            elif mat_global987 == 2:
                material_=self.material1_
                tolerance_add = float(self.tolerance1.text())
            else:
                print("Matrix "+str(ijk+1)+" is not found")
                material_ = None
                tolerance_add = None
                sim_twotheta = []
                sim_chi = []
                list_spots = []
                residues = []
                theo_index = []
                sim_energy = []
                sim_hkl = []
            
            if np.all(rotation_matrix987==0):
                material_ = None
                sim_twotheta = []
                sim_chi = []
                list_spots = []
                residues = []
                theo_index = []
                sim_energy = []
                sim_hkl = []
                print("No rotation matrix found")
            
            if material_ != None:
                sim_twotheta, sim_chi, sim_energy, sim_hkl,\
                list_spots, residues, theo_index = simulate_spots(rotation_matrix987, 
                                                                    material_, self.emax, self.emin, 
                                                                    dict_dp['detectorparameters'], dict_dp,
                                                                    tolerance_add, data_theta*2.0,
                                                                    data_chi)
                if len(sim_twotheta) == 0:
                    sim_twotheta = []
                    sim_chi = []
                    list_spots = []
                    residues = []
                    theo_index = []
                    sim_energy = []
                    sim_hkl = []
                    print("Nothing simulated")
            sim_twotheta1.append(sim_twotheta)
            sim_chi1.append(sim_chi)
            list_spots1.append(list_spots)
            residues1.append(residues)
            theo_index1.append(theo_index)
            sim_energy1.append(sim_energy)
            sim_hkl1.append(sim_hkl)
            
        w = MyPopup(match_rate12, rotation_matrix12, mat_global12, fR_pix12, \
                    filenameSingleExp, strain_matrix12, strain_matrixs12, end_time, mode_analysis,
                    data_theta, data_chi, intensity, sim_twotheta1, sim_chi1, sim_energy1, sim_hkl1,
                    list_spots1, residues1, theo_index1, pred_hkl)
        # w.setGeometry(QRect(100, 100, 400, 200))
        w.show()       
        self.popups.append(w)

        #TODO cnt12 is 0 i.e. does not correspond to the image number
        # cnt_mpdata = cnt12
        # for i_mpdata in files_treated12:
        #     self.files_treated.append(i_mpdata)

        # for intmat_mpdata in range(int(self.ubmat.text())):
        #     self.check[cnt_mpdata,intmat_mpdata] = check12[0,intmat_mpdata]
        #     self.mat_global[intmat_mpdata][0][cnt_mpdata] = mat_global12[intmat_mpdata][0][0]
        #     self.strain_matrix[intmat_mpdata][0][cnt_mpdata,:,:] = strain_matrix12[intmat_mpdata][0][0,:,:]
        #     self.strain_matrixs[intmat_mpdata][0][cnt_mpdata,:,:] = strain_matrixs12[intmat_mpdata][0][0,:,:]
        #     self.rotation_matrix[intmat_mpdata][0][cnt_mpdata,:,:] = rotation_matrix12[intmat_mpdata][0][0,:,:]
        #     self.col[intmat_mpdata][0][cnt_mpdata,:] = col12[intmat_mpdata][0][0,:]
        #     self.colx[intmat_mpdata][0][cnt_mpdata,:] = colx12[intmat_mpdata][0][0,:]
        #     self.coly[intmat_mpdata][0][cnt_mpdata,:] = coly12[intmat_mpdata][0][0,:]
        #     self.match_rate[intmat_mpdata][0][cnt_mpdata] = match_rate12[intmat_mpdata][0][0]
        #     self.spots_len[intmat_mpdata][0][cnt_mpdata] = spots_len12[intmat_mpdata][0][0]
        #     self.iR_pix[intmat_mpdata][0][cnt_mpdata] = iR_pix12[intmat_mpdata][0][0]
        #     self.fR_pix[intmat_mpdata][0][cnt_mpdata] = fR_pix12[intmat_mpdata][0][0]
        #     self.best_match[intmat_mpdata][0][cnt_mpdata] = best_match12[intmat_mpdata][0][0]
        # self.update_plot()

    def save_btn(self,):
        curr_time = time.time()
        now = datetime.datetime.fromtimestamp(curr_time)
        c_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        
        save_directory_ = self.model_direc+"//results_"+self.material_+"_"+c_time
        if not os.path.exists(save_directory_):
            os.makedirs(save_directory_)
        
        np.savez_compressed(save_directory_+ "//results.npz", 
                            self.best_match, self.mat_global, self.rotation_matrix, self.strain_matrix, 
                            self.strain_matrixs,
                            self.col, self.colx, self.coly, self.match_rate, self.files_treated,
                            self.lim_x, self.lim_y, self.spots_len, self.iR_pix, self.fR_pix,
                            self.material_, self.material1_)
        
        ## intermediate saving of pickle objects with results
        with open(save_directory_+ "//results.pickle", "wb") as output_file:
                cPickle.dump([self.best_match, self.mat_global, self.rotation_matrix, self.strain_matrix, 
                              self.strain_matrixs,
                              self.col, self.colx, self.coly, self.match_rate, self.files_treated,
                              self.lim_x, self.lim_y, self.spots_len, self.iR_pix, self.fR_pix,
                              self.material_, self.material1_, self.lattice_, self.lattice1_,
                              self.symmetry, self.symmetry1, self.crystal, self.crystal1], output_file)     

        try:
            ## Write global text file with all results
            if self.material_ != self.material1_:
                text_file = open(save_directory_+"//prediction_stats_"+self.material_+"_"+self.material1_+".txt", "w")
            else:
                text_file = open(save_directory_+"//prediction_stats_"+self.material_+".txt", "w")
    
            filenames = list(np.unique(self.files_treated))
            filenames.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
            
            for i in range(self.lim_x*self.lim_y):
                text_file.write("# ********** \n")
                text_file.write("# Filename: "+ filenames[i] + "\n")
                for j in range(len(self.best_match)):
                    stats_ = self.best_match[j][0][i]
                    dev_eps_sample = self.strain_matrixs[j][0][i,:,:]
                    dev_eps = self.strain_matrix[j][0][i,:,:]
                    initial_residue = self.iR_pix[j][0][i][0]
                    final_residue = self.fR_pix[j][0][i][0]
                    mat = int(self.mat_global[j][0][i][0])
                    if mat == 0:
                        case = "None"
                    elif mat == 1:
                        case = self.material_
                    elif mat == 2:
                        case = self.material1_
                    
                    text_file.write("# ********** UB MATRIX "+str(j+1)+" \n")
                    text_file.write("Spot_index for 2 HKL are "+ str(stats_[0])+" ; "+ str(stats_[1])+ "\n")
                    text_file.write("HKL1 "+str(stats_[2])+"; HKL2 "+str(stats_[3])+"\n")
                    text_file.write("Coords of HKL1 "+str(stats_[4])+\
                                    "; coords of HKL2 "+str(stats_[5])+"\n")
                    text_file.write("Distance between 2 spots is "+ str(stats_[6])+ "\n")
                    text_file.write("Distance between 2 spots in LUT is "+ str(stats_[7])+ "\n")
                    text_file.write("Accuracy of NN for 2 HKL is "+ str(stats_[8])+\
                                    "% ; "+str(stats_[9])+ "% \n")
                    string1 = "Matched, Expected, Matching rate(%) : " + \
                                str(stats_[10]) +", "+str(stats_[11]) +", "+str(stats_[12])+" \n"
                    text_file.write(string1)
                    text_file.write("Rotation matrix for 2 HKL (multiplied by symmetry) is \n")
                    temp_ = stats_[14].flatten()
                    string1 = "[["+str(temp_[0])+","+str(temp_[1])+","+str(temp_[2])+"],"+  \
                                "["+str(temp_[3])+","+str(temp_[4])+","+str(temp_[5])+"],"+  \
                                    "["+str(temp_[6])+","+str(temp_[7])+","+str(temp_[8])+"]]"+ " \n"  
                    text_file.write(string1)
                    
                    text_file.write("dev_eps_sample is \n")
                    temp_ = dev_eps_sample.flatten()
                    string1 = "[["+str(temp_[0])+","+str(temp_[1])+","+str(temp_[2])+"],"+  \
                                "["+str(temp_[3])+","+str(temp_[4])+","+str(temp_[5])+"],"+  \
                                    "["+str(temp_[6])+","+str(temp_[7])+","+str(temp_[8])+"]]"+ " \n"  
                    text_file.write(string1)
    
                    text_file.write("dev_eps is \n")
                    temp_ = dev_eps.flatten()
                    string1 = "[["+str(temp_[0])+","+str(temp_[1])+","+str(temp_[2])+"],"+  \
                                "["+str(temp_[3])+","+str(temp_[4])+","+str(temp_[5])+"],"+  \
                                    "["+str(temp_[6])+","+str(temp_[7])+","+str(temp_[8])+"]]"+ " \n"  
                    text_file.write(string1)
    
                    text_file.write("Initial_pixel, Final_pixel residues are : "+str(initial_residue)+", "+str(final_residue)+" \n")
                    
                    text_file.write("Mat_id is "+str(mat)+"\n")
                    text_file.write("Material indexed is "+case+"\n")
                    text_file.write("\n")
            text_file.close()
            print("prediction statistics are generated") 
        except:
            # text_file.close()
            print("Errors with writing prediction output text file; could be the prediction was stopped midway")

        try:
            ## write MTEX file
            rotation_matrix = [[] for i in range(len(self.rotation_matrix))]
            for i in range(len(self.rotation_matrix)):
                rotation_matrix[i].append(np.zeros((self.lim_x*self.lim_y,3,3)))

            for i in range(len(self.rotation_matrix)):
                temp_mat = self.rotation_matrix[i][0]    
                for j in range(len(temp_mat)):
                    orientation_matrix = temp_mat[j,:,:]                    
                    ## rotate orientation by 40degrees to bring in Sample RF
                    omega = np.deg2rad(-40)
                    # # rotation de -omega autour de l'axe x (or Y?) pour repasser dans Rsample
                    cw = np.cos(omega)
                    sw = np.sin(omega)
                    mat_from_lab_to_sample_frame = np.array([[cw, 0.0, sw], [0.0, 1.0, 0.0], [-sw, 0, cw]]) #Y
                    # mat_from_lab_to_sample_frame = np.array([[1.0, 0.0, 0.0], [0.0, cw, -sw], [0.0, sw, cw]]) #X
                    # mat_from_lab_to_sample_frame = np.array([[cw, -sw, 0.0], [sw, cw, 0.0], [0.0, 0.0, 1.0]]) #Z
                    orientation_matrix = np.dot(mat_from_lab_to_sample_frame.T, orientation_matrix)

                    if np.linalg.det(orientation_matrix) < 0:
                        orientation_matrix = -orientation_matrix
                    rotation_matrix[i][0][j,:,:] = orientation_matrix
                              
            if self.material_ == self.material1_:
                lattice = self.lattice_
                material0_LG = material0_lauegroup
                header = [
                        "Channel Text File",
                        "Prj     lauetoolsnn",
                        "Author    [Ravi raj purohit]",
                        "JobMode    Grid",
                        "XCells    "+str(self.lim_x),
                        "YCells    "+str(self.lim_y),
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
                lattice = self.lattice_
                lattice1 = self.lattice1_
                material0_LG = material0_lauegroup
                material1_LG = material1_lauegroup
                header = [
                        "Channel Text File",
                        "Prj     lauetoolsnn",
                        "Author    [Ravi raj purohit]",
                        "JobMode    Grid",
                        "XCells    "+str(self.lim_x),
                        "YCells    "+str(self.lim_y),
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
            for index in range(len(self.rotation_matrix)):
                euler_angles = np.zeros((len(rotation_matrix[index][0]),3))
                phase_euler_angles = np.zeros(len(rotation_matrix[index][0]))
                for i in range(len(rotation_matrix[index][0])):
                    if np.all(rotation_matrix[index][0][i,:,:] == 0):
                        continue
                    euler_angles[i,:] = OrientationMatrix2Euler(rotation_matrix[index][0][i,:,:])
                    phase_euler_angles[i] = self.mat_global[index][0][i]        
                
                euler_angles = euler_angles.reshape((self.lim_x,self.lim_y,3))
                phase_euler_angles = phase_euler_angles.reshape((self.lim_x,self.lim_y,1))
                
                a = euler_angles
                if self.material_ != self.material1_:
                    filename125 = save_directory_+ "//"+self.material_+"_"+self.material1_+"_MTEX_UBmat_"+str(index)+"_LT.ctf"
                else:
                    filename125 = save_directory_+ "//"+self.material_+"_MTEX_UBmat_"+str(index)+"_LT.ctf"
                    
                f = open(filename125, "w")
                for ij in range(len(header)):
                    f.write(header[ij]+" \n")
                        
                for i123 in range(euler_angles.shape[1]):
                    y_step = 1 * i123
                    for j123 in range(euler_angles.shape[0]):
                        x_step = 1 * j123
                        phase_id = int(phase_euler_angles[j123,i123,0])
                        eul =  str(phase_id)+'\t' + "%0.4f" % x_step +'\t'+"%0.4f" % y_step+'\t8\t0\t'+ \
                                            "%0.4f" % a[j123,i123,0]+'\t'+"%0.4f" % a[j123,i123,1]+ \
                                                '\t'+"%0.4f" % a[j123,i123,2]+'\t0.0001\t180\t0\n'
                        string = eul
                        f.write(string)
                f.close()
        except:
            print("Error writing the MTEX file, could be the prediction data is not completed and save function was called")

        #%  Plot some data  
        try:
            global_plots(self.lim_x, self.lim_y, self.rotation_matrix, self.strain_matrix, self.strain_matrixs, 
                         self.col, self.colx, self.coly, self.match_rate, self.mat_global, self.spots_len, 
                         self.iR_pix, self.fR_pix, save_directory_, self.material_, self.material1_,
                         match_rate_threshold=5, bins=30)
        except:
            print("Error in the global plots module")

        # try:
        #     save_sst(self.lim_x, self.lim_y, self.strain_matrix, self.strain_matrixs, self.col, 
        #             self.colx, self.coly, self.match_rate, self.mat_global, self.spots_len, 
        #             self.iR_pix, self.fR_pix, save_directory_, self.material_, self.material1_,
        #             self.lattice_, self.lattice1_, self.symmetry, self.symmetry1, self.crystal, self.crystal1,
        #             self.rotation_matrix, self.symmetry_name, self.symmetry1_name,
        #                   mac_axis = [0., 0., 1.], axis_text="Z", match_rate_threshold=5)
        # except:
        #     print("Error in the SST plots module")

        ## HKL selective plots (in development)
        hkls_list = ast.literal_eval(self.hkl_plot.text())
        if self.ipf_axis.currentText() == "Z":
            mac_axis = [0., 0., 1.]
        elif self.ipf_axis.currentText() == "Y":
            mac_axis = [0., 1., 0.]
        elif self.ipf_axis.currentText() == "X":
            mac_axis = [1., 0., 0.]
        print(mac_axis, hkls_list)
        # save_hkl_stats(self.lim_x, self.lim_y, self.strain_matrix, self.strain_matrixs, self.col, 
        #               self.colx, self.coly, self.match_rate, self.mat_global, self.spots_len, 
        #               self.iR_pix, self.fR_pix, save_directory_, self.material_, self.material1_,
        #               self.lattice_, self.lattice1_, self.symmetry, self.symmetry1, self.rotation_matrix, 
        #              hkls_list=hkls_list, angle=10., mac_axis = mac_axis, axis_text = self.ipf_axis.currentText())

    def plot_pc(self):
        ## update matrix plot box?
        if self.matrix_plot.count() < int(self.ubmat.text()):
            for intmat in range(int(self.ubmat.text())):
                if intmat == 0 or intmat < self.matrix_plot.count():
                    continue
                self.matrix_plot.addItem(str(intmat+1))
            self.modify_array()
            
        self.btn_config.setEnabled(False)
        self.model_direc = self.modelDirec

        ## load model related files and generate the model
        if self.material_ != self.material1_:
            json_file = open(self.model_direc+"//model_"+self.material_+"_"+self.material1_+".json", 'r')
        else:
            json_file = open(self.model_direc+"//model_"+self.material_+".json", 'r')
                
        self.classhkl = np.load(self.model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_0"]
        self.angbins = np.load(self.model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_1"]
        
        if self.material_ != self.material1_:
            self.ind_mat = np.load(self.model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_5"]
            self.ind_mat1 = np.load(self.model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_6"]
        else: 
            self.ind_mat = None
            self.ind_mat1 = None  
        
        load_weights = self.filenameModel[0]
        self.wb = read_hdf5(load_weights)
        self.temp_key = list(self.wb.keys())
        
        # # load json and create model
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        load_weights = self.filenameModel[0]
        self.model.load_weights(load_weights)
        
        if self.file_state==0:
            ct = time.time()
            now = datetime.datetime.fromtimestamp(ct)
            self.c_time = now.strftime("%Y-%m-%d_%H-%M-%S")
            self.file_state = 1
        
        self.initialize_plot()

        self.mode_spotCycle = self.analysis_plot_tech.currentText()
        
        if self.matrix_plot_tech.currentText() == "MultiProcessing":
            self.ncpu = cpu_count_user
            self._inputs_queue = Queue()
            self._outputs_queue = Queue()
            run_flag = multip.Value('I', True)
            self._worker_processes = {}
            for i in range(self.ncpu):
                self._worker_processes[i]= Process(target=worker, args=(self._inputs_queue, self._outputs_queue, i+1, run_flag))#, mp_rotation_matrix))
            for i in range(self.ncpu):
                self._worker_processes[i].start()
            ### Update data from multiprocessing
            self.timermp1212.setInterval(500) ## check every second (update the list of files in folder)
            self.timermp1212.timeout.connect(self.update_data_mp1212)
            self.timermp1212.start()
    
        self.out_name = None
        self.run = True
        self.temp_ = threading.Thread(target=self.plot_pcv1, daemon=False)
        self.temp_.start()
        self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(False)
    
    def update_plot(self):
        index_plotfnc = int(self.matrix_plot.currentText())-1

        if self.ipf_axis.currentText() == "Z":
            col_plot_plotfnc = self.col[index_plotfnc][0]
        elif self.ipf_axis.currentText() == "Y":
            col_plot_plotfnc = self.coly[index_plotfnc][0]
        elif self.ipf_axis.currentText() == "X":
            col_plot_plotfnc = self.colx[index_plotfnc][0]
        
        col_plot_plotfnc = col_plot_plotfnc.reshape((self.lim_x, self.lim_y, 3))
        mr_plot_plotfnc = self.match_rate[index_plotfnc][0]
        mr_plot_plotfnc = mr_plot_plotfnc.reshape((self.lim_x, self.lim_y))        
        mat_glob_plotfnc = self.mat_global[index_plotfnc][0]
        mat_glob_plotfnc = mat_glob_plotfnc.reshape((self.lim_x, self.lim_y))
        
        self.im_axes.set_data(col_plot_plotfnc)
        self.im_axes1.set_data(mr_plot_plotfnc)
        if self.im_axes3 != None:
            self.im_axes3.set_data(mat_glob_plotfnc)
            
        strain_index_plotfnc = self.strain_plot.currentText()
        if "sample" in strain_index_plotfnc:
            strain_matrix_plot_plotfnc = self.strain_matrixs[index_plotfnc][0]
        elif "crystal" in strain_index_plotfnc:
            strain_matrix_plot_plotfnc = self.strain_matrix[index_plotfnc][0]
        if "11" in strain_index_plotfnc:
            strain_matrix_plot_plotfnc = strain_matrix_plot_plotfnc[:,0,0]
        elif "22" in strain_index_plotfnc:
            strain_matrix_plot_plotfnc = strain_matrix_plot_plotfnc[:,1,1]
        elif "33" in strain_index_plotfnc:
            strain_matrix_plot_plotfnc = strain_matrix_plot_plotfnc[:,2,2]
        elif "12" in strain_index_plotfnc:
            strain_matrix_plot_plotfnc = strain_matrix_plot_plotfnc[:,0,1]
        elif "13" in strain_index_plotfnc:
            strain_matrix_plot_plotfnc = strain_matrix_plot_plotfnc[:,0,2]
        elif "23" in strain_index_plotfnc:
            strain_matrix_plot_plotfnc = strain_matrix_plot_plotfnc[:,1,2]
        strain_tensor_plot_plotfnc = strain_matrix_plot_plotfnc.reshape((self.lim_x, self.lim_y))
        self.im_axes2.set_data(strain_tensor_plot_plotfnc)
        self.canvas.draw()
        
    def initialize_plot(self):
        ## get color matrix to plot
        index_plotfnc = int(self.matrix_plot.currentText())-1
        strain_index_plotfnc = self.strain_plot.currentText()

        if "sample" in strain_index_plotfnc:
            strain_matrix_plot_plotfnc = self.strain_matrixs[index_plotfnc][0]
        elif "crystal" in strain_index_plotfnc:
            strain_matrix_plot_plotfnc = self.strain_matrix[index_plotfnc][0]
        
        if "11" in strain_index_plotfnc:
            strain_matrix_plot_plotfnc = strain_matrix_plot_plotfnc[:,0,0]
        elif "22" in strain_index_plotfnc:
            strain_matrix_plot_plotfnc = strain_matrix_plot_plotfnc[:,1,1]
        elif "33" in strain_index_plotfnc:
            strain_matrix_plot_plotfnc = strain_matrix_plot_plotfnc[:,2,2]
        elif "12" in strain_index_plotfnc:
            strain_matrix_plot_plotfnc = strain_matrix_plot_plotfnc[:,0,1]
        elif "13" in strain_index_plotfnc:
            strain_matrix_plot_plotfnc = strain_matrix_plot_plotfnc[:,0,2]
        elif "23" in strain_index_plotfnc:
            strain_matrix_plot_plotfnc = strain_matrix_plot_plotfnc[:,1,2]
        
        try:
            strain_tensor_plot_plotfnc = strain_matrix_plot_plotfnc.reshape((self.lim_x, self.lim_y))
        except:
            print("Reshape error, verify the grid xlim and ylim and change them")
            return
        
        if self.ipf_axis.currentText() == "Z":
            col_plot_plotfnc = self.col[index_plotfnc][0]
        elif self.ipf_axis.currentText() == "Y":
            col_plot_plotfnc = self.coly[index_plotfnc][0]
        elif self.ipf_axis.currentText() == "X":
            col_plot_plotfnc = self.colx[index_plotfnc][0]
        
        col_plot_plotfnc = col_plot_plotfnc.reshape((self.lim_x, self.lim_y, 3))
        mr_plot_plotfnc = self.match_rate[index_plotfnc][0]
        mr_plot_plotfnc = mr_plot_plotfnc.reshape((self.lim_x, self.lim_y))        
        mat_glob_plotfnc = self.mat_global[index_plotfnc][0]
        mat_glob_plotfnc = mat_glob_plotfnc.reshape((self.lim_x, self.lim_y))
        
        # Drop off the first y element, append a new one.
        self.canvas.axes.cla()
        self.canvas.axes.set_title("IPF map (rectangle selector)", loc='center', fontsize=10)
        self.im_axes = self.canvas.axes.imshow(col_plot_plotfnc, origin='lower')
        self.canvas.axes1.cla()
        self.canvas.axes1.set_title("Matching rate (line selector)", loc='center', fontsize=10) 
        self.im_axes1 = self.canvas.axes1.imshow(mr_plot_plotfnc, origin='lower', cmap="jet", vmin=0, vmax=100)
        self.canvas.axes2.cla()
        self.canvas.axes2.set_title("Deviatoric strain", loc='center', fontsize=10) 
        self.im_axes2 = self.canvas.axes2.imshow(strain_tensor_plot_plotfnc, origin='lower', cmap="jet", vmin=-1, vmax=1)
        
        if self.material_ != self.material1_:
            self.canvas.axes3.cla()
            self.canvas.axes3.set_title("Material Index (1: "+self.material_+"; 2: "+self.material1_+")", loc='center', fontsize=10) 
            self.im_axes3 = self.canvas.axes3.imshow(mat_glob_plotfnc, origin='lower', vmin=0, vmax=2)
        else:
            self.im_axes3 = None
            
        toggle_selector.RS = RectangleSelector(self.canvas.axes, self.box_select_callback,
                                               drawtype='box', useblit=True,
                                               button=[1],  # don't use middle/right button
                                               minspanx=5, minspany=5,
                                               spancoords='pixels',
                                               interactive=True)
        
        toggle_selector1.RS = RectangleSelector(self.canvas.axes1, self.line_select_callback,
                                               drawtype='line', useblit=True,
                                               button=[1],  # don't use middle/right button
                                               minspanx=5, minspany=5,
                                               spancoords='pixels',
                                               interactive=True)
        # Trigger the canvas to update and redraw.
        self.canvas.draw()
    
    def line_select_callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        if eclick.button == 1 and erelease.button == 1:
            x1, y1 = int(np.round(eclick.xdata)), int(np.round(eclick.ydata))
            x2, y2 = int(np.round(erelease.xdata)), int(np.round(erelease.ydata))
            print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
            # print(" The button you used were: %s %s" % (eclick.button, erelease.button))
            try:
                index_plotfnc = int(self.matrix_plot.currentText())-1
                title_plotfnc = "Deviatoric strain (crystal frame)"
                strain_matrix_plot_plotfnc = self.strain_matrix[index_plotfnc][0]
                try:
                    strain_tensor_plot_plotfnc = strain_matrix_plot_plotfnc.reshape((self.lim_x, self.lim_y,3,3))
                    num = int(np.hypot(x2-x1, y2-y1)) #np.max((abs(x2-x1),abs(y2-y1)))
                    x, y = np.linspace(x1, x2, num), np.linspace(y1, y2, num)
                    # Extract the values along the line
                    strain_tensor_cropped = strain_tensor_plot_plotfnc[y.astype(int), x.astype(int),:,:]
                except:
                    print("Reshape error, verify the grid xlim and ylim and change them")
                    return
            except:
                print("No stats could be generated for the selected range of pixels")
                return
            w = MyPopup_image_v2(strain_tensor_cropped, title_plotfnc, flag=1)
            w.show()       
            self.popups.append(w)
            
    def box_select_callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        if eclick.button == 1 and erelease.button == 1:
            x1, y1 = int(np.round(eclick.xdata)), int(np.round(eclick.ydata))
            x2, y2 = int(np.round(erelease.xdata)), int(np.round(erelease.ydata))
            print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
            # print(" The button you used were: %s %s" % (eclick.button, erelease.button))
            try:
                index_plotfnc = int(self.matrix_plot.currentText())-1
                title_plotfnc = "Deviatoric strain (crystal frame)"
                strain_matrix_plot_plotfnc = self.strain_matrix[index_plotfnc][0]
                try:
                    strain_tensor_plot_plotfnc = strain_matrix_plot_plotfnc.reshape((self.lim_x, self.lim_y,3,3))
                except:
                    print("Reshape error, verify the grid xlim and ylim and change them")
                    return
            except:
                print("No stats could be generated for the selected range of pixels")
                return
            ## crop the strain array with the coordinates of the rectangle
            strain_tensor_cropped = strain_tensor_plot_plotfnc[y1:y2,x1:x2,:,:]
            w = MyPopup_image_v2(strain_tensor_cropped, title_plotfnc)
            w.show()       
            self.popups.append(w)
                
    def onclickImage(self, event123):
        if event123.dblclick or event123.button == 2:
            ix, iy = event123.xdata, event123.ydata
            try:
                ## read the saved COR file and extract exp spots info.## avoid zero index problem
                ix = int(round(ix))
                iy = int(round(iy))
                try:
                    if iy == 0 and ix == 0:
                        image_no = 0
                    elif iy == 0 and ix != 0:
                        image_no = ix
                    elif iy != 0 and ix == 0:
                        image_no = iy * self.lim_y
                    elif iy != 0 and ix != 0:
                        image_no = iy * self.lim_y + ix
                    
                    ccd_label = self.ccd_label.currentText()
                    path = os.path.normpath(self.filenm[image_no].decode())                    
                    Data, framedim, fliprot = IOimage.readCCDimage(path,
                                                                    stackimageindex=-1,
                                                                    CCDLabel=ccd_label,
                                                                    dirname=None,
                                                                    verbose=0)   
                except:
                    print(path)
                    print('chosen pixel coords are x = %d, y = %d'%(ix, iy))
                    print("No IMAGE file could be found for the selected pixel")
                    return
                w = MyPopup_image_v1(ix, iy, path, Data, ccd_label, 
                                     self.predict_single_file_nodialog, image_no,
                                     self.detectorparameters)
                w.show()       
                self.popups.append(w)
                print('chosen pixel coords are x = %d, y = %d'%(ix, iy))
            except:
                return
        elif event123.button == 3:
            ix, iy = event123.xdata, event123.ydata
            try:
                ## read the saved COR file and extract exp spots info.## avoid zero index problem
                ix = int(round(ix))
                iy = int(round(iy))
                try:
                    if iy == 0 and ix == 0:
                        image_no = 0
                    elif iy == 0 and ix != 0:
                        image_no = ix
                    elif iy != 0 and ix == 0:
                        image_no = iy * self.lim_y
                    elif iy != 0 and ix != 0:
                        image_no = iy * self.lim_y + ix
                        
                    # image_no = int(ix*iy+(iy-1)-1)
                    index_plotfnc = int(self.matrix_plot.currentText())-1
                    rotation_matrix = self.rotation_matrix[index_plotfnc][0][image_no,:,:]
                    mat_glob_plotfnc = self.mat_global[index_plotfnc][0][image_no]
                    path = os.path.normpath(self.filenm[image_no].decode())
                    files = self.cor_file_directory+"//"+path.split(os.sep)[-1].split(".")[0]+".cor"        
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
                    dict_dp['detectordiameter']=pixelsize*framedim[0]#TODO*2
                    dict_dp['pixelsize']=pixelsize
                    dict_dp['dim']=framedim
                    dict_dp['peakX']=peakx
                    dict_dp['peakY']=peaky
                    dict_dp['intensity']=intensity
                except:
                    print(self.cor_file_directory+"//"+path.split(os.sep)[-1].split(".")[0]+".cor")
                    print('chosen pixel coords are x = %d, y = %d'%(ix, iy))
                    print("No COR file could be found for the selected pixel")
                    return
                        
                if mat_glob_plotfnc == 1:
                    material_=self.material_
                    tolerance_add = float(self.tolerance.text())                
                elif mat_glob_plotfnc == 2:
                    material_=self.material1_
                    tolerance_add = float(self.tolerance1.text())
                else:
                    print("No Material is indexed for this pixel")
                    material_ = None
                    tolerance_add = None
                    sim_twotheta = []
                    sim_chi = []
                    list_spots = []
                    residues = []
                    theo_index = []
                    sim_energy = []
                    sim_hkl = []
                
                if np.all(rotation_matrix==0):
                    material_ = None
                    sim_twotheta = []
                    sim_chi = []
                    list_spots = []
                    residues = []
                    theo_index = []
                    sim_energy = []
                    sim_hkl = []
                    print("No rotation matrix found")
                
                if material_ != None:
                    sim_twotheta, sim_chi, sim_energy, sim_hkl, \
                    list_spots, residues, theo_index = simulate_spots(rotation_matrix, 
                                                                        material_, self.emax, self.emin, 
                                                                        dict_dp['detectorparameters'], dict_dp,
                                                                        tolerance_add, data_theta*2.0,
                                                                        data_chi)
                    if len(sim_twotheta) == 0:
                        sim_twotheta = []
                        sim_chi = []
                        list_spots = []
                        residues = []
                        theo_index = []
                        sim_energy = []
                        sim_hkl = []
                        print("Nothing simulated")
                
                w = MyPopup_image(data_theta, data_chi, intensity, sim_twotheta, sim_chi, sim_energy,
                                  sim_hkl, ix, iy, files,
                                  list_spots, residues, theo_index, rotation_matrix)
                w.show()       
                self.popups.append(w)
                print('chosen pixel coords are x = %d, y = %d'%(ix, iy))
            except:
                print("Error occured")
                return
        else:
            print("Right Left for plotting the Indexation results; Left double click (or middle mouse) for Raw Laue patter and left click drag for lasso")

    def update_data_mp1212(self):
        if not self._outputs_queue.empty():            
            self.timermp1212.blockSignals(True)         
            n_range = self._outputs_queue.qsize()
            for _ in range(n_range):
                r_message_mpdata = self._outputs_queue.get()
                strain_matrix_mpdata, strain_matrixs_mpdata, rotation_matrix_mpdata, col_mpdata, \
                                 colx_mpdata, coly_mpdata, match_rate_mpdata, mat_global_mpdata, \
                                     cnt_mpdata, meta_mpdata, files_treated_mpdata, spots_len_mpdata, \
                                         iR_pixel_mpdata, fR_pixel_mpdata, best_match_mpdata, check_mpdata = r_message_mpdata
    
                for i_mpdata in files_treated_mpdata:
                    self.files_treated.append(i_mpdata)
                                
                for intmat_mpdata in range(int(self.ubmat.text())):
                    self.check[cnt_mpdata,intmat_mpdata] = check_mpdata[cnt_mpdata,intmat_mpdata]
                    self.mat_global[intmat_mpdata][0][cnt_mpdata] = mat_global_mpdata[intmat_mpdata][0][cnt_mpdata]
                    self.strain_matrix[intmat_mpdata][0][cnt_mpdata,:,:] = strain_matrix_mpdata[intmat_mpdata][0][cnt_mpdata,:,:]
                    self.strain_matrixs[intmat_mpdata][0][cnt_mpdata,:,:] = strain_matrixs_mpdata[intmat_mpdata][0][cnt_mpdata,:,:]
                    self.rotation_matrix[intmat_mpdata][0][cnt_mpdata,:,:] = rotation_matrix_mpdata[intmat_mpdata][0][cnt_mpdata,:,:]
                    self.col[intmat_mpdata][0][cnt_mpdata,:] = col_mpdata[intmat_mpdata][0][cnt_mpdata,:]
                    self.colx[intmat_mpdata][0][cnt_mpdata,:] = colx_mpdata[intmat_mpdata][0][cnt_mpdata,:]
                    self.coly[intmat_mpdata][0][cnt_mpdata,:] = coly_mpdata[intmat_mpdata][0][cnt_mpdata,:]
                    self.match_rate[intmat_mpdata][0][cnt_mpdata] = match_rate_mpdata[intmat_mpdata][0][cnt_mpdata]
                    self.spots_len[intmat_mpdata][0][cnt_mpdata] = spots_len_mpdata[intmat_mpdata][0][cnt_mpdata]
                    self.iR_pix[intmat_mpdata][0][cnt_mpdata] = iR_pixel_mpdata[intmat_mpdata][0][cnt_mpdata]
                    self.fR_pix[intmat_mpdata][0][cnt_mpdata] = fR_pixel_mpdata[intmat_mpdata][0][cnt_mpdata]
                    self.best_match[intmat_mpdata][0][cnt_mpdata] = best_match_mpdata[intmat_mpdata][0][cnt_mpdata] 
                    
                if self.use_previous_UBmatrix_name:
                    try:
                    #Perhaps save only the best matching rate UB matricies in the file, instead of all UB matricies
                    #Or select only the best UB matricies when opening the file in propose_UBmatrix function
                        ## calculate average matching rate and save it
                        avg_match_rate1 = [[] for i in range(int(self.ubmat.text()))]
                        for intmat_mpdata in range(int(self.ubmat.text())):
                            avg_match_rate = []
                            for j in self.match_rate[intmat_mpdata][0][:]:
                                if j != 0:
                                    avg_match_rate.append(j)
                            avg_match_rate1[intmat_mpdata].append(np.median(avg_match_rate))
                        np.savez_compressed(self.model_direc+'//rotation_matrix_indexed_1.npz', 
                                            self.rotation_matrix, self.mat_global, 
                                            self.match_rate, avg_match_rate1)
                    except:
                        print("Warning : Error saving the NPZ file; nothing to worry")
            ## update plot now
            self.update_plot()
            self.timermp1212.blockSignals(False)
    
    def plot_pcv1(self):
        if self.use_previous_UBmatrix_name:
            np.savez_compressed(self.model_direc+'//rotation_matrix_indexed_1.npz', self.rotation_matrix, self.mat_global, self.match_rate, 0.0)
        
        cond = self.strain_plot_tech.currentText()
        self.strain_calculation = False
        if cond == "YES":
            self.strain_calculation = True
            
        cond_mode = self.matrix_plot_tech.currentText()
        # =============================================================================
        #         ## Multi-processing routine
        # =============================================================================
        ## Number of files to generate
        grid_files = np.zeros((self.lim_x,self.lim_y))
        self.filenm = np.chararray((self.lim_x,self.lim_y), itemsize=1000)
        grid_files = grid_files.ravel()
        self.filenm = self.filenm.ravel()
        count_global = self.lim_x * self.lim_y

        if self.ccd_label.currentText() == "Cor" or self.ccd_label.currentText() == "cor":
            format_file = "cor"
        else:
            format_file = dictLT.dict_CCD[self.ccd_label.currentText()][7]

        list_of_files = glob.glob(self.filenameDirec+'//'+self.experimental_prefix.text()+'*.'+format_file)
        ## sort files
        ## TypeError: '<' not supported between instances of 'str' and 'int'
        list_of_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

        if len(list_of_files) == count_global:
            for ii in range(len(list_of_files)):
                grid_files[ii] = ii
                self.filenm[ii] = list_of_files[ii]               
        else:
            print("expected "+str(count_global)+" files based on the XY grid ("+str(self.lim_x)+","+str(self.lim_y)+") defined by user")
            print("But found "+str(len(list_of_files))+" files (either all data is not written yet or maybe XY grid definition is not proper)")
            digits = len(str(count_global))
            digits = max(digits,4)

            for ii in range(count_global):
                text = str(ii)
                if ii < 10000:
                    string = text.zfill(4)
                else:
                    string = text.zfill(5)
                file_name_temp = self.filenameDirec+'//'+self.experimental_prefix.text()+string+'.'+format_file
                ## store it in a grid 
                self.filenm[ii] = file_name_temp
            ## access grid files to process with multi-thread
        # self.check = np.zeros((count_global,int(self.ubmat.text())))
        # =============================================================================
        try:
            blacklist = self.blacklist_file[0]
        except:
            blacklist = None
        
        ### Create a COR directory to be loaded in LaueTools
        self.cor_file_directory = self.filenameDirec + "//" + self.experimental_prefix.text()+"CORfiles"
        if format_file in ['cor',"COR","Cor"]:
            self.cor_file_directory = self.filenameDirec
        if not os.path.exists(self.cor_file_directory):
            os.makedirs(self.cor_file_directory)
        
        # while True:
        if cond_mode == "Sequential":
            self.predict_preprocess(cnt=self.cnt, 
                                      rotation_matrix=self.rotation_matrix,
                                      strain_matrix=self.strain_matrix,
                                      strain_matrixs=self.strain_matrixs,
                                      col=self.col,
                                      colx=self.colx,
                                      coly=self.coly,
                                      match_rate=self.match_rate,
                                      spots_len=self.spots_len, 
                                      iR_pix=self.iR_pix, 
                                      fR_pix=self.fR_pix,
                                      best_match = self.best_match,
                                      mat_global=self.mat_global,
                                      check=self.check,
                                      detectorparameters=self.detectorparameters,
                                      pixelsize=self.pixelsize,
                                      angbins=self.angbins,
                                      classhkl=self.classhkl,
                                      hkl_all_class0=self.hkl_all_class0,
                                      hkl_all_class1=self.hkl_all_class1,
                                      emin=self.emin,
                                      emax=self.emax,
                                      material_=self.material_,
                                      material1_=self.material1_,
                                      symmetry=self.symmetry,
                                      symmetry1=self.symmetry1,   
                                      lim_x= self.lim_x,
                                      lim_y=self.lim_y,
                                      strain_calculation=self.strain_calculation, 
                                      ind_mat=self.ind_mat, ind_mat1=self.ind_mat1,
                                      model_direc=self.model_direc, tolerance=float(self.tolerance.text()),
                                      tolerance1=float(self.tolerance1.text()),
                                      matricies=int(self.ubmat.text()), ccd_label=self.ccd_label.currentText(), 
                                      filename_bkg=None, #self.filenamebkg,
                                      intensity_threshold=float(self.intensity_threshold.text()),
                                      boxsize=int(self.boxsize.text()),bkg_treatment=self.bkg_treatment.text(),
                                      filenameDirec=self.filenameDirec, 
                                      experimental_prefix=self.experimental_prefix.text(),
                                      blacklist_file =blacklist,
                                      text_file=None,
                                      files_treated=self.files_treated,
                                      try_previous1=True,
                                      wb = self.wb,
                                      temp_key = self.temp_key,
                                      cor_file_directory=self.cor_file_directory,
                                      mode_spotCycle1 = self.mode_spotCycle,
                                      softmax_threshold_global123 = self.softmax_threshold_global,
                                      mr_threshold_global123=self.mr_threshold_global,
                                      cap_matchrate123=self.cap_matchrate,
                                      tolerance_strain123=self.tolerance_strain,
                                      tolerance_strain1231=self.tolerance_strain1,
                                      NumberMaxofFits123=self.NumberMaxofFits,
                                      fit_peaks_gaussian_global123=self.fit_peaks_gaussian_global,
                                      FitPixelDev_global123=self.FitPixelDev_global,
                                      coeff123 = self.coeff,
                                      coeff_overlap=self.coeff_overlap,
                                      material0_limit=self.material0_limit,
                                      material1_limit=self.material1_limit,
                                      use_previous_UBmatrix_name=self.use_previous_UBmatrix_name,
                                      material_phase_always_present = self.material_phase_always_present,
                                      crystal=self.crystal,
                                      crystal1=self.crystal1,
                                      strain_free_parameters=self.strain_free_parameters,
                                      mode_peaksearch=self.mode_peaksearch.text())
            
        elif cond_mode == "MultiProcessing":
            try_prevs = False
            if self.mode_spotCycle == "beamtime":
                try_prevs = True
            
            valu12 = [[ self.filenm[ii].decode(), ii,
                        self.rotation_matrix,
                        self.strain_matrix,
                        self.strain_matrixs,
                        self.col,
                        self.colx,
                        self.coly,
                        self.match_rate,
                        self.spots_len, 
                        self.iR_pix, 
                        self.fR_pix,
                        self.best_match,
                        self.mat_global,
                        self.check,
                        self.detectorparameters,
                        self.pixelsize,
                        self.angbins,
                        self.classhkl,
                        self.hkl_all_class0,
                        self.hkl_all_class1,
                        self.emin,
                        self.emax,
                        self.material_,
                        self.material1_,
                        self.symmetry,
                        self.symmetry1,   
                        self.lim_x,
                        self.lim_y,
                        self.strain_calculation, 
                        self.ind_mat, self.ind_mat1,
                        self.model_direc, float(self.tolerance.text()),
                        float(self.tolerance1.text()),
                        int(self.ubmat.text()), self.ccd_label.currentText(), 
                        None,
                        float(self.intensity_threshold.text()),
                        int(self.boxsize.text()),self.bkg_treatment.text(),
                        self.filenameDirec, 
                        self.experimental_prefix.text(),
                        blacklist,
                        None,
                        self.files_treated,
                        try_prevs, ## try previous is kept true, incase if its stuck in loop
                        self.wb,
                        self.temp_key,
                        self.cor_file_directory,
                        self.mode_spotCycle,
                        self.softmax_threshold_global,
                        self.mr_threshold_global,
                        self.cap_matchrate,
                        self.tolerance_strain,
                        self.tolerance_strain1,
                        self.NumberMaxofFits,
                        self.fit_peaks_gaussian_global,
                        self.FitPixelDev_global,
                        self.coeff,
                        self.coeff_overlap,
                        self.material0_limit,
                        self.material1_limit,
                        self.use_previous_UBmatrix_name,
                        self.material_phase_always_present,
                        self.crystal,
                        self.crystal1,
                        self.strain_free_parameters,
                        self.mode_peaksearch.text()] for ii in range(count_global)]
            
            chunks = chunker_list(valu12, self.ncpu)
            chunks_mp = list(chunks)

            meta = {'t1':time.time()}
            for ijk in range(int(self.ncpu)):
                self._inputs_queue.put((chunks_mp[ijk], self.ncpu, meta))
                
        if cond_mode == "MultiProcessing":
            print("Launched all processes")

    def plot_btn_stop(self):
        if self.matrix_plot_tech.currentText() == "MultiProcessing":
            self.timermp1212.blockSignals(False)
            run_flag = multip.Value('I', False)
            while not self._outputs_queue.empty():            
                n_range = self._outputs_queue.qsize()
                for _ in range(n_range):
                    continue  
            print("Flag for mp module: ",run_flag)
            time.sleep(0.1)
            self.timermp1212.stop()
        self.cnt = 1
        self.run = False
        self.btn_config.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(True)
        
    def getfiles(self):
        self.modelDirec = QFileDialog.getExistingDirectory(self, 'Select Folder in which model files are located')
    
    def getfiles1(self):
        self.filenameDirec = QFileDialog.getExistingDirectory(self, 'Select Folder in which Experimental data is or will be stored')
    
    def getfileModel(self):
        self.filenameModel = QFileDialog.getOpenFileName(self, 'Select the model weights H5 or HDF5 file')
    
    def predict_preprocess(self,cnt,rotation_matrix,strain_matrix,strain_matrixs,
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
                           blacklist_file =None, text_file=None, files_treated=None,try_previous1=False,
                           wb=None, temp_key=None, cor_file_directory=None, mode_spotCycle1=None,
                           softmax_threshold_global123=None,mr_threshold_global123=None,cap_matchrate123=None,
                           tolerance_strain123=None,tolerance_strain1231=None,NumberMaxofFits123=None,fit_peaks_gaussian_global123=None,
                           FitPixelDev_global123=None, coeff123=None,coeff_overlap=None,
                           material0_limit=None, material1_limit=None, use_previous_UBmatrix_name=None,
                           material_phase_always_present=None, crystal=None, crystal1=None, strain_free_parameters=None,
                           mode_peaksearch=None):
        
        
        if ccd_label in ["Cor", "cor"]:
            format_file = "cor"
        else:
            format_file = dictLT.dict_CCD[ccd_label][7]

        list_of_files = glob.glob(filenameDirec+'//'+experimental_prefix+'*.'+format_file)
        ## sort files
        ## TypeError: '<' not supported between instances of 'str' and 'int'
        list_of_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
        
        for files in list_of_files:
            print("# Predicting for "+ files)
            self.update_plot()
            call_global()
            peak_detection_error = False
            if self.run == False:
                print("Analysis stopped")
                break

            if files in files_treated:
                continue
            
            files_treated.append(files)
                        
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
                    #print(CCDLabel)
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
                                                mode = mode_peaksearch
                                                )
                    peak_XY = peak_XY[0]#[:,:2] ##[2] Integer peak lists
                except:
                    print("Error in Peak detection for "+ files)
                    for intmat in range(matricies):
                        rotation_matrix[intmat][0][self.cnt,:,:] = np.zeros((3,3))
                        strain_matrix[intmat][0][self.cnt,:,:] = np.zeros((3,3))
                        strain_matrixs[intmat][0][self.cnt,:,:] = np.zeros((3,3))
                        col[intmat][0][self.cnt,:] = 0,0,0
                        colx[intmat][0][self.cnt,:] = 0,0,0
                        coly[intmat][0][self.cnt,:] = 0,0,0
                        match_rate[intmat][0][self.cnt] = 0
                        mat_global[intmat][0][self.cnt] = 0
                    
                    cnt += 1
                    self.cnt += 1
                    peak_detection_error = True
                    continue
                
                try:
                    s_ix = np.argsort(peak_XY[:, 2])[::-1]
                    peak_XY = peak_XY[s_ix]
                except:
                    print("No peaks found for "+ files)
                    for intmat in range(matricies):
                        rotation_matrix[intmat][0][self.cnt,:,:] = np.zeros((3,3))
                        strain_matrix[intmat][0][self.cnt,:,:] = np.zeros((3,3))
                        strain_matrixs[intmat][0][self.cnt,:,:] = np.zeros((3,3))
                        col[intmat][0][self.cnt,:] = 0,0,0
                        colx[intmat][0][self.cnt,:] = 0,0,0
                        coly[intmat][0][self.cnt,:] = 0,0,0
                        match_rate[intmat][0][self.cnt] = 0
                        mat_global[intmat][0][self.cnt] = 0
                    
                    cnt += 1
                    self.cnt += 1
                    peak_detection_error = True
                    continue
                
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
                #print(framedim)
                #print(pixelsize)
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
                # print('detectorparameters from file are: '+ str(detectorparameters))
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
            
            if peak_detection_error:
                continue
            
            sorted_data = np.transpose(np.array([data_theta, data_chi]))
            tabledistancerandom = np.transpose(GT.calculdist_from_thetachi(sorted_data, sorted_data))

            codebars_all = []
            
            if len(data_theta) == 0:
                print("No peaks Found for : " + files)
                for intmat in range(matricies):
                    rotation_matrix[intmat][0][self.cnt,:,:] = np.zeros((3,3))
                    strain_matrix[intmat][0][self.cnt,:,:] = np.zeros((3,3))
                    strain_matrixs[intmat][0][self.cnt,:,:] = np.zeros((3,3))
                    col[intmat][0][self.cnt,:] = 0,0,0
                    colx[intmat][0][self.cnt,:] = 0,0,0
                    coly[intmat][0][self.cnt,:] = 0,0,0
                    match_rate[intmat][0][self.cnt] = 0
                    mat_global[intmat][0][self.cnt] = 0
                        
                cnt += 1
                self.cnt += 1
                continue
            
            spots_in_center = np.arange(0,len(data_theta))

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
            prediction = predict_DNN(codebars, wb, temp_key)
            max_pred = np.max(prediction, axis = 1)
            class_predicted = np.argmax(prediction, axis = 1)
            # print("Total spots attempted:"+str(len(spots_in_center)))
            # print("Took "+ str(time.time()-strat_time_P)+" seconds to predict spots")       
            predicted_hkl123 = classhkl[class_predicted]
            predicted_hkl123 = predicted_hkl123.astype(int)
            
            #print(predicted_hkl123)
            s_tth = data_theta * 2.
            s_chi = data_chi
            
            rotation_matrix1, mr_highest, mat_highest, \
                strain_crystal, strain_sample, iR_pix1, \
                            fR_pix1, spots_len1, best_match1,\
                                check12 = predict_ubmatrix(seednumber, spots_in_center, classhkl, 
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
                                                                        cnt=self.cnt,
                                                                        dict_dp=dict_dp,
                                                                        rotation_matrix=self.rotation_matrix,
                                                                        mat_global=self.mat_global,
                                                                        strain_calculation=strain_calculation,
                                                                        ind_mat=ind_mat, 
                                                                        ind_mat1=ind_mat1,
                                                                        tolerance=tolerance,
                                                                        tolerance1 =tolerance1,
                                                                        matricies=matricies,
                                                                        tabledistancerandom=tabledistancerandom,
                                                                        text_file = text_file,
                                                                        try_previous1=try_previous1,
                                                                        mode_spotCycle = mode_spotCycle1,
                                                                        softmax_threshold_global123=softmax_threshold_global123,
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
                                                                        match_rate=self.match_rate,
                                                                        check=self.check[self.cnt,:],
                                                                        crystal=crystal,
                                                                        crystal1=crystal1,
                                                                        angbins=angbins,
                                                                        wb=wb, temp_key=temp_key,
                                                                        strain_free_parameters=strain_free_parameters)
                
            for intmat in range(matricies):

                if len(rotation_matrix1[intmat]) == 0:
                    col[intmat][0][self.cnt,:] = 0,0,0
                    colx[intmat][0][self.cnt,:] = 0,0,0
                    coly[intmat][0][self.cnt,:] = 0,0,0
                else:
                    # mat_global[intmat][0][self.cnt] = mat_highest[intmat][0]
                    self.mat_global[intmat][0][self.cnt] = mat_highest[intmat][0]
                    
                    final_symm =symmetry
                    final_crystal = crystal
                    if mat_highest[intmat][0] == 1:
                        final_symm = symmetry
                        final_crystal = crystal
                    elif mat_highest[intmat][0] == 2:
                        final_symm = symmetry1
                        final_crystal = crystal1
                    symm_operator = final_crystal._hklsym
                    # strain_matrix[intmat][0][cnt,:,:] = strain_crystal[intmat][0]
                    # strain_matrixs[intmat][0][cnt,:,:] = strain_sample[intmat][0]
                    self.strain_matrix[intmat][0][self.cnt,:,:] = strain_crystal[intmat][0]
                    self.strain_matrixs[intmat][0][self.cnt,:,:] = strain_sample[intmat][0]
                    # rotation_matrix[intmat][0][cnt,:,:] = rotation_matrix1[intmat][0]
                    self.rotation_matrix[intmat][0][self.cnt,:,:] = rotation_matrix1[intmat][0]
                    col_temp = get_ipf_colour(rotation_matrix1[intmat][0], np.array([0., 0., 1.]), final_symm, symm_operator)
                    # col[intmat][0][cnt,:] = col_temp
                    self.col[intmat][0][self.cnt,:] = col_temp
                    col_tempx = get_ipf_colour(rotation_matrix1[intmat][0], np.array([1., 0., 0.]), final_symm, symm_operator)
                    # colx[intmat][0][cnt,:] = col_tempx
                    self.colx[intmat][0][self.cnt,:] = col_tempx
                    col_tempy = get_ipf_colour(rotation_matrix1[intmat][0], np.array([0., 1., 0.]), final_symm, symm_operator)
                    # coly[intmat][0][cnt,:] = col_tempy
                    self.coly[intmat][0][self.cnt,:] = col_tempy
                    # match_rate[intmat][0][cnt] = mr_highest[intmat][0]    
                    self.match_rate[intmat][0][self.cnt] = mr_highest[intmat][0]
                    # spots_len[intmat][0][cnt] = spots_len1[intmat][0]    
                    self.spots_len[intmat][0][self.cnt] = spots_len1[intmat][0]
                    # iR_pix[intmat][0][cnt] = iR_pix1[intmat][0]    
                    self.iR_pix[intmat][0][self.cnt] = iR_pix1[intmat][0]
                    # fR_pix[intmat][0][cnt] = fR_pix1[intmat][0]    
                    self.fR_pix[intmat][0][self.cnt] = fR_pix1[intmat][0]
                    # best_match[intmat][0][cnt] = best_match1
                    self.best_match[intmat][0][self.cnt] = best_match1[intmat][0]
                    self.check[self.cnt,intmat] = check12[intmat]
            cnt += 1
            self.cnt += 1
            
def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)

def toggle_selector1(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector1.RS.active:
        print(' LineSelector deactivated.')
        toggle_selector1.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector1.RS.active:
        print(' LineSelector activated.')
        toggle_selector1.RS.set_active(True)
        
def start():
    """ start of GUI for module launch"""
    # Handle high resolution displays:
    # fixes same widgets size across different screens
    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    try:
        screen = app.primaryScreen()
        print('Screen: %s' % screen.name())
        size = screen.size()
        print('Size: %d x %d' % (size.width(), size.height()))
        rect = screen.availableGeometry()
        print('Available: %d x %d' % (rect.width(), rect.height()))
        win = Window(rect.width()//2.5, rect.height()//1.2)
    except:
        win = Window()
    win.show()
    sys.exit(app.exec_()) 

if __name__ == "__main__":
    start()
