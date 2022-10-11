# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:00:12 2022

@author: PURUSHOT
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
import sys
import re
import glob
import itertools
import _pickle as cPickle
import time, datetime
import threading
import multiprocessing as multip
from multiprocessing import Process, Queue, cpu_count
import configparser
from skimage.transform import (hough_line, hough_line_peaks)

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QPushButton, QWidget, QFormLayout, \
                            QApplication, QSlider, QVBoxLayout, QTextEdit, \
                            QComboBox, QLineEdit, QFileDialog, QMenuBar
## util library with MP function
try:
    from utils_lauenn import chunker_list,\
        predict_preprocessMultiMatProcess, simulate_spots, mse_images, \
        new_MP_multimat_functionGUI, computeGnomonicImage, global_plots_MM
    from NNmodels import read_hdf5 
except:
    from lauetoolsnn.utils_lauenn import chunker_list,\
        predict_preprocessMultiMatProcess, simulate_spots, mse_images, \
        new_MP_multimat_functionGUI, computeGnomonicImage, global_plots_MM
    from lauetoolsnn.NNmodels import read_hdf5 

try:
    from lauetools import dict_LaueTools as dictLT
    from lauetools import IOLaueTools as IOLT
    from lauetools import LaueGeometry as Lgeo
    from lauetools import readmccd as RMCCD
    from lauetools import IOimagefile as IOimage
    from lauetools import imageprocessing as ImProc
except:
    import lauetoolsnn.lauetools.dict_LaueTools as dictLT
    import lauetoolsnn.lauetools.IOLaueTools as IOLT
    import lauetoolsnn.lauetools.LaueGeometry as Lgeo
    import lauetoolsnn.lauetools.readmccd as RMCCD
    import lauetoolsnn.lauetools.IOimagefile as IOimage
    import lauetoolsnn.lauetools.imageprocessing as ImProc

from keras.models import model_from_json

def resource_path(relative_path, verbose=0):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = os.path.dirname(__file__)
    if verbose:
        print("Base path of the library: ",base_path)
    return os.path.join(base_path, relative_path)

Logo = resource_path("lauetoolsnn_logo.png",  verbose=0)
gui_state = np.random.randint(1e6)
cpu_count_user = cpu_count()

class Window(QMainWindow):
    """Main Window."""
    def __init__(self, 
                winx, 
                winy, 
                model_direc,
                material_,
                emin,
                emax,
                symmetry,
                detectorparameters,
                pixelsize,
                lattice_material,
                mode_spotCycleglobal,
                softmax_threshold_global,
                mr_threshold_global,
                cap_matchrate,
                coeff,
                coeff_overlap,
                fit_peaks_gaussian_global,
                FitPixelDev_global,
                NumberMaxofFits,
                tolerance_strain,
                material0_limit,
                use_previous_UBmatrix,
                material_phase_always_present,
                crystal,
                strain_free_parameters,
                additional_expression,
                strain_label_global, 
                UB_matrix_global, 
                boxsize_global, 
                intensity_threshold_global,
                ccd_label_global, 
                exp_prefix_global, 
                image_grid_globalx, 
                image_grid_globaly,
                tolerance_global, 
                expfile_global, 
                weightfile_global,
                model_annote):
        """Initializer."""
        super(Window, self).__init__()

        app_icon = QtGui.QIcon()
        app_icon.addFile(Logo, QtCore.QSize(16,16))
        self.setWindowIcon(app_icon)
        
        if winx==None or winy==None:
            self.setFixedSize(16777215,16777215)
        else:
            self.setFixedSize(winx, winy)

        self.setWindowTitle("Laue Neural-Network v3")
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
        self.setDisplayText("Load a config file first (for example see the example_config tab)")
        self.popups = []
        # self.showMaximized()
        self.setFixedSize(16777215,16777215)
        
        
        ### Variables needed for PLOT functions
        self.model_direc = model_direc
        self.material_ = material_
        self.emin = emin
        self.emax = emax
        self.symmetry = symmetry
        self.detectorparameters = detectorparameters
        self.pixelsize = pixelsize
        self.lattice_material = lattice_material
        self.mode_spotCycleglobal = mode_spotCycleglobal
        self.softmax_threshold_global = softmax_threshold_global
        self.mr_threshold_global = mr_threshold_global
        self.cap_matchrate = cap_matchrate
        self.coeff = coeff
        self.coeff_overlap = coeff_overlap
        self.fit_peaks_gaussian_global = fit_peaks_gaussian_global
        self.FitPixelDev_global = FitPixelDev_global
        self.NumberMaxofFits = NumberMaxofFits
        self.tolerance_strain = tolerance_strain
        self.material0_limit = material0_limit
        self.use_previous_UBmatrix = use_previous_UBmatrix
        self.material_phase_always_present = material_phase_always_present
        self.crystal = crystal
        self.strain_free_parameters = strain_free_parameters
        self.additional_expression = additional_expression
        self.strain_label_global = strain_label_global
        self.UB_matrix_global = UB_matrix_global
        self.boxsize_global = boxsize_global 
        self.intensity_threshold_global = intensity_threshold_global
        self.ccd_label_global = ccd_label_global
        self.exp_prefix_global = exp_prefix_global 
        self.image_grid_globalx = image_grid_globalx 
        self.image_grid_globaly = image_grid_globaly
        self.tolerance_global = tolerance_global
        self.expfile_global = expfile_global
        self.weightfile_global = weightfile_global
        self.model_annote = model_annote
        
        self.show_window_liveprediction()
        
    def _createDisplay(self):
        """Create the display."""
        self.display = QTextEdit()
        self.display.setReadOnly(True)
        self.layout.addWidget(self.display)

    def setDisplayText(self, text):
        self.display.append('%s'%text)
        self.display.moveCursor(QtGui.QTextCursor.End)
        self.display.setFocus()

    def show_window_liveprediction(self):
        try:
            hkl_all_class0 = []
            for ino, imat in enumerate(self.material_):
                with open(self.model_direc+"//classhkl_data_nonpickled_"+imat+".pickle", "rb") as input_file:
                    hkl_all_class_load = cPickle.load(input_file)[0]
                hkl_all_class0.append(hkl_all_class_load)
        except:
            print("Couldn't load the data properly! please check")
            return
        
        w2 = AnotherWindowLivePrediction(material_=self.material_, 
                                         emin=self.emin, 
                                         emax=self.emax, 
                                         symmetry=self.symmetry,
                                         detectorparameters=self.detectorparameters, 
                                         pixelsize=self.pixelsize,
                                         lattice_=self.lattice_material,
                                         hkl_all_class0 = hkl_all_class0,
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
                                         material0_limit = self.material0_limit,
                                         use_previous_UBmatrix_name = self.use_previous_UBmatrix,
                                         material_phase_always_present = self.material_phase_always_present,
                                         crystal=self.crystal,
                                         strain_free_parameters=self.strain_free_parameters,
                                         additional_expression=self.additional_expression,
                                         strain_label_global = self.strain_label_global,
                                         UB_matrix_global = self.UB_matrix_global,
                                         boxsize_global = self.boxsize_global ,
                                         intensity_threshold_global = self.intensity_threshold_global,
                                         ccd_label_global = self.ccd_label_global,
                                         exp_prefix_global = self.exp_prefix_global ,
                                         image_grid_globalx = self.image_grid_globalx ,
                                         image_grid_globaly = self.image_grid_globaly,
                                         tolerance_global = self.tolerance_global,
                                         expfile_global = self.expfile_global,
                                         modelfile_global = self.model_direc,
                                         weightfile_global = self.weightfile_global,
                                         model_annote = self.model_annote)
        w2.show()
        self.popups.append(w2)

class AnotherWindowLivePrediction(QWidget):
    def __init__(self, 
                 material_=None, 
                 emin=None, 
                 emax=None, 
                 symmetry=None, 
                 detectorparameters=None, 
                 pixelsize=None, 
                 lattice_=None, 
                 hkl_all_class0=None, 
                 mode_spotCycleglobal=None,
                 softmax_threshold_global = None, 
                 mr_threshold_global =    None, 
                 cap_matchrate =    None,
                 coeff =    None, 
                 coeff_overlap1212 =    None, 
                 fit_peaks_gaussian_global =    None,
                 FitPixelDev_global =    None, 
                 NumberMaxofFits =    None, 
                 tolerance_strain =    None,
                 material0_limit = None, 
                 use_previous_UBmatrix_name = None, 
                 material_phase_always_present=None, 
                 crystal=None,
                 strain_free_parameters=None, 
                 additional_expression=None,
                 strain_label_global="YES", 
                 UB_matrix_global=5, 
                 boxsize_global=5, 
                 intensity_threshold_global=100,
                 ccd_label_global="cor", 
                 exp_prefix_global=None, 
                 image_grid_globalx=25, 
                 image_grid_globaly=25,
                 tolerance_global=0.5, 
                 expfile_global=None, 
                 modelfile_global=None, 
                 weightfile_global=None,
                 model_annote=None):
        
        super(AnotherWindowLivePrediction, self).__init__()
        app_icon = QtGui.QIcon()
        app_icon.addFile(Logo, QtCore.QSize(16,16))
        self.setWindowIcon(app_icon)
        self.myQMenuBar = QMenuBar(self)
        self._createMenu()
        
        self.material_phase_always_present = material_phase_always_present
        self.material0_limit = material0_limit
        self.softmax_threshold_global = softmax_threshold_global
        self.mr_threshold_global = mr_threshold_global
        self.cap_matchrate = cap_matchrate
        self.coeff = coeff
        self.coeff_overlap = coeff_overlap1212
        self.fit_peaks_gaussian_global = fit_peaks_gaussian_global
        self.FitPixelDev_global = FitPixelDev_global
        self.NumberMaxofFits = NumberMaxofFits        
        self.tolerance_strain = tolerance_strain
        self.mode_spotCycle = mode_spotCycleglobal
        self.material_ = material_
        self.files_treated = []
        self.cnt = 0
        self.emin = emin
        self.emax= emax
        self.lattice_ = lattice_
        self.symmetry = symmetry
        self.crystal = crystal
        self.hkl_all_class0 = hkl_all_class0
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
        self.tolerance = tolerance_global
        self.weightfile_global=weightfile_global
        self.model_annote = model_annote
        self.filenameDirec = expfile_global
        self.modelDirec = modelfile_global
        self.filenameModel = [weightfile_global]
        
        self.ipf_axis = QComboBox()
        choices = ["Z","Y","X"]
        for s in choices:
            self.ipf_axis.addItem(s)
        
        self.filenamebkg = None
        self.blacklist_file = None
        
        self.image_grid = QLineEdit()
        self.image_grid.setText("10,10")
        
        self.ubmat = QLineEdit()
        self.ubmat.setText("1")
        
        self.bkg_treatment = QLineEdit()
        self.bkg_treatment.setText("A-B")

        self.ccd_label = QComboBox()
        self.ccd_label.addItem("Cor")
        choices = dictLT.dict_CCD.keys()
        for s in choices:
            self.ccd_label.addItem(s)
            
        self.intensity_threshold = QLineEdit()
        self.intensity_threshold.setText("1500")
        
        self.experimental_prefix = QLineEdit()
        self.experimental_prefix.setText("")
        
        self.boxsize = QLineEdit()
        self.boxsize.setText("5")

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
        choices = ["MultiProcessing"]
        for s in choices:
            self.matrix_plot_tech.addItem(s)        
        
        self.analysis_plot_tech = QComboBox()
        choices = ["slow", "graphmode"]
        for s in choices:
            self.analysis_plot_tech.addItem(s)
        
        self.strain_plot_tech = QComboBox()
        choices = ["NO", "YES"]
        for s in choices:
            self.strain_plot_tech.addItem(s)
        ### default values here
        if mode_spotCycleglobal != None:
            self.analysis_plot_tech.setCurrentText(mode_spotCycleglobal)
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
        if len(self.material_) == 1:
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
        if len(self.material_) > 1:
            prefix_mat = self.material_[0]
            for ino, imat in enumerate(self.material_):
                if ino == 0:
                    continue
                prefix_mat = prefix_mat + "_" + imat
        else:
            prefix_mat = self.material_[0]

        json_file = open(self.model_direc+"//model_"+prefix_mat+".json", 'r')
                
        self.classhkl = np.load(self.model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_0"]
        self.angbins = np.load(self.model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_1"]
        ind_mat_all = np.load(self.model_direc+"//MOD_grain_classhkl_angbin.npz",allow_pickle=True)["arr_5"]
        ind_mat = []
        for inni in ind_mat_all:
            ind_mat.append(len(inni))
        self.ind_mat = [item for item in itertools.accumulate(ind_mat)]
        
        load_weights = self.filenameModel[0]
        self.wb = read_hdf5(load_weights)
        self.temp_key = list(self.wb.keys())
        # # load json and create model
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        print("Constructing model")
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
        self.ncpu = cpu_count_user
        self.cor_file_directory = self.filenameDirec + "//" + self.experimental_prefix.text()+"CORfiles"
        if format_file in ['cor',"COR","Cor"]:
            self.cor_file_directory = self.filenameDirec
        if not os.path.exists(self.cor_file_directory):
            os.makedirs(self.cor_file_directory)
    
    def load_results(self, filename):
        try:
            with open(filename, "rb") as input_file:
                self.best_match, self.mat_global, self.rotation_matrix, \
                    self.strain_matrix, self.strain_matrixs, self.col,\
                    self.colx, self.coly, self.match_rate, \
                    self.files_treated, self.lim_x, self.lim_y,\
                    self.spots_len, self.iR_pix, self.fR_pix,\
                    self.material_, self.lattice_material,\
                    self.symmetry, self.crystal = cPickle.load(input_file)
        except:
            print("Error loading the result file; please verify")
            return
                                
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
                                     strain_free_parameters,additional_expression):
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
            best_match_mpdata, pred_hkl = predict_preprocessMultiMatProcess(
                                                                       filenameSingleExp, 
                                                                       0, rotation_matrix,strain_matrix,strain_matrixs,
                                                                       col,colx,coly,match_rate,spots_len,iR_pix,fR_pix,best_match,
                                                                       mat_global, check,self.detectorparameters,self.pixelsize,self.angbins,
                                                                       self.classhkl, self.hkl_all_class0,  self.emin, self.emax,
                                                                       self.material_, self.symmetry, lim_x,lim_y,
                                                                       self.strain_calculation, self.ind_mat,
                                                                       self.model_direc, self.tolerance,
                                                                       int(self.ubmat.text()), self.ccd_label.currentText(),
                                                                       None,float(self.intensity_threshold.text()),
                                                                       int(self.boxsize.text()),self.bkg_treatment.text(),
                                                                       self.filenameDirec, self.experimental_prefix.text(),
                                                                       None, None, 
                                                                       [],False,self.wb, self.temp_key, self.cor_file_directory, mode_analysis,
                                                                        self.softmax_threshold_global,self.mr_threshold_global,
                                                                        self.cap_matchrate,self.tolerance_strain,
                                                                        self.NumberMaxofFits,self.fit_peaks_gaussian_global,
                                                                        self.FitPixelDev_global,self.coeff,
                                                                        self.coeff_overlap,self.material0_limit,
                                                                        False,self.material_phase_always_present,
                                                                        self.crystal,self.strain_free_parameters,
                                                                        self.model_annote)
        # predict_preprocessMP_vsingle() #TODO
        
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
            dict_dp['detectordiameter']=pixelsize*framedim[0]
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
            if mat_global987 > 0:
                material_=self.material_[int(mat_global987)-1]
                tolerance_add = self.tolerance[int(mat_global987)-1]
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
        ## Provide path to a single tiff or cor file to predict and write a pickle object
        filenameSingleExp = QFileDialog.getOpenFileName(self, 'Select a single experimental file',
                                                        resource_path("examples"))
        if len(filenameSingleExp[0]) == 0:
            return
        filenameSingleExp = filenameSingleExp[0]
        model_direc = self.modelDirec
        lim_x, lim_y = int(1), int(1)
        ## load model related files and generate the model
        if len(self.material_) > 1:
            prefix_mat = self.material_[0]
            for ino, imat in enumerate(self.material_):
                if ino == 0:
                    continue
                prefix_mat = prefix_mat + "_" + imat
        else:
            prefix_mat = self.material_[0]
        json_file = open(model_direc+"//model_"+prefix_mat+".json", 'r')
                
        classhkl = np.load(model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_0"]
        angbins = np.load(model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_1"]
        ind_mat_all = np.load(model_direc+"//MOD_grain_classhkl_angbin.npz",allow_pickle=True)["arr_5"]
        ind_mat = []
        for inni in ind_mat_all:
            ind_mat.append(len(inni))
        ind_mat = [item for item in itertools.accumulate(ind_mat)]
        
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
            
        strain_matrix12, strain_matrixs12, \
        rotation_matrix12, col12, \
        colx12, coly12,\
        match_rate12, mat_global12, cnt12,\
        files_treated12, spots_len12, \
        iR_pix12, fR_pix12, check12, \
            best_match12, pred_hkl = predict_preprocessMultiMatProcess(
                                                                       filenameSingleExp, 
                                                                       0, rotation_matrix,strain_matrix,strain_matrixs,
                                                                       col,colx,coly,match_rate,spots_len,iR_pix,fR_pix,best_match,
                                                                       mat_global, check,self.detectorparameters,self.pixelsize,angbins,
                                                                       classhkl, self.hkl_all_class0,  self.emin, self.emax,
                                                                       self.material_, self.symmetry, lim_x,lim_y,
                                                                       self.strain_calculation, ind_mat,
                                                                       model_direc, self.tolerance,
                                                                       int(self.ubmat.text()), self.ccd_label.currentText(),
                                                                       None,float(self.intensity_threshold.text()),
                                                                       int(self.boxsize.text()),self.bkg_treatment.text(),
                                                                       self.filenameDirec, self.experimental_prefix.text(),
                                                                       blacklist, None, 
                                                                       [],False,wb, temp_key, cor_file_directory, mode_analysis,
                                                                        self.softmax_threshold_global,self.mr_threshold_global,
                                                                        self.cap_matchrate,self.tolerance_strain,
                                                                        self.NumberMaxofFits,self.fit_peaks_gaussian_global,
                                                                        self.FitPixelDev_global,self.coeff,
                                                                        self.coeff_overlap,self.material0_limit,
                                                                        False,self.material_phase_always_present,
                                                                        self.crystal,self.strain_free_parameters,
                                                                        self.model_annote)
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
            dict_dp['detectordiameter']=pixelsize*framedim[0]
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
                dict_dp['detectordiameter']=pixelsize*framedim[0]
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
            if mat_global987 > 0:
                material_=self.material_[int(mat_global987)-1]
                tolerance_add = self.tolerance[int(mat_global987)-1]
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
        w.show()       
        self.popups.append(w)
  
    def save_btn(self,):
        curr_time = time.time()
        now = datetime.datetime.fromtimestamp(curr_time)
        c_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        
        ## Write global text file with all results
        if len(self.material_) > 1:
            prefix_mat = self.material_[0]
            for ino, imat in enumerate(self.material_):
                if ino == 0:
                    continue
                prefix_mat = prefix_mat + "_" + imat
        else:
            prefix_mat = self.material_[0]
        save_directory_ = self.model_direc+"//results_"+prefix_mat+"_"+c_time
        if not os.path.exists(save_directory_):
            os.makedirs(save_directory_)

        ## intermediate saving of pickle objects with results
        with open(save_directory_+ "//results.pickle", "wb") as output_file:
                cPickle.dump([self.best_match, self.mat_global, self.rotation_matrix, self.strain_matrix, 
                              self.strain_matrixs, self.col, self.colx, self.coly, self.match_rate, 
                              self.files_treated, self.lim_x, self.lim_y, self.spots_len, self.iR_pix, 
                              self.fR_pix, self.material_, self.lattice_, self.symmetry, self.crystal], output_file)     

        try:      
            text_file = open(save_directory_+"//prediction_stats_"+prefix_mat+".txt", "w")
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
                    else:
                        case = self.material_[mat-1]

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
            print("Errors with writing prediction output text file; could be the prediction was stopped midway")
        #%  Plot some data  
        try:
            global_plots_MM(self.lim_x, self.lim_y, self.rotation_matrix, self.strain_matrix, self.strain_matrixs, 
                          self.col, self.colx, self.coly, self.match_rate, self.mat_global, self.spots_len, 
                          self.iR_pix, self.fR_pix, save_directory_, self.material_,
                          match_rate_threshold=5, bins=30, constantlength="a")
            print("global plots done")
        except:
            print("Error in the global plots module")
            
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
        if len(self.material_) > 1:
            prefix_mat = self.material_[0]
            for ino, imat in enumerate(self.material_):
                if ino == 0:
                    continue
                prefix_mat = prefix_mat + "_" + imat
        else:
            prefix_mat = self.material_[0]
            
        json_file = open(self.model_direc+"//model_"+prefix_mat+".json", 'r')

        self.classhkl = np.load(self.model_direc+"//MOD_grain_classhkl_angbin.npz",allow_pickle=True)["arr_0"]
        self.angbins = np.load(self.model_direc+"//MOD_grain_classhkl_angbin.npz",allow_pickle=True)["arr_1"]
        ind_mat_all = np.load(self.model_direc+"//MOD_grain_classhkl_angbin.npz",allow_pickle=True)["arr_5"]
        ind_mat = []
        for inni in ind_mat_all:
            ind_mat.append(len(inni))
        self.ind_mat = [item for item in itertools.accumulate(ind_mat)]
        
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
                self._worker_processes[i]= Process(target=new_MP_multimat_functionGUI, 
                                                   args=(self._inputs_queue, self._outputs_queue, i+1, run_flag))
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
        
        if len(self.material_) > 1:
            self.canvas.axes3.cla()
            prefix_mat = "Material Index "
            for ino, imat in enumerate(self.material_):
                prefix_mat = prefix_mat + str(ino+1)+": "+imat+"; "
            self.canvas.axes3.set_title(prefix_mat, loc='center', fontsize=10) 
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
                    dict_dp['detectordiameter']=pixelsize*framedim[0]
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
                      
                if mat_glob_plotfnc == 0:
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
                else:
                    material_=self.material_[int(mat_glob_plotfnc)-1]
                    tolerance_add = self.tolerance[int(mat_glob_plotfnc)-1]
                
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
                                                                      material_, 
                                                                      self.emax, 
                                                                      self.emin, 
                                                                      dict_dp['detectorparameters'], 
                                                                      dict_dp,
                                                                      tolerance_add, 
                                                                      data_theta*2.0,
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
            np.savez_compressed(self.model_direc+'//rotation_matrix_indexed_1.npz', 
                                self.rotation_matrix, self.mat_global, self.match_rate, 0.0)
        
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

        if cond_mode == "MultiProcessing":
            try_prevs = False

            valu12 = [[ self.filenm[ii].decode(), 
                        ii,
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
                        self.emin,
                        self.emax,
                        self.material_,
                        self.symmetry,
                        self.lim_x,
                        self.lim_y,
                        self.strain_calculation, 
                        self.ind_mat,
                        self.model_direc,
                        self.tolerance,
                        int(self.ubmat.text()), 
                        self.ccd_label.currentText(), 
                        None,
                        float(self.intensity_threshold.text()),
                        int(self.boxsize.text()),
                        self.bkg_treatment.text(),
                        self.filenameDirec,
                        self.experimental_prefix.text(),
                        blacklist,
                        None,
                        self.files_treated,
                        try_prevs,
                        self.wb,
                        self.temp_key,
                        self.cor_file_directory,
                        self.mode_spotCycle,
                        self.softmax_threshold_global,
                        self.mr_threshold_global,
                        self.cap_matchrate,
                        self.tolerance_strain,
                        self.NumberMaxofFits,
                        self.fit_peaks_gaussian_global,
                        self.FitPixelDev_global,
                        self.coeff,
                        self.coeff_overlap,
                        self.material0_limit,
                        self.use_previous_UBmatrix_name,
                        self.material_phase_always_present,
                        self.crystal,
                        self.strain_free_parameters,
                        self.model_annote] for ii in range(count_global)]
            
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

#%%
# =============================================================================
# #%%PLOT FUNCTIONS        
# =============================================================================
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
        self.peak_params.setText("500,15,15")
        
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
            
            if intensity_threshold != None and boxsize != None and FitPixelDev_global123!=None:
                self.peak_params.setText(str(intensity_threshold)+","+str(boxsize)+","+str(FitPixelDev_global123))
                
            self.predict_params.setText(str(float(softmax_threshold_global123))+","+str(float(cap_matchrate123))+
                                        ","+option_global+","+
                                        str(nb_spots_global_threshold)+","+str(residues_threshold)+
                                        ","+str(nb_spots_consider))
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
        except:
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
            print("error with setting config file")
            
        self.corrected_data = None
        self.image_mode = QComboBox()
        mode_ = ["raw","bkg_corrected"]
        for s in mode_:
            self.image_mode.addItem(s)
            
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
        formLayout.addRow('Intensity; box size; pix dev', self.peak_params)
        formLayout.addRow('Image mode', self.image_mode)
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
        config_setting1 = configparser.ConfigParser()
        filepath = resource_path('settings.ini')
        print("Settings path is "+filepath)
        config_setting1.read(filepath)
        config_setting1.set('CALLER', 'residues_threshold',str(self.residues_threshold))
        config_setting1.set('CALLER', 'nb_spots_global_threshold',str(self.nb_spots_global_threshold))
        config_setting1.set('CALLER', 'option_global',self.option_global)
        config_setting1.set('CALLER', 'use_om_user',"false")
        config_setting1.set('CALLER', 'nb_spots_consider',str(self.nb_spots_consider))
        config_setting1.set('CALLER', 'path_user_OM',"")
        config_setting1.set('CALLER', 'intensity', str(self.intensity_threshold))
        config_setting1.set('CALLER', 'boxsize', str(self.boxsize))
        config_setting1.set('CALLER', 'pixdev', str(self.FitPixelDev_global123))
        config_setting1.set('CALLER', 'cap_softmax', str(self.softmax_threshold_global123))
        config_setting1.set('CALLER', 'cap_mr', str(self.cap_matchrate123))
        config_setting1.set('CALLER', 'strain_free_parameters', ",".join(self.strain_free_parameters))
        config_setting1.set('CALLER', 'additional_expression', ",".join(self.additional_expression))
        with open(filepath, 'w') as configfile:
            config_setting1.write(configfile)
        print("Config settings updated")
    
    def neighbor_UB(self):
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
            
    def peak_search(self):
        self.propagate_button.setEnabled(False)
        intens = int(float(self.peak_params.text().split(",")[0]))
        bs = int(float(self.peak_params.text().split(",")[1]))
        pixdev = int(float(self.peak_params.text().split(",")[2]))
        bkg_treatment = self.bkg_treatment.text()
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
                                        )
            peak_XY = peak_XY[0]
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
            
            self.function_predict(self.file, intens, bs, pixdev, smt, mrt, self.image_no, nb_spots_consider_global123,
                                  residues_threshold123, option_global123, nb_spots_global_threshold123, self.peakXY,
                                  self.strain_free_parameters,self.additional_expression)
            self.propagate_button.setEnabled(True)
        except:
            print("Error during prediction; reinitialize the optimize window again, most probably this will fix it")
        
    def draw_something(self):
        # Drop off the first y element, append a new one.
        self.canvas.axes.cla()
        self.canvas.axes.set_title("Laue pattern of pixel x=%d, y=%d (file: %s)"%(self.iy,self.ix,self.file), loc='center', fontsize=8)
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
        self.canvas.axes.imshow(arr.astype('uint8'), origin='lower')        
        self.canvas.draw()
    
    def calculate_image_similarity(self):
        try:
            values = []
            count = 0 
            total = self.diff_data.shape[0]*self.diff_data.shape[1]
            for ix in range(self.diff_data.shape[0]):
                for iy in range(self.diff_data.shape[1]):
                    if iy == 0 and ix == 0:
                        continue
                    elif iy == 0 and ix != 0:
                        image_no = ix
                    elif iy != 0 and ix == 0:
                        image_no = iy * self.lim_y
                    elif iy != 0 and ix != 0:
                        image_no = iy * self.lim_y + ix
                
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
                    self.diff_data[r1[2],r1[1]] = r1[0]
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

def start(  model_direc = None,
            material_ = None,
            emin = None,
            emax = None,
            symmetry = None,
            detectorparameters = None,
            pixelsize = None,
            lattice_material = None,
            mode_spotCycleglobal = None,
            softmax_threshold_global = None,
            mr_threshold_global = None,
            cap_matchrate = None,
            coeff = None,
            coeff_overlap = None,
            fit_peaks_gaussian_global = None,
            FitPixelDev_global = None,
            NumberMaxofFits = None,
            tolerance_strain = None,
            material0_limit = None,
            use_previous_UBmatrix = None,
            material_phase_always_present = None,
            crystal = None,
            strain_free_parameters = None,
            additional_expression = None,
            strain_label_global = None, 
            UB_matrix_global = None, 
            boxsize_global = None, 
            intensity_threshold_global = None,
            ccd_label_global = None, 
            exp_prefix_global = None, 
            image_grid_globalx = None, 
            image_grid_globaly = None,
            tolerance_global = None, 
            expfile_global = None, 
            weightfile_global = None,
            model_annote = None):
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
        win = Window(rect.width()//5, 
                     rect.height()//5,
                     model_direc,
                     material_,
                     emin,
                     emax,
                     symmetry,
                     detectorparameters,
                     pixelsize,
                     lattice_material,
                     mode_spotCycleglobal,
                     softmax_threshold_global,
                     mr_threshold_global,
                     cap_matchrate,
                     coeff,
                     coeff_overlap,
                     fit_peaks_gaussian_global,
                     FitPixelDev_global,
                     NumberMaxofFits,
                     tolerance_strain,
                     material0_limit,
                     use_previous_UBmatrix,
                     material_phase_always_present,
                     crystal,
                     strain_free_parameters,
                     additional_expression,
                     strain_label_global, 
                     UB_matrix_global, 
                     boxsize_global, 
                     intensity_threshold_global,
                     ccd_label_global, 
                     exp_prefix_global, 
                     image_grid_globalx, 
                     image_grid_globaly,
                     tolerance_global, 
                     expfile_global, 
                     weightfile_global,
                     model_annote)
    except:
        win = Window(None,
                     None,
                     model_direc,
                     material_,
                     emin,
                     emax,
                     symmetry,
                     detectorparameters,
                     pixelsize,
                     lattice_material,
                     mode_spotCycleglobal,
                     softmax_threshold_global,
                     mr_threshold_global,
                     cap_matchrate,
                     coeff,
                     coeff_overlap,
                     fit_peaks_gaussian_global,
                     FitPixelDev_global,
                     NumberMaxofFits,
                     tolerance_strain,
                     material0_limit,
                     use_previous_UBmatrix,
                     material_phase_always_present,
                     crystal,
                     strain_free_parameters,
                     additional_expression,
                     strain_label_global, 
                     UB_matrix_global, 
                     boxsize_global, 
                     intensity_threshold_global,
                     ccd_label_global, 
                     exp_prefix_global, 
                     image_grid_globalx, 
                     image_grid_globaly,
                     tolerance_global, 
                     expfile_global, 
                     weightfile_global,
                     model_annote)
    win.show()
    sys.exit(app.exec_()) 

# if __name__ == "__main__":
#     start()