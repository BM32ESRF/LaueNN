# -*- coding: utf-8 -*-
"""
Dictionary of several parameters concerning Detectors, Materials, Transforms etc
that are used in LaueTools and in LaueToolsGUI module

Lauetools project
April 2019

"""
__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"
import numpy as np
import json
import os
##------------------------------------------------------------------------
# --- -----------  Element-materials library
#-------------------------------------------------------------------------
def resource_path(relative_path, verbose=0):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = os.path.dirname(__file__)
    if verbose:
        print("Base path of the library: ",base_path)
    return os.path.join(base_path, relative_path)

def get_dict_mat():
    with open(resource_path('material.json'),'r') as f:
        json_data = json.load(f)
    return json_data

class dict_mat():
    def __init__(self):
        self.dict_Materials = self.getvalue()
    
    def getvalue(self):
        with open(resource_path('material.json'),'r') as f:
            json_data = json.load(f)
        return json_data
    
    
dict_Materials = get_dict_mat()
dict_Materials_short = get_dict_mat()
    
def get_extinct_mat():
    with open(resource_path('extinction.json'),'r') as f:
        extinction_json = json.load(f)
    return extinction_json

dict_Extinc = get_extinct_mat()
dict_Extinc_inv = get_extinct_mat()
 
dict_Stiffness = {"Ge": ["Ge", [126, 44, 67.7], "cubic"]}


######## Geometrey Default  ##############
# Default constant
DEFAULT_DETECTOR_DISTANCE = 70.0  # mm
DEFAULT_DETECTOR_DIAMETER = 165.0  # mm
DEFAULT_TOP_GEOMETRY = "Z>0"

#############   2D DETECTOR ##############
LAUEIMAGING_DATA_FORMAT = "uint16"  # 'uint8'
LAUEIMAGING_FRAME_DIM = (1290, 1970)  # (645, 985)

CCD_CALIBRATION_PARAMETERS = ["dd", "xcen", "ycen", "xbet", "xgam", "pixelsize",
                                "xpixelsize", "ypixelsize", "CCDLabel",
                                "framedim", "detectordiameter", "kf_direction"]

# --- ---  CCD Read Image Parameters
# CCDlabel,
# framedim=(dim1, dim2),
# pixelsize,
# saturation value,
# geometrical operator key,
# header size in byte,
# binary encoding format,
# description,
# file extension
dict_CCD = {
    "MARCCD165": ((2048, 2048), 0.079142, 65535, "no", 4096, "uint16", "MAR Research 165 mm now rayonix", "mccd", ),
    "sCMOS": [(2018, 2016), 0.0734, 65535, "no", 3828, "uint16", "file as produced by sCMOS camera with checked fliplr transform. CCD parameters read from tif header by fabio", "tif"],
    "cor1": [(2018, 2016), 0.0734, 65535, "no", 3828, "uint16", "file as produced by sCMOS camera with checked fliplr transform. CCD parameters read from tif header by fabio", "tif"],
    "cor": [(2 * 2018, 2 * 2016), 0.0734 / 2.0, 65535, "no", 3828, "uint16", "binned 1x1, CCD parameters binned 1x1 read from tif header by fabio ", "tif"],
    "sCMOS_fliplr": [(2018, 2016), 0.0734, 65535, "sCMOS_fliplr", 3828, "uint16", "binned 2x2, CCD parameters read from tif header by fabio", "tif"],
    "sCMOS_fliplr_16M": [(2 * 2018, 2 * 2016), 0.0734 / 2.0, 65535, "sCMOS_fliplr", 3828, "uint16", "binned 1x1, CCD parameters binned 1x1 read from tif header by fabio ", "tif"],
    "sCMOS_16M": [(2 * 2018, 2 * 2016), 0.0734 / 2.0, 65535, "no", 3828, "uint16", "binned 1x1, CCD parameters binned 1x1 read from tif header by fabio ", "tif"],
    "psl_weiwei": [(1247, 1960), 0.075, 65535, "no", -1, "uint16", "camera from desy photonics science 1247*1960 ", "tif", ],
    "VHR_full": ((2671, 4008), 0.031, 10000, "vhr", 4096, "uint16", "NOT USED: very basic vhr settings, the largest frame available without grid correction", "tiff"),
    "VHR_diamond": ((2594, 3764), 0.031, 10000, "vhr", 4096, "uint16", "first vhr settings of Jun 12 close to diamond 2theta axis displayed is vertical, still problem with fit from PeakSearchGUI", "tiff"),
    "VHR_small": ((2594, 2748), 0.031, 10000, "vhr", 4096, "uint16", "vhr close to diamond Nov12 frame size is lower than VHR_diamond", "tiff"),
    "ImageStar": ((1500, 1500), 0.044, 65535, "vhr", 4096, "uint16", "Imagestar photonics Science close to diamond March13  extension is mar.tiff", "tiff"),
    "ImageStar_raw_3056x3056": ((3056, 3056), 0.022, 64000, "vhr", 872, "uint16", "raw image Apr 2018 imagestar for diamond binning 1x1  .tif (could be read by VHR_DLS?)", "tif"),
    "ImageStar_1528x1528": ((1528, 1528), 0.044, 65535, "vhr", 4096, "uint16", "Imagestar photonics Science close to diamond May 2018  extension is mar.tiff  non remapping 1528x1528", "tiff"),
    "ImageDeathStar": ((1500, 1500), 0.044, 65535, "VHR_Feb13", 4096, "uint16", "Imagestar photonics Science close to sample, mounting similar to MARCCD, Sep14", "tif"),
    "ImageStar_raw": ((1500, 1500), 0.044, 64000, "vhr", 872, "uint16", "raw image GISAXS BM32 November 2014 .tif", "tif"),
    "ImageStar_dia_2021": ((3056, 3056), 0.022, 65535, "ImageStar_dia_2021", 4096, "uint16", "Imagestar photonics Science close to diamond March13  extension is mar.tiff", "tif"),
    "VHR_diamond_Mar13": ((2594, 2774), 0.031, 10000, "vhr", 4096, "uint16", "vhr close to diamond Mar13 frame size is lower than VHR_diamond", "tiff"),
    "VHR": ((2594, 3764), 0.031, 10000, "VHR_Feb13", 4096, "uint16", "vhr settings of Jun 12 2theta axis displayed is horizontal, no problem with fit from PeakSearchGUI", "tiff"),
    "VHR_Feb13": ((2594, 2774), 0.031, 10000, "VHR_Feb13", 4096, "uint16", "vhr settings of Feb13 close to sample 2theta axis displayed is vertical, no problem with fit from PeakSearchGUI", "tiff"),
    "VHR_Feb13_rawtiff": ((2594, 2774), 0.031, 10000, "VHR_Feb13", 110, "uint16", " ", "tiff"),
    "VHR_PSI": ((2615, 3891), 0.0312, 65000, "no", 4096, "uint16", "vhr at psi actually read by libtiff (variable header size and compressed data)", "tif"),
    "VHR_DLS": ((3056, 3056), 0.0312, 65000, "no", 4096, "uint16", "vhr at dls actually read by libtiff (variable header size and compressed data)", "tif"),
    "PRINCETON": ((2048, 2048), 0.079, 57000, "no", 4096, "uint16", "ROPER Princeton Quadro 2048x2048 pixels converted from .spe to .mccd", "mccd"),  # 2X2, saturation value depends on gain and DAC
    "FRELON": ((2048, 2048), 0.048, 65000, "frelon2", 1024, "uint16", "FRELON camera 2048x2048 pixels, 2theta axis is horizontal (edf format)", "edf"),
    "TIFF Format": (-1, -1, "", "", "", "" "CCD parameters read from tiff header", "tiff", ),
    "EDF": ((1065, 1030), 0.075, 650000000000, "no", 0, "uint32", "CCD parameters read from edf header EIGER", "edf", ),
    "pnCCD": ((384, 384), 0.075, 65000, "no", 1024, "uint16", "pnCCD from SIEGEN only: pixel size and frame dimensions OK", "tiff"),
    "pnCCD_Tuba": ((384, 384), 0.075, 10000000, "no", 258, "uint16", "pnCCD from Tuba only: pixel size and frame dimensions OK", "tiff"),
    "EIGER_4M": ((2167, 2070), 0.075, 4294967295, "no", 0, "uint32", "CCD parameters read from tif header EIGER4M used at ALS", "tif"),
    "EIGER_4Mstack": ((2167, 2070), 0.075, 4294967295, "no", 0, "uint32", "detector parameters read hdf5 EIGER4M stack used at SLS", "h5"),
    "EIGER_4Munstacked": ((2167, 2070), 0.075, 4294967295, "no", 0, "uint32", "unstacked hdf5 EIGER4M  used at SLS", "unstacked"),
    "EIGER_1M": ((1065, 1030), 0.075, 4294967295, "no", 0, "uint32", "CCD parameters read from edf header EIGER1M at BM32 ESRF", "edf"),
}

# ------------------------------------------------------
# CCD pixels skewness:
# #RECTPIX = 0.0 : square pixels
# #RECTPIX = -1.0e-4
# define rectangular pixels with
# xpixsize = pixelsize
# ypixsize = xpixsize*(1.0+RECTPIX)
# ---------------------------------------------
RECTPIX = 0  # CCD pixel skewness   see find2thetachi

list_CCD_file_extensions = []
for key in list(dict_CCD.keys()):
    list_CCD_file_extensions.append(dict_CCD[key][-1])
list_CCD_file_extensions.append("tif.gz")
# print list_CCD_file_extensions

# ---   ---  general geometry of detector CCD position
DICT_LAUE_GEOMETRIES = {"Z>0": "Top Reflection (2theta=90)",
                        "X>0": "Transmission (2theta=0)",
                        "X<0": "Back Reflection (2theta=180)"}

DICT_LAUE_GEOMETRIES_INFO = {
    "Top Reflection (2theta=90)": ["Z>0", "top reflection geometry camera on top (2theta=90)"],
    "Transmission (2theta=0)": ["X>0", "Transmission geometry, camera in direct beam (2theta=0)"],
    "Back Reflection (2theta=180)": ["X<0", "Back reflection geometry, camera is upstream (2theta=180)"]
}


# --- -------------- History of Calibration Parameters
dict_calib = {
    "ZrO2 Sep08": [69.8076, 878.438, 1034.46, 0.54925, 0.18722],  # as first trial of zrO2 sicardy sep 08
    "Sep09": [68.0195, 934.94, 1033.6, 0.73674, -0.74386],  # Sep09
    "ZrO2 Dec09": [69.66221, 895.29492, 960.78674, 0.84324, -0.32201],  # Dec09 Julie Ge_run41_1_0003.mccd
    "Basic": [68, 930, 1100, 0.0, 0.0],
}



# --- -------------- Transforms 3x3 Matrix
dict_Vect = {
    "Identity": [[1.0, 0, 0], [0, 1, 0], [0, 0, 1]],
    "Default": [[1.0, 0, 0], [0, 1, 0], [0, 0, 1]],
    "sigma3_1": [
        [1.0 / 3, -2.0 / 3, -2.0 / 3],
        [-2.0 / 3, 1.0 / 3, -2.0 / 3],
        [-2.0 / 3, -2.0 / 3, 1.0 / 3],
    ],
    "sigma3_2": [
        [1.0 / 3, -2.0 / 3, -2.0 / 3],
        [2.0 / 3, -1.0 / 3, 2.0 / 3],
        [2.0 / 3, 2.0 / 3, -1.0 / 3],
    ],
    "sigma3_3": [
        [-1.0 / 3, 2.0 / 3, 2.0 / 3],
        [-2.0 / 3, 1.0 / 3, -2.0 / 3],
        [2.0 / 3, 2.0 / 3, -1.0 / 3],
    ],
    "sigma3_4": [
        [-1.0 / 3, 2.0 / 3, 2.0 / 3],
        [2.0 / 3, -1.0 / 3, 2.0 / 3],
        [-2.0 / 3, -2.0 / 3, 1.0 / 3],
    ],
    # "sigma3_1": [
    #     [-1.0 / 3, 2.0 / 3, 2.0 / 3],
    #     [2.0 / 3, -1.0 / 3, 2.0 / 3],
    #     [2.0 / 3, 2.0 / 3, -1.0 / 3],
    # ],
    # "sigma3_2": [
    #     [-1.0 / 3, -2.0 / 3, 2.0 / 3],
    #     [-2.0 / 3, -1.0 / 3, -2.0 / 3],
    #     [2.0 / 3, -2.0 / 3, -1.0 / 3],
    # ],
    # "sigma3_3": [
    #     [-1.0 / 3, 2.0 / 3, -2.0 / 3],
    #     [2.0 / 3, -1.0 / 3, -2.0 / 3],
    #     [-2.0 / 3, -2.0 / 3, -1.0 / 3],
    # ],
    # "sigma3_4": [
    #     [-1.0 / 3, -2.0 / 3, -2.0 / 3],
    #     [-2.0 / 3, -1.0 / 3, 2.0 / 3],
    #     [-2.0 / 3, 2.0 / 3, -1.0 / 3],
    # ],
    "shear1": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.008, 0.0, 1.0]],
    "shear2": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.016, 0.0, 1.0]],
    "shear3": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.024, 0.0, 1.0]],
    "shear4": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.032, 0.0, 1.0]],
    "JSMtest": [[1.0, 1.02, 0.98], [-0.1, 1.1, 0.2], [-0.032, 0.05, 1.15]],
    "JSMtest2": [[1.0, 1.02, 0.98], [-0.1, 1.1, 0.2], [-0.032, 0.05, 1.15]],
    "543_909075": [[0.20705523608201659, -0.066987298107780757, 2.0410779985789219e-17],
        [0.0, 0.24999999999999994, -2.0410779985789219e-17],
        [0.0, 0.0, 0.33333333333333331]]}

dict_Transforms = {
    "Identity": [[1.0, 0, 0], [0, 1, 0], [0, 0, 1]],
    "Default": [[1.0, 0, 0], [0, 1, 0], [0, 0, 1]],
    "sigma3_1": [
        [1.0 / 3, -2.0 / 3, -2.0 / 3],
        [-2.0 / 3, 1.0 / 3, -2.0 / 3],
        [-2.0 / 3, -2.0 / 3, 1.0 / 3],
    ],
    "sigma3_2": [
        [1.0 / 3, -2.0 / 3, -2.0 / 3],
        [2.0 / 3, -1.0 / 3, 2.0 / 3],
        [2.0 / 3, 2.0 / 3, -1.0 / 3],
    ],
    "sigma3_3": [
        [-1.0 / 3, 2.0 / 3, 2.0 / 3],
        [-2.0 / 3, 1.0 / 3, -2.0 / 3],
        [2.0 / 3, 2.0 / 3, -1.0 / 3],
    ],
    "sigma3_4": [
        [-1.0 / 3, 2.0 / 3, 2.0 / 3],
        [2.0 / 3, -1.0 / 3, 2.0 / 3],
        [-2.0 / 3, -2.0 / 3, 1.0 / 3],
    ],
    "shear1": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.008, 0.0, 1.0]],
    "shear2": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.016, 0.0, 1.0]],
    "shear3": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.024, 0.0, 1.0]],
    "shear4": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.032, 0.0, 1.0]],
    "stretch_axe1_0p01": [[1.01, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "stretch_axe2_0p01": [[1.0, 0.0, 0.0], [0.0, 1.01, 0.0], [0.0, 0.0, 1.0]],
    "stretch_axe3_0p01": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.01]],
    "JSMtest": [[1.0, 1.02, 0.98], [-0.1, 1.1, 0.2], [-0.032, 0.05, 1.15]],
    "JSMtest2": [[1.0, 1.02, 0.98], [-0.1, 1.1, 0.2], [-0.032, 0.05, 1.15]],
    "twin100": [[1, 0, 0], [0, -1, 0], [0, 0, -1.0]],
    "twin010": [[-1, 0, 0], [0, 1, 0], [0, 0, -1.0]],
    "twin001": [[-1, 0, 0], [0, -1, 0], [0, 0, 1.0]],
}

sq3 = np.sqrt(3)
sq2 = np.sqrt(2)
sq6 = np.sqrt(6)
# --- --------------  (almost) Rotation Matrices

dict_Rot = {
    "mat203": [
        [0.2168812, 0.00086280000000000005, -0.9761976],
        [0.0143763, 0.99988829999999995, 0.0040777000000000001],
        [0.97609219999999997, -0.014918499999999999, 0.2168445],
    ],
    "Si_001": [
        [-0.55064559927923185, 0.54129537384117710, -0.63548417889810638],
        [-0.70646885891790212, -0.70765027148876392, 0.00938623461321308],
        [-0.44458921907077736, 0.45412248299046004, 0.77207936061212878],
    ],
    "Ge_23Feb09": [
        [-0.48273739999999998, 0.22555140000000001, -0.84622169999999997],
        [-0.80558430000000003, 0.2646423, 0.53009289999999998],
        [0.34350920000000001, 0.9375985, 0.0539479],
    ],
    "matTEST": [
        [-0.069614099999999998, -0.097000299999999998, -0.99284680000000003],
        [-0.71808340000000004, 0.69573390000000002, -0.017623799999999998],
        [0.69246669999999999, 0.71171989999999996, -0.1180872],
    ],
    "OrientSurf001": [
        [0.76604444311897801, 0.0, -0.64278760968653925],
        [0.0, 1.0, 0.0],
        [0.64278760968653925, 0.0, 0.76604444311897801],
    ],
    "OrientSurf101": [
        [0.087155742747658138, 0.0, -0.99619469809174555],
        [0.0, 1.0, 0.0],
        [0.99619469809174555, 0.0, 0.087155742747658138],
    ],
    "matsolCu": [
        [-0.87097440000000004, -0.0123476, -0.49117309999999997],
        [-0.24467739999999999, 0.87780820000000004, 0.4118078],
        [0.42607099999999998, 0.47885309999999998, -0.76756970000000002],
    ],
    "mat112": [
        [0.33640419999999999, -0.18571589999999999, -0.92322360000000003],
        [-0.30955670000000002, 0.90407420000000005, -0.29465999999999998],
        [0.88938569999999995, 0.3849149, 0.24664469999999999],
    ],
    "mat113": [
        [0.44508639999999999, -0.079989099999999994, -0.89190800000000003],
        [-0.26918449999999999, 0.9379864, -0.21845200000000001],
        [0.85407129999999998, 0.3373178, 0.39595320000000001],
    ],
    "Identity": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "mat111": [
        [0.10547339999999999, -0.40906530000000002, -0.906389],
        [-0.40097290000000002, 0.81659619999999999, -0.41520049999999997],
        [0.90999790000000003, 0.40722999999999998, -0.077894900000000003],
    ],
    "mat313": [
        [0.0307196, -0.15987100000000001, -0.98665979999999998],
        [-0.14832409999999999, 0.97546739999999998, -0.1626756],
        [0.98846160000000005, 0.1513428, 0.0062532999999999998],
    ],
    "mat213": [
        [0.2158294, -0.098352499999999995, -0.97146509999999997],
        [-0.23944190000000001, 0.95920539999999999, -0.15030789999999999],
        [0.94661779999999995, 0.26505040000000002, 0.183475],
    ],
    "mat212": [
        [0.043270400000000001, -0.2274217, -0.97283450000000005],
        [-0.21745729999999999, 0.94825649999999995, -0.2313482],
        [0.97511029999999999, 0.22156049999999999, -0.0084230999999999993],
    ],
    "mat001c1": [
        [0.80585162458034332, -0.099044669238326555, -0.58377514453272938],
        [-0.050447334156050835, 0.97084166554927731, -0.23435350971946972],
        [0.58996466978353546, 0.21830403690176159, 0.77735774278669989],
    ],
    "mat311c2": [
        [0.65541179999999999, -0.15574589999999999, -0.73903890000000005],
        [0.2240627, 0.97455190000000003, -0.0066696000000000004],
        [0.72127059999999998, -0.16121969999999999, 0.6736297],
    ],
    "mat311c1": [
        [0.33213278843844807, -0.17992831998350736, -0.9259125783924782],
        [0.36259340405801743, 0.93056390242371512, -0.050766839018154847],
        [0.87075505327300606, -0.31886836727914458, 0.37431154818295931],
    ],
    "Ge2": [
        [-0.074566099999999996, -0.056390999999999997, -0.99562039999999996],
        [-0.71899650000000004, 0.69486250000000005, 0.0144922],
        [0.69100209999999995, 0.71692820000000002, -0.092358099999999999],
    ],
    "Ge1": [
        [-0.95482739999999999, 0.13672049999999999, -0.2638412],
        [-0.29632960000000003, -0.50444549999999999, 0.81100150000000004],
        [-0.022213, 0.85255040000000004, 0.52217279999999999],
    ],
    "OrientSurf111": [
        [-0.68482849999999995, 0.25444870000000003, -0.68283649999999996],
        [-0.7064087, -0.0017916, 0.70780189999999998],
        [0.1788759, 0.96708459999999996, 0.18097170000000001],
    ],
    "mat001": [
        [0.69434583787375137, 6.1436600223703855e-005, -0.71964133190793478],
        [0.0075513758697978803, 0.99994425206834781, 0.0073713100177218091],
        [0.71960175442685925, -0.010552521978912609, 0.69430673722979697],
    ],
    "GeSep08": [
        [-0.6848109, 0.26432260000000002, -0.67909319999999995],
        [-0.70638460000000003, -0.0118251, 0.70772939999999995],
        [0.17903849999999999, 0.96436180000000005, 0.1948114],
    ],
    "mat101": [
        [0.0146839, -0.0047729000000000001, -0.99988080000000001],
        [0.010890199999999999, 0.99993010000000004, -0.0046131999999999996],
        [0.99983290000000002, -0.0108212, 0.014734799999999999],
    ],
    "mat103": [
        [0.44777640324061274, -0.0011584725574786611, -0.89414470645860211],
        [0.012770794767075907, 0.99990539208093965, 0.0050999549848959203],
        [0.89405428621925565, -0.013702578870469915, 0.44774892550143103],
    ],
    "mat102": [
        [0.3019029322698577, 0.00028313402064173831, -0.95333884763451215],
        [0.01362900757190938, 0.99989671500576127, 0.0046129895742762494],
        [0.95324148342890003, -0.014385734294100679, 0.30186788473211684],
    ],
    "mat111alongx": [
        [1 / sq3, 1 / sq2, -1 / sq6],
        [1 / sq3, -1 / sq2, -1 / sq6],
        [1 / sq3, 0, 2.0 / sq6]]
        }

# dictionary of some rotations from a sequence of three elementary rotations
dict_Eul = {
    "Identity": [0.0, 0.0, 0.0],
    "misorient_0": [21.0, 1.0, 53.0],
    "misorient_1": [21.2, 1.0, 53.0],
    "misorient_2": [21.4, 1.0, 53.0],
    "misorient_3": [21.6, 1.0, 53.0],
    "misorient_4": [21.6, 1.2, 53.0],
    "misorient_5": [21.0, 1.4, 53.0],
    "misorient_6": [21.0, 1.6, 53.0],
    "EULER_1": [10.0, 52.0, 45.0],
    "EULER_2": [14.0, 2.0, 56.0],
    "EULER_3": [38.0, 85.0, 1.0],
    "EULER_4": [1.0, 1.0, 53.0],
}

# --- ---------- Example to add a new material
# Umat = dict_Rot['mat311c1']
# Dc = dict_Vect['shear4']
# Bmat = dict_Vect['543_909075']
# Id = np.eye(3)
#
# dict_Materials['mycell_s'] = ['mycell_s', [Id, Umat, Dc, Bmat], 'fcc']
# dict_Materials['mycell'] = ['mycell', [Id, Umat, Id, Bmat], 'fcc']
SAMPLETILT = 40.0

DEG = np.pi / 180.0
PI = np.pi
RotY40 = np.array([[np.cos(SAMPLETILT * DEG), 0, -np.sin(SAMPLETILT * DEG)],
                    [0, 1, 0],
                    [np.sin(SAMPLETILT * DEG), 0, np.cos(SAMPLETILT * DEG)]])
RotYm40 = np.array([[np.cos(SAMPLETILT * DEG), 0, np.sin(SAMPLETILT * DEG)],
                        [0, 1, 0],
                        [-np.sin(SAMPLETILT * DEG), 0, np.cos(SAMPLETILT * DEG)]])

# planck constant h * 2pi in 1e-16 eV.sec  unit
hbarrex1em16 = 6.58211899
# light speed : c in 1e7 m/s units
ccx1e7 = 29.9792458

E_eV_fois_lambda_nm = np.pi * 2.0 * hbarrex1em16 * ccx1e7
# print "E_eV_fois_lambda_nm = ", E_eV_fois_lambda_nm
CST_ENERGYKEV = 12.398  # keV * angstrom  in conversion formula:E (keV) = 12.398 / lambda (angstrom)

# --- ----------- cubic permutation operators
opsymlist = np.zeros((48, 9), float)

opsymlist[0, :] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])  # identity
opsymlist[1, :] = np.array([-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0])

opsymlist[2, :] = np.array([1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0])
opsymlist[3, :] = np.array([-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

opsymlist[4, :] = np.array([-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
opsymlist[5, :] = np.array([1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0])

opsymlist[6, :] = np.array([-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0])
opsymlist[7, :] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0])

opsymlist[8, :] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
opsymlist[9, :] = np.array([0.0, -1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0])

opsymlist[10, :] = np.array([0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0])
opsymlist[11, :] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0])

opsymlist[12, :] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0])
opsymlist[13, :] = np.array([0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
opsymlist[14, :] = np.array([0.0, -1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0])
opsymlist[15, :] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0])

opsymlist[16, :] = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
opsymlist[17, :] = np.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0])
opsymlist[18, :] = np.array([0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0])
opsymlist[19, :] = np.array([0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
opsymlist[20, :] = np.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
opsymlist[21, :] = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0])
opsymlist[22, :] = np.array([0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0])
opsymlist[23, :] = np.array([0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

opsymlist[24, :] = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
opsymlist[25, :] = np.array([0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0])
opsymlist[26, :] = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0])
opsymlist[27, :] = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
opsymlist[28, :] = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0])
opsymlist[29, :] = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
opsymlist[30, :] = np.array([0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
opsymlist[31, :] = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0])

opsymlist[32, :] = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
opsymlist[33, :] = np.array([0.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0])
opsymlist[34, :] = np.array([0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0])
opsymlist[35, :] = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0])
opsymlist[36, :] = np.array([0.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0])
opsymlist[37, :] = np.array([0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0])
opsymlist[38, :] = np.array([0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0])
opsymlist[39, :] = np.array([0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])

opsymlist[40, :] = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
opsymlist[41, :] = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0])
opsymlist[42, :] = np.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0])
opsymlist[43, :] = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
opsymlist[44, :] = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])
opsymlist[45, :] = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0])
opsymlist[46, :] = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0])
opsymlist[47, :] = np.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])

OpSymArray = np.reshape(opsymlist, (48, 3, 3))

# FCC slips systems  plane normal (p) and slip direction (burger b)
# rotation axis is given by cross product of p ^ b
SLIPSYSTEMS_FCC = np.array([[1, 1, 1], [0, -1, 1],
    [1, 1, 1], [-1, 0, 1],
    [1, 1, 1], [-1, 1, 0],
    [-1, 1, 1], [0, -1, 1],
    [-1, 1, 1], [1, 0, 1],
    [-1, 1, 1], [1, 1, 0],
    [1, -1, 1], [0, 1, 1],
    [1, -1, 1], [-1, 0, 1],
    [1, -1, 1], [1, 1, 0],
    [1, 1, -1], [0, 1, 1],
    [1, 1, -1], [1, 0, 1],
    [1, 1, -1], [-1, 1, 0]]).reshape((12, 2, 3))
