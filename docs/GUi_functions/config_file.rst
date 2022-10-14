========================
Config file generation
========================
Example config file below:

### config file for LaueNeuralNetwork 

## comments

[CPU]

n_cpu = 16


[GLOBAL_DIRECTORY]

prefix = _final

## directory where all training related data and results will be saved 

main_directory = C:\Users\purushot\Desktop\pattern_matching\experimental\GUIv0\latest_version


[MATERIAL]

## same material key as lauetools (see Lauetools.dictlauetools.py for complete key)

## as of now symmetry can be cubic, hexagonal, orthorhombic, tetragonal, trigonal, monoclinic, triclinic


material = In2Bi

symmetry = hexagonal

space_group = 194

#general_diffraction_rules will filter out the forbidden spots 

#(slow, keep it false, as they will be eliminated at a later stage; 

#useful if system has low RAM to eliminate some possibilities before)

general_diffraction_rules = false


## if second phase is present, else none

material1 = In_epsilon

symmetry1 = tetragonal

space_group1 = 139

general_diffraction_rules1 = false


[DETECTOR]

## path to detector calibration file (.det)

## Max and Min energy to be used for generating training dataset, as well as for calcualting matching rate

detectorfile = C:\Users\purushot\Desktop\In_JSM\calib.det

# or use as below if you have values

#detectorfile = user_input

#five detector parameters+pixelsize+image_dimensions+ccd_label

#params = 79.51900,1951.6300,1858.1500,0.3480000,0.4560000,0.03670000,4036,4032,sCMOS_16M

emax = 22

emin = 5


[TRAINING]

## classes_with_frequency_to_remove: HKL class with less appearance than specified will be ignored in output

## desired_classes_output : can be all or an integer: to limit the number of output classes

## max_HKL_index : can be auto or integer: Maximum index of HKL to build output classes

## max_nb_grains : Maximum number of grains to simulate per lauepattern

####### Material 0

classes_with_frequency_to_remove = 500

desired_classes_output = all

max_HKL_index = 5

max_nb_grains = 1

####### Material 1

## HKL class with less appearance than specified will be ignored in output

classes_with_frequency_to_remove1 = 500

desired_classes_output1 = all

max_HKL_index1 = 5

max_nb_grains1 = 1


## Max number of simulations per number of grains

## Include single crystal misorientation (1 deg) data in training

## Maximum angular distance to probe (in deg)

## step size in angular distribution to discretize (in deg)

## batch size and epochs for training


max_simulations = 500

include_small_misorientation = false

misorientation_angle = 1

angular_distance = 120

step_size = 0.1

batch_size = 50

epochs = 5


[PREDICTION]

# model_weight_file: if none, it will select by default the latest H5 weight file, else provide a specific model

# softmax_threshold_global: thresholding to limit the predicted spots search zone

# mr_threshold_global: thresholding to ignore all matricies less than the MR threshold

# cap_matchrate: any UB matrix providing MR less than this will be ignored (between 0-1)

# coeff: should be same as cap_matchrate but this is for try previous UB matrix

# coeff_overlap: coefficient to limit the overlapping between spots; if more than this, new solution will be computed

# mode_spotCycle: How to cycle through predicted spots (slow or graphmode) ##slow is more thorough but slow as the name suggests

##use_previous true for few crystal and prefered texture case, otherwise time consuming; advised for single phase alone


UB_matrix_to_detect = 2

matrix_tolerance = 0.6

matrix_tolerance1 = 0.6

cap_matchrate = 0.10

material0_limit = 1

material1_limit = 1


### no need to change the settings below unless the neural network does not give satisfactory results

mode_spotCycle = graphmode

model_weight_file = none

softmax_threshold_global = 0.85

mr_threshold_global = 0.80

coeff = 0.3

coeff_overlap = 0.3

#true for few crystal and prefered texture case, otherwise time consuming; advised for single phase alone

#true also if using single CPU mode to make indexation go faster

use_previous = false


[EXPERIMENT]

experiment_directory = user_path_to_experimental_folder

experiment_file_prefix = file_prefix_without_filenumbers

image_grid_x = 51

image_grid_y = 51


[PEAKSEARCH]

# mode: either LaueTools or skimage

# Two modes of peaksearch is possible

# For skimage, keep intenisty_threshold somewhere between 2 - 10

intensity_threshold = 100

boxsize = 15

fit_peaks_gaussian = 1

FitPixelDev = 15

mode = LaueTools


[STRAINCALCULATION]

# strain computation with multi step refinement for material0 and material1

strain_compute = true

tolerance_strain_refinement = 0.6,0.5,0.4,0.3,0.2

tolerance_strain_refinement1 = 0.6,0.5,0.4,0.3,0.2


[CALLER]

# some additional settings to change hardcoded variables in the code

residues_threshold=0.25

nb_spots_global_threshold=8

option_global = v2

#use_om_user provides the possibility to provide your own path with text file contatining orientations, 

#if true, no prediction will be done with neural network and we just index with user defined orientations

use_om_user = false

#first n number of intense spots to predict the hkl for, for many UB detection keep a high number like 1000

nb_spots_consider = 200


[DEVELOPMENT]

# could be 1 or 2 or none in case of single phase (for pretty plots) or in case of substrate present

material_phase_always_present = none

#writes MTEX input file to be treated with MTEX for advance orientation analysis

write_MTEX_file = true

