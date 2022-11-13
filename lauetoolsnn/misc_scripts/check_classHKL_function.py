# coding: utf-8
"""
Created on June 18 06:54:04 2021

Check generate_classHKL function
@author: Ravi raj purohit PURUSHOTTAM RAJ PUROHIT (purushot@esrf.fr)

"""
__author__ = "Ravi raj purohit PURUSHOTTAM RAJ PUROHIT, CRG-IF BM32 @ ESRF"

if __name__ == '__main__':     #enclosing required because of multiprocessing

    ## Import modules used for this Notebook
    import os

    # =============================================================================
    # Step 0: Define the dictionary with all parameters 
    # =============================================================================
    ## User Input dictionary with parameters
    ## In case of only one phase/material, keep same value for material_ and material1_ key
    input_params = {
                    "global_path" : r"C:\Users\purushot\Desktop\LaueNN_script",
                    "prefix" : "_2phase",                 ## prefix for the folder to be created for training dataset

                    "material_": ["GaN", "Si"],             ## same key as used in dict_LaueTools
                    "symmetry": ["hexagonal", "cubic"],           ## crystal symmetry of material_
                    "SG": [184, 227], #186                    ## Space group of material_ (None if not known)
                    "hkl_max_identify" : [6,5],        ## Maximum hkl index to classify in a Laue pattern
                    "nb_grains_per_lp_mat" : [2,1],        ## max grains to be generated in a Laue Image

                    ## hkl_max_identify : can be "auto" or integer: Maximum index of HKL to build output classes
                    
                    # =============================================================================
                    # ## Data generation settings
                    # =============================================================================
                    "grains_nb_simulate" : 500,    ## Number of orientations to generate (takes advantage of crystal symmetry)
                    "classes_with_frequency_to_remove": [100,100], ## classes_with_frequency_to_remove: HKL class with less appearance than 
                                                                            ##  specified will be ignored in output
                    "desired_classes_output": ["all","all"], ## desired_classes_output : can be all or an integer: to limit the number of output classes

                    "include_small_misorientation": False, ## to include additional data with small angle misorientation
                    "misorientation": 5, ##only used if "include_small_misorientation" is True
                    "maximum_angle_to_search":20, ## Angle of radial distribution to reconstruct the histogram (in deg)
                    "step_for_binning" : 0.1,      ## bin widht of angular radial distribution in degree
                    }

    # =============================================================================
    # Step 1 
    # =============================================================================
    ## if LaueToolsNN is properly installed
    from lauetoolsnn.utils_lauenn import get_material_detail, generate_classHKL
    # ## Get material parameters 
    # ### Generates a folder with material name and gets material unit cell parameters 
    # ### and symmetry object from the get_material_detail function
    material_= input_params["material_"][0]
    material1_= input_params["material_"][1]
    n = input_params["hkl_max_identify"][0]
    n1 = input_params["hkl_max_identify"][1]
    maximum_angle_to_search = input_params["maximum_angle_to_search"]
    step_for_binning = input_params["step_for_binning"]
    symm_ = input_params["symmetry"][0]
    symm1_ = input_params["symmetry"][1]
    SG = input_params["SG"][0]
    SG1 = input_params["SG"][1]
    
    ## read hkl information from a fit file in case too large HKLs
    manual_hkl_list=False
    if manual_hkl_list:
        import numpy as np
        temp = np.loadtxt(r"img_0000_LT_1.fit")
        hkl_array = temp[:,2:5]
        hkl_array1 = None
    else:
        hkl_array = None
        hkl_array1 = None
        
    if material_ != material1_:
        save_directory = os.path.join(input_params["global_path"],material_+"_"+material1_+input_params["prefix"])
    else:
        save_directory = os.path.join(input_params["global_path"],material_+input_params["prefix"])
    print("save directory is : "+save_directory)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    ## get unit cell parameters and other details required for simulating Laue patterns
    rules, symmetry, lattice_material,\
        crystal, SG, rules1, symmetry1,\
        lattice_material1, crystal1, SG1 = get_material_detail(material_, SG, symm_,
                                                               material1_, SG1, symm1_)
    
    # ## Generate Neural network output classes (Laue spot hkls) using the generate_classHKL function
    ## procedure for generation of GROUND TRUTH classes
    # general_diff_cond = True will eliminate the hkl index that does not 
    # satisfy the general reflection conditions, otherwise they will be eliminated in the next stage
    generate_classHKL(n, rules, lattice_material, symmetry, material_, 
                      crystal=crystal, SG=SG, general_diff_cond=False,
                      save_directory=save_directory, write_to_console=print, 
                      ang_maxx = maximum_angle_to_search, 
                      step = step_for_binning, mat_listHKl=hkl_array)