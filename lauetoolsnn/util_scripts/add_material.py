# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 13:14:53 2022

@author: PURUSHOT

Add user material and extinction rule to the json object

NEW: query pymatgen rest API for lattice parameters and spacegroups
"""
__author__ = "Ravi raj purohit PURUSHOTTAM RAJ PUROHIT, CRG-IF BM32 @ ESRF"

import warnings
warnings.filterwarnings('ignore')

def set_laue_geometry():
    ## sets laue geometry
    ## Top/side reflection
    ## Transmission/ back reflection
    import json
    import argparse
    from lauetoolsnn.utils_lauenn import resource_path
    parser = argparse.ArgumentParser(description="Set Laue mode geometry; either Z>0 (top reflection mode); X>0 (Transmission mode); X<0 (back reflection mode)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--mode", required=True, help="User string for the mode of Laue geometry (provide input with double quotes like string)")
    args = parser.parse_args()
    config = vars(args)
    print(config)
    mode = config["mode"]
    filepath = resource_path('xxxx')
    filepathmat = filepath[:-4] + "lauetools//" + 'detector_geometry.json'
    print(filepathmat)
    ## Load the json of dict_geometry
    with open(filepathmat,'r') as f:
        dict_geometry = json.load(f)
    ## Modify/ADD the dictionary values to add new entries
    dict_geometry["default"] = mode
    ## dump the json back with new values
    with open(filepathmat, 'w') as fp:
        json.dump(dict_geometry, fp)
    print("Laue geometry changed to ", dict_geometry[mode])

def add_detector():
    ## sets laue geometry
    ## Top/side reflection
    ## Transmission/ back reflection
    import json
    import argparse
    from lauetoolsnn.utils_lauenn import resource_path
    parser = argparse.ArgumentParser(description="add a new detector to the dictionary",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--name", required=True, help="User string for the detector")
    parser.add_argument("-dx", "--dimx", required=True, help="X dimension of detector in pixels")
    parser.add_argument("-dy", "--dimy", required=True, help="Y dimension of detector in pixels")
    parser.add_argument("-p", "--pix", required=True, help="Detector pixel size")
    parser.add_argument("-f", "--format", required=True, help="Image extension of the detector files")
    
    args = parser.parse_args()
    config = vars(args)
    print(config)
    
    name = config["name"]
    dimx = int(config["dimx"])
    dimy = int(config["dimy"])
    pxsize = float(config["pix"])
    fformat = config["format"]
    
    filepath = resource_path('xxxx')
    filepathmat = filepath[:-4] + "lauetools//" + 'detector_config.json'
    print(filepathmat)
    ## Load the json of dict_geometry
    with open(filepathmat,'r') as f:
        dict_detector = json.load(f)
    ## Modify/ADD the dictionary values to add new entries
    dict_detector[name] = [[dimx, dimy], pxsize, 65535, "no", 4096, "uint16", "user defined detector", fformat]
    ## dump the json back with new values
    with open(filepathmat, 'w') as fp:
        json.dump(dict_detector, fp)
    print("New detector entry added ", dict_detector[name])
    
def example_scripts():
    ## Given a path from the users
    ## Transfer the example scripts there
    ## And additionally if download is true, then download a test dataset
    
    from lauetoolsnn.utils_lauenn import resource_path
    import os
    filepath = resource_path('end_to_end_scripts')
    file1 = os.path.join(filepath,"LaueNN_pyScript_1_or_2phase.py")
    file2 = os.path.join(filepath,"LaueNN_pyScript_3_or_more_phase.py")
    file3 = os.path.join(filepath,"LaueNN_pyScript_1_or_2phase.ipynb")
    file4 = os.path.join(filepath,"LaueNN_pyScript_3_or_more_phase.ipynb")
    file5 = os.path.join(filepath,"example_MTEX.m")
    file6 = os.path.join(filepath,"hdf5_plots.ipynb")
    file7 = os.path.join(filepath,"interactive_plots.ipynb")
    file8 = os.path.join(filepath,"plots_postprocess.py")
    
    current_path = os.getcwd()
    save_directory = os.path.join(current_path, "LaueNN_script")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    import shutil
    shutil.copy(file1, save_directory)
    shutil.copy(file2, save_directory)
    shutil.copy(file3, save_directory)
    shutil.copy(file4, save_directory)
    shutil.copy(file5, save_directory)
    shutil.copy(file6, save_directory)
    shutil.copy(file7, save_directory)
    shutil.copy(file8, save_directory)
    print("Files copied to "+save_directory+" successfully")
    
def pymatgen_query():
    from pymatgen.ext.matproj import MPRester
    import argparse
    import numpy as np
    parser = argparse.ArgumentParser(description="Query the PYMATGEN rest API for lattice parameters and spacegroup",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-e", "--elements", required=True, nargs="*", help="User string for the elements present in the material; provide a list of elements space seperated")
    args = parser.parse_args()
    config = vars(args)

    cat = config["elements"]
    cat = [str(i) for i in cat[0].split(' ')]
    print("Desired Elements", cat)

    ## MY api key to access materialsproject database
    pymatgen_api_key = "87iNakVJgbTSqn3A"

    def spacegroup_extractor(entry_id):
        """Extracts space group for a compound. Is the target property for this project
        :parameter entry_id: A string which is the id of the crystals in Materials Project
        :returns space group symbol as a string
        """
        properties = ['spacegroup']
        with MPRester(pymatgen_api_key) as m:
            results = m.query(entry_id, properties=properties)
        return results[0]['spacegroup']['symbol']

    def lattice_extractor(entry_id):
        """Extracts the lattice parameters for a given crystal
        :parameter entry_id: A string which is the id of the crystals in Materials Project
        :returns list of 6 lattice parameters as float
        """
        properties = ['initial_structure']
        with MPRester(pymatgen_api_key) as m:
            results = m.query(entry_id, properties=properties)
        lattice_parameters = results[0]['initial_structure'].as_dict()['lattice']
        return [lattice_parameters['a'], lattice_parameters['b'], lattice_parameters['c'],
                lattice_parameters['alpha'], lattice_parameters['beta'], lattice_parameters['gamma']]

    with MPRester(pymatgen_api_key) as m:
        mp_entries = m.get_entries_in_chemsys(cat)

    print("Total number found on PYMATGEN database:", len(mp_entries))
    print()
    data_dict = {}
    for index, element in enumerate(mp_entries):
        lattices = np.round(lattice_extractor(element.entry_id), 4)
        sg = spacegroup_extractor(element.entry_id)
        data_dict[element.entry_id] = {'lattice': lattices, 
                                       'spacegroup': sg}
        
        print("Entry", index+1)
        print(element.entry_id)
        print("Composition", element.composition)
        print("Lattice parameters", lattices)
        print("Space group", sg)
        print("***************************************")

def querymat():
    import argparse
    from lauetoolsnn.utils_lauenn import resource_path
    import json
    
    parser = argparse.ArgumentParser(description="Query a user material to laueNN",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--name", required=True, help="User string for the material")
    args = parser.parse_args()
    config = vars(args)
    
    mat_name = config["name"]
    
    filepath = resource_path('xxxx')
    filepathmat = filepath[:-4] + "lauetools//" + 'material.json'
    filepathext = filepath[:-4] + "lauetools//" + 'extinction.json'
    
    print(filepathext)
    print(filepathmat)
    # ## If material key does not exist in Lauetoolsnn dictionary
    # ## you can modify its JSON materials file before import or starting analysis
    
    ## Load the json of material and extinctions
    with open(filepathmat,'r') as f:
        dict_Materials = json.load(f)
    with open(filepathext,'r') as f:
        extinction_json = json.load(f)
        
    ## Modify/ADD the dictionary values to add new entries
    error_free_mat = False
    error_free_ext = False
    try:
        material_queried = dict_Materials[mat_name]
        error_free_mat = True
        try:
            _ = extinction_json[material_queried[2]]
            error_free_ext = True
        except:
            error_free_ext = False
            print("Extinction does not exist in the LaueNN library, please add it with lauenn_addmat command")
    except:
        error_free_mat = False
        print("Material does not exist in the LaueNN library, please add it with lauenn_addmat command")
    if error_free_mat:
        print("Material found in LaueNN library")
        print(material_queried)

    if error_free_ext:
        print("Material Extinction found in LaueNN library")
        print(extinction_json[material_queried[2]])

def start():
    import argparse
    from lauetoolsnn.utils_lauenn import resource_path
    import json
    
    parser = argparse.ArgumentParser(description="Add user material to laueNN",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--name", required=True, help="User string for the material")
    parser.add_argument("-l", "--lattice", required=True, nargs=6, type=float, help="Unit cell lattice parameters of unit cell in the format a b c alpha beta gamma")
    parser.add_argument("-e", "--extinction", required=True, help="String for material extinction; if want space group rules provide the spacegroup number")
    args = parser.parse_args()
    config = vars(args)
    print(config)
    
    mat_name = config["name"]
    lat = config["lattice"]
    ext = str(config["extinction"])
    
    
    filepath = resource_path('xxxx')
    filepathmat = filepath[:-4] + "lauetools//" + 'material.json'
    filepathext = filepath[:-4] + "lauetools//" + 'extinction.json'
    
    print(filepathext)
    print(filepathmat)
    # ## If material key does not exist in Lauetoolsnn dictionary
    # ## you can modify its JSON materials file before import or starting analysis
    
    ## Load the json of material and extinctions
    with open(filepathmat,'r') as f:
        dict_Materials = json.load(f)
    with open(filepathext,'r') as f:
        extinction_json = json.load(f)
        
    ## Modify/ADD the dictionary values to add new entries
    dict_Materials[mat_name] = [mat_name, lat, ext]
    extinction_json[ext] = ext

    ## dump the json back with new values
    with open(filepathmat, 'w') as fp:
        json.dump(dict_Materials, fp)
    with open(filepathext, 'w') as fp:
        json.dump(extinction_json, fp)
        
    print("New material successfully added to the database with the following string")
    print(mat_name, lat, ext)
    ## print dtype also


def query_hklmax():
    import argparse
    from lauetoolsnn.utils_lauenn import simulatemultiplepatterns, _round_indices
    import numpy as np
    import collections

    def prepare_LP( nbgrains, nbgrains1, material_, material1_, verbose, seed=None, sortintensity=False,
                   detectorparameters=None, pixelsize=None, dim1=2048, dim2=2048, removeharmonics=1):
        s_tth, s_chi, s_miller_ind, s_posx, s_posy, \
                                        s_intensity, _, _ = simulatemultiplepatterns(nbgrains, nbgrains1, seed=seed, 
                                                                                    key_material=material_,
                                                                                    key_material1=material1_,
                                                                                    detectorparameters=detectorparameters,
                                                                                    pixelsize=pixelsize,
                                                                                    emin=5,
                                                                                    emax=23,
                                                                                    sortintensity=sortintensity, 
                                                                                    dim1=dim1,dim2=dim2,
                                                                                    removeharmonics=removeharmonics,
                                                                                    misorientation_angle=1,
                                                                                    phase_always_present=None)
        hkl_sol = s_miller_ind
        return hkl_sol, s_posx, s_posy, s_intensity, s_tth, s_chi
    
    parser = argparse.ArgumentParser(description="Calculates the maxHKL available on the Detector with polycrystal Laue Simulation; useful for max_HKL_index parameter estimation in the config file",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--name", required=True, help="User string for the material")
    parser.add_argument("-dd", "--detectordistance", required=False, type=float, help="detector distance from sample in mm")
    # parser.add_argument("-sg", "--spacegroup", required=False, type=float, help="detector distance from sample in mm")
    parser.add_argument("-nb", "--numbergrains", required=False, type=int, help="number of grains used in simulate Laue")
    parser.add_argument("-c", "--category", required=False, nargs="*", help="string of categories to sort the hkl classes into; for example '5 10 15' will provide count of hkls less than index 5, less than index 10 and greater than 5, and less than index 15 and greater than 10")

    args = parser.parse_args()
    config = vars(args)
    print(config)
    
    material_ = config["name"]
    try:
        dd = config["detectordistance"]
        if dd == None:
            print("detector distance not defined, using default value of 79mm")
            dd=79
    except:
        print("detector distance not defined, using default value of 79mm")
        dd =79
    
    try:
        cat = config["category"]
        if len(cat) > 1:
            cat = [float(i.strip("'")) for i in cat]
        else:
            cat = [float(i) for i in cat[0].split(' ')]
        if cat == None:
            print("hkl category not defined, using default index of 5s")
            cat = [5, 10, 15, 20, 25, 30]
    except:
        print("hkl category not defined, using default index of 5s")
        cat = [5, 10, 15, 20, 25, 30]
    
    try:
        nbgrains = int(config["numbergrains"])
        if nbgrains == None:
            print("Number of grains not defined, using default value of 5grains per LP")
            nbgrains=5
    except:
        print("Number of grains not defined, using default value of 5grains per LP")
        nbgrains =5

    nbtestspots = 0
    hkl_sol_all = []
    verbose=0
    for _ in range(10):
        seednumber = np.random.randint(1e6)
        hkl_sol,  _, _, _, _, _ = prepare_LP(nbgrains, 0,
                                            material_,
                                            None,
                                            verbose,
                                            seed=seednumber,
                                            detectorparameters=[dd, 1000, 1000, 0.357, 0.437], 
                                            pixelsize=0.0734,
                                            dim1=2048,
                                            dim2=2048,
                                            removeharmonics=1)
        for i, txt_hkl in enumerate(hkl_sol):
            hkl_sol_all.append(_round_indices(txt_hkl[:3]))
        nbtestspots = nbtestspots + len(hkl_sol)
    hkl_sol_all = np.array(hkl_sol_all)
    indexes = []
    for i in range(len(hkl_sol_all)):
        for k, j in enumerate(cat):
            if k == 0:
                if np.max(np.abs(hkl_sol_all[i,:])) <= j:
                    indexes.append(j)
            else:
                if np.max(np.abs(hkl_sol_all[i,:])) <= j and np.max(np.abs(hkl_sol_all[i,:])) > cat[k-1]:
                    indexes.append(j)
    most_common0 = collections.Counter(indexes).most_common()
    print("Most common occurances of the hkl indexes are as follows")
    print(most_common0)

    print("Total spots created for calculating HKL bounds:"+str(nbtestspots))
    print("Max HKL index:"+str(np.max(hkl_sol_all)))
    print("Min HKL index:"+str(np.min(hkl_sol_all)))  
