# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 13:14:53 2022

@author: PURUSHOT

Add user material and extinction rule to the json object

"""
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
    
    parser = argparse.ArgumentParser(description="Calculates the maxHKL available on the Detector with polycrystal Laue Simulation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--name", required=True, help="User string for the material")
    parser.add_argument("-dd", "--detectordistance", required=False, type=float, help="detector distance from sample in mm")
    parser.add_argument("-nb", "--numbergrains", required=False, type=int, help="number of grains used in simulate Laue")
    parser.add_argument("-c", "--category", required=False, nargs="*", help="detector distance from sample in mm")

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
                                            detectorparameters=[dd, 1000, 1000, 0.3570000, 0.4370000], 
                                            pixelsize=0.0734,
                                            dim1=2048,
                                            dim2=2048,
                                            removeharmonics=1)
        for i, txt_hkl in enumerate(hkl_sol):
            hkl_sol_all.append(_round_indices(txt_hkl[:3]))
        nbtestspots = nbtestspots + len(hkl_sol)
    hkl_sol_all = np.array(hkl_sol_all)
    # print(hkl_sol_all[::100])
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
    