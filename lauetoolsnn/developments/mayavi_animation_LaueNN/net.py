# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 10:35:30 2021

@author: PURUSHOT

Helper module with functions
"""
import h5py
import numpy as np

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
    return layer0, layer1, layer2, layer3

##generate validation dataset to try
try:
    import LaueTools.generaltools as GT
    import LaueTools.lauecore as LT
    import LaueTools.CrystalParameters as CP
except:
    import lauetoolsnn.lauetools.generaltools as GT
    import lauetoolsnn.lauetools.lauecore as LT
    import lauetoolsnn.lauetools.CrystalParameters as CP
    
class GenData():
    def __init__(self, key_material = 'Cu', nbUBs=1, sec_mat=False, seed = 10,
                 key_material1 = 'Si', nbUBs1 = 1, seed1 = 15, ang_maxx = 20,
                 step = 0.1):
        r = 10
        detectorparameters = [79.61200, 977.8100, 932.1700, 0.4770000, 0.4470000]
        pixelsize = 0.079142
        ###########################################################
        self.img_array = np.zeros((2048,2048), dtype=np.uint8)
        app_len=0
        l_miller_ind = np.zeros((1,3))
        l_tth = np.zeros(1)
        l_chi = np.zeros(1)
        l_posx = np.zeros(1)
        l_posy = np.zeros(1)
        l_E = np.zeros(1)
        l_intensity = np.zeros(1)
        colu = []

        
        np.random.seed(seed)
        UBelemagnles = np.random.random((nbUBs,3))*360-180.

        for angle_X, angle_Y, angle_Z in UBelemagnles:
            UBmatrix = GT.fromelemangles_toMatrix([angle_X, angle_Y, angle_Z])
            grain = CP.Prepare_Grain(key_material, UBmatrix)
            s_tth, s_chi, s_miller_ind, s_posx, s_posy, s_E= LT.SimulateLaue_full_np(grain, 5, 23,
                                                                                     detectorparameters,
                                                                                     detectordiameter = pixelsize * 2048,
                                                                                     pixelsize=pixelsize,
                                                                                     removeharmonics=1)
            s_intensity = 1./s_E
            l_tth = np.hstack((l_tth, s_tth))
            l_chi = np.hstack((l_chi, s_chi))
            l_posx = np.hstack((l_posx, s_posx))
            l_posy = np.hstack((l_posy, s_posy))
            l_E = np.hstack((l_E, s_E))
            l_intensity = np.hstack((l_intensity, s_intensity))
            l_miller_ind = np.vstack((l_miller_ind, s_miller_ind))
            for _ in range(len(s_tth)):
                colu.append("k")
            Xpix = s_posx
            Ypix = s_posy
            app_len = app_len + len(s_tth)    
            for i in range(len(Xpix)):
                X_pix = int(Xpix[i])
                Y_pix = int(Ypix[i])
                self.img_array[X_pix-r:X_pix+r, Y_pix-r:Y_pix+r] = 2

        if sec_mat:
            np.random.seed(seed1)
            UBelemagnles = np.random.random((nbUBs1,3))*360-180.
            for angle_X, angle_Y, angle_Z in UBelemagnles:
                UBmatrix = GT.fromelemangles_toMatrix([angle_X, angle_Y, angle_Z])
                grain = CP.Prepare_Grain(key_material1, UBmatrix)
                s_tth, s_chi, s_miller_ind, s_posx, s_posy, s_E= LT.SimulateLaue_full_np(grain, 5, 23,
                                                                                         detectorparameters,
                                                                                         detectordiameter = pixelsize * 2048,
                                                                                         pixelsize=pixelsize,
                                                                                         removeharmonics=1)
                s_intensity = 1./s_E
                l_tth = np.hstack((l_tth, s_tth))
                l_chi = np.hstack((l_chi, s_chi))
                l_posx = np.hstack((l_posx, s_posx))
                l_posy = np.hstack((l_posy, s_posy))
                l_E = np.hstack((l_E, s_E))
                l_intensity = np.hstack((l_intensity, s_intensity))
                l_miller_ind = np.vstack((l_miller_ind, s_miller_ind))
                for _ in range(len(s_tth)):
                    colu.append("r")
                Xpix = s_posx
                Ypix = s_posy
                app_len = app_len + len(s_tth)    
                for i in range(len(Xpix)):
                    X_pix = int(Xpix[i])
                    Y_pix = int(Ypix[i])
                    self.img_array[X_pix-r:X_pix+r, Y_pix-r:Y_pix+r] = 1
        ### RUN ONLY ONCE 
        l_tth = np.delete(l_tth, 0, axis=0)
        l_chi = np.delete(l_chi, 0, axis=0)
        l_posx = np.delete(l_posx, 0, axis=0)
        l_posy = np.delete(l_posy, 0, axis=0)
        l_E = np.delete(l_E, 0, axis=0)
        l_intensity = np.delete(l_intensity, 0, axis=0)
        l_miller_ind = np.delete(l_miller_ind, 0, axis=0)

        print("Number of spots: " + str(app_len))
        allspots_the_chi = np.transpose(np.array([l_tth/2., l_chi]))
        tabledistancerandom = np.transpose(GT.calculdist_from_thetachi(allspots_the_chi, allspots_the_chi))
        tab_angulardist = np.transpose(GT.calculdist_from_thetachi(allspots_the_chi, allspots_the_chi))
        np.putmask(tab_angulardist, np.abs(tab_angulardist) < 0.001, 400)
        
        spots_in_center = np.arange(0,len(l_tth))
        angbins = np.arange(0, ang_maxx+step, step)
        codebars_all = []
        for i in spots_in_center:
            spotangles = tabledistancerandom[i]
            spotangles = np.delete(spotangles, i)# removing the self distance
            codebars = np.histogram(spotangles, bins=angbins)[0]
            ## normalize the same way as training data
            max_codebars = np.max(codebars)
            codebars = codebars/ max_codebars
            codebars_all.append(codebars)
        ## reshape for the model to predict all spots at once
        codebars = np.array(codebars_all)
        ## reshape for the model to predict all spots at once
        self.codebars = np.array(codebars)
        self.l_tth = l_tth
        self.l_chi = l_chi
        self.l_miller_ind = l_miller_ind
        self.l_intensity = l_intensity
        self.tab_angulardist = tab_angulardist
    
    def get_data(self):
        return self.img_array, self.codebars, self.l_tth, self.l_chi, self.l_miller_ind, self.l_intensity, self.tab_angulardist
