# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 15:58:37 2022

@author: PURUSHOT

Check prediction from the model

"""
# else import from a path where LaueToolsNN files are
import sys
sys.path.append(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn")
from lauetools import readmccd as RMCCD
from lauetools import LaueGeometry as Lgeo
from lauetools import generaltools as GT


import numpy as np

exp_data = r"C:\Users\purushot\Desktop\Cu_pads_Bassel\W02_SQ2E2\RT\map_RT_0000.tif"





### Max space = space betzeen pixles
peak_XY = RMCCD.PeakSearch(
                            exp_data,
                            stackimageindex = -1,
                            CCDLabel="sCMOS",
                            NumberMaxofFits=5000,
                            PixelNearRadius=10,
                            removeedge=2,
                            IntensityThreshold=70,
                            local_maxima_search_method=0,
                            boxsize=15,
                            position_definition=1,
                            verbose=0,
                            fit_peaks_gaussian=1,
                            xtol=0.001,                
                            FitPixelDev=15,
                            return_histo=0,
                            # Saturation_value=1e10,  # to be merged in CCDLabel
                            # Saturation_value_flatpeak=1e10,
                            MinIntensity=0,
                            PeakSizeRange=(0.65,200),
                            write_execution_time=1,
                            Data_for_localMaxima = "auto_background",
                            formulaexpression="A-B",
                            Remove_BlackListedPeaks_fromfile=None,
                            reject_negative_baseline=True,
                            Fit_with_Data_for_localMaxima=False,
                            maxPixelDistanceRejection=15.0,
                            )
peak_XY = peak_XY[0]#[:,:2] ##[2] Integer peak lists

detectorparameters = [79.583, 976.202, 931.883, 0.4411, 0.3921]
pixelsize = 0.0734
twicetheta, chi = Lgeo.calc_uflab(peak_XY[:,0], peak_XY[:,1], detectorparameters,
                                    returnAngles=1,
                                    pixelsize=pixelsize,
                                    kf_direction='Z>0')
data_theta, data_chi = twicetheta/2., chi

sorted_data = np.transpose(np.array([data_theta, data_chi]))
tabledistancerandom = np.transpose(GT.calculdist_from_thetachi(sorted_data, sorted_data))


nb_spots_consider = 100

spots_in_center = np.arange(0,len(data_theta))
spots_in_center = spots_in_center[:nb_spots_consider]

angbins = np.load(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\multi_mat_LaueNN\Cu_Si_CNN_DNN\MOD_grain_classhkl_angbin.npz")["arr_1"]

codebars_all = []
for i in spots_in_center:
    spotangles = tabledistancerandom[i]
    spotangles = np.delete(spotangles, i)# removing the self distance
    codebars = np.histogram(spotangles, bins=angbins)[0]
    ## normalize the same way as training data
    max_codebars = np.max(codebars)
    codebars = codebars/ max_codebars
    codebars_all.append(codebars)
    
codebars = np.array(codebars_all)


from NNmodels import read_hdf5, predict_CNN_DNN, predict_DNN

##CNN
classhkl = np.load(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\multi_mat_LaueNN\Cu_Si_CNN_DNN\MOD_grain_classhkl_angbin.npz")["arr_0"]
##Model weights 
model_weights = read_hdf5(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\multi_mat_LaueNN\Cu_Si_CNN_DNN\model_Cu_Si.h5")
model_key = list(model_weights.keys())
prediction = predict_CNN_DNN(codebars, model_weights, model_key)
max_pred = np.max(prediction, axis = 1)
class_predicted = np.argmax(prediction, axis = 1)
predicted_hkl = classhkl[class_predicted]
predicted_hkl = predicted_hkl.astype(int)

##DNN
classhkl1 = np.load(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\multi_mat_LaueNN\Cu_Si_DNN\MOD_grain_classhkl_angbin.npz")["arr_0"]
model_weights1 = read_hdf5(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\multi_mat_LaueNN\Cu_Si_DNN\model_Cu_Si.h5")
model_key1 = list(model_weights1.keys())
prediction1 = predict_DNN(codebars, model_weights1, model_key1)
max_pred1 = np.max(prediction1, axis = 1)
class_predicted1 = np.argmax(prediction1, axis = 1)
predicted_hkl1= classhkl1[class_predicted1]
predicted_hkl1 = predicted_hkl1.astype(int)


diff = np.sum(predicted_hkl - predicted_hkl1, axis=1)







