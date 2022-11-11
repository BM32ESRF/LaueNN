# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:54:36 2022

@author: PURUSHOT

Lets treat the Laue patterns as multi-blob problem with open cv

Specifically for Laue patterns that has multi component (load FIJI segmented images)
"""
import numpy as np
import glob
import re
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import itertools
import matplotlib.pyplot as plt

try:
    from lauetools import imageprocessing as ImProc
    from lauetools import LaueGeometry as Lgeo
    from lauetools import IOLaueTools as IOLT
    from lauetools import CrystalParameters as CP
    from lauetools import lauecore as LT
    from lauetools import readmccd as RMCCD

except:
    import lauetoolsnn.lauetools.imageprocessing as ImProc
    import lauetoolsnn.lauetools.LaueGeometry as Lgeo
    import lauetoolsnn.lauetools.IOLaueTools as IOLT
    import lauetoolsnn.lauetools.CrystalParameters as CP
    import lauetoolsnn.lauetools.lauecore as LT
    import lauetoolsnn.lauetools.readmccd as RMCCD

from skimage.feature import blob_log #, blob_dog
import cv2
from skimage import morphology


def process_images(filename, file_directory):
    
    ##Idea is to get the XY pixel from the oreintation matrix and extract all
    ##closest peaks to a given ori mat.
    # original_data = plt.imread(filename)
    
    data_8bit_raw = plt.imread(filename)
    framedim = data_8bit_raw.shape
    CCDLabel = "sCMOS_16M"

    backgroundimage = ImProc.compute_autobackground_image(data_8bit_raw, boxsizefilter=10)
    # basic substraction
    data_8bit_raw_bg = ImProc.computefilteredimage(data_8bit_raw, backgroundimage, 
                                                CCDLabel, usemask=True, formulaexpression="A-B")
    
    data_8bit_raw = np.copy(data_8bit_raw_bg)
    ## simple thresholding
    bg_threshold = 100
    data_8bit_raw[data_8bit_raw<bg_threshold] = 0
    data_8bit_raw[data_8bit_raw>0] = 255
    
    ### Grayscale image (0 to 255)
    data_8bit_raw = data_8bit_raw.astype(np.uint8)
        
    detectorparameters = [79.51900, 1951.6300, 1858.1500, 0.3480000, 0.4560000]
    pixelsize = 0.0367
    material_ = "ZrO2_1250C"
    UBmatrix = np.array([[-0.35199794, 0.21770188, 0.91422899], 
                         [-0.91543419, 0.09662939, -0.37938011], 
                         [-0.16905938, -0.96440191, 0.16925999]])
    dim1, dim2 = framedim
    grain = CP.Prepare_Grain(material_, UBmatrix)
    _, _, _, s_posx, s_posy, s_E = LT.SimulateLaue_full_np(grain, 5, 23,
                                                            detectorparameters,
                                                            pixelsize=pixelsize,
                                                            dim=(dim1, dim2),
                                                            detectordiameter=pixelsize * dim1,
                                                            removeharmonics=1)
    
    s_intensity = 1./s_E
    indsort = np.argsort(s_intensity)[::-1]
    s_posx=np.take(s_posx, indsort)
    s_posy=np.take(s_posy, indsort)
    s_intensity=np.take(s_intensity, indsort)
    
    blobs_log_all_x= []#, blobs_dog_all_x = [], []
    blobs_log_all_y= []#, blobs_dog_all_y = [], []
    blobs_log_all_ecc= []#, blobs_dog_all_ecc = [], []
    ## Loop through each peak and only extract blobs given by the theoretical position
    cnt = -1
    big_image = np.zeros((framedim))
    for xpos, ypos in zip(s_posx, s_posy):
        cnt += 1
        ## +- 30 pixels on either side of the theo peaks
        lim = 100
        ytheo_min = int(xpos) - lim
        ytheo_max = int(xpos) + lim
        xtheo_min = int(ypos) - lim
        xtheo_max = int(ypos) + lim
    
        if xtheo_min < 0:
            xtheo_min = 0
        if xtheo_max > dim1:
            xtheo_max = dim1
        if xtheo_max < 0:
            xtheo_max = lim
    
        if ytheo_min < 0:
            ytheo_min = 0
        if ytheo_max > dim2:
            ytheo_max = dim2
        if ytheo_max < 0:
            ytheo_max = lim
    
        crop_data_8bit_raw = data_8bit_raw[xtheo_min:xtheo_max, ytheo_min:ytheo_max]

        crop_data_8bit_raw_process = morphology.remove_small_objects(crop_data_8bit_raw.astype(bool), 
                                                             min_size=40, connectivity=2).astype(int)
        
        crop_data_8bit_raw = crop_data_8bit_raw_process.astype(np.uint8)
        crop_data_8bit_raw[crop_data_8bit_raw>0] = 255

        kernel = np.ones((3,3),np.uint8)
        thresh = cv2.morphologyEx(crop_data_8bit_raw, cv2.MORPH_OPEN, kernel, iterations = 2)

        kernel = np.ones((2,2),np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations = 1)

        if np.all(thresh==0):
            continue     
        
        ### make one big image with patches
        big_image[xtheo_min:xtheo_max, ytheo_min:ytheo_max] = data_8bit_raw_bg[xtheo_min:xtheo_max, ytheo_min:ytheo_max]
        # big_image[xtheo_min:xtheo_max, ytheo_min:ytheo_max] = thresh

        minsigma, maxsigma = 5, 15
        threshold_int = 0.01
        #Lapacian of gaussian
        blobs_log = blob_log(thresh, min_sigma=minsigma, max_sigma=maxsigma, 
                             num_sigma=10, threshold=threshold_int)# Compute radii in the 3rd column.
        blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)
        
        #difference of gaussian
        # blobs_dog = blob_dog(thresh, min_sigma=minsigma, max_sigma=maxsigma, threshold=threshold_int)
        # blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)
        
        blobs_log_all_x.append(xtheo_min+blobs_log[:,1])
        # blobs_dog_all_x.append(xtheo_min+blobs_dog[:,1])
        
        blobs_log_all_y.append(ytheo_min+blobs_log[:,0])
        # blobs_dog_all_y.append(ytheo_min+blobs_dog[:,0])
        
        blobs_log_all_ecc.append(blobs_log[:,2])
        # blobs_dog_all_ecc.append(blobs_dog[:,2])
        
        
    blobs_log_all_x = np.concatenate(blobs_log_all_x, axis=0)
    blobs_log_all_y = np.concatenate(blobs_log_all_y, axis=0)
    blobs_log_all_ecc = np.concatenate(blobs_log_all_ecc, axis=0)
    
    # blobs_dog_all_x = np.concatenate(blobs_dog_all_x, axis=0)
    # blobs_dog_all_y = np.concatenate(blobs_dog_all_y, axis=0)
    # blobs_dog_all_ecc = np.concatenate(blobs_dog_all_ecc, axis=0)

    blobs_log = np.column_stack((blobs_log_all_x, blobs_log_all_y, blobs_log_all_ecc))
    # blobs_dog = np.column_stack((blobs_dog_all_x, blobs_dog_all_y, blobs_dog_all_ecc))            
    
    ## delete small circles
    val_del = np.sqrt(2)+0.1
    ind_del_log = np.where(blobs_log[:,2]<val_del)[0]
    # ind_del_dog = np.where(blobs_log[:,2]<val_del)[0]
    
    blobs_log = np.delete(blobs_log, ind_del_log, axis=0)
    # blobs_dog = np.delete(blobs_dog, ind_del_dog, axis=0)
    
    # =============================================================================
    #   Write Cor file for lauetools      
    # =============================================================================
    nbpeaks, _ = blobs_log.shape    
    peak_X_nonfit = blobs_log[:,1]
    peak_Y_nonfit = blobs_log[:,0]
    peak_I_nonfit = blobs_log[:, 2] #np.ones(nbpeaks) * 255.

    twicetheta, chi = Lgeo.calc_uflab(peak_X_nonfit, peak_Y_nonfit, detectorparameters,
                                        returnAngles=1,
                                        pixelsize=pixelsize,
                                        kf_direction='Z>0')
    dict_dp={}
    dict_dp['kf_direction']='Z>0'
    dict_dp['detectorparameters']=detectorparameters
    dict_dp['detectordistance']=detectorparameters[0]
    dict_dp['detectordiameter']=pixelsize*framedim[0]
    dict_dp['pixelsize']=pixelsize
    dict_dp['dim']=framedim
    dict_dp['peakX']=peak_X_nonfit
    dict_dp['peakY']=peak_Y_nonfit
    dict_dp['intensity']=peak_I_nonfit
    CCDcalib = {"CCDLabel":"sCMOS_16M",
                "dd":detectorparameters[0], 
                "xcen":detectorparameters[1], 
                "ycen":detectorparameters[2], 
                "xbet":detectorparameters[3], 
                "xgam":detectorparameters[4],
                "pixelsize": pixelsize}
    path = os.path.normpath(filename)
    IOLT.writefile_cor(file_directory+"//LOG_nofit//"+path.split(os.sep)[-1].split(".")[0], twicetheta, 
                        chi, peak_X_nonfit, peak_Y_nonfit, peak_I_nonfit,
                        param=CCDcalib, sortedexit=1)
    # =============================================================================
    #     FIT PEAKS AGAIN WITH PEAK_SEARCH
    # =============================================================================
    pixel_deviation = 15 #50
    boxsize = 10
    
    type_of_function = "gaussian"
    position_start = "max"

    peaklist = np.vstack((peak_X_nonfit, peak_Y_nonfit)).T
    peak_dataarray = RMCCD.fitoneimage_manypeaks(filename,
                                                peaklist,
                                                boxsize=boxsize,
                                                stackimageindex=-1,
                                                CCDLabel=CCDLabel,
                                                dirname=None,
                                                position_start=position_start,
                                                type_of_function=type_of_function,
                                                xtol=0.00001,
                                                FitPixelDev=pixel_deviation,
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
    
    isorted = peak_dataarray[0]
    peak_X, peak_Y, peak_I = isorted[:,0], isorted[:,1], isorted[:,2]
    twicetheta, chi = Lgeo.calc_uflab(peak_X, peak_Y, detectorparameters,
                                        returnAngles=1,
                                        pixelsize=pixelsize,
                                        kf_direction='Z>0')
    
    dict_dp={}
    dict_dp['kf_direction']='Z>0'
    dict_dp['detectorparameters']=detectorparameters
    dict_dp['detectordistance']=detectorparameters[0]
    dict_dp['detectordiameter']=pixelsize*framedim[0]
    dict_dp['pixelsize']=pixelsize
    dict_dp['dim']=framedim
    dict_dp['peakX']=peak_X
    dict_dp['peakY']=peak_Y
    dict_dp['intensity']=peak_I
    CCDcalib = {"CCDLabel":"sCMOS_16M",
                "dd":detectorparameters[0], 
                "xcen":detectorparameters[1], 
                "ycen":detectorparameters[2], 
                "xbet":detectorparameters[3], 
                "xgam":detectorparameters[4],
                "pixelsize": pixelsize}
    path = os.path.normpath(filename)
    IOLT.writefile_cor(file_directory+"//LT_LOG_ROIs//"+path.split(os.sep)[-1].split(".")[0], twicetheta, 
                        chi, peak_X, peak_Y, peak_I,
                        param=CCDcalib, sortedexit=1)


    # =============================================================================
    #     FIT PEAKS AGAIN WITH PEAK_SEARCH
    # =============================================================================
    ## specific intensity threshold based on beam condition
    intensity_threshold = 100 #85
    pixel_deviation = 15
    boxsize = 10
    
    type_of_function = "gaussian"
    position_start = "max"
    
    peaklist, Ipixmax = ImProc.LocalMaxima_ShiftArrays(big_image,
                                                        framedim=framedim,
                                                        IntensityThreshold=intensity_threshold,
                                                        Saturation_value=65535,
                                                        boxsize_for_probing_minimal_value_background=boxsize,  # 30
                                                        pixeldistance_remove_duplicates=5,  # 25
                                                        nb_of_shift=boxsize)  # 25
    
    peak_dataarray = RMCCD.fitoneimage_manypeaks(filename,
                                                peaklist,
                                                boxsize=boxsize,
                                                stackimageindex=-1,
                                                CCDLabel=CCDLabel,
                                                dirname=None,
                                                position_start=position_start,
                                                type_of_function=type_of_function,
                                                xtol=0.00001,
                                                FitPixelDev=pixel_deviation,
                                                Ipixmax=Ipixmax,
                                                MaxIntensity=65535,
                                                MinIntensity=0,
                                                PeakSizeRange=(0, 200),
                                                verbose=0,
                                                position_definition=1,
                                                NumberMaxofFits=5000,
                                                ComputeIpixmax=True,
                                                use_data_corrected=None,
                                                reject_negative_baseline=True)
    
    isorted = peak_dataarray[0]
    peak_X, peak_Y, peak_I = isorted[:,0], isorted[:,1], isorted[:,2]
    twicetheta, chi = Lgeo.calc_uflab(peak_X, peak_Y, detectorparameters,
                                        returnAngles=1,
                                        pixelsize=pixelsize,
                                        kf_direction='Z>0')
    
    dict_dp={}
    dict_dp['kf_direction']='Z>0'
    dict_dp['detectorparameters']=detectorparameters
    dict_dp['detectordistance']=detectorparameters[0]
    dict_dp['detectordiameter']=pixelsize*framedim[0]
    dict_dp['pixelsize']=pixelsize
    dict_dp['dim']=framedim
    dict_dp['peakX']=peak_X
    dict_dp['peakY']=peak_Y
    dict_dp['intensity']=peak_I
    CCDcalib = {"CCDLabel":"sCMOS_16M",
                "dd":detectorparameters[0], 
                "xcen":detectorparameters[1], 
                "ycen":detectorparameters[2], 
                "xbet":detectorparameters[3], 
                "xgam":detectorparameters[4],
                "pixelsize": pixelsize}
    path = os.path.normpath(filename)
    IOLT.writefile_cor(file_directory+"//LT_ROIs//"+path.split(os.sep)[-1].split(".")[0], twicetheta, 
                        chi, peak_X, peak_Y, peak_I,
                        param=CCDcalib, sortedexit=1)

def process_imagesv1(filename, file_directory, dat_flag=False, cor_flag=True):

    data_8bit = plt.imread(filename)
    framedim = data_8bit.shape
    ### Grayscale image (0 to 255)
    data_8bit = data_8bit.astype(np.uint8)
    #Lapacian of gaussian
    blobs_log = blob_log(data_8bit, min_sigma=2, max_sigma=30, num_sigma=30, threshold=0.01)# Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)
    del data_8bit
    # =============================================================================
    #   Write DAT file for lauetools      
    # =============================================================================
    path = os.path.normpath(filename)
    nbpeaks, _ = blobs_log.shape    
    eccentricity = blobs_log[:, 2]
    peak_X = blobs_log[:,1]
    peak_Y = blobs_log[:,0]
    peak_I = blobs_log[:, 2] #np.ones(nbpeaks) * 255.
    peak_bkg = np.zeros(nbpeaks)
    peak_inclination = np.zeros(nbpeaks)
    Xdev = np.zeros(nbpeaks)
    Ydev = np.zeros(nbpeaks)
    peak_bkg = np.zeros(nbpeaks)
    Ipixmax = np.zeros(nbpeaks)
    
    if dat_flag:
        # =============================================================================
        #     DAT file
        # =============================================================================
        outputfilename = file_directory+"//"+path.split(os.sep)[-1].split(".")[0] + ".dat"
        outputfile = open(outputfilename, "w")
        outputfile.write("peak_X peak_Y peak_Itot peak_Isub peak_fwaxmaj peak_fwaxmin peak_inclination Xdev Ydev peak_bkg Ipixmax \n")        
        outputfile.write(
            "\n".join(
                ["%.02f   %.02f   %.02f   %.02f   %.02f   %.02f    %.03f   %.02f   %.02f   %.02f   %d"
                    % tuple(list(zip(peak_X.round(decimals=2),
                                peak_Y.round(decimals=2),
                                (peak_I + peak_bkg).round(decimals=2),
                                peak_I.round(decimals=2),
                                eccentricity.round(decimals=2),
                                eccentricity.round(decimals=2),
                                peak_inclination.round(decimals=2),
                                Xdev.round(decimals=2),
                                Ydev.round(decimals=2),
                                peak_bkg.round(decimals=2),
                                Ipixmax))[i] ) for i in list(range(nbpeaks))]))
    
        outputfile.write("\n")
        outputfile.write("# File created with IOLaueTools.py \n")
        outputfile.write("# From: "+filename + " \n")
        outputfile.write("# Comments: nb of peaks "+str(nbpeaks)+" \n")
        outputfile.close()
        
    if cor_flag:
        # =============================================================================
        #     COR file
        # =============================================================================
        detectorparameters = [79.51900, 1951.6300, 1858.1500, 0.3480000, 0.4560000]
        pixelsize = 0.0367
        twicetheta, chi = Lgeo.calc_uflab(peak_X, peak_Y, detectorparameters,
                                            returnAngles=1,
                                            pixelsize=pixelsize,
                                            kf_direction='Z>0')
        
        dict_dp={}
        dict_dp['kf_direction']='Z>0'
        dict_dp['detectorparameters']=detectorparameters
        dict_dp['detectordistance']=detectorparameters[0]
        dict_dp['detectordiameter']=pixelsize*framedim[0]
        dict_dp['pixelsize']=pixelsize
        dict_dp['dim']=framedim
        dict_dp['peakX']=peak_X
        dict_dp['peakY']=peak_Y
        dict_dp['intensity']=peak_I
        
        CCDcalib = {"CCDLabel":"sCMOS_16M",
                    "dd":detectorparameters[0], 
                    "xcen":detectorparameters[1], 
                    "ycen":detectorparameters[2], 
                    "xbet":detectorparameters[3], 
                    "xgam":detectorparameters[4],
                    "pixelsize": pixelsize}
        
        path = os.path.normpath(filename)
        IOLT.writefile_cor(file_directory+"//"+path.split(os.sep)[-1].split(".")[0], twicetheta, 
                            chi, peak_X, peak_Y, peak_I,
                            param=CCDcalib, sortedexit=1)
    
if __name__ == "__main__":
    folder_tiff = r"C:\Users\purushot\Desktop\Laue_Zr_HT\1250Cfriday"
    dat_flag=False
    cor_flag=True
    file_directory = folder_tiff + "\\cor_files"
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)
    
    temp_ = folder_tiff + "\\cor_files\\LT_ROIs"
    if not os.path.exists(temp_):
        os.makedirs(temp_)
        
    temp_ = folder_tiff + "\\cor_files\\LT_LOG_ROIs"
    if not os.path.exists(temp_):
        os.makedirs(temp_)
        
    temp_ = folder_tiff + "\\cor_files\\LOG_nofit"
    if not os.path.exists(temp_):
        os.makedirs(temp_)
    
    
    experimental_prefix= "Zr3_1250_"
    format_file = "tif"
    
    list_of_files = glob.glob(folder_tiff+'//'+experimental_prefix+'*.'+format_file)
    list_of_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    n_processes = cpu_count()
    
    
    if n_processes == 1: ## TEST
        for ino, filename in enumerate(list_of_files):
            if ino != 0:
                continue
            process_images(filename, file_directory)#, dat_flag, cor_flag)
        
    else:
        args = zip(list_of_files, itertools.repeat(file_directory))#, itertools.repeat(dat_flag), itertools.repeat(cor_flag))
        with Pool(processes=n_processes) as proc_pool:
            results = proc_pool.starmap(process_images, tqdm(args, total=len(list_of_files)))
            
            