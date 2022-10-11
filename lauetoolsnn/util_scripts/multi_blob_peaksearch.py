# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:54:36 2022

@author: PURUSHOT

Lets treat the Laue patterns as multi-blob problem with open cv

Specifically for Laue patterns that has multi component
"""
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
import os

try:
    from lauetools import IOimagefile as IOimage
    from lauetools import imageprocessing as ImProc
    from lauetools import readmccd as RMCCD
    from lauetools import LaueGeometry as Lgeo
    from lauetools import IOLaueTools as IOLT
    from lauetools import CrystalParameters as CP
    from lauetools import lauecore as LT
except:
    import lauetoolsnn.lauetools.IOimagefile as IOimage
    import lauetoolsnn.lauetools.imageprocessing as ImProc
    import lauetoolsnn.lauetools.readmccd as RMCCD
    import lauetoolsnn.lauetools.LaueGeometry as Lgeo
    import lauetoolsnn.lauetools.IOLaueTools as IOLT
    import lauetoolsnn.lauetools.CrystalParameters as CP
    import lauetoolsnn.lauetools.lauecore as LT

from skimage.feature import blob_dog, blob_log
import cv2
from skimage import morphology

experimental_prefix= "Zr3_1250_"
format_file = "tif"
CCDLabel = "sCMOS_16M"
folder_tiff = r"C:\Users\purushot\Desktop\Laue_Zr_HT\1250Cfriday"
list_of_files = glob.glob(folder_tiff+'//'+experimental_prefix+'*.'+format_file)
list_of_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

show_img = False
cropped = False
multi_blob_detection = False
stackoverflow = False
ori_mat_reinforced = True

vmin = 0
threshold = 0

for ino, filename in enumerate(list_of_files):
    if ino != 0:
        continue
    ## introduce FIJI image treatment procedure here to avoid using multiple softwares
    
    if ori_mat_reinforced:
        ##Idea is to get the XY pixel from the oreintation matrix and extract all
        ##closest peaks to a given ori mat.
        data_8bit_raw = plt.imread(filename)
        framedim = data_8bit_raw.shape
        
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
        
        blobs_log_all_x, blobs_dog_all_x = [], []
        blobs_log_all_y, blobs_dog_all_y = [], []
        blobs_log_all_ecc, blobs_dog_all_ecc = [], []
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
            
            # kernel = np.ones((3,3),np.uint8)
            # thresh = cv2.morphologyEx(crop_data_8bit_raw, cv2.MORPH_OPEN, kernel, iterations = 2)
            
            kernel = np.ones((3,3),np.uint8)
            thresh = cv2.morphologyEx(crop_data_8bit_raw, cv2.MORPH_OPEN, kernel, iterations = 2)
            
            # kernel = np.ones((3,3),np.uint8)
            # closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 2)

            kernel = np.ones((2,2),np.uint8)
            thresh = cv2.erode(thresh, kernel, iterations = 1)
            
            # kernel = np.ones((3,3),np.uint8)
            # thresh = cv2.dilate(thresh, kernel, iterations = 2)
            
            # print(xtheo_min,xtheo_max, ytheo_min,ytheo_max)
            # print(xpos,ypos)

            if np.all(thresh==0):
                continue     
            
            ### make one big image with patches
            big_image[xtheo_min:xtheo_max, ytheo_min:ytheo_max] = thresh

            minsigma, maxsigma = 10, 15
            threshold_int = 0.01
            #Lapacian of gaussian
            blobs_log = blob_log(thresh, min_sigma=minsigma, max_sigma=maxsigma, 
                                 num_sigma=10, threshold=threshold_int)# Compute radii in the 3rd column.
            blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)
            
            #difference of gaussian
            blobs_dog = blob_dog(thresh, min_sigma=minsigma, max_sigma=maxsigma, threshold=threshold_int)
            blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)
            
            blobs_log_all_x.append(xtheo_min+blobs_log[:,1])
            blobs_dog_all_x.append(xtheo_min+blobs_dog[:,1])
            
            blobs_log_all_y.append(ytheo_min+blobs_log[:,0])
            blobs_dog_all_y.append(ytheo_min+blobs_dog[:,0])
            
            blobs_log_all_ecc.append(blobs_log[:,2])
            blobs_dog_all_ecc.append(blobs_dog[:,2])
            
            #all in list
            blobs_list = [blobs_log, blobs_dog]
            colors = ['yellow', 'lime']
            titles = ['Laplacian of Gaussian', 'Difference of Gaussian']
            sequence = zip(blobs_list, colors, titles)
            
            fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
            ax = axes.ravel()
            ax[2].set_title("Raw image (BG_subtracted)")
            ax[2].imshow(data_8bit_raw_bg[xtheo_min:xtheo_max, ytheo_min:ytheo_max], vmin=0, vmax=50)
            for idx, (blobs, color, title) in enumerate(sequence):
                # print("Nb of blobs in "+title+" is "+str(len(blobs)))
                ax[idx].set_title(title)
                ax[idx].imshow(thresh, interpolation='nearest', cmap='gray')
                for blob in blobs:
                    y, x, r = blob
                    if r > np.sqrt(2)+0.1:
                        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
                        ax[idx].add_patch(c)
                ax[idx].set_axis_off()
                plt.tight_layout()
            plt.savefig(str(cnt)+'.png', bbox_inches='tight',format='png', dpi=200) 
            # plt.show()
            plt.close()
                    
        blobs_log_all_x = np.concatenate(blobs_log_all_x, axis=0)
        blobs_log_all_y = np.concatenate(blobs_log_all_y, axis=0)
        blobs_log_all_ecc = np.concatenate(blobs_log_all_ecc, axis=0)
        
        blobs_dog_all_x = np.concatenate(blobs_dog_all_x, axis=0)
        blobs_dog_all_y = np.concatenate(blobs_dog_all_y, axis=0)
        blobs_dog_all_ecc = np.concatenate(blobs_dog_all_ecc, axis=0)
    
        blobs_log = np.column_stack((blobs_log_all_x, blobs_log_all_y, blobs_log_all_ecc))
        blobs_dog = np.column_stack((blobs_dog_all_x, blobs_dog_all_y, blobs_dog_all_ecc))            
        
        ## delete small circles
        val_del = np.sqrt(2)+0.1
        ind_del_log = np.where(blobs_log[:,2]<val_del)[0]
        ind_del_dog = np.where(blobs_log[:,2]<val_del)[0]
        
        blobs_log = np.delete(blobs_log, ind_del_log, axis=0)
        blobs_dog = np.delete(blobs_dog, ind_del_dog, axis=0)
        
        #all in list
        blobs_list = [blobs_log, blobs_dog]
        colors = ['yellow', 'lime']
        titles = ['Laplacian of Gaussian', 'Difference of Gaussian']
        sequence = zip(blobs_list, colors, titles)
        
        fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
        ax = axes.ravel()
        ax[2].set_title("Raw image (BG_subtracted)")
        ax[2].imshow(data_8bit_raw_bg, vmin=0, vmax=50)
        for idx, (blobs, color, title) in enumerate(sequence):
            # print("Nb of blobs in "+title+" is "+str(len(blobs)))
            ax[idx].set_title(title)
            ax[idx].imshow(big_image, interpolation='nearest', cmap='gray')
            for blob in blobs:
                y, x, r = blob
                if r > val_del:
                    c = plt.Circle((x, y), r, color=color, linewidth=0.5, fill=False)
                    ax[idx].add_patch(c)
            ax[idx].set_axis_off()
            plt.tight_layout()
        plt.savefig('big_image.png', bbox_inches='tight',format='png', dpi=200) 
        # plt.show()
        plt.close()

        # =============================================================================
        #   Write Cor file for lauetools      
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

        dat_file_directory = folder_tiff + "\\cor_files"
        if not os.path.exists(dat_file_directory):
            os.makedirs(dat_file_directory)
        
        # =============================================================================
        #     COR file
        # =============================================================================
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
        IOLT.writefile_cor(dat_file_directory+"//"+path.split(os.sep)[-1].split(".")[0], twicetheta, 
                            chi, peak_X, peak_Y, peak_I,
                            param=CCDcalib, sortedexit=1)
    
    
#     continue
# # =============================================================================
# #     
# # =============================================================================
#     # Data are read from image file
#     Data, framedim, fliprot = IOimage.readCCDimage(filename,
#                                             stackimageindex=-1,
#                                             CCDLabel=CCDLabel,
#                                             dirname=None,
#                                             verbose=0)
        
#     # backgroundimage = ImProc.compute_autobackground_image(Data, boxsizefilter=10)
#     # # basic substraction
#     # Data = ImProc.computefilteredimage(Data, backgroundimage, CCDLabel, usemask=True,
#     #                                                     formulaexpression="A-B")
    
#     # peak_XY = RMCCD.PeakSearch(
#     #                             filename,
#     #                             stackimageindex = -1,
#     #                             CCDLabel=CCDLabel,
#     #                             NumberMaxofFits=5000,
#     #                             PixelNearRadius=10,
#     #                             removeedge=2,
#     #                             IntensityThreshold=100,
#     #                             local_maxima_search_method=0,
#     #                             boxsize=100,
#     #                             position_definition=1,
#     #                             verbose=0,
#     #                             fit_peaks_gaussian=0,
#     #                             xtol=0.001,                
#     #                             FitPixelDev=50,
#     #                             return_histo=0,
#     #                             MinIntensity=0,
#     #                             PeakSizeRange=(0.65,200),
#     #                             write_execution_time=1,
#     #                             Data_for_localMaxima = None, #"auto_background",
#     #                             formulaexpression="A-B",
#     #                             Remove_BlackListedPeaks_fromfile=None,
#     #                             reject_negative_baseline=True,
#     #                             Fit_with_Data_for_localMaxima=False,
#     #                             maxPixelDistanceRejection=15.0,
#     #                             )
#     # peak_XY = peak_XY[0]
    
#     if threshold > 0:
#         Data[Data<threshold] = 0
    
    
#     if show_img:
#         fig = plt.figure()
#         plt.imshow(Data, cmap='hot', vmin=vmin, vmax=vmin+1)
#         plt.show(fig)


#     if cropped:
#         xmin, xmax = 1700, 2000
#         ymin, ymax = 1300, 1700
#         fig1 = plt.figure()
#         plt.imshow(Data[xmin:xmax,ymin:ymax], cmap='hot', vmin=vmin, vmax=vmin+1)
#         plt.show(fig1)

        
#     if multi_blob_detection:
#         # xmin, xmax = 1700, 2000
#         # ymin, ymax = 1300, 1700
#         # #threshold
#         # Data[Data<threshold] = 0
#         # data_8bit = np.copy(Data[xmin:xmax,ymin:ymax])
        
        
#         Data[Data<threshold] = 0
#         data_8bit = np.copy(Data)
#         ### Grayscale image (0 to 255)
#         data_8bit = data_8bit.astype(np.uint8)
        
#         # #Temp step
#         data_8bit = data_8bit[1790:1970,1425:1625]
#         # data_8bit[data_8bit<100] = 0
        
#         ## cropped main image
#         # fig1 = plt.figure()
#         # plt.imshow(data_8bit, cmap='gray')
#         # plt.show(fig1)
        
#         #Lapacian of gaussian
#         blobs_log = blob_log(data_8bit, min_sigma=2, max_sigma=30, num_sigma=30, threshold=0.01)# Compute radii in the 3rd column.
#         blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)
        
        
#         # =============================================================================
#         #   Write DAT file for lauetools      
#         # =============================================================================
#         dat_file_directory = folder_tiff + "\\dat_files"
#         if not os.path.exists(dat_file_directory):
#             os.makedirs(dat_file_directory)
            
#         path = os.path.normpath(filename)
#         # writing ascii peak list    
#         nbpeaks, _ = blobs_log.shape
        
#         eccentricity = blobs_log[:, 2]
#         peak_X = blobs_log[:,0]
#         peak_Y = blobs_log[:,1]
#         peak_I = np.ones(nbpeaks) * 255.
#         peak_bkg = np.zeros(nbpeaks)
#         peak_fwaxmaj = np.ones(nbpeaks)
#         peak_fwaxmin = np.ones(nbpeaks)
#         peak_inclination = np.zeros(nbpeaks)
#         Xdev = np.zeros(nbpeaks)
#         Ydev = np.zeros(nbpeaks)
#         peak_bkg = np.zeros(nbpeaks)
#         Ipixmax = np.ones(nbpeaks) * 255.
        
#         outputfilename = dat_file_directory+"//"+path.split(os.sep)[-1].split(".")[0] + ".dat"
#         outputfile = open(outputfilename, "w")
#         outputfile.write("#peak_X peak_Y peak_Itot peak_Isub peak_fwaxmaj peak_fwaxmin peak_inclination Xdev Ydev peak_bkg Ipixmax Eccentricity \n")
#         # peak_X peak_Y peak_Itot peak_Isub peak_fwaxmaj peak_fwaxmin peak_inclination Xdev Ydev peak_bkg Ipixmax
        
#         outputfile.write(
#             "\n".join(
#                 ["%.02f   %.02f   %.02f   %.02f   %.02f   %.02f    %.03f   %.02f   %.02f   %.02f   %d   %0.2f"
#                     % tuple(list(zip(peak_X.round(decimals=2),
#                                 peak_Y.round(decimals=2),
#                                 (peak_I + peak_bkg).round(decimals=2),
#                                 peak_I.round(decimals=2),
#                                 peak_fwaxmaj.round(decimals=2),
#                                 peak_fwaxmin.round(decimals=2),
#                                 peak_inclination.round(decimals=2),
#                                 Xdev.round(decimals=2),
#                                 Ydev.round(decimals=2),
#                                 peak_bkg.round(decimals=2),
#                                 Ipixmax,
#                                 eccentricity.round(decimals=2)))[i] ) for i in list(range(nbpeaks))]))
#         nbpeaks = len(peak_X)
#         outputfile.close()
        
#     if stackoverflow:
#         img_src = cv2.imread(filename)
        
#         Data[Data<threshold] = 0
#         data_8bit = np.copy(Data)
#         data_8bit = data_8bit.astype(np.uint8)
        
#         # xmin, xmax = 1700, 2000
#         # ymin, ymax = 1300, 1700
#         # img_src = img_src[xmin:xmax,ymin:ymax]
#         # img = data_8bit[xmin:xmax,ymin:ymax]
#         # gray = img
        
#         img = data_8bit
#         gray = img
        
#         fig1 = plt.figure()
#         plt.imshow(img, cmap='gray')
#         plt.show(fig1)
        
#         # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#         ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                
#         fig1 = plt.figure()
#         plt.imshow(255-thresh, cmap='gray')
#         plt.show(fig1)
        
#         # noise removal
#         kernel = np.ones((3,3),np.uint8)
#         opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
#         opening = 255 - opening
        
#         fig1 = plt.figure()
#         plt.imshow(opening, cmap='gray')
#         plt.show(fig1)

#         sure_bg = cv2.dilate(opening,kernel,iterations=3)
#         fig1 = plt.figure()
#         plt.imshow(sure_bg, cmap='gray')
#         plt.show(fig1)
        
#         # Finding sure foreground area
#         dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
#         fig1 = plt.figure()
#         plt.imshow(dist_transform, cmap='gray')
#         plt.show(fig1)


#         ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
#         fig1 = plt.figure()
#         plt.imshow(sure_fg, cmap='gray')
#         plt.show(fig1)
        
#         # Finding unknown region
#         sure_fg = np.uint8(sure_fg)
#         unknown = cv2.subtract(sure_bg,sure_fg)
#         fig1 = plt.figure()
#         plt.imshow(unknown, cmap='gray')
#         plt.show(fig1)
        
#         # Marker labelling
#         ret, markers = cv2.connectedComponents(sure_fg)
#         # Add one to all labels so that sure background is not 0, but 1
#         markers = markers+1
#         # Now, mark the region of unknown with zero
#         markers[unknown==255] = 0
#         fig1 = plt.figure()
#         plt.imshow(markers, cmap='hot')
#         plt.show(fig1)


#         markers = cv2.watershed(img_src,markers)
#         img[markers == -1] = 255 #[255,0,0]        
#         fig1 = plt.figure()
#         plt.imshow(img, cmap='gray')
#         plt.show(fig1)
