# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:54:36 2022

@author: PURUSHOT

Lets treat the Laue patterns as multi-blob problem with open cv
Detect multi blob locations
Then find through the image series if these blobs exists elsewhere
Do a voting system to know which blob is present where

"""
import numpy as np
import glob
import re
from tqdm import tqdm, trange
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import itertools

try:
    from lauetools import imageprocessing as ImProc
except:
    import lauetoolsnn.lauetools.imageprocessing as ImProc

from skimage.feature import blob_log
import cv2

from vtk.util import numpy_support
import vtk

def process_images(filename, bg_threshold=100, min_sigma=1, max_sigma=15, verbose = 0):
    data_8bit_raw = plt.imread(filename)
    backgroundimage = ImProc.compute_autobackground_image(data_8bit_raw, boxsizefilter=10)
    # basic substraction
    data_8bit_rawtiff = ImProc.computefilteredimage(data_8bit_raw, 
                                                    backgroundimage, 
                                                    "sCMOS", 
                                                    usemask=True, 
                                                    formulaexpression="A-B")
    data_8bit_raw = np.copy(data_8bit_rawtiff)
    ## simple thresholding
    data_8bit_raw[data_8bit_raw < bg_threshold] = 0
    data_8bit_raw[data_8bit_raw > 0] = 255
    data_8bit_raw = data_8bit_raw.astype(np.uint8)
    ### Resize
    data_8bit_raw = cv2.resize(data_8bit_raw,(0,0),fx = 0.25, fy = 0.25)
    # if verbose:
    #     cv2.imshow('Bin image', data_8bit_raw)
    #     cv2.waitKey(0)
        
    #Lapacian of gaussian
    data_8bit_rawtiff = cv2.resize(data_8bit_rawtiff,(0,0),fx = 0.25, fy = 0.25)
    blobs_log = blob_log(data_8bit_raw, min_sigma=min_sigma, 
                         max_sigma=max_sigma, num_sigma=30, threshold=0.01)# Compute radii in the 3rd column.
    blobs_log = blobs_log.astype(np.int64)
    blobs_log[:, 2] = data_8bit_rawtiff[blobs_log[:, 0], blobs_log[:, 1]]
    
    if verbose:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
        ax = axes.ravel()
        ax[0].set_title("Raw image (BG_subtracted)")
        ax[0].imshow(data_8bit_rawtiff, vmin=0, vmax=50)
        ax[0].set_axis_off()
        ax[1].set_title("Raw image (processed)")
        ax[1].imshow(data_8bit_raw, interpolation='nearest', cmap='gray')
        ax[1].set_axis_off()
        ax[2].imshow(data_8bit_raw, interpolation='nearest', cmap='gray')
        for blob in blobs_log:
            y, x, r = blob
            c = plt.Circle((x, y), 10, color="r", linewidth=1, fill=False)
            ax[2].add_patch(c)
        ax[2].set_axis_off()
        plt.tight_layout()
        plt.show()
    return blobs_log

def plt_img_peaks(filename, peaks, verbose = 0, factor=5):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Circle
    
    data_8bit_raw = plt.imread(filename)
    backgroundimage = ImProc.compute_autobackground_image(data_8bit_raw, boxsizefilter=10)
    # basic substraction
    data_8bit_rawtiff = ImProc.computefilteredimage(data_8bit_raw, 
                                                    backgroundimage, 
                                                    "sCMOS", 
                                                    usemask=True, 
                                                    formulaexpression="A-B")
    data_8bit_raw = np.copy(data_8bit_rawtiff)
    ## simple thresholding
    bg_threshold = 100
    data_8bit_raw[data_8bit_raw < bg_threshold] = 0
    data_8bit_raw[data_8bit_raw > 0] = 255
    data_8bit_raw = data_8bit_raw.astype(np.uint8)
    
    if verbose:
        fig,ax = plt.subplots(1)
        ax.set_aspect('equal')
        # Show the image
        ax.imshow(data_8bit_raw)
        plt.savefig('raw_image.png', bbox_inches='tight',format='png', dpi=1000) 
        plt.close(fig)
        # Create a figure. Equal aspect so circles look circular
        fig,ax = plt.subplots(1)
        ax.set_aspect('equal')
        # Show the image
        ax.imshow(data_8bit_raw)
        # Now, loop through coord arrays, and create a circle at each x,y pair
        peaks[:,2] = np.log(peaks[:,2])
        for yy,xx,ii in zip(peaks[:,0],peaks[:,1],peaks[:,2]):
            circ = Circle((xx,yy), ii*factor, color="r", fill=False)
            ax.add_patch(circ)
        plt.savefig('img_peaks.png', bbox_inches='tight',format='png', dpi=1000) 
        plt.close(fig)
    
def numpy_array_as_vtk_image_data(source_numpy_array, nx, ny, nz, 
                                  filename='default.vti', dtype="INT"):
    """
    Convert numpy arrays to VTK class object
    Seems to support only Log-Scale color now
    Need to check the nb of component parameters in VTK IMAGE class TODO
    """
    source_numpy_array = source_numpy_array.reshape((source_numpy_array.size))
    data_array = numpy_support.numpy_to_vtk(source_numpy_array)
    image_data = vtk.vtkImageData()
    vtkinfo = image_data.GetInformation()
    if dtype == "FLOAT":
        image_data.SetPointDataActiveScalarInfo(vtkinfo, vtk.VTK_DOUBLE, 1)
    elif dtype == "INT":
        image_data.SetPointDataActiveScalarInfo(vtkinfo, vtk.VTK_INT, 1)
    image_data.SetOrigin(0, 0, 0)
    image_data.SetSpacing(1, 1, 1)
    image_data.SetExtent(0, nx - 1, 0, ny - 1, 0, nz - 1)
    image_data.GetPointData().AddArray(data_array)
    # export data to file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(image_data)
    writer.Write()

def extract_patch_ori_pp(results, index, lim_x, lim_y, max_len, peak_tolerance, peakind=[1]):
    peaks = results[index]
    int_sort = np.argsort(peaks[:,2])[::-1]
    peaks = peaks[int_sort][peakind,:]
    voting_volume = np.zeros((lim_x*lim_y, max_len), dtype=np.int64)

    for jpeak, peaks1 in enumerate(results):
        diff = (peaks[:,None,:] - peaks1)
        # diff_intensity = -diff[:,:,2] ## +ve indicates high intensity from reference peak
        diff_position = np.abs(diff[:,:,:2]).sum(axis=2)
        diff_position_tol = diff_position < peak_tolerance
        indx, indy = np.where(diff_position_tol)
        voting_volume[jpeak,indx] = peaks1[indy,2] #diff_intensity[indx, indy]
    voting_volume = voting_volume.reshape((lim_x, lim_y, voting_volume.shape[1]))
    return voting_volume

def lower_bound_index(x, l):
    for i, y in enumerate(l):
        if y > x:
            return i
    return len(l)

def create_3d_hist_from2d(max_len, results_mp, bin_range, image_no):
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    yspacing = 1
    for peak_index in trange(max_len):
        histo = results_mp[image_no][:,:,peak_index].flatten()
        hist, bin_edges = np.histogram(list(range(bin_range)), weights=histo, bins=list(range(bin_range))) 
        dx = np.diff(bin_edges)
        dy = np.ones_like(hist)
        y = peak_index * (1 + yspacing) * np.ones_like(hist)
        z = np.zeros_like(hist)
        ax.bar3d(bin_edges[:-1], y, z, dx, dy, hist, color='b', 
                zsort='average', alpha=0.5)
    plt.savefig('3d_histogram.png', bbox_inches='tight',format='png', dpi=1000) 
    plt.close(fig)
# =============================================================================
# calculations
# =============================================================================

if __name__ == "__main__":
    folder_tiff = r"E:\vijaya_lauedata\HS261120b_SH2_S5_B_"
    lim_x, lim_y = 51,51
    experimental_prefix= "HS261120b_SH2_S5_B_"
    format_file = "tif"
    list_of_files = glob.glob(folder_tiff+'//'+experimental_prefix+'*.'+format_file)
    list_of_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    
    segment_images = False
    analyze_peaks = False
    write_vtk = True
    postprocess = True
    
    if segment_images: #30 seconds per iamge
        # process_images(list_of_files[0], bg_threshold=100, min_sigma=1, max_sigma=15, verbose=1)
        n_processes = cpu_count()
        args = zip(list_of_files, itertools.repeat(50), itertools.repeat(2),
                    itertools.repeat(15), itertools.repeat(0))
        with Pool(processes=n_processes) as proc_pool:
            results = proc_pool.starmap(process_images, tqdm(args, total=len(list_of_files)))
        np.savez_compressed("npz_files"+"//"+experimental_prefix+'.npz', results)
    
    if analyze_peaks: # 5 sec per image
        results = list(np.load("npz_files"+"//"+experimental_prefix+".npz", allow_pickle=True)["arr_0"])

        peak_tolerance = 2 #pixels
        max_len = 0
        for i in results:
            if len(i) > max_len:
                max_len = len(i)
                
        val12 = list(range(len(results)))
        
        peakind = [0,1,2,3]   
        max_len = len(peakind)

        args = zip(itertools.repeat(results), val12, itertools.repeat(lim_x),
                    itertools.repeat(lim_y), itertools.repeat(max_len), itertools.repeat(peak_tolerance), 
                    itertools.repeat(peakind))
        with Pool(processes=cpu_count()) as proc_pool:
            results_mp = proc_pool.starmap(extract_patch_ori_pp, tqdm(args, total=len(val12)))
            
        np.savez_compressed("npz_files"+"//"+experimental_prefix+'processed.npz', results_mp)
        
    if postprocess:
        results_mp = list(np.load("npz_files"+"//"+experimental_prefix+'processed.npz', allow_pickle=True)["arr_0"])
        # results = list(np.load("npz_files"+"//"+experimental_prefix+".npz", allow_pickle=True)["arr_0"]) 
        # max_len = 0
        # for i in results:
        #     if len(i) > max_len:
        #         max_len = len(i)
        # ## plot image with peaks to check
        # plt_img_peaks(list_of_files[0], np.copy(results[0]), verbose=1, factor=1)
        
        peakind = [0,1,2,3]   
        max_len = len(peakind)
        

        if write_vtk:
            peak_max = 0
            peak_min = 100
            for image_no in trange(len(results_mp)):
                result_peak = results_mp[image_no]
                # taking the most intense peak
                intense_peak_index = 0
                intense_peak = result_peak[:,:,intense_peak_index]
                peak_max1 = intense_peak.max()
                peak_min1 = intense_peak.min()
                if peak_max1 > peak_max:
                    peak_max = peak_max1
                if peak_min1 < peak_min:
                    peak_min = peak_min1
                    
            ## single VTK file with alll data (with real intensity)
            zdim = np.max((lim_x, lim_y))
            volume_3d = np.zeros((zdim, lim_x, lim_y), dtype=np.int64)
            step = (peak_max - peak_min)//zdim
            if step == 0:
                step = 1
            intensity_range = list(range(peak_min+1, peak_max, step))
            
            for image_no in trange(len(results_mp)):
                result_peak = results_mp[image_no]
                # taking the most intense peak
                # intense_peak_index = 0
                for intense_peak_index in peakind:
                    intense_peak = result_peak[:,:,intense_peak_index]
                    for i in range(lim_x):
                        for j in range(lim_y):
                            index = lower_bound_index(np.log(intense_peak[i,j]), np.log(intensity_range))
                            volume_3d[-index:,i,j] = volume_3d[-index:,i,j] + np.int64(intense_peak[i,j])
            
            numpy_array_as_vtk_image_data(volume_3d, 
                                          volume_3d.shape[0], 
                                          volume_3d.shape[1], 
                                          volume_3d.shape[2],
                                          filename=experimental_prefix+'_all_realintensity_4peaks.vti', 
                                          dtype="INT")
            
            # image_no = 0
            # result_peak = results_mp[image_no]
            
            # # peak_index = 0 # which peak; 0 is the intense peak in image
            # image_no = 0
            # result_peak = results_mp[image_no]
            # # or a list of high intersection peaks
            # peak1_corr = np.where(score[0,:] > 50)[0]
            # zdim = np.max((lim_x, lim_y))
            # volume_3d = np.zeros((zdim, lim_x, lim_y), dtype=np.int32)
            
            # # first find the max and min values
            # peak_max = 0
            # peak_min = 100
            # for peak_index in peak1_corr:
            #     peak_max1 = result_peak[:,:,peak_index].max()
            #     peak_min1 = result_peak[:,:,peak_index].min()
            #     if peak_max1 > peak_max:
            #         peak_max = peak_max1
            #     if peak_min1 < peak_min:
            #         peak_min = peak_min1
             
            # step = (peak_max - peak_min)//zdim
            # if step == 0:
            #     step = 1
            # intensity_range = list(range(peak_min+1, peak_max, step))
            
            # for peak_index in peak1_corr:
            #     for i in range(lim_x):
            #         for j in range(lim_y):
            #             if result_peak[i,j,peak_index] < 10000:
            #                 continue
            #             index = lower_bound_index(result_peak[i,j,peak_index], intensity_range)
            #             volume_3d[-index:,i,j] = volume_3d[-index:,i,j] + 1
                        
            # numpy_array_as_vtk_image_data(volume_3d, 
            #                               volume_3d.shape[0], 
            #                               volume_3d.shape[1], 
            #                               volume_3d.shape[2],
            #                               filename=experimental_prefix+str(image_no)+'.vti', 
            #                               dtype="INT")
            
        

                    
        
        # def return_intersection(hist_1, hist_2):
        #     # intersection of two histogram
        #     minima = np.minimum(hist_1, hist_2)
        #     intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
        #     return intersection*100
        
        # #1D histo with image index
        # bin_range = lim_x * lim_y
        # score = np.zeros((max_len, max_len))
        # for peak_index in trange(max_len):
        #     # histo = results_mp[image_no][:,:,peak_index].flatten()
        #     # h = np.histogram(list(range(bin_range)), weights=histo, bins=list(range(bin_range)))
        #     # plt.hist(list(range(bin_range)), weights=histo, bins=list(range(bin_range)))
        #     for peak_index1 in range(max_len):
        #         score[peak_index, peak_index1] = return_intersection(
        #                                                 results_mp[0][:,:,peak_index].flatten(), 
        #                                                 results_mp[0][:,:,peak_index1].flatten()
        #                                                             )
        # np.savez_compressed('score.npz', score)
        # score = np.array(np.load("score.npz", allow_pickle=True)["arr_0"])
        
        # if 0:            
        #     ## save some histograms        
        #     fig,ax = plt.subplots(1)
        #     histo = results_mp[0][:,:,0].flatten()
        #     plt.hist(list(range(bin_range)), weights=histo, bins=list(range(bin_range)))
        #     plt.savefig('histogram_peak0.png', bbox_inches='tight',format='png', dpi=1000) 
        #     plt.close(fig)
            
            
        #     ##% create 3D histograms from 1D
        #     create_3d_hist_from2d(50, results_mp, bin_range, 0)
            
        #     ## save intersection score as histogram        
        #     fig,ax = plt.subplots(1)
        #     histo = score[0,:]
        #     histo[histo==0] = np.nan
        #     plt.hist(list(range(len(histo))), weights=histo, bins=list(range(len(histo))))
        #     plt.savefig('histogram_score_peak0.png', bbox_inches='tight',format='png', dpi=1000) 
        #     plt.close(fig)
            
            
        #     ## plot 2 histograms    
        #     print(score[0,1], score[1,0])
        #     fig,ax = plt.subplots(1)
        #     histo = results_mp[0][:,:,0].flatten()
        #     plt.hist(list(range(bin_range)), weights=histo, bins=list(range(bin_range)), color="r",
        #              label="1st peak")
        #     histo = results_mp[1][:,:,0].flatten()
        #     plt.hist(list(range(bin_range)), weights=histo, bins=list(range(bin_range)), color="k",
        #              label="2nd peak")
        #     plt.savefig('histogram_peaks.png', bbox_inches='tight',format='png', dpi=1000) 
        #     plt.close(fig)
            
        #     ## 2D density sort of plot based on scores
        #     peak1_corr = np.where(score[0,:] > 50)[0]
        #     plt.imshow(results_mp[0][:,:,peak1_corr].sum(axis=2),origin='lower', aspect='auto',cmap='Blues')
        #     cb = plt.colorbar()
        #     cb.set_label("Intensity")
        
        
        
    
            
            