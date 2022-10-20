# -*- coding: utf-8 -*-
r"""
readmccd module is made for reading data contained in binary image file
fully or partially.
It can process a peak or blob search by various methods
and refine the peak by a gaussian or lorentzian 2D model

More tools can be found in LaueTools package at sourceforge.net and gitlab.esrf.fr
March 2020
"""
__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"

# built-in modules
import sys
import os
import copy
import time as ttt
import numpy as np
import configparser as CONF

try:
    from lauetools import fit2Dintensity as fit2d
    from lauetools import fit2Dintensity_Lorentz as fit2d_l
    from lauetools import IOLaueTools as IOLT
    from lauetools import generaltools as GT
    from lauetools import IOimagefile as IOimage
    from lauetools import dict_LaueTools as DictLT
    from lauetools import imageprocessing as ImProc
except:
    import lauetoolsnn.lauetools.fit2Dintensity as fit2d
    import lauetoolsnn.lauetools.fit2Dintensity_Lorentz as fit2d_l
    import lauetoolsnn.lauetools.IOLaueTools as IOLT
    import lauetoolsnn.lauetools.generaltools as GT
    import lauetoolsnn.lauetools.IOimagefile as IOimage
    import lauetoolsnn.lauetools.dict_LaueTools as DictLT
    import lauetoolsnn.lauetools.imageprocessing as ImProc
    
listfile = os.listdir(os.curdir)

# Default dictionary peak search parameters:

PEAKSEARCHDICT_Convolve = {"PixelNearRadius": 10,
                        "thresholdConvolve": 500,
                        "IntensityThreshold": 10,
                        "removeedge": 2,
                        "local_maxima_search_method": 2,
                        "boxsize": 15,
                        "position_definition": 1,
                        "verbose": 0,
                        "fit_peaks_gaussian": 1,
                        "xtol": 0.001,
                        "FitPixelDev": 2.0,
                        "return_histo": 0,
                        "write_execution_time": 0,
                        "Data_for_localMaxima": "auto_background",
                        "NumberMaxofFits": 5000}
# 'MaxPeakSize':3.0,
# 'MinPeakSize':0.01
# }

IOimagefuncs = ['Add_Images', 'Add_Images2', 'SumImages', 'getIndex_fromfilename', 'get_imagesize',
'getpixelValue', 'getwildcardstring',# 'libtiff_ctypes'
 'listfile', 'readCCDimage', 'read_header_marccd',
 'read_header_marccd2', 'read_header_scmos', 'read_motorsposition_fromheader', 'readheader', 'readoneimage', 'readoneimage_band', 'readoneimage_crop', 'readoneimage_crop_fast', 'readoneimage_full', 'readoneimage_manycrops',
 'readrectangle_in_image', 'setfilename', 'stringint', 'write_rawbinary', 'writeimage','getfilename']

imageprocfuncs = ['ConvolvebyKernel', 'LoG', 'LoGArr', 'LocalMaxima_KernelConvolution', 'LocalMaxima_ShiftArrays',
 'LocalMaxima_from_thresholdarray', 'LocalMaxima_ndimage', 'applyformula_on_images', 'blurCCD', 'blurCCD_with_binning',
 'blur_image', 'check_array_indices', 'circularMask', 'compute_autobackground_image', 'computefilteredimage',
 'diff_pix', 'filter_minimum', 'filterimage', 'gauss_kern', 'getExtrema',
 'getIntegratedIntensity', 'getMinMax', 'getindices2cropArray', 'listfile', 'localmaxima',
 'minmax', 'minmax_fast', 'myfromfunction', 'normalize_shape',
 'plot_image_markers', 'pp', 'radialArr', 'rebin2Darray', 'remove_minimum_background',
 'shiftarrays_accum', 'to8bits']


thismodule = sys.modules[__name__]

def warn_func_is_now_in(func, modulename):
    def inner(*args, **kwargs):
        print("\n Warning!  %s is no longer in readmccd.py but in %s\n" % (func.__name__, modulename))
        return func(*args, **kwargs)
    return inner

for iofunc in IOimagefuncs:
    setattr(thismodule, iofunc, warn_func_is_now_in(getattr(IOimage, iofunc), 'IOimagefile.py'))

for ipfunc in imageprocfuncs:
    setattr(thismodule, ipfunc, warn_func_is_now_in(getattr(ImProc, ipfunc), 'imageprocessing.py'))

def readoneimage_multiROIfit(filename, centers, boxsize, stackimageindex=-1, CCDLabel="PRINCETON",
                                                                        baseline="auto",
                                                                        startangles=0.0,
                                                                        start_sigma1=1.0,
                                                                        start_sigma2=1.0,
                                                                        position_start="max",
                                                                        fitfunc="gaussian",
                                                                        showfitresults=1,
                                                                        offsetposition=0,
                                                                        verbose=0,
                                                                        xtol=0.00000001,
                                                                        addImax=False,
                                                                        use_data_corrected=None):
    r"""
    Fit several peaks in one image

    :param filename: string, full path to image file
    :param centers: list or array like with shape=(n,2)
                    list of centers of selected ROI
    :param boxsize: (Truly HALF boxsize: fuill boxsize= 2(halfboxsize) +1), iterable 2 elements or integer
                    boxsizes [in x, in y] direction or integer to set a square ROI
    :param baseline: string, 'auto' (ie minimum intensity in ROI) or array of floats
    :param startangles: float or iterable of 2 floats, elliptic gaussian angle (major axis with respect to X direction),
                        one value or array of values
    :param start_sigma1, start_sigma2: floats, gaussian standard deviation (major and minor axis) in pixel,
    :param position_start:  string, starting gaussian center:'max' (position of maximum intensity in ROI),
                            "centers" (centre of each ROI)
    :param offsetposition: integer, 0 for no offset, 1  XMAS compatible, since XMAS consider first pixel as index 1 (in array, index starts with 0), 2  fit2d, since fit2d for peaksearch put pixel labelled n at the position n+0.5 (between n and n+1)
    :param use_data_corrected: tuple of 3 elements, Enter data instead of reading data from file:
                         fulldata, framedim, fliprot
                         where fulldata is a 2D ndarray
    :return: list of results:   bkg,  amp  (gaussian height-bkg), X , Y, major axis standard deviation, minor axis standard deviation,
                                major axis tilt angle / Ox
    .. todo:: setting list of initial guesses can be improve with
        scipy.ndimages of a concatenate array of multiple slices?
    """
    # if 1:#verbose > 0:
    #     print("addImax", addImax)
    
    # read data (several arrays)
    ResData = IOimage.readoneimage_manycrops(filename,
                                    centers,
                                    boxsize,
                                    stackimageindex,
                                    CCDLabel=CCDLabel,
                                    addImax=addImax,
                                    use_data_corrected=use_data_corrected)
    if addImax:
        Datalist, Imax = ResData
    else:
        Datalist = ResData

    framedim = DictLT.dict_CCD[CCDLabel][0]
    saturation_value = DictLT.dict_CCD[CCDLabel][2]
   
    Data = np.array(Datalist)

    # setting initial guessed values for each center
    nb_Images = len(Data)
    #     print "nb of images to fitdata ... in  readoneimage_multiROIfit()", nb_Images
    if baseline in ("auto", None):  # background height or baseline level
        list_min = []
        for _, dd in enumerate(Data):
            #            print "k, dd.shape", k, dd.shape
            list_min.append(np.amin(dd))
        start_baseline = list_min
    else:  # input numerical value array
        start_baseline = baseline

    if isinstance(startangles, (int, float)):
        start_anglerot = startangles * np.ones(nb_Images)
    else:
        start_anglerot = startangles

    if isinstance(start_sigma1, (int, float)):
        start_sigma1 = start_sigma1 * np.ones(nb_Images)
    if isinstance(start_sigma2, (int, float)):
        start_sigma2 = start_sigma2 * np.ones(nb_Images)

    if isinstance(boxsize, (int, float)):
        xboxsize, yboxsize = int(boxsize), int(boxsize)
    else:
        xboxsize, yboxsize = boxsize

    Xboxsize = xboxsize * np.ones(nb_Images)
    Yboxsize = yboxsize * np.ones(nb_Images)

    start_j = []
    start_i = []
    start_amplitude = []

    if position_start in ("centers", "center"):  # starting position  from input center
        start_j, start_i = yboxsize, xboxsize
        start_amplitude = []
        d = 0
        for dd in Data:
            start_amplitude.append(dd[int(start_j[d]), int(start_i[d])] - start_baseline[d])
            d += 1
    elif position_start == "max":  # starting position  from maximum intensity in dat

        d = 0
        for dd in Data:
            start_j.append(np.argmax(dd) // dd.shape[1])
            start_i.append(np.argmax(dd) % dd.shape[1])
            start_amplitude.append(np.amax(dd) - start_baseline[d])
            d += 1

    startingparams_zip = np.array([start_baseline, start_amplitude,
                                    start_j, start_i,
                                    start_sigma1, start_sigma2,
                                    start_anglerot])

    RES_params = []
    RES_cov = []
    RES_infodict = []
    RES_errmsg = []

    if verbose:
        print("startingparams_zip", startingparams_zip.T)

    # consider that ROi shape will be constant over all ROIs,
    # so no need to recompute np.indices in gaussfit
    ROIshape = Data[0].shape
    ijindices_array = np.indices(ROIshape)

    k_image = 0
    for startingparams in startingparams_zip.T:
        # if (k_image%25) == 0: print "%d/%d"%(k_image,nb_Images)
        if verbose:
            print("startingparams", startingparams)
        if fitfunc == "gaussian":
            if Data[k_image].shape != ROIshape:
                ijindices_array = None

            # ROIdata = Data[k_image]
            #             print 'ROIdata', ROIdata
            #             print 'np.amax(ROIdata)', np.amax(ROIdata)
            #             print 'np.amin (ROIdata)', np.amin(ROIdata)
            #             print 'np.argmax (ROIdata)', np.argmax(ROIdata)

            params, cov, infodict, errmsg = fit2d.gaussfit(Data[k_image],
                                                    err=None,
                                                    params=startingparams,
                                                    autoderiv=1,
                                                    return_all=1,
                                                    circle=0,
                                                    rotate=1,
                                                    vheight=1,
                                                    xtol=xtol,
                                                    Acceptable_HighestValue=saturation_value,
                                                    Acceptable_LowestValue=0,
                                                    ijindices_array=ijindices_array)

            if cov is not None and verbose:
                print('params  solution ', params)
                print('\n\n *****\n covariance matrix    --- ', cov.tolist())

        elif fitfunc == "lorentzian":
            params, cov, infodict, errmsg = fit2d_l.lorentzfit(Data[k_image],
                                                                err=None,
                                                                params=startingparams,
                                                                autoderiv=1,
                                                                return_all=1,
                                                                circle=0,
                                                                rotate=1,
                                                                vheight=1,
                                                                xtol=xtol)

        if showfitresults:
            # print "startingparams"
            # print startingparams
            print("\n *****fitting results ************\n")
            print(params)
            print("background intensity:                        {:.2f}".format(params[0]))
            print("Peak amplitude above background              {:.2f}".format(params[1]))
            print("pixel position (X)                   {:.2f}".format(
                    params[3] - Xboxsize[k_image] + centers[k_image][0])
            )  # WARNING Y and X are exchanged in params !
            print("pixel position (Y)                   {:.2f}".format(
                    params[2] - Yboxsize[k_image] + centers[k_image][1]))
            print("std 1,std 2 (pix)                    ( {:.2f} , {:.2f} )".format(
                    params[4], params[5]))
            print("e=min(std1,std2)/max(std1,std2)              {:.3f}".format(
                    min(params[4], params[5]) / max(params[4], params[5])))
            print("Rotation angle (deg)                 {:.2f}".format(params[6] % 360))
            print("************************************\n")
        bkg_sol, amp_sol, Y_sol, X_sol, std1_sol, std2_sol, ang_sol = params

        RES_cov.append(cov)
        RES_infodict.append(infodict)
        RES_errmsg.append(errmsg)

        params_sol = np.array([bkg_sol, amp_sol,
                X_sol - Xboxsize[k_image] + centers[k_image][0],
                Y_sol - Yboxsize[k_image] + centers[k_image][1],
                std1_sol,
                std2_sol,
                ang_sol])  # now X,Y in safest order

        if offsetposition == 1:
            # PATCH: To match peak positions given by XMAS
            # (confusion coming from counting array indices from 0 or 1...)
            # array fitted by python module see pixels at lower position
            params_sol[3] = params_sol[3] + 1.0
            params_sol[2] = params_sol[2] + 1.0
            # End of PATCH

        elif offsetposition == 2:  # see Compute_data2thetachi() in find2thetachi.py
            # PATCH: To match peak positions given by peaksearch of fit2D
            # in fit2D graphics window first pixel labelled 1 is for the
            # peaksearch located at position in  between 0 and 1 (ie 0.5)
            params_sol[3] = params_sol[3] + 0.5
            params_sol[2] = (framedim[0] - params_sol[2]) + 0.5  # TODO: tocheck dim[0] or dim[1]
            # End of PATCH

        RES_params.append(params_sol)

        k_image += 1

    if addImax:
        return RES_params, RES_cov, RES_infodict, RES_errmsg, start_baseline, Imax
    else:
        return RES_params, RES_cov, RES_infodict, RES_errmsg, start_baseline


def fitPeakMultiROIs(Data, centers, FittingParametersDict, showfitresults=True, verbose=False):
    r""" refine all peaks  guessed to be at center of several ROIs

    :param Data: list of Data array centered on peaks
    :param centers: list of pixels (x,y) positions of ROI centers
    :param FittingParametersDict: dict of fitting parameters

    :return: RES_params, RES_cov, RES_infodict, RES_errmsg, start_baseline
            which are all list of refinement results
    """
    boxsize = FittingParametersDict["boxsize"]
    if verbose: print("bosize", boxsize)
    if (boxsize[0] % 2 == 0) or (boxsize[1] % 2 == 0):
        print("boxsize", boxsize)
        raise ValueError("boxsizes are not odd !!")

    framedim = FittingParametersDict["framedim"]
    saturation_value = FittingParametersDict["saturation_value"]
    baseline = FittingParametersDict["baseline"]

    startangles = FittingParametersDict["startangles"]
    position_start = FittingParametersDict["position_start"]
    start_sigma1 = FittingParametersDict["start_sigma1"]
    start_sigma2 = FittingParametersDict["start_sigma2"]

    fitfunc = FittingParametersDict["fitfunction"]
    xtol = FittingParametersDict["xtol"]
    offsetposition = FittingParametersDict["offsetposition"]

    # setting initial guessed values for each center
    nb_Images = len(Data)
    if verbose:
        print("nb of images fitPeakMultiROIs", nb_Images)
        print("shape of Data", Data.shape)

    if baseline in ("auto", None):  # background height or baseline level
        list_min = []
        for _, dd in enumerate(Data):
            #            print "k, dd.shape", k, dd.shape
            list_min.append(np.amin(dd))
        start_baseline = list_min
    else:  # input numerical value array
        start_baseline = baseline

    if isinstance(startangles, (int, float)):
        start_anglerot = startangles * np.ones(nb_Images)
    else:
        start_anglerot = startangles

    if isinstance(start_sigma1, (int, float)):
        start_sigma1 = start_sigma1 * np.ones(nb_Images)
    if isinstance(start_sigma2, (int, float)):
        start_sigma2 = start_sigma2 * np.ones(nb_Images)

    if isinstance(boxsize, (int, float)):
        xboxsize, yboxsize = int(boxsize), int(boxsize)
    else:
        xboxsize, yboxsize = boxsize

    halfxboxsize = int(xboxsize / 2.0)
    halfyboxsize = int(yboxsize / 2.0)

    Xhalfboxsize = halfxboxsize * np.ones(nb_Images)
    Yhalfboxsize = halfyboxsize * np.ones(nb_Images)

    start_j = []
    start_i = []
    start_amplitude = []

    if position_start in ("centers", "center"):  # starting position  from input center
        start_j, start_i = halfyboxsize, halfxboxsize
        start_amplitude = []
        d = 0
        for dd in Data:
            start_amplitude.append(dd[int(start_j[d]), int(start_i[d])] - start_baseline[d])
            d += 1
    elif position_start == "max":  # starting position  from maximum intensity in dat

        d = 0
        for dd in Data:
            start_j.append(np.argmax(dd) // dd.shape[1])
            start_i.append(np.argmax(dd) % dd.shape[1])
            start_amplitude.append(np.amax(dd) - start_baseline[d])
            d += 1

    startingparams_zip = np.array([start_baseline, start_amplitude, start_j, start_i,
                            start_sigma1, start_sigma2, start_anglerot])

    RES_params = []
    RES_cov = []
    RES_infodict = []
    RES_errmsg = []

    if verbose:
        print("startingparams_zip", startingparams_zip.T)

    # consider that ROi shape will be constant over all ROIs,
    # so no need to recompute np.indices in gaussfit
    ROIshape = Data[0].shape
    ijindices_array = np.indices(ROIshape)

    k_image = 0
    for startingparams in startingparams_zip.T:
        # if (k_image%25) == 0: print "%d/%d"%(k_image,nb_Images)
        if verbose:
            print("startingparams", startingparams)
        if fitfunc == "gaussian":
            if Data[k_image].shape != ROIshape:
                ijindices_array = None

            params, cov, infodict, errmsg = fit2d.gaussfit(Data[k_image],
                                                            err=None,
                                                            params=startingparams,
                                                            autoderiv=1,
                                                            return_all=1,
                                                            circle=0,
                                                            rotate=1,
                                                            vheight=1,
                                                            xtol=xtol,
                                                            Acceptable_HighestValue=saturation_value,
                                                            Acceptable_LowestValue=0,
                                                            ijindices_array=ijindices_array)

        elif fitfunc == "lorentzian":
            params, cov, infodict, errmsg = fit2d_l.lorentzfit(Data[k_image],
                                                                err=None,
                                                                params=startingparams,
                                                                autoderiv=1,
                                                                return_all=1,
                                                                circle=0,
                                                                rotate=1,
                                                                vheight=1,
                                                                xtol=xtol)

        if showfitresults:
            print("\n *****fitting results ************\n")
            print("  for k_image = ", k_image)
            print(params)
            print("background intensity:                        {:.2f}".format(params[0]))
            print("Peak amplitude above background              {:.2f}".format(params[1]))
            print("pixel position (X)                   {:.2f}".format(
                    params[3] - Xhalfboxsize[k_image] + centers[k_image][0]))  # WARNING Y and X are exchanged
            print("pixel position (Y)                   {:.2f}".format(
                    params[2] - Yhalfboxsize[k_image] + centers[k_image][1]))
            print("std 1,std 2 (pix)                    ( {:.2f} , {:.2f} )".format(
                    params[4], params[5]))
            print("e=min(std1,std2)/max(std1,std2)              {:.3f}".format(
                    min(params[4], params[5]) / max(params[4], params[5])))
            print("Rotation angle (deg)                 {:.2f}".format(params[6] % 360))
            print("- Xboxsize[k_image]", -Xhalfboxsize[k_image])
            print("centers[k_image][0]", centers[k_image][0])
            print("************************************\n")
        bkg_sol, amp_sol, Y_sol, X_sol, std1_sol, std2_sol, ang_sol = params

        RES_cov.append(cov)
        RES_infodict.append(infodict)
        RES_errmsg.append(errmsg)

        params_sol = np.array([bkg_sol, amp_sol,
                X_sol - Xhalfboxsize[k_image] + centers[k_image][0],
                Y_sol - Yhalfboxsize[k_image] + centers[k_image][1],
                std1_sol,
                std2_sol,
                ang_sol])  # now X,Y in safest order

        if offsetposition == 1:
            # PATCH: To match peak positions given by XMAS
            # (confusion coming from counting array indices from 0 or 1...)
            # array fitted by python module see pixels at lower position
            params_sol[3] = params_sol[3] + 1.0
            params_sol[2] = params_sol[2] + 1.0
            # End of PATCH

        elif offsetposition == 2:  # see Compute_data2thetachi() in find2thetachi.py
            # PATCH: To match peak positions given by peaksearch of fit2D
            # in fit2D graphics window first pixel labelled 1 is for the
            # peaksearch located at position in  between 0 and 1 (ie 0.5)
            params_sol[3] = params_sol[3] + 0.5
            params_sol[2] = (framedim[0] - params_sol[2]) + 0.5  # TODO: tocheck dim[0] or dim[1]
            # End of PATCH

        RES_params.append(params_sol)

        k_image += 1

    return RES_params, RES_cov, RES_infodict, RES_errmsg, start_baseline


def getIntegratedIntensities(fullpathimagefile,
                            list_centers,
                            boxsize,
                            CCDLabel="MARCCD165",
                            thresholdlevel=0.2,
                            flipxycenter=True):
    r"""
    read binary image file and compute integrated intensities of peaks
    whose center is given in list_centers

    :return: array whose columns are:
        - integrated intensity
        - absolute minimum intensity threshold
        - nb of pixels composing the peak
    """
    dataimage, framedim, _ = IOimage.readCCDimage(fullpathimagefile, CCDLabel, None, 0)
    res = []
    for center in list_centers:
        res.append(ImProc.getIntegratedIntensity(dataimage, center, boxsize, framedim,
                                                        thresholdlevel, flipxycenter))
    return np.array(res)



def Find_optimal_thresholdconvolveValue(filename, IntensityThreshold, CCDLabel="PRINCETON", verbose=0):
    r"""
    give the lowest value for thresholdconvolve according to IntensityThreshold
    in order not to miss any blob detection in the first filtering
    (by convolution)
    that would be accepted in the second filtering
    (thresholding by IntensityThreshold)
    """
    # image = scipy.misc.lena().astype(float32)
    #    d = readoneimage(filename).reshape((2048,2048))

    res = []
    for tc in (0, 100, 200, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 10000):
        # tstart = ttt.time()
        # Isorted, fitpeak, localpeak, nbrawblobs
        _, fitpeak, _, nbrawblobs = PeakSearch(filename,
                                                CCDLabel=CCDLabel,
                                                PixelNearRadius=20,
                                                removeedge=2,
                                                IntensityThreshold=IntensityThreshold,
                                                thresholdConvolve=tc,
                                                boxsize=15,
                                                position_definition=1,
                                                verbose=0,
                                                fit_peaks_gaussian=0,
                                                xtol=0.001,
                                                return_histo=2)

        # res.append([tc,IntensityThreshold, nbrawblobs,len(fitpeak), ttt.time()-tstart])
        res.append([tc, IntensityThreshold, nbrawblobs, len(fitpeak)])

    Res = np.array(res)
    cond = (Res[:, 2] > Res[:, 3])  # if False  ==> threshold after convolution is too high, some blob may have been rejected before intensity thresholding
    indices_False = np.where(cond == False)[0]
    optim_value = 0
    if len(indices_False) > 1:

        if indices_False[0] != 0:
            optim_value = Res[indices_False[0] - 1, 0]

    if verbose: print("optim value for thresholdConvolve", optim_value)

    return optim_value, Res


def writepeaklist(tabpeaks, output_filename,
                                        outputfolder=None, comments=None, initialfilename=None):
    r"""
    write peaks properties and comments in file with extension .dat added
    """
    outputfilefullpath = IOLT.writefile_Peaklist(output_filename,
                                                tabpeaks,
                                                dirname=outputfolder,
                                                overwrite=1,
                                                initialfilename=initialfilename,
                                                comments=comments)

    return outputfilefullpath


def fitoneimage_manypeaks(filename, peaklist, boxsize, stackimageindex=-1,
                                                    CCDLabel="PRINCETON",
                                                    dirname=None,
                                                    position_start="max",
                                                    type_of_function="gaussian",
                                                    guessed_peaksize=(1.0, 1.0),
                                                    xtol=0.001,
                                                    FitPixelDev=2.0,
                                                    Ipixmax=None,
                                                    MaxIntensity=100000000000,
                                                    MinIntensity=0,
                                                    PeakSizeRange=(0, 200),
                                                    verbose=0,
                                                    position_definition=1,
                                                    NumberMaxofFits=500,
                                                    ComputeIpixmax=False,
                                                    use_data_corrected=None,
                                                    reject_negative_baseline=True,
                                                    purgeDuplicates=True):

    r"""
    fit multiple ROI data to get peaks position in a single image

    Ipixmax  :  highest intensity above background in every ROI centered on element of peaklist

    use_data_corrected   :  enter data instead of reading data from file
                        must be a tuple of 3 elements:
                        fulldata, framedim, fliprot
                        where fulldata is an ndarray

    purgeDuplicates    : True   remove duplicates that are close within pixel distance of 'boxsize' and keep the most intense peak

    use_data_corrected   :  enter data instead of reading data from file
                        must be a tuple of 3 elements:
                        fulldata, framedim, fliprot
                        where fulldata  ndarray

    .. note:: used in PeakSearchGUI
    """
    #     print 'Ipixmax in fitoneimage_manypeaks', Ipixmax
    if len(peaklist) >= NumberMaxofFits:
        print("TOO MUCH peaks to fitdata.")
        print("(in fitoneimage_manypeaks) It may stuck the computer.")
        print("Try to reduce the number of Local Maxima or reduce NumberMaxofFits in fitoneimage_manypeaks()")
        return

    if dirname is not None:
        filename = os.path.join(dirname, filename)

    start_sigma1, start_sigma2 = guessed_peaksize

    tstart = ttt.time()

    ResFit = readoneimage_multiROIfit(filename,
                                        peaklist,
                                        boxsize,
                                        stackimageindex,
                                        CCDLabel=CCDLabel,
                                        baseline="auto",  # min in ROI box
                                        startangles=0.0,
                                        start_sigma1=start_sigma1,
                                        start_sigma2=start_sigma2,
                                        position_start=position_start,  # 'centers' or 'max'
                                        showfitresults=0,
                                        offsetposition=0,  # offset are applied after fit
                                        fitfunc=type_of_function,
                                        xtol=xtol,
                                        addImax=ComputeIpixmax,
                                        use_data_corrected=use_data_corrected)

    if verbose:
        print("fitting time for {} peaks is : {:.4f}".format(len(peaklist), ttt.time() - tstart))
        print("nb of results: ", len(ResFit[0]))

    if ComputeIpixmax:
        params, _, info, _, baseline, Ipixmax = ResFit
    else:
        params, _, info, _, baseline = ResFit

    par = np.array(params)

    #    print "par in fitoneimage_manypeaks", par

    if par == []:
        print("no fitted peaks")
        return

    peak_bkg = par[:, 0]
    peak_I = par[:, 1]
    peak_X = par[:, 2]
    peak_Y = par[:, 3]
    peak_fwaxmaj = par[:, 4]
    peak_fwaxmin = par[:, 5]
    peak_inclination = par[:, 6] % 360

    # pixel deviations from guessed initial position before fitting
    Xdev = peak_X - peaklist[:, 0]
    Ydev = peak_Y - peaklist[:, 1]
    
    # --- --- PEAKS REJECTION -------------------------------
    # print "peaklist[:20]",peaklist[:]
    # number of iteration screening
    to_reject = []
    k = 0
    for inf in info:
        if inf["nfev"] > 1550:
            if verbose:
                print("k= {}   too much iteration".format(k))
            to_reject.append(k)
        k += 1

    if CCDLabel == "FRELONID15_corrected":
        reject_negative_baseline = False
    # negative intensity rejection
    if reject_negative_baseline:
        to_reject2 = np.where((peak_bkg - baseline) < 0)[0]
    else:
        to_reject2 = []

    # too far found peak rejection
    to_reject3 = np.where(np.sqrt(Xdev ** 2 + Ydev ** 2) > FitPixelDev)[0]

    # too intense compared to given threshold (saturation)
    to_reject4 = np.where(peak_I >= MaxIntensity)[0]

    # too weak compared to given threshold
    to_reject5 = np.where(peak_I <= MinIntensity)[0]

    maxpeaksize = np.maximum(peak_fwaxmaj, peak_fwaxmin)

    # too small peak compared to given threshold
    to_reject6 = np.where(maxpeaksize <= PeakSizeRange[0])[0]

    # too large peak compared to given threshold
    to_reject7 = np.where(maxpeaksize >= PeakSizeRange[1])[0]

    if verbose:
        print("to_reject", type(to_reject))
        print("to_reject ...(len)", len(to_reject))
        print(np.take(peaklist, to_reject, axis=0))
        print("to_reject2 ...(len)", len(to_reject2))
        print(np.take(peaklist, to_reject2, axis=0))
        print(np.take(peaklist, to_reject3, axis=0))
        print("After fitting, {}/{} peaks have been rejected\n due to (final - initial position)> FitPixelDev = {}".format(
                len(to_reject3), len(peaklist), FitPixelDev))
        print("{} spots have been rejected\n due to negative baseline".format(len(to_reject2)))
        print("{} spots have been rejected\n due to much intensity ".format(len(to_reject4)))
        print("{} spots have been rejected\n due to weak intensity ".format(len(to_reject5)))
        print("{} spots have been rejected\n due to small peak size".format(len(to_reject6)))
        print("{} spots have been rejected\n due to large peak size".format(len(to_reject7)))

    # spots indices to reject
    ToR = (set(to_reject)
        | set(to_reject2)
        | set(to_reject3)
        | set(to_reject4)
        | set(to_reject5)
        | set(to_reject6)
        | set(to_reject7))  # to reject

    # spot indices to take
    ToTake = set(np.arange(len(peaklist))) - ToR

    if verbose:
        print("index ToTake", ToTake)
        print("nb indices in ToTake", len(ToTake))
    if len(ToTake) < 1:
        return None, par, peaklist

    #     print "Ipixmax",Ipixmax
    if Ipixmax is None:
        Ipixmax = peak_I
    else:
        # ask for maximum intensity in ROI, see
        pass

    # all peaks list building
    tabpeak = np.array([peak_X, peak_Y, peak_I, peak_fwaxmaj, peak_fwaxmin, peak_inclination,
                        Xdev, Ydev, peak_bkg, Ipixmax]).T

    tabpeak = np.take(tabpeak, list(ToTake), axis=0)

    if len(tabpeak.shape) > 1:  # several peaks
        intense_rank = np.argsort(tabpeak[:, 2])[::-1]  # sort by decreasing intensity-bkg
        tabIsorted = tabpeak[intense_rank]
        
    if position_definition == 1:  # XMAS offset
        tabIsorted[:, :2] = tabIsorted[:, :2] + np.array([1, 1])

    if verbose:
        print("\n\nIntensity sorted\n\n")
        print(tabIsorted[:10])
        print("X,Y", tabIsorted[:10, :2])
        print("\n{} fitted peak(s)\n".format(len(tabIsorted)))

    if purgeDuplicates and len(tabIsorted) > 2:
        if verbose: print("Removing duplicates from fit")

        # minimum distance fit solutions
        pixeldistance = boxsize

        # tabXY, index_todelete
        _, index_todelete = GT.purgeClosePoints2(tabIsorted[:, :2], pixeldistance)

        tabIsorted = np.delete(tabIsorted, tuple(index_todelete), axis=0)
        if verbose:
            print(
                "\n{} peaks found after removing duplicates minimum intermaxima distance = {})".format(
                    len(tabIsorted), pixeldistance))

    return tabIsorted, par, peaklist

# =============================================================================
# SKIMAGE CURVE FITTING
# =============================================================================
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage.feature import peak_local_max

def twoD_Gaussian(imgSize, amp0, center_x, center_y, theta, sigma_x, sigma_y):
    theta = np.deg2rad(theta)
    x = np.arange(0, imgSize, 1, float)
    y = np.arange(0, imgSize, 1, float)
    y = y[:,np.newaxis]
    a=np.cos(theta)*x -np.sin(theta)*y
    b=np.sin(theta)*x +np.cos(theta)*y
    a0=np.cos(theta)*center_x -np.sin(theta)*center_y
    b0=np.sin(theta)*center_x +np.cos(theta)*center_y
    g = amp0 * np.exp(-(((a-a0)**2)/(2*(sigma_x**2)) + ((b-b0)**2) /(2*(sigma_y**2))))
    return g

def twoD_GaussianCF(imgSize, amp0, center_x, center_y, theta, sigma_x, sigma_y):
    theta = np.deg2rad(theta)
    x = np.arange(0, imgSize, 1, float)
    y = np.arange(0, imgSize, 1, float)
    y = y[:,np.newaxis]
    a=np.cos(theta)*x -np.sin(theta)*y
    b=np.sin(theta)*x +np.cos(theta)*y
    a0=np.cos(theta)*center_x -np.sin(theta)*center_y
    b0=np.sin(theta)*center_x +np.cos(theta)*center_y
    g = amp0 * np.exp(-(((a-a0)**2)/(2*(sigma_x**2)) + ((b-b0)**2) /(2*(sigma_y**2))))
    return g.ravel()

def peaksearch_skimage(filename, min_dist, pkid_threshold, bs, fit_peaks_gaussian, ccd_label, peak_limits):
    # start_time = time.time()
    data_raw = plt.imread(filename)
    ##BackGround correction with LaueTools
    backgroundimage = ImProc.compute_autobackground_image(data_raw, boxsizefilter=10)
    data_raw = ImProc.computefilteredimage(data_raw, backgroundimage, ccd_label, usemask=True, formulaexpression="A-B")
    # Small median filter to remove any persistent noise
    img = ndi.median_filter(data_raw, size=2)
    # Set the parameters for the peak-finding
    threshold = np.std(img) * pkid_threshold
    # Find the peaks using skimage.feature.peak_local_max
    peak_coords = peak_local_max(img, min_distance=min_dist, threshold_abs=threshold, exclude_border=10)
    peak_coords = np.fliplr(peak_coords)  # This is so that they go more nicely into the following functions
    intensity = []
    boxsize = []
    for i in range(len(peak_coords)):
        px, py = peak_coords[i]
        px, py = int(px), int(py)
        img = data_raw[py-bs:py+bs, px-bs:px+bs]
        if img.shape[0] != 2*bs+1 or img.shape[1] != 2*bs+1:
            boxsize.append(5) #safe box size
        else:
            boxsize.append(bs)
        intensity.append(data_raw[peak_coords[i,1], peak_coords[i,0]])
    intensity = np.array(intensity)
    boxsize = np.array(boxsize)
    intensity = intensity.reshape((len(intensity), 1))
    boxsize = boxsize.reshape((len(boxsize), 1))
    peak_coords = np.hstack((peak_coords, intensity, boxsize))
    
    if len(peak_coords) > peak_limits:
        return None
    
    if fit_peaks_gaussian == 0:
        # all peaks list building
        tabpeak = np.zeros((len(peak_coords),10))
        for ii in range(len(peak_coords)):
            tabpeak[ii,0] = peak_coords[ii,0]#+1
            tabpeak[ii,1] = peak_coords[ii,1]#+1
            tabpeak[ii,2] = peak_coords[ii,2]
            tabpeak[ii,3] = 0
            tabpeak[ii,4] = 0
            tabpeak[ii,5] = 0
            tabpeak[ii,6] = 0
            tabpeak[ii,7] = 0
            tabpeak[ii,8] = 0
            tabpeak[ii,9] = 0
    
    else:
        poptimized = []
        for ii in range(len(peak_coords)):
            # Create data indices
            px, py, Amp, bs = peak_coords[ii]
            px, py, bs = int(px), int(py), int(bs)
            img = data_raw[py-bs:py+bs, px-bs:px+bs]
            px0, py0 = bs, bs
            Amp = np.max(img[py0-1:py0+1, px0-1:px0+1])
            sigmax, sigmay = 1, 1
            theta = 0
            imgSize = img.shape[0]
            p = np.asarray(img).astype('float')
            initial_guess_sum = (Amp, px0, py0, theta, sigmax, sigmay)
            if fit_peaks_gaussian == 1:
                popt, pcov = curve_fit(twoD_GaussianCF, imgSize, np.ravel(p), p0=initial_guess_sum)
            elif fit_peaks_gaussian == 2:
                popt, pcov = curve_fit(twoD_GaussianCF, imgSize, np.ravel(p), p0=initial_guess_sum,
                           bounds=((10, 1, 1, -90, 0.5, 0.5), (1e6, imgSize-1, imgSize-1, 90, 5, 5)))
            elif fit_peaks_gaussian == 3:
                popt, pcov = curve_fit(twoD_GaussianCF, imgSize, np.ravel(p), p0=initial_guess_sum,
                            bounds=((10, bs-1.5, bs-1.5, -90, 0.5, 0.5), (1e6, bs+1.5, bs+1.5, 90, 3, 3)))
            poptimized.append(popt)
    
    
        assert(len(poptimized)==len(peak_coords))
        # all peaks list building
        poptimized = np.array(poptimized)
        tabpeak = np.zeros((len(poptimized),10))
        for ii in range(len(poptimized)):
            px0, py0, _, bs = peak_coords[ii]
            tabpeak[ii,0] = poptimized[ii,1]+px0-bs#+1
            tabpeak[ii,1] = poptimized[ii,2]+py0-bs#+1
            tabpeak[ii,2] = poptimized[ii,0]
            tabpeak[ii,3] = poptimized[ii,4]
            tabpeak[ii,4] = poptimized[ii,5]
            tabpeak[ii,5] = poptimized[ii,3]
            tabpeak[ii,6] = 0
            tabpeak[ii,7] = 0
            tabpeak[ii,8] = 0
            tabpeak[ii,9] = 0

    if len(tabpeak.shape) > 1:  # several peaks
        intense_rank = np.argsort(tabpeak[:, 2])[::-1]  # sort by decreasing intensity-bkg
        tabIsorted = tabpeak[intense_rank]
    # print("Time:", time.time()-start_time)
    
    return tabIsorted, peak_coords


def PeakSearch(filename, stackimageindex=-1, CCDLabel="PRINCETON", center=None,
                                                boxsizeROI=(200, 200),  # use only if center != None
                                                PixelNearRadius=5,
                                                removeedge=2,
                                                IntensityThreshold=400,
                                                thresholdConvolve=200,
                                                paramsHat=(4, 5, 2),
                                                boxsize=15,
                                                verbose=0,
                                                position_definition=1,
                                                local_maxima_search_method=1,
                                                peakposition_definition="max",
                                                fit_peaks_gaussian=1,
                                                xtol=0.00001,
                                                return_histo=1,
                                                FitPixelDev=25,  # to_reject3 parameter
                                                write_execution_time=1,
                                                Saturation_value=65535,  # to be merged in CCDLabel
                                                Saturation_value_flatpeak=65535,
                                                MinIntensity=0,
                                                PeakSizeRange=(0, 200),
                                                Data_for_localMaxima=None,
                                                Fit_with_Data_for_localMaxima=False,
                                                Remove_BlackListedPeaks_fromfile=None,
                                                maxPixelDistanceRejection=15.0,
                                                NumberMaxofFits=5000,
                                                reject_negative_baseline=True,
                                                formulaexpression="A-1.1*B",
                                                listrois=None,
                                                outputIpixmax=True,
                                                mode="skimage"):
    r"""
    Find local intensity maxima as starting position for fittinng and return peaklist.

    :param filename: string, full path to image data file

    :param stackimageindex: integer, index corresponding to the position of image data on a stacked images file
                if -1  means single image data w/o stacking

    :param CCDLabel: string, label for CCD 2D detector used to read the image data file see dict_LaueTools.py

    :param center: position

    .. todo:: to be removed: position of the ROI center in CCD frame

    :param boxsizeROI: dimensions of the ROI to crop the data array
                    only used if center != None

    :param boxsize: half length of the selected ROI array centered on each peak, used for:
                - fitting a peak
                - estimating the background around a peak
                - shifting array in second method of local maxima search (shifted arrays)

    :param IntensityThreshold: integer, pixel intensity level above which potential peaks are kept for fitting position procedure. For local maxima method 0 and 1, this level is relative to zero intensity. For local maxima method 2, this level is relative to lowest intensity in the ROI (local background).
    
    .. note:: Start with high value, because if too high, few peaks are found (only the most important), and if too low, too many local maxima are found leading to time consuming fitting procedure.

    :param thresholdConvolve: integer, pixel intensity level in convolved image above which potential peaks are kept for fitting position procedure. This threshold step on convolved image is applied prior to the local threshold step with IntensityThreshold on initial image (with respect to the local background)

    :param paramsHat: mexican hat kernel parameters (see :func:`LocalMaxima_ndimage`)

    :param PixelNearRadius: integer, pixel distance between two regions considered as peaks.
    
    .. note:: Start rather with a large value. If too low, there are very much peaks duplicates and this is very time consuming.

    :param local_maxima_search_method: integer, Select method for find the local maxima, each of them will fitted
                            - 0   extract all pixel above intensity threshold
                            - 1   find pixels are highest than their neighbours in horizontal, vertica and diagonal direction (up to a given pixel distance)
                            - 2   find local hot pixels which after numerical convolution give high intensity above threshold (thresholdConvolve) then threshold (IntensityThreshold) on raw data

    :param peakposition_definition: 'max' or 'center'  for local_maxima_search_method == 2 to assign to the blob position its hottest pixel position or its center (no weight)

    :param Saturation_value_flatpeak: saturation value of detector for local maxima search method 1

    :param Remove_BlackListedPeaks_fromfile:
        - None
        - file fullpath, str,  to a peaklist file containing peaks that will be deleted in peak list resulting from
        the local maxima search procedure (prior to peak refinement)
        - ndarray of nx2 X Y pixels cooordinates (avoid reading file in peaksearch series)

    :param maxPixelDistanceRejection: maximum distance between black listed peaks and current peaks
                                    (found by peak search) to be rejected

    :param NumberMaxofFits: highest acceptable number of local maxima peak to be refined with a 2D modelPeakSearch

    :param fit_peaks_gaussian:
        - 0  no position and shape refinement procedure performed from local maxima (or blob) result
        - 1  2D gaussian peak refinement
        - 2  2D lorentzian peak refinement

    :param xtol: relative error on solution (x vector)  see args for leastsq in scipy.optimize
    :param FitPixelDev: largest pixel distance between initial (from local maxima search)
                            and refined peak position

    :param position_definition: due to various conventional habits when reading array, add some offset to fitdata XMAS or fit2d peak search values:
        - 0   no offset (python numpy convention)
        - 1   XMAS offset (first pixel is counted as located at 1 instead of 0)
        - 2   fit2d offset (obsolete)

    :param return_histo: - 0   3 output elements
                         - 1   4 elemts, last one is histogram of data
                         - 2   4 elemts, last one is the nb of raw blob found after convolution and threshold

    :param Data_for_localMaxima:  object to be used only for initial step of finding local maxima (blobs) search
                                (and not necessarly for peaks fitting procedure):
                              -  ndarray     = array data
                              - 'auto_background'  = calculate and remove background computed from image data itself (read in file 'filename')
                              - path to image file (string)  = B image to be used in a mathematical operation with Ato current image

    :param Fit_with_Data_for_localMaxima: use 'Data_for_localMaxima' object as image when refining peaks position and shape
                                       with initial peak position guess from local maxima search

    :param formulaexpression: string containing A (raw data array image) and B (other data array image)
                            expressing mathematical operation,e.g:
                            'A-3.2*B+10000'
                            for simple background substraction (with B as background data):
                            'A-B' or 'A-alpha*B' with alpha > 1.

    :param reject_negative_baseline:  True  reject refined peak result if intensity baseline (local background) is negative
                                        (2D model is maybe not suitable)

    :param outputIpixmax: compute maximal pixel intensity for all peaks found

    :return:  -  peak list sorted by decreasing (integrated intensity - fitted bkg)
                -peak_X,peak_Y,peak_I,peak_fwaxmaj,peak_fwaxmin,peak_inclination,Xdev,Ydev,peak_bkg

    for fit_peaks_gaussian == 0 (no fitdata) and local_maxima_search_method==2 (convolution)
        if peakposition_definition ='max' then X,Y,I are from the hottest pixels
        if peakposition_definition ='center' then X,Y are blob center and I the hottest blob pixel

    .. warning:: nb of output elements depends on 'return_histo' argument
    """
    ## New method replacing the old PeakSearch;
    if mode == "skimage":
        print("Skimage mode (strict constraints) of PeakSearch is used for the PeakSearch")
        return peaksearch_skimage(filename, 3, IntensityThreshold, boxsize, 3, CCDLabel, 3000)
    elif mode == "skimage_relaxed":
        print("Skimage mode (relaxed constraints) of PeakSearch is used for the PeakSearch")
        return peaksearch_skimage(filename, 3, IntensityThreshold, boxsize, 2, CCDLabel, 3000)
    elif mode == "skimage_nobounds":
        print("Skimage mode (no constraints) of PeakSearch is used for the PeakSearch")
        return peaksearch_skimage(filename, 3, IntensityThreshold, boxsize, 1, CCDLabel, 3000)
    elif mode == "skimage_nofit":
        print("Skimage mode (No fit) of PeakSearch is used for the PeakSearch")
        return peaksearch_skimage(filename, 3, IntensityThreshold, boxsize, 0, CCDLabel, 3000)
    else:
        print("LaueTools mode of PeakSearch is used for the PeakSearch")
        
    if return_histo in (0, 1):
        return_nb_raw_blobs = 0
    if return_histo in (2,):
        return_nb_raw_blobs = 1
    if write_execution_time:
        t0 = ttt.time()

    # user input its own shaped Data array
    if isinstance(Data_for_localMaxima, np.ndarray):
        # if verbose:
        #     print("Using 'Data_for_localMaxima' ndarray for finding local maxima")
        Data = Data_for_localMaxima

        #         print "min, max intensity", np.amin(Data), np.amax(Data)
        # TODO to test with VHR
        framedim = Data.shape
        ttread = ttt.time()

    # Data are read from image file
    elif isinstance(Data_for_localMaxima, str) or Data_for_localMaxima is None:

        Data, framedim, fliprot = IOimage.readCCDimage(filename,
                                                stackimageindex=stackimageindex,
                                                CCDLabel=CCDLabel,
                                                dirname=None,
                                                verbose=verbose)
        
        # if verbose: print("image from filename {} read!".format(filename))

        # peak search in a single and particular region of image
        if center is not None:

            framediminv = (framedim[1], framedim[0])
            imin, imax, jmin, jmax = ImProc.getindices2cropArray(center, boxsizeROI, framediminv)
            Data = Data[jmin:jmax, imin:imax]

        if write_execution_time:
            dtread = ttt.time() - t0
            ttread = ttt.time()
            # if verbose: print("Read Image. Execution time : {:.3f} seconds".format(dtread))
    if isinstance(Data_for_localMaxima, str):
        # if verbose: print("Using Data_for_localMaxima for local maxima search: --->", Data_for_localMaxima)
        # compute and remove background from this image
        if Data_for_localMaxima == "auto_background":
            # if verbose:
            #     print("computing background from current image ", filename)
            backgroundimage = ImProc.compute_autobackground_image(Data, boxsizefilter=10)
            # basic substraction
            usemask = True
        # path to a background image file
        else:
            if stackimageindex == -1:
                raise ValueError("Use stacked images as background is not implement")
            path_to_bkgfile = Data_for_localMaxima
            if verbose: print("Using image file {} as background".format(path_to_bkgfile))
            try:
                backgroundimage, _, _ = IOimage.readCCDimage(path_to_bkgfile,
                                                                        CCDLabel=CCDLabel)
            except IOError:
                raise ValueError("{} does not seem to be a path file ".format(path_to_bkgfile))

            usemask = False

        # if verbose: print("Removing background for local maxima search")
        Data = ImProc.computefilteredimage(Data, backgroundimage, CCDLabel, usemask=usemask,
                                                            formulaexpression=formulaexpression)

    # if verbose > 1: print("Data.shape for local maxima", Data.shape)

    # --- PRE SELECTION OF HOT PIXELS as STARTING POINTS FOR FITTING ---------
    # first method ---------- "Basic Intensity Threshold"
    if local_maxima_search_method in (0, "0"):

        # if verbose: print("Using simple intensity thresholding to detect local maxima (method 1/3)")
        res = ImProc.LocalMaxima_from_thresholdarray(Data, IntensityThreshold=IntensityThreshold,
                                                    rois=listrois,
                                                    framedim=framedim,
                                                    outputIpixmax=outputIpixmax)
        if res is not None:
            if outputIpixmax:
                peaklist, Ipixmax = res
                ComputeIpixmax = True
            else:
                peaklist = res
                Ipixmax = np.ones(len(peaklist)) * IntensityThreshold
                ComputeIpixmax = False

    # second method ----------- "Local Maxima in a box by shift array method"
    if local_maxima_search_method in (1, "1"):
        # flat top peaks (e.g. saturation) are NOT well detected
        # if verbose: print("Using shift arrays to detect local maxima (method 2/3)")
        peaklist, Ipixmax = ImProc.LocalMaxima_ShiftArrays(Data,
                                        framedim=framedim,
                                        IntensityThreshold=IntensityThreshold,
                                        Saturation_value=Saturation_value_flatpeak,
                                        boxsize_for_probing_minimal_value_background=boxsize,  # 30
                                        pixeldistance_remove_duplicates=PixelNearRadius,  # 25
                                        nb_of_shift=boxsize)  # 25

        ComputeIpixmax = True

    # third method: ------------ "Convolution by a gaussian kernel"
    if local_maxima_search_method in (2, "2"):

        # if verbose: print("Using mexican hat convolution to detect local maxima (method 3/3)")

        peakValConvolve, boxsizeConvolve, central_radiusConvolve = paramsHat

        Candidates = ImProc.LocalMaxima_KernelConvolution(Data,
                                        framedim=framedim,
                                        peakValConvolve=peakValConvolve,
                                        boxsizeConvolve=boxsizeConvolve,
                                        central_radiusConvolve=central_radiusConvolve,
                                        thresholdConvolve=thresholdConvolve,  # 600 for CdTe
                                        connectivity=1,
                                        IntensityThreshold=IntensityThreshold,
                                        boxsize_for_probing_minimal_value_background=PixelNearRadius,
                                        return_nb_raw_blobs=return_nb_raw_blobs,
                                        peakposition_definition=peakposition_definition)

        if Candidates is None:
            # if verbose: print("No local maxima found, change peak search parameters !!!")
            return None

        if return_nb_raw_blobs == 1:
            peaklist, Ipixmax, nbrawblobs = Candidates
        else:
            peaklist, Ipixmax = Candidates

        #         print "len(peaklist)", peaklist.shape
        #         print "Ipixmax", Ipixmax.shape
        ComputeIpixmax = True
    #         print "Ipixmax after convolution method", Ipixmax
    # -------------------------------------------------------------
    # --- END of blobs search methods calls

    if (peaklist is None
        or peaklist is []
        or peaklist is np.array([])
        or (len(peaklist) == 0)):
        # if verbose: print("No local maxima found, change peak search parameters !!!")
        return None
    # pixel origin correction due to ROI croping
    if center is not None:
        x1, y1 = center  # TODO: to ne checked !!
        peaklist = peaklist + np.array([x1, y1])

    if write_execution_time:
        dtsearch = ttt.time() - float(ttread)

        # if verbose: print("Local maxima search. Execution time : {:.3f} seconds".format(dtsearch))

    # removing some duplicates ------------
    if len(peaklist) >= 2:
        nb_peaks_before = len(peaklist)
        #         print "%d peaks in peaklist before purge" % nb_peaks_before
        #         print 'peaklist',in peaklist before purge

        if len(peaklist) >= NumberMaxofFits:
            if verbose:
                print("TOO MUCH peaks to handle.")
                print("(in PeakSearch) It may stuck the computer.")
                print("Try to reduce the number of Local Maxima or\n reduce "
                        "NumberMaxofFits in PeakSearch()")
            return None

        Xpeaklist, Ypeaklist, tokeep = GT.removeClosePoints(peaklist[:, 0], peaklist[:, 1],
                                                                                dist_tolerance=2)

        peaklist = np.array([Xpeaklist, Ypeaklist]).T
        Ipixmax = np.take(Ipixmax, tokeep)

        if verbose: print("Keep {} from {} initial peaks (ready for peak positions and shape fitting)".format(
                len(peaklist), nb_peaks_before))
    # -----------------------------------------------

    #-------------------------------------------------
    # remove black listed peaks option
    # and update peaklist
    if Remove_BlackListedPeaks_fromfile is not None and len(peaklist)>1:
        if not isinstance(Remove_BlackListedPeaks_fromfile, str):
            # array of XY  shape = (n,2)
            XY_blacklisted = Remove_BlackListedPeaks_fromfile

        elif Remove_BlackListedPeaks_fromfile.endswith(('.dat', '.fit')):
            XY_blacklisted = Get_blacklisted_spots(Remove_BlackListedPeaks_fromfile)
        
        if XY_blacklisted is None: print('No or only 1 Blacklisted spots found...')
        else:  #
            X, Y = peaklist[:, :2].T
            (peakX, _, tokeep) = GT.removeClosePoints_two_sets([X, Y], XY_blacklisted,
                                                        dist_tolerance=maxPixelDistanceRejection,
                                                        verbose=0)

            npeak_before = len(X)
            npeak_after = len(peakX)

            if verbose: print("\n Removed {} (over {}) peaks belonging to the blacklist {}\n".format(
                    npeak_before - npeak_after,
                    npeak_before,
                    Remove_BlackListedPeaks_fromfile))

            peaklist = np.take(peaklist, tokeep, axis=0)
            Ipixmax = Ipixmax[tokeep]
    #-------------------------------------------------

    # ---- ----------- no FITTING ----------------------------
    # NO FIT  and return raw list of local maxima
    if fit_peaks_gaussian == 0:

        if position_definition == 1:  # XMAS like offset
            peaklist[:, :2] = peaklist[:, :2] + np.array([1, 1])

        if position_definition == 2:  # fit2D offset
            peaklist[:, 0] = peaklist[:, 0] + 0.5
            peaklist[:, 1] = framedim[0] - peaklist[:, 1] + 0.5

        if verbose:
            print("{} local maxima found".format(len(peaklist)))
            print("20 first peaks", peaklist[:20])

        # tabpeak mimics the array built after fitting procedures
        tabpeak = np.zeros((len(peaklist[:, 0]), 10))
        tabpeak[:, 0] = peaklist[:, 0]
        tabpeak[:, 1] = peaklist[:, 1]
        tabpeak[:, 2] = Ipixmax
        # return tabpeak, peaklist, peaklist, peaklist  # no fitdata return raw list of local maxima
        lastelem = peaklist
        if return_nb_raw_blobs == 1:
            lastelem = nbrawblobs

        return tabpeak, peaklist, peaklist, lastelem

    if (peaklist is None
        or peaklist is []
        or peaklist is np.array([])
        or (len(peaklist) == 0)):
        # print("No local maxima found, no peaks to fit !!!")
        return None

    # ----  ---------------FITTING ----------------------------
    # gaussian fitdata
    elif fit_peaks_gaussian == 1:
        type_of_function = "gaussian"

    # lorentzian fitdata
    elif fit_peaks_gaussian == 2:
        type_of_function = "lorentzian"

    else:
        raise ValueError("optional fit_peaks_gaussian value is not understood! Must be 0,1 or 2")

    if verbose:
        print("\n*****************")
        print("{} local maxima found".format(len(peaklist)))
        print("\n Fitting of each local maxima\n")

    if center is not None:
        position_start = "centers"
    else:
        position_start = "max"

    # if Data_for_localMaxima will be used for refining peak positions
    if Fit_with_Data_for_localMaxima:
        Data_to_Fit = (Data, framedim, fliprot)
    else:
        Data_to_Fit = None

    return fitoneimage_manypeaks(filename,
                                peaklist,
                                boxsize,
                                stackimageindex,
                                CCDLabel=CCDLabel,
                                dirname=None,
                                position_start=position_start,
                                type_of_function=type_of_function,
                                xtol=xtol,
                                FitPixelDev=FitPixelDev,
                                Ipixmax=Ipixmax,
                                MaxIntensity=Saturation_value,
                                MinIntensity=MinIntensity,
                                PeakSizeRange=PeakSizeRange,
                                verbose=verbose,
                                position_definition=position_definition,
                                NumberMaxofFits=NumberMaxofFits,
                                ComputeIpixmax=ComputeIpixmax,
                                use_data_corrected=Data_to_Fit,
                                reject_negative_baseline=reject_negative_baseline)

def Get_blacklisted_spots(filename):
    XY_blacklisted = None
    if filename.endswith('.dat'):

        data_peak_blacklisted = IOLT.read_Peaklist(filename, dirname=None)
        if len(data_peak_blacklisted) > 1:

            XY_blacklisted = data_peak_blacklisted[:, :2].T

    elif filename.endswith('.fit'):
        resdata = IOLT.readfile_fit(filename)
        if resdata is not None:
            allgrainsspotsdata = resdata[4]
            nspots = len(allgrainsspotsdata)
        else:
            nspots = 0

        if nspots > 1:

            XY_blacklisted = allgrainsspotsdata[:, 7:9].T
            
    return XY_blacklisted

def peaksearch_on_Image(filename_in, pspfile, background_flag="no", blacklistpeaklist=None,
                                                        dictPeakSearch={},
                                                        CCDLabel="MARCCD165",
                                                        outputfilename=None,
                                                        psdict_Convolve=PEAKSEARCHDICT_Convolve,
                                                        verbose=0):
    r"""
    Perform a peaksearch by using .psp file

    # still not very used and checked?
    # missing dictPeakSearch   as function argument for formulaexpression  or dict_param??
    """

    dict_param = readPeakSearchConfigFile(pspfile)

    Data_for_localMaxima, formulaexpression = read_background_flag(background_flag)

    blacklistedpeaks_file = set_blacklist_filepath(blacklistpeaklist)

    dict_param["Data_for_localMaxima"] = Data_for_localMaxima
    dict_param["formulaexpression"] = formulaexpression
    dict_param["Remove_BlackListedPeaks_fromfile"] = blacklistedpeaks_file

    # create a data considered as background from an imagefile
    BackgroundImageCreated = False
    flag_for_backgroundremoval = dict_param["Data_for_localMaxima"]

    # flag_for_backgroundremoval is a file path to an imagefile
    # create background data: dataimage_bkg
    if flag_for_backgroundremoval not in ("auto_background", None) and not isinstance(
        flag_for_backgroundremoval, np.ndarray):

        fullpath_backgroundimage = psdict_Convolve["Data_for_localMaxima"]

        #         print "fullpath_backgroundimage ", fullpath_backgroundimage

        # dirname_bkg, imagefilename_bkg = os.path.split(fullpath_backgroundimage)

        # CCDlabel_bkg = CCDLabel

        BackgroundImageCreated = True

        if verbose: print("consider dataimagefile {} as background".format(fullpath_backgroundimage))

        if "formulaexpression" in dictPeakSearch:
            formulaexpression = dictPeakSearch["formulaexpression"]
        else:
            raise ValueError('Missing "formulaexpression" to operate on images before peaksearch in '                                    'peaksearch_fileseries()')

        # saturationlevel = DictLT.dict_CCD[CCDLabel][2]

        # dataimage_corrected = applyformula_on_images(dataimage_raw,
        #                                             dataimage_bkg,
        #                                             formulaexpression=formulaexpression,
        #                                             SaturationLevel=saturationlevel,
        #                                             clipintensities=True)

        if verbose: print("using {} in peaksearch_fileseries".format(formulaexpression))

        # for finding local maxima in image from formula
        psdict_Convolve["Data_for_localMaxima"] = fullpath_backgroundimage

        # for fitting peaks in image from formula
        psdict_Convolve["reject_negative_baseline"] = False
        psdict_Convolve["formulaexpression"] = formulaexpression
        psdict_Convolve["Fit_with_Data_for_localMaxima"] = True

    Res = PeakSearch(filename_in,
                    CCDLabel=CCDLabel,
                    Saturation_value=DictLT.dict_CCD[CCDLabel][2],
                    Saturation_value_flatpeak=DictLT.dict_CCD[CCDLabel][2],
                    **psdict_Convolve)

    if Res in (False, None):
        print("No peak found for image file: ", filename_in)
        return None
    # write file with comments
    Isorted, _, _ = Res[:3]

    if outputfilename:

        params_comments = "Peak Search and Fit parameters\n"

        params_comments += "# {}: {}\n".format("CCDLabel", CCDLabel)

        for key, val in list(psdict_Convolve.items()):
            if not BackgroundImageCreated or key not in ("Data_for_localMaxima",):
                params_comments += "# " + key + " : " + str(val) + "\n"

        if BackgroundImageCreated:
            params_comments += ("# "
                                + "Data_for_localMaxima"
                                + " : {} \n".format(fullpath_backgroundimage))
        # .dat file extension is done in writefile_Peaklist()

        IOLT.writefile_Peaklist("{}".format(outputfilename),
                                Isorted,
                                overwrite=1,
                                initialfilename=filename_in,
                                comments=params_comments)

    return Isorted


# -------------------  CONFIG file functions (.psp)
# import configparser as CONF

# --- ---- Local maxima and fit parameters
CONVERTKEY_dict = {"fit_peaks_gaussian": "fit_peaks_gaussian",
                    "position_definition": "position_definition",
                    "local_maxima_search_method": "local_maxima_search_method",
                    "intensitythreshold": "IntensityThreshold",
                    "thresholdconvolve": "thresholdConvolve",
                    "boxsize": "boxsize",
                    "pixelnearradius": "PixelNearRadius",
                    "xtol": "xtol",
                    "fitpixeldev": "FitPixelDev",
                    "maxpixeldistancerejection": "maxPixelDistanceRejection",
                    "maxpeaksize": "MaxPeakSize",
                    "minpeaksize": "MinPeakSize"}

LIST_OPTIONS_PEAKSEARCH = ["local_maxima_search_method",
                            "IntensityThreshold",
                            "thresholdConvolve",
                            "boxsize",
                            "PixelNearRadius",
                            "fit_peaks_gaussian",
                            "xtol",
                            "FitPixelDev",
                            "position_definition",
                            "maxPixelDistanceRejection",
                            "MinPeakSize",
                            "MaxPeakSize"]

LIST_OPTIONS_TYPE_PEAKSEARCH = ["integer flag",
                                "integer count",
                                "float count",
                                "integer pixel",
                                "integer pixel",
                                "integer flag",
                                "float",
                                "float pixel",
                                "integer flag",
                                "float",
                                "float",
                                "float"]

LIST_OPTIONS_VALUESPARAMS = [1, 1000, 5000, 15, 10, 1, 0.001, 2.0, 1, 15.0, 0.01, 3.0]

if (len(CONVERTKEY_dict) != len(LIST_OPTIONS_PEAKSEARCH)
                            != LIST_OPTIONS_TYPE_PEAKSEARCH != LIST_OPTIONS_VALUESPARAMS):
    raise ValueError(
        "Lists of parameters for config .psp file do not have the same length (readmccd.py)")


def savePeakSearchConfigFile(dict_param, outputfilename=None):
    r""" save peak search parameters in .psp file"""
    # save peaksearch parameter in config file
    config = CONF.RawConfigParser()
    config.add_section("PeakSearch")

    params_comments = "Peak Search and Fit parameters\n"

    for key, val in list(dict_param.items()):
        params_comments += "# " + key + " : " + str(val) + "\n"
        config.set("PeakSearch", key, str(val))

    if outputfilename is None:
        outputfilename = "PeakSearch.psp"

    if not outputfilename.endswith(".psp"):
        if outputfilename.count(".") > 0:

            outputfilename = "".join(outputfilename.split(".")[:-1] + ".psp")
        else:
            outputfilename += ".psp"

    # Writing configuration file to 'PeakSearch.cfg'
    with open(outputfilename, "w") as configfile:
        config.write(configfile)

    return outputfilename


def readPeakSearchConfigFile(filename):
    r""" read peak search parameters in .psp file"""
    config = CONF.RawConfigParser()
    config.optionxform = str
    #    config = MyCasePreservingConfigParser()

    config.read(filename)

    section = config.sections()[0]

    if section not in ("PeakSearch",):
        raise ValueError(
            "wrong section name in config file {}. Must be in {}".format(
                filename, "IndexRefine"))

    dict_param = {}

    list_options = config.options(section)

    for option in list_options:

        #         print "\n option\n", option
        for option_ref, option_type in zip(LIST_OPTIONS_PEAKSEARCH, LIST_OPTIONS_TYPE_PEAKSEARCH):

            #             print "option_ref, option_type", option_ref, option_type

            if option_ref == option or option_ref.lower() == option:

                #                 print "BINGO! I m able to read %s" % option_ref
                #                 print "data type should be: %s" % option_type

                try:
                    optionkey = CONVERTKEY_dict[option_ref]
                except KeyError:
                    optionkey = option_ref

                option_lower = option_ref.lower()

                try:
                    if option_type.startswith("int"):
                        dict_param[optionkey] = int(config.getint(section, option_lower))
                    elif option_type.startswith(("float",)):
                        dict_param[optionkey] = float(config.getfloat(section, option_lower))
                    else:
                        dict_param[optionkey] = config.get(section, option_lower)

                except ValueError:
                    print("Value of option '{}' has not the correct type".format(option))
                    return None

                break

    #     print "Finally, I ve read these parameters"
    #     print dict_param
    return dict_param


def read_background_flag(background_flag, verbose=0):
    r"""
    interpret the background flag (field used in FileSeries/Peak_Search.py)

    return two values to put in dict_param of peaksearch_series
    """
    formulaexpression = "A-B"

    if background_flag in ("auto", "AUTO", "yes", "YES", "y"):
        Data_for_localMaxima = "auto_background"
    elif background_flag in ("n", "no", "NO", None, "None", "NONE"):
        Data_for_localMaxima = None
    else:
        # should be a path to a single imagefile
        # (plus optionally a mathematical formula expression)

        ressplit = background_flag.split(";")

        if len(ressplit) != 2:
            filepath = background_flag
        elif len(ressplit) == 2:
            filepath, formulaexpression = ressplit

        if not os.path.exists(filepath):
            raise ValueError('File %s for background is does not exist'%filepath)

        Data_for_localMaxima = filepath

        if verbose: print("Image file path used for background", Data_for_localMaxima)

    return Data_for_localMaxima, formulaexpression


def set_blacklist_filepath(filepathstr):
    r""" return None or path to file containing black listed spots list"""
    if filepathstr == "None":
        Remove_BlackListedPeaks_fromfile = None
    # fullpath
    else:
        Remove_BlackListedPeaks_fromfile = filepathstr
    return Remove_BlackListedPeaks_fromfile


def set_rois_file(filepathstr):
    """ return None or path to file containing rois list"""
    if filepathstr == "None":
        roisfilepath = None
    # fullpath
    else:
        roisfilepath = filepathstr
    return roisfilepath


# --- -------------- multiple file peak search
def peaksearch_fileseries(fileindexrange,
                            filenameprefix="",
                            suffix="",
                            nbdigits=4,
                            dirname_in="/home/micha/LaueProjects/AxelUO2",
                            outputname=None,
                            dirname_out=None,
                            CCDLABEL="MARCCD165",
                            KF_DIRECTION="Z>0",  # not used yet
                            dictPeakSearch=None,
                            verbose=0,
                            writeResultDicts=0,
                            computetime=0):
    r"""
    peaksearch function to be called for multi or single processing
    """
    if computetime:
        t0 = ttt.time()
    print('\n\n ***** Starting peaksearch_fileseries()  *****\n\n')
    # peak search Parameters update from .psp file
    if isinstance(dictPeakSearch, dict):
        for key, val in list(dictPeakSearch.items()):
            PEAKSEARCHDICT_Convolve[key] = val

    if "MinPeakSize" not in PEAKSEARCHDICT_Convolve:
        PEAKSEARCHDICT_Convolve["MinPeakSize"] = 0.65
        PEAKSEARCHDICT_Convolve["MaxPeakSize"] = 3.
        if verbose: print("Default values for minimal and maximal peaksize are used!. Resp. 0.65 and 3 pixels.")

    PEAKSEARCHDICT_Convolve["PeakSizeRange"] = (copy.copy(PEAKSEARCHDICT_Convolve["MinPeakSize"]),
                                                copy.copy(PEAKSEARCHDICT_Convolve["MaxPeakSize"]))
    del PEAKSEARCHDICT_Convolve["MinPeakSize"]
    del PEAKSEARCHDICT_Convolve["MaxPeakSize"]

    # ----handle reading of filename
    # special case for _mar.tif files...
    if nbdigits in ("varying",):
        pass
    # normal case
    else:
        nbdigits = int(nbdigits)

    if suffix == "":
        suffix = ".mccd"

    if dirname_in != None:
        filenameprefix_in = os.path.join(dirname_in, filenameprefix)
    else:
        filenameprefix_in = filenameprefix

    if outputname != None:
        prefix_outputname = outputname
    else:
        prefix_outputname = filenameprefix  # filename_wo_path[:-len(file_extension) - 1]

    if dirname_out != None:
        prefix_outputname = os.path.join(dirname_out, prefix_outputname)

    if len(fileindexrange) == 2:
        fileindexrange = fileindexrange[0], fileindexrange[1], 1

    # create a data considered as background from an imagefile
    BackgroundImageCreated = False
    flag_for_backgroundremoval = PEAKSEARCHDICT_Convolve["Data_for_localMaxima"]

    # flag_for_backgroundremoval is a file path to an imagefile
    # create background data: dataimage_bkg
    if flag_for_backgroundremoval not in ("auto_background", None) and not isinstance(
        flag_for_backgroundremoval, np.ndarray):

        fullpath_backgroundimage = PEAKSEARCHDICT_Convolve["Data_for_localMaxima"]

        BackgroundImageCreated = True

    # create
    blspots = PEAKSEARCHDICT_Convolve["Remove_BlackListedPeaks_fromfile"]
    if blspots is not None:
        PEAKSEARCHDICT_Convolve["Remove_BlackListedPeaks_fromfile"] = Get_blacklisted_spots(blspots)
        if PEAKSEARCHDICT_Convolve["Remove_BlackListedPeaks_fromfile"] is None:
            print('Warning!! blacklisted spots file may not be well read! blacked spots list is empty', PEAKSEARCHDICT_Convolve["Remove_BlackListedPeaks_fromfile"])

    DictPeaksList = {}
    file_ix, nb_empty_files = 0, 0  # nb of probed file, nb of zero peaks file
    listimageindices = list(range(fileindexrange[0],
                        fileindexrange[1] + 1,
                        fileindexrange[2]))
    nbimages = len(listimageindices)
    nbstepsprogress = 5
    progressstep = 0
    for fileindex in listimageindices:
        # TODO to move this branching elsewhere (readmccd)
        if suffix.endswith("_mar.tif"):
            filename_in = IOimage.setfilename(filenameprefix_in + "{}".format(fileindex) + suffix,
                                                                                        fileindex)
        else:
            #             filename_in = filenameprefix_in + encodingdigits % fileindex + suffix
            filename_in = filenameprefix_in + str(fileindex).zfill(nbdigits) + suffix

        tirets = "-" * 15
        if verbose: print("\n\n {} PeakSearch on filename {}\n{}\n{}{}{}n\n".format(
                tirets, tirets, filename_in, tirets, tirets, tirets))

        if not os.path.exists(filename_in):
            raise ValueError("\n\n*******\nSomething wrong with the filename: {}. Please check "
                                            "carefully the filename!".format(filename_in))

        # remove a single image (considered as background) to current image
        if BackgroundImageCreated:

            if verbose: print("consider dataimagefile {} as background".format(fullpath_backgroundimage))
            # (dataimage_raw, _, _) = IOimage.readCCDimage(filename_in, CCDLabel=CCDLABEL,
            #                                                                         dirname=None)

            if "formulaexpression" in dictPeakSearch:
                formulaexpression = dictPeakSearch["formulaexpression"]
            else:
                raise ValueError('Missing "formulaexpression" to operate on images before '
                                'peaksearch in peaksearch_fileseries()')

            if verbose: print("using {} in peaksearch_fileseries".format(formulaexpression))

            # for finding local maxima in image from formula
            PEAKSEARCHDICT_Convolve["Data_for_localMaxima"] = fullpath_backgroundimage

            # for fitting peaks in image from formula
            PEAKSEARCHDICT_Convolve["reject_negative_baseline"] = False
            PEAKSEARCHDICT_Convolve["formulaexpression"] = formulaexpression
            PEAKSEARCHDICT_Convolve["Fit_with_Data_for_localMaxima"] = True

        # --------------------------
        # launch peaksearch
        # -----------------------
        Res = PeakSearch(filename_in, CCDLabel=CCDLABEL,
                            Saturation_value=DictLT.dict_CCD[CCDLABEL][2],
                            Saturation_value_flatpeak=DictLT.dict_CCD[CCDLABEL][2],
                            **PEAKSEARCHDICT_Convolve)

        if Res in (False, None):
            print("No peak found for image file: ", filename_in)

            nb_empty_files += 1
        #             Isorted, fitpeak, localpeak = None, None, None
        else:  # write file with comments
            Isorted, _, _ = Res[:3]

            params_comments = "Peak Search and Fit parameters\n"

            params_comments += "# {}: {}\n".format("CCDLabel", CCDLABEL)

            for key, val in list(PEAKSEARCHDICT_Convolve.items()):
                if not BackgroundImageCreated or key not in ("Data_for_localMaxima",):
                    params_comments += "# " + key + " : " + str(val) + "\n"

            if BackgroundImageCreated:
                params_comments += ("# " + "Data_for_localMaxima"
                                    + " : {} \n".format(fullpath_backgroundimage))
            # .dat file extension is done in writefile_Peaklist()
            # filename_out = prefix_outputname + encodingdigits % fileindex
            # TODO valid whatever
            filename_out = prefix_outputname + str(fileindex).zfill(nbdigits)
            IOLT.writefile_Peaklist("{}".format(filename_out),
                                        Isorted,
                                        overwrite=1,
                                        initialfilename=filename_in,
                                        comments=params_comments)
            if writeResultDicts:                            
                DictPeaksList[fileindex] = Isorted

        progress = int(np.floor(file_ix/nbimages*nbstepsprogress))
        if progress > progressstep:
            print('Imageindex: %d, Task Progress : %.2f %%' % (fileindex, file_ix / nbimages * 100))
            progressstep += 1
        file_ix += 1

    print("\n\n\n*******************\n\n\n task of peaksearch COMPLETED!")
    if computetime:
        print('Execution time %.2f sec'%(ttt.time()-t0))
    return DictPeaksList, file_ix, nb_empty_files

# def peaksearch_multiprocessing(fileindexrange, filenameprefix, suffix="", nbdigits=4,
#                                                     dirname_in="/home/micha/LaueProjects/AxelUO2",
#                                                     outputname=None,
#                                                     dirname_out=None,
#                                                     CCDLABEL="MARCCD165",
#                                                     KF_DIRECTION="Z>0",
#                                                     dictPeakSearch=None,
#                                                     nb_of_cpu=2,
#                                                     verbose=0,
#                                                     writeResultDicts=0):
#     r"""
#     launch several processes in parallel
#     """
#     import multiprocessing

#     try:
#         if len(fileindexrange) > 2:
#             print("\n\n STEP INDEX is SET to 1 \n\n")
#         index_start, index_final = fileindexrange[:2]
#     except:
#         raise ValueError("Need 2 file indices integers in fileindexrange=(indexstart, indexfinal)")

#     t00 = ttt.time()

#     max_nb_cpus = multiprocessing.cpu_count()
#     nb_cpus = min(nb_of_cpu, max_nb_cpus)

#     fileindexdivision = GT.getlist_fileindexrange_multiprocessing(index_start, index_final, nb_of_cpu)

#     if nb_cpus > 1:
#         print('using %d cpu(s)'%nb_cpus)
#         fileindexdivision = GT.getlist_fileindexrange_multiprocessing(index_start, index_final, nb_cpus)
#         nbimages = index_final - index_start + 1
#         print('dispatch of fileindex ', fileindexdivision)
        
#         peaksearch_fileseries.__defaults__ = (filenameprefix,
#                                             suffix,
#                                             nbdigits,
#                                             dirname_in,
#                                             outputname,
#                                             dirname_out,
#                                             CCDLABEL,
#                                             KF_DIRECTION,
#                                             dictPeakSearch,
#                                             verbose,
#                                             writeResultDicts,
#                                             0) # compute execution time / task

#         pool = multiprocessing.Pool(nb_of_cpu)
#         multiple_results = pool.map(peaksearch_fileseries, fileindexdivision)

#         # DictPeaksList, file_ix, nb_empty_files = multiple_results

#     t_mp = ttt.time() - t00
#     print("Execution time : %.2f" % t_mp)

#     if nb_cpus > 1:
#         nbtreatedimages = 0
#         nbzeropeaksimages = 0
#         for mres in multiple_results:
#             print('nb treated files, nb zero peaks file', mres[1], mres[2])
#             nbtreatedimages += mres[1]
#             nbzeropeaksimages += mres[2]
#         if nbzeropeaksimages != 0:
#             print('total nb of zero peaks file', nbzeropeaksimages)
#         else:
#             print('all %s images contain at least one peak'%nbtreatedimages)

#         # TODO  see end of indexFilesSeries()  to write a log file or hdf5 file with peaks props and other stas, nb of peaks per file, average nb , min and max number

#     return multiple_results, nbtreatedimages, nbzeropeaksimages


def peaklist_dict(prefixfilename, startindex, finalindex, dirname=None):
    r""" create a dict with key=image index and value=list of peaks """
    dict_peaks = {}
    for k in list(range(startindex, finalindex + 1)):
        filename = prefixfilename + "{:04d}.dat".format(k)

        array_peaks = IOLT.read_Peaklist(filename, dirname=dirname)
        dict_peaks[k] = array_peaks

    return dict_peaks


def purgePeaksListFile(filename1, blacklisted_XY, dist_tolerance=0.5, dirname=None):
    r"""
    remove in peaklist .dat file peaks that are in blacklist

    :param blacklisted_XY:         [X1,Y1],[X2,Y2]
    """
    data_peak = IOLT.read_Peaklist(filename1, dirname=dirname)

    XY = data_peak[:, 0:2].T

    blacklisted_XY = np.array(blacklisted_XY).T

    peakX, peakY, tokeep = GT.removeClosePoints_two_sets(XY, blacklisted_XY,
                                            dist_tolerance=dist_tolerance, verbose=0)

    return peakX, peakY, tokeep

def write_PurgedPeakListFile(filename1, blacklisted_XY, outputfilename, dist_tolerance=0.5,
                                                                        dirname=None):
    r"""
    write a new .dat file where peaks in blacklist are omitted
    """
    #peakX, peakY, tokeep
    _, _, tokeep = purgePeaksListFile(filename1, blacklisted_XY, dist_tolerance=0.5, dirname=dirname)

    data_peak = IOLT.read_Peaklist(filename1, dirname=dirname)

    new_data_peak = np.take(data_peak, tokeep, axis=0)

    if dirname is not None:
        outputfilename = os.path.join(dirname, outputfilename)

    IOLT.writefile_Peaklist(outputfilename,
                            new_data_peak,
                            overwrite=1,
                            initialfilename=filename1,
                            comments="Some peaks have been removed by write_PurgedPeakListFile",
                            dirname=dirname)

    print("New peak list file {} has been written".format(outputfilename))


def removePeaks_inPeakList(PeakListfilename,
                            BlackListed_PeakListfilename,
                            outputfilename,
                            dist_tolerance=0.5,
                            dirname=None):
    r"""
    read peaks PeakListfilename and remove those in BlackListed_PeakListfilename
    and write a new peak list file

    .. note:: Not used ??
    """
    data_peak_blacklisted = IOLT.read_Peaklist(BlackListed_PeakListfilename, dirname=dirname)

    XY_blacklisted = data_peak_blacklisted[:, 0:2].T

    write_PurgedPeakListFile(PeakListfilename,
                            XY_blacklisted,
                            outputfilename,
                            dist_tolerance=0.5,
                            dirname=dirname)


def merge_2Peaklist(filename1, filename2, dist_tolerance=5, dirname1=None, dirname2=None, verbose=0):
    r"""
    return merge spots data from two peaklists and removed duplicates within dist_tolerance (pixel)
    """
    data_peak_1 = IOLT.read_Peaklist(filename1, dirname=dirname1)
    data_peak_2 = IOLT.read_Peaklist(filename2, dirname=dirname2)

    XY1 = data_peak_1[:, 0:2]
    XY2 = data_peak_2[:, 0:2]

    #XY, ind_delele_1, ind_delele_2
    _, ind_delele_1, ind_delele_2 = GT.mergelistofPoints(XY1, XY2, dist_tolerance=dist_tolerance,
                                                            verbose=verbose)

    data1 = np.delete(data_peak_1, ind_delele_1, axis=0)
    data2 = np.delete(data_peak_2, ind_delele_2, axis=0)

    return np.concatenate((data1, data2), axis=0)


def writefile_mergedPeaklist(filename1, filename2, outputfilename, dist_tolerance=5,
                                                                    dirname1=None,
                                                                    dirname2=None,
                                                                    verbose=0):
    r"""
    write peaklist file from the merge of spots data from two peaklists
    (and removed duplicates within dist_tolerance (pixel))
    """
    merged_data = merge_2Peaklist(
        filename1, filename2, dist_tolerance, dirname1, dirname2, verbose)
    comments = "Peaks from merging {} and {} with pixel tolerance {:.3f}".format(
        filename1, filename2, dist_tolerance)
    IOLT.writefile_Peaklist(outputfilename, merged_data, 1, None, comments, None)

    print("Merged peak lists written in file: ", outputfilename)
