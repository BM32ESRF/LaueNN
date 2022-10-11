# -*- coding: utf-8 -*-
"""
module of lauetools project

http://sourceforge.net/projects/lauetools/

JS Micha   May 2019

this module gathers functions to read and write ASCII file corresponding
to various data
"""
from __future__ import division
import os
import sys
import time
import string
import copy
from copy import deepcopy
import re

import numpy as np

np.set_printoptions(precision=15)
PYTHON3 = True
try:
    from lauetools.dict_LaueTools import CST_ENERGYKEV, CCD_CALIBRATION_PARAMETERS
except:
    from lauetoolsnn.lauetools.dict_LaueTools import CST_ENERGYKEV, CCD_CALIBRATION_PARAMETERS

DEFAULT_CCDLABEL = 'sCMOS'

# --- ------------  PROCEDURES
def convert2corfile():
    """ from .dat + .det compute .cor file  in LaueGeometry """
    pass


def writefile_cor(prefixfilename, twicetheta, chi, data_x, data_y, dataintensity,
                                            param=None,
                                            initialfilename=None,
                                            comments=None,
                                            sortedexit=0,
                                            overwrite=1,
                                            data_sat=None,
                                            data_props=None,
                                            rectpix=0,  # RECTPIX
                                            dirname_output=None,
                                            verbose=0):
    """
    Write .cor file containing data
    one line   of header
    next lines of data
    last lines of comments

    + comments at the end for calibration CCD parameters that have been used for calculating
    2theta and chi for each peak (in addition to X,Y pixel position)

    :return: outputfilename

    :param sortedexit: - 1 sort peaks by intensity for the outputfile
                - 0 do not sort (e.g. sorting already done in input .dat file . see dataintensity inpu parameter)

    :param overwrite: 1 overwrite existing file
                        0 write a file with '_new' added in the name

    :param rectpix:   to deal with non squared pixel: ypixelsize = xpixelsize * (1.0 + rectpix)

    :param data_props: [array of dataproperties, list columns name]  (ie  peak sizes, pixdev etc...  see .dat file)

    if data_sat list, add column to .cor file to mark saturated peaks
    """
    nbspots = len(twicetheta)

    outputfilename = prefixfilename + ".cor"

    if dirname_output is None:
        dirname_output = os.curdir

    if outputfilename in os.listdir(dirname_output) and not overwrite:
        outputfilename = prefixfilename + "_new" + ".cor"

    outputfile = open(os.path.join(dirname_output, outputfilename), "w")

    if not os.access(dirname_output, os.W_OK):
        print('Can not write in the folder: %s'%dirname_output)
        print('File .cor is not written !')
        return None

    firstline = "2theta chi X Y I"
    format_string = "%.06f   %.06f   %.06f   %.06f   %.03f"
    list_of_data = [twicetheta, chi, data_x, data_y, dataintensity]

    if data_sat is not None:
        firstline += " data_sat"
        format_string += "   %d"
        list_of_data += [data_sat]
    if data_props:
        # print('preparing spots props "list_of_data"')
        data_peaks, columnnames = data_props
        for k in range(len(columnnames)):
            firstline += " %s" % columnnames[k]
            format_string += "   %.06f"
            # TODO clarify ???
            list_of_data += [data_peaks[:, k]]
            #list_of_data += [data_peaks[k, :]]

    firstline += "\n"

    # print('format_string', format_string)
    # print('firstline', firstline)

    if sortedexit:
        # to write in decreasing order of intensity (col intensity =4)
        # print("rearranging exp. spots order according to intensity")
        arraydata = np.array(list_of_data).T
        s_ix = np.argsort(arraydata[:, 4])[::-1]

        sortedarray = arraydata[s_ix]

        list_of_data = sortedarray.T

    outputfile.write(firstline)

    #print('nbspots', nbspots)
    #print('len(list_of_data)', len(list_of_data))
    ldata = [elem for elem in list_of_data]
    #for elem in ldata:
        #print('len ---', len(elem))
        #print('elem ---', elem)
    if PYTHON3:
        liststrs = [format_string % tuple(list(zip(*ldata))[i]) for i in range(nbspots)]
        #liststrs = [format_string % tuple(list(zip((*list_of_data)))[i]) for i in range(nbspots)]
    else:
        liststrs = [format_string % tuple(zip(*ldata)[i]) for i in range(nbspots)]
    outputfile.write("\n".join(liststrs))

    outputfile.write("\n# File created at %s with IOLaueTools.py" % (time.asctime()))

    if initialfilename:
        outputfile.write("\n# From: %s" % initialfilename)

    # metadata on detector position and nature
    if verbose:
        print(' param   in writefile_cor() for prefixfilename %s'%prefixfilename, param)
    if param is not None:
        outputfile.write("\n# Calibration parameters")
        # param is a list
        if isinstance(param, (list, np.ndarray)):
            if len(param) == 6:
                for par, value in list(zip(CCD_CALIBRATION_PARAMETERS[:6], param)):
                    outputfile.write("\n# %s     :   %s" % (par, value))
                ypixelsize = param[5] * (1.0 + rectpix)
                outputfile.write("\n# ypixelsize     :   " + str(ypixelsize))
            elif len(param) == 5:
                for par, value in list(zip(CCD_CALIBRATION_PARAMETERS[:5], param)):
                    outputfile.write("\n# %s     :   %s" % (par, value))
            else:
                raise ValueError("5 or 6 calibration parameters are needed!")
        # param is a dict : CCDCalibdict
        elif isinstance(param, dict):
            if verbose:
                print("param is a dict!")
            for key in CCD_CALIBRATION_PARAMETERS:
                # print('key in CCD_CALIBRATION_PARAMETERS', key)
                if key in param:
                    # print('key in param', key)
                    outputfile.write("\n# %s     :   %s" % (key, str(param[key])))

    if comments:
        outputfile.write("\n# Comments")
        for line in comments:
            outputfile.write("\n# %s" % line)

    outputfile.close()

    # print("(%s) written in %s at the end of writefile_cor()" % (firstline[:-1], outputfilename))
    return outputfilename

def get_otherspotprops(allspotsprops, filename, sortintensities=True):
    """return other spot properties from .cor file (other than 2theta, Chi, X, Y, Intensity)

    :param allspotsprops: array with shape (nb of spots, nb of props)
    """
    assert len(allspotsprops.shape) == 2

    # no additional spot properties
    if allspotsprops.shape[1] == 5:
        return None

    # print('filename', filename, 'allspotsprops  int', allspotsprops[:,4])
    if sortintensities:
        props = allspotsprops[np.argsort(allspotsprops[:, 4])[:: -1]]
    else:
        props = allspotsprops

    # list of props
    otherpropsdata = props[:, 5:].T
    f = open(filename, 'r')
    columnnames = f.readline().split()[5:]
    f.close()

    # print('\n\n      get_otherspotprops()')
    # print('otherpropsdata', otherpropsdata[0])
    # print('columnnames', columnnames)
    # print('\n\n')
    return otherpropsdata, columnnames

def readfile_cor(filename, output_CCDparamsdict=False):
    """
    read peak list in .cor file which is contain 2theta and chi angles for each peak
    .cor file is made of 5 columns

    2theta chi pixX pixY

    :return: alldata                  #array with all spots properties)
            data_theta, data_chi,
            data_pixX, data_pixY,
            data_I,                            # intensity
            detector parameters

    NOTE: detector parameters has been used previously to compute 2theta and chi (angles of kf)
    from pixX and pixY, ie 2theta chi are detector position independent
    (see find2thetachi for definition of kf)

    #TODO: output 2theta ?
    """
    SKIPROWS = 1
    # read first line
    f = open(filename, "r")
    firstline = f.readline()
    unindexeddata = False
    if firstline.startswith("# Unindexed"):
        unindexeddata = True
        SKIPROWS = 7
    f.close()

    if sys.version.split()[0] < "2.6.1":
        f = open(filename, "r")
        alldata = np.loadtxt(f, skiprows=SKIPROWS)
        f.close()
    else:
        #         print "python version", sys.version.split()[0]
        # self.alldata = scipy.io.array_import.read_array(filename, lines = (1,-1))
        alldata = np.loadtxt(filename, skiprows=SKIPROWS)

    # nbspots, nbcolumns = np.shape(self.alldata)
    sha = np.shape(alldata)
    if len(sha) == 2:
        nbcolumns = sha[1]
        nb_peaks = sha[0]
    elif len(sha) == 1:
        nb_peaks = 1
        nbcolumns = sha[0]

    if nb_peaks > 1:

        if nbcolumns == 3:
            data_theta = alldata[:, 0] / 2.0
            data_chi, data_I = alldata.T[1:]
            data_pixX = np.zeros(len(data_chi))
            data_pixY = np.zeros(len(data_chi))
        elif nbcolumns == 5:
            data_theta = alldata[:, 0] / 2.0
            (data_chi, data_pixX, data_pixY, data_I) = alldata.T[1:]
        # case of unindexed file .cor
        elif unindexeddata:
            _, data_I, data_2theta, data_chi, data_pixX, data_pixY = alldata.T
            data_theta = data_2theta / 2.0
        elif nbcolumns > 6:  # .cor file with additional spots properties
            data_2theta, data_chi, data_pixX, data_pixY, data_I = alldata.T[:5]
            data_theta = data_2theta / 2.0
    elif nb_peaks == 1:
        if nbcolumns == 3:
            data_theta = alldata[0] / 2.0
            data_chi, data_I = alldata[1:]
            data_pixX = 0
            data_pixY = 0
        elif nbcolumns == 5:
            data_theta = alldata[0] / 2.0
            (data_chi, data_pixX, data_pixY, data_I) = alldata[1:]

        # case of unindexed file .cor
        elif unindexeddata:
            _, data_I, data_2theta, data_chi, data_pixX, data_pixY = alldata
            data_theta = data_2theta / 2.0
        elif nbcolumns > 6:
            data_2theta, data_chi, data_pixX, data_pixY, data_I = alldata[:5]
            data_theta = data_2theta / 2.0

    #    print "Reading detector parameters if exist"
    openf = open(filename, "r")

    # new way of reading CCD calibration parameters

    CCDcalib = readCalibParametersInFile(openf)
    # print('CCDcalib in readfile_cor() of file %s'%filename, CCDcalib)

    if len(CCDcalib) >= 5:
        detParam = [CCDcalib[key] for key in CCD_CALIBRATION_PARAMETERS[:5]]
        # print("5 CCD Detector parameters read from .cor file: %s"%filename)

    openf.close()

    if output_CCDparamsdict:
        return (alldata, data_theta, data_chi,
                    data_pixX, data_pixY, data_I, detParam, CCDcalib)
    else:
        return (alldata, data_theta, data_chi,
                    data_pixX, data_pixY, data_I, detParam)


def getpixelsize_from_corfile(filename):
    """
    return pixel size if written in .cor file
    """
    xpixelsize = None

    #    print "Reading detector parameters if exist"
    f = open(filename, "r")
    find_xpixelsize = False
    find_ypixelsize = False
    find_pixelsize = False

    for line in f:
        if line.startswith("# pixelsize"):
            find_pixelsize = True
            pixelsize = float(line.split(":")[-1])
            break
        if line.startswith("# xpixelsize"):
            find_xpixelsize = True
            xpixelsize = float(line.split(":")[-1])
        elif line.startswith("# ypixelsize"):
            find_ypixelsize = True
            ypixelsize = float(line.split(":")[-1])
    f.close()

    if find_pixelsize:
        return pixelsize

    if find_xpixelsize and find_ypixelsize:
        if xpixelsize != ypixelsize:
            raise ValueError("Pixels are not square!!")

        return xpixelsize
    else:
        return None


def readfile_det(filename_det, nbCCDparameters=5, verbose=True):
    """
    read .det file and return calibration parameters and orientation matrix used
    """
    f = open(filename_det, "r")
    i = 0

    mat_line = None
    try:
        for line in f:
            i = i + 1
            if i == 1:
                calib = np.array(line.split(",")[:nbCCDparameters], dtype=float)
                if verbose:
                    print("calib = ", calib)
            if i == 6:
                pline = line.replace("[", "").replace("]", "").split(",")
                mat_line = np.array(pline, dtype=float)
                if verbose:
                    print("matrix = ", mat_line.round(decimals=6))
    finally:
        f.close()

    return calib, mat_line


def writeCalibFile():
    """
    # TODO:   put here DetectorCalibration.OnSaveCalib()

    """
    pass

def readCalibParametersInFile(openfile, Dict_to_update=None, guessCCDLabel=True):
    """
    read .det file (detector geometry calibration)

    .. warning:: if CCDLabel is unknown, it is gueesed from pixelsize...

    .. todo:: we could add dimensions to guess CCDLabel

    return dict of parameters in open file
    """
    #CCD_CALIBRATION_PARAMETERS = ["dd", "xcen", "ycen", "xbet", "xgam",
    #                           "pixelsize", "xpixelsize", "ypixelsize",
    #                           "CCDLabel",  "framedim", "detectordiameter", "kf_direction"]
    List_sharpedParameters = ["# %s" % elem for elem in CCD_CALIBRATION_PARAMETERS]

    # print("List_sharpedParameters", List_sharpedParameters)
    if Dict_to_update is None:
        CCDcalib = {}
    else:
        CCDcalib = Dict_to_update

    for line in openfile:
        if line.startswith(tuple(List_sharpedParameters)):
            key, val = line.split(":")
            key_param = key[2:].strip()
            try:
                val = float(val)
            except ValueError:
                val = readStringOfIterable(val.strip())
            CCDcalib[key_param] = val
        if line.startswith("Material"):
            key, val = line.split(":")
            key_param = key[2:].strip()
            CCDcalib[key_param] = val

    if 'ypixelsize' in CCDcalib:
        CCDcalib['pixelsize'] = CCDcalib['ypixelsize']
    if 'xpixelsize' in CCDcalib:
        CCDcalib['pixelsize'] = CCDcalib['xpixelsize']

    # print('CCDcalib in readCalibParametersInFile of file: %s'%openfile, CCDcalib)

    if 'CCDLabel' not in CCDcalib:  #will recognise from pixelsize...
        CCDcalib['CCDLabel'] = None# DEFAULT_CCDLABEL
        if guessCCDLabel:
            if 'pixelsize' in CCDcalib:
                ps = CCDcalib['pixelsize']
                if abs(ps-0.0795) < 0.004:
                    ccdlabel = 'MARCCD165'
                elif abs(ps-0.073) <= 0.002:
                    ccdlabel = 'sCMOS'
                elif abs(ps-0.0365) <= 0.002:
                    ccdlabel = 'sCMOS_16M'
                elif abs(ps-0.075) <= 0.002:
                    ccdlabel = 'psl_weiwei'
                elif abs(ps-0.031) <= 0.002:
                    ccdlabel = 'VHR_Feb13'
                elif abs(ps-0.022) <= 0.002:
                    ccdlabel = 'ImageStar_dia_2021'
                CCDcalib['CCDLabel'] = ccdlabel

    return CCDcalib


def readCalib_det_file(filename_det):
    """
    read .det file and return calibration parameters and orientation matrix used
    """
    f = open(filename_det, "r")

    CCDcalib = readCalibParametersInFile(f)

    f.close()

    calibparam, UB_calib = readfile_det(filename_det, nbCCDparameters=8)

    CCDcalib["framedim"] = calibparam[6:8]
    CCDcalib["detectordiameter"] = max(calibparam[6:8]) * calibparam[5]
    CCDcalib["xpixelsize"] = calibparam[5]
    CCDcalib["pixelsize"] = CCDcalib["xpixelsize"]
    CCDcalib["ypixelsize"] = CCDcalib["xpixelsize"]
    CCDcalib["UB_calib"] = UB_calib

    if "dd" in CCDcalib:
        CCDcalib["CCDCalibParameters"] = [
            CCDcalib[key] for key in CCD_CALIBRATION_PARAMETERS[:5]]
    else:
        CCDcalib["CCDCalibParameters"] = calibparam
        for key, val in list(zip(CCD_CALIBRATION_PARAMETERS[:5], calibparam)):
            CCDcalib[key] = val

    return CCDcalib


def readStringOfIterable(striter):
    """
    extract elements contained in a string and return the list of elements
    (5,9) -> [5,9]
    [2048.0 2048.0] -> [2048,2048]
    """
    if "[" not in striter and not "(" in striter:
        return striter

    ss = striter.strip()[1:-1]

    if "," in ss:
        vals = ss.split(",")
    elif " " in ss:
        vals = ss.split()

    listvals = []
    for elem in vals:
        try:
            val = int(elem)
        except ValueError:
            try:
                val = float(elem)
            except ValueError:
                return striter.strip()
        listvals.append(val)

    return listvals


def writefile_Peaklist(outputprefixfilename, Data_array, overwrite=1,
                                                        initialfilename=None,
                                                        comments=None,
                                                        dirname=None,
                                                        verbose=0):
    """
    Write .dat file

    :param dirname: output file dirname


    containing data
    one line   of header
    next lines of data
    last lines of comments

    WARNING: compute and a column 'peak_Itot'

    TODO: should only write things and not compute !! see intensity calculation!
    (peak_I + peak_bkg)

    TODO: to simplify to deal with single peak recording

    position_definition    0 no offset ,1 XMAS offset , 2 fit2D offset
    (see peaksearch)

    overwrite            : 1 to overwrite the existing file
                            0 to write a file with '_new' added in the name
    """
    if Data_array is None:
        print("No data peak to write")
        return
    # just one row!
    elif len(Data_array.shape) == 1:
        print("single peak to record!")
        nbpeaks, nbcolumns = 1, Data_array.shape[0]
    else:
        nbpeaks, nbcolumns = Data_array.shape
        if Data_array.shape == (1, 10):
            Data_array = Data_array[0]

    if dirname is None:
        dirname = os.curdir

    outputfilename = outputprefixfilename + ".dat"

    if outputfilename in os.listdir(os.curdir) and not overwrite:
        outputfilename = outputfilename + "_new" + ".dat"

    if nbpeaks == 1:
        Data_array = np.array([Data_array, Data_array])

    if nbcolumns == 10:
        (peak_X, peak_Y, peak_I, peak_fwaxmaj, peak_fwaxmin, peak_inclination,
        Xdev, Ydev, peak_bkg, Ipixmax, ) = Data_array.T

    elif nbcolumns == 11:
        (peak_X, peak_Y, _, peak_I, peak_fwaxmaj, peak_fwaxmin, peak_inclination,
            Xdev, Ydev, peak_bkg, Ipixmax, ) = Data_array.T

    elif nbcolumns == 3: # basic X, Y , I
        # need to set fake data
        (peak_X, peak_Y, peak_I) = Data_array.T
        (peak_fwaxmaj, peak_fwaxmin, peak_inclination,
        Xdev, Ydev, peak_bkg) = np.zeros((6, nbpeaks))
        Ipixmax = 500*np.ones(nbpeaks)

    outputfile = open(os.path.join(dirname, outputfilename), "w")

    outputfile.write("peak_X peak_Y peak_Itot peak_Isub peak_fwaxmaj peak_fwaxmin "
                                                "peak_inclination Xdev Ydev peak_bkg Ipixmax\n")

    if nbpeaks == 1:
        print("nbcolumns", nbcolumns)

        outputfile.write(
            "\n%.02f   %.02f   %.02f   %.02f   %.02f   %.02f    %.03f   %.02f   %.02f   %.02f   %d"
            % (np.round(peak_X[0], decimals=2),
                np.round(peak_Y[0], decimals=2),
                np.round(peak_I[0] + peak_bkg[0], decimals=2),
                np.round(peak_I[0], decimals=2),
                np.round(peak_fwaxmaj[0], decimals=2),
                np.round(peak_fwaxmin[0], decimals=2),
                np.round(peak_inclination[0], decimals=2),
                np.round(Xdev[0], decimals=2),
                np.round(Ydev[0], decimals=2),
                np.round(peak_bkg[0], decimals=2),
                int(Ipixmax[0])))

        nbpeaks = 1

    else:

        outputfile.write(
            "\n".join(
                ["%.02f   %.02f   %.02f   %.02f   %.02f   %.02f    %.03f   %.02f   %.02f   %.02f   %d"
                    % tuple(list(zip(peak_X.round(decimals=2),
                                peak_Y.round(decimals=2),
                                (peak_I + peak_bkg).round(decimals=2),
                                peak_I.round(decimals=2),
                                peak_fwaxmaj.round(decimals=2),
                                peak_fwaxmin.round(decimals=2),
                                peak_inclination.round(decimals=2),
                                Xdev.round(decimals=2),
                                Ydev.round(decimals=2),
                                peak_bkg.round(decimals=2),
                                Ipixmax))[i]
                    ) for i in list(range(nbpeaks))]))
        nbpeaks = len(peak_X)

    outputfile.write("\n# File created at %s with IOLaueTools.py" % (time.asctime()))
    if initialfilename:
        outputfile.write("\n# From: %s" % initialfilename)

    outputfile.write("\n# Comments: nb of peaks %d" % nbpeaks)
    if comments:
        outputfile.write("\n# " + comments)

    outputfile.close()

    if verbose:
        print("table of %d peak(s) with %d columns has been written in \n%s"
            % (nbpeaks, nbcolumns, os.path.join(os.path.abspath(dirname), outputfilename)))

    return os.path.join(os.path.abspath(dirname), outputfilename)


def addPeaks_in_Peaklist(
    filename_in, data_new_peaks, filename_out=None, dirname_in=None, dirname_out=None):
    """
    create or update peak list according to a new peaks data
    """

    data_current_peaks = read_Peaklist(filename_in, dirname=dirname_in)

    if data_new_peaks.shape[1] != data_current_peaks.shape[1]:
        raise ValueError("Data to be merged have not the same number of columns")

    # merge data
    raw_merged_data = np.concatenate((data_new_peaks, data_current_peaks), axis=0)

    # sort by peak amplitude (column #3)
    merged_data = raw_merged_data[np.argsort(raw_merged_data[:, 3])[::-1]]

    print("merged_data", merged_data)

    if dirname_in is not None:
        filename_in = os.path.join(dirname_in, filename_in)

    f = open(filename_in, "r")
    comments = ""
    incomments = False
    while True:
        line = f.readline()

        if line.startswith("#"):
            incomments = True
            #print(line)
            comments += line

        elif incomments:
            break
    f.close()

    #print(merged_data.shape)

    if filename_out is None:
        filename_out == filename_in
    else:
        if dirname_out is not None:
            filename_out = os.path.join(dirname_out, filename_out)

    writefile_Peaklist(filename_out,
                        merged_data,  # last column is computed inside functions
                        overwrite=1,
                        initialfilename=None,
                        comments=comments,
                        dirname=dirname_out)

    return merged_data


def readfile_dat(filename_in, dirname=None, returnnbpeaks = False):
    """ call simply read_Peaklist()"""
    return read_Peaklist(filename_in, dirname=dirname, returnnbpeaks=returnnbpeaks)


def read_Peaklist(filename_in, dirname=None, output_columnsname=False, returnnbpeaks=False):
    """
    read peak list .dat file and return the entire array of spots data

    (peak_X,peak_Y,peak_Itot, peak_Isub,peak_fwaxmaj,peak_fwaxmin,
    peak_inclination,Xdev,Ydev,peak_bkg, Pixmax)
    """
    if dirname is not None:
        filename_in = os.path.join(dirname, filename_in)

    SKIPROWS = 1

    with open(filename_in, 'r') as ff:
        lineindex = 0
        commentfound = False
        while not commentfound:
            _line = ff.readline()

            if lineindex == 0:
                columnsname = _line.split()

            if _line.startswith('# File created'):
                nbdatarows = lineindex-1
                commentfound = True
            elif _line.startswith('# Comments: nb of peaks'):
                nbpeaks = int(_line.split('peaks')[-1])
            elif _line.startswith('# Remove_BlackListedPeaks_fromfile'):
                commentfound = True

            lineindex += 1

    data_peak = np.loadtxt(filename_in, skiprows=SKIPROWS, max_rows=nbdatarows)
    if len(data_peak.shape) == 1:
        foundNpeaks = 1
    else:
        foundNpeaks = len(data_peak)


    if output_columnsname:
        return data_peak, columnsname

    elif returnnbpeaks:
        return data_peak, foundNpeaks
    else:
        return data_peak


def writefitfile(outputfilename,
                 datatooutput,
                 nb_of_indexedSpots,
                 dict_matrices=None,
                 meanresidues=None,
                 PeakListFilename=None,
                 columnsname=None,
                 modulecaller=None,
                 refinementtype="Strain and Orientation"):
    """
    write a .fit file
    
    :param outputfilename: full path of outputfilename
    """
    # HEADER
    header = "%s Refinement from experimental file: %s\n" % (refinementtype, PeakListFilename)
    modulecallerstr = ""
    if modulecaller is not None:
        modulecallerstr = " with %s" % modulecaller
    header += "File created at %s%s\n" % (time.asctime(), modulecallerstr)
    header += "Number of indexed spots: %d\n" % nb_of_indexedSpots

    if "Element" in dict_matrices:
        header += "Element\n"
        header += "%s\n" % str(dict_matrices["Element"])

    if "grainIndex" in dict_matrices:
        header += "grainIndex\n"
        header += "G_%d\n" % dict_matrices["grainIndex"]

    if meanresidues is not None:
        header += "Mean Deviation(pixel): %.3f\n" % meanresidues

    if columnsname:
        header += columnsname.rstrip()

    else:
        header += "spot_index : !!columns name missing !!"

    # FOOTER
    footer = ""

    if "UBmat" in dict_matrices:
        footer += "UB matrix in q= (UB) B0 G* \n"
        #            outputfile.write(str(self.UBB0mat) + '\n')
        footer += str(dict_matrices["UBmat"].round(decimals=9)) + "\n"

    # added by OR-------------------------
    if "Umat2" in dict_matrices:
        footer += "Umatrix in q_lab= (UB) B0 G* \n"
        #            outputfile.write(str(self.UBB0mat) + '\n')
        footer += str(dict_matrices["Umat2"].round(decimals=9)) + "\n"

    if "Bmat_tri" in dict_matrices:
        footer += "Bmatrix in q_lab= (UB) B0 G* \n"
        #            outputfile.write(str(self.UBB0mat) + '\n')
        footer += str(dict_matrices["Bmat_tri"].round(decimals=9)) + "\n"

        footer += "(B-I)*1000 \n"
        #            outputfile.write(str(self.UBB0mat) + '\n')
        smattri = (dict_matrices["Bmat_tri"] - np.eye(3)) * 1000.0
        footer += str(smattri.round(decimals=3)) + "\n"

    if ("HKLxyz_names" in dict_matrices) and ("HKLxyz" in dict_matrices):
        footer += "HKL coord. of lab and sample frame axes :\n"
        for k in list(range(6)):
            footer += dict_matrices["HKLxyz_names"][k] + "\t"
            footer += str(dict_matrices["HKLxyz"][k].round(decimals=3)) + "\n"

    # ---------- end OR

    if "B0" in dict_matrices:
        footer += "B0 matrix in q= UB (B0) G*\n"
        footer += str(dict_matrices["B0"].round(decimals=8)) + "\n"

    if "UBB0" in dict_matrices:
        footer += "UBB0 matrix in q= (UB B0) G* i.e. recip. basis vectors are columns "
        footer += "in LT frame: astar = UBB0[0,:], bstar = UBB0[1,:], cstar = UBB0[2,:]. (abcstar as lines on xyzlab1, "
        footer += "xlab1 = ui, ui = unit vector along incident beam)\n"
        footer += str(dict_matrices["UBB0"].round(decimals=8)) + "\n"

    if "euler_angles" in dict_matrices:
        footer += "Euler angles phi theta psi (deg)\n"
        footer += str(dict_matrices["euler_angles"]) + "\n"

    if "mastarlab" in dict_matrices:
        footer += "matstarlab , abcstar on xyzlab2, ylab2 = ui : astar_lab2 = matstarlab[0:3] "
        footer += ",bstar_lab2 = matstarlab[3:6], cstar_lab2 = matstarlab[6:9] \n"
        footer += str(dict_matrices["matstarlab"].round(decimals=7)) + "\n"

    if "matstarsample" in dict_matrices:
        footer += "matstarsample , abcstar on xyzsample2, xyzsample2 obtained by rotating xyzlab2 "
        footer += "by MG.PAR.omega_sample_frame around xlab2, astar_sample2 = matstarsample[0:3] "
        footer += ",bstar_sample2 = matstarsample[3:6], cstar_lab2 = matstarsample[6:9] \n"
        footer += str(dict_matrices["matstarsample"].round(decimals=8)) + "\n"

    if "devstrain_crystal" in dict_matrices:
        footer += "deviatoric strain in direct crystal frame (10-3 unit)\n"
        footer += str((dict_matrices["devstrain_crystal"] * 1000.0).round(decimals=2)) + "\n"

    if "devstrain_sample" in dict_matrices:
        footer += "deviatoric strain in sample2 frame (10-3 unit)\n"
        footer += str((dict_matrices["devstrain_sample"] * 1000.0).round(decimals=2)) + "\n"

    if "LatticeParameters" in dict_matrices:
        footer += "new lattice parameters\n"
        footer += str(dict_matrices["LatticeParameters"].round(decimals=7)) + "\n"

    if "CCDLabel" in dict_matrices:
        footer += "CCDLabel\n"
        footer += str(dict_matrices["CCDLabel"]) + "\n"

    if "detectorparameters" in dict_matrices:
        footer += "DetectorParameters\n"
        footer += str(dict_matrices["detectorparameters"]) + "\n"

    if "pixelsize" in dict_matrices:
        footer += "pixelsize\n"
        footer += str(dict_matrices["pixelsize"]) + "\n"

    if "framedim" in dict_matrices:
        footer += "Frame dimensions\n"
        footer += str(dict_matrices["framedim"]) + "\n"

    if "Ts" in dict_matrices:
        if dict_matrices["Ts"] is not None:
            footer += "Refined T transform elements in %s\n" % dict_matrices["Ts"][1]
            footer += str(dict_matrices["Ts"][2]) + "\n"

    outputfile = open(outputfilename, "wb")
    np.savetxt(outputfilename, datatooutput, fmt="%.6f",
               header=header, footer=footer, comments="#")
    outputfile.close()


def ReadASCIIfile(_filename_data, col_2theta=0, col_chi=1, col_Int=-1, nblineskip=1):
    """ from a file
    return 3 arrays of columns located at index given by
    col_2theta=0, col_chi=1, col_Int=-1:
    [0] theta
    [1] chi
    [2] intensity

    # Quite basic and useless function
    """
    _tempdata = np.loadtxt(_filename_data, skiprows=nblineskip)
    # _tempdata = scipy.io.array_import.read_array(_filename_data, lines = (nblineskip,-1))

    _data_theta = _tempdata[nblineskip - 1 :, col_2theta] / 2.0
    _data_chi = _tempdata[nblineskip - 1 :, col_chi]

    try:
        _data_I = _tempdata[nblineskip - 1 :, col_Int]
    except IndexError:
        print("there are not 5 columns in data.cor file")
        print("I create then a uniform intensity data!")
        _data_I = np.ones(len(_data_theta))
    if (np.array(_data_I) < 0.0).any():
        print("Strange ! I don't like negative intensity...")
        print("I create then a uniform intensity data!")
        _data_I = np.ones(len(_data_theta))

    return (_data_theta, _data_chi, _data_I)


def readfile_fit(fitfilename, verbose=0, readmore=False,
                                        fileextensionmarker=(".fit", ".cor", ".dat"),
                                        returnUnindexedSpots=False,
                                        return_columnheaders=False,
                                        return_toreindex=False):
    """ call alias function readfitfile_multigrains()"""
    return readfitfile_multigrains(fitfilename, verbose, readmore,
                                        fileextensionmarker,
                                        returnUnindexedSpots,
                                        return_columnheaders,
                                        return_toreindex)

def readfitfile_multigrains(fitfilename, verbose=0, readmore=False,
                                        fileextensionmarker=(".fit", ".cor", ".dat"),
                                        returnUnindexedSpots=False,
                                        return_columnheaders=False,
                                        return_toreindex=False):
    """
    JSM version of multigrain.readlt_fit_mg()
    read a single .fit file containing data for several grains

    fileextensionmarker :  '.fit' extension at the end of the line
                            stating that a new grain data starts

    return            : list_indexedgrains_indices,
                        list_nb_indexed_peaks,
                        list_starting_rows_in_data,
                        all_UBmats_flat,
                        allgrains_spotsdata,
                       calibJSM[:, :5],
                       pixdev, strain6, euler

               where   list_indexedgrains_indices   : list of indices of indexed grains
                       list_nb_indexed_peaks        : list of numbers of indexed peaks for each grain
                       list_starting_rows_in_data    : list of starting rows in spotsdata for reading grain's spots data

                        all_UBmats_flat           : all 1D 9 elements UBmat matrix
                                                in q = UBmat B0 G* in Lauetools Frame (ki//x)
                                                WARNING! not OR or labframe (ki//y) !!!
                        allgrains_spotsdata        :   array of all spots sorted by grains
                        calibJSM[:, :5]            : contains 5 detector geometric parameters
                        pixdev                    : list of pixel deviations after fit for each grain
                        strain6                    : list of 6 elements (voigt notation) of deviatoric strain
                                                    in 10-3 unit for each grain in CRYSTAL Frame
                                                    from Lauetools calculation
                        euler                    : list of 3 Euler Angles for each grain

    """
    if verbose:
        print("reading fit file %s by readfitfile_multigrains.py of IOLaueTools (formerly readwriteASCII): " % fitfilename)

    columns_headers = []

    f = open(fitfilename, "r")

    # search for each start of grain dat
    nbgrains = 0
    linepos_grain_list = []
    lineindex = 1
    WrongExtension = False
    try:
        for line in f:
            _line = line.rstrip(string.whitespace)
            if not _line.startswith("# Unindexed and unrefined"):
                if _line.endswith(fileextensionmarker):
                    nbgrains += 1
                    linepos_grain_list.append(lineindex)
                else:
                    WrongExtension = True
            lineindex += 1
        if WrongExtension:
            print('Warning !! Strange extension file for the first line of %s'%fitfilename)
    finally:
        linepos_grain_list.append(lineindex)
        f.close()

    if verbose:
        print("nbgrains = ", nbgrains)
        print("linepos_grain_list = ", linepos_grain_list)

    # nothing has been indexed
    if nbgrains == 0:
        return None

    list_indexedgrains_indices = list(range(nbgrains))

    all_UBmats_flat = np.zeros((nbgrains, 9), float)
    strain6 = np.zeros((nbgrains, 6), float)
    calib = np.zeros((nbgrains, 5), float)
    calibJSM = np.zeros((nbgrains, 7), float)
    euler = np.zeros((nbgrains, 3), float)
    list_nb_indexed_peaks = np.zeros(nbgrains, int)
    list_starting_rows_in_data = np.zeros(nbgrains, int)
    pixdev = np.zeros(nbgrains, float)

    Material_list = []
    GrainName_list = []
    PixDev_list = []

    dataspots_Unindexed = []

    # read .fit file for each grain

    UBmat = np.zeros((3, 3), dtype=np.float)
    strain = np.zeros((3, 3), dtype=np.float)

    matrixfound = 0
    calibfound = 0
    calibfoundJSM = 0
    pixdevfound = 0
    strainfound = 0
    eulerfound = 0
    linecalib = 0
    linepixdev = 0
    #linestrain = 0
    lineeuler = 0

    f = open(fitfilename, "r")
    unindexedspots = False
    for grain_index in list(range(nbgrains)):

        iline = linepos_grain_list[grain_index]

        nb_indexed_spots = 0

        while iline < linepos_grain_list[grain_index + 1]:

            line = f.readline()
            #             print "iline =%d line" % iline, line
            if line.startswith(("# Number of indexed spots", "#Number of indexed spots")):
                # print("iline =%d line" % iline, line)
                try:
                    nb_indexed_spots = int(line.split(":")[-1])
                except ValueError:
                    print("number of indexed should places after ':' ")

            elif line.startswith(("# Number of unindexed spots", "#Number of unindexed spots")):
                nb_indexed_spots = 0
                nb_UNindexed_spots = int(line.split(":")[-1])
                unindexedspots = True

            elif line.startswith(("# Mean Pixel Deviation", "#Mean Deviation", "#Mean Pixel Deviation")):
                meanpixdev = float(line.split(":")[-1])
                PixDev_list.append(meanpixdev)

            elif line.startswith("#Element"):
                line = f.readline()
                Material_list.append(line.rstrip("\n"))
                iline += 1
            elif line.startswith("#grainIndex"):
                line = f.readline()
                GrainName_list.append(line.rstrip("\n"))

                iline += 1
            elif line.startswith(("spot#", "#spot", "##spot", "# Spot")):
                if not unindexedspots:
                    columns_headers = line.replace("#", "").split()

                if nb_indexed_spots > 0:
                    nbspots = nb_indexed_spots

                    dataspots = []
                    for _ in list(range(nbspots)):
                        line = f.readline()
                        iline += 1
                        dataspots.append(
                            line.rstrip("\n").replace("#", "").replace("[", "").replace("]", "").split())

                    dataspots = np.array(dataspots, dtype=np.float)

                elif nb_UNindexed_spots > 0:
                    nbspots = nb_UNindexed_spots

                    dataspots_Unindexed = []
                    for _ in list(range(nbspots)):
                        line = f.readline()
                        iline += 1
                        dataspots_Unindexed.append(
                            line.rstrip("\n").replace("#", "").replace("[", "").replace("]", "").split())

                    dataspots_Unindexed = np.array(dataspots_Unindexed, dtype=np.float)
            #                     print "got dataspots_Unindexed!"

            elif line.startswith("#UB"):
                matrixfound = 1

                #lineendspot = iline - 1
                # print "matrix found"
            elif line.startswith("#Sample"):
                calibfound = 1
                linecalib = iline + 1
            elif line.startswith(("# Calibration", "#Calibration")):
                calibfoundJSM = 1
                linecalib = iline + 1
            elif line.startswith("#pixdev"):
                pixdevfound = 1
                linepixdev = iline + 1
            elif line.startswith("#deviatoric"):
                strainfound = 1

            elif line.startswith("#Euler"):
                eulerfound = 1
                lineeuler = iline + 1

                print('Eulers Angles found', eulerfound)

            if matrixfound:
                for jline_matrix in list(range(3)):
                    line = f.readline()
                    # print("line in matrix matrixfound", line)
                    lineval = (line.rstrip("\n").replace("#", "").replace("[", "").replace("]", "").split())
                    UBmat[jline_matrix, :] = np.array(lineval, dtype=float)
                    iline += 1
                #                 print "got UB matrix:", UBmat
                matrixfound = 0
            if strainfound:
                for jline_matrix in list(range(3)):
                    line = f.readline()
                    lineval = (line.rstrip("\n").replace("#", "").replace("[", "").replace("]", "").split())
                    strain[jline_matrix, :] = np.array(lineval, dtype=float)
                    iline += 1
                #                 print "got strain matrix:", strain
                strainfound = 0
            if calibfoundJSM:
                calibparam = []
                for _ in list(range(7)):
                    line = f.readline()
                    val = float(line.split(":")[-1])
                    calibparam.append(val)
                    iline += 1
                #                 print "got calibration parameters:", calibparam
                calibJSM[grain_index, :] = calibparam
                calibfoundJSM = 0

            if calibfound & (iline == linecalib):
                calib[grain_index, :] = np.array(line.split(",")[:5], dtype=float)
                # print "calib = ", calib[grain_index,:]
            if eulerfound & (iline == lineeuler):
                euler[grain_index, :] = np.array(
                    line.replace("[", "").replace("#", "").replace("]", "").split()[:3], dtype=float)
                # print "euler = ", euler[grain_index,:]
            if pixdevfound & (iline == linepixdev):
                pixdev[grain_index] = float(line.rstrip("\n"))
                # print "pixdev = ", pixdev[grain_index]

            iline += 1

        list_nb_indexed_peaks[grain_index] = np.shape(dataspots)[0]

        #        if min_matLT == True :
        #            matmin, transfmat = FindO.find_lowest_Euler_Angles_matrix(UBmat)
        #            UBmat = matmin
        #            print "transfmat \n", list(transfmat)
        #            # transformer aussi les HKL pour qu'ils soient coherents avec matmin
        #            hkl = data_fit[:, 2:5]
        #            data_fit[:, 2:5] = np.dot(transfmat, hkl.transpose()).transpose()

        all_UBmats_flat[grain_index, :] = np.ravel(UBmat)

        # xx yy zz yz xz xy
        # voigt notation
        strain6[grain_index, :] = np.array([strain[0, 0],
                                            strain[1, 1],
                                            strain[2, 2],
                                            strain[1, 2],
                                            strain[0, 2],
                                            strain[0, 1]])

        if grain_index == 0:
            allgrains_spotsdata = dataspots * 1.0
        elif grain_index:
            allgrains_spotsdata = np.row_stack((allgrains_spotsdata, dataspots))

    f.close()

    for grain_index in list(range(1, nbgrains)):
        list_starting_rows_in_data[grain_index] = (list_starting_rows_in_data[grain_index - 1]
                                                    + list_nb_indexed_peaks[grain_index - 1])

    pixdev = np.array(PixDev_list, dtype=np.float)

    if verbose:
        print("list_indexedgrains_indices = ", list_indexedgrains_indices)
        print("all_UBmats_flat = ")
        print(all_UBmats_flat)
        print("list_nb_indexed_peaks = ", list_nb_indexed_peaks)
        print("list_starting_rows_in_data = ", list_starting_rows_in_data)
        print("pixdev = ", pixdev.round(decimals=4))
        print("strain6 = \n", strain6.round(decimals=2))
        print("euler = \n", euler.round(decimals=3))

    if not readmore:
        toreturn = (list_indexedgrains_indices,
                    list_nb_indexed_peaks,
                    list_starting_rows_in_data,
                    all_UBmats_flat,
                    allgrains_spotsdata,
                    calibJSM[:, :5],
                    pixdev)
    elif readmore:
        toreturn = (list_indexedgrains_indices,
                    list_nb_indexed_peaks,
                    list_starting_rows_in_data,
                    all_UBmats_flat,
                    allgrains_spotsdata,
                    calibJSM[:, :5],
                    pixdev,
                    strain6,
                    euler)
    if return_toreindex:
        toreturn = (list_indexedgrains_indices,
                    list_nb_indexed_peaks,
                    pixdev,
                    Material_list,
                    all_UBmats_flat,
                    calibJSM[:, :5])

    if columns_headers is not []:
        dict_column_header = {}
        for k, col_name in enumerate(columns_headers):
            dict_column_header[col_name] = k
    else:
        if return_columnheaders:
            raise ValueError("problem reading columns name")

    if returnUnindexedSpots:
        _res = toreturn, dataspots_Unindexed
    else:
        _res = toreturn

    if return_columnheaders:
        return _res, dict_column_header
    else:
        return _res

def readfitfile_comments(fitfilepath):
    """read comments in .fit file and return corresponding strings
    #CCDLabel
    #pixelsize
    #Frame dimensions
    #DetectorParameters
    """
    dictcomments = {}
    ccdlabelflag = False
    pixelsizeflag = False
    framedimflag = False
    detectorflag = False

    f = open(fitfilepath, "r")
    nblines = 0
    for line in f.readlines():
        nblines += 1
    f.seek(0)
    lineindex = 0
    while lineindex < nblines:
        line = f.readline()
        # print('lineeeeeeeeee', line)
        if ccdlabelflag:
            dictcomments['CCDLabel'] = line.split('#')[1].strip()
            ccdlabelflag = False
        if pixelsizeflag:
            dictcomments['pixelsize'] = line.split('#')[1].strip()
            pixelsizeflag = False
        if framedimflag:
            dictcomments['framedim'] = line.split('#')[1].strip()
            framedimflag = False
        if detectorflag:
            dictcomments['detectorparameters'] = line.split('#')[1].strip()
            detectorflag = False

        if line.startswith(('#CCDLabel', "# CCDLabel")):
            ccdlabelflag = True
        if line.startswith(('#pixelsize', "# pixelsize")):
            pixelsizeflag = True
        if line.startswith(('#Frame dimensions', "# Frame dimensions")):
            framedimflag = True
        if line.startswith(('#DetectorParameters', "# DetectorParameters")):
            detectorflag = True
        lineindex += 1
    f.close()

    return dictcomments


def convert_fit_to_cor(fitfilepath, verbose=0):
    """ convert .fit file to .cor file"""

    col_2theta, col_chi = 9, 10
    col_Xexp, col_Yexp = 7, 8
    col_intensity = 1

    # output_corfilepath = fitfilepath[-4:] + '.cor'

    folder, filename = os.path.split(fitfilepath)
    prefixfilename = filename.rsplit(".", 1)[0]

    if verbose:
        print('\n in convert_fit_to_cor')
        print('filename', filename)
        print('prefixfilename: %s\n'%prefixfilename)

    # read .fit file
    res = readfitfile_multigrains(fitfilepath)

    alldata = res[4]

    #   (nb spots,  nb properties/spots)  sorted by grainindex
    if verbose:
        print('alldata.shape', alldata.shape)

    (twicetheta, chi,
    data_x, data_y, dataintensity) = (alldata[:, col_2theta], alldata[:, col_chi],
                            alldata[:, col_Xexp], alldata[:, col_Yexp], alldata[:, col_intensity])

    dictcoms = readfitfile_comments(fitfilepath)
    if 'pixelsize' in dictcoms:
        pixelsize = dictcoms['pixelsize']
        # print('pix', pixelsize)
    if 'CCDLabel' in dictcoms:
        CCDLabel = dictcoms['CCDLabel']
        # print('ccdlabel',CCDLabel)

    if 'detectorparameters' in dictcoms:
        detparams = dictcoms['detectorparameters']
        detectorparameters = readStringOfIterable(detparams)
        # print('detparam',detectorparameters)

    listfield = ["dd", "xcen", "ycen", "xbet", "xgam", "pixelsize",
                                "xpixelsize", "ypixelsize", "CCDLabel"]
    listval = detectorparameters + [pixelsize, pixelsize, pixelsize, CCDLabel]
    dictparam = {}
    for key, val in zip(listfield, listval):
        dictparam[key] = val

    # print('dictparam in convert_fit_to_cor()', dictparam)

    # write .cor file
    filecor = writefile_cor(prefixfilename, twicetheta, chi, data_x, data_y, dataintensity,
                                    param=dictparam,
                                    dirname_output=folder,
                                    overwrite=1)

    return os.path.join(folder, filecor)


def read3linesasMatrix(fileobject):
    """
    return matrix from reading 3 lines in fileobject
    """
    matrix = np.zeros((3, 3))
    iline = 0
    for i in list(range(3)):
        line = fileobject.readline()

        listval = re.split("[ ()\[\)\;\,\]\n\t\a\b\f\r\v]", line)
        listelem = []
        for elem in listval:
            if elem not in ("",):
                val = float(elem)
                listelem.append(val)

        #print("listelem", listelem)

        matrix[i, :] = np.array(listelem, dtype=float)
        iline += 1
    if iline == 3:
        return matrix


def readListofIntegers(fullpathtoFile):
    """ parse a string of list of integers """
    fileobject = open(fullpathtoFile, "r")

    nbElements = 0

    lines = fileobject.readlines()
    listelem = []
    for line in lines:

        listval = re.split("[ ()\[\)\;\:\!\,\]\n\t\a\b\f\r\v]", line)

        for elem in listval:
            if elem not in ("",):
                try:
                    val = int(float(elem))
                except ValueError:
                    return None
                listelem.append(val)
                nbElements += 1

    return listelem

def read_roisfile(fullpathtoFile):
    """ read a file where each line is x,y,boxx,boxy (all integers or will be converted in integers)
    """
    f = open(fullpathtoFile, "r")

    nbRois = 0

    lines = f.readlines()
    listrois = []
    for line in lines:

        listval = re.split("[ ()\[\)\;\:\!\,\]\n\t\a\b\f\r\v]", line)
        nbelems = 0
        listroielems = []
        for elem in listval:
            if elem not in ("",):
                try:
                    val = int(float(elem))
                except ValueError:
                    print("can't convert %s into integer...! I give up"%elem)
                    return None
                listroielems.append(val)
                nbelems += 1
        if nbelems == 4:
            listrois.append(listroielems)
            nbRois += 1
        else:
            print("\n\n**********\n\nSorry. I can't extract 4 integers from the line %s. "
                "I give up...****\n\n" % line)
            return None

    return listrois


def readListofMatrices(fullpathtoFile):
    """ Read ASCII file containing n matrices elements. Must be 9*n float numbers

    :param fullpathtoFile: full path
    :type fullpathtoFile: str
    :raises ValueError: if number of floats is not a multiple of 9
    :return: nb of matrices found, list of matrices
    :rtype: (int, list)
    """

    fileobject = open(fullpathtoFile, "r")

    nbElements = 0

    lines = fileobject.readlines()
    listelem = []
    for line in lines:

        listval = re.split("[ ()\[\)\;\,\]\n\t\a\b\f\r\v]", line)

        for elem in listval:
            if elem not in ("",):
                val = float(elem)
                listelem.append(val)
                nbElements += 1

    if (nbElements % 9) != 0:
        raise ValueError("Number of elements is not a multiple of 9")

    nbMatrices = nbElements // 9
    matrices = np.array(listelem, dtype=float).reshape((nbMatrices, 3, 3))
    return nbMatrices, matrices


def readCheckOrientationsFile(fullpathtoFile):
    """
    read .ubs file

    return tuple of two elements:
    [0] nb of Material in output
    [1] list of infos:
        [0] fileindices ROI infos
        [1] Material for indexation
        [2] Energy max and minimum matching rate threshold (nb of coincidence / nb of theo. spots)
        [3] nb of matrices to be tested
        [4] matrix or list of matrices

    # design of .mats file aiming at giving infos of guesses UB matrix solutions
    prior to indexation from scratch

    Hierarchical tree structure  FileIndex  / Grain / Material / EnergyMax / MatchingThreshold / Matrix(ces)

    --- Fileindex1  --Grain- Material 1-1 -EnergyMax -MatchingThreshold- Matrix(ces)
                    |
                    --Grain- Material 1-2 --- Matrix(ces)
                    |
                    --Grain -  ...

    --- Fileindex2  --- Material 2-1 --- Matrix(ces)
                    |
                    --- Material 2-2 --- Matrix(ces)
                    |
                    ---  ...

    --- Fileindex3  --- Material 3-1 --- Matrix(ces)
                    |
                    --- Material 3-2 --- Matrix(ces)
                    |
                    ---  ...

    When using this file, current fileindex will be searched among the  Fileindex3 sets.
    If found, guessed Material and matrices will be then tested before indexation from scratch

    return: list of CheckOrientation

    where each CheckOrientation is a list of:
    - File index (list or -2 for all images)
    - Grain index
    - Material
    - Energy Max
    - MatchingThreshold
    - Matrix(ces)

    .. note:: this .ubs file must be accompanied by .irp with the same nb of materials in the same order (to perform the refinement)

    example.ubs--------
    $FileIndex
    [0,1,2,3,4,5]
    $Grain
    0
    $Material
    Current
    $EnergyMax
    22
    $MatchingThreshold
    50
    $Matrix
    [[0.5165,0.165,-.95165],
    [0.3198951498,-0.148979,0.123126],
    [-.4264896,.654128,-.012595747]]
    $Material
    Cu
    $Matrix
    [[0.8885165,0.0000165,-.777795165],
    [0.100003198951498,-74440.148979,0.155242423126],
    [-.54264896,.99999654128,-.572785747]]
    $FileIndex
    [6,7,8]
    $Material
    Ge
    $Matrix
    [[0.5165,0.165,-.95165],
    [0.3198951498,-0.148979,0.123126],
    [-.4264896,.654128,-.012595747]]
    [[0.8885165,0.0000165,-.777795165],
    [0.100003198951498,-74440.148979,0.155242423126],
    [-.54264896,.99999654128,-.572785747]]
    $Material
    Current
    $Matrix
    [[0.8885165,0.0000165,-.777795165],
    [0.100003198951498,-74440.148979,0.155242423126],
    [-.54264896,.99999654128,-.572785747]]
    $FileIndex
    All
    $Material
    Current
    $Matrix
    [[0.5165,0.165,-.95165],
    [0.3198951498,-0.148979,0.123126],
    [-.4264896,.654128,-.012595747]]
    END

    substrate_and_grains.ubs--------
    $FileIndex
    All
    $Grain
    0
    $Material
    Si
    $EnergyMax
    22
    $MatchingThreshold
    50
    $Matrix
    [[0.5165,0.165,-.95165],
    [0.3198951498,-0.148979,0.123126],
    [-.4264896,.654128,-.012595747]]
    $Grain
    1
    $Material
    Cu
    $Matrix
    [[0.8885165,0.0000165,-.777795165],
    [0.100003198951498,-74440.148979,0.155242423126],
    [-.54264896,.99999654128,-.572785747]]
    END
    """

    List_posImageIndex = []
    List_posGrain = []
    List_posMaterial = []
    List_posEnergyMax = []
    List_posMatchingThreshold = []
    List_posMatrices = []

    List_CheckOrientations = []

    known_values = [False for k in list(range(6))]
    Current_CheckOrientationParameters = [0 for k in list(range(6))]

    f = open(fullpathtoFile, "r")
    lineindex = 0
    while 1:
        line = f.readline()
        #print("line file.ubs",line)
        if line.startswith("$"):
            if line.startswith("$FileIndex"):
                line = str(f.readline())
                #print("$FileIndex: ", line)
                list_indices = getfileindex(line)
                Current_CheckOrientationParameters[0] = list_indices
                known_values[0] = True
                List_posImageIndex.append(lineindex)
                lineindex += 1
            elif line.startswith("$Grain"):
                line = f.readline()
                grain_index = int(line)
                Current_CheckOrientationParameters[1] = grain_index
                known_values[1] = True
                List_posGrain.append(lineindex)
                lineindex += 1
            elif line.startswith("$Material"):
                line = f.readline()
                key_material = str(line).strip()
                Current_CheckOrientationParameters[2] = key_material
                known_values[2] = True
                List_posMaterial.append(lineindex)
                lineindex += 1
            elif line.startswith("$EnergyMax"):
                line = f.readline()
                energymax = int(line)
                Current_CheckOrientationParameters[3] = energymax
                known_values[3] = True
                List_posEnergyMax.append(lineindex)
                lineindex += 1
            elif line.startswith("$MatchingThreshold"):
                line = f.readline()
                matchingthreshold = float(line)
                Current_CheckOrientationParameters[4] = matchingthreshold
                known_values[4] = True
                List_posMatchingThreshold.append(lineindex)
                lineindex += 1
            elif line.startswith("$Matrix"):
                nbMatrices, matrices, nblines, posfile = readdataasmatrices(f)
                #print("nbMatrices,matrices, nblines, posfile",
                    # nbMatrices,
                    # matrices,
                    # nblines,
                    # posfile)

                Current_CheckOrientationParameters[5] = matrices
                known_values[5] = True
                List_posMatrices.append(lineindex)

                #print("Current_CheckOrientationParameters", Current_CheckOrientationParameters)
                #print("known_values", known_values)

                List_CheckOrientations.append(copy.copy(Current_CheckOrientationParameters))

                if posfile != -1:
                    f.seek(posfile)
                    for _ in list(range(nblines)):
                        f.readline()
                        lineindex += 1
                else:
                    f.close()
                    break

    return List_CheckOrientations


def getfileindex(str_expression):
    #print("str_expression", str_expression)
    if str_expression.strip() in ("all", "All"):
        return -1

    list_val = str_expression.strip("[]()\n").split(",")
    #print("list_val", list_val)
    integerlist = [int(float(elem)) for elem in list_val]
    return integerlist


def readdataasmatrices(fileobject):
    """ read matrices in ubs file?
    """
    posfile = fileobject.tell()

    nbElements = 0

    nblines = 1
    lines = []

    while True:
        line = str(fileobject.readline())
        #print("line matrix", line)
        if line.startswith("$"):
            break
        if line.strip() in ("END",):
            posfile = -1
            break
        lines.append(line)
        nblines += 1

    listelem = []
    for line in lines:

        listval = re.split("[ ()\[\)\;\,\]\n\t\a\b\f\r\v]", line)

        for elem in listval:
            if elem not in ("",):
                val = float(elem)
                listelem.append(val)
                nbElements += 1

                #print('listelem', listelem)

    if (nbElements % 9) != 0:
        raise ValueError("Number of elements is not a multiple of 9")

    nbMatrices = nbElements // 9
    matrices = np.array(listelem, dtype=float).reshape((nbMatrices, 3, 3))

    return nbMatrices, matrices, nblines - 1, posfile


def writefile_log(output_logfile_name="lauepattern.log", linestowrite=[[""]]):
    """
    TODO: maybe useless ?
    """
    filou = open(output_logfile_name, "w")
    aecrire = linestowrite
    for line in aecrire:
        lineData = "\t".join(line)
        filou.write(lineData)
        filou.write("\n")
    filou.close()


def Writefile_data_log(grainspot, index_of_grain, linestowrite=[[""]], cst_energykev=CST_ENERGYKEV):
    """
    write a log data file of simulation
    """
    for elem in grainspot:
        linestowrite.append([str(index_of_grain),
                            str(elem.Millers[0]),
                            str(elem.Millers[1]),
                            str(elem.Millers[2]),
                            str(elem.EwaldRadius * cst_energykev),
                            str(elem.Twicetheta),
                            str(elem.Chi),
                            str(elem.Xcam),
                            str(elem.Ycam)])


def writefilegnomon(gnomonx, gnomony, outputfilename, dataselected):
    """
    write file with gnomonic coordinates
    """
    linestowrite = []
    linestowrite.append(["gnomonx gnomony 2theta chi I #spot"])
    nb = len(gnomonx)
    for i in list(range(nb)):
        linestowrite.append([str(gnomonx[i]),
                            str(gnomony[i]),
                            str(2.0 * dataselected[0][i]),
                            str(dataselected[1][i]),
                            str(dataselected[2][i]),
                            str(i)])

    naname = outputfilename + ".gno"
    outputfile = open(naname, "w")
    for line in linestowrite:
        lineData = "\t".join(line)
        outputfile.write(lineData)
        outputfile.write("\n")
    outputfile.close()


def ReadSummaryFile(filename, dirname=None):
    """
    read summary .dat file generated by multigrain

    one line per grain and per image
    """
    fullpath = filename
    if dirname is not None:
        fullpath = os.path.join(dirname, filename)

    f = open(fullpath, "r")
    # skip first line
    f.readline()

    # read columns name
    columns = f.readline()
    list_cols = columns.split(" ")
    list_column_names = []
    dict_column_names = {}
    for k, elem in enumerate(list_cols):
        list_column_names.append(elem)
        dict_column_names[elem] = k

    data = np.loadtxt(f, dtype=np.float)  # , comments, delimiter, converters, skiprows, usecols, unpack, ndmin)

    f.close()

    # remove last elem = '\n'
    del dict_column_names["\n"]
    return data, list_column_names[:-1], dict_column_names


def createselecteddata(tupledata_theta_chi_I, _listofselectedpts, _indicespotmax):
    """
    select part of peaks in peaks data

    From theta,chi,intensity
    returns the same arrays with less points (if _indicespotmax>=1)
    TODO: to document, and to extend to posX and posY
    """
    _data_theta, _data_chi, _data_I = tupledata_theta_chi_I

    if _indicespotmax < 1:
        # all selected data are considered
        _nbmax = len(_listofselectedpts)
    else:
        _nbmax = min(_indicespotmax, len(_listofselectedpts))

    cutlistofselectedpts = _listofselectedpts[:_nbmax]
    # print cutlistofselectedpts
    # print "nb of selected data points",len(cutlistofselectedpts)
    if cutlistofselectedpts is None:
        _dataselected = np.array(np.zeros((3, len(_data_theta)), dtype=np.float))
        _dataselected[1] = _data_chi
        _dataselected[0] = _data_theta
        _dataselected[2] = _data_I
    else:
        # _dataselected=np.array(zeros((3,len(cutlistofselectedpts)),'float'))
        _dataselected = np.array(np.zeros((3, len(cutlistofselectedpts)), dtype=np.float))
        _dataselected[0] = np.take(_data_theta, cutlistofselectedpts)
        _dataselected[1] = np.take(_data_chi, cutlistofselectedpts)
        _dataselected[2] = np.take(_data_I, cutlistofselectedpts)

    return (_dataselected, _nbmax)


def ReadSpec(fname, scan, outputdate=False):
    """
    Procedure very based on that of Vincent Favre Nicolin procedure

    return :
    spec command (str), dict of data (key=counter name, val= values), and optionnaly [date (str)]
    """
    f = open(fname, "r")

    print('ReadSpec of IOLaueTools.py')

    s = "#S %d" % scan
    title = 0

    bigmca = []
    # spec command with motors, steps and exposure
    while 1:
        title = f.readline()
        if s == title[0 : len(s)]:
            break
        if len(title) == 0:
            break
    print(title)
    # date -----------------
    date = 'unknown'
    s = "#D"
    while 1:
        line = f.readline()
        if s == line[0:len(s)]:
            date = line[3:]
            break
        if len(line) == 0:
            break
    # data   (dict with key= counter column name) -------------
    s = "#L"
    coltit = 0

    while 1:
        coltit = f.readline()
        if s == coltit[0 : len(s)]:
            break
        if len(coltit) == 0:
            break
    d = {}
    coltit = coltit.split()
    for i in list(range(1, len(coltit))):
        d[coltit[i]] = []

    ii = 0
    while 1:  # reading data
        l = f.readline()
        if len(l) < 2:
            break

        if l[:2] == "#C":            
            # deal with#C Thu Feb 25 23:54:05 2021.  Erreur com with laueT.
            if 'Erreur com' in l:
                print('line error #C :', l)
                continue
            elif not l.startswith("#C tiltcomp:"):
                print("Scan aborted after %d point(s)" % ii)
                break
            else:
                print(l)
        if l[0] == "#":
            continue
        l = l.split()
        # print "l",l
        #         print "coltit", coltit
        # print "nb columns",len(coltit)-1
        if l[0] != "@A":
            for i in list(range(1, len(coltit))):
                d[coltit[i]].append(float(l[i - 1]))
        else:
            # print "reading mca data array for one point"
            bill = np.zeros((128, 16))  # 2048=128*16
            mcadata = []
            l[-1] = l[-1][:-1]
            mcadata.append(np.array(l[1:]))
            bill[0] = np.array(np.array(l[1:]), dtype=np.int16)
            # fist line has its first element = '@A'
            if ii % 10 == 0:
                print("%d" % ii)
            for k in list(range(1, 127)):  # first and last line off , each line contains 16 integers
                l = f.readline()

                l = l.split()
                l[-1] = l[-1][:-1]
                mcadata.append(np.array(l[:16]))
                # print "uihuihui ",k,"   ",array(l)
                bill[k] = np.array(np.array(l[:16]), dtype=np.int16)
            # last line doesn't finish with \
            l = f.readline()
            l = l.split()
            # print array(l)
            mcadata.append(np.array(l))
            bill[-1] = np.array(np.array(l), dtype=np.int16)

            # bill=array(mcadata,dtype=uint16)
            bigmca.append(np.ravel(bill))
            ii += 1

            d["mca"] = np.array(bigmca)

    nb = len(d[coltit[1]])  # nb of points
    # print "nb",nb
    for i in list(range(1, len(coltit))):
        a = np.zeros(nb, dtype=float)
        for j in list(range(nb)):
            a[j] = d[coltit[i]][j]
        d[coltit[i]] = deepcopy(a)
    f.close()
    if outputdate:
        return title, d, date
    else:
        return title, d


# --- -----  read write Parameters file
class readwriteParametersFile:
    """
    class in (old) developement
    """

    def __init__(self):
        pass

    def loadParamsFile(self, filename, dirname=None):
        with open(filename) as fh:
            self.attrs = []
            for line in fh:
                if not line.startswith(("#", "!", "-")):
                    print("line", line)
                    s = line.strip(" \n").split("=")
                    print("s", s)
                    attr_name = s[0].strip()
                    print("attr_names", attr_name)
                    if s != [""]:
                        if len(s) == 2:
                            setattr(self, attr_name, s[1])
                        elif len(s) == 3:
                            setattr(self, attr_name, s[1:])
                        self.attrs.append(attr_name)

    def getParamsDict(self):
        print(self.attrs)


# ---  --- XMAS file related functions
def readxy_XMASind(filename):
    """
    read XMAS indexation file
    and return:
    x(exp)  y(exp)  h  k  l  ang.dev. xdev(pix)  ydev(pix)  energy(keV)  theta(deg)  intensity   integr  xwidth(pix)   ywidth(pix)  tilt(deg)  rfact   pearson  xcentroid(pix)   ycentroid(pix)

    usage:
    dataindXMAS = readxy_XMASind('Ge_run41_1_0003_1.ind')
    X_XMAS = dataindXMAS[:, 0]
    Y_XMAS = dataindXMAS[:, 1]

    # for nspots=array([0,1,2,3,4])
    pixX, pixY = np.transpose(np.take(dataindXMAS[:, :2], (1, 5, 4, 10, 7), axis=0))

    # not much used now!
    """
    f = open(filename, "r")

    _ = f.readline()  #filename_mccd
    f.readline()
    f.readline()
    l = f.readline()
    nb_peaks = int(l.split()[-1])
    f.readline()
    datalines = []
    for _ in list(range(nb_peaks)):
        datalines.append(f.readline().split())

    return np.array(datalines, dtype=float)


def read_cri(filecri):
    """
    file .cri of XMAS
    """
    # fichier type : attention parametres a b c en nanometres

    # Al2O3 crystal (hexagonal axis)
    # 167
    # 0.47588   0.47588   1.29931  90.00000  90.00000 120.00000
    # 2
    # Al001    0.00000   0.00000   0.35230   1.00000
    # O0001    0.30640   0.00000   0.25000   1.00000

    uc = np.zeros(6, dtype=np.float)
    element_name = []
    #sg_num = 0

    VERBOSE = 0

    print("reading crystal structure from file :  ", filecri)
    f = open(filecri, "r")
    i = 0
    try:
        for line in f:
            if (i == 0) & VERBOSE:
                print("comment : ", line[:-1])
            if i == 1:
                if VERBOSE:
                    print("space group number : ", line[:-1])
                #sg_num = int(line.split()[0])
                # print sg_num
            if i == 2:
                if VERBOSE:
                    print("unit cell parameters (direct space) : ", line[:-1])
                for j in list(range(6)):
                    if j < 3:
                        # print line.split()[j]
                        uc[j] = float(line.split()[j]) * 1.0
                    else:
                        uc[j] = float(line.split()[j])
                # print uc
            if i == 3:
                if VERBOSE:
                    print("number of atoms in asymmetric unit : ", line[:-1])
                num_at = int(line.split()[0])
                # print num_at

            if i > 3:
                # print "new line", line
                # print line.split()
                if np.size(line.split()) > 0:
                    if ((line.split()[0])[1]).isalpha():
                        element_name.append((line.split()[0])[0:2])
                    else:
                        element_name.append((line.split()[0])[0:1])
            i = i + 1

    finally:
        #linetot = i
        f.close()

    linestart = 4
    #lineend = num_at + 4

    # to use loadtxt ?...
    element_coord_and_occ = np.genfromtxt(filecri, usecols=(1, 2, 3, 4), skip_header=linestart)

    if VERBOSE:
        print("element_coord_and_occ = \n", element_coord_and_occ)

        print("element_name =\n", element_name)

        print("%d atom(s) in asymmetric unit :" % num_at)
        if num_at > 1:
            for i in list(range(num_at)):
                print(element_name[i], "\t", element_coord_and_occ[i, :])
        else:
            print(element_name, "\t", element_coord_and_occ)

    return uc


def readfile_str(filename, grain_index):
    """
    read XMAS .str file

    return for one grain (WARNING: grain_index  starting from 1)
    data_str:
    matstr
    calib
    dev_str

    WARNING: endline does not have space character, this is really annoying
    upgraded scipy.io.array_import to np.genfromtxt
    TODO: to be refactored (JSM Feb 2012)
    """

    print("reading info from STR file : \n", filename)
    # print "peak list, calibration, strained orientation matrix, deviations"
    # print "change sign of HKL's"
    f = open(filename, "r")
    i = 0
    grainfound = 0
    calib = np.zeros(5, dtype=np.float)
    # x(exp)  y(exp)  h  k  l xdev  ydev  energy  dspace  intens   integr
    # xwidth   ywidth  tilt  rfactor   pearson  xcentroid  ycentroid
    # 0 x(exp)    # 1 y(exp)
    # 2 3 4 h  k  l
    # 5 6 xdev  ydev
    # 7 energy
    # 8 dspace
    # 9 10 intens   integr

    try:
        for line in f:
            i = i + 1
            if line.startswith("Grain no"):
                gnumloc = np.array((line.split())[2], dtype=int)
                print("gnumloc", gnumloc)
                if gnumloc == grain_index:
                    print("grain ", grain_index)
                    # print  " indexed peaks list starts at line : ", i
                    linestart = i + 1
                    grainfound = 1
            if grainfound == 1:
                # if i == linestart :
                # print line.rstrip("\n")
                if line.startswith("latticeparameters"):
                    # print "lattice parameters at line : ", i
                    dlatstr = np.array(line[18:].split(), dtype=float)
                    print("lattice parameters : \n", dlatstr)
                    # print "indexed peaks list ends at line = ", i
                    lineend = i - 1
                if line.startswith("dd,"):
                    # print "calib starts at line : ", i
                    calib[:3] = np.array(line[17:].split(), dtype=float)
                if line.startswith("xbet,"):
                    # print "calib line 2 at line = ", i
                    calib[3:] = np.array(line[11:].split(), dtype=float)
                if line.startswith("dev1,"):
                    dev_str = np.array(line.split()[3:], dtype=float)
                    print("deviations : \n", dev_str)
                if line.startswith("coordinates of a*"):
                    # print "matrix starts at line : ", i
                    linemat = i
                    grainfound = 0
    finally:
        linetot = i
        f.close()

    #    matstr = scipy.io.array_import.read_array(filestr, columns=(0, 1, 2),
    #                                              lines = (linemat, linemat + 1, linemat + 2))
    #    print "linemat", linemat
    matstr = np.genfromtxt(filename,
                            usecols=(0, 1, 2),
                            skip_header=linemat,
                            skip_footer=linetot - (linemat + 3))

    print("matstr", matstr)

    # print "linestart = ", linestart
    # print "lineend =", lineend
    # TODO: upgrade scipy.io.array_import to np.loadtxt
    #    data_str = scipy.io.array_import.read_array(filestr, columns=(0, (1, 11)), \
    #                                              lines = (linestart, (linestart + 1, lineend)))
    #    print "linestart", linestart
    #    print "linetot - lineend", linetot - lineend
    data_ = np.genfromtxt(filename,
                            dtype=None,
                            delimiter="\n",
                            names=True,
                            #                                 usecols=tuple(range(5)),
                            skip_header=linestart,
                            skip_footer=linetot - lineend)

    data_str = np.array([elem[0].split() for elem in data_], dtype=np.float)[:, :11]

    print("number of indexed peaks :", len(data_str))
    print("first peak : ", data_str[0, 2:5])
    print("last peak : ", data_str[-1, 2:5])

    # print "return(data_str, satocrs, calib, dev_str)"
    # print "data_str :  xy(exp) 0:2 hkl 2:5 xydev 5:7 energy 7 dspacing 8  intens 9 integr 10"

    return data_str, matstr, calib, dev_str


def read_indexationfile(filename, grainindex_mat=0):
    r"""
    Read indexation file created by lauetools with extension .res

    Return arrays of all colums [spot# grainindex 2theta chi pixX pixY intensity h k l energy]

    .. todo:: adapt matrix reader to new file .res format

    """
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line.startswith('# Number of spots:'):
                nbspots = int(line.split(':')[1])
                break

    nb_lines_header = 4

    f = open(filename, 'r')
    for _ in range(nb_lines_header):
        f.readline()

    dat = np.fromfile(f, dtype=float, sep=" ")
    nbelems = len(dat)
    f.close()

    data_millerExy = dat.reshape((nbspots, nbelems//nbspots))

    spotindex = np.array(data_millerExy[:, 0])
    twicetheta = np.array(data_millerExy[:, 2])  # target values
    chi = np.array(data_millerExy[:, 3])  # target values
    pixX = np.array(data_millerExy[:, 4])
    pixY = np.array(data_millerExy[:, 5])
    Intensity = np.array(data_millerExy[:, 6])
    miller = np.array(data_millerExy[:, 7:10])
    energy = np.array(data_millerExy[:, 10])

    Grain_index = np.array(data_millerExy[:, 1])

    # removing spots that were in ambiguous positions to be indexed (energy set to 0)
    pos = np.where(energy != 10000.)[0]

    spotindex = np.take(spotindex, pos, axis=0)
    twicetheta = np.take(twicetheta, pos, axis=0)
    chi = np.take(chi, pos, axis=0)
    pixX = np.take(pixX, pos, axis=0)
    pixY = np.take(pixY, pos, axis=0)
    Intensity = np.take(Intensity, pos, axis=0)
    miller = np.take(miller, pos, axis=0)
    energy = np.take(energy, pos, axis=0)
    Grain_index = np.take(Grain_index, pos, axis=0)

    f = open(filename, "r")
    accum_mat = 0  # U matrix
    lineaccum = 0
    mat_index = 0
    datamat = {}

    accum_B = 0  # B matrix
    lineaccum_B = 0
    mat_index_B = 0
    datamat_B = {}

    accum_E = 0  # E Matrix (deviatoric strain)
    lineaccum_E = 0
    mat_index_E = 0
    datamat_E = {}

    accum_param = 0
    lineaccum_param = 0
    detector_param = {}

    for line in f.readlines():
        if accum_mat == 1 and lineaccum < 3:  # collect orientation matrix elements
            tru = line.split()[1:]
            datamat[grainindex_mat].append(tru)
            lineaccum += 1
        if accum_B == 1 and lineaccum_B < 3:  # collect Bmatrix elements
            tru = line.split()[1:]
            datamat_B[grainindex_mat].append(tru)
            lineaccum_B += 1
        if accum_E == 1 and lineaccum_E < 3:  # collect Ematrix elements
            tru = line.split()[1:]
            datamat_E[grainindex_mat].append(tru)
            lineaccum_E += 1
        if (accum_param == 1 and lineaccum_param < 5):  # collect calibration parameters elements
            tru = line.split()
            variable = tru[1]
            value = float(tru[3])
            detector_param[variable] = value
            lineaccum_param += 1

        if accum_mat == 1 and lineaccum == 3:
            accum_mat = 0
            mat_index += 1
        if accum_B == 1 and lineaccum_B == 3:
            accum_B = 0
            mat_index_B += 1
        if accum_E == 1 and lineaccum_E == 3:
            accum_E = 0
            mat_index_E += 1

        if line[:4] == "#Ori":
            datatype = "orientation"
        if line[:4] == "#BMa":
            datatype = "Bmatrix"
        if line[:4] == "#EMa":
            datatype = "Ematrix"
        if line[:2] == "#G":  # Grain matrix orientation start
            tra = line.split(" ")[1]
            # print "tra",tra
            grainindex_mat = tra.split(":")[0]
            # print "datatype",datatype
            if datatype == "orientation":
                accum_mat = 1
                lineaccum = 0
                datamat[grainindex_mat] = []
            if datatype == "Bmatrix":
                accum_B = 1
                lineaccum_B = 0
                datamat_B[grainindex_mat] = []
            if datatype == "Ematrix":
                accum_E = 1
                lineaccum_E = 0
                datamat_E[grainindex_mat] = []
        if line[:4] == "#Cal":  # calibration parameter start
            accum_param = 1
            lineaccum_param = 0

    f.close()

    for key, _ in datamat.items():
        datamat[key] = np.array(datamat[key], dtype=float)

    for key, _ in datamat_B.items():
        datamat_B[key] = np.array(datamat_B[key], dtype=float)

    for key, _ in datamat_E.items():
        datamat_E[key] = np.array(datamat_E[key], dtype=float)

    calib = []
    if detector_param:
        calib = []
        for key in ["dd", "xcen", "ycen", "xbet", "xgam"]:
            calib.append(detector_param[key])

    return (spotindex, twicetheta, chi, pixX, pixY, Intensity, miller, energy,
                Grain_index, datamat, datamat_B, datamat_E, calib)

def getpeaks_fromfit2d(filename):
    """
    read peaks list created by fit2d peak search

    #TODO: to remove function to read old data format, not used any longer
    """
    frou = open(filename, "r")
    alllines = frou.readlines()
    frou.close()
    peaklist = alllines[1:]

    print(" %d peaks in %s" % (len(peaklist), filename))
    outputfilename = filename[:-6] + ".pik"
    fric = open(outputfilename, "w")
    for line in peaklist:
        fric.write(line)
    fric.close()
    print("X,Y, int list in %s" % (outputfilename))
    return len(peaklist)


def start_func():
    print("main of IOLaueTools.py")
    # print("numpy version", np.__version__)

    print("print current", time.asctime())

    for k in list(range(20)):
        print("k=%d, k**2=%d" % (k, k ** 2))


# ----------------------------------
# Lauetools .fit file parser
# rev. : 2016-08-03
# S. Tardif (samuel.tardif@gmail.com)
# --------------------------


class Peak:
    def __init__(self, p):
        if len(p) == 18:
            self.spot_index = float(p[0])
            self.Intensity = float(p[1])
            self.h = int(float(p[2]))
            self.k = int(float(p[3]))
            self.l = int(float(p[4]))
            self.pixDev = float(p[5])
            self.energy = float(p[6])
            self.Xexp = float(p[7])
            self.Yexp = float(p[8])
            self.twotheta_exp = float(p[9])
            self.chi_exp = float(p[10])
            self.Xtheo = float(p[11])
            self.Ytheo = float(p[12])
            self.twotheta_theo = float(p[13])
            self.chi_theo = float(p[14])
            self.Qx = float(p[15])
            self.Qy = float(p[16])
            self.Qz = float(p[17])
        elif len(p) == 12:
            self.spot_index = float(p[0])
            self.Intensity = float(p[1])
            self.h = int(float(p[2]))
            self.k = int(float(p[3]))
            self.l = int(float(p[4]))
            self.twotheta_exp = float(p[5])
            self.chi_exp = float(p[6])
            self.Xexp = float(p[7])
            self.Yexp = float(p[8])
            self.energy = float(p[9])
            self.GrainIndex = float(p[10])
            self.pixDev = float(p[11])


class LT_fitfile:
    """
    Parse the .fit file in a LT_fitfile object
    """

    # dictionary definitions for handling the LaueTools .fit files lines
    def __param__(self):
        return {
            "#UB matrix in q= (UB) B0 G* ": self.__UB__,
            "#B0 matrix in q= UB (B0) G*": self.__B0__,
            "#UBB0 matrix in q= (UB B0) G* i.e. recip. basis vectors are columns in LT frame: astar = UBB0[0,:], bstar = UBB0[1,:], cstar = UBB0[2,:]. (abcstar as lines on xyzlab1, xlab1 = ui, ui = unit vector along incident beam)": self.__UBB0__,
            "#UBB0 matrix in q= (UB B0) G* , abcstar as lines on xyzlab1, xlab1 = ui, ui = unit vector along incident beam : astar = UBB0[0,:], bstar = UBB0[1,:], cstar = UBB0[2,:]": self.__UBB0__,
            "#deviatoric strain in crystal frame (10-3 unit)": self.__devCrystal__,
            "#deviatoric strain in direct crystal frame (10-3 unit)": self.__devCrystal__,
            "#deviatoric strain in sample2 frame (10-3 unit)": self.__devSample__,
            "#DetectorParameters": self.__DetectorParameters__,
            "#pixelsize": self.__PixelSize__,
            "#Frame dimensions": self.__FrameDimension__,
            "#CCDLabel": self.__CCDLabel__,
            "#Element": self.__Element__,
            "#grainIndex": self.__GrainIndex__,
            "#spot_index intensity h k l 2theta Chi Xexp Yexp Energy GrainIndex PixDev": self.__Peaks__,
            "#spot_index Intensity h k l pixDev energy(keV) Xexp Yexp 2theta_exp chi_exp Xtheo Ytheo 2theta_theo chi_theo Qx Qy Qz": self.__Peaks__,
            "# Number of indexed spots": self.__NumberIndexedSpots__,
            "# Mean Deviation(pixel)": self.__MeanDev__,
        }

    def __UB__(self, f, l):
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").split()
        ub11, ub12, ub13 = float(l[0]), float(l[1]), float(l[2])
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").split()
        ub21, ub22, ub23 = float(l[0]), float(l[1]), float(l[2])
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").split()
        ub31, ub32, ub33 = float(l[0]), float(l[1]), float(l[2])
        self.UB = np.array([[ub11, ub12, ub13], [ub21, ub22, ub23], [ub31, ub32, ub33]])

    def __B0__(self, f, l):
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").split()
        b011, b012, b013 = float(l[0]), float(l[1]), float(l[2])
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").split()
        b021, b022, b023 = float(l[0]), float(l[1]), float(l[2])
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").split()
        b031, b032, b033 = float(l[0]), float(l[1]), float(l[2])
        self.B0 = np.array([[b011, b012, b013], [b021, b022, b023], [b031, b032, b033]])

    def __UBB0__(self, f, l):
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").split()
        ubb011, ubb012, ubb013 = float(l[0]), float(l[1]), float(l[2])
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").split()
        ubb021, ubb022, ubb023 = float(l[0]), float(l[1]), float(l[2])
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").split()
        ubb031, ubb032, ubb033 = float(l[0]), float(l[1]), float(l[2])
        self.UBB0 = np.array([[ubb011, ubb012, ubb013],
                            [ubb021, ubb022, ubb023],
                            [ubb031, ubb032, ubb033]])

    def __devCrystal__(self, f, l):
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").split()
        ep11, ep12, ep13 = float(l[0]) * 1e-3, float(l[1]) * 1e-3, float(l[2]) * 1e-3
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").split()
        ep22, ep23 = float(l[1]) * 1e-3, float(l[2]) * 1e-3
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").split()
        ep33 = float(l[2]) * 1e-3
        self.deviatoric = np.array([[ep11, ep12, ep13], [ep12, ep22, ep23], [ep13, ep23, ep33]])

    def __devSample__(self, f, l):
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").split()
        ep_sample11, ep_sample12, ep_sample13 = (float(l[0]) * 1e-3,
                                                float(l[1]) * 1e-3,
                                                float(l[2]) * 1e-3)
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").split()
        ep_sample22, ep_sample23 = float(l[1]) * 1e-3, float(l[2]) * 1e-3
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").split()
        ep_sample33 = float(l[2]) * 1e-3
        self.dev_sample = np.array([[ep_sample11, ep_sample12, ep_sample13],
                                    [ep_sample12, ep_sample22, ep_sample23],
                                    [ep_sample13, ep_sample23, ep_sample33]])

    def __DetectorParameters__(self, f, l):
        l = (f.readline()
            .replace("[", "")
            .replace("]", "")
            .replace("\n", "")
            .replace(" ", "")
            .split(","))
        self.dd = float(l[0])
        self.xcen = float(l[1])
        self.ycen = float(l[2])
        self.xbet = float(l[3])
        self.xgam = float(l[4])
        self.DetectorParameters = [self.dd, self.xcen, self.ycen, self.xbet, self.xgam]

    def __PixelSize__(self, f, l):
        l = (f.readline()
            .replace("[", "")
            .replace("]", "")
            .replace("\n", "")
            .replace(" ", "")
            .split(","))
        self.PixelSize = float(l[0])

    def __FrameDimension__(self, f, l):
        l = f.readline().replace("\n", "")
        if l[0] == "[":
            l = l.replace("[", "").replace("]", "").split("  ")
        elif l[0] == "(":
            l = l.replace("(", "").replace(")", "").split(", ")
        self.FrameDimension = [float(l[0]), float(l[1])]

    def __CCDLabel__(self, f, l):
        l = (f.readline()
            .replace("[", "")
            .replace("]", "")
            .replace("\n", "")
            .replace(" ", "")
            .split(","))
        self.CCDLabel = l[0]

    def __Element__(self, f, l):
        l = (f.readline()
            .replace("[", "")
            .replace("]", "")
            .replace("\n", "")
            .replace(" ", "")
            .split(","))
        self.Element = l[0]

    def __GrainIndex__(self, f, l):
        l = (f.readline()
            .replace("[", "")
            .replace("]", "")
            .replace("\n", "")
            .replace(" ", "")
            .split(","))
        self.GrainIndex = l[0]

    def __Peaks__(self, f, l):
        self.peak = {}
        for _ in list(range(self.NumberOfIndexedSpots)):
            l = f.readline().split()
            self.peak["{:d} {:d} {:d}".format(int(float(l[2])), int(float(l[3])), int(float(l[4])))] = Peak(l)

    def __NumberIndexedSpots__(self, _, l):
        self.NumberOfIndexedSpots = int(l.split(" ")[-1])

    def __MeanDev__(self, _, l):
        self.MeanDevPixel = float(l.split(" ")[-1])

    def __init__(self, filename, verbose=False):
        try:
            with open(filename, "rU") as f:
                self.filename = filename

                # read the header
                l = f.readline()
                self.corfile = l.split(" ")[-1]

                l = f.readline()
                self.timestamp, self.software = l.lstrip("# File created at ").split(" with ")

                # read the footer
                l = f.readline().replace("\n", "")
                while l != "\n" and l != "":
                    try:
                        self.__param__()[l](f, l)
                        if verbose:
                            print("read ", l)
                        l = f.readline().replace("\n", "")

                    except KeyError:
                        try:
                            # print l.split(':')[0]
                            self.__param__()[l.split(":")[0]](f, l)
                            l = f.readline().replace("\n", "")

                        except KeyError:
                            print("could not read line {}".format(l))
                            l = f.readline().replace("\n", "")

                # some extra calculations to get the direct and reciprocal lattice basis vector
                # NOTE: the scale of the lattice basis vector is UNKNOWN !!!
                #       they are given here with a arbitrary scale factor
                if not hasattr(self, "UBB0"):
                    self.UBB0 = np.dot(self.UB, self.B0)

                try:
                    self.astar_prime = self.UBB0[:, 0]
                    self.bstar_prime = self.UBB0[:, 1]
                    self.cstar_prime = self.UBB0[:, 2]

                    self.a_prime = np.cross(self.bstar_prime, self.cstar_prime
                    ) / np.dot(self.astar_prime, np.cross(self.bstar_prime, self.cstar_prime))
                    self.b_prime = np.cross(self.cstar_prime, self.astar_prime
                    ) / np.dot(self.bstar_prime, np.cross(self.cstar_prime, self.astar_prime))
                    self.c_prime = np.cross(self.astar_prime, self.bstar_prime
                    ) / np.dot(self.cstar_prime, np.cross(self.astar_prime, self.bstar_prime))

                    self.boa = np.linalg.linalg.norm(self.b_prime
                    ) / np.linalg.linalg.norm(self.a_prime)
                    self.coa = np.linalg.linalg.norm(self.c_prime
                    ) / np.linalg.linalg.norm(self.a_prime)

                    self.alpha = (np.arccos(np.dot(self.b_prime, self.c_prime)
                            / np.linalg.linalg.norm(self.b_prime)
                            / np.linalg.linalg.norm(self.c_prime)) * 180.0 / np.pi)
                    self.beta = (np.arccos(np.dot(self.c_prime, self.a_prime)
                            / np.linalg.linalg.norm(self.c_prime)
                            / np.linalg.linalg.norm(self.a_prime)) * 180.0 / np.pi)
                    self.gamma = (np.arccos(np.dot(self.a_prime, self.b_prime)
                            / np.linalg.linalg.norm(self.a_prime)
                            / np.linalg.linalg.norm(self.b_prime)) * 180.0 / np.pi)

                except ValueError:
                    print("could not compute the reciprocal space from the UBB0")

        except IOError:
            print("file {} not found! or problem of reading it!".format(filename))


# if __name__ == "__main__":

#     filepath = "checkubs.ubs"
#     filepath = "SiHgCdTe.ubs"
#     filepath = "/home/micha/LaueToolsPy3/LaueTools/Examples/CuSi/testUBS.ubs"
#     res = readCheckOrientationsFile(filepath)


# #     start_func()
# #
# #     pp = readwriteParametersFile()
# #     pp.loadParamsFile('myparams.txt')
