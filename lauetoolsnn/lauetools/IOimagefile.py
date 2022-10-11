# -*- coding: utf-8 -*-
r"""
IOimagefile module is made for reading data contained in binary image file
fully or partially.

More tools can be found in LaueTools package at sourceforge.net and gitlab.esrf.fr
March 2020
"""
__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"

# built-in modules
# import sys
import os
import copy
import struct

# third party modules
try:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=FutureWarning)
        import fabio
    FABIO_EXISTS = True
except ImportError:
    FABIO_EXISTS = False

try:
    from libtiff import TIFF, libtiff_ctypes
    libtiff_ctypes.suppress_warnings()
    LIBTIFF_EXISTS = True
except (ImportError, ValueError):
    LIBTIFF_EXISTS = False

try:
    from PIL import Image
    PIL_EXISTS = True
except ImportError:
    PIL_EXISTS = False
    
try:
    import matplotlib.pyplot as plt
    MLAB_EXISTS = True
except ImportError:
    MLAB_EXISTS = False
    
import numpy as np

try:
    from lauetools import dict_LaueTools as DictLT
    from lauetools import imageprocessing as ImProc
except:
    import lauetoolsnn.lauetools.dict_LaueTools as DictLT
    import lauetoolsnn.lauetools.imageprocessing as ImProc
     
listfile = os.listdir(os.curdir)

def stringint(k, n):
    r""" returns string of integer k with n zeros padding
    (by placing zeros before to have n characters)

    :param k: integer to convert
    :param n: nb of digits for zero padding

    :return: string of length n containing integer k

    Example: 1 -> '0001'
    15 -> '0015'
    """
    #     strint = str(k)
    #     res = '0' * (n - len(strint)) + strint

    encodingdigits = "{" + ":0{}".format(int(n)) + "}"
    res = encodingdigits % k

    return res

# --- ---------------- read images functions
def setfilename(imagefilename, imageindex, nbdigits=4, CCDLabel=None, verbose=0):
    r"""
    reconstruct filename string from imagefilename and update filename index with imageindex

    :param imagefilename: filename string (full path or not)
    :param imageindex: index in filename
    :type imageindex: string

    :return filename: input filename with index replaced by input imageindex
    :rtype: string
    """

    #     print "imagefilename",imagefilename
    if imagefilename.endswith("mccd"):
        lenext = 5 #length of extension including '.'
        imagefilename = imagefilename[: -(lenext + nbdigits)] + "{:04d}.mccd".format(imageindex)

    elif CCDLabel in ("sCMOS", "sCMOS_fliplr"):
        #default file extension for sCMOS camera
        ext = "tif"
        lenext = 4  #length of extension including '.'

        if imagefilename.endswith("tiff"):
            ext = "tiff"
            lenext = 5
        # zero padded index for filename
        if nbdigits is not None:
            if imagefilename.endswith(ext):
                #imagefilename = imagefilename[: - (lenext + nbdigits)] + "{:04d}.{}".format(imageindex, ext)

                #imagefilename = imagefilename[: - (lenext + nbdigits)].zfill(nbdigits) + "%s" % ext

                prepart = str(imagefilename).rsplit('.',1)[0]
                prefix, digitpart = str(prepart).rsplit('_', 1)

                current_nbdigits = len(digitpart)
                imagefilename = prefix + "_" + str(imageindex).zfill(current_nbdigits) + "." + ext
                
            elif imagefilename.endswith(ext+".gz"):
                imagefilename = imagefilename[: -(lenext+3 + nbdigits)] + "{:04d}.{}.gz".format(imageindex, ext)
        # no zero padded index for filename
        else:
            if imagefilename.endswith(ext):
                prefix, _ = imagefilename.split(".")
                prefix0 = prefix.rsplit("_")[0]
                if imageindex > 9999:
                    imagefilename = prefix0 + "_{}.{}".format(imageindex, ext)
                else:
                    imagefilename = prefix0 + "_{:04d}.{}".format(imageindex, ext)

    elif CCDLabel in ("EIGER_4Mstack",):
        # only stackimageindex is changed not imagefilename
        pass

    #     print "imagefilename archetype", imagefilename
    elif imagefilename.endswith("mar.tiff"):

        imagefilename = imagefilename[: -(9 + nbdigits)] + "{:04d}_mar.tiff".format(imageindex)

    elif imagefilename.endswith("mar.tif"):
        # requires two underscores with number in between
        # map1_105_mar.tif
        sname = imagefilename.split("_")
        if len(sname) >= 4:  # pathfilename contain more than 1 underscore

            sname2 = sname[-2] + "_" + sname[-1]
            prefix = imagefilename.rstrip(sname2)
        else:
            prefix = imagefilename.rsplit("_")[0]
        imagefilename = prefix + "_{}_mar.tif".format(imageindex)

    # special case for image id15 frelon corrected form distorsion
    elif imagefilename.endswith((".tif", ".edf")):
        if CCDLabel in ("ImageStar_raw", ):
            prefixwihtindex = imagefilename[:-4]
            prefixwithindex_list = list(prefixwihtindex)
            indexnodigit = 0
            while len(prefixwithindex_list) > 0:
                lastelem = prefixwithindex_list.pop(-1)
                if verbose>0: print("lastelem", lastelem)
                if not lastelem.isdigit():
                    break
                indexnodigit += 1
            prefix = prefixwihtindex[:-(indexnodigit)]
            if verbose > 0: print("prefix", prefix)
            imagefilename = prefix + "{}.tif".format(imageindex)

        else:
            suffix = imagefilename[-4:]
            prefix = imagefilename[: -(4 + nbdigits)]

            imagefilename = prefix + "{:04d}{}".format(imageindex, suffix)

    elif imagefilename.endswith("mccd"):

        imagefilename = imagefilename[: -(5 + nbdigits)] + "{:04d}.mccd".format(imageindex)

    elif imagefilename.endswith("edf"):

        imagefilename = imagefilename[: -(4 + nbdigits)] + "{:04d}.edf".format(imageindex)

    elif imagefilename.endswith("unstacked"):

        imagefilename = imagefilename[: -(10 + nbdigits)] + "{:04d}.unstacked".format(imageindex)

    #     print "set filename to:", imagefilename
    return imagefilename


def getIndex_fromfilename(imagefilename, nbdigits=4, CCDLabel=None, stackimageindex=-1, verbose=0):
    r"""
    get integer index from imagefilename string

    :param imagefilename: filename string (full path or not)

    :return: file index
    """
    if CCDLabel in ("sCMOS", "sCMOS_fliplr"):
        #default file extension for sCMOS camera
        ext = "tif"
        lenext = 4 #length of extension including '.'

        if imagefilename.endswith("tiff"):
            ext = "tiff"
            lenext = 5
        if verbose > 0: print('imagefilename', imagefilename)
        if nbdigits is not None:
            if imagefilename.endswith(ext):
                #imageindex = int(imagefilename[-(lenext + nbdigits): - (lenext)])
                prepart = str(imagefilename).rsplit('.',1)[0]
                digitpart = str(prepart).rsplit('_',1)[-1]
                print("prepart", prepart)
                print("digitpart",digitpart)
                imageindex = int(digitpart)
            elif imagefilename.endswith(ext+".gz"):
                imageindex = int(imagefilename[-(lenext+3 + nbdigits) : -(lenext+3)])
        else:
            if imagefilename.endswith(ext):
                prefix, _ = imagefilename.rsplit(".")
                imageindex = int(prefix.rsplit("_")[1])

    # for stacked images we return the position of image data in the stack as imagefileindex
    elif CCDLabel in ("EIGER_4Mstack",):
        imageindex = stackimageindex

    elif imagefilename.endswith("mar.tiff"):
        imageindex = int(imagefilename[-(9 + nbdigits) : -9])
    elif imagefilename.endswith("tiff"):
        imageindex = int(imagefilename[-(5 + nbdigits) : -5])
    elif imagefilename.endswith("mar.tif"):
        imageindex = int(imagefilename.split("_")[-2])
    elif imagefilename.endswith(".tif"):
        # TODO: treat the case of HN56.tif  without underscore (see setfilename)
        imageindex = int(imagefilename.split("_")[-1][:-4])
    elif imagefilename.endswith("mccd"):
        imageindex = int(imagefilename[-(5 + nbdigits) : -5])
    elif imagefilename.endswith("edf"):
        imageindex = int(imagefilename[-(4 + nbdigits) : -4])
    elif imagefilename.endswith("unstacked"):
        imageindex = int(imagefilename[-(10 + nbdigits) : -10])

    return imageindex


def getfilename(dirname, imfileprefix, imfilesuffix=None, numim=None,
                nbdigits_filename=4):
    """
    to get the global file name (name+path) for given components of the name
    put %4d instead of stringint
    """

    fileim = imfileprefix + stringint(numim, nbdigits_filename) + imfilesuffix
    filename = os.path.join(dirname, fileim)

    return filename

def getwildcardstring(CCDlabel):
    r"""  return smart wildcard to open binary CCD image file with priority of CCD type of CCDlabel

    :param CCDlabel: string label defining the CCD type
    :type CCDlabel: str
    :return: string from concatenated strings to be used in wxpython open file dialog box
    :rtype: str

    .. see also::
        - :func:`getIndex_fromfilename`

        - LaueToolsGUI.AskUserfilename

        - wx.FileDialog

    .. example::
        >>> from readmccd import getwildcardstring
        >>> getwildcardstring('MARCCD165')
        'MARCCD, ROPER(*.mccd)|*mccd|mar tif(*.tif)|*_mar.tiff|tiff(*.tiff)|*tiff|Princeton(*.spe)|*spe|Frelon(*.edf)|*edf|tif(*.tif)|*tif|All files(*)|*'
    """
    ALL_EXTENSIONS = ["mccd", "_mar.tiff", "tiff", "spe", "edf", "tif", "h5", ""]
    INFO_EXTENSIONS = ["MARCCD, ROPER(*.mccd)", "mar tif(*.tif)", "tiff(*.tiff)",
                    "Princeton(*.spe)", "Frelon(*.edf)", "tif(*.tif)", "hdf5(*.h5)",
                    "All files(*)"]

    extensions = copy.copy(ALL_EXTENSIONS)
    infos = copy.copy(INFO_EXTENSIONS)
    chosen_extension = DictLT.dict_CCD[CCDlabel][7]

    if chosen_extension in ALL_EXTENSIONS:
        index = ALL_EXTENSIONS.index(chosen_extension)
        ce = extensions.pop(index)
        extensions.insert(0, ce)

        inf = infos.pop(index)
        infos.insert(0, inf)

    wcd = ""
    for inf, ext in zip(infos, extensions):
        wcd += "{}|*{}|".format(inf, ext)

    wildcard_extensions = wcd[:-1]

    return wildcard_extensions


def getpixelValue(filename, x, y, ccdtypegeometry="edf"):
    r"""return pixel value at x,y

    .. warning::
        Very old function. To be checked. Use better readpixelvalue in plotdip.py

    :param filename: path to image file
    :type filename: str
    :param x: x pixel value
    :type x: int
    :param y: y pixel value
    :type y: int
    :param ccdtypegeometry: CCD label, defaults to "edf"
    :type ccdtypegeometry: str, optional
    :return: pixel intensity
    :rtype: int
    """
    if ccdtypegeometry == "edf":
        # frelon camera as mounted on BM32 Oct2012
        y = 2047 - y
    if ccdtypegeometry in ("mccd",):
        pass

    f = open(filename, "rb")
    f.seek(1024 + 2 * (2048 * y + x))
    val = struct.unpack("H", f.read(2))
    f.close()
    return val[0]


def readheader(filename, offset=4096, CCDLabel="MARCCD165"):
    r"""
    return header in a raw format

    default offset for marccd image
    """
    if CCDLabel.startswith("sCMOS"):
        filesize = os.path.getsize(filename)
        framedim = DictLT.dict_CCD[CCDLabel][0]
        offset = filesize - np.prod(framedim) * 2
    elif CCDLabel.startswith("MARCCD165"):
        offset = 4096
    with open(filename, "rb") as f:
        myheader = f.read(offset)
        #myheader.replace("\x00", " ")

    return myheader


def read_header_marccd(filename):
    r"""
    return string of parameters found in header in marccd image file .mccd

    - print allsentences  displays the header
    - use allsentences.split('\n') to get a list
    """
    f = open(filename, "rb")
    f.seek(2048)
    posbyte = 0
    allsentences = ""
    for _ in list(range(32)):
        tt = f.read(32)
        s1 = tt.strip("\x00")
        if s1 != "":
            allsentences += s1 + "\n"
        # print posbyte, s1
        posbyte += 32
    tt = f.read(1024)
    s1 = tt.strip("\x00")
    if s1 != "":
        allsentences += s1 + "\n"

    f.close()
    return allsentences


def read_header_marccd2(filename):
    r"""
    return string of parameters comments and exposure time
    found in header in marccd image file .mccd

    - print allsentences  displays the header
    - use allsentences.split('\n') to get a list
    """
    f = open(filename, "rb")
    f.seek(3072)
    tt = f.read(512)
    # from beamline designed input
    dataset_comments = str(tt).strip("\x00")

    f.seek(1024 + 2 * 256 + 128 + 12)
    s = struct.Struct("I I I")
    unpacked_data = s.unpack(f.read(3 * 4))
    #     print 'Unpacked Values:', unpacked_data
    #integration_time, expo_time, readout_time = unpacked_data
    _, expo_time, _ = unpacked_data

    f.close()
    return dataset_comments, expo_time


def read_header_scmos(filename, verbose=0):
    r"""
    return string of parameters comments and exposure time
    found in header in scmis image file .tif

    - print allsentences  displays the header
    - use allsentences.split('\n') to get a list
    """
    if not PIL_EXISTS:
        return {}

    img = Image.open(filename)


    # img.tag.keys()
    # tag[270]   = (u'0 (thf=-50.4850 mon=24923 exposure=0.400)',)

    dictpar = {}
    strcom = img.tag[270][0]
    if verbose>0: print('read_header_scmos', strcom, type(strcom))

    si = strcom.index("(")
    fi = strcom.index(")")
    listpar = strcom[si + 1 : fi].split()
    #     print "listpar",listpar
    for elem in listpar:
        if "=" in elem:
            key, val = elem.split("=")
            dictpar[key] = float(val)

    return dictpar


def read_motorsposition_fromheader(filename, CCDLabel="MARCCD165"):
    r""" return xyzpositions, expo_time from image file header
    available for "MARCCD165", "sCMOS", "sCMOS_fliplr"
    """
    if CCDLabel in ("MARCCD165",):
        dataset_comments, expo_time = read_header_marccd2(filename)
        xyz = dataset_comments.split(" ")[:3]
        xyzpositions = []
        for pos in xyz:
            xyzpositions.append(float(pos))
    elif CCDLabel in ("sCMOS", "sCMOS_fliplr"):
        dictpar = {}
        import re

        img = Image.open(filename)
        ff = img.tag[315]
        aa = re.findall(r"([^:]+)=([^:]+)(?:,|$)", ff[0])
        listeq = aa[0][0].split(" ")
        for elem in listeq:
            if "Xech" in elem or "Yech" in elem or "Zech" in elem:
                dictpar[elem.lstrip("(")] = float(elem.rstrip(")"))

        headerdict = read_header_scmos(filename)
        if headerdict:
            expo_time = headerdict["exposure"]
        xyzpositions = []
        for motorname in ("Xech", "Yech", "Zech"):
            xyzpositions.append(dictpar[motorname])

    return xyzpositions, expo_time


def readoneimage_full(filename, frametype="mccd", dirname=None):
    r"""
    too SLOW!
    reads 1 entire image (marCCD format)
    :return: PILimage, image object of PIL module (16 bits integer) and
    arrayofdata: 2D array of intensity
    #TODO: manage framedim like readoneimage() just below"""

    if frametype == "mccd":
        shapeCCD = (2048, 2048)
    elif frametype == "martiff":
        # opposite of fit2d displays in loading data
        shapeCCD = (2591, 2751)
    elif frametype == "spe":
        # opposite of fit2d displays in loading data
        shapeCCD = (2048, 2048)

    if dirname is None:
        dirname = os.curdir

    pilimage = Image.open(os.path.join(dirname, filename))
    ravdata = np.array(pilimage.getdata())

    return pilimage, np.reshape(ravdata, shapeCCD)


def readCCDimage(filename, CCDLabel="MARCCD165", dirname=None, stackimageindex=-1, verbose=0):
    r"""Read raw data binary image file.

    Read raw data binary image file and return pixel intensity 2D array such as
    to fit the data (2theta, chi) scattering angles representation convention.

    :param filename: path to image file (fullpath if ` dirname` =None)
    :type filename: str
    :param CCDLabel: label, defaults to "MARCCD165"
    :type CCDLabel: str, optional
    :param dirname: folder path, defaults to None
    :type dirname: str, optional
    :param stackimageindex: index of images bunch, defaults to -1
    :type stackimageindex: int, optional
    :param verbose: 0 or 1, defaults to 0
    :type verbose: int, optional
    :raises ValueError: if data format and CCD parameters from label are not compatible
    :return:
        - dataimage, 2D array image data pixel intensity properly oriented
        - framedim, iterable of 2 integers shape of dataimage
        - fliprot : string, key for CCD frame transform to orient image
    :rtype: tuple of 3 elements
    """
    (framedim, _, _, fliprot, offsetheader, formatdata, _, _) = DictLT.dict_CCD[CCDLabel]

    USE_RAW_METHOD = False

    if verbose > 1:
        print("CCDLabel in readCCDimage", CCDLabel)
        print('FABIO_EXISTS',FABIO_EXISTS)
        print('LIBTIFF_EXISTS',LIBTIFF_EXISTS)
        print('PIL_EXISTS',PIL_EXISTS)
    #    if extension != extension:
    #        print "warning : file extension does not match CCD type set in Set CCD File Parameters"
    if FABIO_EXISTS:
        if CCDLabel in ('MARCCD165', "EDF", "EIGER_4M", "EIGER_1M",
                        "sCMOS", "sCMOS_fliplr", "sCMOS_fliplr_16M", "sCMOS_16M",
                        "Rayonix MX170-HS", 'psl_weiwei', 'ImageStar_dia_2021'):

            if verbose > 1:
                print('----> Using fabio ... to open %s\n'%filename)
            # warning import Image  # for well read of header only

            if dirname is not None:
                img = fabio.open(os.path.join(dirname, filename))
            else:
                img = fabio.open(filename)

            dataimage = img.data
            framedim = dataimage.shape

            # pythonic way to change immutable tuple...
            initframedim = list(DictLT.dict_CCD[CCDLabel][0])
            initframedim[0] = framedim[0]
            initframedim[1] = framedim[1]
            initframedim = tuple(initframedim)
        else:
            USE_RAW_METHOD = True
    
    elif MLAB_EXISTS:
        if CCDLabel in ('MARCCD165', "EDF", "EIGER_4M", "EIGER_1M",
                        "sCMOS", "sCMOS_fliplr", "sCMOS_fliplr_16M", "sCMOS_16M",
                        "Rayonix MX170-HS", 'psl_weiwei', 'ImageStar_dia_2021'):

            if verbose > 1:
                print('----> Using matplotlib ... to open %s\n'%filename)
            # warning import Image  # for well read of header only

            if dirname is not None:
                dataimage = plt.imread(os.path.join(dirname, filename))
            else:
                dataimage = plt.imread(filename)
            framedim = dataimage.shape
            # pythonic way to change immutable tuple...
            initframedim = list(DictLT.dict_CCD[CCDLabel][0])
            initframedim[0] = framedim[0]
            initframedim[1] = framedim[1]
            initframedim = tuple(initframedim)
        else:
            USE_RAW_METHOD = True
    
    elif CCDLabel in ("EIGER_4Mstack", ):

        import tables as Tab

        if dirname is not None:
            pathtofile = os.path.join(dirname, filename)
        else:
            pathtofile = filename

        # TODO check to correct version transition
        if Tab.__version__ >= "3.4.2":
            hdf5file = Tab.open_file(pathtofile)
        else:
            hdf5file = Tab.openFile(pathtofile)

        if verbose > 0: print("opening hdf5 stacked data table")
        alldata = hdf5file.root.entry.data.data
        if verbose > 0: print("alldata.shape", alldata.shape)

        dataimage = alldata[stackimageindex]
        framedim = dataimage.shape

    elif LIBTIFF_EXISTS:
        if verbose > 1:
            print("----> Using libtiff...")
        if CCDLabel in ("sCMOS", "MARCCD165", "TIFF Format", "FRELONID15_corrected", "VHR_PSI",
                            "VHR_DLS", "MARCCD225", "Andrea", "pnCCD_Tuba", "sCMOS_16M"):

            #         print "tiff format", CCDLabel
            #             print "dirname, filename", dirname, filename
            if dirname is not None:
                tifimage = TIFF.open(os.path.join(dirname, filename), mode="r")
            else:
                tifimage = TIFF.open(filename, mode="r")

            dataimage = tifimage.read_image()
            framedim = (int(tifimage.GetField("ImageLength")),
                        int(tifimage.GetField("ImageWidth")))
            if tifimage.IsByteSwapped():
                dataimage = dataimage.byteswap()
        else:
            USE_RAW_METHOD = True

    elif PIL_EXISTS:
        if verbose > 1:
            print("using PIL's module Image")
        if CCDLabel in ("sCMOS", "MARCCD165", "sCMOS_16M"):
            if verbose > 1: print('PIL is too slow. Better install libtiff or fabio. Meanwhile ...')
            USE_RAW_METHOD = True
        elif CCDLabel in ("VHR_PSI", "VHR_DLS", "MARCCD225", "Andrea", "pnCCD_Tuba"):
            # data are compressed!

            if dirname is not None:
                fullpath = os.path.join(dirname, filename)
            else:
                fullpath = filename

            im = Image.open(fullpath, "r")
            dataimage = np.array(im.getdata()).reshape(framedim)

    # RAW method knowing or deducing offsetheader and dataformat
    if USE_RAW_METHOD:
        print("----> !!! not using libtiff, nor fabio, nor PIL!!!  ")
        if CCDLabel in ("MARCCD165",):
            print("for MARCCD not using libtiff, raw method ...")
            # offsetheader may change ...
            #filesize=getfilesize(dirname, filename)
            offsetheader = 4096 #filesize - 2048*2048 * 2
        # offset header varying
        elif CCDLabel.startswith("ImageStar_raw"):
            filesize=getfilesize(dirname, filename)
            bytes_per_pixels = 2
            if CCDLabel.endswith("32bits"):
                bytes_per_pixels = 4

            nbpixels = 1500
            if CCDLabel in ("ImageStar_1528x1528",):
                nbpixels = 1528
            offsetheader = filesize - nbpixels * nbpixels * bytes_per_pixels

        elif CCDLabel == "ImageStar_dia_2021":
            filesize=getfilesize(dirname, filename)
            bytes_per_pixels = 2
            nbpixels = 3056

            offsetheader = filesize - nbpixels * nbpixels * bytes_per_pixels
        elif CCDLabel in ("sCMOS",):
            if verbose > 0: print("for sCMOS not using libtiff, raw method ...")
            # offsetheader may change ...
            filesize = getfilesize(dirname, filename)
            offsetheader = filesize - 2016*2018 * 2

        dataimage = readoneimage(filename,
                                framedim=framedim,
                                dirname=dirname,
                                offset=offsetheader,
                                formatdata=formatdata)

        print("dataimage", dataimage)

        if CCDLabel in ("FRELONID15_corrected",):

            dataimage = dataimage.byteswap()

        if CCDLabel in ("EIGER_4Munstacked",):
            dataimage = np.ma.masked_where(dataimage > 4000000, dataimage)
            if verbose > 0:
                print("framedim", framedim)
                print("offsetheader", offsetheader)
                print("formatdata", formatdata)
                print("dataimage", dataimage)

    if verbose:
        print("CCDLabel: ", CCDLabel)
        print("nb of pixels", np.shape(dataimage))

    # need to reshape data from 1D to 2D
    try:
        if len(dataimage.shape) == 1:
            if verbose:
                print("nb elements", len(dataimage))
                print("framedim", framedim)
                print("framedim nb of elements", framedim[0] * framedim[1])
            dataimage = np.reshape(dataimage, framedim)
    except ValueError:
        raise ValueError(
            "Selected CCD type :{} may be wrong (or nb of pixels, dimensions...)".format(
                CCDLabel))

    # some array transformations if needed depending on the CCD mounting
    if fliprot == "spe":
        dataimage = np.rot90(dataimage, k=1)

    elif fliprot == "VHR_Feb13":
        #            self.dataimage_ROI = np.rot90(self.dataimage_ROI, k=3)
        dataimage = np.fliplr(dataimage)

    elif fliprot == "sCMOS_fliplr":
        dataimage = np.fliplr(dataimage)

    elif fliprot == "vhr":  # july 2012 close to diamond monochromator crystal
        dataimage = np.rot90(dataimage, k=3)
        dataimage = np.fliplr(dataimage)

    elif fliprot == "vhrdiamond":  # july 2012 close to diamond monochromator crystal
        dataimage = np.rot90(dataimage, k=3)
        dataimage = np.fliplr(dataimage)

    elif fliprot == "Imagestar_dia_2021":  # starting Feb july 2021 close to diamond monochromator crystal
        dataimage = np.rot90(dataimage, k=1)
        dataimage = np.fliplr(dataimage)
    elif fliprot == "frelon2":
        dataimage = np.flipud(dataimage)
    return dataimage, framedim, fliprot

def readCCDimagev1(filename):
    img = fabio.open(filename)
    return img.data

def getfilesize(dirname,filename):
    if dirname is not None:
        filesize = os.path.getsize(os.path.join(dirname, filename))
    else:
        filesize = os.path.getsize(filename)
    return filesize

def readoneimage(filename, framedim=(2048, 2048), dirname=None, offset=4096, formatdata="uint16"):
    r""" returns a 1d array of integers from a binary image file (full data)

    :param filename: image file name (full path if dirname=0)
    :type filename: str
    :param framedim: detector dimensions, defaults to (2048, 2048)
    :type framedim: tuple of 2 integers, optional
    :param dirname: folder path, defaults to None
    :type dirname: str, optional
    :param offset: file header in byte (octet), defaults to 4096
    :type offset: int, optional
    :param formatdata: numpy format of raw binary image pixel value, defaults to "uint16"
    :type formatdata: str, optional
    :return: dataimage : image data pixel intensity
    :rtype: 1D array
    """

    nb_elem = framedim[0] * framedim[1]

    if dirname is None:
        dirname = os.curdir

    f = open(os.path.join(dirname, filename), "rb")
    f.seek(offset)

    # d=scipy.io.fread(f,2048*2048,np.oldnumeric.Int16)
    # d = scipy.io.fread(f,nb_elem,np.oldnumeric.UInt16)
    d = np.fromfile(f, dtype=formatdata, count=nb_elem)
    f.close()
    return d


def readoneimage_band(filename,
                        framedim=(2048, 2048),
                        dirname=None,
                        offset=4096,
                        line_startindex=0,
                        line_finalindex=2047,
                        formatdata="uint16"):
    r"""
    returns a 1d array of integers from a binary image file. Data located in band according shape of data (framedim)

    :param filename: string
               path to image file (fullpath if `dirname`=None)
    :param offset: integer
             nb of file header bytes
    :param framedim: iterable of 2 integers
               shape of expected 2D data
    :param formatdata: string
                 key for numpy dtype to decode binary file

    :return: dataimage, 1D array, image data pixel intensity
    """
    if dirname is None:
        dirname = os.curdir

    if formatdata in ("uint16",):
        nbBytesPerElement = 2
    if formatdata in ("uint32",):
        nbBytesPerElement = 4

    nbElems = (line_finalindex - line_startindex + 1) * framedim[1]

    f = open(os.path.join(dirname, filename), "rb")
    f.seek(offset + line_startindex * framedim[1] * nbBytesPerElement)

    # d=scipy.io.fread(f,2048*2048,np.oldnumeric.Int16)
    # d = scipy.io.fread(f,nb_elem,np.oldnumeric.UInt16)
    #     print "line_finalindex-line_startindex",line_finalindex-line_startindex
    #     print "line_startindex*framedim[1]*nbBytesPerElement",line_startindex*framedim[1]*nbBytesPerElement
    #     print "line_startindex*framedim[1]",line_startindex*framedim[1]
    #     print "nbElems",nbElems

    band = np.fromfile(f, dtype=formatdata, count=nbElems)
    f.close()
    return band


def readoneimage_crop_fast(filename, dirname=None, CCDLabel="MARCCD165",
                                                    firstElemIndex=0, lastElemIndex=2047, verbose=0):
    r""" Returns a 2d array of integers from a binary image file. Data are taken only from a rectangle

    with respect to firstElemIndex and lastElemIndex.

    :param filename: string, path to image file (fullpath if ` dirname`=None)
    :param offset: integer, nb of file header bytes
    :param framedim: iterable of 2 integers, shape of expected 2D data
    :param formatdata: string, key for numpy dtype to decode binary file

    :return: dataimage : 1D array image data pixel intensity
    """
    (framedim, _, _, fliprot, offsetheader, formatdata, _, _) = DictLT.dict_CCD[CCDLabel]
    if verbose > 0:
        print("framedim read from DictLT.dict_CCD in readoneimage_crop_fast()", framedim)
        print("formatdata", formatdata)
        print("offsetheader", offsetheader)

    if dirname is None:
        dirname = os.curdir

    # if formatdata in ("uint16",):
    #     nbBytesPerElement = 2
    # if formatdata in ("uint32",):
    #     nbBytesPerElement = 4

    dataimage2D = np.zeros(framedim)

    # colFirstElemIndex = firstElemIndex % framedim[1]
    lineFirstElemIndex = firstElemIndex // framedim[1]
    # colLastElemIndex = lastElemIndex % framedim[1]
    lineLastElemIndex = lastElemIndex // framedim[1]

    band1D = readoneimage_band(filename, framedim=framedim, dirname=dirname, offset=offsetheader,
                                                            line_startindex=lineFirstElemIndex,
                                                            line_finalindex=lineLastElemIndex,
                                                            formatdata=formatdata)

    #     print "band1D.shape",band1D.shape
    #     print "(lineLastElemIndex-lineFirstElemIndex,framedim[1])",(lineLastElemIndex-lineFirstElemIndex,framedim[1])
    #     print "((lineLastElemIndex-lineFirstElemIndex)*framedim[1])",((lineLastElemIndex-lineFirstElemIndex)*framedim[1])

    band2D = band1D.reshape((lineLastElemIndex - lineFirstElemIndex + 1, framedim[1]))

    dataimage2D[lineFirstElemIndex : lineLastElemIndex + 1, :] = band2D

    return dataimage2D, framedim, fliprot


def readrectangle_in_image(filename, pixx, pixy, halfboxx, halfboxy, dirname=None,
                                                                            CCDLabel="MARCCD165",
                                                                            verbose=0):
    r"""
    returns a 2d array of integers from a binary image file. Data are taken only from a rectangle
    centered on pixx, pixy

    :return: dataimage : 2D array, image data pixel intensity
    """
    (framedim, _, _, fliprot, offsetheader, formatdata, _, _) = DictLT.dict_CCD[CCDLabel]

    if verbose > 0:
        print("framedim read from DictLT.dict_CCD in readrectangle_in_image()", framedim)
        print("formatdata", formatdata)
        print("offsetheader", offsetheader)
    # recompute headersize
    if dirname is not None:
        fullpathfilename = os.path.join(dirname, filename)
    else:
        dirname = os.curdir
        fullpathfilename = filename

    if formatdata in ("uint16",):
        nbBytesPerElement = 2
    if formatdata in ("uint32",):
        nbBytesPerElement = 4

    if verbose > 0:
        print("fullpathfilename", fullpathfilename)
    try:
        filesize = os.path.getsize(fullpathfilename)
    except OSError:
        print("missing file {}\n".format(fullpathfilename))
        return None

    # uint16
    offsetheader = filesize - (framedim[0] * framedim[1]) * nbBytesPerElement

    if verbose > 0:
        print("calculated offset of header from file size...", offsetheader)

    x = int(pixx)
    y = int(pixy)

    if fliprot in ("sCMOS_fliplr",):
        x = framedim[1] - x

    boxx = int(halfboxx)
    boxy = int(halfboxy)

    xpixmin = x - boxx
    xpixmax = x + boxx

    ypixmin = y - boxy
    ypixmax = y + boxy

    #     ymin= (y-boxy)*framedim[1]
    #     ymax= (y+boxy)*framedim[1]

    lineFirstElemIndex = ypixmin
    lineLastElemIndex = ypixmax

    if verbose > 0:
        print("lineFirstElemIndex", lineFirstElemIndex)
        print("lineLastElemIndex", lineLastElemIndex)

    band = readoneimage_band(fullpathfilename,
                                framedim=framedim,
                                dirname=None,
                                offset=offsetheader,
                                line_startindex=lineFirstElemIndex,
                                line_finalindex=lineLastElemIndex,
                                formatdata=formatdata)

    nblines = lineLastElemIndex - lineFirstElemIndex + 1

    band2D = np.reshape(band, (nblines, framedim[1]))

    rectangle2D = band2D[:, xpixmin : xpixmax + 1]

    if verbose > 0:
        print("band2D.shape", band2D.shape)
        print("rectangle2D.shape", rectangle2D.shape)

    return rectangle2D


def readoneimage_crop(filename, center, halfboxsize, CCDLabel="PRINCETON", dirname=None):
    r"""
    return a cropped array of data read in an image file

    :param filename: string, path to image file (fullpath if ` dirname`=None)
    :param center: iterable of 2 integers, (x,y) pixel coordinates
    :param halfboxsize: integer or iterable of 2 integers, ROI half size in both directions

    :return: dataimage : 1D array, image data pixel intensity

    .. todo:: useless?
    """
    if dirname is None:
        dirname = os.curdir
    else:
        filename = os.path.join(dirname, filename)

    if isinstance(halfboxsize, int):
        boxsizex, boxsizey = halfboxsize, halfboxsize
    elif len(halfboxsize) == 2:
        boxsizex, boxsizey = halfboxsize

    xpic, ypic = center

    dataimage, framedim, _ = readCCDimage(filename, CCDLabel=CCDLabel, dirname=None)

    x1 = np.maximum(0, xpic - boxsizex)
    x2 = np.minimum(framedim[1], xpic + boxsizex + 1)  # framedim[0]
    y1 = np.maximum(0, ypic - boxsizey)
    y2 = np.minimum(framedim[0], ypic + boxsizey + 1)  # framedim[1]

    return None, dataimage[y1:y2, x1:x2]


def readoneimage_manycrops(filename, centers, boxsize, stackimageindex=-1, CCDLabel="MARCCD165",
                                                                        addImax=False,
                                                                        use_data_corrected=None,
                                                                        verbose=0):
    r"""
    reads 1 image and extract many regions
    centered on center_pixel with xyboxsize dimensions in pixel unit

    :param filename: string,fullpath to image file
    :param centers: list or array of [int,int] centers (x,y) pixel coordinates
    :param use_data_corrected: enter data instead of reading data from file
                         must be a tuple of 3 elements:
                         fulldata, framedim, fliprot
                         where fulldata is a numpy.ndarray
                         as output by :func:`readCCDimage`
    :param boxsize: iterable 2 elements or integer
              boxsizes [in x, in y] direction or integer to set a square ROI

    :return: Data, list of 2D array pixel intensity or Data and Imax
    """
    # use alternate data  (for instance for data from filename without background)
    if use_data_corrected is not None:
        if isinstance(use_data_corrected, tuple):
            if len(use_data_corrected) == 3:
                fulldata, framedim, _ = use_data_corrected
    # use data by reading file
    else:
        fulldata, framedim, _ = readCCDimage(filename, stackimageindex=stackimageindex,
                                                            CCDLabel=CCDLabel,
                                                            dirname=None)

    if isinstance(boxsize, int):
        boxsizex, boxsizey = boxsize, boxsize
    elif len(boxsize) == 2:
        boxsizex, boxsizey = boxsize

    # xpic, ypic = np.array(centers).T

    #    x1 = np.array(np.maximum(0, xpic - boxsizex), dtype=np.int)
    #    x2 = np.array(np.minimum(framedim[0], xpic + boxsizex), dtype=np.int)
    #    y1 = np.array(np.maximum(0, ypic - boxsizey), dtype=np.int)
    #    y2 = np.array(np.minimum(framedim[1], ypic + boxsizey), dtype=np.int)

    Data = []

    Imax = []
        
    if verbose > 0:
        print("framedim in readoneimage_manycrops", framedim)
    framedim = framedim[1], framedim[0]

    for center in centers:
        i1, i2, j1, j2 = ImProc.getindices2cropArray(center, (boxsizex, boxsizey), framedim)
        #        print "i1, i2, j1, j2-----", i1, i2, j1, j2
        cropdata = fulldata[i1:i2, j1:j2]
        #         # print "cropdata.shape", cropdata.shape
        #         print "i2-i1",i2-i1
        #         print "boxsizey*2+1",boxsizey*2+1
        #         print "j2-j1",j2-j1
        #         print "boxsizex*2+1",boxsizex*2+1

        # for spot near border, replace by zeros array
        if i2 - i1 != boxsizey * 2 or j2 - j1 != boxsizex * 2:
            cropdata = np.zeros((boxsizey * 2 + 1, boxsizex * 2 + 1))

        Data.append(cropdata)
        if addImax:
            Imax.append(np.amax(cropdata))

    #        print "max in cropped data", np.amax(cropdata)
    if addImax:
        return Data, Imax
    else:
        return Data

def writeimage(outputname, _header, data, dataformat=np.uint16, verbose=0):
    r"""
    from data 1d array of integers
    with header coming from a f.open('imagefile'); f.read(headersize);f.close()
    .. warning:: header contain dimensions for subsequent data. Check before the compatibility of
    data with header infos(nb of byte per pixel and array dimensions
    """
    newfile = open(outputname, "wb")
    newfile.write(_header)
    data = np.array(data, dtype=dataformat)
    data.tofile(newfile)
    newfile.close()
    if verbose > 0: print("image written in ", outputname)


def write_rawbinary(outputname, data, dataformat=np.uint16, verbose=0):
    r"""
    write a binary file without header of a 2D array

    used ?
    """
    newfile = open(outputname, "wb")
    data = np.array(data, dtype=dataformat)
    data.tofile(newfile)

    newfile.close()
    if verbose > 0: print("image written in ", outputname)


def SumImages(prefixname, suffixname, ind_start, ind_end, dirname=None,
                                                            plot=0,
                                                            output_filename=None,
                                                            CCDLabel=None,
                                                            nbdigits=0):
    r"""
    sum images and write image with 32 bits per pixel format (4 bytes)

    used?
    """
    #     prefixname = 'HN08_'
    #     suffixname = '.tif'
    #     CCDLabel = 'ImageStar_raw'
    #     dirname = '/home/micha/LaueProjects/Vita'

    output_filename = "mean_{}_{:04d}_{}{}".format(prefixname, ind_start, ind_end, suffixname)

    filename = "{}{:04d}{}".format(prefixname, ind_start, suffixname)

    data, shape, _ = readCCDimage(filename, CCDLabel=CCDLabel, dirname=dirname)

    if CCDLabel == "ImageStar_raw":
        # Add addition of 32 bits image => replace 2 by 4  nb of bytes per pixel
        filesize = os.path.getsize(os.path.join(dirname, filename))
        offsetheader = filesize - 1500 * 1500 * 2
    if CCDLabel == "EIGER_1M":
        filesize = os.path.getsize(os.path.join(dirname, filename))
        offsetheader = filesize - 1065 * 1030 * 4
    else:
        # Add addition of 32 bits image => replace 2 by 4  nb of bytes per pixel
        filesize = os.path.getsize(os.path.join(dirname, filename))
        offsetheader = filesize - 2048 * 2048 * 2
    # print "shape = ", np.shape(datastart)
    datasum = np.zeros(shape, dtype=np.uint32)

    #     nb_images = ind_end - ind_start + 1

    indexscanlist = list(range(ind_start, ind_end + 1, 1))
    for k in indexscanlist:
        # print k
        filename = "{}{:04d}{}".format(prefixname, k, suffixname)
        # print filename1
        data, shape, _ = readCCDimage(filename, CCDLabel=CCDLabel, dirname=dirname)
        # print max(data1), np.argmax(data1)
        datasum = data + datasum

    if output_filename:
        outputfilename = output_filename
        header = readheader(os.path.join(dirname, filename), offset=offsetheader)
        writeimage(os.path.join(dirname, outputfilename), header, datasum, dataformat=np.uint32)

        print("Added images with prefix {} from {} to {} written in {}".format(
                prefixname, ind_start, ind_end, outputfilename))
    if plot:
        print("later")

    return datasum


def Add_Images2(prefixname,
                ind_start,
                ind_end,
                plot=0,
                writefilename=None,
                CCDLabel="MARCCD165",
                average=True):
    """
    in dev
    """
    suffixname = "." + DictLT.dict_CCD[CCDLabel][-1]

    filename = prefixname + stringint(ind_start, 4) + suffixname
    datastart, _, _ = readCCDimage(filename, CCDLabel)

    # print "shape = ", np.shape(datastart)
    datastart = np.array(datastart, dtype=float)

    indexscanlist = list(range(ind_start + 1, ind_end + 1, 1))
    for k in indexscanlist:
        # print k
        filename1 = prefixname + stringint(k, 4) + suffixname
        # print filename1
        data1, _, _ = readCCDimage(filename1, CCDLabel)
        # print max(data1), np.argmax(data1)
        datastart = np.array(data1, dtype=float) + datastart
    # datastart= datastart/((len(indexscanlist)+1)*1.)
    # print "final"
    # print max(datastart), np.argmax(datastart)
    if average:
        datastart = datastart / (float(len(indexscanlist)) + 1.0)
    # print max(datastart), np.argmax(datastart)
    datastart = np.array(datastart, dtype=np.uint16)
    # print max(datastart), np.argmax(datastart)
    if writefilename:
        outputfilename = writefilename
        header = readheader(filename1, CCDLabel=CCDLabel)
        writeimage(outputfilename, header, datastart)
        print("written in ", outputfilename)
    if plot:
        print("later")

    return datastart


def Add_Images(prefixname, ind_start, ind_end, plot=0, writefilename=None):
    r"""
    Add continuous sequence of images

    .. note::
        Add_Images2 exists

    :param prefixname: string, prefix common part of name of files

    :param ind_start: int, starting image index

    :param ind_end: int, final image index

    :param writefilename: string, new image filename where to write datastart (with last image file header read)

    :return:  datastart, array accumulation of 2D data from each image
    """
    suffixname = ".mccd"

    datastart = readoneimage(prefixname + stringint(ind_start, 4) + suffixname)
    # print "shape = ", np.shape(datastart)
    datastart = np.array(datastart, dtype=float)

    indexscanlist = list(range(ind_start + 1, ind_end + 1, 1))
    for k in indexscanlist:
        # print k
        filename1 = prefixname + stringint(k, 4) + suffixname
        # print filename1
        data1 = readoneimage(filename1)
        # print max(data1), np.argmax(data1)
        datastart = np.array(data1, dtype=float) + datastart
    # datastart= datastart/((len(indexscanlist)+1)*1.)
    # print "final"
    # print max(datastart), np.argmax(datastart)
    datastart = datastart / (float(len(indexscanlist)) + 1.0)
    # print max(datastart), np.argmax(datastart)
    datastart = np.array(datastart, dtype=np.uint16)
    # print max(datastart), np.argmax(datastart)
    if writefilename:
        outputfilename = writefilename
        header = readheader(filename1)
        writeimage(outputfilename, header, datastart)
        print("written in ", outputfilename)
    if plot:
        print("later")

    return datastart

def get_imagesize(framedim, nbbits_per_pixel, headersize_bytes):
    r"""
    return size of image in byte (= 1 octet = 8 bits)
    """
    return (framedim[0] * framedim[1] * nbbits_per_pixel + headersize_bytes * 8) // 8
