# -*- coding: utf-8 -*-
r"""
imageprocessing module is made to modify filter data array

More tools can be found in LaueTools package at sourceforge.net and gitlab.esrf.fr
March 2020
"""
__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"

# built-in modules
import os

# third party modules

# import scipy.interpolate as sci
import scipy.ndimage as ndimage
import scipy.signal
import scipy.spatial.distance as ssd

try:
    from PIL import Image
    PIL_EXISTS = True
except ImportError:
    PIL_EXISTS = False
    
import numpy as np
import pylab as pp

# lauetools modules
try:
    from lauetools import generaltools as GT
    from lauetools import dict_LaueTools as DictLT
except:
    import lauetoolsnn.lauetools.generaltools as GT
    import lauetoolsnn.lauetools.dict_LaueTools as DictLT
    
listfile = os.listdir(os.curdir)

def getindices2cropArray(center, halfboxsizeROI, arrayshape, flipxycenter=False):
    r"""
    return array indices limits to crop array data

    :param center: iterable of 2 elements
             (x,y) pixel center of the ROI
    :param halfboxsizeROI: integer or iterable of 2 elements
                     half boxsize ROI in two dimensions
    :param arrayshape: iterable of 2 integers
                 maximal number of pixels in both directions

    :param flipxycenter: boolean
                   True: swap x and y of center with respect to others
                   parameters that remain fixed
    :return: imin, imax, jmin, jmax : 4 integers
                             4 indices allowing to slice a 2D np.ndarray

    .. todo::  merge with check_array_indices()
    """
    xpic, ypic = center
    if flipxycenter:
        ypic, xpic = center

    xpic, ypic = int(xpic), int(ypic)

    if isinstance(halfboxsizeROI, int):
        boxsizex, boxsizey = halfboxsizeROI, halfboxsizeROI
    else:
        boxsizex, boxsizey = halfboxsizeROI

    # TODO check also  if xpic and ypic is within framedim

    x1 = max(0, xpic - boxsizex)
    x2 = min(arrayshape[0], xpic + boxsizex)
    y1 = max(0, ypic - boxsizey)
    y2 = min(arrayshape[1], ypic + boxsizey)

    imin, imax, jmin, jmax = int(y1), int(y2), int(x1), int(x2)

    return imin, imax, jmin, jmax


def check_array_indices(imin, imax, jmin, jmax, framedim=None):
    r"""
    Return 4 indices for array slice compatible with framedim

    :param imin, imax, jmin, jmax: 4 integers
                            mini. and maxi. indices in both directions
    :param framedim: iterable of 2 integers
               shape of the array to be sliced by means of the 4 indices

    :return: imin, imax, jmin, jmax: 4 integers
                            mini. and maxi. indices in both directions

    .. todo:: merge with getindices2cropArray()
    """
    if framedim is None:
        print("framedim is empty in check_array_indices()")
        return
    imin = max(imin, 0)
    jmin = max(jmin, 0)
    imax = min(framedim[0], imax)
    jmax = min(framedim[1], jmax)

    return imin, imax, jmin, jmax


### Modify images
def to8bits(PILimage, normalization_value=None):
    r"""
    convert PIL image (16 bits) in 8 bits PIL image

    :return:    - [0]  8 bits image
                - [1] corresponding pixels value array

    .. todo:: since not used, may be deleted
    """

    imagesize = PILimage.size
    image8bits = Image.new("L", imagesize)
    rawdata = np.array(PILimage.getdata())
    if not normalization_value:
        normalization_value = 1.0 * np.amax(rawdata)
    datatoput = np.array(rawdata / normalization_value * 255, dtype="uint8")
    image8bits.putdata(datatoput)

    return image8bits, datatoput


# --- -------------  getting data from images or ROI
def diff_pix(pix, array_pix, radius=1):
    r"""
    returns
    index in array_pix which is the closest to pix if below the tolerance radius

    array_pix: array of 2d pixel points
    pix: one 2elements pixel point
    """
    dist2 = sum((pix - array_pix) ** 2, axis=1)
    closepix = np.argmin(dist2)
    if closepix:
        if dist2[closepix] <= radius ** 2:
            return closepix
        else:
            return None


def minmax(D_array, center, boxsize, framedim=(2048, 2048), withmaxpos=False):
    r"""
    extract min and max from a 2d array in a ROI

    Obsolete? Still used in LocalMaxima_ShiftArrays()

    Parameters
    D_array : 2D array
              data array
    center : iterable of 2 integers
             (x,y) pixel center
    boxsize : integer or iterable of 2 integers
              full boxsize defined in both directions
    framedim : iterable of 2 integers
               shape of D_array

    Return
    [min, max]: minimium and maximum pixel internsity in ROI
    [min, max],absolute_max_pos : if withmaxpos is True  add in output
                                  the absolute position of the largest pixel

    #TODO: replace by scipy.ndimage.extrema
    # see next functions below
    # framedim = from dictionary of CCDs
    D_array shape is flip(framedim)
    """
    if not isinstance(boxsize, int):
        boxsize = (boxsize, boxsize)
    #    print "framedim in minmax", framedim
    #    print "D_array.shape", D_array.shape

    # halfbox = int(boxsize / 2)

    #    xc, yc = center
    #    imin, imax, jmin, jmax = max(0, yc - halfbox), \
    #                            min(yc + halfbox, framedim[0]), \
    #                            max(0, xc - halfbox), \
    #                            min(framedim[1], xc + halfbox)
    #
    #    print "imin, imax, jmin, jmax", imin, imax, jmin, jmax
    framedim = framedim[1], framedim[0]
    imin, imax, jmin, jmax = getindices2cropArray(center, boxsize, framedim, flipxycenter=1)

    fulldata = D_array
    array_short = fulldata[imin:imax, jmin:jmax]

    mini_in_ROI = np.amin(array_short)
    maxi_in_ROI = np.amax(array_short)

    absolute_max_pos = ndimage.maximum_position(array_short) + np.array([imin, jmin])

    if withmaxpos:
        return [mini_in_ROI, maxi_in_ROI], absolute_max_pos

    else:
        return [mini_in_ROI, maxi_in_ROI]


def getExtrema(data2d, center, boxsize, framedim, ROIcoords=0, flipxycenter=True, verbose=0):
    r"""
    return  min max XYposmin, XYposmax values in ROI

    :param ROIcoords: 1 in local array indices coordinates
                0 in X,Y pixel CCD coordinates
    :param flipxycenter: boolean like
                   swap input center coordinates
    :param data2d: 2D array
             data array as read by :func:`readCCDimage`

    :return: min, max, XYposmin, XYposmax:
        - min : minimum pixel intensity
        - max : maximum pixel intensity
        - XYposmin : list of absolute pixel coordinates of lowest pixel
        - XYposmax : list of absolute pixel coordinates of largest pixel

    """
    if center is None or len(center) == 0:
        raise ValueError("center (peak list) in getExtrema is empty")

    indicesborders = getindices2cropArray(center, [boxsize, boxsize], framedim,
                                                                        flipxycenter=flipxycenter)
    imin, imax, jmin, jmax = indicesborders

    if verbose > 0: print("imin, imax, jmin, jmax", imin, imax, jmin, jmax)
    datacropped = data2d[imin:imax, jmin:jmax]

    # mini, maxi, posmin, posmax

    mini, maxi, posmin, posmax = ndimage.measurements.extrema(datacropped)

    if ROIcoords:
        return mini, maxi, posmin, posmax
    else:
        max_i, max_j = posmax
        min_i, min_j = posmin

        if flipxycenter:
            centery, centerx = center
        else:
            centerx, centery = center
        #        print "local position of maximum i,j", max_i, max_j

        globalXmax = max_j + centerx - boxsize
        globalYmax = max_i + centery - boxsize

        globalXmin = min_j + centerx - boxsize
        globalYmin = min_i + centery - boxsize

        #        print "Highest intensity %.f at (X,Y): (%d,%d) " % (maxi, globalX, globalY)
        if flipxycenter:
            return mini, maxi, [globalYmin, globalXmin], [globalYmax, globalXmax]
        else:
            return mini, maxi, [globalXmin, globalYmin], [globalXmax, globalYmax]

def getIntegratedIntensity(data2d, center, boxsize, framedim,
                                                        thresholdlevel=0.2, flipxycenter=True):
    r"""
    return  crude estimate of integrated intensity of peak above a given relative threshold

    :param ROIcoords: 1 in local array indices coordinates
                0 in X,Y pixel CCD coordinates
    :param flipxycenter: boolean like
                   swap input center coordinates
    :param data2d: 2D array
             data array as read by :func:`readCCDimage`

    :param Thresholdlevel:  relative level above which pixel intensity must be taken into account
                I(p)- minimum> Thresholdlevel* (maximum-minimum)

    :return: integrated intensity, minimum absolute intensity, nbpixels used for the summation

    """
    if not center:
        raise ValueError("center (peak list) in getExtrema is empty")

    indicesborders = getindices2cropArray(center, [boxsize, boxsize], framedim, flipxycenter=flipxycenter)
    imin, imax, jmin, jmax = indicesborders

    #     print "imin, imax, jmin, jmax", imin, imax, jmin, jmax
    datacropped = data2d[imin:imax, jmin:jmax]

    # mini, maxi, posmin, posmax
    mini, maxi, _, _ = ndimage.measurements.extrema(datacropped)

    minimum_amplitude = thresholdlevel * (maxi - mini) + mini
    print("integration for pixel intensity higher than: ", minimum_amplitude)
    pixelsabove = datacropped[datacropped > minimum_amplitude]
    nbpixels = len(pixelsabove)
    print("nb pixels above threshold  ", nbpixels)
    return ndimage.measurements.sum(pixelsabove), minimum_amplitude, nbpixels


def getMinMax(data2d, center, boxsize, framedim):
    r"""
    return min and max values in ROI

    Parameters:

    data2d : 2D array
             array as read by readCCDimage
    """
    return getExtrema(data2d, center, boxsize, framedim)[:2]


def minmax_fast(D_array, centers, boxsize=(25, 25)):
    r"""
    extract min (considered as background in boxsize) and intensity at center
    from a 2d array at different places (centers)

    centers is tuple a two array (  array([slow indices]),  array([fast indices]))

    return:

    [0] background values
    [1] intensity value

    used?
    """

    min_array = ndimage.minimum_filter(D_array, size=boxsize)

    return [min_array[centers], D_array[centers]]





# --- ------------- Mexican Hat 2D kernel
def myfromfunction(f, s, t):
    return np.fromfunction(f, s).astype(t)


def normalize_shape(shape):
    r"""
    return shape
    in case a scalar was given:
    return (shape,)
    """
    try:
        len(shape)
        return shape
    except TypeError:
        return (shape,)


def LoG(r, sigma=None, dim=1, r0=None, peakVal=None):
    r"""note:
         returns *negative* Laplacian-of-Gaussian (aka. mexican hat)
         zero-point will be at sqrt(dim)*sigma
         integral is _always_ 0
         if peakVal is None:  uses "mathematical" "gaussian derived" norm
         if r0 is not None: specify radius of zero-point (IGNORE sigma !!)
    """
    r2 = r ** 2

    if sigma is None:
        if r0 is not None:
            sigma = float(r0) / np.sqrt(dim)
        else:
            raise ValueError("One of sigma or r0 have to be non-None")
    else:
        if r0 is not None:
            raise ValueError("Only one of sigma or r0 can be non-None")
    s2 = sigma ** 2
    dsd = dim * sigma ** dim

    if peakVal is not None:
        norm = peakVal / dsd
    else:
        norm = 1.0 / (s2 * (2.0 * np.pi * sigma) ** (dim / 2.0))
    return np.exp(-r2 / (2.0 * s2)) * (dsd - r2) * norm


def LoGArr(shape=(256, 256),
                r0=None,
                sigma=None,
                peakVal=None,
                orig=None,
                wrap=0,
                dtype=np.float32):
    r"""returns n-dim Laplacian-of-Gaussian (aka. mexican hat)
    if peakVal   is not None
         result max is peakVal
    if r0 is not None: specify radius of zero-point (IGNORE sigma !!)

    credits: "Sebastian Haase <haase@msg.ucsf.edu>"
    """
    shape = normalize_shape(shape)
    dim = len(shape)
    return radialArr(shape,
                    lambda r: LoG(r, sigma=sigma, dim=dim, r0=r0, peakVal=peakVal),
                    orig,
                    wrap,
                    dtype)


def radialArr(shape, func, orig=None, wrap=False, dtype=np.float32):
    r"""generates and returns radially symmetric function sampled in volume(image) of shape shape
    if orig is None the origin defaults to the center
    func is a 1D function with 1 paramater: r

    if shape is a scalar uses implicitely `(shape,)`
    wrap tells if functions is continued wrapping around image boundaries
    wrap can be True or False or a tuple same length as shape:
       then wrap is given for each axis sperately
    """
    shape = normalize_shape(shape)
    try:
        len(shape)
    except TypeError:
        shape = (shape,)

    if orig is None:
        orig = (np.array(shape, dtype=np.float) - 1) / 2.0
    else:
        try:
            oo = float(orig)
            orig = np.ones(shape=len(shape)) * oo
        except:
            pass

    if len(shape) != len(orig):
        raise ValueError("shape and orig not same dimension")

    try:
        if len(wrap) != len(shape):
            raise ValueError("wrap tuple must be same length as shape")
    except TypeError:
        wrap = (wrap,) * len(shape)

    def wrapIt(ax, q):
        if wrap[ax]:
            nq = shape[ax]
            return np.where(q > nq // 2, q - nq, q)
        else:
            return q

    #     if wrap:
    #         def wrapIt(q, nq):
    #             return np.where(q>nq/2,q-nq, q)
    #     else:
    #         def wrapIt(q, nq):
    #             return q

    #     if len(shape) == 1:
    #         x0 = orig[0]  # 20060606: [0] prevents orig (as array) promoting its dtype (e.g. Float64) into result
    #         nx = shape[0]
    #         return myfromfunction(lambda x: func(wrapIt(np.absolute(x-x0),nx)), shape, dtype)
    #     elif len(shape) == 2:
    #         y0,x0 = orig
    #         ny,nx=shape
    #         return myfromfunction(lambda y,x: func(
    #                                              np.sqrt( \
    #             (wrapIt((x-x0),nx))**2 + (wrapIt((y-y0),ny))**2 ) ), shape, dtype)
    #     elif len(shape) == 3:
    #         z0,y0,x0 = orig
    #         nz,ny,nx=shape
    #         return myfromfunction(lambda z,y,x: func(
    #                                              np.sqrt( \
    #             (wrapIt((x-x0),nx))**2 + (wrapIt((y-y0),ny))**2 + (wrapIt((z-z0),nz))**2 ) ), shape, dtype)
    if len(shape) == 1:
        x0 = orig[0]  # 20060606: [0] prevents orig (as array) promoting its dtype (e.g. Float64) into result
        return myfromfunction(lambda x: func(wrapIt(0, np.absolute(x - x0))), shape, dtype)
    elif len(shape) == 2:
        y0, x0 = orig
        return myfromfunction(lambda y, x: func(
                np.sqrt((wrapIt(-1, x - x0)) ** 2 + (wrapIt(-2, y - y0)) ** 2)
            ), shape, dtype)
    elif len(shape) == 3:
        z0, y0, x0 = orig
        return myfromfunction(lambda z, y, x: func(
                np.sqrt((wrapIt(-1, x - x0)) ** 2
                    + (wrapIt(-2, y - y0)) ** 2
                    + (wrapIt(-3, z - z0)) ** 2)), shape, dtype)
    else:
        raise ValueError("only defined for dim < 3 (#TODO)")


# --- --------------------  Local Maxima or Local Hot pixels search
def LocalMaxima_ndimage(Data,
                        peakVal=4,
                        boxsize=5,
                        central_radius=2,
                        threshold=1000,
                        connectivity=1,
                        returnfloatmeanpos=0,
                        autothresholdpercentage=None):

    r"""
    returns (float) i,j positions in array of each blob
    (peak, spot, assembly of hot pixels or whatever)

    .. note:: used only in LocalMaxima_KernelConvolution

    inputs

    peakVal, boxsize, central_radius:
        parameters for numerical convolution with a mexican-hat-like kernel

    threshold:
        intensity threshold of filtered Data (by convolution with the kernel)
        above which blob signal will be considered
        if = 0 : take all blobs at the expense of processing time

    connectivity :
        1 for filled square 3*3 connectivity
        0 for 3*3 star like connectivity

    autothresholdpercentage :
        threshold in filtered image with respect to the maximum intensity in filtered image

    output:
    array (n,2): array of 2 indices
    """
    aa = ConvolvebyKernel(Data, peakVal=peakVal, boxsize=boxsize, central_radius=central_radius)

    # print("Histogram after convolution with Mexican Hat")
    # print(np.histogram(aa))

    if autothresholdpercentage is None:
        thraa = np.where(aa > threshold, 1, 0)
    else:
        thraa = np.where(aa > (autothresholdpercentage / 100.0) * np.amax(aa), 1, 0)

    if connectivity == 0:
        star = np.eye(3)
        ll, nf = ndimage.label(thraa, structure=star)
    elif connectivity == 1:
        ll, nf = ndimage.label(thraa, structure=np.ones((3, 3)))
    elif connectivity == 2:
        ll, nf = ndimage.label(thraa, structure=np.array([[1, 1, 1], [0, 1, 0], [1, 1, 1]]))
    elif connectivity == 3:

        ll, nf = ndimage.label(thraa, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    meanpos = np.array(ndimage.measurements.maximum_position(thraa, ll, np.arange(1, nf + 1)),
                                                                                    dtype=np.float)

    if returnfloatmeanpos:
        return meanpos
    else:
        return np.array(meanpos, dtype=np.int)


def ConvolvebyKernel(Data, peakVal=4, boxsize=5, central_radius=2):
    r"""
    Convolve Data array witn mexican-hat kernel

    inputs:
    Data                            : 2D array containing pixel intensities
    peakVal > central_radius        : defines pixel distance from box center where weights are positive
                                    (in the middle) and negative farther to converge back to zero
    boxsize                            : size of the box

    ouput:
    array  (same shape as Data)
    """

    # from scipy import ndimage

    # outa = np.zeros(Data.shape)
    # ndimage.filters.gaussian_laplace(d,(5,5),output=outa)

    # whole_structure= createstructure(10, 10)-2*createstructure(10, 7)+4*createstructure(10, 5)
    # bb= ndimage.convolve(d,whole_structure)

    # bb=ndimage.morphology.white_tophat(Data,(boxsize,boxsize))
    # mexicanhat = array(LoGArr((10,10),r0=6,peakVal=4),dtype= int16)

    mexicanhat = LoGArr((boxsize, boxsize), r0=central_radius, peakVal=peakVal)
    mexicanhat = mexicanhat - sum(mexicanhat) / mexicanhat.size
    bb = ndimage.convolve(np.array(Data, dtype=np.float32), mexicanhat)

    return bb


def LocalMaxima_KernelConvolution(Data, framedim=(2048, 2048),
                            peakValConvolve=4, boxsizeConvolve=5, central_radiusConvolve=2,
                            thresholdConvolve=1000,
                            connectivity=1,
                            IntensityThreshold=500,
                            boxsize_for_probing_minimal_value_background=30,
                            return_nb_raw_blobs=0,
                            peakposition_definition="max"):  # full side length
    r"""
    return local maxima (blobs) position and amplitude in Data by using
    convolution with a mexican hat like kernel.

    Two Thresholds are used sequently:
        - thresholdConvolve : level under which intensity of kernel-convolved array is discarded
        - IntensityThreshold : level under which blob whose local intensity amplitude in raw array is discarded

    :param Data: 2D array containing pixel intensities

    :param peakValConvolve, boxsizeConvolve, central_radiusConvolve: convolution kernel parameters

    :param thresholdConvolve: minimum threshold (expressed in unit of convolved array intensity)
                        under which convoluted blob is rejected.It can be zero
                        (all blobs are accepted but time consuming)
    :param connectivity: shape of connectivity pattern to consider pixels belonging to the
                   same blob.
                       - 1: filled square  (1 pixel connected to 8 neighbours)
                       - 0: star (4 neighbours in vertical and horizontal direction)

    :param IntensityThreshold: minimum local blob amplitude to accept

    :param boxsize_for_probing_minimal_value_background: boxsize to evaluate the background
                                                        and the blob amplitude

    :param peakposition_definition: string ('max' or 'center')
                              key to assign to the blob position its hottest pixel position
                              or its center (no weight)
    :return:
        peakslist : array like (n,2)
                list of peaks position (pixel)
        Ipixmax : array like (n,1) of integer
             list of highest pixel intensity in the vicinity of each peak
        npeaks : integer
             nb of peaks (if return_nb_raw_blobs =1)

    """
    # print("framedim in LocalMaxima_KernelConvolution", framedim)
    # print("Data.shape", Data.shape)
    dataimage_ROI = Data

    peak = LocalMaxima_ndimage(dataimage_ROI,
                                peakVal=peakValConvolve,
                                boxsize=boxsizeConvolve,
                                central_radius=central_radiusConvolve,
                                threshold=thresholdConvolve,
                                connectivity=connectivity,
                                returnfloatmeanpos=0)

    if len(peak) == 0:
        return None

    peak = (peak[:, 0], peak[:, 1])

    intensity_localmaxima = dataimage_ROI[peak]

    peaki = peak[0]
    peakj = peak[1]

    # building an array of hot pixels (2 coordinates)
    Yarray = peakj
    Xarray = peaki
    peaklist = np.array([Xarray, Yarray]).T

    #    print "peaklist", peaklist
    #     print "%d local maxima have been found from convolution method" % len(peaklist)
    #    print "peaklist[:3]", peaklist[:3]

    # probing background and maximal intensity in boxsize
    #
    ptp_boxsize = boxsize_for_probing_minimal_value_background

    # first method ---------------------------
    tabptp = []  # tab of min and max around each peak
    tabposmax = []  # # tab of position of hottest pixel close to that found after convolution
    for k in list(range(len(peaklist))):
        #        print "k in LocalMaxima_KernelConvolution", k
        #        print "dataimage_ROI.shape", dataimage_ROI.shape
        minimaxi, maxpos = minmax(dataimage_ROI, peaklist[k], ptp_boxsize, framedim=framedim, withmaxpos=1)
        tabptp.append(minimaxi)
        #         if minimaxi[1] > 4000:
        #             print "k, peaklist[k]", k, peaklist[k]
        #             print maxpos, minimaxi[1]

        #        mini, maxi, minpos, maxpos = getExtrema(dataimage_ROI, peaklist[k], ptp_boxsize, framedim,
        #                                                ROIcoords=0,
        #                                                flipxycenter=1)
        #
        #        tabptp.append([mini, maxi])
        #
        #        if maxi > 4000:
        #            print "k, peaklist[k]", k, peaklist[k]
        #            print maxpos, maxi

        tabposmax.append(maxpos)  # # new

    #    # to test peaks position / framedim
    #    print minmax(dataimage_ROI,
    #                                 [2580, 2750],
    #                                ptp_boxsize,
    #                                framedim=framedim,
    #                                withmaxpos=1)

    ar_ptp = np.array(tabptp)
    ar_posmax = np.array(tabposmax)  # # new

    #     print ' ar_posmax[:3]', ar_posmax[:10]
    #     print ' ar_posmax[:3]', ar_posmax[10:20]
    #     print ' ar_posmax[:3]', ar_posmax[20:30]

    #     print "saturation at", np.where(ar_ptp > 65000.)
    # -------------------------------------------

    # second method ---------------
    # tabptp = minmax_fast(dataimage_ROI,
    #                 tuple(transpose(peaklist)),
    #                    boxsize=(boxsize_for_probing_minimal_value_background,
    #                                boxsize_for_probing_minimal_value_background))
    # ar_ptp = array(tabptp).T
    # ---------------------------------

    # ar_amp = np.subtract(ar_ptp[:,1],ar_ptp[:,0])
    ar_amp = np.subtract(intensity_localmaxima, ar_ptp[:, 0])
    amp_rank = np.argsort(ar_amp)[::-1]

    peaklist_sorted = peaklist[amp_rank]
    ptp_sorted = ar_ptp[amp_rank]
    amp_sorted = ar_amp[amp_rank]
    posmax_sorted = ar_posmax[amp_rank]  # # new
    # thresholding on peak-to-peak amplitude
    threshold_amp = IntensityThreshold

    cond = np.where(amp_sorted > threshold_amp)
    th_peaklist = peaklist_sorted[cond]
    th_ar_ptp = ptp_sorted[cond]
    th_ar_amp = amp_sorted[cond]
    th_ar_pos = posmax_sorted[cond]  # # new

    #     print "th_ar_ptp", th_ar_ptp[:10]

    ##### peak positions that will be returned are the hottest pixels
    if peakposition_definition == "max":
        th_peaklist = th_ar_pos
    else:
        # using th_peaklist which is float position
        pass

    # print("{} local maxima found after thresholding above {} (amplitude above local background)".format(
    #         len(th_ar_amp), threshold_amp))

    # NEW --- from method array shift!
    # remove duplicates (close points), the most intense pixel is kept
    # minimum distance between hot pixel
    # it corresponds both to distance between peaks and peak size ...
    pixeldistance = 10  # pixeldistance_remove_duplicates

    purged_pklist, index_todelete = GT.purgeClosePoints2(th_peaklist, pixeldistance)

    purged_amp = np.delete(th_ar_amp, index_todelete)
    purged_ptp = np.delete(th_ar_ptp, index_todelete, axis=0)

    #     print 'shape of purged_ptp method conv.', purged_ptp.shape
    # print("{} local maxima found after removing duplicates (minimum intermaxima distance = {})".format(
    #         len(purged_amp), pixeldistance))

    # print "purged_pklist", purged_pklist
    #     print "shape(purged_pklist)", np.shape(purged_pklist)

    npeaks = np.shape(purged_pklist)[0]
    # print np.shape(Data)

    Ipixmax = purged_ptp[:, 1]
    # print "Ipixmax = ", Ipixmax

    peakslist = np.fliplr(purged_pklist)

    if return_nb_raw_blobs:
        return peakslist, Ipixmax, npeaks
    else:
        return peakslist, Ipixmax

    # NEW --- !
    # -----------------------


#     npeaks = np.shape(th_peaklist)[0]
#     Ipixmax = np.zeros(npeaks, dtype=int)
#     # print np.shape(Data)
#     for i in list(range(npeaks)):
#         # Ipixmax[i]=Data[th_peaklist[i,0],th_peaklist[i,1]]
#         Ipixmax[i] = th_ar_ptp[i][1]
#
#     if return_nb_raw_blobs == 1:
#         return np.fliplr(th_peaklist), Ipixmax, len(peaklist)
#     else:
#         return np.fliplr(th_peaklist), Ipixmax


def LocalMaxima_ShiftArrays(Data, framedim=(2048, 2048), IntensityThreshold=500,
                                Saturation_value=65535,
                                boxsize_for_probing_minimal_value_background=30,  # full side length
                                nb_of_shift=25,
                                pixeldistance_remove_duplicates=25,
                                verbose=0):
    r""" blob search or local maxima search by shift array method (kind of derivative)

    .. warning:: Flat peak (= two neighbouring pixel with rigourouslty the same intensity)
            is not detected
    """

    try:
        import networkx as NX
    except ImportError:
        print("\n***********************************************************")
        print("networkx module is missing! Some functions may not work...\nPlease install it at http://networkx.github.io/")
        print("***********************************************************\n")

    xminfit2d, xmaxfit2d, yminfit2d, ymaxfit2d = (1, framedim[1], 1, framedim[0])

    # warning i corresponds to y
    # j corresponds to x
    # change nom xminf2d => xminfit2d pour coherence avec le reste

    imin, _, jmin, _ = (framedim[0] - ymaxfit2d,
                                framedim[0] - yminfit2d,
                                xminfit2d - 1,
                                xmaxfit2d - 1)

    # dataimage_ROI=dataimage[imin:imax,jmin:jmax]# array index   i,j
    # # fit2d index:  X=j Y=2048-i

    dataimage_ROI = Data

    # print("searching local maxima for non saturated consecutive pixels")

    peak = localmaxima(dataimage_ROI, nb_of_shift, diags=1)

    # print("Done...!")
    # print(peak)
    # print "execution time : %f  secondes"%(ttt.time()-time_0)

    intensity_localmaxima = dataimage_ROI[peak]
    # print intensity_localmaxima

    # SATURATION handling ------------------------------
    # if the top of the local maximum has at least two pixels with the same intensity, this maximum is not detected
    # this generally the case for saturated peaks
    # saturation value : saturation above which we will take into account the pixel
    # this value may be lower than the 2^n bits value to handle unfortunately very flat weak peak with 2 neighbouring pixels
    # Saturation_value = 65535 for mccd
    Size_of_pixelconnection = 20
    # print("Saturation value for flat top peak handling", Saturation_value)
    sat_pix = np.where(dataimage_ROI >= Saturation_value)

    if verbose:
        print("positions of saturated pixels \n", sat_pix)
    # print("nb of saturated pixels", len(sat_pix[0]))
    sat_pix_mean = None

    # there is at least one peak above or equal to the Saturation_value threshold
    # loop over saturated pixels
    if len(sat_pix[0]) > 0:

        if verbose:
            print("positions of saturated pixels \n", sat_pix)

        # use of graph algorithms
        sat_pix = np.column_stack(sat_pix)

        disttable_sat = ssd.pdist(sat_pix, "euclidean")
        sqdistmatrix_sat = ssd.squareform(disttable_sat)
        # building adjencymat

        a, b = np.indices(sqdistmatrix_sat.shape)
        indymat = np.triu(b) + np.tril(a)
        cond2 = np.logical_and(sqdistmatrix_sat < Size_of_pixelconnection, sqdistmatrix_sat > 0)
        adjencymat = np.where(cond2, indymat, 0)

        # print "before networkx"
        # print("networkx version", NX.__version__)
        GGraw = NX.to_networkx_graph(adjencymat, create_using=NX.Graph())
        list_of_cliques = NX.find_cliques(GGraw)
        # print "after networkx"

        # now find average pixel of each clique
        sat_pix_mean = []
        for clique in list_of_cliques:
            ii, jj = np.mean(sat_pix[clique], axis=0)
            sat_pix_mean.append([int(ii), int(jj)])

        sat_pix_mean = np.array(sat_pix_mean)
        # print("Mean position of saturated pixels blobs = \n", sat_pix_mean)

        # if 0:  # of scipy.ndimage

        #     df = ndimage.gaussian_filter(dataimage_ROI, 10)

        #     # histo = np.histogram(df)
        #     # print "histogram",histo
        #     # print "maxinten",np.amax(df)
        #     threshold_for_measurements = (
        #         np.amax(df) / 10.0
        #     )  # histo[1][1]# 1000  pour CdTe # 50 pour Ge

        #     tG = np.where(df > threshold_for_measurements, 1, 0)
        #     ll, nf = ndimage.label(tG)  # , structure = np.ones((3,3)))
        #     meanpos = np.array(
        #         ndimage.measurements.center_of_mass(tG, ll, np.arange(1, nf + 1)), dtype=float)
        #     # meanpos = np.fliplr(meanpos)  # this done later

        #     # print "meanpos",meanpos

        #     sat_pix_mean = meanpos
    # else:
    #     print("No pixel saturation")
    # SATURATION handling -(End) --------------------------------------------------------

    # x,y from localmaxima is a matter of convention

    peaki = peak[0] + imin
    peakj = peak[1] + jmin

    # building an array of hot pixels (2 coordinates)
    Yarray = peakj
    Xarray = peaki
    peaklist = np.array([Xarray, Yarray]).T

    # print peaklistfit2D
    # print peaklist[100:150]
    # print("{} local maxima have been found".format(len(peaklist)))

    # probing background and maximal intensity in boxsize
    #
    ptp_boxsize = boxsize_for_probing_minimal_value_background

    tabptp = []
    for k in list(range(len(peaklist))):
        tabptp.append(minmax(dataimage_ROI, peaklist[k], ptp_boxsize, framedim=framedim))

    ar_ptp = np.array(tabptp)
    # ar_amp = np.subtract(ar_ptp[:,1],ar_ptp[:,0])
    ar_amp = np.subtract(intensity_localmaxima, ar_ptp[:, 0])
    amp_rank = np.argsort(ar_amp)[::-1]

    peaklist_sorted = peaklist[amp_rank]
    # ptp_sorted = ar_ptp[amp_rank]
    amp_sorted = ar_amp[amp_rank]
    # thresholding on peak-to-peak amplitude
    threshold_amp = IntensityThreshold

    cond = np.where(amp_sorted > threshold_amp)
    th_peaklist = peaklist_sorted[cond]
    th_ar_amp = amp_sorted[cond]

    # print("{} local maxima found after thresholding above {} amplitude above local background".format(
    #         len(th_ar_amp), threshold_amp))

    # remove duplicates (close points), the most intense pixel is kept
    # minimum distance between hot pixel
    # it corresponds both to distance between peaks and peak size ...
    pixeldistance = pixeldistance_remove_duplicates

    purged_pklist, index_todelete = GT.purgeClosePoints2(th_peaklist, pixeldistance)

    purged_amp = np.delete(th_ar_amp, tuple(index_todelete))
    # print(
    #     "{} local maxima found after removing duplicates (minimum intermaxima distance = {})".format(
    #         len(purged_amp), pixeldistance))

    # print "execution time : %f  secondes"%( ttt.time() - time_0)

    # merging different kind of peaks
    if sat_pix_mean is not None:
        # print("Merging saturated and normal peaks")
        # print("number of saturated peaks : ", np.shape(sat_pix_mean)[0])
        purged_pklist = np.vstack((sat_pix_mean, purged_pklist))

    if 0:  # check if there are still close hot pixels
        disttable_c = ssd.pdist(purged_pklist, "euclidean")
        maxdistance_c = np.amax(disttable_c)
        sqdistmatrix_c = ssd.squareform(disttable_c)
        distmatrix_c = sqdistmatrix_c + np.eye(sqdistmatrix_c.shape[0]) * maxdistance_c
        # must be (array([], dtype=int64), array([], dtype=int64))
        # print("close hotpixels", np.where(distmatrix_c < pixeldistance))
    # print "purged_pklist", purged_pklist
    # print("shape(purged_pklist)", np.shape(purged_pklist))
    npeaks = np.shape(purged_pklist)[0]
    Ipixmax = np.zeros(npeaks, dtype=int)
    # print np.shape(Data)

    for i in list(range(npeaks)):
        Ipixmax[i] = Data[purged_pklist[i, 0], purged_pklist[i, 1]]
        # print "Ipixmax = ", Ipixmax

    return np.fliplr(purged_pklist), Ipixmax

def shiftarrays_accum(Data_array, n, dimensions=1, diags=0):
    r"""
    idem than shiftarrays() but with all intermediate shifted arrays
    1D
    returns 3 arrays corresponding to shifted arrays
    by n in two directions and original one
    2D
    returns 5 arrays corresponding to shifted arrays
    by n in two directions and original one

    these arrays are ready for comparison with eg np.greater

    Data_array must have shape (slowdim,fastdim) so that
    slowdim-2*n>=1 and fastdim-2*n>=1
    (ie central array with zero shift has some elements)

    TODO: replace append by a pre allocated array

    .. note:: readmccd.localmaxima is better

    """
    if n <= 0:
        raise ValueError("shift value must be positive")

    if dimensions == 2:
        if diags:
            shift_zero = Data_array[n:-n, n:-n]

            allleft = []
            allright = []
            allup = []
            alldown = []
            alldiagleftdown = []  # diag "y=x"
            alldiagrightup = []  # diag "y=x"
            alldiagrightdown = []  #  diah "y=-x"
            alldiagleftup = []  #  diah "y=-x"

            for k in np.arange(1, n + 1)[::-1]:

                allleft.append(Data_array[n - k : -(n + k), n:-n])
                alldown.append(Data_array[n:-n, n - k : -(n + k)])
                alldiagrightdown.append(Data_array[n - k : -(n + k), n - k : -(n + k)])

                if (n - k) != 0:
                    allright.append(Data_array[k + n : -(n - k), n:-n])
                    allup.append(Data_array[n:-n, k + n : -(n - k)])
                    alldiagleftdown.append(
                        Data_array[k + n : -(n - k), n - k : -(n + k)])
                    alldiagleftup.append(Data_array[k + n : -(n - k), k + n : -(n - k)])
                    alldiagrightup.append(
                        Data_array[n - k : -(n + k), k + n : -(n - k)])

                else:  # correct python array slicing at the end :   a[n:0]  would mean a[n:]

                    allright.append(Data_array[k + n :, n:-n])
                    allup.append(Data_array[n:-n, k + n :])
                    alldiagleftdown.append(Data_array[k + n :, n - k : -(n + k)])
                    alldiagleftup.append(Data_array[k + n :, k + n :])
                    alldiagrightup.append(Data_array[n - k : -(n + k), k + n :])

            return (shift_zero,
                allleft,
                allright,
                alldown,
                allup,
                alldiagleftdown,
                alldiagrightup,
                alldiagrightdown,
                alldiagleftup)

        else:
            shift_zero = Data_array[n:-n, n:-n]

            allleft = []
            allright = []
            allup = []
            alldown = []

            allleft.append(Data_array[: -2 * n, n:-n])
            alldown.append(Data_array[n:-n, : -2 * n])

            for k in np.arange(1, n)[::-1]:
                allleft.append(Data_array[n - k : -(n + k), n:-n])
                allright.append(Data_array[k + n : -(n - k), n:-n])
                alldown.append(Data_array[n:-n, n - k : -(n + k)])
                allup.append(Data_array[n:-n, k + n : -(n - k)])

            allright.append(Data_array[2 * n :, n:-n])
            allup.append(Data_array[n:-n, 2 * n :])

            return shift_zero, allleft, allright, alldown, allup

    elif dimensions == 1:
        shift_zero = Data_array[n:-n]
        allleft = []
        allright = []
        allleft.append(Data_array[: -2 * n])
        for k in np.arange(1, n)[::-1]:
            allright.append(Data_array[k + n : -(n - k)])
            allleft.append(Data_array[n - k : -(n + k)])
        allright.append(Data_array[2 * n :])

        return shift_zero, allleft, allright

def LocalMaxima_from_thresholdarray(Data, IntensityThreshold=400, rois=None, framedim=None,
                                                                                    verbose=False,
                                                                                    outputIpixmax=False):
    r"""
    return center of mass of each blobs composes by pixels above IntensityThreshold

    if Centers = list of (x,y, halfboxsizex, halfboxsizey)  perform only blob search in theses ROIs

    .. warning:: center of mass of blob where all intensities are set to 1
    """
    if rois is not None:
        # print('\n>>>>> Finding only peaks in %d ROIs.\n' % len(rois))
        listmeanpos_roi = []
        for x, y, boxx, boxy in rois:

            centerj, centeri = x, y
            boxj, boxi = boxx, boxy
            (imin, imax, jmin, jmax) = (centeri - boxi, centeri + boxi + 1,
                                    centerj - boxj, centerj + boxj + 1)

            # avoid to wrong indices when slicing the data
            imin, imax, jmin, jmax = check_array_indices(imin, imax, jmin, jmax,
                                                                    framedim=framedim)
            # print("imin, imax, jmin, jmax", imin, imax, jmin, jmax)
            dataroi = Data[imin : imax, jmin : jmax]
            if verbose:
                print("\n------------------\nx,y, boxx, boxy", x, y, boxx, boxy)
                print('max intensity in dataroi', np.amax(dataroi))
                print('min intensity in dataroi', np.amin(dataroi))
                print('IntensityThreshold', IntensityThreshold)

            # other way equivalent
            # print("framedim in LocalMaxima_from_thresholdarray", framedim)
            # framedim = framedim[1], framedim[0]

            # i1, i2, j1, j2 = getindices2cropArray((x,y), (boxx, boxy), framedim)
            # #        print "i1, i2, j1, j2-----", i1, i2, j1, j2
            # dataroi = Data[i1:i2, j1:j2]

            # # for spot near border, replace by zeros array
            # if i2 - i1 != boxy * 2 or j2 - j1 != boxx * 2:
            #     dataroi = np.zeros((boxy * 2 + 1, boxx * 2 + 1))
            # print('max intensity in dataroi  2  :  ', np.amax(dataroi))

            # blob seach in dataroi
            thrData_for_label = np.where(dataroi > IntensityThreshold, 1, 0)

            ll, nf = ndimage.label(thrData_for_label)
            if nf == 0:
                print('sad! No blobs there in this roi...')
                continue

            meanpos_roi = np.array(ndimage.measurements.maximum_position(dataroi, ll, np.arange(1, nf + 1)),
                dtype=float)

            if len(np.shape(meanpos_roi)) > 1:
                meanpos_roi = np.fliplr(meanpos_roi)
            else:
                meanpos_roi = np.roll(meanpos_roi, 1)
            if verbose:
                print('meanpos_roi  =>', meanpos_roi)
            for pos in meanpos_roi:
                listmeanpos_roi.append([pos[0] + x - boxx, pos[1] + y - boxy])

        meanpos = np.array(listmeanpos_roi)

        # if verbose:
        #     print('meanpos', meanpos)

    # single ROI is whole Data
    else:
        thrData_for_label = np.where(Data > IntensityThreshold, 1, 0)

        #     thrData = np.where(Data > IntensityThreshold, Data, 0)

        #    star = array([[0,1,0],[1,1,1],[0,1,0]])
        # ll, nf = ndimage.label(thrData_for_label, structure=np.ones((3,3)))
        # ll, nf = ndimage.label(thrData_for_label, structure=star)
        ll, nf = ndimage.label(thrData_for_label)

        #     print "nb of blobs in LocalMaxima_from_thresholdarray()", nf

        if nf == 0:
            return None

        meanpos = np.array(ndimage.measurements.maximum_position(Data, ll, np.arange(1, nf + 1)))

        if len(np.shape(meanpos)) > 1:  # more than 1 peak
            meanpos = np.fliplr(meanpos)
        else:  # found a single peak
            meanpos = np.roll(meanpos, 1)

    if outputIpixmax:
        if len(np.shape(meanpos)) > 1:
            # print('meanpos' ,meanpos)
            JJ, II = meanpos.T
            Ipixmax = Data[II, JJ]
            # print('Ipixmax', Ipixmax)
        else:
            meanpos, Data[meanpos[1], meanpos[0]]
        return meanpos, Ipixmax

    else:
        return meanpos


def localmaxima(DataArray, n, diags=1, verbose=0):
    r"""
    from DataArray 2D  returns (array([i1,i2,...,ip]),array([j1,j2,...,jp]))
    of indices where pixels value is higher in two direction up to n pixels

    this tuple can be easily used after in the following manner:
    DataArray[tupleresult] is an array of the intensity of the hottest pixels in array

    in similar way with only four cardinal directions neighbouring (found in the web):
    import numpy as N
    def local_minima(array2d):
        return ((array2d <= np.roll(array2d,  1, 0)) &
                (array2d <= np.roll(array2d, -1, 0)) &
                (array2d <= np.roll(array2d,  1, 1)) &
                (array2d <= np.roll(array2d, -1, 1)))

    WARNING: flat top peak are not detected !!
    """
    dim = len(np.shape(DataArray))

    if diags:
        c, alll, allr, alld, allu, diag11, diag12, diag21, diag22 = shiftarrays_accum(
            DataArray, n, dimensions=dim, diags=diags)
        flag = np.greater(c, alll[0])
        for elem in alll[1:] + allr + alld + allu + diag11 + diag12 + diag21 + diag22:
            flag = flag * np.greater(c, elem)
    else:
        c, alll, allr, alld, allu = shiftarrays_accum(DataArray, n, dimensions=dim, diags=diags)
        flag = np.greater(c, alll[0])
        for elem in alll[1:] + allr + alld + allu:
            flag = flag * np.greater(c, elem)

    peaklist = np.nonzero(flag)  # in c frame index

    if verbose:
        print("value local max", c[peaklist])
        print("value from original array ", DataArray[tuple(np.array(peaklist) + n)])
        print("positions of local maxima in original frame index",
            tuple(np.array(peaklist) + n),)

    # first slow index array , then second fast index array
    return tuple(np.array(peaklist) + n)


# --- ----  Filtering and background removal function
def gauss_kern(size, sizey=None):
    r"""
    Returns a normalized 2D gauss kernel array for convolutions
    """
    size = int(size)
    if sizey is not None:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-1*size : size + 1, -1*sizey: sizey + 1]
    g = np.exp(-(x ** 2 / float(size) + y ** 2 / float(sizey)))
    return g / g.sum()


def blur_image(im, n, ny=None):
    r"""
    blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    g = gauss_kern(n, sizey=ny)
    improc = scipy.signal.convolve(im, g, mode="valid")
    return improc


def blurCCD(im, n):
    r"""
    apply a blur filter to image ndarray
    """
    framedim = im.shape
    minipix = np.amin(im)
    tab = minipix * np.ones(framedim)

    blurredpart = blur_image(im, n)
    tab[n:-n, n:-n] = blurredpart
    return tab


def circularMask(center, radius, arrayshape):
    r"""
    return a boolean ndarray of elem in array inside a mask
    """
    II, JJ = np.mgrid[0 : arrayshape[0], 0 : arrayshape[1]]

    cond = (II - center[0]) ** 2 + (JJ - center[1]) ** 2 <= radius ** 2
    return cond


def compute_autobackground_image(dataimage, boxsizefilter=10):
    r"""
    return 2D array of filtered data array
    :param dataimage: array of image data
    :type dataimage: 2D array
    """

    bkgimage = filter_minimum(dataimage, boxsize=boxsizefilter)

    return bkgimage


def computefilteredimage(dataimage, bkg_image, CCDlabel, kernelsize=5, formulaexpression="A-B",
                                                        usemask=True, verbose=0):
    r"""
    return 2D array of initial image data without background given by bkg_image data

    usemask        : True  then substract bkg image on masked raw data
                    False  apply formula on all pixels (no mask)

    :param dataimage: array of image data
    :type dataimage: 2D array
    :param bkg_image: array of filtered image data (background)
    :type bkg_image: 2D array
    :param CCDlabel: key for CCD dictionary
    :type CCDlabel: string
    """

    framedim = DictLT.dict_CCD[CCDlabel][0]
    SaturationLevel = DictLT.dict_CCD[CCDlabel][2]
    dataformat = DictLT.dict_CCD[CCDlabel][5]

    if verbose:
        print("framedim in computefilteredimage ", framedim)
        print("CCDlabel in computefilteredimage ", CCDlabel)
    #
    #     if CCDlabel in ('EDF',):
    #         return dataimage

    #     if framedim not in ((2048, 2048), [2048, 2048]):
    #         raise ValueError, "Background removal still implemented for non squared camera "
    # computing substraction on whole array
    if CCDlabel in ("sCMOS", "sCMOS_fliplr", "sCMOS_16M"):
        usemask = False

    if usemask:
        # mask parameter to avoid high intensity steps at border:
        # TODO: to compute for all CCD types
        #     center, mask_radius, minvalue = (1024, 1024), 1010, 0

        # radius of the mask a bit smaller than real radius avoiding circular intensity step

        mask_radius = min(framedim[0] // 2, framedim[1] // 2) - 15
        center, minvalue = (framedim[0] // 2, framedim[1] // 2), 0

        # print("mask_radius", mask_radius)
        # print("center", center)

        dataarray2D_without_background = filterimage(dataimage, framedim,
                                                    blurredimage=bkg_image,
                                                    kernelsize=kernelsize,
                                                    mask_parameters=(center, mask_radius, minvalue),
                                                    clipvalues=(0, SaturationLevel),
                                                    imageformat=dataformat)

    else:
        dataarray2D_without_background = applyformula_on_images(dataimage, bkg_image,
                                                            formulaexpression=formulaexpression,
                                                            SaturationLevel=SaturationLevel,
                                                            clipintensities=True)

    return dataarray2D_without_background


def filterimage(image_array, framedim, blurredimage=None,
                                        kernelsize=5,
                                        mask_parameters=None,
                                        clipvalues=None,
                                        imageformat=np.uint16):
    r"""
    compute a difference of images inside a region defined by a mask

    :param blurredimage:    ndarray image to substract to image_array
    :param kernelsize:    pixel size of gaussian kernel if blurredimage is None

    :param mask_parameters: circular mask parameter: center=(x,y), radius, value outside mask
    """
    if blurredimage is None:
        dblur = blurCCD(image_array, kernelsize)
    else:
        dblur = blurredimage

    if clipvalues is not None:
        minival, maxival = clipvalues
        tab = np.clip(image_array - dblur, minival, maxival)
    else:
        tab = image_array - dblur

    if mask_parameters is not None:
        center, radius, minvalue = mask_parameters
        if minvalue == "minvalue":
            minivalue = np.amin(image_array)
        else:
            minivalue = 0
        maskcd = np.where(circularMask(center, radius, framedim), tab, minivalue)

    return np.array(maskcd, dtype=imageformat)


def rebin2Darray(inputarray, bin_dims=(2, 2), operator="mean"):
    r"""
    rebin 2D array by applying an operator to define the value of one element from the other

    :param operator: mean, min, max, sum
    :param bin_dims: side sizes of binning. (2,3) means 2X3
    """
    rows, cols = inputarray.shape
    binr, binc = bin_dims
    if operator == "mean":
        op = np.mean
    if operator == "max":
        op = np.max
    if operator == "min":
        op = np.max
    if operator == "sum":
        op = np.sum
    if rows % binr == 0 and cols % binc == 0:
        return op(op(inputarray.reshape(rows // binr, binr, cols // binc, binc), axis=3), axis=1)
    else:
        print("array and binning size are not compatible")
        return None


def blurCCD_with_binning(im, n, binsize=(2, 2)):
    r"""
    blur the array by rebinning before and after aplying the filter
    """
    # framedim = im.shape
    imrebin = rebin2Darray(im, bin_dims=binsize, operator="min")
    if imrebin is None:
        return None

    dblur = blurCCD(imrebin, n)

    return np.repeat(np.repeat(dblur, binsize[0], axis=0), binsize[1], axis=1)


def filter_minimum(im, boxsize=10):
    r""" return filtered image using minimum filter"""
    return ndimage.filters.minimum_filter(im, size=boxsize)


def remove_minimum_background(im, boxsize=10):
    r"""
    remove to image array the array resulting from minimum_filter
    """
    return im - filter_minimum(im, boxsize=boxsize)

# --- -------------- Plot image and peaks
def plot_image_markers(image, markerpos, position_definition=1):
    r"""
    plot 2D array (image) with markers at first two columns of (markerpos)

    .. note:: used in LaueHDF5. Could be better implementation in some notebooks
    """
    fig = pp.figure()
    ax = fig.add_subplot(111)

    numrows, numcols = image.shape

    def format_coord(x, y):
        """ return string with x, y values """
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = image[row, col]
            return "x = {:.4f}, y = {:.4f}, z = {:.4f}".format(x, y, z)
        else:
            return "x = {:.4f}, y = {:.4f}".format(x, y)

    ax.format_coord = format_coord

    from matplotlib.patches import Circle

    # correction only to fitdata peak position to the display
    if position_definition == 1:
        PointToPlot = np.zeros(markerpos.shape)
        PointToPlot[:, :2] = markerpos[:, :2] - np.array([1, 1])

    for po in PointToPlot[:, :2]:

        large_circle = Circle(po, 7, fill=False, color="b")
        center_circle = Circle(po, 0.5, fill=True, color="r")
        ax.add_patch(large_circle)
        ax.add_patch(center_circle)

    ax.imshow(np.log(image), interpolation="nearest")

    pp.show()


def applyformula_on_images(A, B, formulaexpression="A-B", SaturationLevel=None,
                                                            clipintensities=True):
    r"""
    calculate image data array from math expression

    :param A, B: ndarray  of the same shape

    :param SaturationLevel:  saturation level of intensity

    :param clipintensities:   clip resulting intensities to zero and saturation value
    """
    if A.shape != B.shape:
        raise ValueError("input arrays in applyformula_on_images() have not the same shape.")

    A = np.array(A, dtype="float32")
    B = np.array(B, dtype="float32")

    #        nbpix = A.shape[0] * A.shape[1]
    #        resformula = np.zeros_like(A)
    #        resformula.dtype = 'int32'

    #        A.dtype = 'int32'
    #        B.dtype = 'int32'
    resformula = eval(formulaexpression)
    #        print resformula.dtype
    newarray = resformula

    if clipintensities:
        if SaturationLevel is not None:
            newarray = np.clip(resformula, 0, SaturationLevel)
        else:
            raise ValueError("saturation level is unknown to clip data for large intensity!\n "
                "Missing argument in applyformula_on_images()")

    return newarray
