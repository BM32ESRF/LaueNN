# -*- coding: utf-8 -*-
r"""
Module of lauetools project to compute Laue spots position on CCD camera.
It handles detection and source geometry.

.. warning::
    The frame (LT2) considered in this package (with y axis parallel to the incoming beam) in not the LaueTools frame (for which x is parallel to the incoming beam)

JS Micha June 2019

* Vectors Definitions
    - **q** momentum transfer vector from resp. incoming and outgoing wave vector **ki** and **kf**, :math:`q=kf-ki`

    - When a Laue spot exists, **q** is equal to the one node of the reciprocal lattice given by **G*** vector

    - **G*** is perpendicular to atomic planes defined by the three Miller indices h,k,l
        such as **G***=h**a*** + k**b*** +l**c*** where **a***, **b***, and **c*** are the unit cell lattice basis vectors.

    - **kf**: scattered beam vector whose corresponding unit vector is **uf**

    - **ki** incoming beam vector, **ui** corresponding unit vector

* Laboratory Frame LT2
    - I: origin

    - z vertical up perpendicular to CCD plane (top camera geometry)

    - y along X-ray horizontal

    - x towards wall behind horizontal

    - O: origin of pixel CCD frame in detecting plane

    - **j** // **ui** incoming beam unit vector

    - z axis is defined by the CCD camera position. z axis is perpendicular to CCD plane
        such as IO belongs to the plane Iyz
    - bet: angle between **IO** and **k**

    - **i**= **j**^**k** (when entering the BM32 hutch **i** is approximately towards the wall
        (in CCD on top geometry and beam coming from the right)
    - M: point lying in CCD plane corresponding to Laue spot

    - **uf** is the unit vector relative to vector **IM**

**kf** is also a vector collinear to **IM** with a length of R=1/wavelength=E/12.398 [keV]
with wavelength and Energy of the corresponding bragg's reflections.

I is the point from which calibration parameters (CCD position) are deduced (from a perfectly known crystal structure Laue pattern)
Iprime is an other source of emission (posI or offset in functions)

:math:`2 \theta` is the scattering angle between **ui** and **uf**, i.e.

:math:`\cos(2 \theta)=u_i.u_f`

.. math::

    {\bf k_f} = ( -\sin 2 \theta \sin \chi, \cos 2\theta, \sin 2\theta \cos \chi)

    {\bf k_i} = (0, 1, 0)

Energy= 12.398*  q**2/(2* **q**.**ui**)=12.398 * q**2/ (-2 sin theta)

*Calibration parameters (CCD position and detection geometry)*
    - calib: list of the 5 calibration parameters [dd,xcen,ycen,xbet,xgam]
    - dd: norm of **IO**  [mm]
    - xcen,ycen [pixel unit]: pixels values in CCD frame of point O with respect to Oprime where
        Oprime is the origin of CCD pixels frame (at a corner of the CCD array)
    - xbet: angle between **IO** and **k** [degree]
    - xgam: azimutal rotation angle around z axis. Angle between CCD array axes
        and (**i**,**j**) after rotation by xbet [degree].

*sample frame*

Origin is I and unit frame vectors (**is**, **js**, **ks**) are derived
from absolute frame by the rotation (axis= - **i**, angle= wo) where wo is the angle between **js** and **j**
"""
from __future__ import print_function
import os
import numpy as np
import pylab as P

try:
    from lauetools import findorient as FindO
    from lauetools import generaltools as GT
    from lauetools import IOLaueTools as IOLT
    from lauetools import CrystalParameters as CP
    from lauetools import dict_LaueTools as DictLT
except:
    import lauetoolsnn.lauetools.findorient as FindO
    import lauetoolsnn.lauetools.generaltools as GT
    import lauetoolsnn.lauetools.IOLaueTools as IOLT
    import lauetoolsnn.lauetools.CrystalParameters as CP
    import lauetoolsnn.lauetools.dict_LaueTools as DictLT
# -----------  CONSTANTS ------------------
RECTPIX = DictLT.RECTPIX  # see above  camera skewness

PI = np.pi
DEG = PI / 180.0

CST_CONV_LAMBDA_KEV = DictLT.CST_ENERGYKEV

# --- -----   old function  ---------------
norme = GT.norme_vec

# --- -------- geometrical functions relating 2theta, chi, pixel X, pixel Y, detector plane ----
def calc_uflab(xcam, ycam, detectorplaneparameters, offset=0, returnAngles=1, verbose=0,
                                                                    pixelsize=165.0 / 2048,
                                                                    rectpix=RECTPIX,
                                                                    kf_direction="Z>0",
                                                                    version=1):
    r"""
    Computes scattered unit vector :math:`{\bf u_f}=\frac{\bf k_f}{\|k_f\|}` in laboratory frame corresponding to :math:`k_f`
    (angle scattering angles 2theta and chi) from X, Y pixel Laue spot position

    Unit vector uf correspond to normalized kf vector: q = kf - ki
    from lists of X and Y Laue spots pixels positions on detector

    :param xcam: list of pixel X position
    :type xcam: list of floats
    :param ycam: list of pixel Y position
    :type ycam: list of floats
    :param detectorplaneparameters: list of 5 calibration parameters

    :param offset: float, offset in position along incoming beam of source of scattered rays
                if positive: offset in sample depth
                units: mm

    :returns:
        * if returnAngles=1   : twicetheta, chi   *(default)*
        * if returnAngles!=1  : uflab, IMlab
    """
    calib = detectorplaneparameters[:5]
    detect, xcen, ycen, xbet, xgam = np.array(calib) * 1.0
    #    print "pixelsize in calc_uflab ", pixelsize

    # transmission geometry
    if kf_direction in ("X>0",):
        if version == 1:
            return calc_uflab_trans(xcam,
                                ycam,
                                calib,
                                returnAngles=returnAngles,
                                verbose=verbose,
                                pixelsize=pixelsize,
                                rectpix=rectpix)
        elif version == 2:
            return calc_uflab_trans_2(xcam,
                                ycam,
                                calib,
                                returnAngles=returnAngles,
                                verbose=verbose,
                                pixelsize=pixelsize,
                                rectpix=rectpix)

    # back reflection geometry
    elif kf_direction in ("X<0",):
        return calc_uflab_back(xcam,
                                ycam,
                                calib,
                                returnAngles=returnAngles,
                                verbose=verbose,
                                pixelsize=pixelsize,
                                rectpix=rectpix)
    # 2theta=90 deg reflection geometry (top side+ and side -)
    elif kf_direction in ("Z>0", "Y>0", "Y<0"):
        cosbeta = np.cos(PI / 2.0 - xbet * DEG)
        sinbeta = np.sin(PI / 2.0 - xbet * DEG)

    else:
        raise ValueError("kf_direction = %s not implemented in calc_uflab" % str(kf_direction))

    cosgam = np.cos(- xgam * DEG)
    singam = np.sin(- xgam * DEG)

    xcam1 = (np.array(xcam) - xcen) * pixelsize
    ycam1 = (np.array(ycam) - ycen) * pixelsize * (1.0 + rectpix)

    xca0 = cosgam * xcam1 - singam * ycam1
    yca0 = singam * xcam1 + cosgam * ycam1

    # I impact point on sample (location of x-ray scattering or emission)
    # O centre of origin of pixel CCD
    # M belong to CCD plane
    # IM is parallel to kf
    # frame is not Lauetools'one
    # since here:  y // ki

    # IOlab = detect * array([0.0, cosbeta, sinbeta])
    # warning : cos and sin are exchanged due to definition above
    # cosbeta = np.cos(PI / 2.0 - xbet * DEG)  just above !!!!
    xO, yO, zO = detect * np.array([0.0, cosbeta, sinbeta])

    # OMlab = array([xca0, yca0*sinbeta, -yca0*cosbeta])
    xOM = xca0
    yOM = yca0 * sinbeta
    zOM = -yca0 * cosbeta

    # IMlab = IOlab + OMlab
    xM = xO + xOM
    yM = yO + yOM
    zM = zO + zOM
    IMlab = np.array([xM, yM, zM]).T

    # norm of IM vector
    # nIMlab=sqrt(dot(IMlab,IMlab))
    nIMlab = 1.0 * np.sqrt(xM ** 2 + yM ** 2 + zM ** 2)

    # print transpose(array([xM,yM,zM])) # vector joining source and pt on CCD in abs frame

    uflab = np.transpose(np.array([xM, yM, zM]) / nIMlab)

    #     print "uflab w/o source offset",uflab

    if offset not in (None, 0, 0.0):
        # with source offset along y (>0 if along the beam and in sample depth)
        # ufprimelab = unit(IpMlab) = unit(IpIlab+IMlab)
        IpMlab = np.array([0, offset, 0]) + IMlab
        normedIpM = np.sqrt(np.sum(IpMlab ** 2, axis=1)).reshape((len(IpMlab), 1))

        ufprime = 1.0 * IpMlab / normedIpM

        # print("ufprime, uflab with source offset", ufprime)

        uflab = ufprime

    # calculus of scattering angles
    EPS = 1e-17
    chi = np.arctan(-uflab[:, 0] / (uflab[:, 2] + EPS)) / DEG  # JSM convention
    #     chiXMAS = np.arctan(uflab[:, 0] / np.sqrt(uflab[:, 1] ** 2 + uflab[:, 2] ** 2)) / DEG
    #     chiXMAS2 = np.arctan(np.sqrt(uflab[:, 0] ** 2 + uflab[:, 1] ** 2) / uflab[:, 2]) / DEG

    twicetheta = np.arccos(uflab[:, 1]) / DEG

    if verbose:
        print("chi_JSM", chi)
        #         print "chi_XMAS", chiXMAS
        #         print "chi_XMAS2", chiXMAS2
        print("2theta", twicetheta)

    if returnAngles != 1:
        return uflab, IMlab
    else:  # default return
        return twicetheta, chi


def calc_uflab_trans(xcam, ycam, calib, returnAngles=1,
                                        verbose=0,
                                        pixelsize=165.0 / 2048,
                                        rectpix=RECTPIX):
    r"""
    compute :math:`2 \theta` and :math:`\chi` scattering angles or **uf** and **kf** vectors
    from lists of X and Y Laue spots positions
    in TRANSMISSION geometry

    :param xcam: list of pixel X position
    :type xcam: list of floats
    :param ycam: list of pixel Y position
    :type ycam: list of floats
    :param calib: list of 5 calibration parameters

    :returns:
        - if returnAngles=1   : twicetheta, chi   *(default)*
        - if returnAngles!=1  : uflab, IMlab

    # TODO: add offset like in reflection geometry
    """
    print("transmission GEOMETRY")
    detect, xcen, ycen, xbet, xgam = np.array(calib) * 1.0
    #    print "pixelsize in calc_uflab ", pixelsize

    # TODO: this is strange beta defintion is different...
    # but it has been checked by data from Poitiers
    # cosbeta definition differs from
    # cosbeta = np.cos(PI / 2.0 - xbet * DEG)
    # sinbeta = np.sin(PI / 2.0 - xbet * DEG)
    # of top reflection geometry ...
    cosbeta = np.cos(-xbet * DEG)
    sinbeta = np.sin(-xbet * DEG)   # negative

    cosgam = np.cos(- xgam * DEG)
    singam = np.sin(- xgam * DEG)  # negative

    xcam1 = (np.array(xcam) - xcen) * pixelsize
    ycam1 = (np.array(ycam) - ycen) * pixelsize * (1.0 + rectpix)

    xca0 = cosgam * xcam1 - singam * ycam1
    yca0 = singam * xcam1 + cosgam * ycam1

    # I centre
    # O centre of origin of pixel CCD
    # M belong to CCD plane
    # IM is parallel to kf

    # IOlab = detect * array([0.0, cosbeta, sinbeta])
    xO, yO, zO = detect * np.array([0.0, cosbeta, sinbeta])

    # OMlab = array([xca0, yca0*sinbeta, -yca0*cosbeta])
    xOM = xca0
    yOM = yca0 * sinbeta
    zOM = -yca0 * cosbeta

    # IMlab = IOlab + OMlab
    xM = xO + xOM
    yM = yO + yOM
    zM = zO + zOM
    IMlab = np.array([xM, yM, zM]).T

    # norm of IM vector
    # nIMlab=sqrt(dot(IMlab,IMlab))
    nIMlab = 1.0 * np.sqrt(xM ** 2 + yM ** 2 + zM ** 2)

    # print transpose(array([xM,yM,zM])) # vector joining source and pt on CCD in abs frame
    # print nIMlab #distance source pt on CCD (mm)

    uflab = np.transpose(np.array([xM, yM, zM]) / nIMlab)
    # print "uflab",uflab

    print("transmission mode ", uflab[:, 1])

    chi = np.arctan2(-xM, zM) / DEG
    twicetheta = np.arccos(uflab[:, 1]) / DEG
    #     chiXMAS = np.arctan(uflab[:, 0] / np.sqrt(uflab[:, 1] ** 2 + uflab[:, 2] ** 2)) / DEG
    #     chiXMAS2 = np.arctan(np.sqrt(uflab[:, 0] ** 2 + uflab[:, 1] ** 2) / uflab[:, 2]) / DEG

    if verbose:
        print("chi_JSM", chi)
        print("2theta", twicetheta)

    if returnAngles != 1:
        return uflab, IMlab
    else:  # default return
        return twicetheta, chi

def calc_uflab_trans_2(xcam, ycam, calib, returnAngles=1,
                                        verbose=0,
                                        pixelsize=165.0 / 2048,
                                        rectpix=RECTPIX):
    r"""
    compute :math:`2 \theta` and :math:`\chi` scattering angles or **uf** and **kf** vectors
    from lists of X and Y Laue spots positions
    in TRANSMISSION geometry

    in LaueToolsFrame
    see calc_xycam_transmission_2

    :param xcam: list of pixel X position
    :type xcam: list of floats
    :param ycam: list of pixel Y position
    :type ycam: list of floats
    :param calib: list of 5 calibration parameters

    :returns:
        - if returnAngles=1   : twicetheta, chi   *(default)*
        - if returnAngles!=1  : uflab, IMlab

    # TODO: add offset like in reflection geometry
    """
    print("transmission GEOMETRY")
    detect, xcen, ycen, xbet, xgam = np.array(calib) * 1.0

    cosbeta = np.cos(xbet * DEG)
    sinbeta = np.sin(xbet * DEG)

    cosgam = np.cos(xgam * DEG)
    singam = np.sin(xgam * DEG)

    xcam1 = (np.array(xcam) - xcen) * pixelsize
    ycam1 = (np.array(ycam) - ycen) * pixelsize * (1.0 + rectpix)

    # coordinates (mm) along tilted by gamma of X' Y'
    # xcam1 = cosgam * xca0 + singam * yca0
    # ycam1 = -singam * xca0 + cosgam * yca0

    xca0 = cosgam * xcam1 - singam * ycam1
    yca0 = singam * xcam1 + cosgam * ycam1

    # I centre
    # O centre of origin of pixel CCD
    # M belong to CCD plane
    # IM is parallel to kf

    # for Z>0 top reflection geometry IOlab = detect * array([0.0, cosbeta, sinbeta])

    # But Here for transmission X>0
    # xca0 length  along x pixel direction (w/o gamma correction)
    # yca0 legnth  along y pixel direction  (// Z) (w/o gamma correction)
    # OMlab = array([-xca0*sinbeta, -xca0*cosbeta, yca0])
    # yca0 = OMlab[:, 2]
    # if sinbeta != 0.0:
    #     xca0 = -OMlab[:, 0] / sinbeta
    # else:
    #     xca0 = -OMlab[:, 1] / cosbeta
    # and
    # IOlab = distance_IO * np.array([cosbeta, -sinbeta,0])

    xO, yO, zO = detect * np.array([cosbeta, -sinbeta, 0])

    xOM = -xca0*sinbeta
    yOM = -xca0*cosbeta
    zOM = yca0

    # IMlab = IOlab + OMlab
    xM = xO + xOM
    yM = yO + yOM
    zM = zO + zOM
    IMlab = np.array([xM, yM, zM]).T

    # norm of IM vector
    # nIMlab=sqrt(dot(IMlab,IMlab))
    nIMlab = 1.0 * np.sqrt(xM ** 2 + yM ** 2 + zM ** 2)

    # print transpose(array([xM,yM,zM])) # vector joining source and pt on CCD in abs frame
    # print nIMlab #distance source pt on CCD (mm)

    uflab = np.transpose(np.array([xM, yM, zM]) / nIMlab)
    # print "uflab",uflab

    print("transmission mode ", uflab[:, 0])

    chi = np.arctan2(yM, zM) / DEG
    twicetheta = np.arccos(uflab[:, 0]) / DEG

    if verbose:
        print("chi_JSM", chi)
        print("2theta", twicetheta)

    if returnAngles != 1:
        return uflab, IMlab
    else:  # default return
        return twicetheta, chi

def calc_uflab_back(xcam, ycam, calib, returnAngles=1,
                                        verbose=0,
                                        pixelsize=165.0 / 2048,
                                        rectpix=RECTPIX):
    r"""
    compute :math:`2 \theta` and :math:`\chi` scattering angles or **uf** and **kf** vectors
    from lists of X and Y Laue spots positions
    in back reflection geometry

    :param xcam: list of pixel X position
    :type xcam: list of floats
    :param ycam: list of pixel Y position
    :type ycam: list of floats
    :param calib: list of 5 calibration parameters

    :returns:
        - if returnAngles=1   : twicetheta, chi   *(default)*
        - if returnAngles!=1  : uflab, IMlab

    # TODO: add offset like in reflection geometry and merge with transmission geometry
    """
    print("Back reflection GEOMETRY")
    detect, xcen, ycen, xbet, xgam = np.array(calib) * 1.0
    #    print "pixelsize in calc_uflab ", pixelsize

    cosbeta = np.cos(-xbet * DEG)
    sinbeta = np.sin(-xbet * DEG)

    cosgam = np.cos(- xgam * DEG)
    singam = np.sin(- xgam * DEG)  # negative

    xcam1 = (np.array(xcam) - xcen) * pixelsize
    ycam1 = (np.array(ycam) - ycen) * pixelsize * (1.0 + rectpix)

    xca0 = cosgam * xcam1 - singam * ycam1
    yca0 = singam * xcam1 + cosgam * ycam1

    # I centre
    # O centre of origin of pixel CCD plane array
    # M belong to CCD plane
    # IM is parallel to kf (or uf = kf/||kf||)

    # IOlab = detect * array([0.0, -cosbeta, sinbeta])
    xO, yO, zO = detect * np.array([0.0, -cosbeta, sinbeta])

    # OMlab = array([xca0, yca0*sinbeta, yca0*cosbeta])
    xOM = xca0
    yOM = yca0 * sinbeta
    zOM = yca0 * cosbeta

    # IMlab = IOlab + OMlab
    xM = xO + xOM
    yM = yO + yOM
    zM = zO + zOM
    IMlab = np.array([xM, yM, zM]).T

    # norm of IM vector
    # nIMlab=sqrt(dot(IMlab,IMlab))
    nIMlab = 1.0 * np.sqrt(xM ** 2 + yM ** 2 + zM ** 2)

    # print transpose(array([xM,yM,zM])) # vector joining source and pt on CCD in abs frame
    # print nIMlab #distance source pt on CCD (mm)

    uflab = np.transpose(np.array([xM, yM, zM]) / nIMlab)
    # print "uflab",uflab
    EPS = 1e-17

    # print("back reflection mode ", uflab)

    chi = np.arctan2(-xM, zM + EPS) / DEG
    twicetheta = 180-np.arccos(-uflab[:, 1]) / DEG
    #     chiXMAS = np.arctan(uflab[:, 0] / np.sqrt(uflab[:, 1] ** 2 + uflab[:, 2] ** 2)) / DEG
    #     chiXMAS2 = np.arctan(np.sqrt(uflab[:, 0] ** 2 + uflab[:, 1] ** 2) / uflab[:, 2]) / DEG

    if verbose:
        print("chi_JSM", chi)
        print("2theta", twicetheta)

    if returnAngles != 1:
        return uflab, IMlab
    else:  # default return
        return twicetheta, chi

def OM_from_uf(uflab, calib, energy=0, offset=None, verbose=0):
    r"""
    2D vector position of point OM in detector frame plane in pixels
    alias function to calc_xycam
    """
    return calc_xycam(uflab, calib, energy=energy, offset=offset, verbose=verbose)


def IprimeM_from_uf(uflab, posI, calib, verbose=0):
    r"""
    from:
    uflab
    posI= IIprime = position (3elemts vector) of source with respect to I (calibrated emission source) in millimeter

    returns:
    IprimeM vector joining shifted source emission to point M lying on CCD
    """

    return calc_xycam(uflab, calib, energy=0, offset=posI, verbose=verbose, returnIpM=True)


def calc_xycam(uflab, calib, energy=0, offset=None, verbose=0, returnIpM=False,
                                                                        pixelsize=165.0 / 2048.,
                                                                        rectpix=RECTPIX):
    r"""
    Computes Laue spots position x and y in pixels units in CCD frame
    from unit scattered vector uf expressed in Lab. frame

    computes coordinates of point M on CCD from point source and **uflab**.
    Point Ip (source Iprime of x-ray scattered beams)
    (for each Laue spot **uflab** is the unit vector of **IpM**)
    Point Ip is shifted by offset (if not None) from the default point I
    (used to calibrate the CCD camera and 2theta chi determination)

    th0 (theta in degrees)
    Energy (energy in keV)

    :param uflab: list or array of [kf_x,kf_y,kf_z] (kf or uf unit vector)
    :type uflab: list or array (length must > 1)

    :param calib: list 5 detector calibration parameters
    :type calib: list of floats

    :param offset: offset (in mm) in the scattering source (origin of Laue spots)
            position with respect to the position which has been used
            for the calibration of  the CCD detector plane. Offset is positive when in the same
            direction as incident beam (i.e. in sample depth)
            (incident beam direction remains constant)
    :type offset: list of floats ([x,y,z])

    :returns:
        - xcam: list of pixel X coordinates
        - ycam: list of pixel Y coordinates
        - theta: list half scattering angle "theta" (in degree)

        - optionally energy=1: add in output list of spot energies (in keV)

        - if returnIpM and offset not None: return list of vectors **IprimeM**
    """
    detect, xcen, ycen, xbet, xgam = np.array(calib) * 1.0

    # beta = PI/2 - xbet*DEG
    # xbet angle between IO and z axis
    # beta angle between y and IO
    # cosbeta= sin xbet
    # sinbeta = cos xbet

    cosbeta = np.cos(PI / 2.0 - xbet * DEG)
    sinbeta = np.sin(PI / 2.0 - xbet * DEG)

    # IOlab: vector joining O nearest point of CCD plane and I (origin of lab frame and emission source)
    IOlab = detect * np.array([0.0, cosbeta, sinbeta])

    # unitary normal vector of CCD plane
    # joining O nearest point of CCD plane and I (origin of lab frame and emission source)
    unlab = IOlab / np.sqrt(np.dot(IOlab, IOlab))

    # normalization of all input uflab
    norme_uflab = np.sqrt(np.sum(uflab ** 2, axis=1))
    uflab = uflab / np.reshape(norme_uflab, (len(norme_uflab), 1))

    # un is orthogonal to any vector joining O and a point M lying in the CCD frame plane
    scal = np.dot(uflab, unlab)
    normeIMlab = detect / scal

    # IMlab = normeIMlab*uflab
    IMlab = uflab * np.reshape(normeIMlab, (len(normeIMlab), 1))

    OMlab = IMlab - IOlab

    if offset not in (None, 0, 0.0):  # offset input in millimeter
        # OO'=II'-(II'.un)un  # 1 vector
        # dd'=  dd - II'.un # scalar
        # I'M= dd'/(uf.un) uf # n vector
        # I'O'= dd' un # 1 vectorin
        # O'M=I'M - I'O' # n vector
        # OM = OO' + O'M # n vector
        IIprime = offset * np.array([1, 0, 0])
        IIprime_un = np.dot(IIprime, unlab)
        OOprime = IIprime - IIprime_un * unlab
        ddprime = detect + IIprime_un
        IprimeM_norm = ddprime / scal
        IprimeM = uflab * np.reshape(IprimeM_norm, (len(uflab), 1))
        IprimeOprime = ddprime * unlab
        OMlab = OOprime + IprimeM - IprimeOprime

        if verbose:
            print("IIprime", IIprime)
            print("IIprime_un", IIprime_un)
            print("OOprime", OOprime)
            print("IprimeM_norm", IprimeM_norm)
            print("IprimeM", IprimeM)
            print("dd", detect)
            print("dd'", ddprime)
            print("OM", OMlab)

        if returnIpM:
            return IprimeM

    # OMlab = array([xca0, yca0*sinbeta, -yca0*cosbeta])
    xca0 = OMlab[:, 0]
    if sinbeta != 0.0:
        yca0 = OMlab[:, 1] / sinbeta
    else:
        yca0 = -OMlab[:, 2] / cosbeta
    # zca0 = 0

    cosgam = np.cos(- xgam * DEG)
    singam = np.sin(- xgam * DEG)

    xcam1 = cosgam * xca0 + singam * yca0
    ycam1 = -singam * xca0 + cosgam * yca0

    xcam = xcen + xcam1 / pixelsize
    ycam = ycen + ycam1 / (pixelsize * (1.0 + rectpix))

    twicetheta = (1. / DEG) * np.arccos(uflab[:, 1])
    th0 = twicetheta / 2.0

    # q = kf - ki
    qlab = uflab - np.array([0.0, 1.0, 0.0])
    norme_qlab = np.sqrt(np.sum(qlab ** 2, axis=1))

    Energy = CST_CONV_LAMBDA_KEV * norme_qlab ** 2 / (2.0 * np.sin(th0 * DEG))

    if energy:
        return xcam, ycam, th0, Energy
    else:
        return xcam, ycam, th0


def calc_xycam_backreflection(uflab, calib, energy=0, offset=None, verbose=0, returnIpM=False,
                                                                            pixelsize=165.0 / 2048,
                                                                            rectpix=RECTPIX):
    r"""
    Computes Laue spots position x and y in pixels units (in CCD frame) from scattered vector kf or uf

    As calc_xycam() but in BACK REFLECTION geometry


     cosbeta = np.cos(-xbet * DEG)
    sinbeta = np.sin(-xbet * DEG)

    cosgam = np.cos(- xgam * DEG)
    singam = np.sin(- xgam * DEG)  # negative

    xcam1 = (np.array(xcam) - xcen) * pixelsize
    ycam1 = (np.array(ycam) - ycen) * pixelsize * (1.0 + rectpix)

    xca0 = cosgam * xcam1 - singam * ycam1
    yca0 = singam * xcam1 + cosgam * ycam1

    # I centre
    # O centre of origin of pixel CCD plane array
    # M belong to CCD plane
    # IM is parallel to kf (or uf = kf/||kf||)

    # IOlab = detect * array([0.0, -cosbeta, sinbeta])
    xO, yO, zO = detect * np.array([0.0, -cosbeta, sinbeta])

    # OMlab = array([xca0, yca0*sinbeta, yca0*cosbeta])
    xOM = xca0
    yOM = yca0 * sinbeta
    zOM = yca0 * cosbeta

    # IMlab = IOlab + OMlab
    xM = xO + xOM
    yM = yO + yOM
    zM = zO + zOM
    IMlab = np.array([xM, yM, zM]).T

    # norm of IM vector
    # nIMlab=sqrt(dot(IMlab,IMlab))
    nIMlab = 1.0 * np.sqrt(xM ** 2 + yM ** 2 + zM ** 2)

    # print transpose(array([xM,yM,zM])) # vector joining source and pt on CCD in abs frame
    # print nIMlab #distance source pt on CCD (mm)

    uflab = np.transpose(np.array([xM, yM, zM]) / nIMlab)
    # print "uflab",uflab
    EPS = 1e-17

    print("back reflection mode ", uflab)

    chi = np.arctan2(-xM, zM + EPS) / DEG
    twicetheta = 180-np.arccos(-uflab[:, 1]) / DEG
    #     chiXMAS = np.arctan(uflab[:, 0] / np.sqrt(uflab[:, 1] ** 2 + uflab[:, 2] ** 2)) / DEG
    #     chiXMAS2 = np.arctan(np.sqrt(uflab[:, 0] ** 2 + uflab[:, 1] ** 2) / uflab[:, 2]) / DEG

    if verbose:
        print("chi_JSM", chi)
        print("2theta", twicetheta)

    if returnAngles != 1:
        return uflab, IMlab
    else:  # default return
        return twicetheta, chi
    """
    distance_IO, xcen, ycen, xbet, xgam = np.array(calib) * 1.0

    # beta = PI/2 - xbet*DEG
    # xbet angle between IO and z axis
    # beta angle between y and IO
    # cosbeta= sin xbet
    # sinbeta = cos xbet

    #     cosbeta = np.cos(PI / 2. - xbet * DEG)
    #     sinbeta = np.sin(PI / 2. - xbet * DEG)

    cosbeta = np.cos(-xbet * DEG)
    sinbeta = np.sin(-xbet * DEG)
    # if xbet positive bottom part of CCD is closer to sample than top part

    #    print "cosbeta", cosbeta
    #    print "sinbeta", sinbeta

    # IOlab: vector joining O nearest point of CCD plane and I (origin of lab frame and emission source)

    IOlab = distance_IO * np.array([0.0, -cosbeta, sinbeta])

    # sinbeta negative for xbet positive

    #    print "IOlab", IOlab

    # unitary normal vector of CCD plane
    # joining O nearest point of CCD plane and I (origin of lab frame and emission source)
    unlab = IOlab / np.sqrt(np.dot(IOlab, IOlab))

    # normalization of all input uflab
    norme_uflab = np.sqrt(np.sum(uflab ** 2, axis=1))
    uflab = uflab / np.reshape(norme_uflab, (len(norme_uflab), 1))

    # un is orthogonal to any vector joining O and a point M lying in the CCD frame plane
    scal = np.dot(uflab, unlab)
    normeIMlab = distance_IO / scal

    # IMlab = normeIMlab*uflab
    IMlab = uflab * np.reshape(normeIMlab, (len(normeIMlab), 1))

    OMlab = IMlab - IOlab

    #    print "OMlab", OMlab

    if offset not in (None, 0, 0.0):  # offset input in millimeter
        # OO'=II'-(II'.un)un  # 1 vector
        # dd'=  dd - II'.un # scalar
        # I'M= dd'/(uf.un) uf # n vector
        # I'O'= dd' un # 1 vector
        # O'M=I'M - I'O' # n vector
        # OM = OO' + O'M # n vector
        IIprime = offset
        IIprime_un = np.dot(IIprime, unlab)
        OOprime = IIprime - IIprime_un * unlab
        ddprime = distance_IO + IIprime_un
        IprimeM_norm = ddprime / scal
        IprimeM = uflab * np.reshape(IprimeM_norm, (len(uflab), 1))
        IprimeOprime = ddprime * unlab
        OMlab = OOprime + IprimeM - IprimeOprime

        if verbose:
            print("IIprime", IIprime)
            print("IIprime_un", IIprime_un)
            print("OOprime", OOprime)
            print("IprimeM_norm", IprimeM_norm)
            print("IprimeM", IprimeM)
            print("dd", distance_IO)
            print("dd'", ddprime)
            print("OM", OMlab)

        if returnIpM:
            return IprimeM

    # OMlab = array([xca0, yca0*sinbeta, yca0*cosbeta])
    xca0 = OMlab[:, 0]
    if sinbeta != 0.0:
        yca0 = OMlab[:, 1] / sinbeta
    else:
        yca0 = OMlab[:, 2] / cosbeta
    # zca0 = 0

    cosgam = np.cos(-xgam * DEG)
    singam = np.sin(-xgam * DEG)

    xcam1 = cosgam * xca0 + singam * yca0
    ycam1 = -singam * xca0 + cosgam * yca0

    #    print "xcam1", xcam1
    #    print "ycam1", ycam1

    xcam = xcen + xcam1 / pixelsize
    ycam = ycen + ycam1 / (pixelsize * (1.0 + rectpix))

    twicetheta = (1 / DEG) * np.arccos(uflab[:, 1])
    th0 = twicetheta / 2.0

    # q = kf - ki
    qf = uflab - np.array([0.0, 1.0, 0.0])
    norme_qflab = np.sqrt(np.sum(qf ** 2, axis=1))

    Energy = CST_CONV_LAMBDA_KEV * norme_qflab ** 2 / (2.0 * np.sin(th0 * DEG))

    if energy:
        return xcam, ycam, th0, Energy
    else:
        return xcam, ycam, th0

def calc_xycam_transmission(uflab, calib, energy=0, offset=None, verbose=0, returnIpM=False,
                                                                            pixelsize=165.0 / 2048,
                                                                            rectpix=RECTPIX):
    r"""
    Computes Laue spots position x and y in pixels units (in CCD frame) from scattered vector uf or kf
    As calc_xycam() but in TRANSMISSION geometry
    """

    distance_IO, xcen, ycen, xbet, xgam = np.array(calib) * 1.0

    # beta = PI/2 - xbet*DEG
    # xbet angle between IO and z axis
    # beta angle between y and IO
    # cosbeta= sin xbet
    # sinbeta = cos xbet

    #     cosbeta = np.cos(PI / 2. - xbet * DEG)
    #     sinbeta = np.sin(PI / 2. - xbet * DEG)

    cosbeta = np.cos(-xbet * DEG)
    sinbeta = np.sin(-xbet * DEG)
    # if xbet positive bottom part of CCD is closer to sample than top part

    #    print "cosbeta", cosbeta
    #    print "sinbeta", sinbeta

    # IOlab: vector joining O nearest point of CCD plane and I (origin of lab frame and emission source)

    IOlab = distance_IO * np.array([0.0, cosbeta, sinbeta])

    # sinbeta negative for xbet positive

    #    print "IOlab", IOlab

    # unitary normal vector of CCD plane
    # joining O nearest point of CCD plane and I (origin of lab frame and emission source)
    unlab = IOlab / np.sqrt(np.dot(IOlab, IOlab))

    # normalization of all input uflab
    norme_uflab = np.sqrt(np.sum(uflab ** 2, axis=1))
    uflab = uflab / np.reshape(norme_uflab, (len(norme_uflab), 1))

    # un is orthogonal to any vector joining O and a point M lying in the CCD frame plane
    scal = np.dot(uflab, unlab)
    normeIMlab = distance_IO / scal

    # IMlab = normeIMlab*uflab
    IMlab = uflab * np.reshape(normeIMlab, (len(normeIMlab), 1))

    OMlab = IMlab - IOlab

    #    print "OMlab", OMlab

    if offset not in (None, 0, 0.0):  # offset input in millimeter
        # OO'=II'-(II'.un)un  # 1 vector
        # dd'=  dd - II'.un # scalar
        # I'M= dd'/(uf.un) uf # n vector
        # I'O'= dd' un # 1 vector
        # O'M=I'M - I'O' # n vector
        # OM = OO' + O'M # n vector
        IIprime = offset
        IIprime_un = np.dot(IIprime, unlab)
        OOprime = IIprime - IIprime_un * unlab
        ddprime = distance_IO + IIprime_un
        IprimeM_norm = ddprime / scal
        IprimeM = uflab * np.reshape(IprimeM_norm, (len(uflab), 1))
        IprimeOprime = ddprime * unlab
        OMlab = OOprime + IprimeM - IprimeOprime

        if verbose:
            print("IIprime", IIprime)
            print("IIprime_un", IIprime_un)
            print("OOprime", OOprime)
            print("IprimeM_norm", IprimeM_norm)
            print("IprimeM", IprimeM)
            print("dd", distance_IO)
            print("dd'", ddprime)
            print("OM", OMlab)

        if returnIpM:
            return IprimeM

    # OMlab = array([xca0, yca0*sinbeta, -yca0*cosbeta])
    xca0 = OMlab[:, 0]
    if sinbeta != 0.0:
        yca0 = OMlab[:, 1] / sinbeta
    else:
        yca0 = -OMlab[:, 2] / cosbeta
    # zca0 = 0

    cosgam = np.cos(-xgam * DEG)
    singam = np.sin(-xgam * DEG)

    xcam1 = cosgam * xca0 + singam * yca0
    ycam1 = -singam * xca0 + cosgam * yca0

    #    print "xcam1", xcam1
    #    print "ycam1", ycam1

    xcam = xcen + xcam1 / pixelsize
    ycam = ycen + ycam1 / (pixelsize * (1.0 + rectpix))

    twicetheta = (1 / DEG) * np.arccos(uflab[:, 1])
    th0 = twicetheta / 2.0

    # q = kf - ki
    qf = uflab - np.array([0.0, 1.0, 0.0])
    norme_qflab = np.sqrt(np.sum(qf ** 2, axis=1))

    Energy = CST_CONV_LAMBDA_KEV * norme_qflab ** 2 / (2.0 * np.sin(th0 * DEG))

    if energy:
        return xcam, ycam, th0, Energy
    else:
        return xcam, ycam, th0

def calc_xycam_transmission_2(uflabframe0, calib, energy=0, offset=None, verbose=0, returnIpM=False,
                                                                            pixelsize=165.0 / 2048,
                                                                            rectpix=RECTPIX,
                                                                            convert2LTframe=True):
    r"""
    Computes Laue spots position x and y in pixels units (in CCD frame) from scattered vector uf or kf
    As calc_xycam() but in TRANSMISSION geometry

    X // ki incoming beam
    Z vertical
    Y = Z ^ X (pointing towards the door, or at the left hand side when riding incoming x-ray)

    without gamma correction, X' // pixel Xcam and Y' // pixel Ycam
    X' tilted by xbet from -Y
    Y' = Z

    WARNING: uflabframe0 must be converted in LaueTools frame in this function, if uflab components in this module frame  (Y // ki , X towards the wall, Z vertical)...
    """
    if convert2LTframe:
        ux, uy, uz = uflabframe0.T
        uflab = np.array([uy, -ux, uz]).T

    distance_IO, xcen, ycen, xbet, xgam = np.array(calib) * 1.0

    cosbeta = np.cos(xbet * DEG)
    sinbeta = np.sin(xbet * DEG)

    # IOlab: vector joining O nearest point of CCD plane and I (origin of lab frame and emission source)

    IOlab = distance_IO * np.array([cosbeta, -sinbeta, 0])

    # unitary normal vector of CCD plane
    # joining O nearest point of CCD plane and I (origin of lab frame and emission source)
    unlab = IOlab / np.sqrt(np.dot(IOlab, IOlab))

    # normalization of all input uflab
    norme_uflab = np.sqrt(np.sum(uflab ** 2, axis=1))
    uflab = uflab / np.reshape(norme_uflab, (len(norme_uflab), 1))

    # un is orthogonal to any vector joining O and a point M lying in the CCD frame plane
    scal = np.dot(uflab, unlab)
    normeIMlab = distance_IO / scal

    # IMlab = normeIMlab*uflab
    IMlab = uflab * np.reshape(normeIMlab, (len(normeIMlab), 1))

    OMlab = IMlab - IOlab

    #    print "OMlab", OMlab

    # to check
    # if offset not in (None, 0, 0.0):  # offset input in millimeter
    #     # OO'=II'-(II'.un)un  # 1 vector
    #     # dd'=  dd - II'.un # scalar
    #     # I'M= dd'/(uf.un) uf # n vector
    #     # I'O'= dd' un # 1 vector
    #     # O'M=I'M - I'O' # n vector
    #     # OM = OO' + O'M # n vector
    #     IIprime = offset
    #     IIprime_un = np.dot(IIprime, unlab)
    #     OOprime = IIprime - IIprime_un * unlab
    #     ddprime = distance_IO + IIprime_un
    #     IprimeM_norm = ddprime / scal
    #     IprimeM = uflab * np.reshape(IprimeM_norm, (len(uflab), 1))
    #     IprimeOprime = ddprime * unlab
    #     OMlab = OOprime + IprimeM - IprimeOprime

    #     if verbose:
    #         print("IIprime", IIprime)
    #         print("IIprime_un", IIprime_un)
    #         print("OOprime", OOprime)
    #         print("IprimeM_norm", IprimeM_norm)
    #         print("IprimeM", IprimeM)
    #         print("dd", distance_IO)
    #         print("dd'", ddprime)
    #         print("OM", OMlab)

    #     if returnIpM:
    #         return IprimeM

    # for Z>0 top reflection geometry :
    # OMlab = array([xca0, yca0*sinbeta, -yca0*cosbeta])

    # Here for transmission X>0
    # xca0 length  along x pixel direction (w/o gamma correction)
    # yca0 legnth  along y pixel direction  (// Z) (w/o gamma correction)
    # OMlab = array([-xca0*sinbeta, -xca0*cosbeta, yca0])
    yca0 = OMlab[:, 2]
    if sinbeta != 0.0:
        xca0 = -OMlab[:, 0] / sinbeta
    else:
        xca0 = -OMlab[:, 1] / cosbeta
    # zca0 = 0  (along dir normal to detector)

    cosgam = np.cos(xgam * DEG)
    singam = np.sin(xgam * DEG)

    # coordinates (mm) along tilted by gamma of X' Y'
    xcam1 = cosgam * xca0 + singam * yca0
    ycam1 = -singam * xca0 + cosgam * yca0

    # same coordinates but in pixel units
    # taking into account the point of normal incidence (xcen,ycen) in pixels unit
    xcam = xcen + xcam1 / pixelsize
    ycam = ycen + ycam1 / (pixelsize * (1.0 + rectpix))

    twicetheta = (1 / DEG) * np.arccos(uflab[:, 0])
    th0 = twicetheta / 2.0

    # q = kf - ki    ki // X
    qf = uflab - np.array([1.0, 0.0, 0.0])
    norme_qflab = np.sqrt(np.sum(qf ** 2, axis=1))

    Energy = CST_CONV_LAMBDA_KEV * norme_qflab ** 2 / (2.0 * np.sin(th0 * DEG))

    if energy:
        return xcam, ycam, th0, Energy
    else:
        return xcam, ycam, th0



def calc_xycam_from2thetachi(twicetheta, chi, calib, offset=0, verbose=0,
                                                        pixelsize=165.0 / 2048,
                                                        kf_direction="Z>0", version=1):
    r"""
    calculate spots coordinates in pixel units in detector plane
    from 2theta, chi angles (kf)

    :param offset: offset (in mm) in the scattering source (origin of Laue spots)
        position with respect to the position which has been used
        for the calibration of  the CCD detector plane. Offset is positive when in the same
        direction as incident beam (i.e. in sample depth)
        (incident beam direction remains constant)
    :type offset: list of floats ([x,y,z])
    """
    # scattered vector not in Lauetools frame (y//ki)
    uflab = uflab_from2thetachi(twicetheta, chi, verbose=0)

    if verbose:
        print("uflab", uflab)

    if kf_direction in ("Z>0",):  # , '[90.0, 45.0]'):
        return calc_xycam(uflab, calib, offset=offset, pixelsize=pixelsize)
    elif kf_direction in ("Y>0", "Y<0"):
        print("CAUTION: not checked yet")
        # TODO raise ValueError, print "not checked yet"
        return calc_xycam(uflab, calib, offset=offset, pixelsize=pixelsize)
    elif kf_direction in ("X>0",):  # transmission
        if version == 1:
            return calc_xycam_transmission(uflab, calib, offset=offset, pixelsize=pixelsize)
        elif version == 2:
            return calc_xycam_transmission_2(uflab, calib, offset=offset, pixelsize=pixelsize)
    elif kf_direction in ("X<0",):  # back-reflection
        # patch JSM March 2020
        return calc_xycam_backreflection(uflab, calib, offset=offset, pixelsize=pixelsize)
    else:
        sentence = "kf_direction = %s is not implemented yet " % kf_direction
        sentence += "in calc_xycam_from2thetachi() in LaueGeometry  (new find2thetachi)"
        raise ValueError(sentence)


def uflab_from2thetachi(twicetheta, chi, verbose=0):
    r"""
    Computes :math:`{\bf u_f}` vectors coordinates in lauetools LT2 frame
    from :math:`{\bf k_f}` scattering angles :math:`2 \theta` and :math:`2 \chi` angles

    :param twicetheta: (list) :math:`2 \theta` angle(s) ( in degree)
    :param chi: (list) :math:`2 \chi` angle(s) ( in degree)

    :returns: list of `{\bf u_f}` =  [:math:`uf_x,uf_y,uf_z`]
    :rtype: list
    """

    ctw = np.cos(np.array(twicetheta) * DEG)
    stw = np.sin(np.array(twicetheta) * DEG)
    cchi = np.cos(np.array(chi) * DEG)
    schi = np.sin(np.array(chi) * DEG)

    xuflab = -stw * schi
    yuflab = ctw
    zuflab = stw * cchi

    uflab = np.array([xuflab, yuflab, zuflab]).T

    if verbose:
        print("uflab", uflab)

    return uflab


def q_unit_XYZ(twicetheta, chi):
    r"""
    Computes unit vector of :math:`{\bf q}` (scattering transfer moment) :math:`{\bf u_q}`
    from scattered :math:`{\bf k_f}` angles
    # TODO: useful ? check with from_twchi_to_qunit()
    lauetools frame
    #in degrees
    """
    THETA = twicetheta / 2.0 * DEG
    CHI = chi * DEG
    return np.array([-np.sin(THETA),
                    np.cos(THETA) * np.sin(CHI),
                    np.cos(THETA) * np.cos(CHI)])


def q_unit_2thetachi(vec):
    r"""
    Computes 2theta and chi scattering angles from a u_q vector expressed in LaueTools frame
    #result in deg
    # TODO: useful ? check with from_qunit_to_twchi()
    lauetools frame ?
    """
    X, Y, Z = tuple(vec)
    # TODO: sign of chi must be checked
    chi = np.arctan2(Y, Z) / DEG
    theta = -np.arcsin(X) / DEG
    return np.array([2.0 * theta, chi])


def from_twchi_to_qunit(Angles):
    r"""
    from kf 2theta, chi to q unit in LaueTools frame (xx// ki) q=kf-ki
    returns array = (all x's, all y's, all z's)

    Angles in degrees !!
    Angles[0] 2theta deg values,
    Angles[1] chi values in deg

    this is the inverse function of from_qunit_to_twchi(), useful to check it
    """

    twthe = np.array(Angles[0]) * DEG
    chi = np.array(Angles[1]) * DEG
    no = 2.0 * np.sin(twthe / 2.0)
    qx = np.cos(twthe) - 1
    qy = np.sin(twthe) * np.sin(chi)
    qz = np.sin(twthe) * np.cos(chi)
    return np.array([qx, qy, qz]) / no


def from_twchi_to_q(Angles):
    r"""
    From kf 2theta,chi to q (arbitrary lenght) in lab frame (xx// ki) q=kf-ki
    returns array = (all qx's, all qy's, all qz's)

    Angles in degrees !!
    Angles[0] 2theta deg values,
    Angles[1] chi values in deg
    """

    twthe = np.array(Angles[0]) * DEG
    chi = np.array(Angles[1]) * DEG
    qx = np.cos(twthe) - 1
    qy = np.sin(twthe) * np.sin(chi)
    qz = np.sin(twthe) * np.cos(chi)
    return np.array([qx, qy, qz])


def from_qunit_to_twchi(arrayXYZ, labXMAS=0):
    r"""
    Returns 2theta chi from a q unit vector (defining a direction) expressed in LaueTools frame (**xx**// **ki**) **q=kf-ki**

    .. math:: \left [ \begin{matrix}
        -\sin \theta \\ \cos \theta \sin \chi \\ \cos \theta \cos \chi
        \end{matrix}
        \right ]

    .. note::
        in LaueTools frame

        .. math::
            kf = \left [ \begin{matrix}
            \cos 2\theta \\ \sin 2\theta \sin \chi \\ \sin 2\theta \cos \chi
            \end{matrix}
            \right ]

            q = 2 \sin \theta \left [ \begin{matrix}
            -\sin \theta \\ \cos \theta \sin \chi \\ \cos \theta \cos \chi
            \end{matrix}
            \right ]

        In LT2 Frame   labXMAS=1

        .. math::
            kf = \left [ \begin{matrix}
            \sin 2\theta \sin \chi \\ \cos 2\theta \\ \sin 2\theta \cos \chi
            \end{matrix}
            \right ]

            q = 2 \sin \theta \left [ \begin{matrix}
            \cos \theta \sin \chi \\ -\sin \theta \\ \cos \theta \cos \chi
            \end{matrix}
            \right ]
    """
    X, Y, Z = arrayXYZ

    if labXMAS:
        chi = np.arctan2(-X * 1.0, Z)
        twthe = 2 * np.arcsin(-Y * 1.0)
    else:  # labXMAS=0  lauetools
        chi = np.arctan2(Y * 1.0, Z)
        twthe = 2 * np.arcsin(-X * 1.0)

    return np.array([twthe, chi]) / DEG


def qvector_from_xy_E(xcamList, ycamList, energy, detectorplaneparameters, pixelsize):
    r"""
    Returns q vectors in Lauetools frame given x and y pixel positions on detector
    for a given Energy (keV)

    :param xcamList: list pixel x postions
    :param ycamList: list pixel y postions
    :param energy: list pf energies
    :param detectorplaneparameters: list of 5 calibration parameters
    :param pixelsize: pixel size in mm
    """
    # in LT's frame (x// ki)

    #     print "xcamList",xcamList
    #     print "ycamList",ycamList
    twtheta, chi = calc_uflab(xcamList, ycamList, detectorplaneparameters, returnAngles=1,
                                                                        verbose=0,
                                                                        pixelsize=pixelsize,
                                                                        rectpix=RECTPIX,
                                                                        kf_direction="Z>0")

    thetarad = twtheta * DEG / 2.0
    chirad = chi * DEG

    qx = -np.sin(thetarad)
    qy = np.cos(thetarad) * np.sin(chirad)
    qz = np.cos(thetarad) * np.cos(chirad)

    newq = np.array([qx, qy, qz])
    normnewq = np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)

    qvec = newq * (1.0 / normnewq)
    #     print "qvec",qvec

    #     print "energy",energy

    qvector_Lauetoolsframe = qvec * np.sin(thetarad) * (2 * energy / 12.398)

    #     print "qvector_Lauetoolsframe",qvector_Lauetoolsframe

    return qvector_Lauetoolsframe


def unit_q(ttheta, chi, frame="lauetools", anglesample=40.0):
    r"""
    Returns unit q vector from 2theta,chi coordinates

    :param ttheta: list of 2theta angles (in degrees)
    :param chi: list of chi angles (in degrees)
    :param anglesample: incidence angle of beam to surface plane (degrees)
    :param frame: frame to express vectors in: 'lauetools' , 'XMASlab' (LT2 frame),'XMASsample'

    :returns: list of 3D u_f (unit vector of scattering transfer q)
    """
    thetarad = ttheta * DEG / 2.0
    chirad = chi * DEG

    if frame == "lauetools":
        qx = -np.sin(thetarad)
        qy = np.cos(thetarad) * np.sin(chirad)
        qz = np.cos(thetarad) * np.cos(chirad)

        newq = np.array([qx, qy, qz])
        normnewq = np.sqrt(np.dot(newq, newq))

        return newq / normnewq
    # LT2 frame
    elif frame == "XMASlab":
        # chi convention:
        # y along Xray horizontal
        # x towards wall behind horizontal
        # z vertical up

        # kf=( - sin 2theta sin chi, cos 2theta  , sin 2theta cos chi) XMAS convention
        # ki=( 0, 1  , 0) XMAS convention
        # q = 2 sin theta (- costheta sinchi , - sintheta  ,  costheta coschi)
        #  unitkf  =  unitki  +   2sintheta unitq

        qx = np.cos(thetarad) * np.sin(chirad)
        qy = -np.sin(thetarad)
        qz = np.cos(thetarad) * np.cos(chirad)

        newq = np.array([qx, qy, qz])
        normnewq = np.sqrt(np.dot(newq, newq))

        return newq / normnewq
    # LT2 sample frame
    elif frame == "XMASsample":

        # kf=( - sin 2theta sin chi, cos 2theta  , sin 2theta cos chi) XMAS convention
        # ki=( 0, 1  , 0) XMAS convention
        # q = 2 sin theta (- costheta sinchi , - sintheta  ,  costheta coschi)
        #  unitkf  =  unitki  +   2sintheta unitq

        angrad = anglesample * np.pi / 180.0  # Must include -xbet/2 correction ???
        ca = np.cos(angrad)
        sa = np.sin(angrad)

        matrot = np.array([[1, 0, 0.0], [0.0, ca, sa], [0, -sa, ca]])

        qx = np.cos(thetarad) * np.sin(chirad)
        qy = -np.sin(thetarad)
        qz = np.cos(thetarad) * np.cos(chirad)

        newq = np.dot(matrot, np.array([qx, qy, qz]))
        normnewq = np.sqrt(np.dot(newq, newq))

        return newq / normnewq


def plotXY2thetachi(datX, datY, dat2the, datchi, mostintense=None):
    r"""
    old script to combine plot of pixel x,y and 2theta chi plot
    """
    if mostintense is not None and mostintense < len(datX):
        data_x = datX[:mostintense]
        data_y = datY[:mostintense]
        twicetheta = dat2the[:mostintense]
        chi = datchi[:mostintense]
    else:
        data_x = datX
        data_y = datY
        twicetheta = dat2the
        chi = datchi

    plot1 = P.subplot(121)
    plot1.set_aspect(aspect="equal")
    P.xlabel("X")
    P.ylabel("Y")
    plot1.scatter(tuple(data_x), tuple(data_y))

    plot2 = P.subplot(122)
    plot2.set_aspect(aspect=0.5)
    P.xlabel("chi")
    P.ylabel("2theta")
    plot2.scatter(tuple(chi), tuple(twicetheta), c="r", marker="d")

    P.show()


# ---------------    Frame Matrix conversion
def matxmas_to_OrientMatrix(satocr, calib):
    r"""
    thanks to Odile robach's reverse engineering hard work

    From XMAS matrices in IND file to matrix in lauetools frame
    convert matrix from XMAS sample axes to lab axes + normalize by astar

    - calib with last angles in degrees

    - satocrs = transposee de la matrice numero 2 du .STR :   matrice UB
    "coordinates of a*, b*, c* in X, Y, Z"
    - satocru = matrice numero 2 du .IND :   matrice U
    "matrix hkl => XYZ"
    """
    astar = np.sqrt(np.sum(satocr[:, 0] ** 2))
    # print "satocr \n", satocr
    satocrnorm = satocr / astar  # sample to crystal: satocr
    # print "satocrnorm \n", satocrnorm

    omega0 = 40.0  # deg
    xbet = calib[3] * DEG  # rad
    # print "xbet in rad" , xbet
    omega = omega0 * DEG - xbet / 2.0
    # print "omega" , omega*180.0/np.pi

    # rotation de omega autour de l'axe x pour repasser dans Rlab
    matrot = np.array([[1.0, 0.0, 0.0],
            [0.0, np.cos(omega), np.sin(omega)],
            [0.0, -np.sin(omega), np.cos(omega)]])
    # print "matrot \n" , matrot

    labtocr = np.dot(matrot, satocrnorm)
    astarlab = labtocr[:, 0]
    bstarlab = labtocr[:, 1]
    cstarlab = labtocr[:, 2]

    matstarlab1 = np.hstack((astarlab, bstarlab, cstarlab))
    changesign = np.array([-1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0])

    matstarlab2 = np.multiply(matstarlab1, changesign)
    # print "matstarlab from satocr \n", matstarlab2

    # mm=matstarlab1
    mm = matstarlab2

    # matstarlabLaueTools= array([[mm[1],-mm[4],-mm[7]],[mm[0],-mm[6],-mm[3]],[-mm[2],mm[5],mm[8]]])
    matstarlabLaueTools = np.array([[mm[1], mm[4], mm[7]],
                                    [-mm[0], -mm[3], -mm[6]],
                                    [mm[2], mm[5], mm[8]]])

    # resulting matrix contains orientation + strain

    return matstarlabLaueTools


def matstarlabLaueTools_to_matstarlabOR(UBmat):
    r"""
    convert matrix from lauetools frame: ki//x, z towards CCD (top), y = z^x
                    to ORobach or XMAS's frame: ki//y, z towards CCD (top), y = z^x

    convert the so called UBmat to matstarlab
    (matstarlab stands for 'matrix of reciprocal unit cell basis vectors in lab. frame')

    see the reciprocal function: matstarlabOR_to_matstarlabLaueTools
    """
    mm = UBmat

    # print "check matrix before normalizing"
    # print "normes astar1 bstar1 cstar1 = ", GT.norme_vec(astar1),GT.norme_vec(astar1),GT.norme_vec(astar1)
    # print "inner products astar1.bstar1, bstar1.cstar1, cstar1.astar1 \n",\
    #      inner(astar1,bstar1), inner(bstar1,cstar1), inner(cstar1,astar1)
    # print "cross products sign(astar1xbstar1).cstar1", sign(inner(cross(astar1,bstar1),cstar1))

    # matstarlab = array([-mm[1,0],mm[0,0],mm[2,0],mm[1,1],-mm[0,1],-mm[2,1],mm[1,2],-mm[0,2],-mm[2,2]])
    matstarlab = np.array([-mm[1, 0],
                            mm[0, 0],
                            mm[2, 0],
                            -mm[1, 1],
                            mm[0, 1],
                            mm[2, 1],
                            -mm[1, 2],
                            mm[0, 2],
                            mm[2, 2]])

    matstarlab = matstarlab / GT.norme_vec(matstarlab[:3])

    return matstarlab


def matstarlabOR_to_matstarlabLaueTools(matstarlab):
    r"""
    reciprocal function of matstarlabLaueTools_to_matstarlabOR
    """
    mm = matstarlab

    # print "check matrix before normalizing"
    # print "normes astar1 bstar1 cstar1 = ", GT.norme_vec(astar1),GT.norme_vec(astar1),GT.norme_vec(astar1)
    # print "inner products astar1.bstar1, bstar1.cstar1, cstar1.astar1 \n",\
    #      inner(astar1,bstar1), inner(bstar1,cstar1), inner(cstar1,astar1)
    # print "cross products sign(astar1xbstar1).cstar1", sign(inner(cross(astar1,bstar1),cstar1))

    UBmat = np.array([[mm[1], mm[4], mm[7]], [-mm[0], -mm[3], -mm[6]], [mm[2], mm[5], mm[8]]])

    return UBmat


def matstarlab_to_matwithlatpar(matstarlab, dlatu_rad):
    """  OR method to convert UB matrix to matrix with vec recirpocical basis.
    .. todo:: to put in CrustalParameters or LaueGeometry?
    """
    norm_vec0 = np.sqrt(np.inner(matstarlab[0:3], matstarlab[0:3]))
    matnorm = matstarlab / norm_vec0
    rlatsr = CP.matrix_to_rlat(GT.matline_to_mat3x3(matnorm), angles_in_deg=0)
    dlatsr = CP.dlat_to_rlat(rlatsr, angles_in_deg=0)

    # print "matstarlab = \n", matstarlab
    dil = CP.dlat_to_dil(dlatu_rad, dlatsr, angles_in_deg=0)
    # print "dilatation =", dil
    rlats1 = np.hstack((rlatsr[0:3] * (1.0 + dil), rlatsr[3:6]))
    # print "rlats1 = ", rlats1
    mat = matnorm * rlats1[0]
    # dlats1 = CP.dlat_to_rlat(rlats1, angles_in_deg=0)
    # print "dlats1 = ", dlats1

    return mat


def readlt_det(filedet, returnmatLT=False, min_matLT=False):
    """ OR method to read .det file
    .. todo:: use better IOLauetools method
    """
    print("reading info from LaueTools det file : \n", filedet)
    print("calibration, orientation matrix")
    print("convert matrix to matstarlabOR")

    calib, mat_line = IOLT.readfile_det(filedet)

    matLT3x3 = (GT.matline_to_mat3x3(mat_line)).T

    if min_matLT:
        matmin, _ = FindO.find_lowest_Euler_Angles_matrix(matLT3x3)
        matLT3x3 = matmin

    matstarlab = matstarlabLaueTools_to_matstarlabOR(matLT3x3)

    print("matstarlab = \n", matstarlab.round(decimals=6))

    if not returnmatLT:
        return (calib, matstarlab)
    else:
        return (calib, matstarlab, matLT3x3)


def readlt_fit(filefit, returnmatLT=False, min_matLT=False, readmore=False, verbose=1, verbose2=0,
                                                                                readmore2=False):
    """
    .. todo::

        to put in IOLauetools

    modif 03Aug12 : genfromtxt removed (problem with skip_footer)
    add transfo of HKL's if matmin_LT  == True
    """

    if verbose:
        print("reading info from LaueTools fit file : \n", filefit)
        print("strained orientation matrix, peak list")
        print("convert matrix to matstarlabOR")

    matLT3x3 = np.zeros((3, 3), dtype=np.float)
    strain = np.zeros((3, 3), dtype=np.float)
    f = open(filefit, "r")
    i = 0
    matrixfound = 0
    calibfound = 0
    pixdevfound = 0
    strainfound = 0
    eulerfound = 0
    linecalib = 0
    linepixdev = 0
    linestrain = 0
    lineeuler = 0
    list1 = []
    linestartspot = 10000
    lineendspot = 10000
    try:
        for line in f:
            i = i + 1
            # print i
            if line[:5] == "spot#":
                # linecol = line.rstrip("\n")
                linestartspot = i + 1
            if line[:3] == "#UB":
                # print line
                matrixfound = 1
                linestartmat = i
                lineendspot = i
                j = 0
                # print "matrix found"
            if line[:3] == "#Sa":
                # print line
                calibfound = 1
                linecalib = i + 1
            if line[:3] == "#pi":
                # print line
                pixdevfound = 1
                linepixdev = i + 1
            if line[:3] == "#de":
                # print line
                strainfound = 1
                linestrain = i
                j = 0
            if line[:3] == "#Eu":
                # print line
                eulerfound = 1
                lineeuler = i + 1
            if matrixfound:
                if i in (linestartmat + 1, linestartmat + 2, linestartmat + 3):
                    strline = line.rstrip("\n").replace("[", "").replace("]", "").split()
                    matLT3x3[j, :] = np.array(strline, dtype=float)
                    j = j + 1
            if strainfound:
                if i in (linestrain + 1, linestrain + 2, linestrain + 3):
                    strline = line.rstrip("\n").replace("[", "").replace("]", "").split()
                    strain[j, :] = np.array(strline, dtype=float)
                    j = j + 1
            if calibfound & (i == linecalib):
                calib = np.array(line.split(",")[:5], dtype=float)
                # print "calib = ", calib
            if eulerfound & (i == lineeuler):
                euler = np.array(line.replace("[", "").replace("]", "").split()[:3], dtype=float)
                # print "euler = ", euler
            if pixdevfound & (i == linepixdev):
                pixdev = float(line.rstrip("\n"))
                # print "pixdev = ", pixdev
            if (i >= linestartspot) & (i < lineendspot):
                list1.append(line.rstrip("\n").replace("[", "").replace("]", "").split())
    finally:
        f.close()
        # linetot = i

    # print "linetot = ", linetot

    data_fit = np.array(list1, dtype=float)

    if verbose:
        print(np.shape(data_fit))
        print(data_fit[0, :])
        print(data_fit[-1, :])

    # print "UB matrix = \n", matLT3x3.round(decimals=6)

    if verbose2:
        print("before transfo")
        print(data_fit[0, 2:5])
        print(data_fit[-1, 2:5])
        q0 = np.dot(matLT3x3, data_fit[0, 2:5])
        print("q0 = ", q0.round(decimals=4))
        qm1 = np.dot(matLT3x3, data_fit[-1, 2:5])
        print("qm1 = ", qm1.round(decimals=4))

    if min_matLT:
        matmin, transfmat = FindO.find_lowest_Euler_Angles_matrix(
            matLT3x3, verbose=verbose
        )
        matLT3x3 = matmin
        if verbose:
            print("transfmat \n", list(transfmat))
        # transformer aussi les HKL pour qu'ils soient coherents avec matmin
        hkl = data_fit[:, 2:5]
        data_fit[:, 2:5] = np.dot(transfmat, hkl.transpose()).transpose()

    if verbose2:
        print("after transfo")
        print(data_fit[0, 2:5])
        print(data_fit[-1, 2:5])
        q0 = np.dot(matLT3x3, data_fit[0, 2:5])
        print("q0 = ", q0.round(decimals=4))
        qm1 = np.dot(matLT3x3, data_fit[-1, 2:5])
        print("qm1 = ", qm1.round(decimals=4))

    matstarlab = matstarlabLaueTools_to_matstarlabOR(matLT3x3)

    if verbose:
        print("matstarlab = \n", matstarlab.round(decimals=6))

    if readmore2:
        readmore = False

    # xx yy zz yz xz xy
    strain6 = np.array([strain[0, 0],
                        strain[1, 1],
                        strain[2, 2],
                        strain[1, 2],
                        strain[0, 2],
                        strain[0, 1]])

    if not returnmatLT:
        if readmore:
            return (matstarlab, data_fit, calib, pixdev)
        elif readmore2:
            return (matstarlab, data_fit, calib, pixdev, strain6, euler)
        else:
            return (matstarlab, data_fit)
    else:
        if readmore:
            return (matstarlab, data_fit, matLT3x3, calib, pixdev)
        elif readmore2:
            return (matstarlab, data_fit, matLT3x3, calib, pixdev, strain6, euler)
        else:
            return (matstarlab, data_fit, matLT3x3)


def readall_str(grain_index, filemane_str, returnmatLT=False, min_matLT=False):
    r"""
    .. todo::

        to put in IOLauetools
    """

    data_str, matstr, calib, dev_str = IOLT.readfile_str(filemane_str, grain_index)

    # postprocessing

    data_str[:, 2:5] = -data_str[:, 2:5]

    satocrs = matstr.transpose()
    # print "strained orientation matrix (satocrs) = \n", satocrs

    matstarlab = matxmas_to_matstarlab(satocrs, calib)

    if min_matLT:
        matLT3x3 = matstarlabOR_to_matstarlabLaueTools(matstarlab)
        matmin, transfmat = FindO.find_lowest_Euler_Angles_matrix(matLT3x3)
        matLT3x3 = matmin
        matstarlab = matstarlabLaueTools_to_matstarlabOR(matLT3x3)
        # transfo des HKL a verifier
        hklmin = np.dot(transfmat, data_str[:, 2:5].transpose()).T
        data_str[:, 2:5] = hklmin

    if not returnmatLT:
        return (data_str, matstarlab, calib, dev_str)
    else:
        return (data_str, matstarlab, calib, dev_str, matLT3x3)


def matxmas_to_matstarlab(satocr, calib):
    r"""
    Original function to correctly use matrix from STR or IND

    # modif 04 Mar 2010 xbet en degres au lieu de radians

    # satocrs = transposee de la matrice numero 2 du .STR :
    # "coordinates of a*, b*, c* in X, Y, Z"
    # satocru = matrice numero 2 du .IND :
    #  "matrix hkl => XYZ"
    #print "convert matrix from XMAS sample axes to lab axes + normalize by astar"
    """
    astar = GT.norme_vec(satocr[:, 0])
    # print "satocr \n", satocr
    satocrnorm = satocr / astar
    # print "satocrnorm \n", satocrnorm

    omega0 = 40.0
    xbetrad = calib[3] * DEG

    # print "xbetrad = " , xbetrad

    omega = omega0 * DEG - xbetrad / 2.0

    # print "omega" , omega*180.0/np.pi

    # rotation de omega autour de l'axe x pour repasser dans Rlab
    matrot = np.array([[1.0, 0.0, 0.0],
            [0.0, np.cos(omega), np.sin(omega)],
            [0.0, -np.sin(omega), np.cos(omega)]])

    labtocr = np.dot(matrot, satocrnorm)
    astarlab = labtocr[:, 0]
    bstarlab = labtocr[:, 1]
    cstarlab = labtocr[:, 2]

    matstarlab1 = np.hstack((astarlab, bstarlab, cstarlab))
    changesign = np.array([-1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0])
    matstarlab2 = np.multiply(matstarlab1, changesign)
    # print "matstarlab from satocr \n", matstarlab2
    return matstarlab2


def Compute_data2thetachi(filename, sorting_intensity="yes", detectorparams=None,
                                                        kf_direction="Z>0",
                                                        verbose=1,
                                                        pixelsize=165.0 / 2048,
                                                        dim=(2048, 2048),
                                                        saturation=0,
                                                        forceextension_lines_to_extract=None,
                                                        col_isbadspot=None,
                                                        alpha_xray_incidence_correction=None):
    r"""
    Read a file and convert spot positions x,y to scattering angles 2theta, chi according to detector parameters

    :param filename: fullpath to peaks list ASCII file
    :type filename: string

    :param detectorparams: list of CCD calibration parameters [det, xcen, ycen, xbet, xgam]
    :param pixelsize: pixelsize in mm
    :param dim: (nb pixels x, nb pixels y)

    :param kf_direction: label of detection geometry (CCD position): 'Z>0','X>0',...
    :type kf_direction: string

    :param sorting_intensity: 'yes' sort spots list by decreasing intensity

    saturation = 0 : do not read Ipixmax column of DAT file from LaueTools peaksearch
    saturation > 0 : read Ipixmax column and create data_sat list
    data_sat[i] = 1 if Ipixmax[i]> saturation, =0 otherwise

    col_Ipixmax = 10 for .dat from LT peak search using method "Local Maxima"
    (TODO : bug in Ipixmax for method "convolve")

    :returns: twicetheta, chi, dataintensity, data_x, data_y  [, other data]
    """
    extension = filename.split(".")[-1]

    if forceextension_lines_to_extract is not None:
        extension = "forcedextension"

    if extension == "peaks":  # single line header   # TODO to remove
        data_xyI = np.loadtxt(filename, usecols=(0, 1, 2), skiprows=1)

    elif extension in ("dat", "DAT"):  # peak list single line header
        alldata, nbpeaks = IOLT.readfile_dat(filename, returnnbpeaks=True)
        # print('nbpeaks', nbpeaks)
        # print('alldata', alldata)
        if nbpeaks > 1:
            data_xyI = np.take(alldata, (0, 1, 3), axis=1)
        elif nbpeaks == 1:
            data_xyI = np.take(alldata, (0, 1, 3), axis=0)

        print("nb of spots and columns in .dat file", data_xyI.shape)

        if saturation:
            data_Ipixmax = alldata[:, -1]
            indsat = np.where(data_Ipixmax >= saturation)
            data_sat = np.zeros(len(data_Ipixmax), dtype=np.int)
            data_sat[indsat[0]] = 1

            if col_isbadspot is not None:
                data_isbadspot = alldata[:, col_isbadspot]
                #print(data_isbadspot)

    elif extension == "forcedextension":
        data_xyI = np.loadtxt(filename, usecols=(0, 1, 2), skiprows=1)
    elif extension == "cor":  # single line header
        # TODO use better  IOLT function...
        try:
            data_xyI = np.loadtxt(filename, usecols=(2, 3, 4), skiprows=1)
        except:
            raise ValueError("%s can not be read" % filename)
    else:
        raise ValueError("Unknown file extension for %s" % filename)

    sha = data_xyI.shape

    # manage if there is a single spot
    if len(sha) == 1:
        nb_peaks = 1
    else:
        nb_peaks = sha[0]

    if detectorparams is None:
        raise ValueError("Missing detectorparams arg in Compute_data2thetachi() of find2thetachi module")
    else:
        param_det = detectorparams

    if verbose:
        print("file :%s" % filename)
        print("containing %d peaks" % nb_peaks)

    if filename.split(".")[-1] in ("pik", "peaks"):
        data_x = data_xyI[:, 0]  # + 0.5  # 0.5 for being closer to XMAS peaks position
        data_y = (dim[1] - data_xyI[:, 1])  # + 0.5 # 0.5 for being closer to XMAS peaks position
        data_I = data_xyI[:, 2]  # for fit2d pixels convention

    elif filename.split(".")[-1] in ("dat", "DAT"):
        if nb_peaks > 1:
            data_x = data_xyI[:, 0]
            data_y = data_xyI[:, 1]
            data_I = data_xyI[:, 2]
        elif nb_peaks == 1:
            data_x = [data_xyI[0], data_xyI[0]]
            data_y = [data_xyI[1], data_xyI[1]]
            data_I = [data_xyI[2], data_xyI[2]]

    if extension in ("forcedextension", "cor"):
        data_x = data_xyI[:, 0]
        data_y = data_xyI[:, 1]
        data_I = data_xyI[:, 2]

    # 21Jul14  O. Robach---------------
    if alpha_xray_incidence_correction != None:

        print("Using alpha_xray_incidence_correction = ", alpha_xray_incidence_correction)
        xystart = np.column_stack((data_x, data_y))
        #        print "xystart = ", xystart
        npics = np.shape(xystart)[0]
        xynew = np.zeros((npics, 2), float)
        xycen = np.array([param_det[1], param_det[2]])
        #        print "xycen = ", xycen
        dxy = xystart - xycen
        dxynorm2 = (np.multiply(dxy, dxy)).sum(axis=1)
        #        print "dxynorm2 = ", dxynorm2
        dxynorm = np.power(dxynorm2, 0.5)
        #        print "dxynorm =", dxynorm
        # dxynorminv = 1.0 / dxynorm
        scale_factor = pixelsize / param_det[0]
        scale_factor = scale_factor * scale_factor
        #        print "dd = ", param_det[0]
        #        print "pixelsize = ", pixelsize
        #        print "scale_factor = ", scale_factor
        for i in list(range(npics)):
            xynew[i, :] = (xystart[i, :]
                                + alpha_xray_incidence_correction
                                * scale_factor
                                * dxy[i, :]
                                * dxynorm[i])

        delta_xy = xynew - xystart
        #        print "delta_xy = ", delta_xy
        print("maximum spot displacement |dx| |dy| : ",
            (abs(delta_xy)).max(axis=0).round(decimals=3))

        data_x = xynew[:, 0]
        data_y = xynew[:, 1]

    # ----compute scattering angles2theta and chi --------------------------
    twicethetaraw, chiraw = calc_uflab(data_x, data_y, param_det[:5], returnAngles=1,
                                                                        pixelsize=pixelsize,
                                                                        kf_direction=kf_direction)
    #-----------------------------------------------------------------------
    # print chi,twicetheta
    if nb_peaks > 1 and sorting_intensity == "yes":
        listsorted = np.argsort(data_I)[::-1]
        chi = np.take(chiraw, listsorted)
        twicetheta = np.take(twicethetaraw, listsorted)
        data_x = np.take(data_x, listsorted)
        data_y = np.take(data_y, listsorted)
        dataintensity = np.take(data_I, listsorted)
        if saturation:
            data_sat = np.take(data_sat, listsorted)
        if col_isbadspot != None:
            data_isbadspot = np.take(data_isbadspot, listsorted)

    else:
        dataintensity = data_I
        chi = chiraw
        twicetheta = twicethetaraw

    if nb_peaks == 1:
        dataintensity = [dataintensity[0]]
        chi = [chi[0]]
        twicetheta = [twicetheta[0]]
        data_x = [data_x[0]]
        data_y = [data_y[0]]

    if saturation:
        print("adding flag column for saturated peaks")
        return twicetheta, chi, dataintensity, data_x, data_y, data_sat
    if col_isbadspot != None:
        return twicetheta, chi, dataintensity, data_x, data_y, data_sat, data_isbadspot

    else:
        if col_isbadspot != None:
            return twicetheta, chi, dataintensity, data_x, data_y, data_isbadspot
        else:
            return twicetheta, chi, dataintensity, data_x, data_y


def convert2corfile(filename, calibparam, dirname_in=None, dirname_out=None, pixelsize=165.0 / 2048,
                                                                                CCDCalibdict=None,
                                                                                add_props=False):
    r"""
    Convert .dat (peaks list from peaksearch procedure) to .cor (adding scattering angles 2theta chi)

    From X,Y pixel positions in peak list file (x,y,I,...) and detector plane geometry comptues scattering angles 2theta chi
    and creates a .cor file (ascii peaks list (2theta chi X Y int ...))

    :param calibparam: list of 5 CCD calibration parameters (used if CCDCalibdict is None or  CCDCalibdict['CCDCalibPameters'] is missing)

    :param pixelsize: CCD pixelsize (in mm) (used if CCDCalibdict is None or CCDCalibdict['pixelsize'] is missing)

    :param CCDCalibdict: dictionary of CCD file and calibration parameters

    :param add_props: add all peaks properties to .cor file instead of the 5 columns
    """
    if dirname_in != None:
        filename_in = os.path.join(dirname_in, filename)
    else:
        filename_in = filename

    print('CCDCalibdict in convert2corfile of %s'%filename, CCDCalibdict)
    if CCDCalibdict is not None:
        if "CCDCalibParameters" in CCDCalibdict:
            calibparam = CCDCalibdict["CCDCalibParameters"]

        if "xpixelsize" in CCDCalibdict:
            pixelsize = CCDCalibdict["xpixelsize"]

    (twicetheta, chi, dataintensity, data_x, data_y) = Compute_data2thetachi(filename_in,
                                                               sorting_intensity="yes",
                                                                            detectorparams=calibparam,
                                                                            pixelsize=pixelsize)
    if add_props:
        if len(twicetheta) > 1:
            rawdata, allcolnames = IOLT.read_Peaklist(filename_in, output_columnsname=True)
            # need to sort data by intensity (col 2)
            sortedind = np.argsort(rawdata[:, 2])[:: -1]
            data = rawdata[sortedind]

            add_props = (data[:, 4:], allcolnames[4:])
        else:
            rawdata, allcolnames = IOLT.read_Peaklist(filename_in, output_columnsname=True)
            # need to sort data by intensity (col 2)
            add_props = (np.array([rawdata[4:]]), allcolnames[4:])

    # TODO: handle windowsOS path syntax
    filename_wo_path = filename.split("/")[-1]

    file_extension = filename_wo_path.split(".")[-1]

    prefix_outputname = filename_wo_path[: -len(file_extension) - 1]

    if dirname_out != None:
        filename_out = os.path.join(dirname_out, prefix_outputname)
    else:
        filename_out = prefix_outputname

    if CCDCalibdict is not None:
        for kk, key in enumerate(DictLT.CCD_CALIBRATION_PARAMETERS[:5]):
            CCDCalibdict[key] = calibparam[kk]

        CCDCalibdict["xpixelsize"] = pixelsize
        CCDCalibdict["ypixelsize"] = pixelsize
        CCDCalibdict["pixelsize"] = pixelsize

        param = CCDCalibdict

        # update dict according to values in file .cor
        f = open(filename_in, "r")
        param = IOLT.readCalibParametersInFile(f, Dict_to_update=CCDCalibdict)
        f.close()

    else:
        param = calibparam + [pixelsize]

    # print('add_props', data.shape, add_props)

    IOLT.writefile_cor(filename_out, twicetheta, chi, data_x, data_y, dataintensity,
                                                            data_props=add_props,
                                                            sortedexit=0,
                                                            param=param,
                                                            initialfilename=filename)


def convert2corfile_fileseries(fileindexrange, filenameprefix, calibparam, suffix="",
            nbdigits=4,
            dirname_in=None,
            dirname_out=None,
            pixelsize=165.0 / 2048,
            fliprot="no"):
    r"""
    convert a serie of peaks list ascii files to .cor files (adding scattering angles).

    Filename is decomposed as following for incrementing file index in ####:
    prefix####suffix
    example: myimage_0025.myccd => prefix=myimage_ nbdigits=4 suffix=.myccd

    :param nbdigits: nb of digits of file index in filename (with zero padding)
        (example: for myimage_0002.ccd nbdigits = 4

    :param calibparam: list of 5 CCD cakibration parameters
    """
    encodingdigits = "%%0%dd" % nbdigits

    if suffix == "":
        suffix = ".dat"

    for fileindex in list(range(fileindexrange[0], fileindexrange[1] + 1)):
        filename_in = filenameprefix + encodingdigits % fileindex + suffix
        print("filename_in", filename_in)
        convert2corfile(filename_in,
                        calibparam,
                        dirname_in=dirname_in,
                        dirname_out=dirname_out,
                        pixelsize=pixelsize)


# def convert2corfile_multiprocessing(fileindexrange,
#                                     filenameprefix,
#                                     calibparam,
#                                     dirname_in=None,
#                                     suffix="",
#                                     nbdigits=4,
#                                     dirname_out=None,
#                                     pixelsize=165.0 / 2048,
#                                     fliprot="no",
#                                     nb_of_cpu=6):
#     """
#     launch several processes in parallel to convert .dat file to .cor file
#     """
#     import multiprocessing

#     try:
#         index_start, index_final = fileindexrange
#     except:
#         raise ValueError("Need 2 file indices integers in fileindexrange=(indexstart, indexfinal)")

#     fileindexdivision = GT.getlist_fileindexrange_multiprocessing(index_start, index_final, nb_of_cpu)

#     #    t00 = time.time()
#     jobs = []
#     for ii in list(range(nb_of_cpu)):
#         proc = multiprocessing.Process(
#             target=convert2corfile_fileseries,
#             args=(fileindexdivision[ii],
#                 filenameprefix,
#                 calibparam,
#                 suffix,
#                 nbdigits,
#                 dirname_in,
#                 dirname_out,
#                 pixelsize))
#         jobs.append(proc)
#         proc.start()


#    t_mp = time.time() - t00
#    print "Execution time : %.2f" % t_mp


def fromlab_tosample(UB, anglesample_deg=40):  # in deg
    """
    compute UBs

    qs = UBs G   with G =ha*+kb*+lc* a*,b*,c* are aligned with lab frame x,y,z
    qs is q in sample frame deduced from lab by a 40* rotation around y (lauetools convention)

    lauetools convention: x // ki ie. ki = 2pi/lambda(1,0,0), z perpendicular to x and contained
    in the plane defined by x and dd* u where u is a unit vector normal to the CCD plane
    and dd is the shortest distance between the CCD plane and the emission source of scattered beams

    """
    anglesample = anglesample_deg * DEG  # in rad
    Rot = np.array([[np.cos(anglesample), 0, np.sin(anglesample)],
                            [0, 1, 0],
                            [-np.sin(anglesample), 0, np.cos(anglesample)]])

    # = GT.matRot([0,1,0], 40)
    # invRot = np.linalg.inv(Rot)
    UBs = np.dot(Rot, UB)
    return UBs


def vec_normalTosurface(mat_labframe):
    r"""
    solve Mat * X = (0,0,1) for X
    for pure rotation invMat = transpose(Mat)

    TODO: add option sample angle and axis
    """
    # last row of matrix in sample frame
    return fromlab_tosample(mat_labframe)[2]


def vec_onsurface_alongys(mat_labframe):
    r"""
    solve Mat * X = (0,1,0) for X
    for pure rotation invMat = transpose(Mat)
    """
    return fromlab_tosample(mat_labframe)[1]


# ---------------------------------------------------------------------------
# ---------------------------WIRE TECHNIQUE ---------------------------------
# ---------------------------------------------------------------------------


# Following functions are for dealing with in depth  x-ray emission source

def find_yzsource_from_IM_uf(IM, uf, depth_z=0, anglesample=40.0):
    r"""
    from vector IM in absolute frame  I origin, M point in CCD plane
    uf: unit vector in absolute frame joining Iprime (source) and M
    depth_z: in microns known vertical offset

    returns x and y position of emission source
    """
    ux, uy, uz = uf.T
    x, y, z = IM.T

    deltaZ = z - depth_z * 1000.0  # in millimeters
    ratio = deltaZ / uz

    xsource = x - ratio * ux
    ysource = y - ratio * uy
    zsource = depth_z

    # rotation matrix from absolute to sample frame
    # Xs= R Xabs
    anglerad = anglesample * DEG
    ca = np.cos(anglerad)
    sa = np.sin(anglerad)
    R = np.array([[1, 0, 0], [0, ca, sa], [0, -sa, ca]])
    # source position in sample frame
    tsource = np.array(xsource, ysource, zsource)
    xsource_s, ysource_s, zsource_s = np.dot(R, tsource)

    IIprime = tsource.T
    IIprime_s = np.array([xsource_s, ysource_s, zsource_s]).T

    return IIprime, IIprime_s


def IMlab_from_xycam(xcam, ycam, calib, verbose=0):
    r"""
    returns list of vector position of M (on CCD camera) in absolute frame
    from pixels position vector in CCD frame
    """
    _, IMlab = calc_uflab(xcam, ycam, calib, returnAngles="uflab")

    if verbose:
        print("IMlab", IMlab)

    return IMlab


def IW_from_IM_onesource(IIprime, IM, depth_wire, anglesample=40.0, anglewire=40.0):
    r"""
    from:
    II': single vector II' (2 elements= y,z) source position in absolute frame
    IM: array of vectors IM (3 elements= x,y,z) point on CCD in absolute frame
    depth_wire: height normal to the surface of the wire
    I origin of absolute frame and calibrated source emission

    returns:
    array of vectors wire position (y,z)  (hypothesis: wire parallel to Ox)
    """
    try:
        yIp, zIp = IIprime
    except ValueError:
        print("Next time, please use a 2elements source position (y,z)")
        _, yIp, zIp = IIprime

    angs = anglesample * DEG
    angw = anglewire * DEG

    _, y, z = (IM * 1.0).T

    IH = np.array([0, -depth_wire * np.sin(angs), depth_wire * np.cos(angs)])

    slopeM = (z - zIp) / (y - yIp)
    slopefil = -np.tan(angw)

    cstM = slopeM * yIp + zIp
    cstH = slopefil * IH[1] + IH[2]

    yw = (cstM - cstH) / (slopeM - slopefil)
    zw = cstH - slopefil * yw

    return np.array([yw, zw])


def IW_from_source_oneIM(IIprime, IM, depth_wire, anglesample=40.0):
    r"""
    TODO : MAY BE FALSE
    from:
    II': array of  vectors II' (2 elements= y,z) source position in absolute frame
    IM: SINGLE vector IM (3 elements= x,y,z) point on CCD in absolute frame
    depth_wire: height normal to the surface of the wire
    I origin of absolute frame and calibrated source emission

    returns:
    array of vectors wire position (y,z)  (hypothesis: wire parallel to Ox)
    """

    yIp, zIp = (IIprime * 1.0).T
    _, y, z = IM

    slope = (zIp - z) / (yIp - y)

    ang = anglesample * DEG
    tw0 = np.tan(ang)

    yw = (yIp * slope + zIp + depth_wire) / (slope - tw0)
    zw = tw0 * yw + depth_wire

    return np.array([yw, zw])


def find_yzsource_from_xycam_uf(OM, uf, calib, depth_z=0, anglesample=40.0):
    r"""
    from:
    OM: list of vectors OM (2 elements) in CCD plane in CCD frame (pixels unit)
    uf: list of unit vectors (3 elements) in absolute frame
    depth_z:  known vertical offset of the beam with respect in microns
                default value = 0 (source is along the line passing through I origin for CCD calibration)


    returns:
    list of position [y,z] of emission source in absolute frame
    list of position [y,z] of emission source in sample frame
    """
    xcam, ycam = OM.T

    IMlab = IMlab_from_xycam(xcam, ycam, calib)

    IMlab_yz = IMlab[:, 1:]

    return find_yzsource_from_IM_uf(IMlab_yz, uf, depth_z=depth_z, anglesample=anglesample)


def find_yzsource_from_2xycam_2yzwire(OMs, IWs, calib):
    """
    rfrom:
    OMs: array of 2 vectors OM (2 elements) in CCD plane in CCD frame (pixels unit): array([OM1,OM2])
    IWs: array of 2 position vectors (2 elements= [y,z]) in absolute frame of wire (which is parallel to Ox): array([IW1,IW2]

    returns:
    y,z position of source of emission (hypothesis x=0)
    """
    xcam, ycam = OMs.T

    IMlab = IMlab_from_xycam(xcam, ycam, calib)

    _, Y_IM, Z_IM = IMlab.T
    y1, y2 = Y_IM
    z1, z2 = Z_IM

    YW, ZW = IWs.T
    u1, u2 = YW
    v1, v2 = ZW

    A = u1 - y1
    B = v1 - z1
    C = u2 - y2
    D = v2 - z2

    E = -y1 * B + z1 * A
    F = -y2 * D + z2 * C

    determ = A * D - B * C

    ysource = (C * E - A * F) / determ
    zsource = (D * E - B * F) / determ

    return np.array([ysource, zsource])


def find_yzsource_from_2xycam_2yzwire_version2(OMs, IWs, calib, verbose=0):
    r"""
    from:
    OMs: array of 2 vectors OM (2 elements) in CCD plane in CCD frame (pixels unit): array([OM1,OM2])
    IWs: array of 2 position vectors (2 elements= [y,z]) in absolute frame of wire (which is parallel to Ox): array([IW1,IW2]

    assumption: xsource = 0

    returns:
    position of source of emission

    """
    xcam, ycam = OMs.T

    IMlab = IMlab_from_xycam(xcam, ycam, calib)

    IM_1, IM_2 = IMlab[:2, 1:]  # y,z of IMs

    IW_1, IW_2 = IWs[:2]  # y,z of IW

    A, B = GT.ShortestLine(IM_1, IW_1, IM_2, IW_2)

    if verbose:
        print("line1 points", IM_1, IW_1)
        print("line2 points", IM_2, IW_2)
        print("two points", A, B)
        print("distance ", np.sqrt(np.dot(A - B, A - B)))

    return (A + B) / 2.0


def find_multiplesourcesyz_from_multiplexycam_multipleyzwire(OMs, Wire_abscissae, calib,
                                                                                anglesample=40.0,
                                                                                wire_height=0.3,
                                                                                verbose=0):
    r"""
    from:
    OMs: array of n vectors OM (2 elements) in CCD plane in CCD frame (pixels unit): array([OM1,OM2, ..., OMn])
    IWs: array of n wire abscissae of wire (which is parallel to Ox): array([W1,W2,...,Wn]
    Wire (strictly parallel to x) travels strictly at anglesample from horizontal plane (defined by ui,xbet and CCD plane in calibration). Wire abscissa is zero when wire is on top of point I of calibration at height wire_height. Wire abscissa increases in same direction as y (and Xray beam).

    assumption: 1) xsource = 0
                2) sources lying in yOz plane may only differ from incident direction in this plane

    returns:
    all positions of source of emission from all pairs [spot,Wire_abscissa]
    """

    IWs = IW_from_wireabscissa(Wire_abscissae, wire_height, anglesample=anglesample)  # array of [y,z]
    IWs = IWs.T
    pairs = GT.pairs_of_indices(len(IWs))

    if verbose:
        print(Wire_abscissae)
        print(OMs)
        print(IWs)
        print(pairs)

    results = []
    for k in list(range(len(pairs))):
        p = pairs[k]
        if verbose:
            print("p", p)
            print("oms", np.take(OMs, p, axis=0))
            print("iws", np.take(IWs, p, axis=0))

        results.append(
            find_yzsource_from_2xycam_2yzwire_version2(np.take(OMs, p, axis=0),
                                                        np.take(IWs, p, axis=0),
                                                        calib,
                                                        verbose=0))

    return np.array(results)


def IW_from_wireabscissa(abscissa, wire_height, anglesample=40.0):
    r"""
    from:
    abscissa of wire and wire height from sample surface inclined by anglesample (deg)

    returns:
    absolute coordinate of wire (hypothesis x is undetermined)
    """
    ang = anglesample * DEG
    sw0 = np.sin(ang)
    cw0 = np.cos(ang)
    y = abscissa * cw0 - wire_height * sw0
    z = abscissa * sw0 + wire_height * cw0
    return np.array([y, z])


def Wireabscissa_from_IW(IWy, IWz, wire_height, anglesample=40.0):
    r"""
    from:
    absolute coordinate of wire (hypothesis x is undetermined)
    wire height from sample surface inclined by anglesample (deg)

    returns:
    abscissa of wire along ysample inclined by anglesample from point at IIw0 distance from I
    """
    angs = anglesample * DEG

    # IH = np.array([0, -wire_height * np.sin(angs), wire_height * np.cos(angs)])

    # WH = array([0, IWy, IWz]) -  IH

    WHy = IWy - (-wire_height * np.sin(angs))
    WHz = IWz - wire_height * np.cos(angs)

    # add sign simply for angle wire  around 40 deg
    signe = np.where(IWz > wire_height * np.cos(angs), 1, -1)

    return signe * np.sqrt(WHy ** 2 + WHz ** 2)


def twotheta_from_wire_and_source(ysource, Height_wire, Abscissa_wire, anglesample=40):
    r"""
    point moving parallel to sample surface in Oyz plane

    ysource= II'  abscissa of spots emission (I') from calibration origin source (I)
    Height_wire= height of moving point Iw from sample surface
    Abscissa_wire= abscissa of moving point along its straight trajectory from point Iw0
            which is in between I and microscope  (normal at sample surface)

    returns scattering angle 2theta from y direction to I'Iw (no x component)
    """

    lambda_angle = np.arctan(Height_wire * 1.0 / Abscissa_wire) / DEG + anglesample
    coslambda = np.cos(lambda_angle * DEG)
    yprime = ysource * 1.0 / Abscissa_wire

    return (np.arccos((coslambda - yprime) / np.sqrt(1 + yprime ** 2 - 2 * yprime * coslambda)
        ) / DEG)


def convert_xycam_from_sourceshift(OMs, IIp, calib, verbose=0):
    r"""
    From x,y on CCD camera (OMs) and source shift (IIprime)
    compute modified x,y values for the SAME calibration (calib)(for further analysis)

    return new value of x,y
    """
    xcam, ycam = OMs.T
    uflab, IM = calc_uflab(xcam, ycam, calib, returnAngles=0, verbose=0, pixelsize=165.0 / 2048)  # IM normalized
    # IM vectors
    # IM=IprimeM_from_uf(uflab,array([0,0,0]),calib,verbose=0)
    OM = calc_xycam(uflab, calib, energy=1, offset=None, verbose=0, returnIpM=False,
                                                                pixelsize=165.0 / 2048)
    x0cam, y0cam, th0, E0 = OM
    # IpM=IpI + IM= IM-IIP
    IpM = IM - IIp
    nor = np.sqrt(np.sum(IpM ** 2, axis=1))
    # new uf prime
    ufp = IpM * 1.0 / np.reshape(nor, (len(nor), 1))
    OpMp = calc_xycam(ufp, calib, energy=1, offset=None, verbose=0, returnIpM=False,
                                                                pixelsize=165.0 / 2048)
    xpcam, ypcam, thp, Ep = OpMp

    if verbose:
        print("\n for source at I and Iprime")
        print("X in pixel")
        print(x0cam, xpcam)
        print("Y in pixel")
        print(y0cam, ypcam)
        print("2theta in deg")
        print(2.0 * th0, 2.0 * thp)
        print("Energy")
        print(E0, Ep)
        # print uflab,ufp
        print("\n for IIprime (mm)", IIp)
        print("And calibration ", calib)
        print("\n shift to add to X")
        print(xpcam - x0cam)
        print("shift to add to Y")
        print(ypcam - y0cam)
        print("shift to add to 2theta in deg")
        print(2.0 * (thp - th0))
        print("shift to add to Energy (IN eV)")
        print(1000.0 * (Ep - E0))


def absorbprofile(x, R, mu, x0):
    """ absorption profile"""
    cond = np.fabs(x - x0) <= R
    print(cond)
    print(cond.dtype)

    def Absorbfunction(x, radius=R, coef=mu, center=x0):
        """ absorption function """
        return np.exp(-2 * coef * np.sqrt(radius ** 2 - (x - center) ** 2))

    yabs = list(map(Absorbfunction, x))
    print(yabs)
    # y=np.piecewise(x,[cond],[Absorbfunction,1.],radius=R,coef=mu,center=x0)
    y = np.select([cond], [yabs], 1.0)
    return y


def lengthInSample(depth, twtheta, chi, omega, verbose=False):
    r""" compute geometrical lengthes in sample from impact point (I) at the surface to a point (B)
    where xray are scattered (or fluorescence is emitted) and finally escape from inside at point (C) lying at the sample surface
    (intersection of line with unit vector u with sample surface plane tilted by omega)

    .. warning::

        twtheta and chi angles can be misleading.
        Assumption is made that angles of unit vector from B to C (or to detector frame pixel) are
        :math:`2 \theta` and :math:`\chi`.
        For large depth D, unit vector scattered beam direction is not given by :math:`2 \theta` and :math:`\chi` angles
        as they are used for describing the scattering direction from point I and a given detector frame position
        (you should then compute the two angles correction , actually :math:`\chi` is unchanged, and the :math:`2 \theta` change is approx
        d/ distance .i.e. 3 10-4 for d=20 µm and CCD at 70 mm)

    .. note::

        incoming beam coming from the right positive x direction with
            - IB = (-D,0,0)
            - BC =(xc+D,yc,zc)
            - and length BC is proportional to the depth D

    """
    D = depth * 1.0
    c2theta = np.cos(twtheta * DEG)
    s2theta = np.sin(twtheta * DEG)
    cchi = np.cos(chi * DEG)
    schi = np.sin(chi * DEG)
    comega = np.cos(omega * DEG)
    somega = np.sin(omega * DEG)

    factor = D * somega / (-c2theta * somega + s2theta * cchi * comega)
    xc = factor * (-c2theta)
    yc = factor * (-s2theta * schi)
    zc = factor * (s2theta * cchi)

    BC = np.sqrt((xc + D) ** 2 + yc ** 2 + zc ** 2)

    if verbose:
        print("[x,y,z] of BC")
        print(np.array([twtheta, chi, xc, yc, zc]).T)

    Ratio_BC_over_D = BC / D

    return D + BC, BC, Ratio_BC_over_D, D
