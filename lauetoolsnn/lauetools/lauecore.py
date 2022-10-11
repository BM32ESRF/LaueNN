# -*- coding: utf-8 -*-
r"""
Core module to compute Laue Pattern in various geometry

Main author is J. S. Micha:   micha [at] esrf [dot] fr

version July 2019
from LaueTools package hosted in

http://sourceforge.net/projects/lauetools/

or

https://gitlab.esrf.fr/micha/lauetools
"""
from __future__ import division

import math
import sys

import numpy as np

np.set_printoptions(precision=15)

import builtins

try:
    from lauetools import CrystalParameters as CP
    from lauetools import IOLaueTools as IOLT
    from lauetools import generaltools as GT
    from lauetools import LaueGeometry as LTGeo
    from lauetools.dict_LaueTools import (dict_Materials, dict_Extinc, CST_ENERGYKEV,
            DEFAULT_DETECTOR_DISTANCE, DEFAULT_DETECTOR_DIAMETER, DEFAULT_TOP_GEOMETRY)
except:
    import lauetoolsnn.lauetools.CrystalParameters as CP
    import lauetoolsnn.lauetools.IOLaueTools as IOLT
    import lauetoolsnn.lauetools.generaltools as GT
    import lauetoolsnn.lauetools.LaueGeometry as LTGeo
    from lauetoolsnn.lauetools.dict_LaueTools import (dict_Materials, dict_Extinc, CST_ENERGYKEV,
            DEFAULT_DETECTOR_DISTANCE, DEFAULT_DETECTOR_DIAMETER, DEFAULT_TOP_GEOMETRY)

USE_CYTHON = False

DEG = np.pi / 180.0

# --- ---------- Spot class
class spot:
    r"""
    Laue Spot class still used...

    .. todo:: To be avoided for greater performance... To be replaced by full numpy computations
    """
    def __init__(self, indice):
        self.Millers = indice
        self.Qxyz = None
        self.EwaldRadius = None
        self.Xcam = None
        self.Ycam = None
        self.Twicetheta = None
        self.Chi = None

    def __hash__(self):
        r"""
        to used indices for defining a key and using set() sorted()  ... object
        comparison
        In our case used for eliminating spots which equal fondamental key
        Warning: Indice must not be changed , intead troubles:
            see http://wiki.python.org/moin/DictionaryKeys
        """
        # listint = tuple(self.have_fond().Millers)
        # still have -2 -4 -2   =  -2 -2 -4
        III = self.have_fond_indices()
        listint = tuple(np.array([100000, 1000, 1]) + III)
        return listint.__hash__()

    def have_fond_indices(self):
        r"""
        return prime indices (fondamental direction)
        """
        interlist = [x for x in list(self.Millers) if x != 0]
        inter = [int(math.fabs(elem)) for elem in interlist]
        # print "interlist ",interlist
        # print "inter ", inter
        return np.array(self.Millers) / GT.pgcdl(inter)

# --- ---------------   PROCEDURES
def Quicklist(OrientMatrix, ReciprocBasisVectors, listRSnorm, lambdamin, verbose=0):
    r"""
    return 6 indices min and max boundary values for each Miller index h, k, l
    to be contained in the largest Ewald Sphere.

    :param OrientMatrix:  orientation matrix (3*3 matrix)
    :param ReciprocBasisVectors:      list of the three vectors a*,b*,c* in the lab frame
                            before rotation with OrientMatrix
    :param listRSnorm:      : list of the three reciprocal space lengthes of a*,b*,c*
    :param lambdamin:       : lambdamin (in Angstrom) corresponding to energy max

    :return: [[hmin,hmax],[kmin,kmax],[lmin,lmax]]
    """
    #     print "OrientMatrix in Quicklist", OrientMatrix

    assert lambdamin > 0, "lambdamin in Quicklist is not positive! %s" % str(lambdamin)

    if isinstance(OrientMatrix, list):
        OrientMatrix = np.array(OrientMatrix)

    if len(OrientMatrix) != 3:
        raise ValueError("Matrix is not 3*3 array !!!%s"%str(OrientMatrix))

    if 0. in listRSnorm or 0 in listRSnorm:
        raise ValueError("listRSnorm contains 0 !! %s"%str(listRSnorm))

    # lengthes of a*,b*,c*
    astarnorm, bstarnorm, cstarnorm = listRSnorm[:3]

    if verbose:
        print("norms of input a*,b*,c* : in Angstrom-1")
        print([astarnorm, bstarnorm, cstarnorm])

    # vec a*,b*,c* input
    vecastar, vecbstar, veccstar = np.array(ReciprocBasisVectors)

    # unit vectors
    # input vec a* normalized , ie vec a* direction
    vecastar_n = vecastar / math.sqrt(sum(vecastar ** 2))
    vecbstar_n = vecbstar / math.sqrt(sum(vecbstar ** 2))
    veccstar_n = veccstar / math.sqrt(sum(veccstar ** 2))

    # vec a* with appropriate length
    astar = vecastar_n * astarnorm
    bstar = vecbstar_n * bstarnorm
    cstar = veccstar_n * cstarnorm

    if verbose:
        print("(a*,b*,c*) in Angstrom-1")
        print(astar)
        print(bstar)
        print(cstar)

    normastar = math.sqrt(sum(astar ** 2))
    normbstar = math.sqrt(sum(bstar ** 2))
    normcstar = math.sqrt(sum(cstar ** 2))

    if verbose:
        print("norms of a*,b*,c* : (must be equal to input norms) in Angstrom-1")
        print(normastar, normbstar, normcstar)

    # rotation matrix or orientation matrix
    matrice_rot = np.array(OrientMatrix)

    if verbose:
        print(matrice_rot)

    rotAstar = np.dot(matrice_rot, astar)  # R(vec a* with appropriate length)
    rotBstar = np.dot(matrice_rot, bstar)
    rotCstar = np.dot(matrice_rot, cstar)

    if verbose:
        print("R(a*, b*, c*) in Angstrom-1")
        print(rotAstar)
        print(rotBstar)
        print(rotCstar)

    # transfer matrix from R(a*),R(b*),R(c*) to X, Y, Z
    matricerotvecstar = np.array([rotAstar, rotBstar, rotCstar]).transpose()
    # transfer matrix from X, Y, Z to R(a*),R(b*),R(c*)
    # inv_matricerotstar = scipy.linalg.basic.inv(matricerotvecstar)
    inv_matricerotstar = np.linalg.inv(matricerotvecstar)

    if verbose:
        print(" ------------------------")
        print("transfer matrix from R(a*),R(b*),R(c*) to X, Y, Z")
        print(matricerotvecstar)

        print(" ------------------------")
        print("transfer matrix from X, Y, Z to R(a*),R(b*),R(c*)")
        print(inv_matricerotstar)

        print("1 / lambdamin:", 1.0 / lambdamin, " Angstrom-1")

    OC = np.array([-1.0 / lambdamin, 0.0, 0.0])  # centre of the largest Ewald sphere in X, Y, Z frame

    OCrot = np.dot(inv_matricerotstar, OC)  # same centre in the R(a*),R(b*),R(c*) frame

    if verbose:
        print("center of largest Ewald Sphere in the R(a*),R(b*),R(c*) frame")
        print(OCrot)

        print("1 / lambdamin in corresponding R(a*),R(b*),R(c*) units")
        print([1.0 / lambdamin / normastar,
                1.0 / lambdamin / normbstar,
                1.0 / lambdamin / normcstar])

    # for non alpha = beta = gamma = 90 deg case
    # Calculate the crossproduct of rotated vector to correct properly the range on each rotated vector

    crossastar = np.cross(rotBstar, rotCstar) / math.sqrt(sum(np.cross(rotBstar, rotCstar) ** 2))  # normalised cross vector for reference
    crossbstar = np.cross(rotCstar, rotAstar) / math.sqrt(sum(np.cross(rotCstar, rotAstar) ** 2))
    crosscstar = np.cross(rotAstar, rotBstar) / math.sqrt(sum(np.cross(rotAstar, rotBstar) ** 2))

    if verbose:
        print("crossedvector", crossastar, crossbstar, crosscstar)

    cosanglea = np.dot(rotAstar / normastar, crossastar)
    cosangleb = np.dot(rotBstar / normbstar, crossbstar)
    cosanglec = np.dot(rotCstar / normcstar, crosscstar)

    if verbose:
        print("cosangle", [cosanglea, cosangleb, cosanglec])

    hmin = GT.properinteger(OCrot[0] - 1.0 / lambdamin / normastar / cosanglea)
    kmin = GT.properinteger(OCrot[1] - 1.0 / lambdamin / normbstar / cosangleb)
    lmin = GT.properinteger(OCrot[2] - 1.0 / lambdamin / normcstar / cosanglec)

    hmax = GT.properinteger(OCrot[0] + 1.0 / lambdamin / normastar / cosanglea)
    kmax = GT.properinteger(OCrot[1] + 1.0 / lambdamin / normbstar / cosangleb)
    lmax = GT.properinteger(OCrot[2] + 1.0 / lambdamin / normcstar / cosanglec)

    # SECURITY -1
    Hmin = min(hmin, hmax) - 1
    Kmin = min(kmin, kmax) - 1
    Lmin = min(lmin, lmax) - 1

    # SECURITY +1 and +1 because of exclusion convention for slicing in python
    # (superior limit in range is excluded)
    Hmax = max(hmin, hmax) + 2
    Kmax = max(kmin, kmax) + 2
    Lmax = max(lmin, lmax) + 2

    try:
        list_hkl_limits = [[int(Hmin), int(Hmax)],
                        [int(Kmin), int(Kmax)],
                        [int(Lmin), int(Lmax)]]
        return list_hkl_limits
    except ValueError:
        return None

def joel():
    print("hello joel")

def genHKL_np(listn, Extinc):
    r"""
    Generate all Miller indices hkl from indices limits given  by listn
    and taking into account for systematic exctinctions

    :param listn: Miller indices limits (warning: these lists are used in python range (last index is excluded))
    :type listn: [[hmin,hmax],[kmin,kmax],[lmin,lmax]]

    :param Extinc: label corresponding to systematic exctinction
        rules on h k and l miller indics such as ('fcc', 'bcc', 'dia', ...) or 'no' for any rules
    :type Extinc: string

    :return: array of [h,k,l]

    .. note:: node [0,0,0] is excluded
    """
    # print(Extinc)
    # print(listn)
    if listn is None:
        raise ValueError("hkl ranges are undefined")

    #    print "inside genHKL_np"
    if isinstance(listn, list) and len(listn) == 3:
        try:
            n_h_min, n_h_max = listn[0]
            n_k_min, n_k_max = listn[1]
            n_l_min, n_l_max = listn[2]
        except:
            raise TypeError("arg #1 has not the shape (3, 2)")
    else:
        raise TypeError("arg #1 is not a list or has not 3 elements")

    nbelements = (n_h_max - n_h_min) * (n_k_max - n_k_min) * (n_l_max - n_l_min)

    assert nbelements > 0

    if (not isinstance(nbelements, int)) or nbelements <= 0.0:
        raise ValueError("Needs (3,2) list of sorted integers")

    if Extinc not in list(dict_Extinc.values()):
        raise ValueError('Could not understand extinction code: " %s " ' % str(Extinc))

    HKLraw = np.mgrid[n_h_min:n_h_max, n_k_min:n_k_max, n_l_min:n_l_max]
    HKLs = HKLraw.shape
    #    print "HKLs", HKLs
    HKL = HKLraw.T.reshape((HKLs[1] * HKLs[2] * HKLs[3], HKLs[0]))

    return CP.ApplyExtinctionrules(HKL, Extinc)


# --- -----------------------  Main procedures
def parse_grainparameters(SingleCrystalParams):
    r"""
    extract and check type of the 4 elements of SingleCrystalParams

    .. todo:: to be used
    """
    try:
        Bmatrix, Extinc, Orientmatrix, key_for_dict = SingleCrystalParams
    except IndexError:
        raise IndexError("List of input parameters for crystal Laue pattern simulation "
                            "has %d elements instead of 4 !" % len(SingleCrystalParams))

    if Extinc not in list(dict_Extinc.values()):
        raise ValueError("wrong code %s for extinction !" % Extinc)
    if np.array(Bmatrix).shape != (3, 3):
        raise ValueError("wrong B matrix format!")
    if np.array(Orientmatrix).shape != (3, 3):
        raise ValueError("wrong Orientmatrix format!")
    return Bmatrix, Extinc, Orientmatrix, key_for_dict


def getLaueSpots(wavelmin, wavelmax, crystalsParams, kf_direction=DEFAULT_TOP_GEOMETRY,
                                            OpeningAngleCollection=22.0,
                                            fastcompute=0,
                                            ResolutionAngstrom=False,
                                            verbose=1,
                                            dictmaterials=None):
    r"""
    Compute Qxyz vectors and corresponding HKL miller indices for nodes in recicprocal space that can be measured
    for the given detection geometry and energy bandpass configuration.

    :param wavelmin:   smallest wavelength in Angstrom
    :param wavelmax:  largest wavelength in Angstrom

    :param crystalsParams: list of *SingleCrystalParams*, each of them being a list
        of 4 elements for crystal orientation and strain properties:

        * [0](array): is the B matrix a*,b*,c* vectors are expressed in column
            in LaueTools frame in reciprocal angstrom units

        * [1](str): peak Extinction rules ('no','fcc','dia', etc...)

        * [2](array): orientation matrix

        * [3](str): key for material element

    :param kf_direction: string defining the average geometry, mean value of exit scattered vector:
        'Z>0'   top spots

        'Y>0'   one side spots (towards hutch door)

        'Y<0'   other side spots

        'X>0'   transmission spots

        'X<0'   backreflection spots
    :param fastcompute:
        * 1, compute reciprocal space (RS) vector BUT NOT the Miller indices

        * 0, returns both RS vectors (normalised) and Miller indices

    :param ResolutionAngstrom:
        * scalar, smallest interplanar distance ordered in crystal in angstrom.

        * None, all reflections will be calculated that can be time-consuming for large unit cell

    :return:
        * list of [Qx,Qy,Qz]s for each grain, list of [H,K,L]s for each grain (fastcompute = 0)

        * list of [Qx,Qy,Qz]s for each grain, None  (fastcompute = 1)

    .. caution::
        This method doesn't create spot instances.

        This is done in filterLaueSpots with fastcompute = 0
    .. caution::
        finer selection of nodes : on camera , without harmonics can be
        done later with filterLaueSpots()

    .. note:: lauetools laboratory frame is in this case:
        x// ki (center of ewald sphere has negative x component)
        z perp to x and belonging to the plane defined by x and dd vectors
        (where dd vector is the smallest vector joining sample impact point and points on CCD plane)
        y is perpendicular to x and z

    """
    if isinstance(wavelmin, (float, int)) and isinstance(wavelmax, (float, int)):
        if wavelmin < wavelmax and wavelmin > 0:
            wlm = wavelmin * 1.0
            wlM = wavelmax * 1.0
        else:
            raise ValueError("wavelengthes must be positive and ordered")
    else:
        raise ValueError("wavelengthes must have numerical values")

    if not isinstance(crystalsParams, list):
        raise ValueError("Grains parameters list is not correct. It Must be: [grain] for 1 grain or [grain1, grain2] for 2 grains...")

    nb_of_grains = len(crystalsParams)

    if verbose:
        print("energy range: %.2f - %.2f keV" % (CST_ENERGYKEV / wlM, CST_ENERGYKEV / wlm))
        print("Number of grains: ", nb_of_grains)

    if dictmaterials is None:
        dictmaterials = dict_Materials

    # loop over grains
    for i in list(range(nb_of_grains)):
        try:
            key_material = dictmaterials[crystalsParams[i][3]][0]
        except (IndexError, TypeError, KeyError):
            smsg = "wrong type of input paramters: must be a list of 4 elements"
            raise ValueError(smsg)
        if verbose:
            print("# grain:  ", i, " made of ", key_material)
            print(" crystal parameters:", crystalsParams[i])

    wholelistvecfiltered = []
    wholelistindicesfiltered = []

    # calculation of RS lattice nodes in lauetools laboratory frame and indices

    # loop over grains
    for i in list(range(nb_of_grains)):

        (Bmatrix, Extinc,
        Orientmatrix, key_for_dict) = parse_grainparameters(crystalsParams[i])

        if verbose:
            print("\nin getLaueSpots()")
            print("Bmatrix", Bmatrix)
            print("Orientmatrix", Orientmatrix)
            print("key_for_dict", key_for_dict)
            print("Extinc", Extinc)
            print("")

        Bmatrix = np.array(Bmatrix)

        # generation of hkl nodes from Bmatrix

        listvecstarlength = np.sqrt(np.sum(Bmatrix ** 2, axis=0))

        # print "listvecstarlength",listvecstarlength

        # in Bmatrix.T,  a*, b* ,c* are rows of this argument
        #  B matrix in q= Orientmatrix B G formula
        # limitation of probed h k and l ranges
        list_hkl_limits = Quicklist(Orientmatrix, Bmatrix.T, listvecstarlength, wlm, verbose=0)

        # Loop over h k l
        # ----cython optimization
        global USE_CYTHON

        if not USE_CYTHON:
            table_vec = genHKL_np(list_hkl_limits, Extinc)

        Orientmatrix = np.array(Orientmatrix)

        listrotvec = np.dot(Orientmatrix, np.dot(Bmatrix, table_vec.T))

        listrotvec_X = listrotvec[0]
        listrotvec_Y = listrotvec[1]
        listrotvec_Z = listrotvec[2]

        arraysquare = listrotvec_X ** 2 + listrotvec_Y ** 2 + listrotvec_Z ** 2

        # removing all spots that have positive X component
        cond_Xnegativeonly = listrotvec_X < 0.
        listrotvec_X = listrotvec_X[cond_Xnegativeonly]
        listrotvec_Y = listrotvec_Y[cond_Xnegativeonly]
        listrotvec_Z = listrotvec_Z[cond_Xnegativeonly]
        arraysquare = arraysquare[cond_Xnegativeonly]

        table_vec = table_vec[cond_Xnegativeonly]

        if sys.version_info.major == 3:
            listObj = builtins.list
        else:
            listObj = list

        # Kf direction selection
        # top reflection 2theta = 90
        if kf_direction == "Z>0":
            KF_condit = listrotvec_Z > 0.0
        # side reflection  2theta = 90
        elif kf_direction == "Y>0":
            KF_condit = listrotvec_Y > 0
        # side reflection  2theta = 90
        elif kf_direction == "Y<0":
            KF_condit = listrotvec_Y < 0

        # x > -R transmission 2theta = 0
        elif kf_direction == "X>0":
            KF_condit = (listrotvec_X + 1.0 / (2.0 * np.abs(listrotvec_X) / arraysquare) > 0)

        # x < -R back reflection 2theta = 180
        elif kf_direction == "X<0":
            KF_condit = (listrotvec_X + 1.0 / (2.0 * np.abs(listrotvec_X) / arraysquare) < 0)
        #   all spots inside the two ewald's sphere with scattered beams in any direction'
        elif kf_direction == "4PI":
            KF_condit = np.ones_like(listrotvec_X) * True
        # user's definition of mean kf vector
        # [2theta , chi]= / kf = [cos2theta,sin2theta*sinchi,sin2theta*coschi]
        elif isinstance(kf_direction, (listObj, np.array)):
            print("\nUSING user defined LauePattern Region\n")
            if len(kf_direction) != 2:
                raise ValueError("kf_direction must be defined by a list of two angles !")
            else:
                kf_2theta, kf_chi = kf_direction
                kf_2theta, kf_chi = kf_2theta * DEG, kf_chi * DEG

                qmean_theta, qmean_chi = kf_2theta / 2.0, kf_chi

                print("central q angles", qmean_theta, qmean_chi)

                # q.qmean >0
                #    KF_condit = (-listrotvec_X * np.sin(qmean_theta) + \
                #                 listrotvec_Y * np.cos(qmean_theta) * np.sin(qmean_chi) + \
                #                 listrotvec_Z * np.cos(qmean_theta) * np.cos(qmean_chi)) > 0

                #    # angle(q, qmean) < 45 deg
                #     AngleMax = OpeningAngleCollection * DEG
                #   KF_condit = np.arccos(((-listrotvec_X * np.sin(qmean_theta) + \
                #                            listrotvec_Y * np.cos(qmean_theta) * np.sin(qmean_chi) + \
                #                            listrotvec_Z * np.cos(qmean_theta) * np.cos(qmean_chi))) / np.sqrt(arraysquare)) < AngleMax

                # angle(kf, kfmean) < 45 deg
                AngleMax = OpeningAngleCollection * DEG

                Rewald = arraysquare / (2 * np.fabs(listrotvec_X))
                kfsquare = Rewald ** 2
                KF_condit = (np.arccos((
                            (listrotvec_X + Rewald) * np.cos(kf_2theta)
                            + listrotvec_Y * np.sin(kf_2theta) * np.sin(kf_chi)
                            + listrotvec_Z * np.sin(kf_2theta) * np.cos(kf_chi)
                        ) / np.sqrt(kfsquare)) < AngleMax)
        else:
            raise ValueError("kf_direction '%s' is not understood!" % str(kf_direction))

        #        print "nb of KF_condit", len(np.where(KF_condit == True)[0])

        # vectors inside the two Ewald's spheres
        # last condition could be perhaps put before on
        # listrotvec_X[listrotvec_Z > 0]...
        Condit = np.logical_and(np.logical_and(
                                        ((listrotvec_X * 2.0 / wlm + arraysquare) <= 0.0),
                                        ((listrotvec_X * 2.0 / wlM + arraysquare) > 0.0),
                                    ), (KF_condit))
        #   print "nb of spots in spheres and in towards CCD region", len(np.where(Condit == True)[0])

        if fastcompute == 0:
            #             print 'Using detailled computation mode'

            if ResolutionAngstrom:
                # #crude resolution limitation 2Angstrom for lattice [20 4.8 49]
                # Hcond = np.abs(table_vec[:,0])<10
                # Kcond = np.abs(table_vec[:,1])<3
                # Lcond = np.abs(table_vec[:,2])<25
                # ConditResolution = np.logical_and(np.logical_and((Hcond) ,
                #                                                  (Kcond)),
                #                                                   (Lcond))
                ConditResolution = arraysquare < (1.0 / ResolutionAngstrom) ** 2

                Condit = np.logical_and(Condit, ConditResolution)

            # print "Condit",Condit
            fil_X = np.compress(Condit, listrotvec_X)
            fil_Y = np.compress(Condit, listrotvec_Y)
            fil_Z = np.compress(Condit, listrotvec_Z)

            # print "fil_Z",fil_Z
            fil_H = np.compress(Condit, table_vec[:, 0])
            fil_K = np.compress(Condit, table_vec[:, 1])
            fil_L = np.compress(Condit, table_vec[:, 2])

            listvecfiltered = np.transpose(np.array([fil_X, fil_Y, fil_Z]))
            listindicesfiltered = np.transpose(np.array([fil_H, fil_K, fil_L]))

            #  print "listindicesfiltered", len(listindicesfiltered)

            # if 0:
            # print "listrotvec[0]",listrotvec[0]
            # print "fgfghfg",fil_X
            # Energyarray = CST_ENERGYKEV*2*np.abs(listrotvec_X) / arraysquare
            # Efiltered = np.compress(Condit, Energyarray)
            # print "energyfiltered",Efiltered
            # print "listindicesfiltered",listindicesfiltered

            # to return
            wholelistvecfiltered.append(listvecfiltered)
            wholelistindicesfiltered.append(listindicesfiltered)

            if verbose:
                print("Orientation matrix", Orientmatrix)
                print("# grain: ", i)
                print("Number of spots for # grain ", i, ": ", len(listvecfiltered))

        # only q vectors are calculated, not Miller indices
        elif fastcompute == 1:

            if ResolutionAngstrom:

                # #crude resolution limitation 2Angstrom for lattice [20 4.8 49]
                # Hcond = np.abs(table_vec[:,0])<10
                # Kcond = np.abs(table_vec[:,1])<3
                # Lcond = np.abs(table_vec[:,2])<25
                # ConditResolution = np.logical_and(np.logical_and((Hcond) ,
                #                                                  (Kcond)),
                #                                                   (Lcond))

                ConditResolution = arraysquare < (1.0 / ResolutionAngstrom) ** 2

                Condit = np.logical_and(Condit, ConditResolution)

            fil_X = np.compress(Condit, listrotvec_X)
            fil_Y = np.compress(Condit, listrotvec_Y)
            fil_Z = np.compress(Condit, listrotvec_Z)

            # to return
            listvecfiltered = np.transpose(np.array([fil_X, fil_Y, fil_Z]))
            wholelistvecfiltered.append(listvecfiltered)
            # and wholelistindicesfiltered (hkl) is empty

            if verbose:
                print("Orientation matrix", Orientmatrix)
                print("# grain: ", i)
                print("Rotating all spots")
                print("Nb of spots for # grain  ", i, ": ", len(listvecfiltered))

    return wholelistvecfiltered, wholelistindicesfiltered


def create_spot(pos_vec, miller, detectordistance, allattributes=False, pixelsize=165.0 / 2048,
                                                                                dim=(2048, 2048)):
    r""" From reciprocal space position and 3 miller indices
    create a spot instance (on top camera geometry)

    :param pos_vec: 3D vector
    :type pos_vec: list of 3 float
    :param  miller: list of 3 miller indices
    :param detectordistance: approximate distance detector sample (to compute complementary spots attributes)
    :param  allattributes: False or 0  not to compute complementary spot attributes
    :param  allattributes: boolean

    :return: spot instance

    .. note:: spot.Qxyz is a vector expressed in lauetools frame

    X along x-ray and Z towards CCD when CCD on top and y towards experimental hutch door
    """
    spotty = spot(miller)
    spotty.Qxyz = pos_vec
    vecres = np.array(pos_vec)
    spotty.EwaldRadius = np.dot(vecres, vecres) / (2.0 * math.fabs(vecres[0]))
    if spotty.Qxyz[2] > 0:
        normkout = math.sqrt((spotty.Qxyz[0] + spotty.EwaldRadius) ** 2
            + spotty.Qxyz[1] ** 2
            + spotty.Qxyz[2] ** 2)

        if allattributes not in (False, 0):
            X = (detectordistance
                * (spotty.Qxyz[0] + spotty.EwaldRadius)
                / spotty.Qxyz[2])
            Y = detectordistance * (spotty.Qxyz[1]) / spotty.Qxyz[2]
            spotty.Xcam = -X / pixelsize + dim[0] / 2
            spotty.Ycam = -Y / pixelsize + dim[1] / 2

        spotty.Twicetheta = (math.acos((spotty.Qxyz[0] + spotty.EwaldRadius) / normkout) / DEG)
        #        spotty.Chi = math.atan(spotty.Qxyz[1] * 1. / spotty.Qxyz[2]) / DEG
        spotty.Chi = math.atan2(spotty.Qxyz[1] * 1.0, spotty.Qxyz[2]) / DEG

    return spotty


def create_spot_np(Qxyz, miller, detectordistance, allattributes=False,
                                                    pixelsize=165.0 / 2048, dim=(2048, 2048)):
    r""" From reciprocal space position and 3 miller indices
    create a spot instance (on top camera geometry)

    :param pos_vec: 3D vector
    :type pos_vec: list of 3 float
    :param  miller: list of 3 miller indices
    :param detectordistance: approximate distance detector sample (to compute complementary spots attributes)
    :param  allattributes: False or 0  not to compute complementary spot attributes
    :param  allattributes: boolean

    :return: spot instance

    .. note:: spot.Qxyz is a vector expressed in lauetools frame

    X along x-ray and Z towards CCD when CCD on top and y towards experimental hutch door
    """
    #     print "Qxyz", Qxyz
    EwaldRadius = np.sum(Qxyz ** 2, axis=1) / (2.0 * np.abs(Qxyz[:, 0]))
    ki = np.zeros((len(Qxyz), 3))
    ki[:, 0] = EwaldRadius

    kout = ki + Qxyz
    #     if spotty.Qxyz[2] > 0:
    normkout = np.sqrt(np.sum(kout ** 2, axis=1))

    Twicetheta = np.arccos((kout[:, 0]) / normkout) / DEG
    Chi = np.arctan2(kout[:, 1] * 1.0, kout[:, 2]) / DEG

    Energy = EwaldRadius * CST_ENERGYKEV

    return Twicetheta, Chi, Energy, miller


def create_spot_4pi(pos_vec, miller, detectordistance, allattributes=0, pixelsize=165.0 / 2048,
                                                                                dim=(2048, 2048)):
    r""" From reciprocal space position and 3 miller indices
    create a spot scattered in 4pi steradian no camera position

    .. note:: spot.Qxyz is a vector expressed in lauetools frame

    X along x-ray and Z towards CCD when CCD on top and y towards experimental hutch door
    """
    spotty = spot(miller)
    spotty.Qxyz = pos_vec
    vecres = np.array(pos_vec)
    spotty.EwaldRadius = np.dot(vecres, vecres) / (2.0 * math.fabs(vecres[0]))

    normkout = math.sqrt((spotty.Qxyz[0] + spotty.EwaldRadius) ** 2
        + spotty.Qxyz[1] ** 2
        + spotty.Qxyz[2] ** 2)

    spotty.Twicetheta = (math.acos((spotty.Qxyz[0] + spotty.EwaldRadius) / normkout) / DEG)
    #    spotty.Chi = math.atan(spotty.Qxyz[1] * 1. / spotty.Qxyz[2]) / DEG
    spotty.Chi = math.atan2(spotty.Qxyz[1] * 1.0, spotty.Qxyz[2]) / DEG

    return spotty


def create_spot_side_pos(pos_vec, miller, detectordistance,
                                        allattributes=0,
                                        pixelsize=165.0 / 2048,
                                        dim=(2048, 2048)):
    r""" From reciprocal space position and 3 miller indices
    create a spot on side camera
    """
    spotty = spot(miller)
    spotty.Qxyz = pos_vec
    vecres = np.array(pos_vec)
    spotty.EwaldRadius = np.dot(vecres, vecres) / (2.0 * math.fabs(vecres[0]))
    if spotty.Qxyz[1] > 0:
        normkout = math.sqrt(
            (spotty.Qxyz[0] + spotty.EwaldRadius) ** 2
            + spotty.Qxyz[1] ** 2
            + spotty.Qxyz[2] ** 2)

        if not allattributes:
            X = (detectordistance * (spotty.Qxyz[0] + spotty.EwaldRadius) / spotty.Qxyz[1])
            Y = detectordistance * (spotty.Qxyz[2]) / spotty.Qxyz[1]
            #             spotty.Xcam = X / pixelsize + dim[0] / 2
            #             spotty.Ycam = Y / pixelsize + dim[1] / 2
            spotty.Xcam = X / pixelsize
            spotty.Ycam = Y / pixelsize

        spotty.Twicetheta = (math.acos((spotty.Qxyz[0] + spotty.EwaldRadius) / normkout) / DEG)
        # spotty.Chi = math.atan(spotty.Qxyz[1]*1. / spotty.Qxyz[2])/DEG
        spotty.Chi = math.atan2(spotty.Qxyz[1] * 1.0, spotty.Qxyz[2]) / DEG

    return spotty


def create_spot_side_neg(pos_vec, miller, detectordistance, allattributes=0, pixelsize=165.0 / 2048,
                                            dim=(2048, 2048)):
    r""" From reciprocal space position and 3 miller indices
    create a spot on neg side camera

    .. todo:: Update with dim as other create_spot()
    """
    spotty = spot(miller)
    spotty.Qxyz = pos_vec
    vecres = np.array(pos_vec)
    spotty.EwaldRadius = np.dot(vecres, vecres) / (2.0 * math.fabs(vecres[0]))

    if spotty.Qxyz[1] < 0.0:
        normkout = math.sqrt(
            (spotty.Qxyz[0] + spotty.EwaldRadius) ** 2
            + spotty.Qxyz[1] ** 2
            + spotty.Qxyz[2] ** 2)

        if not allattributes:
            X = (-detectordistance * (spotty.Qxyz[0] + spotty.EwaldRadius) / spotty.Qxyz[1])
            Y = detectordistance * (spotty.Qxyz[2]) / spotty.Qxyz[1]
            spotty.Xcam = X / pixelsize + dim[0] / 2.0
            spotty.Ycam = Y / pixelsize + dim[1] / 2.0

        spotty.Twicetheta = (math.acos((spotty.Qxyz[0] + spotty.EwaldRadius) / normkout) / DEG)
        # spotty.Chi = math.atan(spotty.Qxyz[1]*1. / spotty.Qxyz[2])/DEG
        spotty.Chi = math.atan2(spotty.Qxyz[1] * 1.0, spotty.Qxyz[2]) / DEG

    return spotty


def create_spot_front(pos_vec, miller, detectordistance, allattributes=0,
                                                        pixelsize=165.0 / 2048,
                                                        dim=(2048, 2048)):
    r""" From reciprocal space position and 3 miller indices
    create a spot on forward direction transmission geometry
    """
    #     print "use create_spot_front"
    spotty = spot(miller)
    spotty.Qxyz = pos_vec
    vecres = np.array(pos_vec)
    spotty.EwaldRadius = np.dot(vecres, vecres) / (2.0 * math.fabs(vecres[0]))

    if miller[0] == 0 and miller[1] == 0 and miller[2] == 0:
        # print("MILLER == MILER ==0")
        return None

    if spotty.Qxyz[0] < 0.0:
        #print('good reflection for transmission')
        abskx = math.fabs(spotty.Qxyz[0] + spotty.EwaldRadius)
        normkout = 1.0 * math.sqrt(abskx ** 2 + spotty.Qxyz[1] ** 2 + spotty.Qxyz[2] ** 2)

        # if qx < -R   no spot in transmission mode (qx is <0)
        if spotty.Qxyz[0] + spotty.EwaldRadius <= 0.:
            return None

        X = -detectordistance * (spotty.Qxyz[1]) / abskx
        Y = -detectordistance * (spotty.Qxyz[2]) / abskx
        spotty.Xcam = X / pixelsize + dim[0] / 2.0
        spotty.Ycam = Y / pixelsize + dim[1] / 2.0

        spotty.Twicetheta = (math.acos((spotty.Qxyz[0] + spotty.EwaldRadius) / normkout) / DEG)
        # spotty.Chi = math.atan(spotty.Qxyz[1]*1. / spotty.Qxyz[2])/DEG
        spotty.Chi = math.atan2(spotty.Qxyz[1] * 1.0, spotty.Qxyz[2]) / DEG

        return spotty

    else:
        return None


def create_spot_back(pos_vec, miller, detectordistance, allattributes=0,
                                                        pixelsize=165.0 / 2048, dim=(2048, 2048)):
    r""" From reciprocal space position and 3 miller indices
    create a spot on backward directions i.e.  back reflection geometry
    """
    spotty = spot(miller)
    spotty.Qxyz = pos_vec
    vecres = np.array(pos_vec)
    spotty.EwaldRadius = np.dot(vecres, vecres) / (2.0 * math.fabs(vecres[0]))
    if spotty.Qxyz[0] < 0.0:
        abskx = math.fabs(spotty.Qxyz[0] + spotty.EwaldRadius)
        normkout = math.sqrt(abskx ** 2 + spotty.Qxyz[1] ** 2 + spotty.Qxyz[2] ** 2)

        if not allattributes:
            X = detectordistance * (spotty.Qxyz[1]) / abskx
            Y = detectordistance * (spotty.Qxyz[2]) / abskx
            spotty.Xcam = X / pixelsize + dim[0] / 2.0
            spotty.Ycam = Y / pixelsize + dim[1] / 2.0

        spotty.Twicetheta = (math.acos((spotty.Qxyz[0] + spotty.EwaldRadius) / normkout) / DEG)
        # spotty.Chi = math.atan(spotty.Qxyz[1]*1. / spotty.Qxyz[2])/DEG
        spotty.Chi = math.atan2(spotty.Qxyz[1] * 1.0, spotty.Qxyz[2]) / DEG

    return spotty

def filterQandHKLvectors(vec_and_indices, detectordistance, detectordiameter, kf_direction='Z>0'):
    """filter vector Q and HKL in vec_and_indices

    :param vec_and_indices: arrays of vectors Q and HKL
    :type vec_and_indices: [Qs, HKLs]
    :param detectordistance: distance detector sample (mm)
    :type detectordistance: float
    :param detectordiameter: detector diameter (mm)
    :type detectordiameter: float
    :param kf_direction: geometry of detection label or two angles giving 2theta chi direction of detector, defaults to 'Z>0'
    :type kf_direction: str or 2 floats, optional
    :raises ValueError: [description]
    :raises ValueError: [description]
    :return: [oncam_vec], [oncam_HKL]
    :rtype: [np.array of 3 elements, np.array of 3 elements]
    """
    Qvectors_list, HKLs_list = vec_and_indices
    Qx = Qvectors_list[0][:, 0] * 1.0
    Qy = Qvectors_list[0][:, 1] * 1.0
    Qz = Qvectors_list[0][:, 2] * 1.0

    indi_H = HKLs_list[0][:, 0]
    indi_K = HKLs_list[0][:, 1]
    indi_L = HKLs_list[0][:, 2]

    Qsquare = Qx ** 2 + Qy ** 2 + Qz ** 2

    # (proportional to photons Energy)
    Rewald = Qsquare / 2.0 / np.abs(Qx)

    # Kf direction selection
    if kf_direction == "Z>0":  # top reflection geometry
        ratiod = detectordistance / Qz
        Ycam = ratiod * (Qx + Rewald)
        Xcam = ratiod * (Qy)
    elif kf_direction == "Y>0":  # side reflection geometry (for detector between the GMT hutch door and the sample (beam coming from right to left)
        ratiod = detectordistance / Qy
        Xcam = ratiod * (Qx + Rewald)
        Ycam = ratiod * (Qz)
    elif kf_direction == "Y<0":  # other side reflection
        ratiod = detectordistance / np.abs(Qy)
        Xcam = -ratiod * (Qx + Rewald)
        Ycam = ratiod * (Qz)
    elif kf_direction == "X>0":  # transmission geometry
        ratiod = detectordistance / np.abs(Qx + Rewald)
        Xcam = -1.0 * ratiod * (Qy)
        Ycam = -ratiod * (Qz)
    elif kf_direction == "X<0":  # back reflection geometry
        ratiod = detectordistance / np.abs(Qx + Rewald)
        Xcam = ratiod * (Qy)
        Ycam = ratiod * (Qz)
    elif kf_direction == "4PI":  # to keep all scattered spots
        Xcam = np.zeros_like(Qy)
        Ycam = np.zeros_like(Qy)

    elif isinstance(kf_direction, (list, np.array)):
        if len(kf_direction) != 2:
            raise ValueError(
                "kf_direction must be defined by a list of two angles !")
        else:
            Xcam = np.zeros_like(Qy)
            Ycam = np.zeros_like(Qy)
    else:
        raise ValueError("Unknown laue geometry code for kf_direction parameter")

    # print "Xcam, Ycam",Xcam
    # print Ycam
    # print "******************"
    # On camera filter
    # print "detectordiameter", detectordiameter
    halfCamdiametersquare = (detectordiameter / 2.0) ** 2
    # TODO: should contain Xcam-Xcamcen (mm) and Ycam-Ycamcen with Xcamcen, Ycamcen
    # given by user
    onCam_cond = Xcam ** 2 + Ycam ** 2 <= halfCamdiametersquare
    # resulting arrays
    oncam_Qx = np.compress(onCam_cond, Qx)
    oncam_Qy = np.compress(onCam_cond, Qy)
    oncam_Qz = np.compress(onCam_cond, Qz)

    oncam_vec = np.array([oncam_Qx, oncam_Qy, oncam_Qz]).T

    oncam_H = np.compress(onCam_cond, indi_H)
    oncam_K = np.compress(onCam_cond, indi_K)
    oncam_L = np.compress(onCam_cond, indi_L)
    oncam_HKL = np.transpose(np.array([oncam_H, oncam_K, oncam_L]))

    return [oncam_vec], [oncam_HKL]


def filterLaueSpots(vec_and_indices, HarmonicsRemoval=1,
                                    fastcompute=0,
                                    kf_direction=DEFAULT_TOP_GEOMETRY,
                                    fileOK=0,
                                    detectordistance=DEFAULT_DETECTOR_DISTANCE,
                                    detectordiameter=DEFAULT_DETECTOR_DIAMETER,
                                    pixelsize=165.0 / 2048,
                                    dim=(2048, 2048),
                                    linestowrite=[[""]],
                                    verbose=0):
    r""" Calculates list of grains spots on camera and without harmonics
    and on CCD camera from [[spots grain 0],[spots grain 1],etc] =>
    returns [[spots grain 0],[spots grain 1],etc] w / o harmonics and on camera  CCD

    :param vec_and_indices: list of elements corresponding to 1 grain, each element is composed by
        * [0] array of vector

        * [1] array of indices

    :param HarmonicsRemoval: 1, removes harmonics according to their miller indices
        (only for fastcompute = 0)

    :param fastcompute:
        * 1, outputs a list for each grain of 2theta spots and a list for each grain of chi spots
            (HARMONICS spots are still HERE!)
        * 0, outputs list for each grain of spots with

    :param kf_direction: label for detection geometry (CCD plane with respect to the incoming beam and sample)
    :type kf_direction: string

    :return:
        * list of spot instances if fastcompute=0

        * 2theta, chi          if fastcompute=1

    .. note::
        * USED IMPORTANTLY in lauecore.SimulateResults  lauecore.SimulateLaue
        * USED in matchingrate.AngularResidues
        * USED in ParametricLaueSimulator.dosimulation_parametric
        * USED in AutoindexationGUI.OnSimulate_S3, DetectorCalibration.Reckon_2pts, and others

    .. todo::
        add dim in create_spot in various geometries
    """
    # print("filterLaueSpots !!!!!")
    try:
        Qvectors_list, HKLs_list = vec_and_indices
    except ValueError:
        raise ValueError("vec_and_indices has not two elements!")

    try:
        nbofgrains = len(Qvectors_list)
    except TypeError:
        raise TypeError("Qvectors_list is not a iterable !")

    if not isinstance(Qvectors_list[0], np.ndarray):
        raise TypeError("first element of Qvectors_list is not surprisingly an array")

    # preparing list of results
    if fastcompute == 0:
        ListSpots_Oncam_wo_harmonics = emptylists(nbofgrains)
    elif fastcompute == 1:
        Oncam2theta = emptylists(nbofgrains)
        Oncamchi = emptylists(nbofgrains)

    # loop over grains
    totalnbspots = 0
    for grainindex in list(range(nbofgrains)):
        try:
            Qx = Qvectors_list[grainindex][:, 0] * 1.0
            Qy = Qvectors_list[grainindex][:, 1] * 1.0
            Qz = Qvectors_list[grainindex][:, 2] * 1.0
            if np.shape(Qvectors_list[grainindex])[1] != 3:
                raise TypeError
        except IndexError:
            raise IndexError("vec_and_indices has not the proper shape")

        Qsquare = Qx ** 2 + Qy ** 2 + Qz ** 2

        # corresponding Ewald sphere radius for each spots
        # (proportional to photons Energy)
        Rewald = Qsquare / 2.0 / np.abs(Qx)

        # Kf direction selection
        if kf_direction == "Z>0":  # top reflection geometry
            ratiod = detectordistance / Qz
            Ycam = ratiod * (Qx + Rewald)
            Xcam = ratiod * (Qy)
        elif kf_direction == "Y>0":  # side reflection geometry (for detector between the GMT hutch door and the sample (beam coming from right to left)
            ratiod = detectordistance / Qy
            Xcam = ratiod * (Qx + Rewald)
            Ycam = ratiod * (Qz)

        elif kf_direction == "Y<0":  # other side reflection
            ratiod = detectordistance / np.abs(Qy)
            Xcam = -ratiod * (Qx + Rewald)
            Ycam = ratiod * (Qz)
        elif kf_direction == "X>0":  # transmission geometry
            ratiod = detectordistance / np.abs(Qx + Rewald)
            Xcam = -1.0 * ratiod * (Qy)
            Ycam = -ratiod * (Qz)
        elif kf_direction == "X<0":  # back reflection geometry
            ratiod = detectordistance / np.abs(Qx + Rewald)
            Xcam = ratiod * (Qy)
            Ycam = ratiod * (Qz)
        elif kf_direction == "4PI":  # to keep all scattered spots
            Xcam = np.zeros_like(Qy)
            Ycam = np.zeros_like(Qy)

        elif isinstance(kf_direction, (list, np.array)):
            if len(kf_direction) != 2:
                raise ValueError(
                    "kf_direction must be defined by a list of two angles !")
            else:
                Xcam = np.zeros_like(Qy)
                Ycam = np.zeros_like(Qy)
        else:
            raise ValueError("Unknown laue geometry code for kf_direction parameter")

        #print("Xcam, Ycam",Xcam,Ycam)
        # print Ycam
        # print "******************"
        # On camera filter
        # print "detectordiameter", detectordiameter
        halfCamdiametersquare = (detectordiameter / 2.0) ** 2
        # TODO: should contain Xcam-Xcamcen (mm) and Ycam-Ycamcen with Xcamcen, Ycamcen
        # given by user
        onCam_cond = Xcam ** 2 + Ycam ** 2 <= halfCamdiametersquare

        #print('onCam_cond',onCam_cond)
        #print('onCam_cond   true',np.where(onCam_cond==True))
        # resulting arrays
        oncam_Qx = np.compress(onCam_cond, Qx)
        oncam_Qy = np.compress(onCam_cond, Qy)
        oncam_Qz = np.compress(onCam_cond, Qz)

        oncam_R = np.compress(onCam_cond, Rewald)
        oncam_XplusR = oncam_Qx + oncam_R

        # compute 2theta, chi  (in radians)
        oncam_2theta = np.arccos(oncam_XplusR / np.sqrt(oncam_XplusR ** 2 + oncam_Qy ** 2 + oncam_Qz ** 2))

        #print('oncam_2theta',oncam_2theta)
        # be careful of the of sign
        oncam_chi = np.arctan(1.0 * oncam_Qy / oncam_Qz)
        #         oncam_chi = np.arctan2(1. * oncam_Qy, oncam_Qz)
        # TODO: replace by arctan2(1. * oncam_Qy ,oncam_Qz) ??

        #        print "oncam_2theta", oncam_2theta
        #        print "oncam_chi", oncam_chi
        #        print "len(oncam)", len(oncam_2theta)

        # creates spot instances which takes some times...
        # will compute spots on the camera
        # (if HarmonicsRemoval = 0 harmonics still exist)
        # (if HarmonicsRemoval = 1 harmonics are removed, fundamentals are kept)
        # (those of lowest energy)
        if fastcompute == 0:
            try:
                indi_H = HKLs_list[grainindex][:, 0]
                indi_K = HKLs_list[grainindex][:, 1]
                indi_L = HKLs_list[grainindex][:, 2]
                if np.shape(HKLs_list[grainindex])[1] != 3:
                    raise TypeError
            except TypeError:
                emsg = "HKLs_list must be a list of array of 3D vectors"
                raise TypeError(emsg)

            oncam_vec = np.array([oncam_Qx, oncam_Qy, oncam_Qz]).T

            # print('onCam_cond', onCam_cond)
            oncam_H = np.compress(onCam_cond, indi_H)
            oncam_K = np.compress(onCam_cond, indi_K)
            oncam_L = np.compress(onCam_cond, indi_L)
            oncam_HKL = np.transpose(np.array([oncam_H, oncam_K, oncam_L]))

            # build list of spot objects
            listspot = get2ThetaChi_geometry(oncam_vec,
                                            oncam_HKL,
                                            detectordistance=detectordistance,
                                            pixelsize=pixelsize,
                                            dim=dim,
                                            kf_direction=kf_direction)
            #print("listspot", listspot)
            # print("oncam_HKL", oncam_HKL.tolist())
            # Creating list of spot with or without harmonics
            if HarmonicsRemoval and listspot:
                # ListSpots_Oncam_wo_harmonics[grainindex] = RemoveHarmonics(listspot)
                # (oncam_HKL_filtered, toremove)
                (_, toremove) = CP.FilterHarmonics_2(oncam_HKL, return_indices_toremove=1)
                #print('toremove',toremove)
                listspot = np.delete(np.array(listspot), toremove).tolist()

            # feeding final list of spots
            ListSpots_Oncam_wo_harmonics[grainindex] = listspot

            if listspot is not None:
                totalnbspots += len(listspot)

            if fileOK:
                IOLT.Writefile_data_log(ListSpots_Oncam_wo_harmonics[grainindex],
                                        grainindex,
                                        linestowrite=linestowrite)

            # print "Number of spot in camera w / o harmonics",len(ListSpots_Oncam_wo_harmonics[grainindex])

        # (fastcompute = 1) no instantiation of spot object
        # will return 2theta, chi for each grain
        # no harmonics removal
        elif fastcompute == 1:
            Oncam2theta[grainindex] = oncam_2theta
            Oncamchi[grainindex] = oncam_chi

            totalnbspots += len(oncam_2theta)

    if verbose: print('total number of spots for all the %d grain(s) in filterLaueSpots():  '%nbofgrains, totalnbspots)
    if totalnbspots == 0:
        return
    # outputs and returns
    if fileOK:
        linestowrite.append(["\n"])
        linestowrite.append(["--------------------------------------------------------"])
        linestowrite.append(["------------- Simulation Data --------------------------"])
        linestowrite.append(["--------------------------------------------------------"])
        linestowrite.append(["\n"])
        linestowrite.append(["#grain, h, k, l, energy(keV), 2theta (deg), chi (deg), X_Xmas, Y_Xmas, X_JSM, Y_JSM, Xtest, Ytest"])
    if fastcompute == 0:
        # list of elements which are list of objects of spot class (1 element / grain)
        return ListSpots_Oncam_wo_harmonics
    elif fastcompute == 1:
        # list of elements which are composed by two arrays (2theta, chi) (1 element / grain)
        return (np.concatenate(Oncam2theta) / DEG, np.concatenate(Oncamchi) / DEG)


def filterLaueSpots_full_np(veccoord, indicemiller, onlyXYZ=False, HarmonicsRemoval=1,
                                                        fastcompute=0,
                                                        kf_direction=DEFAULT_TOP_GEOMETRY,
                                                        detectordistance=DEFAULT_DETECTOR_DISTANCE,
                                                        detectordiameter=DEFAULT_DETECTOR_DIAMETER,
                                                        pixelsize=165.0 / 2048,
                                                        dim=(2048, 2048),
                                                        grainindex=0):
    r""" Calculates spots data for an individual grain
    on camera and without harmonics
    and on CCD camera

    :param veccoord : list of elements corresponding to 1 grain,
                    each element is composed by: array of q vector [Qx,Qy,Qz]
    :param indicemiller : list of Miller indices [h,k,l] array of indices

    :param HarmonicsRemoval: 1 removes harmonics according to their miller indices
                            (only for fastcompute = 0)

    :param fastcompute:  1 output a list for each grain of 2theta spots and a list for each grain of chi spots
                            (HARMONICS spots are still HERE!)
                           0 output list for each grain of spots with


    :param kf_direction: label for detection geometry
                    (CCD plane with respect to
                    the incoming beam and sample)
    :type kf_direction: string

    :return: tuple of lists of Twtheta Chi Energy Millers if fastcompute=0

                tuple of lists 2theta, chi          if fastcompute=1

    .. note::
        USED in detectorCalibration...simulate_theo
        and tentatively on matching rate)
    """
    VecX = veccoord[:, 0] * 1.0
    VecY = veccoord[:, 1] * 1.0
    VecZ = veccoord[:, 2] * 1.0

    Vecsquare = VecX ** 2 + VecY ** 2 + VecZ ** 2

    # correspondinf Ewald sphere radius for each spots
    # (proportional to photons Energy)
    Rewald = Vecsquare / 2.0 / np.abs(VecX)

    # Kf direction selection
    if kf_direction == "Z>0":  # top reflection geometry
        # VecZ is >0
        ratiod = detectordistance / VecZ
        Ycam = ratiod * (VecX + Rewald)
        Xcam = ratiod * (VecY)
    elif kf_direction == "Y>0":  # side reflection geometry (for detector between the GMT hutch door and the sample (beam coming from right to left)
        ratiod = detectordistance / VecY
        Xcam = ratiod * (VecX + Rewald)
        Ycam = ratiod * (VecZ)
    elif kf_direction == "Y<0":  # other side reflection
        ratiod = detectordistance / np.abs(VecY)
        Xcam = -ratiod * (VecX + Rewald)
        Ycam = ratiod * (VecZ)
    elif kf_direction == "X>0":  # transmission geometry
        ratiod = detectordistance / np.abs(VecX + Rewald)
        Xcam = -1.0 * ratiod * (VecY)
        Ycam = -ratiod * (VecZ)
    elif kf_direction == "X<0":  # back reflection geometry
        # pos0 = np.where(np.abs(VecX + Rewald) < 0.0000000001)
        # print('pos0', pos0)
        # print(VecX[pos0])
        # print(Rewald[pos0])
        # if VecX +Rewald = 0 then kf is // z axis
        # peaks are likely not to be intercepted bt detector plane
        # except if plane is
        ratiod = detectordistance / np.abs(VecX + Rewald)
        Xcam = ratiod * (VecY)
        Ycam = ratiod * (VecZ)
    elif kf_direction == "4PI":  # to keep all scattered spots
        Xcam = np.zeros_like(VecY)
        Ycam = np.zeros_like(VecY)

    elif isinstance(kf_direction, (list, np.array)):
        if len(kf_direction) != 2:
            raise ValueError("kf_direction must be defined by a list of two angles !")
        else:
            Xcam = np.zeros_like(VecY)
            Ycam = np.zeros_like(VecY)
    else:
        raise ValueError("Unknown laue geometry code for kf_direction parameter")

    # print "Xcam, Ycam",Xcam
    # print Ycam
    # ----------------------------------------------------------
    # Approximate selection of spots in camera accroding to Xcam and Ycam
    # onCam_cond  conditions
    # ------------------------------------------------------------

    # print "******************"
    # On camera filter
    # print "detectordiameter", detectordiameter
    halfCamdiametersquare = (detectordiameter / 2.0) ** 2
    # TODO: should contain Xcam-Xcamcen (mm) and Ycam-Ycamcen with Xcamcen, Ycamcen
    # given by user
    onCam_cond = Xcam ** 2 + Ycam ** 2 <= halfCamdiametersquare
    # resulting arrays
    oncam_vecX = np.compress(onCam_cond, VecX)
    oncam_vecY = np.compress(onCam_cond, VecY)
    oncam_vecZ = np.compress(onCam_cond, VecZ)

    if onlyXYZ:
        return np.array([oncam_vecX, oncam_vecY, oncam_vecZ]).T

    # creates spot instances which takes some times...
    # will compute spots on the camera
    # (if HarmonicsRemoval = 0 harmonics still exist)
    # (if HarmonicsRemoval = 1 harmonics are removed, fundamentals are kept)
    # (those of lowest energy)
    if fastcompute == 0:

        indi_H = indicemiller[:, 0]
        indi_K = indicemiller[:, 1]
        indi_L = indicemiller[:, 2]

        oncam_vec = np.array([oncam_vecX, oncam_vecY, oncam_vecZ]).T

        oncam_H = np.compress(onCam_cond, indi_H)
        oncam_K = np.compress(onCam_cond, indi_K)
        oncam_L = np.compress(onCam_cond, indi_L)
        oncam_HKL = np.transpose(np.array([oncam_H, oncam_K, oncam_L]))

        # build list of spot objects
        TwthetaChiEnergyMillers_list_one_grain = get2ThetaChi_geometry_full_np(
                                                                oncam_vec,
                                                                oncam_HKL,
                                                                detectordistance=detectordistance,
                                                                pixelsize=pixelsize,
                                                                dim=dim,
                                                                kf_direction=kf_direction)

        #         print 'TwthetaChiEnergy_list_one_grain', TwthetaChiEnergy_list_one_grain
        # Creating list of spot with or without harmonics
        if HarmonicsRemoval and True:  # listspot:
            raise ValueError("Harmonic removal is not implemented and listspot does not exist anymore")
            #                ListSpots_Oncam_wo_harmonics[i] = RemoveHarmonics(listspot)
            # (oncam_HKL_filtered, toremove) = CP.FilterHarmonics_2(
            #     oncam_HKL, return_indices_toremove=1
            # )
            # listspot = np.delete(np.array(listspot), toremove).tolist()

        # print "Number of spot in camera w / o harmonics",len(ListSpots_Oncam_wo_harmonics[i])

    # (fastcompute = 1) no instantiation of spot object
    # will return 2theta, chi for each grain
    # no harmonics removal
    elif fastcompute == 1:

        oncam_R = np.compress(onCam_cond, Rewald)
        oncam_XplusR = oncam_vecX + oncam_R
        # compute 2theta, chi
        oncam_2theta = (
            np.arccos(oncam_XplusR / np.sqrt(oncam_XplusR ** 2 + oncam_vecY ** 2 + oncam_vecZ ** 2))
            / DEG)
        # be careful of the of sign
        oncam_chi = np.arctan(1.0 * oncam_vecY / oncam_vecZ) / DEG
        #         oncam_chi = np.arctan2(1. * oncam_vecY, oncam_vecZ)
        # TODO: replace by arctan2(1. * oncam_vecY ,oncam_vecZ) ??

    if fastcompute == 0:
        # list spots data
        return TwthetaChiEnergyMillers_list_one_grain
    elif fastcompute == 1:
        # list of elements which are composed by two arrays (2theta, chi) (1 element / grain)
        return oncam_2theta, oncam_chi


def get2ThetaChi_geometry(oncam_vec, oncam_HKL, detectordistance=DEFAULT_DETECTOR_DISTANCE,
                                                                pixelsize=165.0 / 2048,
                                                                dim=(2048, 2048),
                                                                kf_direction=DEFAULT_TOP_GEOMETRY):
    r"""
    computes list of spots instances from oncam_vec (q 3D vectors)
    and oncam_HKL (miller indices 3D vectors)

    :param oncam_vec: q vectors [qx,qy,qz] (corresponding to kf collected on camera)
    :type oncam_vec: array with 3D elements (shape = (n,3))

    :param dim: CCD frame dimensions (nb pixels, nb pixels)
    :type dim: list or tuple of 2 integers

    :param detectordistance: approximate distance detector sample
    :param detectordistance: float or integer

    :param: kf_direction : label for detection geometry
                    (CCD plane with respect to
                    the incoming beam and sample)
    :type: kf_direction: string

    :param pixelsize: pixel size in mm
    :type pixelsize: float

    :return: list of spot instances

    .. note::
        USED in lauecore.filterLaueSpots

    .. todo::
        * to be replaced by something else not using spot class
        * put this function in LaueGeometry module ?
    """
    if len(oncam_vec) != len(oncam_HKL):
        raise ValueError("Wrong input for get2ThetaChi_geometry()")

    listspot = []
    options_createspot = {"allattributes": 0, "pixelsize": pixelsize, "dim": dim}

    dictcase = {"Z>0": create_spot,   # top reflection geom
                "Y>0": create_spot_side_pos,  # side + reflection geom
                "Y<0": create_spot_side_neg,  # side - reflection geom
                "X>0": create_spot_front, # transmission geom
                "X<0": create_spot_back,  # back reflection geom
                "4PI": create_spot_4pi}   # all spots

    for position, indices in zip(oncam_vec, oncam_HKL):
        try:
            function_create_spot = dictcase[kf_direction]
        except TypeError:
            function_create_spot = create_spot_4pi

        spotcreated = function_create_spot(position, indices, detectordistance,
                                                **options_createspot)
        if spotcreated is not None:
            listspot.append(spotcreated)
    return listspot


def get2ThetaChi_geometry_full_np(oncam_vec, oncam_HKL, detectordistance=DEFAULT_DETECTOR_DISTANCE,
                                                                pixelsize=165.0 / 2048,
                                                                dim=(2048, 2048),
                                                                kf_direction=DEFAULT_TOP_GEOMETRY):
    r"""
    computes 2theta chi from oncam_vec (only 3D Q vectors corresponding to reflections on camera)
    and oncam_HKL (miller indices 3D vectors) for all spots of one grain

    :param oncam_vec: q vectors [qx,qy,qz] (corresponding to kf collected on camera)
    :type oncam_vec: array with 3D elements (shape = (n,3))

    :param dim: CCD frame dimensions (nb pixels, nb pixels)
    :type dim: list or tuple of 2 integers

    :param detectordistance: approximate distance detector sample
    :param detectordistance: float or integer

    :param: kf_direction: label for detection geometry
                    (CCD plane with respect to
                    the incoming beam and sample)
    :type: kf_direction: string

    :param pixelsize: pixel size in mm
    :type pixelsize: float

    :return: list of spot

    .. note::
        * USED in lauecore.filterLaueSpots_full_np
        * USED in DetectorCalibration.simultate_theo

    .. todo::
        * Only geometry Z>0 (top reflection) and X>0 (transmission) are vectorized by numpy
        * TODO: put this function obviously in find2thetachi ?
    """
    if len(oncam_vec) != len(oncam_HKL):
        raise ValueError("Wrong input for get2ThetaChi_geometry_full_np()")

    # TwthetaChiEnergy_list = []
    options_createspot = {"allattributes": 0, "pixelsize": pixelsize, "dim": dim}

    dictcase = {"Z>0": create_spot_np,
                "Y>0": create_spot_side_pos,
                "Y<0": create_spot_side_neg,
                "X>0": create_spot_np,
                "X<0": create_spot_np, #create_spot_back,
                "4PI": create_spot_4pi}

    try:
        function_create_spot = dictcase[kf_direction]
    except TypeError:
        function_create_spot = create_spot_4pi

    TwthetaChiEnergyMillers_list = function_create_spot(
        oncam_vec, oncam_HKL, detectordistance, **options_createspot)
    return TwthetaChiEnergyMillers_list


def RemoveHarmonics(listspot):
    r"""
    removes harmonics present in listspot (list of objects of spot class)

    .. todo:: NOT USED ANYMORE!
    """
    _invdict = {}

    # print "len after sorting wtih respect to __hash__()",len(sorted(listspot, reverse = True))
    for elem in sorted(listspot, reverse=True):
        # print "elem, elem.__hash__()",elem.Millers, elem.__hash__()

        _invdict[elem.__hash__()] = elem
        _oncamsansh = [_invdict[cle] for cle in list(_invdict.keys())]

    # print "Number of fundamental spots (RS directions): %d"%len(_oncamsansh)
    return _oncamsansh


def calcSpots_fromHKLlist(UB, B0, HKL, dictCCD):
    r"""
    computes all Laue Spots properties on 2D detector from a list of hkl
    (given structure by B0 matrix, orientation by UB matrix, and detector geometry by dictCCD)

    :param UB: orientation matrix (rotation -and if any- strain)
    :type UB: 3x3 array (or list)

    :param B0: initial a*,b*,c* reciprocal unit cell basis vector in Lauetools frame (x// ki))
    :type B0: 3x3 array (or list)

    :param HKL: array of Miller indices
    :type HKL: array with shape = (n,3)

    :param dictCCD: dictionnary of CCD properties (with key 'CCDparam', 'pixelsize','dim')
        for 'ccdparam' 5 CCD calibration parameters [dd,xcen,ycen,xbet,xgam], pixelsize in mm, and (dim1, dim2)
    :param dictCCD: dict object

    :returns: list of arrays H, K, L, Qx, Qy, Qz, X, Y, twthe, chi, Energy

    Fundamental equation
    :math:`{\bf q} = UB*B0 * {\bf G^*}`
    with :math:`{\bf G^*} = h{\bf a^*}+k{\bf b^*}+l{\bf c^*}`

    .. note::
        USED in DetectorCalibration.OnWriteResults, and PlotRefineGUI.onWriteFitFile
    """

    detectorparam = dictCCD["CCDparam"]
    pixelsize = dictCCD["pixelsize"]
    # dim = dictCCD["dim"]
    if "kf_direction" in dictCCD:
        kf_direction = dictCCD["kf_direction"]
    else:
        kf_direction = "Z>0"

    # H,K,L
    tHKL = np.transpose(HKL)

    # initial lattice rotation and distorsion

    tQ = np.dot(np.dot(UB, B0), tHKL)
    # results are qx,qy,qz

    # Q**2
    Qsquare = np.sum(tQ ** 2, axis=0)
    # norms of Q vectors
    Qn = 1.0 * np.sqrt(Qsquare)

    twthe, chi = LTGeo.from_qunit_to_twchi(tQ / Qn)

    X, Y, _ = LTGeo.calc_xycam_from2thetachi(twthe,
                                                chi,
                                                detectorparam,
                                                verbose=0,
                                                pixelsize=pixelsize,
                                                kf_direction=kf_direction)

    # E = (C)*(-q**2/qx/2)
    Energy = (CST_ENERGYKEV) * (-0.5 * Qsquare / tQ[0])

    # theoretical values
    H, K, L = tHKL
    Qx, Qy, Qz = tQ

    return H, K, L, Qx, Qy, Qz, X, Y, twthe, chi, Energy


def emptylists(n):
    r"""
    builds a list of n empty lists : [[],[],[], ...,[]]
    """
    return [[] for k in list(range(n))]


def SimulateLaue_merge( grains, emin, emax, detectorparameters, only_2thetachi=True,
                                                output_nb_spots=False,
                                                kf_direction=DEFAULT_TOP_GEOMETRY,
                                                ResolutionAngstrom=False,
                                                removeharmonics=0,
                                                pixelsize=165 / 2048.0,
                                                dim=(2048, 2048),
                                                detectordiameter=None,
                                                dictmaterials=dict_Materials):
    r"""
    Simulates Laue pattern full data from a list of grains and concatenate results data

    :param grains: list of 4 elements grain parameters

    :param only_2thetachi: * True, return only concatenated grains data 2theta and chi,
                           * False, return All_Twicetheta, All_Chi, All_Miller_ind,
                                    All_posx, All_posy, All_Energy

    :param output_nb_spots: * True, output a second element (in addition to data)
                            with list of partial nb of spots per grain
                            (to know the grain origin of spots)
    """
    # use SimulateLaue
    if not only_2thetachi:
        All_Twicetheta = []
        All_Chi = []
        All_Miller_ind = []
        All_posx = []
        All_posy = []
        All_Energy = []
        All_nb_spots = []

        for grain in grains:
            (Twicetheta, Chi, Miller_ind, posx, posy, Energy) = SimulateLaue(
                grain,
                emin,
                emax,
                detectorparameters,
                kf_direction=kf_direction,
                ResolutionAngstrom=ResolutionAngstrom,
                removeharmonics=removeharmonics,
                pixelsize=pixelsize,
                dim=dim,
                detectordiameter=detectordiameter,
                dictmaterials=dictmaterials)

            nb_spots = len(Twicetheta)

            All_Twicetheta.append(Twicetheta)
            All_Chi.append(Chi)
            All_Miller_ind.append(Miller_ind)
            All_posx.append(posx)
            All_posy.append(posy)
            All_Energy.append(Energy)
            All_nb_spots.append(nb_spots)

        All_Twicetheta = np.concatenate(All_Twicetheta)
        All_Chi = np.concatenate(All_Chi)
        All_Miller_ind = np.concatenate(All_Miller_ind)
        All_posx = np.concatenate(All_posx)
        All_posy = np.concatenate(All_posy)
        All_Energy = np.concatenate(All_Energy)

        toreturn = (All_Twicetheta,
                    All_Chi,
                    All_Miller_ind,
                    All_posx,
                    All_posy,
                    All_Energy)

    # Use SimulateResult
    else:
        simulparameters = {}
        simulparameters["detectordiameter"] = detectordiameter
        simulparameters["kf_direction"] = kf_direction
        simulparameters["detectordistance"] = detectorparameters[0]
        simulparameters["pixelsize"] = pixelsize

        All_TwicethetaChi = []
        All_nb_spots = []
        for grain in grains:
            TwicethetaChi = SimulateResult(grain,
                                            emin,
                                            emax,
                                            simulparameters,
                                            fastcompute=1,
                                            ResolutionAngstrom=False,
                                            dictmaterials=dictmaterials)
            nb_spots = len(TwicethetaChi[0])

            All_TwicethetaChi.append(TwicethetaChi)
            All_nb_spots.append(nb_spots)

        All_TwicethetaChi = np.concatenate(All_TwicethetaChi)

        toreturn = All_TwicethetaChi

    if output_nb_spots:
        return toreturn, All_nb_spots
    else:
        return toreturn


def SimulateLaue_twins(grainparent, twins_operators, emin, emax, detectorparameters,
                                                                only_2thetachi=True,
                                                                output_nb_spots=False,
                                                                kf_direction=DEFAULT_TOP_GEOMETRY,
                                                                ResolutionAngstrom=False,
                                                                removeharmonics=0,
                                                                pixelsize=165 / 2048.0,
                                                                dim=(2048, 2048),
                                                                detectordiameter=None,
                                                                dictmaterials=dict_Materials):
    r"""
    Simulates Laue pattern full data for twinned grain

    :param grainparent: list of 4 elements grain parameter

    :param twins_operators: list of 3*3 matrices corresponding of Matrices

    :param output_nb_spots: True, output a second element with list of partial nb of spots per grain

    .. note:: USED in test only in detectorCalibration...simulate_theo  to simulate 2 twinned crystals
    """
    # nb_twins = len(twins_operators)

    Bmat, dilat, Umat, extinction = grainparent

    grains = [grainparent]

    for _, twin_op in enumerate(twins_operators):
        twinUmat = np.dot(Umat, twin_op)
        grains.append([Bmat, dilat, twinUmat, extinction])

    return SimulateLaue_merge(grains, emin, emax, detectorparameters, only_2thetachi=only_2thetachi,
                                                            output_nb_spots=output_nb_spots,
                                                            kf_direction=kf_direction,
                                                            ResolutionAngstrom=ResolutionAngstrom,
                                                            removeharmonics=removeharmonics,
                                                            pixelsize=pixelsize,
                                                            dim=dim,
                                                            detectordiameter=detectordiameter,
                                                            dictmaterials=dictmaterials)


def SimulateLaue(grain, emin, emax, detectorparameters, kf_direction=DEFAULT_TOP_GEOMETRY,
                                                            ResolutionAngstrom=False,
                                                            removeharmonics=0,
                                                            pixelsize=165 / 2048.0,
                                                            dim=(2048, 2048),
                                                            detectordiameter=None,
                                                            force_extinction=None,
                                                            dictmaterials=dict_Materials,
                                                            version=1):
    r"""Computes Laue Pattern spots positions, scattering angles, miller indices
                            for a SINGLE grain or Xtal

    :param grain: crystal parameters made of a 4 elements list
    :param emin: minimum bandpass energy (keV)
    :param emax: maximum bandpass energy (keV)

    :param removeharmonics:
        * 1, removes harmonics spots and keep fondamental spots (or reciprocal direction)
            (with lowest Miller indices)

        * 0 keep all spots (including harmonics)

    :return: single grain data: Twicetheta, Chi, Miller_ind, posx, posy, Energy

    .. todo::
        To update to accept kf_direction not only in reflection geometry

    .. note::
        USED in detectorCalibration...simulate_theo  for non routine geometry (ie except Z>0 (reflection top) X>0 (transmission)
    """

    if detectordiameter is None:
        DETECTORDIAMETER = pixelsize * dim[0] #TODO * 2
    else:
        DETECTORDIAMETER = detectordiameter

    key_material = grain[3]

    grain = CP.Prepare_Grain(
        key_material, grain[2], force_extinction=force_extinction, dictmaterials=dictmaterials)

    Spots2pi = getLaueSpots(CST_ENERGYKEV / emax,
                            CST_ENERGYKEV / emin,
                            [grain],
                            fastcompute=0,
                            verbose=0,
                            kf_direction=kf_direction,
                            ResolutionAngstrom=ResolutionAngstrom,
                            dictmaterials=dictmaterials)

    #     print "len Spots2pi", len(Spots2pi[0][0])

    # list of spot which are on camera (with harmonics)
    ListofListofSpots = filterLaueSpots(Spots2pi,
                                            fileOK=0,
                                            fastcompute=0,
                                            detectordistance=detectorparameters[0],
                                            detectordiameter=DETECTORDIAMETER,
                                            kf_direction=kf_direction,
                                            HarmonicsRemoval=removeharmonics,
                                            pixelsize=pixelsize)

    ListofSpots = ListofListofSpots[0]

    Twicetheta = np.array([spot.Twicetheta for spot in ListofSpots])
    Chi = np.array([spot.Chi for spot in ListofSpots])
    Miller_ind = np.array([list(spot.Millers) for spot in ListofSpots])
    Energy = np.array([spot.EwaldRadius * CST_ENERGYKEV for spot in ListofSpots])

    posx, posy = LTGeo.calc_xycam_from2thetachi(Twicetheta,
                                                Chi,
                                                detectorparameters,
                                                verbose=0,
                                                pixelsize=pixelsize,
                                                kf_direction=kf_direction, version=version)[:2]

    return Twicetheta, Chi, Miller_ind, posx, posy, Energy


def SimulateLaue_full_np(grain, emin, emax, detectorparameters, kf_direction=DEFAULT_TOP_GEOMETRY,
                                                                ResolutionAngstrom=False,
                                                                removeharmonics=0,
                                                                pixelsize=165 / 2048.0,
                                                                dim=(2048, 2048),
                                                                detectordiameter=None,
                                                                force_extinction=None,
                                                                dictmaterials=dict_Materials,
                                                                verbose=0,
                                                                depth=None): 
    r"""Compute Laue Pattern spots positions, scattering angles, miller indices
                            for a SINGLE grain or Xtal using numpy vectorization

    :param grain:    crystal parameters in a 4 elements list
    :param emin: minimum bandpass energy (keV)
    :param emax: maximum bandpass energy (keV)

    :param removeharmonics: 1, remove harmonics spots and keep fondamental spots
                            (with lowest Miller indices)
    :param depth: depth (in microns) of the sample point that produces Laue pattern. Default = 0 (impact point at sample surface). Positive depth towards inside the sample  (// k_i)

    :return: single grain data: Twicetheta, Chi, Miller_ind, posx, posy, Energy

    .. todo::
        update to accept kf_direction not only in reflection geometry

    .. note::
        USED in detectorCalibration...simulate_theo for routine geometry Z>0 (reflection top) X>0 (transmission)
    """

    if detectordiameter is None:
        DETECTORDIAMETER = pixelsize * dim[0] #TODO * 2.0
    else:
        DETECTORDIAMETER = detectordiameter
    # use DEFAULT_TOP_GEOMETRY <=> kf_direction = 'Z>0'

    key_material = grain[3]
    grain = CP.Prepare_Grain(key_material, grain[2],
                                    force_extinction=force_extinction, dictmaterials=dictmaterials)

    Qxyz, HKL = getLaueSpots(CST_ENERGYKEV / emax,
                            CST_ENERGYKEV / emin,
                            [grain],
                            fastcompute=0,
                            verbose=0,
                            kf_direction=kf_direction,
                            ResolutionAngstrom=ResolutionAngstrom,
                            dictmaterials=dictmaterials)

    # list of spot which are on camera (with harmonics)
    TwthetaChiEnergyMillers_list_one_grain_wo_harmonics = filterLaueSpots_full_np(
                                                Qxyz[0],
                                                HKL[0],
                                                fastcompute=0,
                                                detectordistance=detectorparameters[0],
                                                detectordiameter=DETECTORDIAMETER,
                                                kf_direction=kf_direction,
                                                HarmonicsRemoval=0,
                                                pixelsize=pixelsize)

    Twicetheta_zerodepth, Chi_zerodepth, Energy, Miller_ind = TwthetaChiEnergyMillers_list_one_grain_wo_harmonics[:4]

    posx_zerodepth, posy_zerodepth = LTGeo.calc_xycam_from2thetachi(Twicetheta_zerodepth,
                                                Chi_zerodepth,
                                                detectorparameters,
                                                verbose=0,
                                                pixelsize=pixelsize,
                                                kf_direction=kf_direction)[:2]

    if depth is not None:
        posx, posy = posx_zerodepth, posy_zerodepth + depth/1000./pixelsize
        Twicetheta, Chi = LTGeo.calc_uflab(posx, posy, detectorparameters,
                                                offset=0, returnAngles=1,
                                                verbose=0,
                                                pixelsize=pixelsize,
                                                rectpix=0,
                                                kf_direction=kf_direction)
    else:
        posx, posy = posx_zerodepth, posy_zerodepth
        Twicetheta, Chi = Twicetheta_zerodepth, Chi_zerodepth

    if removeharmonics:
        # remove harmonics:
        _, _, tokeep = GT.removeClosePoints(posx, posy, 0.05)
        if verbose: print('removeharmonics = 1 in SimulateLaue_full_np() tokeep',tokeep)

        s_tth = Twicetheta[tokeep]
        s_chi = Chi[tokeep]
        s_miller_ind = Miller_ind[tokeep]
        s_posx = posx[tokeep]
        s_posy = posy[tokeep]
        s_E = Energy[tokeep]
        return s_tth, s_chi, s_miller_ind, s_posx, s_posy, s_E
    else:
        return Twicetheta, Chi, Miller_ind, posx, posy, Energy


def SimulateResult(grain, emin, emax, simulparameters,
                    fastcompute=1, ResolutionAngstrom=False, dictmaterials=dict_Materials):
    r"""Simulates 2theta chi of Laue Pattern spots for ONE SINGLE grain

    :param grain: crystal parameters in a 4 elements list
    :param emin: minimum bandpass energy (keV)
    :param emax: maximum bandpass energy (keV)

    :return: 2theta, chi

    .. warning:: Need of approximate detector distance and diameter to restrict simulation to a limited solid angle

    .. note::
        * USED: in AutoindexationGUI.OnStart, LaueToolsGUI.OnCheckOrientationMatrix
        * USED also IndexingImageMatching, lauecore.SimulateLaue_merge
    """

    detectordiameter = simulparameters["detectordiameter"]
    kf_direction = simulparameters["kf_direction"]
    detectordistance = simulparameters["detectordistance"]
    pixelsize = simulparameters["pixelsize"]

    # PATCH: redefinition of grain to simulate any unit cell(not only cubic)
    key_material = grain[3]
    grain = CP.Prepare_Grain(key_material, grain[2], dictmaterials=dictmaterials)
    # -----------------------------------------------------------------------------

    # print "grain in SimulateResult()",grain

    spots2pi = getLaueSpots(CST_ENERGYKEV / emax,
                            CST_ENERGYKEV / emin,
                            [grain],
                            fastcompute=fastcompute,
                            ResolutionAngstrom=ResolutionAngstrom,
                            verbose=0,
                            kf_direction=kf_direction,
                            dictmaterials=dictmaterials)
    # ---------------------------------------------------------------------------

    # array(vec) and array(indices)  of spots exiting the crystal in 2pi steradian
    # if fastcompute = 0 array(indices) = 0 and TwicethetaChi is a list of spot object
    # k_direction =(Z>0)
    # TODO: to be argument if camera is far from preset kf_direction!!
    # spots2pi = LAUE.generalfabriquespot_fromMat_veryQuick(CST_ENERGYKEV/emax, CST_ENERGYKEV/emin,[grain],1,
    # fastcompute = fastcompute, fileOK = 0, verbose = 0, kf_direction = 'Z>0')

    # 2theta, chi of spot which are on camera(without harmonics)
    TwicethetaChi = filterLaueSpots(spots2pi,
                                    fileOK=0,
                                    fastcompute=fastcompute,
                                    detectordistance=detectordistance,
                                    detectordiameter=detectordiameter * 1.2,
                                    kf_direction=kf_direction,
                                    pixelsize=pixelsize)

    return TwicethetaChi

def B_DebyeWaller(U):
    """ compute B term of exp Debye waller factor
    exp - B  (sintheta/lambda)
    with U mean square displacement
    """
    return 8 * np.pi**2 * U


def StructureFactorCubic(h, k, l, extinctions="dia"):
    """
    computes structure factor of cubic
    """
    assert extinctions == 'dia'
    pi = np.pi
    F = (1 + np.exp(-1.0j * pi / 2.0 * (h + k + l))) * (1 + np.exp(-1.0j * pi * (k + l))
                                                        + np.exp(-1.0j * pi * (h + l))
                                                        + np.exp(-1.0j * pi * (h + k)))
    return F


def StructureFactorUO2(h, k, l, qvector, U_U, U_O):
    """
    computes structure factor of CaF2 flurine type structure  SG 225 Fm-3m
    """
    # CaF2 structure
    pi = np.pi
    # q = 4pi*sintheta/lambda
    sol = qvector/(4*pi)
    B_U = B_DebyeWaller(U_U)
    B_O = B_DebyeWaller(U_O)
    fu = atomicformfactor(qvector, "U")*np.exp(-B_U * sol**2)
    fo = atomicformfactor(qvector, "O")*np.exp(-B_O * sol**2)
    F = 4*(fu + 2 * fo * np.cos(pi / 2*((h + k + l)))) * (1 + np.exp(1.0j * pi * (k + l))
                                                        + np.exp(1.0j * pi * (l + h))
                                                        + np.exp(1.0j * pi * (h + k)))
    return F


def atomicformfactor(q, element="Ge"):
    """
    Computes x-ray atomic scattering factor following
    http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php

    :param q: q vector norm in Angst-1
    :returns: scalar, f(q)

    """
    if element == "Ge":
        p = (16.0816, 2.8509, 6.3747, 0.2516, 3.7068, 11.4468, 3.683, 54.7625, 2.1313)

    elif element == "U":
        p = (5.3715, 0.516598, 22.5326, 3.05053, 12.0291, 12.5723, 4.79840, 23.4582, 13.2671)

    elif element == "O":
        p = (3.04850, 13.2771, 2.28680, 5.70110, 1.54630, 0.323900, 0.867000, 32.9089, 0.250800)

    val = 0
    for k in list(range(4)):
        val += p[2 * k] * np.exp(-p[2 * k + 1] * (q / 4 / np.pi) ** 2)
    val += p[-1]
    return val


def simulatepurepattern_np(grain, emin, emax, kf_direction, data_filename, PlotLaueDiagram=1,
                                                        Plot_Data=0,
                                                        verbose=0,
                                                        detectordistance=DEFAULT_DETECTOR_DISTANCE,
                                                        ResolutionAngstrom=False,
                                                        Display_label=1,
                                                        HarmonicsRemoval=1,
                                                        dictmaterials=dict_Materials):
    """
    .. warning:: In test. NOT USED anywhere !!???
    """

    vecind = getLaueSpots(CST_ENERGYKEV / emax, CST_ENERGYKEV / emin, grain,
                    fastcompute=0, kf_direction=kf_direction,
                    verbose=verbose, ResolutionAngstrom=ResolutionAngstrom,
                    dictmaterials=dictmaterials)

    print("len(vecind[0])", len(vecind[0][0]))

    # selecting RR nodes without harmonics (fastcompute = 1 loses the miller indices and RR positions associations for quicker computation)

    oncam_sansh = filterLaueSpots_full_np(vecind[0], vecind[1],
                                        fastcompute=0,
                                        kf_direction=kf_direction,
                                        detectordistance=detectordistance,
                                        HarmonicsRemoval=HarmonicsRemoval)

    return True
