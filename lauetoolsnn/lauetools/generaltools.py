# -*- coding: utf-8 -*-
from __future__ import print_function
"""
module of lauetools project

JS Micha May 2019

package gathering general tools
"""

import copy
import pickle
# import multiprocessing
import numpy as np
import scipy.spatial.distance as ssd
import matplotlib as mpl

if mpl.__version__ < "2.2":
    MATPLOTLIB2p2 = False
else:
    MATPLOTLIB2p2 = True

import matplotlib.cm as mplcm

DEG = np.pi / 180.0

IDENTITYMATRIX = np.eye(3)

try:
    from lauetools import IOLaueTools as IOLT
except:
    import lauetoolsnn.lauetools.IOLaueTools as IOLT

# --- --------------  Vectors
def AngleBetweenVectors(Vectors1, Vectors2, metrics=IDENTITYMATRIX, verbose=False):
    """compute angles between all pairs of vectors from Vectors1 and Vectors2

    inputs:
    Vectors1,Vectors2            :  list of n1 3D vectors, list of n2 3D vectors
    Gstar            : metrics , default np.eye(3)
                    """
    HKL1r = np.array(Vectors1)
    HKL2r = np.array(Vectors2)

    if HKL1r.shape == (3,):
        H1 = np.array([HKL1r])
        n1 = 1
    elif HKL1r.shape == (1, 3):
        H1 = HKL1r
        n1 = 1
    else:
        H1 = HKL1r
        n1 = HKL1r.shape[0]

    if HKL2r.shape == (3,):
        H2 = np.array([HKL2r])
        n2 = 1
    elif HKL2r.shape == (1, 3):
        H2 = HKL2r
        n2 = 1
    else:
        H2 = HKL2r
        n2 = HKL2r.shape[0]

    dstar_square_1 = np.diag(np.inner(np.inner(H1, metrics), H1))
    dstar_square_2 = np.diag(np.inner(np.inner(H2, metrics), H2))

    scalar_product = np.inner(np.inner(H1, metrics), H2) * 1.0

    if n1 != 1:
        # 1d
        d1 = np.sqrt(dstar_square_1.reshape((n1, 1))) * 1.0
    else:
        d1 = np.sqrt(dstar_square_1)
    if n2 != 1:
        # 1d
        d2 = np.sqrt(dstar_square_2.reshape((n2, 1))) * 1.0
    else:
        d2 = np.sqrt(dstar_square_2)

    outy = np.outer(d1, d2)

    if verbose:
        print("H1", H1)
        print("H2", H2)
        print("d1", d1)
        print("d2", d2)
        print("len(d1)", len(d1))
        print("len(d2)", len(d2))
        print("outy", outy)
        print(outy.shape)

        print("scalar_product", scalar_product)
        print(scalar_product.shape)

    ratio = scalar_product / outy
    ratio = np.round(ratio, decimals=7)
    #    print "ratio", ratio
    #    np.putmask(ratio, np.abs(ratio + 1) <= .0001, -1)
    #    np.putmask(ratio, ratio == 0, 0)

    return np.arccos(ratio) / DEG


def calculdist2D(listpoints1, listpoints2):
    """
    return CLOSEST:
    array of closest spot index in listpoints2 for each point in listpoints1

    len(CLOSEST) == len(listpoints1)

    values of CLOSEST are indices of points in listpoints2

    For cartesian coordinates
    """
    # return CLOSEST
    return np.argmin(calcdistancetab(listpoints1, listpoints2))


def calcdistancetab(listpoints1, listpoints2):
    data1 = np.array(listpoints1)
    data2 = np.array(listpoints2)
    xdata1 = data1[:, 0]
    ydata1 = data1[:, 1]
    xdata2 = data2[:, 0]
    ydata2 = data2[:, 1]
    deltax = xdata1 - np.reshape(xdata2, (len(xdata2), 1))
    deltay = ydata1 - np.reshape(ydata2, (len(ydata2), 1))
    didist = np.sqrt(deltax ** 2 + deltay ** 2)

    return didist


def distfrom2thetachi(points1, points2):
    """
    returns angular distance (deg) from two single spots
    defined by kf=(2theta,chi) between corresponding q vectors
    (ie corresponding to the two atomic planes normals)

    q = kf - ki

    points1, points2 must be two elements array: [2theta_1, chi_1], [2theta_2, chi_2]
    """

    longdata1 = points1[0] * DEG / 2.0  # theta
    latdata1 = points1[1] * DEG  # chi

    longdata2 = points2[0] * DEG / 2.0
    latdata2 = points2[1] * DEG

    deltalat = latdata1 - latdata2
    cosang = np.sin(longdata1) * np.sin(longdata2) + np.cos(longdata1) * np.cos(
                                                                    longdata2) * np.cos(deltalat)

    return np.arccos(cosang) / DEG


def calculdist_from_thetachi(listpoints1, listpoints2):
    """
    From two lists of pairs (THETA, CHI) return:

    return:
    tab_angulardist:   matrix of all mutual angular distance
                whose shape is (len(list2),len(list1))

    WARNING: theta angle is used, i.e. NOT 2THETA!
    TIP: used with listpoints1 = expspots  and listpoints2 = theospots
    """
    data1 = np.array(listpoints1)
    data2 = np.array(listpoints2)
    # print "data1",data1
    # print "data2",data2
    longdata1 = data1[:, 0] * DEG  # theta
    latdata1 = data1[:, 1] * DEG  # chi

    longdata2 = data2[:, 0] * DEG  # theta
    latdata2 = data2[:, 1] * DEG  # chi

    deltalat = latdata1 - np.reshape(latdata2, (len(latdata2), 1))
    longdata2new = np.reshape(longdata2, (len(longdata2), 1))
    prodcos = np.cos(longdata1) * np.cos(longdata2new)
    prodsin = np.sin(longdata1) * np.sin(longdata2new)

    arccos_arg = np.around(prodsin + prodcos * np.cos(deltalat), decimals=9)

    tab_angulardist = (1.0 / DEG) * np.arccos(arccos_arg)

    return tab_angulardist


# if NUMBAINSTALLED:

#     @njit(fastmath=True, parallel=True)
#     def mycalcangle(a, b):

#         Lo1 = a[0]*DEG / 2.0
#         Lo2 = b[0]*DEG / 2.0

#         La1 = a[1]*DEG
#         La2 = b[1]*DEG
#         delta = La1 - La2
#         cang = math.sin(Lo1)*math.sin(Lo2) + math.cos(Lo1)*math.cos(Lo2)*math.cos(delta)

#         return math.acos(cang)/DEG

# def pairwise_mutualangles(XY1, XY2):
#     """
#     From two lists of pairs (2THETA, CHI) return:

#     return:
#     tab_angulardist:   matrix of all mutual angular distance
#                 whose shape is (len(list2),len(list1))

#     TIP: used with listpoints1 = expspots  and listpoints2 = theospots
#     """
#     M1 = XY1.shape[0]
#     M2 = XY2.shape[0]
#     D = np.empty((M1, M2), dtype=np.float)
#     #k=0
#     for i in range(M1):
#         #print("XY1[i]",XY1[i])
#         for j in range(M2):
#             D[i, j] = mycalcangle(XY1[i], XY2[j])
#             #k+=1
#     return D

# def pairwise_mutualangles_1D(XY1, XY2):
#     """
#     From two lists of pairs (2THETA, CHI) return:

#     return:
#     tab_angulardist:   matrix of all mutual angular distance
#                 whose shape is (len(list2),len(list1))

#     TIP: used with listpoints1 = expspots  and listpoints2 = theospots
#     """
#     M1 = XY1.shape[0]
#     M2 = XY2.shape[0]
#     D = np.empty(M1*M2, dtype=np.float)
#     k = 0
#     for i in range(M1):
#         #print("XY1[i]",XY1[i])
#         for j in range(M2):
#             D[k] = mycalcangle(XY1[i], XY2[j])
#             k += 1
#     return D

# if NUMBAINSTALLED:
#     def computeMutualAngles(listpoints1, listpoints2, TwicethetaInput=True):

#         if TwicethetaInput:
#             pairwiseangles_numba = autojit(pairwise_mutualangles)

#         return pairwiseangles_numba(listpoints1, listpoints2)

def cartesiandistance(x1, x2, y1, y2):
    """
    return the cartesian distance between two points
    """
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def norme_vec(vec1):
    """
    return the scalar cartesian norm of a single 3 elements vector
    """
    #     if isinstance(vec1, list):
    #         if len(vec1) != 3:
    #             raise TypeError, "%s is not a list or array of 3 elements" % str(vec1)
    #     elif isinstance(vec1, np.ndarray):
    #         if len(vec1.shape) != 1 or vec1.shape[0] != 3:
    #             raise TypeError, "%s is not a list or array of 3 elements" % str(vec1)
    norm = np.sqrt(np.inner(vec1, vec1))
    return norm


def getAngle_2Vectors(vec1, vec2):
    """
    return angle between two vectors in degree
    """
    n1 = np.sqrt(np.dot(vec1, vec1)) * 1.0
    n2 = np.sqrt(np.dot(vec2, vec2)) * 1.0

    return 180.0 / np.pi * np.arccos(np.dot(vec1, vec2) / n1 / n2)


def norme_list(listvec):
    """
    return 1d array of cartesian norms of an array of 3 elements vector

    """
    normarray = np.sqrt(np.diag(np.inner(listvec, listvec)))
    return normarray


def tensile_along_u(v, tensile, u="zsample"):
    """
    from list of vectors of q vectors expressed in absolute frame,
    transform them so that to expand or compress the q vector component along u axis by factor 'tensile'.

    result is an array of 3 elements vectors

    v*_in = v*_in_plane+v*_in_along_u
    v*_out = v*_in_plane+factor * v*_in_along_u
    v*_in_along_u = (v*_in.u) u   (along u)
    v*_out = v*_in+(factor-1)(v_in.u) u
    (or v*_in_plane= (u ^ v)^u)

    example: 1.02 means real expansion of 2% i.e. 1/1.02 of variation of reciprocal vector component along u
    """
    # print "lolo",v # all v vectors
    # print wholelistindicesfiltered
    omegasurfacesample = 40 * DEG  # 40 deg sample inclination
    real_expansion_coef = tensile
    if u == "zsample":
        # u direction traction in q space in absolute frame
        direction_traction = np.array([-np.sin(omegasurfacesample), 0, np.cos(omegasurfacesample)])
    else:
        # normalized axis vector u
        UU = np.array(u)
        nUU = 1.0 * np.sqrt(np.sum(UU ** 2))
        # u direction traction in q space in absolute frame
        direction_traction = np.array(UU) / nUU

    # u must be normalized
    scalaruv = np.inner(v, direction_traction)  # array of all scalar product (u, v)
    # print "scalaruv",scalaruv

    # crossuvu = cross(cross(direction_traction, wholelistvecfiltered[0]),direction_traction)
    # wholelistvecfiltered = [crossuvu+1./real_expansion_coef*np.reshape(scalaruv,(len(scalaruv),1))*direction_traction]
    wholelistvecfiltered = (v
                            + (1.0 / real_expansion_coef - 1)
                            * np.reshape(scalaruv, (len(scalaruv), 1))
                            * direction_traction)
    # print "lala",wholelistvecfiltered

    return wholelistvecfiltered


def rotate_around_u(v, angle, u):
    """
    from list of vectors of v in absolute frame, rotate q vector component around u
    angle in deg
    result is an array
    """
    UU = np.array(u)
    nUU = 1.0 * np.sqrt(np.sum(UU ** 2))
    unit_axis = np.array(UU) / nUU  # u direction traction in q space in absolute frame

    mat = matRot(unit_axis, angle)

    wholelistvecfiltered = np.transpose(np.dot(mat, np.transpose(np.array(v))))

    return wholelistvecfiltered


def reflect_on_u(v, u):
    """
    from list of vectors of v in absolute frame, reflect vector on plane defined by its normal u
    angle in deg
    result is an array
    """
    UU = np.array(u)
    nUU = 1.0 * np.sqrt(np.sum(UU ** 2))
    unit_axis = np.array(UU) / nUU  # u plane's normal

    mat = np.eye(3) - 2.0 * np.outer(unit_axis, unit_axis)

    wholelistvecfiltered = np.dot(mat, v.T).T

    return wholelistvecfiltered


def strain_along_u(v, alpha, u="zsample", anglesample=40):
    """
    from list of vectors of v in absolute frame,
    /alpha expand or contract one vector component along u
    alpha in real space
    1/alpha in reciprocal space
    result is an array
    """
    omegasurfacesample = anglesample * DEG  # 40 deg sample inclination
    if u == "zsample":
        # u direction traction in q space in absolute frame
        direction_traction = np.array([-np.sin(omegasurfacesample), 0, np.cos(omegasurfacesample)])
    else:
        UU = np.array(u)
        nUU = 1.0 * np.sqrt(np.sum(UU ** 2))
        # u direction traction in q space
        direction_traction = np.array(UU) / nUU

    mat = np.eye(3) + (1.0 / alpha - 1) * np.outer(direction_traction, direction_traction)

    wholelistvecfiltered = np.dot(mat, v.T).T

    return wholelistvecfiltered


# ------ ---------  Matrices
def matline_to_mat3x3(mat):
    """
    arrange  9 elements in columns in a 3*3 matrix
    """
    # print "mat ligne \n", mat
    mat1 = np.column_stack((mat[0:3], mat[3:6], mat[6:9]))
    # print "mat 3x3 \n", mat1
    return mat1


def mat3x3_to_matline(mat):
    """
    convert the three columns of 3*3 matrix in a 9 elements vector

    WARNING: not ravel(mat) but ravel(mat.T)
    """
    # print "mat 3x3 \n", mat
    mat1 = np.hstack((mat[:, 0], mat[:, 1], mat[:, 2]))
    # print "mat ligne \n", mat1
    return mat1


def epsline_to_epsmat(epsline):
    """
    Arrange the 6 elements in line in symetric deviatoric strain matrix (3x3)

    NOTE: # deviatoric strain 11 22 33 -dalf 23, -dbet 13, -dgam 12
    """
    if len(epsline) != 6:
        raise ValueError("%s argument in epsline_to_epsmat has not 6 elements" % epsline)

    epsmat = np.identity(3, float)

    epsmat[0, 0] = epsline[0]
    epsmat[1, 1] = epsline[1]
    epsmat[2, 2] = epsline[2]

    epsmat[1, 2] = epsline[3]
    epsmat[0, 2] = epsline[4]
    epsmat[0, 1] = epsline[5]

    epsmat[2, 1] = epsline[3]
    epsmat[2, 0] = epsline[4]
    epsmat[1, 0] = epsline[5]

    return epsmat


def epsmat_to_epsline(epsmat):
    """
    Arrange matrix elements of symetric deviatoric strain (6 independent elements)
    in a row matrix

    From Odile robach
    """
    # deviatoric strain 11 22 33 -dalf 23, -dbet 13, -dgam 12

    epsline = np.zeros(6, float)

    epsline[0] = epsmat[0, 0]
    epsline[1] = epsmat[1, 1]
    epsline[2] = epsmat[2, 2]

    epsline[3] = epsmat[1, 2]
    epsline[4] = epsmat[0, 2]
    epsline[5] = epsmat[0, 1]

    return epsline


def Orthonormalization(mat):
    """
    return orthonormalized matrix M from a matrix where columns are expression
    of non unit and non orthogonal expression of basis vector in absolute frame

    first vector is considered as the first vector of the new basis

    TODO : to check
    """
    vec1, vec2, _ = np.array(mat).T

    new1 = vec1 / np.sqrt(np.dot(vec1, vec1))

    new3 = np.cross(vec1, vec2)
    new3_n = np.sqrt(np.dot(new3, new3))
    new3 = new3 / new3_n

    new2 = np.cross(new3, new1)

    return np.array([new1, new2, new3]).T


def UBdecomposition_RRPP(UBmat):
    """
    decomposes UBmat in matrix product RR*PP
    where RR is pure rotation and PP symetric matrix
    """
    # polar decomposition from singular value decomposition
    # see http://en.wikipedia.org/wiki/Polar_decomposition and np.svd() help
    U, ss, Vc = np.linalg.svd(UBmat)

    SS = np.zeros_like(UBmat)
    SS = np.diag(ss)
    RR = np.dot(U, Vc)  # rotation matrix
    Vcinv = np.linalg.inv(Vc)
    PP = np.dot(np.dot(Vcinv, SS), Vc)  # symetric matrix (strain) in inverse distance
    # UBmat = RR*PP
    PP = PP / np.amax(PP)

    return RR, PP


# ---- ------------------  TENSORS -------
##http://www.continuummechanics.org/coordxforms.html
#

def rotT(T, g):
    """
    rotate tensor of 4th rank

    T tensor of 3*3*3*3 numpy array shape
    g rotation transform matrix 3*3 shape

    http://stackoverflow.com/questions/4962606/fast-tensor-rotation-with-numpy
    """
    gg = np.outer(g, g)
    gggg = np.outer(gg, gg).reshape(4 * g.shape)
    axes = ((0, 2, 4, 6), (0, 1, 2, 3))
    return np.tensordot(gggg, T, axes)


def rotT_faster(T, gggg):
    """
    even 2x faster
    """
    return np.dot(gggg.transpose((1, 3, 5, 7, 0, 2, 4, 6)).reshape((81, 81)), T.reshape(81, 1)
    ).reshape((3, 3, 3, 3))


# -------------- ----------   ALGEBRA
def pgcd(a, b):
    """return the highest common divisor of two positive integers
    Method of sequential substractions
    """
    sentenceerror = "%s,%s must be positive integers" % (str(a), str(b))
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError(sentenceerror)
    if a <= 0 or b <= 0:
        raise TypeError(sentenceerror)
    while a != b:
        if a < b:
            a, b = b, a
        a, b = a - b, b
    return a


def pgcdl(l):
    """return the largest common divisor of a list of POSITIVE integers
    aa = np.array(l)
    absaa = abs(aa)
    reaa = np.reshape(absaa,(len(l),1))
    pot = [pgcd(val[0],val[1]) for val in broadcast(absaa, reaa)]
    return min(pot)
    """
    lepgcd = l[0]
    for element in l:
        lepgcd = pgcd(lepgcd, element)
    return lepgcd


def signe(a):
    """ return 0 if a<0 , 1 else"""
    if a < 0:
        return 0
    else:
        return 1


def properinteger(flo):
    """ return the furthest integer of the input float from zero
    gives ceil(flo) for flo > 0
    or floor(flo) for flo  < 0
    """
    if not isinstance(flo, (float, int)):
        raise TypeError("properinteger() needs float or integer")

    if flo >= 0:
        return np.ceil(flo)
    else:
        return np.floor(flo)


# ----- ------------  SET
def FindClosestPoint(arraypts, XY, returndist=0):
    """
    Returns the index of the closest point in arraypts from point XY =[X,Y]

    arraypts = [[x1,y1],[x2,y2],...]
    """
    dist = np.array(arraypts) - XY
    indclose = np.argmin(np.hypot(dist[:, 0], dist[:, 1]))
    if returndist:
        return indclose, np.sqrt(np.sum(dist ** 2, axis=1))
    else:
        return indclose


def FindTwoClosestPoints(arraypts, XY):
    """
    Returns the index of the two closest points in arraypts from point XY =[X,Y]

    arraypts = [[x1,y1],[x2,y2],...]
    """
    if arraypts.shape[0] <= 2:
        raise ValueError("arraypts does not contain more than 2 elements: %s" % str(arraypts))
    dist = np.array(arraypts) - XY
    indclose = np.argsort(np.hypot(dist[:, 0], dist[:, 1]))[:2]

    return indclose, np.sqrt(np.sum(dist ** 2, axis=1))[indclose]


def SortPoints_fromPositions(TestPoints, ReferencePoints, tolerancedistance=5):
    """
    to sort list of points as a function of their proximity to a sorted reference list of spots

    TestPoints    : list of spots to be sorted accordingly to ReferencePoints
    ReferencePoints: list of spots whose sorting is considered as a reference

    Sort points in TestPoints according to the rank of the NEAREST points in ReferencePoints

    Loop is performed on points ReferencePoints:
    for each point of ReferencePoints, the closest point of TestPoints (within tolerancedistance) is selected.

    return:

    isolated_pts_in_ReferencePoints    : list of indices of isolated (within tolerance) spots of ReferencePoints
    """
    # i_ReferencePoints  index of element in list ReferencePoints
    # list_closest_ind[i_ReferencePoints] = index of closest point in TestPoints of point i_ReferencePoints
    list_closest_ind = []
    # list_allclose_ind[i_ReferencePoints] = all point indices in TestPoints
    # close enough (within tolerancedistance) to point i_ReferencePoints
    list_allclose_ind = []

    # list_all_dist[i_ReferencePoints] = all distances to points in TestPoints
    list_all_dist = []
    # list_mindist[i_ReferencePoints] = smallest distance to points in TestPoints
    list_mindist = []

    for XY_Ref in ReferencePoints:
        indclose_XY_Ref, distances_to_XY_Ref = FindClosestPoint(TestPoints, XY_Ref, returndist=1)

        list_closest_ind.append(indclose_XY_Ref)

        sortdist_ind = np.argsort(distances_to_XY_Ref)
        #         print 'sortdist_ind', sortdist_ind
        close_pts = np.where(distances_to_XY_Ref[sortdist_ind] <= tolerancedistance)[0]

        list_allclose_ind.append(sortdist_ind[close_pts])

        list_all_dist.append(distances_to_XY_Ref)
        list_mindist.append(min(distances_to_XY_Ref))

    selected_testpoints_indices = []

    isolated_pts_in_ReferencePoints = []
    for k_refpoint, closests_spot_index in enumerate(list_allclose_ind):
        if not closests_spot_index:
            isolated_pts_in_ReferencePoints.append(k_refpoint)
            continue

        for closest_spot_index in closests_spot_index:
            if closest_spot_index in selected_testpoints_indices:
                continue

            selected_testpoints_indices.append(closest_spot_index)
            break

    print("select points index in TestPoints", selected_testpoints_indices)

    selected_testpoints_indices = prepend_in_list(list(list(range(len(TestPoints)))),
                                                    selected_testpoints_indices)

    print("select points index in TestPoints", selected_testpoints_indices)

    #     return list_closest_ind, list_allclose_ind, list_all_dist, list_mindist
    selected_testpoints = np.take(TestPoints, selected_testpoints_indices, axis=0)

    return (selected_testpoints,
        selected_testpoints_indices,
        isolated_pts_in_ReferencePoints)


def prepend_in_list(list_to_modify, elems):
    """
    move elements of list_to_modify those that are at the beginning of list_to_modify

    .. example::
    prepend_in_list(range(10),[6,2,5])
    [6, 2, 5, 0, 1, 3, 4, 7, 8, 9]

    .. warning:: in each list, elements must appear once
    """
    if isinstance(list_to_modify, np.ndarray):
        list_to_modify = list_to_modify.tolist()

    if isinstance(elems, np.ndarray):
        elems = elems.tolist()

    for el in list_to_modify:
        if list_to_modify.count(el) > 1:
            raise ValueError("%s (input list to modify) contains duplicates!!" % str(list_to_modify))

    todelete = []
    for k in elems:
        if elems.count(k) > 1:
            raise ValueError("%s (elements list to prepend) contains duplicates!!" % elems)

        nb_occurences = list_to_modify.count(k)
        if nb_occurences == 0:
            todelete.append(k)
        else:
            list_to_modify.remove(k)

    for elem_not_in_list in todelete:
        elems.remove(elem_not_in_list)

    return elems + list_to_modify


def find_closest(input_array, target_array, tol):
    """
    Find the set of elements in input_array that are closest to
    elements in target_array.  Record the indices of the elements in
    target_array that are within tolerance, tol, of their closest
    match. Also record the indices of the elements in target_array
    that are outside tolerance, tol, of their match.

    For example, given an array of observations with irregular
    observation times along with an array of times of interest, this
    routine can be used to find those observations that are closest to
    the times of interest that are within a given time tolerance.

    NOTE: input_array must be sorted! The array, target_array, does not have to be sorted.

    Inputs:
      input_array:  a sorted Float64 numarray
      target_array: a Float64 numarray
      tol:          a tolerance

    Returns:
      closest_indices:  the array of indices of elements in input_array that are closest to elements in target_array
      accept_indices:  the indices of elements in target_array that have a match in input_array within tolerance
      reject_indices:  the indices of elements in target_array that do not have a match in input_array within tolerance

    from stats.py
    """
    input_array_len = len(input_array)
    closest_indices = np.searchsorted(input_array, target_array)  # determine the locations of target_array in input_array
    acc_rej_indices = [-1] * len(target_array)
    curr_tol = [tol] * len(target_array)

    est_tol = 0.0
    for i in list(range(len(target_array))):
        best_off = 0  # used to adjust closest_indices[i] for best approximating element in input_array

        if closest_indices[i] >= input_array_len:
            # the value target_array[i] is >= all elements in input_array so check whether it is within tolerance of the last element
            closest_indices[i] = input_array_len - 1
            est_tol = target_array[i] - input_array[closest_indices[i]]
            if est_tol < curr_tol[i]:
                curr_tol[i] = est_tol
                acc_rej_indices[i] = i
        elif target_array[i] == input_array[closest_indices[i]]:
            # target_array[i] is in input_array
            est_tol = 0.0
            curr_tol[i] = 0.0
            acc_rej_indices[i] = i
        elif closest_indices[i] == 0:
            # target_array[i] is <= all elements in input_array
            est_tol = input_array[0] - target_array[i]
            if est_tol < curr_tol[i]:
                curr_tol[i] = est_tol
                acc_rej_indices[i] = i
        else:
            # target_array[i] is between input_array[closest_indices[i]-1] and input_array[closest_indices[i]]
            # and closest_indices[i] must be > 0
            top_tol = input_array[closest_indices[i]] - target_array[i]
            bot_tol = target_array[i] - input_array[closest_indices[i] - 1]
            if bot_tol <= top_tol:
                est_tol = bot_tol
                best_off = -1  # this is the only place where best_off != 0
            else:
                est_tol = top_tol

            if est_tol < curr_tol[i]:
                curr_tol[i] = est_tol
                acc_rej_indices[i] = i

        if est_tol <= tol:
            closest_indices[i] += best_off

    accept_indices = np.compress(np.greater(acc_rej_indices, -1), acc_rej_indices)
    reject_indices = np.compress(
        np.equal(acc_rej_indices, -1), np.arange(len(acc_rej_indices)))

    return closest_indices, accept_indices, reject_indices


def mutualpairs(ind1, ind2):
    """return all pairs from elements of 2 lists

    .. example::
        mutualpairs([2,99,5],[8,6])
        array([[ 2,  8], [99,  8], [ 5,  8], [ 2,  6], [99,  6], [ 5,  6]])

    :param ind1: 1D list of indices (len = n)
    :type ind1: iterable
    :param ind2: 1D list of indices (len = m)
    :type ind2: iterable
    :return: list of pairs with shape (n*m, 2)
    :rtype: array
    """
    return np.transpose([np.tile(ind1, len(ind2)), np.repeat(ind2, len(ind1))])


def pairs_of_indices(n):
    """
    return indice position of non zero and non diagonal elements in triangular up matrix (n*n)
    pairs_of_indices(5)
    array([[0, 1],
       [0, 2],
       [0, 3],
       [0, 4],
       [1, 2],
       [1, 3],
       [1, 4],
       [2, 3],
       [2, 4],
       [3, 4]])

    Useful to get elements coming from expression derived from two non identical elements of a set
    """
    pairs = []
    for i in list(range(n)):
        for j in list(range(i + 1, n)):
            pairs.append([i, j])
    return np.array(pairs)


def allpairs_in_set(list_of_indices):
    """
    return all combinations by pairs of elements in list_of_indices

    list_of_indices: must contain elements of the same type

    allpairs_in_set([0,20,5,9,1])

    array([[ 0, 20],
       [ 0,  5],
       [ 0,  9],
       [ 0,  1],
       [20,  5],
       [20,  9],
       [20,  1],
       [ 5,  9],
       [ 5,  1],
       [ 9,  1]])

    allpairs_in_set(['a','b','c'])
    array([['a', 'b'],
       ['a', 'c'],
       ['b', 'c']],
      dtype='|S1')
    """
    ar = np.array(list_of_indices)
    pairs = pairs_of_indices(len(ar))
    return np.take(list_of_indices, pairs)


def return_pair(n, pairs):
    """
    return array of integer that are in correspondence in pairs with integer n
    """
    Pairs = np.array(pairs)

    i, j = np.where(Pairs == n)
    j = (j + 1) % 2
    return Pairs[(i, j)]


def getSets(pairs):
    """
    find indices pairs from index connections given by a list from pairs of indices

    TODO: not really doing the thing

    example:

    getSets([[ 0,  1],[ 0,  2],[ 1,  2],[ 3,  7],[ 3, 11],[ 4,  9],[ 4, 14],[ 7, 11],[ 9, 14],
    [15, 31],[15, 47],[16, 33],[16, 50],[19, 39],[19, 59],[20, 41],[20, 62],[31, 47],[19,14],
    [39, 59],[33, 0]])

    [[0, 1, 2],
     [0, 33],
     [3, 11, 7],
     [4, 9, 14],
     [14, 19],
     [15, 47, 31],
     [16, 33],
     [16, 50],
     [19, 59, 39],
     [20, 41],
     [20, 62]]
    """
    try:
        import networkx as NX

    except ImportError:
        print("\n***********************************************************")
        print(
            "networkx module is missing! Some functions may not work...\nPlease install it at http://networkx.github.io/")
        print("***********************************************************\n")
        return None

    sizemat = np.amax(pairs)
    adjencymat = np.zeros((sizemat + 1, sizemat + 1))

    pairs = np.array(pairs, dtype=np.int)

    for pair in pairs:
        i, j = pair
        adjencymat[i, j] = 1
        adjencymat[j, i] = 1

    if NX.__version__ <= "0.99":
        GGraw = NX.from_whatever(adjencymat, create_using=NX.Graph())  # old syntax
    else:
        GGraw = NX.to_networkx_graph(adjencymat, create_using=NX.Graph())

    # cliques
    res_sets = []
    for cli in NX.find_cliques(GGraw):
        if len(cli) > 1:
            res_sets.append(cli)

    return res_sets


def Set_dict_frompairs(pairs_index, verbose=0):
    """
    from association pairs of integers return dictionnary of associated integer

    example:
    array([[ 0,  1],[ 0,  2],[ 1,  2],[ 3,  7],[ 3, 11],[ 4,  9],[ 4, 14],[ 7, 11],[ 9, 14],
    [15, 31],[15, 47],[16, 33],[16, 50],[19, 39],[19, 59],[20, 41],[20, 62],[31, 47],[19,14],
    [39, 59],[33, 0]])
    ->
    {0: [0, 1, 2, 33, 16, 50],
    3: [3, 7, 11],
    4: [4, 9, 14, 59, 19, 39],
    15: [15, 47, 31],
    20: [20, 41, 62]}
    """
    if len(pairs_index) == 1:
        # print "pairs_index",pairs_index
        res_final = {}
        res_final[pairs_index[0][0]] = [pairs_index[0][0], pairs_index[0][1]]
        return res_final, res_final

    res_dict = {}

    for elem in set(np.ravel(np.array(pairs_index))):
        pairs = return_pair(elem, pairs_index)
        # print "\nelem ",elem," pairs ",pairs
        # print "res_dict",res_dict
        if len(pairs) > 1:
            classmembers = [elem] + pairs.tolist()
            set_min = set()
            for _elem in classmembers:
                if _elem in res_dict:
                    set_min = set_min.union(res_dict[_elem])
                else:
                    set_min.add(_elem)

                res_dict[_elem] = set_min

    if verbose:
        print("res_dict", res_dict)

    res_final = {}

    for key, val in list(res_dict.items()):
        listval = list(val)
        smallest_index = min(listval)

        if smallest_index in res_final:
            res_final[smallest_index].append(key)
        else:
            res_final[smallest_index] = [key]

    return res_final, res_dict


def getCommonPts(XY1, XY2, dist_tolerance=0.5, samelist=False):
    """
    return indices in XY1 and in XY2 of common pts (2D) and
    a flag is closest distances are below dist_tolerance

    :param XY1: list of 2D elements
    :param XY2: list of 2D elements
    :param dist_tolerance: largest distance (in unit of XY1, XY2) to consider two elements close enough
    :param samelist: boolean, default is False (when XY1 and XY2 are different). True if XY1=XY2 to
    find close spots in a single list of points

    :return:
    [0] ind_XY1: index of points in XY1 which are seen in XY2
    [1]  ind_XY2: index of points in XY2 which are seen in XY1
    [2] boolean: True if at least point belong to XY1 and XY2. False if no common points are found (within the tolerance)

    example:
    pA =  [[0, 0], [0, 1], [1, 2], [10, 3], [1, 20], [5, 3]]
    pB= [[14, 1], [1, 2], [1, 20], [15, 1]]

    getCommonPts(pA,pB,0.5)
    => (array([2, 4]), array([1, 2]), True)
    """
    WITHINTOLERANCE = True
    x1, y1 = np.array(XY1).T

    x2, y2 = np.array(XY2).T

    diffx = x1[:, np.newaxis] - x2
    diffy = y1[:, np.newaxis] - y2

    dist = np.hypot(diffx, diffy)

    if samelist:
        # add big distance in diagonal
        np.fill_diagonal(dist, np.amax(dist)+2*dist_tolerance)

    # print('dist.shape', dist.shape)
    _, n2 = dist.shape

    resmin = np.where(dist <= dist_tolerance)

    # closest distance beyond tolerance distance
    #print('resmin in getCommonPts', resmin)
    if len(resmin[0]) == 0:
        posmin = np.argmin(dist)
        ind_XY1, ind_XY2 = posmin // n2, posmin % n2
        # print('closest distance:', dist[ind_XY1, ind_XY2])
        WITHINTOLERANCE = False
    else:
        ind_XY1, ind_XY2 = resmin[0], resmin[1]
    return ind_XY1, ind_XY2, WITHINTOLERANCE


def sortclosestpoints(pt0, pts):
    """return pt index in pts sorted by increasing distance from pt0

    Note: cartesian distance
    """
    x1, y1 = pt0

    x2, y2 = np.array(pts).T

    diffx = x1 - x2
    diffy = y1 - y2

    dist = np.hypot(diffx, diffy)

    sortedindices = np.argsort(dist)
    sortedistances = dist[sortedindices]

    return sortedindices, sortedistances

def sortclosestspots(kf0, kfs, dist_tolerance):
    """return spot or scattered vector kf (2theta, chi) index in kfs sorted by increasing angular distance from kf0

    kf0 and kfs    2theta and chi coordinates in degrees
    """

    tth0, chi0 = kf0

    tths, chis = np.array(kfs).T

    ar1 = np.array([[tth0/2., chi0]])
    ar2 = np.array([tths/2., chis]).T

    angletab = calculdist_from_thetachi(ar1, ar2)[:, 0]

    #print('angletab',angletab)

    sortedindices = np.argsort(angletab)
    sortedistances = angletab[sortedindices]

    return sortedindices, sortedistances


def removeClosePoints_two_sets(XY1, XY2, dist_tolerance=0.5, verbose=0):
    """
    remove the spots in XY1 spots list which are present in XY2 spots list
    within the cartesian distance dist_tolerance

    XY1 : array([[x1,x2,...],[y1,y2,...]])
    XY2 : array([[x1,x2,...],[y1,y2,...]])
    """
    X, Y = np.array(XY1)

    coord_1 = np.array(XY1).T
    coord_2 = np.array(XY2).T

    nb_pts1 = len(coord_1)

    coord12 = np.vstack((coord_1, coord_2))

    #    print "coord_1", coord_1
    #    print "coord_2", coord_2
    # print("coord12", coord12)

    tabdist = ssd.squareform(ssd.pdist(coord12, metric="euclidean"))

    close_pos1 = np.where(tabdist < dist_tolerance)

    i1, j1 = close_pos1

    ionly = i1[np.logical_and(i1 != j1, i1 < nb_pts1)]

    toremove = ionly

    #     nb_of_spots_set1 = len(X)

    #     angular_tab = calculdist_from_thetachi(coord_1, coord_2)
    #
    #     # angular_tab.shape = (len(coord_2),len(coord_1))
    # #    print "angular_tab.shape", angular_tab.shape
    #
    #     close_pos = np.where(angular_tab < dist_tolerance)
    #
    #     i, j = close_pos
    #
    # #    print "len(i)", len(i)
    # #    print "len(j)", len(j)
    # #    # len(i) = len(coord_2)
    # #    print "close_pos", close_pos
    # #    print "i", i
    # #    print "j", j
    #
    #     toremove = j

    if verbose:
        print("nb of spots to remove", len(toremove))

    tokeep = list(set(list(range(nb_pts1))) - set(toremove))

    return X[tokeep], Y[tokeep], tokeep


def mergelistofPoints(XY1, XY2, dist_tolerance=0.5, verbose=0):
    """
    merge two list of points (concatenate) and without duplicates

    keeping one spot from XY1 when two spots from XY1 and XY2 are closer than dist_tolerance
    keeping the first spot (in the list) from XY1 when two spots from XY1 are closer than dist_tolerance

    XY1 : array([[x1,x2,...],[y1,y2,...]])
    XY2 : array([[x1,x2,...],[y1,y2,...]])

    return merged list XY, list spot index of XY1 to delete, list spot index of XY2 to delete
    """

    # concatenate
    c12 = np.concatenate((XY1, XY2), axis=0)
    n1 = len(XY1)
    n2 = len(XY2)
    # then purged from duplicates with localisation of them
    purged_c12, index_todelete_in_c12 = purgeClosePoints2(c12, dist_tolerance, verbose=verbose)
    print("c12", c12)
    print("purged_c12", purged_c12)
    print("index_todelete_in_c12", index_todelete_in_c12)
    print("n1 %d,n2 %d" % (n1, n2))
    print(index_todelete_in_c12 < n1)
    print(index_todelete_in_c12 >= n1)

    index_todelete_in_1 = index_todelete_in_c12[index_todelete_in_c12 < n1]
    index_todelete_in_2 = index_todelete_in_c12[index_todelete_in_c12 >= n1] - n1

    return purged_c12, index_todelete_in_1, index_todelete_in_2


def removeClosePoints_2(Twicetheta, Chi, dist_tolerance=0.5):
    """
    remove very close spots within dist_tolerance

    dist_tolerance (in deg) is a crude criterium since angular distance are computed
    as if 2theta,chi coordinates were cartesian ones !

    NOTE: this can be used for harmonics removal (harmonics enhance too much some sinusoids and
    leads to artefact when searching zone axes by gnomonic-hough transform
    and digital image processing methods)

    TODO: to generalise to cartesian distance
    """
    coord = np.array([Twicetheta, Chi]).T
    angular_tab = calculdist_from_thetachi(coord, coord)

    close_pos = np.where(angular_tab < dist_tolerance)

    i, j = close_pos

    #    print "close_pos", close_pos
    #    print "i", i
    #    print "j", j

    dict_sets = Set_dict_frompairs(np.array([i, j]).T, verbose=0)[0]

    #    print "dict_sets", dict_sets

    toremove = []
    for val in list(dict_sets.values()):
        if len(val) > 1:
            toremove += val[1:]

    tokeep = list(set(list(range(len(Twicetheta)))) - set(toremove))

    return Twicetheta[tokeep], Chi[tokeep], tokeep


def removeClosePoints(X, Y, dist_tolerance=0.5):
    r"""
    remove very close spots within dist_tolerance (cartesian distance)
    """
    coord = np.array([X, Y]).T
    dist_tab = calcdistancetab(coord, coord)

    close_pos = np.where(dist_tab < dist_tolerance)

    i, j = close_pos

    #    print "close_pos", close_pos
    #    print "i", i
    #    print "j", j

    dict_sets = Set_dict_frompairs(np.array([i, j]).T, verbose=0)[0]

    #    print "dict_sets", dict_sets

    toremove = []
    for val in list(dict_sets.values()):
        if len(val) > 1:
            toremove += val[1:]

    tokeep = list(set(list(range(len(X)))) - set(toremove))

    return X[tokeep], Y[tokeep], tokeep


def purgeClosePoints(peaklist, dist_tolerance=0.5):

    """
    remove points in peaklist that are too close one to the other within dist_tolerance
    """
    X, Y, _ = removeClosePoints(peaklist[:, 0], peaklist[:, 1], dist_tolerance=dist_tolerance)

    return np.array([X, Y]).T


def purgeClosePoints2(peaklist, maxdistance, verbose=0):
    """
    return peaks list without peaks closer than pixeldistance (maxdistance)
    """

    if np.shape(peaklist)[0] < 2:
        print("GT.purgeClosePoints2 : shape(peaklist) = ", np.shape(peaklist))
        return peaklist, []

    pixeldistance = maxdistance

    disttable = ssd.pdist(peaklist, "euclidean")

    maxdistance = np.amax(disttable)
    sqdistmatrix = ssd.squareform(disttable)
    # we add on diagonal a large number to avoid the zero (=self interdistance!)
    # that is annoying when finding minimum
    distmatrix = sqdistmatrix + np.eye(sqdistmatrix.shape[0]) * maxdistance

    si, fi = np.where(distmatrix < pixeldistance)

    index_to_delete = np.where(fi > si, fi, si)

    # there could be several spots close to one.
    # so remove duplicates in index_todelete !
    itd = set(index_to_delete.tolist())
    index_todelete = np.array(list(itd))

    if verbose:
        print("disttable", disttable)
        print("maxdistance", maxdistance)
        print("sqdistmatrix", sqdistmatrix)
        print("distmatrix", distmatrix)
        print("index si, fi where distmatrix < pixeldistance", si, fi)
        print("index_todelete", index_todelete)

    purged_pklist = np.delete(peaklist, tuple(index_todelete), axis=0)
    return purged_pklist, index_todelete


def getCommonSpots(file1, file2, toldistance, dirname=None, data1=None, fulloutput=False):
    """
    return nb of spots in common in two list of peaks file1 and file2

    if data1 is provided, file1 is not read

    fulloutput: True, return all results
    """
    if data1 is None:
        data1 = IOLT.read_Peaklist(file1, dirname)

    try:
        data2 = IOLT.read_Peaklist(file2, dirname)
    except IOError:
        print("file %s does not exist" % file2)
        return 0

    # nbspots1 = len(data1)
    # nbspots2 = len(data2)
    # print('nb peaks in data1',nbspots1)
    # print('nb peaks in data2',nbspots2)

    XY1 = data1[:, :2]
    XY2 = data2[:, :2]

    toldistance = 1.0

    res = getCommonPts(XY1, XY2, toldistance)

    nbcommonspots = len(res[0])

    #         print 'nb of common spots (< %.f): '%toldistance, nbcommonspots

    #         print "mean % of common spots: ", 100.*nbcommonspots/((nbspots2+nbspots1)/2.)

    # common spots:
    # x,y in XY1
    inXY1 = XY1[res[0]]
    # x,y in XY2
    inXY2 = XY2[res[1]]
    # dist between x1,y1 and x2,y2
    distances = np.sqrt(np.sum((inXY1.T - inXY2.T) ** 2, axis=0))

    if fulloutput:
        return res[0], res[1], inXY1, inXY2, nbcommonspots, distances

    return nbcommonspots


# def computingFunction(fileindexrange, Parameters_dict=None, saveObject=0):
#     """
#     Core procedure to compute common spots over a list of peaks list files
#     """
#     p = multiprocessing.current_process()
#     print("Starting:", p.name, p.pid)

#     fileprefix = Parameters_dict["prefixfilename"]
#     toldistance = Parameters_dict["toldistance"]
#     dirname = Parameters_dict["dirname"]
#     dataref = Parameters_dict["dataref"]

#     commonspotsnb = []
#     file1 = "dummy"
#     for imageindex in list(range(fileindexrange[0], fileindexrange[1] + 1)):
#         if imageindex >= 185:
#             print("imageindex", imageindex)
#         file2 = fileprefix + "%04d" % (imageindex) + ".dat"
#         commonspotsnb.append(
#             [imageindex, getCommonSpots(file1, file2, toldistance, dirname=dirname, data1=dataref)])

#     return commonspotsnb


# def LaueSpotsCorrelator_multiprocessing(fileindexrange, imageindexref, Parameters_dict=None,
#                                         saveObject=0, nb_of_cpu=5):
#     """
#     launch several processes in parallel
#     """
#     try:
#         if len(fileindexrange) > 2:
#             print("\n\n ---- Warning! file STEP INDEX is SET to 1 !\n\n")
#         index_start, index_final = fileindexrange[:2]
#     except:
#         raise ValueError("Need 2 file indices (integers) in fileindexrange=(indexstart, indexfinal)")

#     fileindexdivision = getlist_fileindexrange_multiprocessing(
#         index_start, index_final, nb_of_cpu)

#     saveObject = 0

#     print("fileindexdivision", fileindexdivision)

#     nbimagesperline = Parameters_dict["nbimagesperline"]
#     # prefixfortitle = Parameters_dict["prefixfilename"]
#     prefixfilename = Parameters_dict["prefixfilename"]
#     dirname = Parameters_dict["dirname"]

#     file1 = prefixfilename + "%04d" % (imageindexref) + ".dat"
#     data1 = IOLT.read_Peaklist(file1, dirname)
#     Parameters_dict["dataref"] = data1

#     computingFunction.__defaults__ = (Parameters_dict, saveObject)

#     pool = multiprocessing.Pool()
#     #     for ii in list(range(len(fileindexdivision)):  # range(nb_of_cpu):
#     #         pool.apply_async(computingFunction, args=(fileindexdivision[ii],), callback=log_result)  # make our results with a map call

#     results = []
#     for ii in list(range(len(fileindexdivision))):  # range(nb_of_cpu):
#         print("ii", ii)
#         # make our results with a map call
#         results.append(pool.apply_async(
#             computingFunction, args=(fileindexdivision[ii],), callback=log_result))

#     pool.close()
#     pool.join()

#     print("results", results)
#     print("HOURRA it's FINISHED")

#     dictcorrelval = {}
#     for k, result in enumerate(results):
#         dictk = dict(result.get())
#         print("dict {}".format(k), dictk)
#         dictcorrelval = dict(list(dictk.items()) + list(dictcorrelval.items()))

#     listindval = []
#     for k, val in dictcorrelval.items():
#         listindval.append([k, val])

#     arr_correl = np.array(listindval)
#     sortedindex = np.argsort(arr_correl[:, 0])
#     myc = arr_correl[sortedindex][:, 1].reshape(
#         (len(arr_correl) / nbimagesperline, nbimagesperline)
#     )

#     return myc, dictcorrelval, listindval


def log_result(result):
    if len(result) == 2:
        print("********************\n\n\n\n %s \n\n\n\n\n******************" % result[1])
        # list_produced_files.append(str(result[1]))
    print("mylog print")

# ------------- -----------  COMBINATORICS -----------------------
def threeindices_up_to_old(n):
    """
    build major hkl indices up to n  (each scanned from -n to n)
    """
    if type(n) != type(5):
        return None
    else:
        if n > 0:

            nbpos = n + 1

            gripos = np.mgrid[0 : n : nbpos * 1j, 0 : n : nbpos * 1j, 0 : n : nbpos * 1j]
            majorindices_pos = np.reshape(gripos.T, (nbpos ** 3, 3))[1:]

            majorindices_neg = -majorindices_pos
            return np.vstack((majorindices_neg, majorindices_pos))


def threeindices_up_to(n, remove_negative_l=False):
    """
    build major hkl indices up to n  (each scanned from -n to n)

    remove_negative_l    : flag to remove vectors hkl with l negative
                        Useful for building nodes of LUT in cubic crystal

    Warning!  hkl = [0,0,0] is removed
    """
    if type(n) != type(5):
        return None
    else:
        if n > 0:

            nbpos = 2 * n + 1

            gripos = np.mgrid[-n : n : nbpos * 1j, -n : n : nbpos * 1j, -n : n : nbpos * 1j]
            majorindices = np.reshape(gripos.T, (nbpos ** 3, 3))

            nbelem = len(majorindices)

            majorindices_000removed = np.delete(majorindices, nbelem // 2, axis=0)

            if remove_negative_l:
                majorindices_000removed = majorindices_000removed[n * nbpos ** 2 :]

            return majorindices_000removed


def twoindices_up_to(n):
    """
    build major hkl indices up to n  (each scanned from -n to n)
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("%s is not a positive integer" % str(n))

    nbpos = 2 * n + 1

    gripos = np.mgrid[-n : n : nbpos * 1j, -n : n : nbpos * 1j]
    indices_pos = np.reshape(gripos.T, (nbpos ** 2, 2))

    return indices_pos


def twoindices_positive_up_to(n, m):
    """
    build  2D integer indices up to n  (each scanned from 0 to n)
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("%s is not a positive integer" % str(n))

    nbpos_n = n + 1
    nbpos_m = m + 1

    gripos = np.mgrid[: n : nbpos_n * 1j, : m : nbpos_m * 1j]
    indices_pos = np.reshape(gripos.T, (nbpos_n * nbpos_m, 2))

    return indices_pos


def closest_array_elements(i, j, array2Dshape, maxdist=1, startingindex=0):
    """find closest (in position) other element in 2D array around element at i, j

    :param i:  int or even float
    :param j:  int or even float

    returns solution:
        - closeelem_ij: (array of i, array of j)
        - correponding list of indices of 1D range indices rearranged in 2D array

    example:
    >>>GT.closest_array_element(5,12,(7,23))
    out: ((array([4, 5, 5, 6]), array([12, 11, 13, 12])),
    array([104, 126, 128, 150]))
    """
    n1, n2 = array2Dshape
    ii, jj = np.indices(array2Dshape)
    distfrom_ij = np.hypot(ii - i, jj - j)
    cond = np.logical_and(distfrom_ij <= maxdist, distfrom_ij > 0)
    closeelem_ij = np.where(cond)

    tabindices = np.arange(startingindex, startingindex + n1 * n2).reshape((n1, n2))
    return closeelem_ij, tabindices[closeelem_ij], distfrom_ij[closeelem_ij]


def best_prior_array_element(i, j, array2Dshape, maxdist=1, startingindex=0, existingabsindices=None):
    """ find closest and prior element of 2D array from given i,j

    prior in the sense of a raster scan

    :return:
         indices i,j and  absolute index of elem

    example:
    >>>GT.best_prior_array_element(5,1,(7,23))
    out: ([4, 1], 93)

    """
    (_cli, _clj), _absoluteindices, _dist = closest_array_elements(i, j, array2Dshape, maxdist, startingindex)

    # filter if asked
    # check if surrounding elements in absoluteindices are available in existingabsindices
    if bool(existingabsindices):

        absoluteindices = []
        cli = []
        clj = []
        dist = []
        for k, abselem in enumerate(_absoluteindices):
            if abselem in existingabsindices:
                absoluteindices.append(abselem)
                cli.append(_cli[k])
                clj.append(_clj[k])
                dist.append(_dist[k])

        dist = np.array(dist)

        close_ij = (cli, clj)
    else:
        absoluteindices, dist = _absoluteindices, _dist
        close_ij = (_cli, _clj)

    ic, jc = close_ij

    b = np.argmin(dist)
    return [ic[b], jc[b]], absoluteindices[b], dist[b]


def GCD(ar_hkl, verbose=0):
    """
    return GCD for each element of an array of hkl:
    """
    HKL = np.array(ar_hkl, dtype=np.int16)
    ts = np.sort(HKL)

    Z, Y, X = ts.T  # sorted by increasing order

    # print "starting ts",np.array([Z,Y,X]).T

    # if there is a zero this happens to lead to wrong result
    # put the highest value instead
    condzeroZ = Z == 0
    condzeroY = Y == 0

    Z[condzeroZ] = X[condzeroZ]
    Y[condzeroY] = X[condzeroY]

    ts = np.sort(np.array([Z, Y, X]).T)

    # print "modified ts",ts

    GCD = np.zeros(len(HKL))

    counter = 0

    # while np.any(np.logical_and(condX,condY)): # there is still at least one non zero remainder
    while counter < len(HKL):
        if verbose:
            print("\n\nnew loop in  while")

        ts = np.sort(np.array([Z, Y, X]).T)
        Z, Y, X = ts.T  # sorted by increasing order

        X = np.remainder(X, Z)
        Y = np.remainder(Y, Z)
        # print "Y",Y
        # print "X",X

        # if X,Y have zero at the same place, i , then GCD[i] = Z[i]
        X0 = X == 0
        Y0 = Y == 0
        # print "X0,Y0", X0,Y0

        indices_finish = np.where(np.logical_and(X0, Y0) == True)[0]

        # print "indices_finish",indices_finish
        if len(indices_finish) > 0:
            for k in indices_finish:
                if GCD[k] == 0:  # then fill GCD, otherwise keep the value
                    GCD[k] = Z[k]
                    if verbose:
                        print("GCD[%d] = Z[%d] = %d" % (k, k, Z[k]))
                    counter += 1

        t = np.array([X, Y, Z]).T
        ts = np.sort(t)
        Z, Y, X = ts.T

        # print "Z",Z
        # print "ts",ts

        # if there is a zero this happens to lead to wrong result
        # put the highest value instead
        condzeroZ = Z == 0
        condzeroY = Y == 0

        Z[condzeroZ] = X[condzeroZ]
        Y[condzeroY] = X[condzeroY]

    return GCD


def threeindicesfamily(n):
    """
    #TODO to be OPTIMIZED
    remove harmonics
    """
    listhkl = []
    for hh in list(range(n + 1)):
        for kk in list(range(n + 1)):
            for ll in list(range(n + 1)):
                if hh >= kk and kk >= ll:
                    listhkl.append([hh, kk, ll])
    return listhkl[1:]


def reduceHKL(ar_hkl):
    """
    return hkl expressed in irreductible element (with h,k,l prime to each other)

    [-4,2,2] -> [2,1,1]
    [-8,0,0] -> [-1,0,0]
    """
    GCDs = GCD(np.abs(ar_hkl))
    # print "GCDs",GCDs
    ra = np.array(np.reshape(GCDs, (len(GCDs), 1)), dtype=np.int16)

    res = np.true_divide(np.array(ar_hkl, dtype=np.int16), ra)

    return res


def find_parallel_hkl(HKLs):
    """
    use the tip: 'better looping than broadcasting ...'
    http://www.scipy.org/EricsBroadcastingDoc

    return mutual cross product of elements in HKLs
    """

    HKLs = np.array(HKLs)
    nb_elem = len(HKLs)
    res = np.zeros((nb_elem, nb_elem, 3))
    for k in list(range(nb_elem)):
        for j in list(range(nb_elem)):
            res[k, j] = np.cross(HKLs[k], HKLs[j])

    return res


def extract2Dslice(center, halfsizes, inputarray2D):
    """
    extract a rectangular 2D slice array from inputarray2D centered on 'center' (value in inputarray2D)

    halfsizes : tuple of 2 integers (half height, half width) ie (half slow axis length, half fast axis length)

    example:

    aa= array([[ 0,  1,  2,  3,  4,  5,  6],
       [ 7,  8,  9, 10, 11, 12, 13],
       [14, 15, 16, 17, 18, 19, 20],
       [21, 22, 23, 24, 25, 26, 27],
       [28, 29, 30, 31, 32, 33, 34],
       [35, 36, 37, 38, 39, 40, 41],
       [42, 43, 44, 45, 46, 47, 48],
       [49, 50, 51, 52, 53, 54, 55]])

    extract2Dslice(25,(2,1),aa)

    array([[10, 11, 12],
       [17, 18, 19],
       [24, 25, 26],
       [31, 32, 33],
       [38, 39, 40]])
    """

    indices_center = np.where(inputarray2D == center)
    #     print "indices_center", indices_center
    if len(indices_center[0]) == 0:
        raise ValueError("value %s is not in array!" % center)
    elif len(indices_center[0]) > 1:
        raise ValueError("value %s is not  unique in array!" % center)

    return extract_array(indices_center, halfsizes, inputarray2D)


def extract_array(indices_center, halfsizes, inputarray2D):
    """
    extract a 2D slice array from inputarray2D

    aa= array([[ 0,  1,  2,  3,  4,  5,  6],
       [ 7,  8,  9, 10, 11, 12, 13],
       [14, 15, 16, 17, 18, 19, 20],
       [21, 22, 23, 24, 25, 26, 27],
       [28, 29, 30, 31, 32, 33, 34],
       [35, 36, 37, 38, 39, 40, 41],
       [42, 43, 44, 45, 46, 47, 48],
       [49, 50, 51, 52, 53, 54, 55]])

    extract_array((3,4),(2,1),aa)

    array([[10, 11, 12],
       [17, 18, 19],
       [24, 25, 26],
       [31, 32, 33],
       [38, 39, 40]])
    """
    slowindex_center, fastindex_center = indices_center
    slowindex_halfsize, fastindex_halfsize = halfsizes

    nslow, nfast = inputarray2D.shape

    imax = min(slowindex_center + slowindex_halfsize, nslow - 1)[0]
    imin = max(slowindex_center - slowindex_halfsize, 0)[0]

    jmax = min(fastindex_center + fastindex_halfsize, nfast - 1)[0]
    jmin = max(fastindex_center - fastindex_halfsize, 0)[0]

    print("imin:imax + 1, jmin:jmax + 1", imin, imax + 1, jmin, jmax + 1)

    return inputarray2D[imin : imax + 1, jmin : jmax + 1]


def to2Darray(a, n2):
    """ return a zero padded array from elements of a and new shape =(n1,n2)
    n2 is the number element in the 2nd axis (fast axis)
    """
    l = len(a)
    dt = a.dtype
    n1 = l // n2
    if l % n2 != 0:
        n1 += 1

    b = np.zeros(n1 * n2, dtype=dt)
    b[:l] = a
    return b.reshape((n1, n2))


def splitarray(a, ndivisions):
    """ split array into several subarrays

    ndivisions: tuple of (n1divisions,n2divisions)
    n1divisions and resp. n2divisions subarrayarrays
    along first axis (slow axis) and resp. second axis (fast axis)
    if subarrays dimensions multiples don't fit to input array's shape
    then split is performed from a suited part of input array

    a = [[ 0  1  2  3  4  5]
        [ 6  7  8  9 10 11]
        [12 13 14 15 16 17]
        [18 19 20 21 22 23]]
    c = splitarray(a,2,3)   (division by 2 vertically and 3 horizontally) so 6 subarrays
    c = [[[ 0  1]
        [ 6  7]],
        [[12 13]
        [18 19]],
        [[ 2  3]
        [ 8  9]],
        [[14 15]
        [20 21]],
        [[ 4  5]
        [10 11]],
        [[16 17]
        [22 23]]]

    In this case final ROI index arranged as follows:
    [[ 0  2  4]
    [ 1  3  5]]
    """
    n1, n2 = a.shape

    n1divisions, n2divisions = ndivisions
    remainder2 = n2%n2divisions
    remainder1 = n1%n1divisions

    #print("a.shape", a.shape)
    #print('remainder2',remainder2)
    #print('remainder1',remainder1)

    m1 = n1 // n1divisions
    if remainder1 == 0:
        b = np.vsplit(a, n1divisions)
    else:
        print('m1', m1)
        b = np.vsplit(a[:n1divisions * m1, :], n1divisions)

    b1 = np.array(b)
    d1, _, d3 = b1.shape
    #b1.shape=(d1,d3)
    #print(b1.shape)
    #print(b1)

    # horizontal split (fast (second) axis)
    m2 = n2//n2divisions
    if remainder2 == 0:
        c = np.dsplit(b1, n2divisions)
    else:
        #print('m2', m2)
        #print('cropb1 shape',b1[:,:n2divisions*m2].shape)
        c = np.dsplit(b1[:, :, :n2divisions * m2], n2divisions)

    ar = np.array(c)
    d1, d2, d3, d4 = ar.shape
    ar.shape = (d1 * d2, d3, d4)

    roi_index = np.arange(n1divisions * n2divisions).reshape((n2divisions, n1divisions)).T
    boxsizes = (m1, m2)  # slow axis (vert.), fast axis (horiz)
    return ar, roi_index, boxsizes


def Positiveindices_up_to(n):
    """
    build major positive hkl indices up to n  (each scanned from 0 to n)
    """
    if type(n) != type(5):
        return None
    else:
        if n > 0:

            nbpos = n + 1

            gripos = np.mgrid[0 : n : nbpos * 1j, 0 : n : nbpos * 1j, 0 : n : nbpos * 1j]
            majorindices_pos = np.reshape(gripos.T, (nbpos ** 3, 3))[1:]

            return majorindices_pos


def fct_j(_, q):
    """
    return second called argument
    TODO: likely to exist a python built in function for that
    """
    return q


def indices_in_flatTriuMatrix(n):
    """
    return index in flattened array of triangular up element
    (excluding diagonal ones) in n*n matrix

    ex: [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]] -> [1,2,3,6,7,11]
    """
    #    toc = []
    #    for i in list(range(n - 1):
    #        for j in list(range(i + 1, n):
    #            toc.append(n * i + j)
    #    return np.array(toc)

    return np.where(np.ravel(np.triu(np.fromfunction(fct_j, (n, n)), k=1)) != 0)[0]


def indices_in_TriuMatrix(ar_indices, n):
    """
     convert 1d triangular up (excluded diagonal) indices
     to  i,j 2D array indices of a square n*n M matrix

     :param ar_indices: numpy array witth square shape

    ex: [1,2,3,6,7,11] from a 4*4 matrix
    -> i,j indices [0,1] [0,2] [0,3] [1,2] [1,3] [2,3]
    ( ie M =  [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]) )
    """
    i = ar_indices // n
    j = ar_indices % n
    return np.array([i, j]).T


def convert2indices(ar_indices, array_shape):
    """
    convert 1d indices in 2D indices arranged in a 2D array with array_shape convert indices in  i,j 2D array indices

    Ok for non square matrix

    example: for k in range(10):
                print(k, GT.convert2indices(k,(5,2)))

    0 [0 0]
    1 [0 1]
    2 [1 0]
    3 [1 1]
    4 [2 0]
    5 [2 1]
    6 [3 0]
    7 [3 1]
    8 [4 0]
    9 [4 1]
    """
    slow_dim_i = array_shape[1]
    ar_indices = np.array(ar_indices)
    i = ar_indices // slow_dim_i
    j = ar_indices % slow_dim_i
    return np.array([i, j]).T


# -----  --------------------    3D GEOMETRY
def ShortestLine(P1, P2, P3, P4):
    """
    thanks to http://local.wasp.uwa.edu.au/~pbourke/geometry/lineline3d/L3D.py
    Copyright (c) 2006 Bruce Vaughan, BV Detailing & Design, Inc.

    P1,P2 define the first line L1
    P3,P4 define the second line L2

    Pa=P1+ma(P2-P1)
    Pb=P3+mb(P4-P3)

    returns 2 points Pa,Pb of respectively L1,L2 which are the closest
    """
    # print "P1",P1
    # print "P3",P3
    A = P1 - P3
    B = P2 - P1
    C = P4 - P3

    dCB = np.dot(C, B)
    dAB = np.dot(A, B)
    dAC = np.dot(A, C)
    C2 = np.dot(C, C)
    B2 = np.dot(B, B)
    # TODO: error? why A2 is unused ?
    # A2 = np.dot(A, A)

    ma = 1.0 * (dCB * dAC - dAB * C2) / (B2 * C2 - dCB ** 2)
    mb = 1.0 * (dAB + ma * B2) / (dCB)

    return P1 + ma * B, P3 + mb * C


def ShortestDistance(P1, P2, P3, P4):
    """
    returns distance  between lines L1, L2
    """

    Pa, Pb = ShortestLine(P1, P2, P3, P4)

    return norme_vec(Pa - Pb)


# ------------------  miscellaneous
def getlist_fileindexrange_multiprocessing(index_start, index_final, nb_of_cpu):
    """
    returns list of 2 elements (index_start, index_final) for each cpu

    TODO: implement step != 1
    """
    nb_files = index_final - index_start + 1

    if nb_of_cpu > nb_files:
        nb_of_cpu = nb_files
    step = nb_files // nb_of_cpu  # integer division

    fileindexdivision = []
    st_ind = index_start
    fi_ind = index_start + step - 1
    for _ in list(range(nb_of_cpu)):
        fileindexdivision.append([st_ind, fi_ind])
        st_ind += step
        fi_ind += step

    # last cpu will to the remaining file
    fileindexdivision[-1][-1] = index_final

    return fileindexdivision


# ---- ---------------- Rotation matrices
def rotY(angle):
    """
    return rotation matrix around 2 basis vector (ie Y) with angle in DEGREES
    """
    angrad = angle * np.pi / 180.0
    ca = np.cos(angrad)
    sa = np.sin(angrad)

    return np.array([[1, 0, 0.0], [0.0, ca, sa], [0, -sa, ca]])


def matRot(axis, angle):
    """
    gives rotation matrix around axis and angle deg
    """
    #     print "axis, angle", axis, angle
    axis = np.array(axis)
    norm = 1.0 * np.sqrt(np.sum(axis * axis))
    unitvec = axis / norm

    syme = np.outer(unitvec, unitvec)
    antisyme = np.array([[0, -unitvec[2], unitvec[1]],
                    [unitvec[2], 0.0, -unitvec[0]],
                    [-unitvec[1], unitvec[0], 0.0]])
    angrad = angle * DEG

    return (np.cos(angrad) * IDENTITYMATRIX
        + (1 - np.cos(angrad)) * syme
        + np.sin(angrad) * antisyme)

def propose_orientation_from_hkl(HKL, target2theta=90., randomrotation=False):
    """
    propose one (non unique) orientation matrix to put reflection hkl at 2theta=target2theta, chi =0)

    .. note:: tested for cubic lattice
    """
    hkl_central = np.array(HKL)

    qdir = np.array([-np.sin(target2theta / 2. * DEG), 0, np.cos(target2theta / 2. * DEG)])
    # print('qdir',qdir)

    n_hklcentral = np.sqrt(np.sum(hkl_central**2))
    axrot1 = np.cross(hkl_central, np.array([-1, 0, 0]))
    angrot1 = np.arccos(np.dot(hkl_central, np.array([-1, 0, 0])) / n_hklcentral) / DEG
    matrot1 = matRot(axrot1, angrot1)
    matrot2 = matRot([0, 1, 0], 90. - target2theta / 2.)  # positive angle between qdir and -x
    matrot3 = np.eye(3)
    # random rotation around qdir
    if randomrotation:
        matrot3 = matRot(qdir, np.random.random() * 360 - 180)

    return np.dot(matrot3, np.dot(matrot2, matrot1))


def getRotationAngleFrom2Matrices(A, B):
    """
    return rotation angle (in degree) of operator R between two pure rotations A,B: B=RA
    tr(R)=1+2cos(theta)
    with R = BA-1
    """
    return (np.arccos(0.5 * (np.trace(np.dot(np.array(B), np.linalg.inv(np.array(A)))) - 1))
        * 180.0
        / np.pi)


def randomRotationMatrix():
    """
    return a random rotation matrix
    """
    axis = np.random.randn(3)
    angle = np.random.rand() * 360.0 - 180
    return matRot(axis, angle)


def frommatGLtomat(orientmatrix_fromQuat):  # TODO: to BE DELETED ...?
    """ gives correct orientation matrix from orientation matrix found by a the quaternion:
    fromMatrix_toQuat(extract_rawmatrix_fromGL())
    """
    sqrt2 = np.sqrt(1 / 2.0)
    rotY45 = np.array([[sqrt2, 0, sqrt2], [0, 1, 0], [-sqrt2, 0, sqrt2]])
    rotY45_inv = np.array([[sqrt2, 0, -sqrt2], [0, 1, 0], [sqrt2, 0, sqrt2]])
    # Trick: Oientation matrix from quat (calculated by fromMatrix_toQuat(extract_rawmatrix_fromGL())   ) doesn't need a transposition !!
    result = np.dot(np.dot(rotY45_inv, orientmatrix_fromQuat), rotY45)

    return result


def OrientMatrix_fromGL(filename="matrixfromopenGL.dat"):
    """
    frame basis conversion from OpenGL laue3D.py  to lauetools frame
    """
    sqrt2 = np.sqrt(1 / 2.0)
    rotY45 = np.array([[sqrt2, 0, sqrt2], [0, 1, 0], [-sqrt2, 0, sqrt2]])
    rotY45_inv = np.array([[sqrt2, 0, -sqrt2], [0, 1, 0], [sqrt2, 0, sqrt2]])

    # Laboratory and OPENGL frames are different (only a 45 deg of rotation around Y)

    # result = rotY45_inv.prodmat((extract_rawmatrix_fromGL(extfilename = filename).transpo()).prodmat(rotY45))
    result = np.dot(
        rotY45_inv, np.dot(extract_rawmatrix_fromGL(extfilename=filename).T, rotY45))
    # print "mat3x3fromGL",mat3x3fromGL

    return result


def extract_rawmatrix_fromGL(extfilename="matrixfromopenGL.dat"):
    """
        return orientation matrix from that given by openGL laue3d.py
    """

    filefrompickle = open(extfilename, "r")
    matfromGL = pickle.load(filefrompickle)  # 4x4 matrix of openGL
    filefrompickle.close()
    mat3x3fromGLtemp = np.array(matfromGL[:3, :3])  # extraction of orientation matrix (3x3)
    return mat3x3fromGLtemp


def fromMatrix_to_elemangles(mat):  # PROBLEME D'UNICITE de la decomposition
    # exemple :fromelemangles_toMatrix(fromMatrix_to_elemangles(
    #    [[-0.68125975000000005, 0.093353649999999996, -0.72606490999999995],
    #     [0.68874564999999999, 0.41666590999999997, -0.59331184999999997],
    # [-0.24610646, 0.90477821000000003, 0.34757445999999997]]))
    # ne retourne pas la meme matrice ...
    """ gives the three angles of rotations around X, Y, Z
    (Rz is applied first then Ry and finally Rx)
    A = cosX, B = sinX
    C = cosY, D = sinY
    E = cosZ, F = sinZ
    M= [ [EC  -CF   D] ,
        [ AF  +  BDE     AE-BDF   -BC ] ,
        [BF-ADE     BE + ADF   AC] ]
    """
    thetaY = np.arcsin(mat[0][2])  # from D
    cosY = np.cos(thetaY)

    if cosY != 0:
        # thetaZ = np.atan2(-mat[0][1],mat[0][0])
        # thetaX = np.atan2(-mat[1][2],mat[2][2]) # general procedure
        thetaZ = np.arctan(-mat[0][1] * 1.0 / mat[0][0])
        thetaX = np.arctan(-mat[1][2] * 1.0 / mat[2][2])

    thetaX = thetaX / DEG
    thetaY = thetaY / DEG
    thetaZ = thetaZ / DEG

    return [thetaX, thetaY, thetaZ]


def fromelemangles_toMatrix(threeangles):
    """ gives the orientation matrix from the three angles of rotations
    around X, Y, Z (Rz is applied first then Ry and finally Rx
    A = cosX, B = sinX
    C = cosY, D = sinY
    E = cosZ, F = sinZ
    M= [ [EC  -CF   D] ,
        [ AF+BDE     AE-BDF   -BC ] ,
        [BF-ADE     BE+ADF   AC] ]
    """
    thetaX = threeangles[0] * DEG
    thetaY = threeangles[1] * DEG
    thetaZ = threeangles[2] * DEG

    mat = np.zeros((3, 3))

    A = np.cos(thetaX)
    B = np.sin(thetaX)
    C = np.cos(thetaY)
    D = np.sin(thetaY)
    E = np.cos(thetaZ)
    F = np.sin(thetaZ)

    mat[0][0] = E * C
    mat[0][1] = -C * F
    mat[0][2] = D

    mat[1][0] = A * F + B * D * E
    mat[1][1] = A * E - B * D * F
    mat[1][2] = -B * C

    mat[2][0] = B * F - A * D * E
    mat[2][1] = B * E + A * D * F
    mat[2][2] = A * C

    return mat


def fromEULERangles_toMatrix(threeangles):
    """ gives the orientation matrix from the three EULER angles of rotations
    (Rz is applied first then Rx around the new rotated X,
    then Rz again around a rotated z axis

    Orientation is defined by (X, Y, Z)   = Phi, theta, psi

    A = cosX, B = sinX
    C = cosY, D = sinY
    E = cosZ, F = sinZ
    M= transpose ([ [EA-CBF  EB+CAF    FD] ,
            [-FA-CBE  -FB+CAE   ED ] ,
            [ DB     -DA    C] ])
    """

    thetaX = threeangles[0] * DEG
    thetaY = threeangles[1] * DEG
    thetaZ = threeangles[2] * DEG

    mat = np.zeros((3, 3))

    A = np.cos(thetaX)
    B = np.sin(thetaX)
    C = np.cos(thetaY)
    D = np.sin(thetaY)
    E = np.cos(thetaZ)
    F = np.sin(thetaZ)

    mat[0][0] = E * A - C * B * F
    mat[0][1] = E * B + C * A * F
    mat[0][2] = F * D

    mat[1][0] = -A * F - B * C * E
    mat[1][1] = -F * B + C * A * E
    mat[1][2] = E * D

    mat[2][0] = D * B
    mat[2][1] = -A * D
    mat[2][2] = C

    return mat


def fromEULERangles_toMatrix2(threeangles):
    """ gives the orientation matrix from the three EULER angles of rotations
    (Rz is applied first then Rx around the new rotated X,
    then Rz again around a rotated z axis

    following bunge's euler definition

    Orientation is defined by three angles (X, Y, Z)   = phi1, PHI, phi2

    A = cosX, B = sinX
    C = cosY, D = sinY
    E = cosZ, F = sinZ
    M= transpose([ [EA-CBF  EB+CAF    FD] ,
            [-FA-CBE  -FB+CAE   ED ] ,
            [ DB     -DA    C] ])

    """
    thetaX = threeangles[0] * DEG
    thetaY = threeangles[1] * DEG
    thetaZ = threeangles[2] * DEG

    A = np.cos(thetaX)
    B = np.sin(thetaX)
    C = np.cos(thetaY)
    D = np.sin(thetaY)
    E = np.cos(thetaZ)
    F = np.sin(thetaZ)

    mat1 = np.array([[A, -B, 0], [B, A, 0], [0, 0, 1]])
    mat2 = np.array([[1, 0, 0], [0, C, -D], [0, D, C]])
    mat3 = np.array([[E, -F, 0], [F, E, 0], [0, 0, 1]])

    matproduct = np.dot(mat1, np.dot(mat2, mat3))

    return matproduct.T


def norme(vec1):
    """
    computes norm of one vector
    from O Robach
    TODO: use better generaltools module
    """
    nvec = np.sqrt(np.inner(vec1, vec1))
    return nvec


def matstarlab_to_matstarlabOND(matstarlab):
    """
    transform matrix in a orthonormalized frame
    (Schmid orthogonalisation procedure)

    From O. Robach

    TODO: to be moved to generaltools
    """
    astar1 = matstarlab[:3]
    bstar1 = matstarlab[3:6]
    #    cstar1 = matstarlab[6:]

    astar0 = 1.0 * astar1 / norme(astar1)
    cstar0 = np.cross(astar0, bstar1)
    cstar0 = 1.0 * cstar0 / norme(cstar0)
    bstar0 = np.cross(cstar0, astar0)

    matstarlabOND = np.hstack((astar0, bstar0, cstar0)).T

    # print matstarlabOND

    return matstarlabOND


def calc_Euler_angles(mat3x3):
    """
    Calculates unique 3 euler angles representation of mat3x3
    from O Robach
    """
    # mat3x3 = matrix "minimized" in LT lab frame
    # see F2TC.find_lowest_Euler_Angles_matrix
    # phi 0, theta 1, psi 2

    mat = matline_to_mat3x3(matstarlab_to_matstarlabOND(mat3x3_to_matline(mat3x3)))

    RAD = 180.0 / np.pi

    euler = np.zeros(3, float)
    euler[1] = RAD * np.arccos(mat[2, 2])

    if np.abs(np.abs(mat[2, 2]) - 1.0) < 1e-5:
        # if theta is zero, phi+psi is defined, if theta is pi, phi-psi is defined */
        # put psi = 0 and calculate phi */
        # psi */
        euler[2] = 0.0
        # phi */
        euler[0] = RAD * np.arccos(mat[0, 0])
        if mat[0, 1] < 0.0:
            euler[0] = -euler[0]
    else:
        # psi */
        st = np.sqrt(1 - mat[2, 2] * mat[2, 2])  # sin theta - >0 */
        euler[2] = RAD * np.arccos(mat[1, 2] / st)
        # phi */
        euler[0] = RAD * np.arccos(-mat[2, 1] / st)
        if mat[2, 0] < 0.0:
            euler[0] = 360.0 - euler[0]

        # print "Euler angles phi theta psi (deg)"
        # print euler.round(decimals = 3)

        return euler


def fromMatrix_to_EulerAngles(mat):
    """
    following bunge's euler definition

    Orientation is defined by three angles (X, Y, Z)   = phi1, PHI, phi2

    A = cosX, B = sinX
    C = cosY, D = sinY
    E = cosZ, F = sinZ

    M= [ [EA-CBF  -FA-CBE    DB] ,
        [EB+CAF  -FB+CAE   -DA ] ,
        [ FD     ED        C] ]
    """
    PHI = np.arccos(mat[2, 2]) / DEG
    phi2 = np.arctan(mat[2, 0] / mat[2, 1]) / DEG
    phi1 = np.arctan(-mat[0, 2] / mat[1, 2]) / DEG

    # something strange:
    return -phi2, np.abs(PHI), -phi1
    # would expect to be the inverse function of fromEULERangles_toMatrix!!
#    return phi1, PHI, phi2


def getdirectbasiscosines(UBmatrix_array, B0=np.eye(3), frame="sample",
                                                    vec1=[1, 0, 0], vec2=[0, 1, 0], vec3=[0, 0, 1]):
    """
    returns 3 cosines of for each of three vectors given the orientation matrix array of n matrices (shape = n,3,3)

    By default: 3 vectors are  three direction given in direct unstrained unit cell

    coordinates of vec1 direction
    qvec1 =UB.B0.vec1  in LT frame

    qsample = RotYm40 qLT
    cos1 = qsample.(100)/norme(qsample)
    cos2 = qsample.(010)/norme(qsample)
    cos3 = qsample.(001)/norme(qsample)
    """
    SAMPLETILT = 40.0

    DEG = np.pi / 180.0
    # RotY40 = np.array([[np.cos(SAMPLETILT * DEG), 0, -np.sin(SAMPLETILT * DEG)],
    #         [0, 1, 0],
    #         [np.sin(SAMPLETILT * DEG), 0, np.cos(SAMPLETILT * DEG)]])
    RotYm40 = np.array([
            [np.cos(SAMPLETILT * DEG), 0, np.sin(SAMPLETILT * DEG)],
            [0, 1, 0],
            [-np.sin(SAMPLETILT * DEG), 0, np.cos(SAMPLETILT * DEG)]])

    vecs = np.array([vec1, vec2, vec3]).T

    #     normes_vecs = norme_list(vecs)
    #
    #     nbmatrices= len(UBmatrix_array)

    UBB0s = 1.0 * np.dot(UBmatrix_array, B0)

    qvec1s = np.dot(UBB0s, vecs[0])
    qvec2s = np.dot(UBB0s, vecs[1])
    qvec3s = np.dot(UBB0s, vecs[2])

    normqvec1 = norme_list(qvec1s)
    qvec1s_sample = np.dot(RotYm40, qvec1s.T).T

    cosvec1_X = (
        np.inner(qvec1s_sample, np.array([1, 0, 0])) / normqvec1)  # cosine component along x sample
    cosvec1_Y = (
        np.inner(qvec1s_sample, np.array([0, 1, 0])) / normqvec1)  # cosine component along y sample
    cosvec1_Z = (
        np.inner(qvec1s_sample, np.array([0, 0, 1])) / normqvec1)  # cosine component along z sample

    normqvec2 = norme_list(qvec2s)
    qvec2s_sample = np.dot(RotYm40, qvec2s.T).T

    cosvec2_X = (
        np.inner(qvec2s_sample, np.array([1, 0, 0])) / normqvec2)  # cosine component along x sample
    cosvec2_Y = (
        np.inner(qvec2s_sample, np.array([0, 1, 0])) / normqvec2)  # cosine component along y sample
    cosvec2_Z = (
        np.inner(qvec2s_sample, np.array([0, 0, 1])) / normqvec2)  # cosine component along z sample

    normqvec3 = norme_list(qvec3s)
    qvec3s_sample = np.dot(RotYm40, qvec3s.T).T

    cosvec3_X = (
        np.inner(qvec3s_sample, np.array([1, 0, 0])) / normqvec3)  # cosine component along x sample
    cosvec3_Y = (
        np.inner(qvec3s_sample, np.array([0, 1, 0])) / normqvec3)  # cosine component along y sample
    cosvec3_Z = (
        np.inner(qvec3s_sample, np.array([0, 0, 1])) / normqvec3)  # cosine component along z sample

    cosinesarray = np.array([cosvec1_X,
                            cosvec1_Y,
                            cosvec1_Z,
                            cosvec2_X,
                            cosvec2_Y,
                            cosvec2_Z,
                            cosvec3_X,
                            cosvec3_Y,
                            cosvec3_Z])
    # return array with (nb matrices, 9) shape
    return cosinesarray.T, vecs


# --- ------------ Quaternions
def fromQuat_to_MatrixRot(inputquat):
    """
    Converts the H quaternion quat into a new equivalent 3x3 rotation matrix.
    """
    X = 0
    Y = 1
    Z = 2
    W = 3

    NewObj = np.zeros((3, 3), dtype=float)
    n = np.dot(inputquat, inputquat)
    # print "n",n
    s = 0.0
    if n > 0.0:
        s = 2.0 / n

    xs = inputquat[X] * s
    ys = inputquat[Y] * s
    zs = inputquat[Z] * s
    wx = inputquat[W] * xs
    wy = inputquat[W] * ys
    wz = inputquat[W] * zs
    xx = inputquat[X] * xs
    xy = inputquat[X] * ys
    xz = inputquat[X] * zs
    yy = inputquat[Y] * ys
    yz = inputquat[Y] * zs
    zz = inputquat[Z] * zs
    # This math all comes about by way of algebra, complex math, and trig identities.
    # See Lengyel pages 88-92

    # print "xs",xs
    NewObj[X][X] = 1.0 - (yy + zz)
    NewObj[Y][X] = xy - wz
    NewObj[Z][X] = xz + wy
    NewObj[X][Y] = xy + wz
    NewObj[Y][Y] = 1.0 - (xx + zz)
    NewObj[Z][Y] = yz - wx
    NewObj[X][Z] = xz - wy
    NewObj[Y][Z] = yz + wx
    NewObj[Z][Z] = 1.0 - (xx + yy)

    # return NewObj # this function is originally made to produce a matrix read by OpenGL
    return np.transpose(NewObj)  # transpose to read matrix correctly (OpenGL matrix are transposed of this one)


# def fromQuat_to_matrix_2(quat):
# X, Y, Z, W = quat[0],quat[1],quat[2],quat[3]
# a00 = 1-2*(Y*Y+Z*Z)
# a01 = 2*(X*Y-Z*W)
# a02 = 2*(X*Z+Y*W)

# a10 = 2*(X*Y+Z*W)
# a11 = 1-2*(X*X+Z*Z)
# a12 = 2*(Y*Z-X*W)

# a20 = 2*(X*Z-Y*W)
# a21 = 2*(Y*Z+X*W)
# a22 = 1-2*(X*X+Y*Y)

# return np.array([[a00, a01, a02],[a10, a11, a12],[a20, a21, a22]])


def fromMatrix_toQuat(matrix):
    qw = np.sqrt(1 + matrix[0][0] + matrix[1][1] + matrix[2][2]) / 2.0
    qx = (matrix[2][1] - matrix[1][2]) / (4 * qw)
    qy = (matrix[0][2] - matrix[2][0]) / (4 * qw)
    qz = (matrix[1][0] - matrix[0][1]) / (4 * qw)
    return [qx, qy, qz, qw]


def fromMatrix_toQuat_test(matrix):
    tracemat = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if tracemat > 0.000001:
        SS = np.sqrt(tracemat) * 2.0
        XX = (matrix[1][2] - matrix[2][1]) / SS
        YY = (matrix[2][0] - matrix[0][2]) / SS
        ZZ = (matrix[0][1] - matrix[1][0]) / SS
        WW = SS / 4.0
    qw = np.sqrt(1 + matrix[0][0] + matrix[1][1] + matrix[2][2]) / 2.0
    # TODO: ???
    qx = (matrix[2][1] - matrix[1][2]) / (4 * qw)
    qy = (matrix[0][2] - matrix[2][0]) / (4 * qw)
    qz = (matrix[1][0] - matrix[0][1]) / (4 * qw)

    qx, qy, qz, qw = XX, YY, ZZ, WW
    return [qx, qy, qz, qw]


def fromQuat_to_vecangle(quat):
    """
    from quat = [vec, scalar] = [sin angle / 2 (unitvec(x, y, z)), cos angle / 2]
    gives unitvec and angle of rotation around unitvec
    """
    normvectpart = np.sqrt(quat[0] ** 2 + quat[1] ** 2 + quat[2] ** 2 + quat[3] ** 2)
    # print "nor",normvectpart
    angle = np.arccos(quat[3] / normvectpart) * 2.0  # in radians
    unitvec = np.array(quat[:3]) / np.sin(angle / 2) / normvectpart
    return unitvec, angle


def fromvec_to_directionangles(vec):
    """
    from vec gives two angles of direction (long and lat)

    Long in the plane Oxy
    lat +- from this plane
    """
    # normalisation
    myvec = np.array(vec)
    norm = np.sqrt(np.dot(vec, vec))
    finalvec = myvec * 1.0 / norm

    if np.sqrt(finalvec[0] ** 2 + finalvec[1] ** 2) != 0:
        lat = (np.arctan(finalvec[2] / (np.sqrt(finalvec[0] ** 2 + finalvec[1] ** 2)))
            * 180.0 / np.pi)  # latitude
    else:
        if finalvec[2] > 0:
            lat = 90.0
        else:
            lat = -90.0
    if finalvec[0] != 0:
        if finalvec[0] > 0:
            longit = np.arctan(finalvec[1] / finalvec[0]) * 180.0 / np.pi  # longitude
        else:
            longit = (180. + np.arctan(finalvec[1] / finalvec[0]) * 180.0 / np.pi)  # longitude
    else:
        if finalvec[1] > 0:
            longit = 90.0
        else:
            longit = -90.0

    return longit, lat


def from3rotangles_toQuat(listangles):
    """
    from  angle (rotation angle in deg) and 2 angles (in deg) defining a
    unitvector direction quat=[vec, scalar]=[sin angle / 2 (unitvec(x, y, z)),
                                             cos angle / 2]
    :return: Quatuertion

    .. note::
        take the first 3 elements of listangles
    """
    # print "listangles dans from3rotangles_toQuat",listangles
    rotangle, longit, lat = listangles[:3]

    myquat = np.zeros(4, dtype=float)

    sinhalfrotangle = np.sin(rotangle / 2.0 * np.pi / 180.0)
    coshalfrotangle = np.cos(rotangle / 2.0 * np.pi / 180.0)

    myquat[3] = coshalfrotangle

    xunit = np.cos(lat * np.pi / 180) * np.cos(longit * np.pi / 180)
    yunit = np.cos(lat * np.pi / 180) * np.sin(longit * np.pi / 180)
    zunit = np.sin(lat * np.pi / 180)

    myquat[0] = xunit * sinhalfrotangle
    myquat[1] = yunit * sinhalfrotangle
    myquat[2] = zunit * sinhalfrotangle

    return myquat.tolist()


def fromQuat_to3rotangles(initquat):
    """
    from quaternion=[vec, scalar]=
    [sin angle / 2 (unitvec(x, y, z)), cos angle / 2] ie 4 elements
    returns rotation angles and axis longitude and latitude coordinates
    (in degrees)
    """
    quat = tuple(np.array(initquat))
    unitvec, rotangle = fromQuat_to_vecangle(quat)
    longit, lat = fromvec_to_directionangles(unitvec)

    return rotangle * 180.0 / np.pi, longit, lat


def prodquat(quat1, quat2):
    """
    returns product quaternion od  quat1.quat2
    Seems to be not used in Lauetools package
    """
    return [[quat1[3] * quat2[0] + quat1[0] * quat2[3] + quat1[1] * quat2[2] - quat1[2] * quat2[1],
            quat1[3] * quat2[1] + quat1[1] * quat2[3] + quat1[2] * quat2[0] - quat1[0] * quat2[2],
            quat1[3] * quat2[2] + quat1[2] * quat2[3] + quat1[0] * quat2[1] - quat1[1] * quat2[0],
            quat1[3] * quat2[3] - quat1[0] * quat2[0] - quat1[1] * quat2[1] - quat1[2] * quat2[2]]]

# ----- ------------  plot tools: colormap
COPPER = mplcm.get_cmap("copper")
GIST_EARTH_R = mplcm.get_cmap("gist_earth_r")
JET = mplcm.get_cmap("jet")
GREENS = mplcm.get_cmap("Greens")
REDS = mplcm.get_cmap("Reds")
GREYS = mplcm.get_cmap("Greys")
BWR = mplcm.get_cmap("bwr")
ORRD = mplcm.get_cmap("OrRd")
SEISMIC = mplcm.get_cmap("seismic")
if MATPLOTLIB2p2:
    SPECTRAL = mplcm.get_cmap("Spectral")
    SPECTRAL_R = mplcm.get_cmap("Spectral_r")

else:
    SPECTRAL = mplcm.get_cmap("spectral")
    SPECTRAL_R = mplcm.get_cmap("spectral_r")

# ----  terminal coloredprint
class bcolors:
    WHITE = "\033[97m"
    LIGHTBLUE = "\033[96m"
    CYAN = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"


dictcoloredprint = {"r": bcolors.RED, "g": bcolors.GREEN, "b": bcolors.BLUE, "y": bcolors.YELLOW,
                            "c": bcolors.CYAN}

def pcolor(message, color="r"):
    if color in dictcoloredprint:
        print("%s%s%s" % (dictcoloredprint[color], message, bcolors.ENDC))


def printred(message):
    pcolor(message, "r")


def printgreen(message):
    pcolor(message, "g")


def printyellow(message):
    pcolor(message, "y")


def printcyan(message):
    pcolor(message, "c")


# ---------------- object manipulation  ----------------
def put_on_top_list(top_elements_list, raw_list, forceinsertion=False):
    """
    modify and return raw_list with elements of top_elements_list as first elements

    forceinsertion: True   insert anyway elements in top_elements_list not present in raw_list
    """
    all_elements = copy.copy(raw_list)
    chosen_elements = top_elements_list

    for chosen_element in chosen_elements[::-1]:
        if chosen_element in all_elements:
            index = all_elements.index(chosen_element)
            ce = all_elements.pop(index)
            all_elements.insert(0, ce)
        elif forceinsertion:
            all_elements.insert(0, chosen_element)

    return all_elements


def fromstringtolist(strlist, map_function=int):
    """
    read list written as string

    map_function  function to convert from string to suited element type
    """
    if strlist in ("None", None):
        return None

    list_elems = strlist[1:-1].split(",")
    toreturn = [map_function(elem) for elem in list_elems]
    return toreturn


def read_elems_from_string(strlist, map_function=int, highest_index=0):
    """
    '[1,2,3,5]'-> [1,2,3,5]
    '[0]'-> [0]
    '5' -> 5
    '8:20'  -> [8,9,...,19]
    ':10' -> [0,1,...,9]
    '5:'  ->  [5,6,...,highest_index]
    """
    if len(strlist) == 1:
        return [map_function(strlist)]

    if ":" in strlist:
        spstrlist = strlist.split(":")
        if len(spstrlist) == 2:
            ss, ff = spstrlist
            if not ss:
                ss = 0
            if not ff:
                ff = map_function(ff)
                if ss == 0:
                    ff = min(highest_index, ff)
                if ff == map_function(ss):
                    ff += 1
            return list(list(range(map_function(ss)), map_function(ff)))

    lcore = strlist[1:-1]
    if "," not in lcore and len(lcore) == 1:
        return [map_function(lcore)]
    else:
        return fromstringtolist(strlist, map_function=map_function)


def findfirstnumberpos(s):
    """
    return position in string where the first digit lies and number of digits found

    pattern12345.cor
    => 7
    pattern_0005.cor
    => 8
    """
    if "." in s:
        splits = s.split(".")
        if len(splits) != 2:
            raise ValueError("%s must contain a single '.'!" % s)
        beforedot = splits[-2]
    else:
        # filename without . is a prefix
        beforedot = s

    l = len(beforedot)
    for _k, elem in enumerate(beforedot[::-1]):
        if elem not in "0123456789":
            break

    nbdigits_found = _k
    indexpos_first_digit = l - _k

    #     print "len", l
    #     print "nbdigits_found",nbdigits_found
    #     print "indexpos_first_digit", indexpos_first_digit

    return indexpos_first_digit, nbdigits_found


def lognorm(x, mu, s):

    return (1.0 / (x * s * np.sqrt(2 * np.pi))
        * np.exp(-np.log(x / (1.0 * mu)) ** 2 / (2 * s ** 2)))


def CCDintensitymodel(x):
    """
    function to model response of CCD
    """
    return np.piecewise(x, [x < 7.0, (x >= 7.0) & (x <= 10.0), x > 10.0],
        [lambda x: 0.2, lambda x: 0.8 / 3.0 * x - 5.0 / 3, lambda x: 10.0 / x])

def CCDintensitymodel2(x):
    """
    function to model response of CCD
    """
    return np.piecewise(x, [x < 6.0, (x >= 6.0) & (x <= 10.0), x > 10.0],
        [lambda x: 0.05, lambda x: 0.95 / 4.0 * x - 5.5 / 4, lambda x: 10.0 / np.power(x, 0.95)])


# -------------------------  IN DEVELOPMENT  ----------------
def removeduplicate2(listindice, tabangledist, ang_tol=1.0):
    """ retourne la liste d'indice de proximite sans dupliques
    (remplace par un signe negatif)

    TODO: NOT CHECKED
    """
    verbose = 0

    # print "tabangledist rtgttgr", tabangledist
    list_exp_vers_theo = copy.copy(listindice)
    if verbose:
        print("listindice dans removeduplicate2", listindice)
        print("longueur listindice", len(listindice))
    #    mylist = list(listindice)
    for i in list(range(len(listindice))):  # scan over index of the experimental points
        if verbose:
            print("---------------------------------------")
            print("experimental point index i= ", i)
        nbrevoisinsupp = np.sum(np.where(listindice[i + 1 :] == listindice[i], 1, 0))
        # pour un point theo donne, nbr de voisin exp supplementaire pour qui ce point est le plus proche voisin
        # =0 = >  OK      1 pt theo pour 1 seul point exp
        #
        # print "nbrevoisinsupp",nbrevoisinsupp
        # CLAUSE A SUPPRIMER ?
        if 0:  # nbrevoisinsupp > 0: # il existe plusieurs point exp ayant en premier voisin le meme pt theo
            if verbose:
                print("spot theo d'indice j= ", listindice[i])

            mycopy = copy.copy(listindice)
            localcopy = list(mycopy)
            listvoisin_i = []
            for k in list(range(nbrevoisinsupp + 1)):
                element = localcopy.index(listindice[i])
                if verbose:
                    print(" a le point voisin exp. d'indice", element)
                del localcopy[element]
                listvoisin_i.append(element + k)

            for indiceexp in listvoisin_i:
                list_exp_vers_theo[indiceexp] = -listindice[i] - 10000
                # -10000 in order to keep memory of the old indices even for 0 (which becomes -10000 ( < 0)! because -0 is not  <  0)!!

        elif np.amin(tabangledist[i, :]) > ang_tol:
            # si le spot exp d'indice i possede son spot theo le plus proche a plus de ang_tol deg,
            # je tue le spot exp
            if verbose:
                print("spot exp d'indice ", i, " un voisin trop eloigne!")
            list_exp_vers_theo[i] = -listindice[i] - 10000
        elif np.sum(np.sort(tabangledist[i, :]) < ang_tol) > 1:
            # si il y a deux spot theoriques a moins de ang_tol degres du spot exp. d'indice i,
            # je tue le spot exp
            if verbose:
                print("spot exp indice ", i, " avec plusieurs points theo.")
            list_exp_vers_theo[i] = -listindice[i] - 10000
        elif list_exp_vers_theo[i] > 0:
            if verbose:
                print("RAS ")
            pass
        else:
            pass

    # print "list ",list_exp_vers_theo
    return list_exp_vers_theo


def AngleBetweenKfVectors(HKL1s, HKL2s, B0, UBmatrix, verbose=False):
    """compute angles between all pairs of kf vectors corresponding to HKL1s and HKL2s Vectors

    inputs:
    HKL1s, HKL2s            :  list of n1 3D vectors, list of n2 3D vectors
    Gstar            : metrics , default np.eye(3)
    .. warning::
        NOT FINISHED
                    """
    HKL1r = np.array(HKL1s)
    HKL2r = np.array(HKL2s)

    metrics = None # TO BE DEFINED

    if HKL1r.shape[0] == 1:
        pass

    elif HKL1r.shape == (3,):
        HKL1r = np.array([HKL1r])

    n1 = len(HKL1r)
    n2 = len(HKL2r)
    dstar_square_1 = np.diag(np.inner(np.inner(HKL1r, metrics), HKL1r))
    dstar_square_2 = np.diag(np.inner(np.inner(HKL2r, metrics), HKL2r))
    scalar_product = np.inner(np.inner(HKL1r, metrics), HKL2r) * 1.0

    d1 = np.sqrt(dstar_square_1.reshape((n1, 1))) * 1.0
    d2 = np.sqrt(dstar_square_2.reshape((n2, 1))) * 1.0

    outy = np.outer(d1, d2)
    if verbose:
        print("d1", d1)
        print("d2", d2)
        print("len(d1)", len(d1))
        print("len(d2)", len(d2))
        print("outy", outy)
        print(outy.shape)

        print("scalar_product", scalar_product)
        print(scalar_product.shape)

    ratio = scalar_product / outy
    ratio = np.round(ratio, decimals=7)
    #    print "ratio", ratio
    #    np.putmask(ratio, np.abs(ratio + 1) <= .0001, -1)
    #    np.putmask(ratio, ratio == 0, 0)

    return np.arccos(ratio) / DEG


def nearestValuesindices(A, B):
    """
    A, B : 1D arrays
    return for each element of B the index of element A closest to B

    len(indexInA)=len(B)
    """
    indexInA = np.abs(np.subtract.outer(A, B)).argmin(0)
    return indexInA
