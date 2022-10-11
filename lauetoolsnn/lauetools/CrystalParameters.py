# -*- coding: utf-8 -*-
r"""
This module belong to LaueTools package. It gathers procedures to define crystal
lattice parameters and strain calculations

Main authors are JS Micha, O. Robach, S. Tardif June 2019
"""
import copy
import numpy as np
from numpy.linalg import inv

try:
    from lauetools.dict_LaueTools import dict_Materials
    from lauetools.wyckpos_lauetools import testhklcond_generalrules_array
    from lauetools import generaltools as GT
except:
    from lauetoolsnn.lauetools.dict_LaueTools import dict_Materials
    from lauetoolsnn.lauetools.wyckpos_lauetools import testhklcond_generalrules_array
    import lauetoolsnn.lauetools.generaltools as GT    
    
DEG = np.pi / 180.0
RAD = 1 / DEG
IDENTITYMATRIX = np.eye(3)


def hasCubicSymmetry(key_material, dictmaterials=dict_Materials):
    r"""
    return True if material  has cubic symmetrys in LaueTools Dictionary

    :param key_material: material or structure label
    :type key_material: string

    :return: True or False
    """
    latticeparams = dictmaterials[key_material][1]
    return isCubic(latticeparams)


def isCubic(latticeparams):
    r"""
    :param latticeparams: 6 elements list
    :type latticeparams: iterable object with float or integers elements

    :return: True or False
    """
    if not isinstance(latticeparams, (list, tuple, np.ndarray)):
        raise ValueError("latticeparams is not a list of the 6 lattice parameters")

    if len(latticeparams) != 6:
        raise ValueError("latticeparams is not a list of the 6 lattice parameters")

    Cubic = False
    if (latticeparams[0] == latticeparams[1] and latticeparams[2] == latticeparams[1]
        and latticeparams[3] * 1.0 == 90.0
        and latticeparams[4] * 1.0 == 90.0
        and latticeparams[5] * 1.0 == 90.0):
        Cubic = True

    return Cubic


def ApplyExtinctionrules(HKL, Extinc, verbose=0):
    r"""
    Apply selection rules to hkl reflections to remove forbidden ones

    :param HKL: numpy array (n,3) of [H,K,L]
    :param Extinc: label for extinction (see genHKL_np())

    :returns: numpy array (m,3) of [H,K,L]  m<=n
    """
    if not isinstance(HKL, (np.ndarray,)):
        HKL = np.array(HKL)

    H, K, L = HKL.T

    if verbose:
        print("nb of reflections before extinctions %d" % len(HKL))

    if Extinc.isdecimal():
        array_hkl = testhklcond_generalrules_array(Extinc, HKL)
    # 'dia' adds selection rules to those of 'fcc'
    elif Extinc in ("fcc", "dia"):
        cond = ((H - K) % 2 == 0) * ((H - L) % 2 == 0)
        if Extinc == "dia":
            conddia = ((H + K + L) % 4) != 2
            cond = cond * conddia
        array_hkl = np.take(HKL, np.where(cond == True)[0], axis=0)
    elif Extinc == "bcc":
        cond1 = (H + K + L) % 2 == 0
        array_hkl = np.take(HKL, np.where(cond1 == True)[0], axis=0)
    elif Extinc == "h+k=2n":  # group space 12  I2/m
        cond1 = (H + K) % 2 == 0
        array_hkl = np.take(HKL, np.where(cond1 == True)[0], axis=0)
    elif Extinc == "h+k+l=2n":  # group space 139 indium
        cond1 = (H + K + L) % 2 == 0
        array_hkl = np.take(HKL, np.where(cond1 == True)[0], axis=0)
    elif Extinc == "Al2O3":
        cond1 = (-H + K + L) % 3 == 0
        cond2 = (L) % 2 == 0
        cond = cond1 * cond2
        array_hkl = np.take(HKL, np.where(cond == True)[0], axis=0)
    # elif Extinc == "SG166":
    #     cond = (-H + K + L) % 3 == 0
    #     array_hkl = np.take(HKL, np.where(cond == True)[0], axis=0)
    elif Extinc == "SG63":
        # general existence conditions for any h k l (take!)
        cond1 = (H + K) % 2 == 0
        array_hkl_1 = np.take(HKL, np.where(cond1 == True)[0], axis=0)
        H, K, L = array_hkl_1.T
        # general existence conditions for specific cases h k l (delete!)
        c0kl = (H == 0) * (K % 2 != 0)
        ch0l = (K == 0) * (L % 2 != 0) * (H % 2 != 0)
        chk0 = (L == 0) * ((H + K) % 2 != 0)
        ch00 = (K == 0) * (L == 0) * (H % 2 != 0)
        c0k0 = (H == 0) * (L == 0) * (K % 2 != 0)
        c00l = (H == 0) * (K == 0) * (L % 2 != 0)
        cond = c0kl+ ch0l + chk0 + ch00 + c0k0 + c00l
        array_hkl = np.delete(array_hkl_1, np.where(cond == True)[0], axis=0)
    elif Extinc == "SG62":
        # general existence conditions for any h k l (take!)
        # general existence conditions for specific cases h k l (delete!)
        c0kl = (H == 0) * ((K+L) % 2 != 0)
        chk0 = (L == 0) * (H % 2 != 0)
        ch00 = (K == 0) * (L == 0) * (H % 2 != 0)
        c0k0 = (H == 0) * (L == 0) * (K % 2 != 0)
        c00l = (H == 0) * (K == 0) * (L % 2 != 0)
        cond = c0kl + chk0 + ch00 + c0k0 + c00l
        array_hkl = np.delete(HKL, np.where(cond == True)[0], axis=0)
    elif Extinc == "Ti2AlN":
        # wurtzite condition
        condfirst = (H == K) * ((L % 2) != 0)
        array_hkl_1 = np.delete(HKL, np.where(condfirst == True)[0], axis=0)
        H, K, L = array_hkl_1.T
        # other conditions due to symmetries
        cond_lodd = (L) % 2 == 1
        cond1 = (H - K) % 3 == 0
        cond = cond_lodd * cond1
        array_hkl = np.delete(array_hkl_1, np.where(cond == True)[0], axis=0)
    elif Extinc == "wurtzite":
        cond1 = H - K == 0
        cond2 = (L) % 2 != 0
        cond = cond1 * cond2
        array_hkl = np.delete(HKL, np.where(cond == True)[0], axis=0)
    elif Extinc == "SG141":  # SG 141 ( Sn beta)
        # general existence conditions for any h k l (take!)
        cond1 = (H + K + L) % 2 == 0
        array_hkl_1 = np.take(HKL, np.where(cond1 == True)[0], axis=0)
        H, K, L = array_hkl_1.T
        # general existence conditions for specific cases h k l (delete!)
        chk0 = (L == 0) * ((H) % 2 != 0)*((K) % 2 != 0)
        c0kl = (H == 0) * ((K + L) % 2 != 0)
        chhl = (H == K) * ((2* H + L) % 4 != 0)
        c00l = (H == 0) * (K == 0) * ((L) % 4 != 0)
        ch00 = (K == 0) * (L == 0) * ((H) % 2 != 0)
        chhb0 =  (K == -H) * (L == 0) * ((H) % 2 != 0)
        cond = chk0+ c0kl + chhl + c00l + ch00 + chhb0
        array_hkl = np.delete(array_hkl_1, np.where(cond == True)[0], axis=0)
    elif Extinc == "SG227":  # SG 227 ( Sn alpha, magnetite)
        #print("len start", len(H))
        cond = ((H + K) % 2 == 0) * ((H + L) % 2 == 0) * ((K + L) % 2 == 0)

        array_hkl_1 = np.take(HKL, np.where(cond == True)[0], axis=0)

        H, K, L = array_hkl_1.T

        cond0kl_1 = (H==0) * ((K) % 2 != 0) * ((L) % 2 != 0) * ((K + L) % 4 != 0)
        cond0kl_2 = (K==0)  * ((H) % 2 != 0) * ((L) % 2 != 0) * ((H + L) % 4 != 0)
        cond0kl_3 = (L==0)  * ((H) % 2 != 0) * ((K) % 2 != 0) * ((H + K) % 4 != 0)

        cond0kl = cond0kl_1 + cond0kl_2 + cond0kl_3

        array_hkl_2 = np.delete(array_hkl_1, np.where(cond0kl == True)[0], axis=0)

        H, K, L = array_hkl_2.T

        condhhl_1 = (H == K) * ((H + L) % 2 != 0)
        condhhl_2 = (K == L) * ((L + H) % 2 != 0)
        condhhl_3 = (L == H) * ((H + K) % 2 != 0)

        condhhl = condhhl_1 + condhhl_2 + condhhl_3

        array_hkl_3 = np.delete(array_hkl_2, np.where(condhhl == True)[0], axis=0)

        H, K, L = array_hkl_3.T

        condh00_1 = (K == 0) * (L == 0) * (H % 4 != 0)
        condh00_2 = (L == 0) * (H == 0) * (K % 4 != 0)
        condh00_3 = (H == 0) * (K == 0) * (L % 4 != 0)

        condh00 = condh00_1 + condh00_2 + condh00_3

        array_hkl = np.delete(array_hkl_3, np.where(condh00 == True)[0], axis=0)

    elif Extinc == "SG225":  # SG 227 ( Sn alpha, magnetite)
        #print("len start", len(H))
        cond = ((H + K) % 2 == 0) * ((H + L) % 2 == 0) * ((K + L) % 2 == 0)

        array_hkl_1 = np.take(HKL, np.where(cond == True)[0], axis=0)

        H, K, L = array_hkl_1.T

        cond0kl_1 = (H==0) * ((K) % 2 != 0) * ((L) % 2 != 0)
        cond0kl_2 = (K==0)  * ((H) % 2 != 0) * ((L) % 2 != 0)
        cond0kl_3 = (L==0)  * ((H) % 2 != 0) * ((K) % 2 != 0)

        cond0kl = cond0kl_1 + cond0kl_2 + cond0kl_3

        array_hkl_2 = np.delete(array_hkl_1, np.where(cond0kl == True)[0], axis=0)

        H, K, L = array_hkl_2.T

        condhhl_1 = (H == K) * ((H + L) % 2 != 0)
        condhhl_2 = (K == L) * ((L + H) % 2 != 0)
        condhhl_3 = (L == H) * ((H + K) % 2 != 0)

        condhhl = condhhl_1 + condhhl_2 + condhhl_3

        array_hkl_3 = np.delete(array_hkl_2, np.where(condhhl == True)[0], axis=0)

        H, K, L = array_hkl_3.T

        condh00_1 = (K == 0) * (L == 0) * (H % 2 != 0)
        condh00_2 = (L == 0) * (H == 0) * (K % 2 != 0)
        condh00_3 = (H == 0) * (K == 0) * (L % 2 != 0)

        condh00 = condh00_1 + condh00_2 + condh00_3

        array_hkl = np.delete(array_hkl_3, np.where(condh00 == True)[0], axis=0)
        
    elif Extinc == "SG82":  # SG 227 ( Sn alpha, magnetite)
        #print("len start", len(H))
        cond = (H + K + L) % 2 == 0
        array_hkl_1 = np.take(HKL, np.where(cond == True)[0], axis=0)

        H, K, L = array_hkl_1.T

        cond0kl_1 = (H==0) * ((K + L) % 2 != 0)
        cond0kl_2 = (K==0) * ((H + L) % 2 != 0)
        cond0kl_3 = (L==0) * ((H + K) % 2 != 0)

        cond0kl = cond0kl_1 + cond0kl_2 + cond0kl_3

        array_hkl_2 = np.delete(array_hkl_1, np.where(cond0kl == True)[0], axis=0)

        H, K, L = array_hkl_2.T

        condhhl_1 = (H == K) * ((H + L) % 2 != 0)
        condhhl_2 = (K == L) * ((L + H) % 2 != 0)
        condhhl_3 = (L == H) * ((H + K) % 2 != 0)

        condhhl = condhhl_1 + condhhl_2 + condhhl_3

        array_hkl_3 = np.delete(array_hkl_2, np.where(condhhl == True)[0], axis=0)

        H, K, L = array_hkl_3.T

        condh00_1 = (K == 0) * (L == 0) * (H % 2 != 0)
        condh00_2 = (L == 0) * (H == 0) * (K % 2 != 0)
        condh00_3 = (H == 0) * (K == 0) * (L % 2 != 0)

        condh00 = condh00_1 + condh00_2 + condh00_3

        array_hkl = np.delete(array_hkl_3, np.where(condh00 == True)[0], axis=0)

    elif Extinc == "SG166":  # SG 227 ( Sn alpha, magnetite)
        #print("len start", len(H))
        cond = (-H + K + L) % 3 == 0
        array_hkl_1 = np.take(HKL, np.where(cond == True)[0], axis=0)

        H, K, L = array_hkl_1.T

        cond0kl_1 = (H==0) * ((K + L) % 3 != 0)
        cond0kl_2 = (K==0) * ((H - L) % 3 != 0)
        cond0kl_3 = (L==0) * ((H - K) % 3 != 0)

        cond0kl = cond0kl_1 + cond0kl_2 + cond0kl_3

        array_hkl_2 = np.delete(array_hkl_1, np.where(cond0kl == True)[0], axis=0)

        H, K, L = array_hkl_2.T

        condhhl_1 = (H == K) * ( L % 3 != 0)
        condhhl_2 = (H == -K) * ((H + L) % 3 != 0)
        condhhl_3 = (H == -K) * (K == 0) * (H % 3 != 0)
        
        condhhl = condhhl_1 + condhhl_2 + condhhl_3
        array_hkl_3 = np.delete(array_hkl_2, np.where(condhhl == True)[0], axis=0)

        H, K, L = array_hkl_3.T

        condh00_1 = (K == 0) * (L == 0) * (H % 3 != 0)
        condh00_2 = (L == 0) * (H == 0) * (K % 3 != 0)
        condh00_3 = (H == 0) * (K == 0) * (L % 3 != 0)

        condh00 = condh00_1 + condh00_2 + condh00_3

        array_hkl = np.delete(array_hkl_3, np.where(condh00 == True)[0], axis=0)
        
    elif Extinc == "Al2O3_rhombo":
        cond1 = (H - K) == 0
        cond2 = (L) % 2 == 0
        cond3 = (H + K + L) % 2 == 0
        cond = cond1 * cond2 * cond3
        array_hkl = np.take(HKL, np.where(cond == True)[0], axis=0)

    elif Extinc == "VO2_mono":
        cond1a = K == 0
        cond1b = (L) % 2 != 0
        cond1 = cond1a * cond1b
        array_hkl_1 = np.delete(HKL, np.where(cond1 == True)[0], axis=0)

        H, K, L = array_hkl_1.T
        cond2a = H == 0
        cond2b = L == 0
        cond2c = (K) % 2 != 0
        cond2 = cond2a * cond2b * cond2c
        array_hkl_2 = np.delete(array_hkl_1, np.where(cond2 == True)[0], axis=0)
        
        H, K, L = array_hkl_2.T
        cond3a = H == 0
        cond3b = K == 0
        cond3c = (L) % 2 != 0
        cond3 = cond3a * cond3b * cond3c
        array_hkl = np.delete(array_hkl_2, np.where(cond3 == True)[0], axis=0)
    
    elif Extinc == "VO2_mono2tet":
        cond4a = L == 0
        cond4c = (H + K) % 2 != 0
        cond4 = cond4a * cond4c
        array_hkl_1 = np.delete(HKL, np.where(cond4 == True)[0], axis=0)
        
        H, K, L = array_hkl_1.T
        cond2a = H == K
        cond2c = (L) % 2 != 0
        cond2 = cond2a * cond2c
        array_hkl_2 = np.delete(array_hkl_1, np.where(cond2 == True)[0], axis=0)
        
        H, K, L = array_hkl_2.T
        cond6a = K == 0
        cond6b = L == 0
        cond6c = (H) % 2 != 0
        cond6 = cond6a * cond6b * cond6c
        array_hkl_3 = np.delete(array_hkl_2, np.where(cond6 == True)[0], axis=0)
        
        H, K, L = array_hkl_3.T
        cond5a = H == 0
        cond5b = L == 0
        cond5c = (K) % 2 != 0
        cond5 = cond5a * cond5b * cond5c
        array_hkl_4 = np.delete(array_hkl_3, np.where(cond5 == True)[0], axis=0)
        
        H, K, L = array_hkl_4.T
        cond3a = H == 0
        cond3b = K == 0
        cond3c = (L) % 2 != 0
        cond3 = cond3a * cond3b * cond3c
        array_hkl = np.delete(array_hkl_4, np.where(cond3 == True)[0], axis=0)
        
    elif Extinc == "hcp":
        cond2a = H == K
        cond2c = (L) % 2 != 0
        cond2 = cond2a * cond2c
        array_hkl_1 = np.delete(HKL, np.where(cond2 == True)[0], axis=0)
        
        H, K, L = array_hkl_1.T
        cond3a = H == 0
        cond3b = K == 0
        cond3c = (L) % 2 != 0
        cond3 = cond3a * cond3b * cond3c
        array_hkl = np.delete(array_hkl_1, np.where(cond3 == True)[0], axis=0)
        
    elif Extinc == "VO2_mono2":

        cond1 = (H + K) % 2 != 0

        array_hkl_1 = np.delete(HKL, np.where(cond1 == True)[0], axis=0)

        H, K, L = array_hkl_1.T

        cond2a = K == 0
        cond2c = (H) % 2 != 0

        cond2 = cond2a * cond2c
        array_hkl_2 = np.delete(array_hkl_1, np.where(cond2 == True)[0], axis=0)

        cond3a = H == 0
        cond3c = (K) % 2 != 0
        cond3 = cond3a * cond3c

        array_hkl_3 = np.delete(array_hkl_2, np.where(cond3 == True)[0], axis=0)

        cond4a = L == 0
        cond4c = (H + K) % 2 != 0
        cond4 = cond4a * cond4c

        array_hkl_4 = np.delete(array_hkl_3, np.where(cond4 == True)[0], axis=0)

        cond5a = H == 0
        cond5b = L == 0
        cond5c = (K) % 2 != 0
        cond5 = cond5a * cond5b * cond5c

        array_hkl_5 = np.delete(array_hkl_4, np.where(cond5 == True)[0], axis=0)

        cond6a = K == 0
        cond6b = L == 0
        cond6c = (H) % 2 != 0
        cond6 = cond6a * cond6b * cond6c

        array_hkl = np.delete(array_hkl_5, np.where(cond6 == True)[0], axis=0)

    elif Extinc == "rutile":

        cond1a = H == 0
        cond1b = (K + L) % 2 != 0
        cond1 = cond1a * cond1b

        array_hkl_1 = np.delete(HKL, np.where(cond1 == True)[0], axis=0)

        H, K, L = array_hkl_1.T

        cond2a = H == 0
        cond2b = K == 0
        cond2c = (L) % 2 != 0

        cond2 = cond2a * cond2b * cond2c
        array_hkl_2 = np.delete(array_hkl_1, np.where(cond2 == True)[0], axis=0)

        cond3a = K == 0
        cond3b = L == 0
        cond3c = (H) % 2 != 0
        cond3 = cond3a * cond3b * cond3c

        array_hkl = np.delete(array_hkl_2, np.where(cond3 == True)[0], axis=0)

    # no extinction rules
    else:
        array_hkl = HKL

    # removing the node 000
    pos000 = np.where(np.all((array_hkl == 0), axis=1) == True)[0]
    array_hkl = np.delete(array_hkl, pos000, axis=0)

    if verbose:
        print("nb of reflections after extinctions %d" % len(array_hkl))
    return array_hkl

def GrainParameter_from_Material(key_material, dictmaterials=dict_Materials):
    r"""
    create grain parameters list for the Laue pattern simulation

    Can handle material defined in dictionary by four elements instead of 6 lattice parameters

    :param key_material: material or structure label
    :type key_material: string

    :return: grain (4 elements list),  contains_U (boolean)
    """
    try:
        elem_key, unitCellparameters, Structure_extinction = dictmaterials[key_material]
    except KeyError:
        raise KeyError("Unknown key '%s'for material" % str(key_material))

    if len(unitCellparameters) == 6:  # a,b,c,alpha,beta,gamma

        Bmat = calc_B_RR(unitCellparameters, directspace=1)
        # Gstar = CP.Gstar_from_directlatticeparams(unitCellparameters*)

        grain = [Bmat, Structure_extinction, np.zeros((3, 3)), elem_key]
        contains_U = False
        # U matrix needs to be added in grain

    # 4 operators Da, U, B, Dc
    elif len(unitCellparameters) == 4:
        Da, U, B, Dc = unitCellparameters
        print("Da", Da)
        print(U)
        print(B)
        print(Dc)
        Bmat = np.dot(Dc, B)
        Umat = np.dot(Da, U)
        grain = [Bmat, Structure_extinction, Umat, elem_key]
        contains_U = True

    else:
        raise TypeError("Something is wrong in the material definition in dict_Materials")

    return grain, contains_U


def isOrientMatrix(mat):
    r"""
    test simply if the matrix is inversible

    :param mat: matrix
    :type mat: numpy.array or list

    :return: boolean
    """
    try:
        val = np.linalg.det(np.array(mat))
    except (TypeError, ValueError, np.linalg.LinAlgError):
        raise TypeError("OrientMatrix is not a matrix!")
    if val == 0:
        raise ValueError("OrientMatrix has determinant equals to 0!")
    return True


def Prepare_Grain(key_material, OrientMatrix, force_extinction=None, dictmaterials=dict_Materials):
    r"""
    Constructor of the grain (crystal) parameters for Laue pattern simulation

    if in key_material definition (see dict_Materials) orient matrix is missing
    (i.e. only lattice parameter are input)

    :param key_material: material label
    :type key_material: str

    then list parameter will consider the provided value of the optional
    OrientMatrix argument

    :param force_extinction:  None, use default extinction rules,
            otherwise use other extinction correspondoing to the label
    :type force_extinction: str
    """

    if key_material not in list(dictmaterials.keys()):
        raise KeyError("%s is unknown! You need to create before using"
            " Prepare_Grain." % key_material)

    grain, contains_U = GrainParameter_from_Material(key_material, dictmaterials)

    if force_extinction is not None:
        grain[1] = force_extinction

    if contains_U:  # grain contains an orient matrix
        return grain
    else:
        if OrientMatrix is not None:
            if isOrientMatrix(OrientMatrix):
                grain[2] = OrientMatrix
            return grain
        else:
            raise ValueError("An OrientMatrix is needed !!! in Prepare_Grain()")


def AngleBetweenNormals(HKL1s, HKL2s, Gstar):
    r"""
    compute pairwise angles (in degrees) between reflections or lattice plane normals
    of two sets according to unit cell metrics Gstar

    :param  HKL1s: list of [H1,K1,L1]
    :param  HKL2s: list of [H2,K2,L2]
    :param Gstar: 3*3 matrix corresponding to reciprocal metric tensor of unit cell (as provided by Gstar_from_directlatticeparams())

    :return: array of pairwise angles between reflections
    """
    HKL1r = np.array(HKL1s)
    HKL2r = np.array(HKL2s)

    return GT.AngleBetweenVectors(HKL1r, HKL2r, metrics=Gstar)


def FilterHarmonics_2(hkl, return_indices_toremove=0):
    r"""
    keep only hkl 3d vectors that are representative of direction nh,nk,nl
    for any h,k,l signed integers

    It removes only parallel vector but KEEPs antiparallel vectors (vec , -n vec) with n>0

    :param hkl: array of 3d hkl indices
    :param return_indices_toremove: 1, returns indices of element in hkl that have been removed
    """
    if not isinstance(hkl, (np.ndarray, list)):
        print("hkl", hkl)
        print("len(hkl)", len(hkl))
        print("hkl.type", type(hkl))
        raise ValueError("hkl is not an array!!")
    if isinstance(hkl, list):
        hkl = np.array(hkl)
    if np.array(hkl).shape[0] == 1:
        # print "input array has only one element!"
        return hkl
    # square array with element[i,j] = angle between hkl[i] and hkl[j]
    nb_hkl = len(hkl)
    angles = np.round(AngleBetweenNormals(hkl, hkl, np.eye(3)), decimals=9)
    ra = np.ravel(angles)
    ind_in_flat_a = GT.indices_in_flatTriuMatrix(nb_hkl)
    # 1D array
    angle_pairs = np.take(ra, ind_in_flat_a)

    # index of zeros
    pos_zeros = np.where(angle_pairs == 0)[0]

    # print "pos_zeros",pos_zeros

    if len(pos_zeros) > 0:
        hkls_pairs_index = GT.indices_in_TriuMatrix(ind_in_flat_a[pos_zeros], nb_hkl)
        cliques_of_harmonics = GT.getSets(hkls_pairs_index)

        allelem_in_cliques = []
        for cli in cliques_of_harmonics:
            allelem_in_cliques += cli
        initial_set = set(allelem_in_cliques)
        #        print "initial toremove_set", allelem_in_cliques

        tokeep = []
        for clique in cliques_of_harmonics:
            #            print "clique", clique
            abshkl = np.abs(np.take(hkl, clique, axis=0))
            fond_index = np.argmin(np.sum(abshkl, axis=1))

            #            print abshkl
            #            print fond_index
            #            print "tokeep", clique[fond_index]
            tokeep.append(clique[fond_index])

        toremove = list(initial_set - set(tokeep))
        filtered_hkl = np.delete(hkl, toremove, axis=0)
        if return_indices_toremove:
            return filtered_hkl, toremove
        else:
            return filtered_hkl

    else:
        #         print "hkl doesn't contain harmonics ..."
        if return_indices_toremove:
            return hkl, []
        else:
            return hkl


# ---- -----Unit Cell parameters - Reciprocal and Direct Lattice Parameters  -----
def calc_B_RR(latticeparameters, directspace=1, setvolume=False):
    r"""
    * Calculate B0 matrix (columns = vectors a*,b*,c*) from direct (real) space lattice parameters (directspace=1)
    * Calculate a matrix (columns = vectors a,b,c) from direct (real) space lattice parameters (directspace=0)

    :math:`\boldsymbol q_{ortho}=B_0 {\bf G^*}` where :math:`{\bf G^*}=h{\bf a^*}+k{\bf b^*}+l{\bf c^*}`

    :param latticeparameters:
        * [a,b,c, alpha, beta, gamma]    (angles are in degrees) if directspace=1
        * [a*,b*,c*, alpha*, beta*, gamma*] (angles are in degrees) if directspace=0
    :param directspace:
        * 1 (default) converts  (reciprocal) direct lattice parameters
            to (direct) reciprocal space calculates "B" matrix in the reciprocal space of input latticeparameters
        * 0  converts  (reciprocal) direct lattice parameters to (reciprocal) direct space
            calculates "B" matrix in same space of  input latticeparameters

    :param setvolume:
        * False, sets direct unit cell volume to the true volume from lattice parameters
        * 1,      sets direct unit cell volume to 1
        * 'a**3',  sets direct unit cell volume to a**3
        * 'b**3', sets direct unit cell volume to b**3
        * 'c**3',  sets direct unit cell volume to c**3

    :return: B Matrix (triangular up) from  crystal (reciprocal space) frame to orthonormal frame matrix
    :rtype: numpy array

    B matrix is used in q=U B G* formula or
        as B0  in q= (UB) B0 G*

    after Busing Levy, Acta Crysta 22 (1967), p 457

    .. math::
    
            \left( \begin{matrix}
            a^*  & b^*\cos \gamma^* & c^*\cos beta^*\\
            0  & b^*\sin \gamma^* &-c^*\sin \beta^*\cos \alpha\\
            0 &  0    &      c^*\sin \beta^*\sin \alpha\\
                    \end{matrix} \right)

    with

    .. math :: \cos(\alpha)=(\cos \beta^*\cos \gamma^*-\cos \alpha^*)/(\sin \beta^*\sin \gamma^*)

    and

    .. math :: c^* \sin \beta^* \sin \alpha = 1/c
    """
    B = np.zeros((3, 3), dtype=float)

    lat = 1.0 * np.array(latticeparameters)

    if directspace:  # from lattice param in one space to a matrix in other space
        rlat = dlat_to_rlat(lat, setvolume=setvolume)

        rlat[3:] *= DEG  # convert angles elements in radians

        B[0, 0] = rlat[0]
        B[0, 1] = rlat[1] * np.cos(rlat[5])
        B[1, 1] = rlat[1] * np.sin(rlat[5])
        B[0, 2] = rlat[2] * np.cos(rlat[4])
        B[1, 2] = -rlat[2] * np.sin(rlat[4]) * np.cos(lat[3] * DEG)
        B[2, 2] = rlat[2] * np.sin(rlat[4]) * np.sin(lat[3] * DEG)
        return B

    else:  # from lattice parameters in one space to a matrix in the same space
        lat = np.array(lat)
        lat[3:] *= DEG  # convert angles elements in radians

        # A = B[0,0]x
        # B = B[0,1]x+B[1,1]y
        # C = B[0,2]x+B[1,2]y+B[2,2]
        # A=(a,0,0)
        # B=(bcosgam,bsingam,0)
        # C=(ccosbeta,c/singamma*(cosalpha-cosgamma*cosbeta),0)
        # C=(cx,cy,c*sqrt(1-cx**2-cy**2))
        B[0, 0] = lat[0]

        B[0, 1] = lat[1] * np.cos(lat[5])  # gamma angle
        B[1, 1] = lat[1] * np.sin(lat[5])
        B[0, 2] = lat[2] * np.cos(lat[4])  # beta angle
        B[1, 2] = (lat[2] / np.sin(lat[5]) * (np.cos(lat[3]) - np.cos(lat[5]) * np.cos(lat[4])))
        B[2, 2] = lat[2] * np.sqrt(1.0 - B[0, 2] ** 2 - B[1, 2] ** 2)

        return B


# ---- ----------------------Strain computations --------------
def DeviatoricStrain_LatticeParams(newUBmat, latticeparams, constantlength="a", verbose=0):
    r"""
    Computes deviatoric strain and new direct (real) lattice parameters
    from matrix newUBmat (rotation and deformation)
    considering that one lattice length is chosen to be constant

    Zero strain corresponds to reference state of input `lattice parameters`

    :param newUBmat: (3x3) matrix operator including rotation and deformation
    :param latticeparams: 6 lattice parameters  [a,b,c,:math:`\alpha, \beta, \gamma`] in Angstrom and degrees
    :param constantlength: 'a','b', or 'c' to set one length according to the value in `latticeparams`

    :returns: * 3x3 deviatoric strain tensor)
            * lattice_parameter_direct_strain (direct (real) lattice parameters)
    :rtype: 3x3 numpy array, 6 elements list

    .. note::

        * q = newUBmat . B0 . G*  where B0 (triangular up matrix) comes from lattice parameters input.

        * equivalently, q = UBstar_s . G*
    """
    # q = newUBmat . B0 . G*  where B0 (triangular up matrix) comes from lattice parameters input
    # q = UBstar_s . G*
    # print "new UBs matrix in q= UBs G"
    B0 = calc_B_RR(latticeparams)
    UBstar_s = np.dot(newUBmat, B0)
    # print UBstar_s

    lattice_parameter_reciprocal = matrix_to_rlat(UBstar_s)
    lattice_parameter_direct_strain = dlat_to_rlat(lattice_parameter_reciprocal)

    Bmatrix_direct_strain = calc_B_RR(lattice_parameter_direct_strain, directspace=0)
    Bmatrix_direct_unstrained = calc_B_RR(latticeparams, directspace=0)

    Trans = np.dot(Bmatrix_direct_strain, np.linalg.inv(Bmatrix_direct_unstrained))
    strain_direct = (Trans + Trans.T) / 2.0 - IDENTITYMATRIX

    # print "strain_direct",strain_direct

    devstrain = strain_direct - np.trace(strain_direct) / 3.0 * IDENTITYMATRIX

    # print "deviatoric strain", devstrain

    # print "final lattice parameter"
    # print "a set reference a = %.5f Angstroms"%latticeparams[0]

    # since absolute scale is unknown , lattice parameter are rescaled with a_reference

    # rescaling to set one length of lattice to its original value
    if constantlength == "a":
        index_constant_length = 0
    elif constantlength == "b":
        index_constant_length = 1
    if constantlength == "c":
        index_constant_length = 2
    if verbose:
        print("For comparison: a,b,c are rescaled with respect to the reference value of %s = %f Angstroms"
        % (constantlength, latticeparams[index_constant_length]))
    ratio = (latticeparams[index_constant_length]
        / lattice_parameter_direct_strain[index_constant_length])
    lattice_parameter_direct_strain[0] *= ratio
    lattice_parameter_direct_strain[1] *= ratio
    lattice_parameter_direct_strain[2] *= ratio

    if verbose:
        print("lattice_parameter_direct_strain", lattice_parameter_direct_strain)

    return devstrain, lattice_parameter_direct_strain


def evaluate_strain_fromUBmat(UBmat, key_material, constantlength="a", dictmaterials=dict_Materials, verbose=0):
    r"""
    Evaluate strain from UBmat matrix  (q = UBmat B0 G*)

    :returns:   devstrain, deviatoricstrain_sampleframe, lattice_parameters
    """
    # compute new lattice parameters  -----
    latticeparams = dictmaterials[key_material][1]
    B0matrix = calc_B_RR(latticeparams)

    UBmat = copy.copy(UBmat)

    (devstrain, lattice_parameters) = compute_deviatoricstrain(UBmat, B0matrix, latticeparams)
    # overwrite and rescale possibly lattice lengthes
    lattice_parameters = computeLatticeParameters_from_UB(UBmat, key_material, constantlength, verbose=verbose)
    if verbose:
        print("final lattice_parameters", lattice_parameters)

    deviatoricstrain_sampleframe = strain_from_crystal_to_sample_frame2(devstrain, UBmat)

    # devstrain_sampleframe_round = np.round(
    #     deviatoricstrain_sampleframe * 1000, decimals=3
    # )
    # devstrain_round = np.round(devstrain * 1000, decimals=3)

    return devstrain, deviatoricstrain_sampleframe, lattice_parameters


def compute_deviatoricstrain(newUBmat, B0matrix, latticeparams, verbose=0):
    r"""
    # starting B0matrix corresponding to the unit cell   -----
        latticeparams = DictLT.dict_Materials[key_material][1]
        B0matrix = CP.calc_B_RR(latticeparams)

        q = newUBmat B0 G*
    """
    if verbose: print("new UBs matrix in q= UBs G (s for strain)")

    Bstar_s = np.dot(newUBmat, B0matrix)
    #     print Bstar_s

    lattice_parameter_reciprocal = matrix_to_rlat(Bstar_s)
    lattice_parameter_direct_strain = dlat_to_rlat(lattice_parameter_reciprocal)

    Bmatrix_direct_strain = calc_B_RR(lattice_parameter_direct_strain, directspace=0)
    Bmatrix_direct_unstrained = calc_B_RR(latticeparams, directspace=0)

    Trans = np.dot(Bmatrix_direct_strain, np.linalg.inv(Bmatrix_direct_unstrained))
    # keeping non rotating part (symmetrical)
    strain_direct = (Trans + Trans.T) / 2.0 - np.eye(3)

    if verbose: print("strain_direct", strain_direct)

    devstrain = strain_direct - np.trace(strain_direct) / 3.0 * np.eye(3)

    if verbose: print("deviatoric strain", devstrain)

    return devstrain, lattice_parameter_direct_strain


def computeLatticeParameters_from_UB(UBmatrix, key_material,
                                            constantlength="a", dictmaterials=dict_Materials,
                                            verbose=0):
    r"""
    Computes  direct (real) lattice parameters
    from matrix UBmatrix (rotation and deformation)
    """
    # starting B0matrix corresponding to the unit cell   -----
    latticeparams = dictmaterials[key_material][1]
    B0matrix = calc_B_RR(latticeparams)

    UBmat = copy.copy(UBmatrix)

    (_, lattice_parameter_direct_strain) = compute_deviatoricstrain(UBmat, B0matrix, latticeparams)

    if constantlength == "a":
        index_constant_length = 0
    elif constantlength == "b":
        index_constant_length = 1
    if constantlength == "c":
        index_constant_length = 2

    
    ratio = (latticeparams[index_constant_length]
        / lattice_parameter_direct_strain[index_constant_length])
    lattice_parameter_direct_strain[0] *= ratio
    lattice_parameter_direct_strain[1] *= ratio
    lattice_parameter_direct_strain[2] *= ratio

    if verbose:
        print(
            "For comparison: a,b,c are rescaled with respect to the reference value of %s = %f Angstroms"
            % (constantlength, latticeparams[index_constant_length]))
        print("lattice_parameter_direct_strain", lattice_parameter_direct_strain)

    return lattice_parameter_direct_strain


def computeDirectUnitCell_from_Bmatrix(Bmatrix):
    r"""
    computes direct space unit cell lattice parameters from Bmatrix
    (i.e. columns are unit cell basis vector a,b,c in absolute Lauetools frame).

    Computes a,b,c from a*,b*,c*

    Corresponds to Matrix from real unit cell frame to reciprocal unit cell frame
    V)in a*,b*,c* = P * V)in a,b,c

    :param Bmatrix: Matrix (3x3) whose columns are a*,*b,c* reciprocal unit cell vectors
    expressed in LaueTools frame
    :type Bmatrix: numpy array

    :returns: matrix (3x3) whose columns are a,b,c expressed in LaueTools frame
    :rtype: numpy array
    """
    Bm = np.array(Bmatrix)
    Astar, Bstar, Cstar = Bm.T
    volumestar = 1.0 * np.dot(np.cross(Astar, Bstar), Cstar)
    # print("volumestar", volumestar)
    a = np.cross(Bstar, Cstar) / volumestar
    b = np.cross(Cstar, Astar) / volumestar
    c = np.cross(Astar, Bstar) / volumestar

    return np.array([a, b, c]).T


def mat_to_rlat(matstarlab):
    r"""
    Computes reciprocal lattice parameters from orientation and deformation matrix

    :param matstarlab: 9 elements inline matrix
    :returns: 6 reciprocal unit cell lattice parameters

    .. note:: from Odile's scripts
    """

    rlat = np.zeros(6, float)

    astarlab = matstarlab[0:3]
    bstarlab = matstarlab[3:6]
    cstarlab = matstarlab[6:9]
    rlat[0] = norme(astarlab)
    rlat[1] = norme(bstarlab)
    rlat[2] = norme(cstarlab)
    rlat[5] = np.arccos(np.inner(astarlab, bstarlab) / (rlat[0] * rlat[1]))
    rlat[4] = np.arccos(np.inner(cstarlab, astarlab) / (rlat[2] * rlat[0]))
    rlat[3] = np.arccos(np.inner(bstarlab, cstarlab) / (rlat[1] * rlat[2]))

    # print "rlat = ",rlat

    return rlat

def rlat_to_Bstar(rlat):  # 29May13
    r"""
        # Xcart = Bstar*Xcrist_rec
        # changement de coordonnees pour le vecteur X entre
        # le repere de la maille reciproque Rcrist_rec
        # et le repere OND Rcart associe a Rcrist_rec
        # rlat  reciprocal lattice parameters
        # dlat  direct lattice parameters
        # en radians
        """
    Bstar = np.zeros((3, 3), dtype=float)
    dlat = dlat_to_rlat(rlat)

    Bstar[0, 0] = rlat[0]
    Bstar[0, 1] = rlat[1] * np.cos(rlat[5])
    Bstar[1, 1] = rlat[1] * np.sin(rlat[5])
    Bstar[0, 2] = rlat[2] * np.cos(rlat[4])
    Bstar[1, 2] = -rlat[2] * np.sin(rlat[4]) * np.cos(dlat[3])
    Bstar[2, 2] = 1.0 / dlat[2]

    return Bstar

def fromrealframe_to_reciprocalframe(vector, Bmatrix):
    r"""
    Computes :math:`{\bf X_{recipr}}` (components of `vector`  in reciprocal unit cell vectors basis)
        from  `vector` (:math:`{\bf X_{real}}`, expressed in real a,b,c unit cell vectors basis)

    :param Bmatrix: Matrix whose columns are a*,b*,c* vectors in LaueTools frame
    :param vector: 3 elements array of components in real unit cell a,b,c vectors basis

    .. note::

        v= ua+vb+wc (real unit cell a,b,c)

        [h,k,l] = fromrealframe_to_reciprocalframe([u,v,w], UB,Bmatrix)   i.e. ha*+kb*+lc*

        :math:`{\bf X_{recipr}}= B^{-1} P  {\bf X_{real}}`
    """
    P = computeDirectUnitCell_from_Bmatrix(Bmatrix)
    invBmatrix = np.linalg.inv(Bmatrix)
    return np.dot(np.dot(invBmatrix, P), vector)


def fromreciprocalframe_to_realframe(vector, Bmatrix):
    r"""
    convert components of vector expressed in reciprocal unit cell basis vectors
    to direct (real) ones

    v= ha*+kb*+lc*
    v= fromreciprocalframe_to_realframe([h,k,l])  = [u,v,w]

    Xreal= P-1 * B* Xrecipr

    """
    P = computeDirectUnitCell_from_Bmatrix(Bmatrix)
    invP = np.linalg.inv(P)
    return np.dot(np.dot(invP, Bmatrix), vector)


def DirectUnitCellVectors_from_UB(UB, Bmatrix):
    r"""
    returns matrix whose columns are unit cell basis vector a,b,c in absolute Lauetools frame

    get a,b,c from  astar=UB Bmatrix (1 0 0)
                    bstar=UB Bmatrix (0 1 0)
                    cstar=UB Bmatrix (0 0 1)
    computeDirectUnitCell_from_Bmatrix(dot(UB,Bmatrix))

    ASSUMPTION: small deformation case where UB is considered as a pure rotation
    then dot(UB,computeDirectUnitCell_from_Bmatrix(Bmatrix))
    """
    return computeDirectUnitCell_from_Bmatrix(np.dot(UB, Bmatrix))


def VolumeCell(latticeparameters):
    r"""
    Computes unit cell volume from lattice parameters (either real or reciprocal)

    :param latticeparameters: 6 lattice parameters
    :returns: scalar volume
    """
    a, b, c, alpha, beta, gamma = latticeparameters
    Alp = alpha * DEG
    Bet = beta * DEG
    Gam = gamma * DEG
    return (a * b * c * np.sqrt(1 - np.cos(Alp) ** 2 - np.cos(Bet) ** 2 - np.cos(Gam) ** 2
                                + 2 * np.cos(Alp) * np.cos(Bet) * np.cos(Gam)))


def matstarlab_to_matdirlab(matstarlab, angles_in_deg=1, vec_in_columns=True):
    r"""
    compute the direct lattice matrix (a,b,c vectors) from the reciprocal matrix (a*,b*,c*)

    :param matstarlab: matrix of reciprocal basis vector a*,b*,c* in lab. frame
                    (1rst  column is made of components of a* vector in lab. frame bases)

    :param matdirlab:  matrix of direct (non reciprocal) basis vectors a,b,c in lab. frame
                    (1rst  column is made of components of a vector in lab. frame bases)

    :param vec_in_columns:  boolean, convention to express the 9 elements matrix in line (False) such as
                            first 3 elements correspond to the first column (when reshaped 3*3)

    :returns: matdirlab, reciprocal_lattice_parameters

    .. note::

        * lab frame is following;
        O. Robach's frame  y//ki, z towards CCD (in 2theta=90deg geometry), x=y^z

        * Lauetools (UBmat and B0 definition):
        x//ki, z towards CCD (in 2theta=90deg geometry), y=z^x
    """
    matstarlab = matstarlab[:]

    # if matstarlab is 9 elements array
    if len(matstarlab) == 9:
        matstarlab = np.array(matstarlab).reshape((3, 3))
        if vec_in_columns:
            matstarlab = matstarlab.T

    reciprocal_lattice_parameters = matrix_to_rlat(matstarlab, angles_in_deg=angles_in_deg)
    # print reciprocal_lattice_parameters
    vol = vol_cell(reciprocal_lattice_parameters, angles_in_deg=angles_in_deg)

    astar1 = matstarlab[:, 0]
    bstar1 = matstarlab[:, 1]
    cstar1 = matstarlab[:, 2]

    adir = np.cross(bstar1, cstar1) / vol
    bdir = np.cross(cstar1, astar1) / vol
    cdir = np.cross(astar1, bstar1) / vol

    matdirlab = np.column_stack((adir, bdir, cdir))

    # print " matdirlab =\n", matdirlab.round(decimals=6)

    return matdirlab, reciprocal_lattice_parameters


def matrix_to_rlat(mat, angles_in_deg=1):
    r"""
    Returns RECIPROCAL lattice parameters of the unit cell a*,b*,c* in columns of `mat`

    :param mat: matrix where columns are respectively a*,b*,c* coordinates in orthonormal frame

    :returns: [a*,b*,c*, alpha*, beta*, gamma*] (angles are in degrees)

    .. note::

        Reciprocal lattice parameters are contained in UB matrix : q =  mat G*
    """
    rlat = np.zeros(6)

    rlat[0] = np.sqrt(np.inner(mat[:, 0], mat[:, 0]))  # A
    rlat[1] = np.sqrt(np.inner(mat[:, 1], mat[:, 1]))  # B
    rlat[2] = np.sqrt(np.inner(mat[:, 2], mat[:, 2]))  # C

    rlat[3] = np.arccos(np.inner(mat[:, 1], mat[:, 2]) / (rlat[1] * rlat[2]))  # cos-1 (B,C)/(B,C)
    rlat[4] = np.arccos(np.inner(mat[:, 2], mat[:, 0]) / (rlat[2] * rlat[0]))
    rlat[5] = np.arccos(np.inner(mat[:, 0], mat[:, 1]) / (rlat[0] * rlat[1]))

    if angles_in_deg:
        rlat = rlat * np.array([1, 1, 1, RAD, RAD, RAD])

    return rlat


def dlat_to_rlat(dlat, angles_in_deg=1, setvolume=False):
    r"""
    Computes RECIPROCAL lattice parameters from DIRECT lattice parameters `dlat`

    :param dlat: [a,b,c, alpha, beta, gamma] angles are in degrees
    :param angles_in_deg: 1 when last three parameters are angle in degrees
    (then results angles are in degrees)

    :returns: [a*,b*,c*, alpha*, beta*, gamma*] angles are in degrees

    .. note::

        dlat stands for DIRECT (real space) lattice, rlat for RECIPROCAL lattice

    .. todo:: To remove setvolume
    """
    rlat = np.zeros(6)
    dlat = np.array(dlat)

    if angles_in_deg:
        dlat[3:] *= DEG
        # convert deg into radian

    # Compute reciprocal lattice parameters. The convention used is that
    # a[i]*b[j] = d[ij], i.e. no 2PI's in reciprocal lattice.

    # compute volume of real lattice cell

    #     print 'dlat[:6]', dlat[:6]

    if not setvolume:
        dvolume = (dlat[0] * dlat[1] * dlat[2] * np.sqrt(1
                        + 2 * np.cos(dlat[3]) * np.cos(dlat[4]) * np.cos(dlat[5])
                        - np.cos(dlat[3]) * np.cos(dlat[3])
                        - np.cos(dlat[4]) * np.cos(dlat[4])
                        - np.cos(dlat[5]) * np.cos(dlat[5])))
    elif setvolume == 1:
        dvolume = 1
    elif setvolume == "a**3":
        dvolume = dlat[0] ** 3
    elif setvolume == "b**3":
        dvolume = dlat[1] ** 3
    elif setvolume == "c**3":
        dvolume = dlat[2] ** 3

    # compute reciprocal lattice parameters

    rlat[0] = dlat[1] * dlat[2] * np.sin(dlat[3]) / dvolume
    rlat[1] = dlat[0] * dlat[2] * np.sin(dlat[4]) / dvolume
    rlat[2] = dlat[0] * dlat[1] * np.sin(dlat[5]) / dvolume
    rlat[3] = np.arccos((np.cos(dlat[4]) * np.cos(dlat[5]) - np.cos(dlat[3]))
                        / (np.sin(dlat[4]) * np.sin(dlat[5])))
    rlat[4] = np.arccos((np.cos(dlat[3]) * np.cos(dlat[5]) - np.cos(dlat[4]))
                        / (np.sin(dlat[3]) * np.sin(dlat[5])))
    rlat[5] = np.arccos((np.cos(dlat[3]) * np.cos(dlat[4]) - np.cos(dlat[5]))
                        / (np.sin(dlat[3]) * np.sin(dlat[4])))

    if angles_in_deg:
        rlat[3:] *= RAD
        # convert radians into degrees

    return rlat


def vol_cell(dlat, angles_in_deg=1):
    r"""
    Computes volume of unit cell (direct space) defined from direct lattice parameters
     dlat=[a,b,c, alpha, beta, gamma]
    (lengthes in angstrom, angles are in degrees)

    Volume is in unit**3 of unit given by the first three elements

    .. note::
        from O Robach's scripts
    """
    dlat = dlat[:]

    if angles_in_deg:
        dlat[3:] *= DEG

    volume = (dlat[0] * dlat[1] * dlat[2] * np.sqrt(1
                    + 2 * np.cos(dlat[3]) * np.cos(dlat[4]) * np.cos(dlat[5])
                    - np.cos(dlat[3]) * np.cos(dlat[3])
                    - np.cos(dlat[4]) * np.cos(dlat[4])
                    - np.cos(dlat[5]) * np.cos(dlat[5])))

    if angles_in_deg:
        dlat[3:] *= RAD

    return volume


def dlat_to_dil(dlat_unstrained, dlat_strained, angles_in_deg=1):
    r"""
    Computes hydrostatic expansion coefficient in direct space
    from two lattice parameters sets (strained and reference) in direct space

    .. todo:: check if dil formula -1 in inside or outsite the power 1/3??

    .. note::
        from O Robach's scripts
    """
    volu = vol_cell(dlat_unstrained, angles_in_deg=angles_in_deg)
    vols = vol_cell(dlat_strained, angles_in_deg=angles_in_deg)
    # print "cell volume : unstrained / strained ", volu, vols
    dil = pow((vols / volu), 1.0 / 3.0) - 1.0
    # print "dilatation ((vols/volu)^(1/3) - 1 ) = ", dil
    return dil


def calc_epsp(dlat):
    r"""
    From direct space lattice parameter dlat=[a,b,c, alpha, beta, gamma]
    (alpha, beta, gamma in DEGREES)
    calculates deviatoric strain from initially cubic lattice (a=b=c, alpha=beta=gamma=90)
    and from SMALL deformation

    .. note::

        from O Robach's scripts

    .. todo:: to be deleted since not general
    """

    epsp = np.zeros(6)
    dlat = np.array(dlat)

    dlat[3:] *= DEG

    meanlattice = (dlat[0] + dlat[1] + dlat[2]) / 3.0

    epsp[0] = (dlat[0] - meanlattice) / dlat[0]
    epsp[1] = (dlat[1] - meanlattice) / dlat[0]
    epsp[2] = (dlat[2] - meanlattice) / dlat[0]

    epsp[3] = -(dlat[3] - np.pi / 2) / 2.0
    epsp[4] = -(dlat[4] - np.pi / 2) / 2.0
    epsp[5] = -(dlat[5] - np.pi / 2) / 2.0

    # print "deviatoric strain 11 22 33 -dalf 23, -dbet 13, -dgam 12 \n", epsp
    return epsp


def strain_from_crystal_to_sample_frame2(strain, UBmat, sampletilt=40.0):
    r"""
    Compute strain components in sample frame:
    Zsample perpendicular to sample surface
    Xsample in the same plane thanZ and incoming vector, Xsample is tilted by sampletilt from incoming beam (XLauetools)
    Ysample is horizontal and equal to Ylauetools

    :param strain: 3x3 symmetric array describing the strain in crystal frame
    :param UBmat: 3x3 array, orientation matrix
    :param sampletilt: float, tilt angle in degree
    :return: 3x3 symmetric array describing the strain in sample frame
 
    .. note::
        qxyzLT= UB B0 q_a*b*c*

        if cubic , B0 is proportional to Id, then frame transfrom matrix is UBmat

        Normally pure rotational part of UBmat must be considered...It should be Ok for small deformation in UBmat

        qxyzLT=P q sample  (P = (cos40,0,-sin40),(0,1,0),(sin40,0,cos40))

        then:

        operator_sample= P-1 UB operator_crystal UB-1 P
    """
    P = GT.matRot([0, 1, 0], -sampletilt)
    #    M = np.dot(np.linalg.inv(P), UBmat)
    # P pure rotation matrix : inverse = transposed
    M = np.dot(P.transpose(), UBmat) 
    invM = np.dot(np.linalg.inv(UBmat), P)

    strain_sampleframe = np.dot(M, np.dot(strain, invM))

    return strain_sampleframe

# def strain_from_crystal_to_LaueToolsframe(strain, UBmat):
#     r"""
#     qxyzLT= UB B0 q_a*b*c*

#     if cubic , B0 is proportional to Id, then frame transfrom matrix is UBmat

#     Normally pure rotational part of UBmat must be considered...It should be Ok for small deformation in UBmat

#     operator_LT= UB operator_crystal UB-1

#     """
#     strain_LaueToolsframe = np.dot(UBmat, np.dot(strain, np.linalg.inv(UBmat)))

#     return strain_LaueToolsframe


# def strain_from_crystal_to_sample_frame(
#     deviat_strain, UBmat, omega0=40.0, LaueToolsFrame_for_UBmat=False):
#     r"""
#     Compute deviatoric strain in sample frame from orientation matrix

#     :param deviat_strain: symetric deviatoric strain tensor (matrix 3x3)
#     :param UBmat: orientation matrix with strain

#     :param LaueToolsFrame_for_UBmat: must be False if UBmat is the orientation matrix expressed in OR lab frame

#     :param omega0: tilt angle of the sample surface with respect to the incoming beam  (in degrees)
#     """
#     if not LaueToolsFrame_for_UBmat:
#         # from matstarlab in Odile's frame
#         matstarlab = UBmat
#         if len(matstarlab) != 3:
#             raise TypeError("strain_from_crystal_to_sample_frame function "
#                 "needs 3x3 orientation matrix")
#     else:
#         # from UB matrix in Lauetools frame
#         matstarlab = matstarlabLaueTools_to_matstarlabOR(
#             np.array(UBmat), returnMatrixInLine=False)

#     matdirONDsample = matstarlab_to_matdirONDsample(
#         matstarlab, omega0=omega0, matrix_in_LaueToolsFrame=False)

#     deviatoric_strain_sampleframe = np.dot(
#         matdirONDsample, np.dot(deviat_strain, matdirONDsample.T))

#     return deviatoric_strain_sampleframe

def matstarlab_to_matstarlabOND(matstarlab=None, matLT3x3=None, verbose=0):  # OR
    r"""
    Orthonormalisation of matrix with to a*,b*,c* as columns

    .. note::

        from O Robach's scripts
    """

    if verbose:
        print("entering CP.matstarlab_to_matstarlabOND")
    if matstarlab is not None:
        if verbose:
            print("matstarlab = ", matstarlab)
        astar1 = matstarlab[:3]
        bstar1 = matstarlab[3:6]
        # cstar1 = matstarlab[6:]

    elif matLT3x3 is not None:
        if verbose:
            print("matLT3x3 = ", matLT3x3)
        astar1 = matLT3x3[:, 0]
        bstar1 = matLT3x3[:, 1]
        # cstar1 = matLT3x3[:, 2]

    astar0 = astar1 / GT.norme_vec(astar1)
    cstar0 = np.cross(astar0, bstar1)
    cstar0 = cstar0 / GT.norme_vec(cstar0)
    bstar0 = np.cross(cstar0, astar0)

    if matstarlab is not None:
        matstarlabOND = np.hstack((astar0, bstar0, cstar0)).transpose()
        if verbose > 1:
            print("exiting CP.matstarlab_to_matstarlabOND")
        return matstarlabOND
    elif matLT3x3 is not None:
        matLT3x3OND = np.column_stack((astar0, bstar0, cstar0))
        if verbose > 1:
            print("matLT3x3OND = ", matLT3x3OND)
            print("exiting CP.matstarlab_to_matstarlabOND")
        return matLT3x3OND


def matstarlab_to_matdirONDsample(
    matstarlab, omega0=40.0, matrix_in_LaueToolsFrame=False):
    r"""
    Return matrix whose columns are basis vectors of frame OND related to
    direct crystal expressed in sample frame basis vectors

    matdirONDsample[:,0]   (ie 1rst column) : 3 components of vector a of OND related direct crystal
    in sample frame basis

    matstarlab   :   matrix with columns expressing components of a*,b*,c* in laboratory frame
                            (UBmat expressed in OR lab. frame)

    .. note::

        from O Robach's scripts
    """
    # uc unit cell
    # dir direct
    # uc_dir_OND : cartesian frame obtained by orthonormalizing direct unit cell

    matdirlab, rlat = matstarlab_to_matdirlab(matstarlab, angles_in_deg=1)
    latticeparameters = dlat_to_rlat(rlat, angles_in_deg=1)
    # dir_bmatrix = uc_dir on uc_dir_OND

    #     dir_bmatrix = dlat_to_Bstar(rlat)
    dir_bmatrix = calc_B_RR(latticeparameters, directspace=0)

    # matdirONDlab = uc_dir_OND on lab
    # orientation matrix of the OND frame deduced from a,b,c (direct lattice vectors)
    matdirONDlab = np.dot(matdirlab, np.linalg.inv(dir_bmatrix))

    # matrot:
    omega = omega0 * DEG
    cw = np.cos(omega)
    sw = np.sin(omega)
    # matrix from lab to sample frame
    # each column of matrot is composed of
    # components of one lab basis vector in the sample frame basis vectors
    # Xsample = matrot.Xlab
    if not matrix_in_LaueToolsFrame:
        # OR lab frame
        # rotation de -omega autour de l'axe x pour repasser dans Rsample
        matrot = np.array([[1.0, 0.0, 0.0], [0.0, cw, sw], [0.0, -sw, cw]])

    else:
        # LAuetools lab frame
        matrot = np.array([[cw, 0.0, sw], [0.0, 1.0, 0.0], [-sw, 0, cw]])

    # matdirONDsample = uc_dir_OND on sample
    # rsample = matdirONDsample * ruc_dir_OND
    # orientation of the OND frame related to a,b,c (direct lattice vectors)
    # 1rst column is components of vector basis 'a' of OND frame related to
    # crystal (direct space) expressed in sample frame
    matdirONDsample = np.dot(matrot, matdirONDlab)

    return matdirONDsample


def directlatticeparameters_fromBmatrix(Bmatrix):
    r"""
    computes direct space lattice parameters from Bmatrix (in reciprocal space)

    :param Bmatrix: Bmatrix  columns are a*,b*,c* expressed in lauetools frame
    :type Bmatrix: 3x3 Matrix

    :returns: lattice parameters   [a,b,c,alpha,beta,gamma]

    .. todo:: To be merged with functions above
    """
    lattice_parameter_reciprocal = matrix_to_rlat(Bmatrix)
    lattice_parameter_direct = dlat_to_rlat(lattice_parameter_reciprocal)
    return lattice_parameter_direct


def matstarlabLaueTools_to_matstarlabOR(UBmat, returnMatrixInLine=True):
    r"""
    Convert matrix from lauetools frame: ki//x, z towards CCD (top), y = z^x
                    to ORobach (or equivalent XMAS)'s frame: ki//y, z towards CCD (top), y = z^x

    convert the so called UBmat to matstarlab
    (matstarlab stands for 'matrix of reciprocal unit cell basis vectors in lab. frame')

    see the reciprocal function: matstarlabOR_to_matstarlabLaueTools

    WARNING: convention for writing in line:
    mat = [a0,a1,a2,a3,a4,a5,a6,a7,a8,a9]
    in 3*3 representation: [[a0,a3,a6],[a1,a4,a7],[a2,a5,a8]] !!
    ie a0,a1,a2 forms the 1rst column!
    so take care of transpose the natural reshaped matrix
    mat3x3 = mat.reshape((3,3)).T

    warning : here input UBmat should include B0  # OR

    """
    mm = UBmat

    matstarlab = np.array([-mm[1, 0], mm[0, 0], mm[2, 0],
                            -mm[1, 1], mm[0, 1], mm[2, 1],
                            -mm[1, 2], mm[0, 2], mm[2, 2]])

    matstarlab = matstarlab / GT.norme_vec(matstarlab[:3])

    if returnMatrixInLine:
        return matstarlab
    else:
        return matstarlab.reshape((3, 3)).T


# X_OR = M. X_LT
M_LT_to_OR = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
# inverse operator
M_OR_to_LT = M_LT_to_OR.T


def from_Lauetools_to_ORlabframe(mat):
    return np.dot(M_LT_to_OR, mat)


def from_ORlabframe_to_Lauetools(mat):
    return np.dot(M_OR_to_LT, mat)


def matstarlabOR_to_matstarlabLaueTools(matstarlab, vec_in_columns=True):
    r"""
    reciprocal function of matstarlabLaueTools_to_matstarlabOR
    see corresponding doc

    warning : here resulting UBmat includes B0   # OR
    """
    matstarlab = matstarlab[:]

    # if matstarlab is 9 elements array
    if len(matstarlab) == 9:
        matstarlab = np.array(matstarlab).reshape((3, 3))
        if vec_in_columns:
            matstarlab = matstarlab.T

    #     mm = matstarlab  # line convention
    #
    #     UBmat = np.array([[mm[1], mm[4], mm[7]],
    #                     [-mm[0], -mm[3], -mm[6]],
    #                     [mm[2], mm[5], mm[8]]])

    return from_ORlabframe_to_Lauetools(matstarlab)


def matstarlab_to_matstarsample3x3(matstarlab, omega=None,  # 40.
                        mat_from_lab_to_sample_frame=np.array([[1.0, 0.0, 0.0],
                                                [0.0, 0.766044443118978, 0.642787609686539],
                                                [0.0, -0.642787609686539, 0.766044443118978]])):
    matstarlab3x3 = GT.matline_to_mat3x3(matstarlab)

    if (omega is not None) & (mat_from_lab_to_sample_frame is None):  # deprecated - only for retrocompatibility
        omega = omega * DEG
        # rotation de -omega autour de l'axe x pour repasser dans Rsample
        mat_from_lab_to_sample_frame = np.array([[1.0, 0.0, 0.0],
                                                [0.0, np.cos(omega), np.sin(omega)],
                                                [0.0, -np.sin(omega), np.cos(omega)]])

    matstarsample3x3 = np.dot(mat_from_lab_to_sample_frame, matstarlab3x3)

    # print  "matstarsample3x3 =\n" , matstarsample3x3.round(decimals=6)
    return matstarsample3x3


def matrix_to_HKLs_along_xyz_sample_and_along_xyz_lab(matstarlab=None,  # OR
                                                    UBmat=None,  # LT , UBB0 ici
                                                    omega=None,  # 40. was MG.PAR.omega_sample_frame,
                                                    mat_from_lab_to_sample_frame=None,
                                                    results_in_OR_frames=1,
                                                    results_in_LT_frames=0,
                                                    sampletilt=40.0):
    """Compute HKLs which are parallel to xyz axis of sample frame and lab frame.

    One of the two optional arguments (matstarlab or UBmat) must be input.

    :param matstarlab: ORs matrix, defaults to None
    :type matstarlab: list or array (3x3), optional
    :param UBmat: UB.B0 matrix , defaults to None
    :type UBmat: list or array (3x3), optional
    :param omega: tilt of sample surface, defaults to None
    :type omega: scalar, optional
    :param mat_from_lab_to_sample_frame: [description], defaults to None
    :type mat_from_lab_to_sample_frame: [type], optional
    :param results_in_OR_frames: [description], defaults to 1
    :type results_in_OR_frames: int, optional
    :param results_in_LT_frames: [description], defaults to 0
    :type results_in_LT_frames: int, optional
    :param sampletilt: [description], defaults to 40.0
    :type sampletilt: float, optional
    :return: [description]
    :rtype: [type]

    .. warning::
        UBmat stands for UBB0  (i.e. UB times B0)

    """
    if UBmat is not None:
        matstarlab = matstarlabLaueTools_to_matstarlabOR(UBmat, returnMatrixInLine=True)

    # print("matstarlab = ", matstarlab)

    matstarlab3x3 = GT.matline_to_mat3x3(matstarlab)

    if results_in_OR_frames:
        str_end = "_OR"
    elif results_in_LT_frames:
        str_end = "_LT"

    HKL_xyz = np.zeros((6, 3), float)

    strlist = ["x", "y", "z"]
    list_HKL_names = []
    for i in list(range(3)):
        str1 = "HKL" + strlist[i] + "_lab" + str_end
        list_HKL_names.append(str1)

    for i in list(range(3)):
        str1 = "HKL" + strlist[i] + "_sample" + str_end
        list_HKL_names.append(str1)

    if results_in_OR_frames:

        matstarsample3x3 = matstarlab_to_matstarsample3x3(matstarlab,
                                        omega=omega,
                                        mat_from_lab_to_sample_frame=mat_from_lab_to_sample_frame)

        mat3 = inv(matstarlab3x3)
        for i in list(range(3)):
            HKL_xyz[i] = (mat3[:, i] / max(abs(mat3[:, i]))).round(decimals=3)

        mat2 = inv(matstarsample3x3)
        for i in list(range(3)):
            HKL_xyz[i + 3] = (mat2[:, i] / max(abs(mat2[:, i]))).round(decimals=3)

    elif results_in_LT_frames:

        if UBmat is None:
            UBmat = from_ORlabframe_to_Lauetools(matstarlab3x3)
        #            UBmat_sample =  CP.from_ORlabframe_to_Lauetools(matstarsample3x3) # variante 1
        PP = GT.matRot([0, 1, 0], -sampletilt)
        UBmat_sample = np.dot(PP.transpose(), UBmat)  # variante 2

        mat3 = inv(UBmat)
        for i in list(range(3)):
            HKL_xyz[i] = (mat3[:, i] / max(abs(mat3[:, i]))).round(decimals=3)
        mat2 = inv(UBmat_sample)
        for i in list(range(3)):
            HKL_xyz[i + 3] = (mat2[:, i] / max(abs(mat2[:, i]))).round(decimals=3)

    # print("HKL coordinates of lab and sample frame axes :")
    # for i in list(range(6)):
    #     print(list_HKL_names[i], "\t", HKL_xyz[i, :])

    return (list_HKL_names, HKL_xyz)


# ---------------------    Metric tensor
def ComputeMetricTensor(a, b, c, alpha, beta, gamma):
    r"""
    computes metric tensor G or G* from lattice parameters
    (either direct or reciprocal * ones)

    :param a,b,c,alpha,beta,gamma: lattice parameters (angles in degrees)

    :returns: 3x3 metric tensor

    .. todo:: Clarify G or G*
    """

    Alpha = alpha * DEG
    Beta = beta * DEG
    Gamma = gamma * DEG

    row1 = a * np.array([a, b * np.cos(Gamma), c * np.cos(Beta)])
    row2 = b * np.array([a * np.cos(Gamma), b, c * np.cos(Alpha)])
    row3 = c * np.array([a * np.cos(Beta), b * np.cos(Alpha), c])

    return np.array([row1, row2, row3])


def Gstar_from_directlatticeparams(a, b, c, alpha, beta, gamma):
    r"""
    G  = G*-1
    """
    return inv(ComputeMetricTensor(a, b, c, alpha, beta, gamma))


def DSpacing(HKL, Gstar):
    r"""
    computes dspacing, or interatomic distance between lattice plane),
    or d(hkl)  = 1/d(hkl)* in unit of 1/length in sqrt(Gstar)

    :param HKL: [H,K,L]
    :param Gstar: 3*3 matrix corresponding to reciprocal metric tensor of unit cell
                    (use Gstar_from_directlatticeparams())
    """
    HKLr = np.array(HKL)
    dstar_square = np.dot(np.inner(HKLr, Gstar), HKLr)
    dstar = np.sqrt(dstar_square)
    return 1.0 / dstar


def Gnorm(HKL, Gstar):
    r"""
    compute norm of G = [H,K,L] = d(hkl)* in unit of length in sqrt(Gstar)

    inputs:
    HKL            :  [H,K,L]
    Gstar            : 3*3 matrix corresponding to reciprocal metric tensor of unit cell
                    (use Gstar_from_directlatticeparams())
    """
    HKLr = np.array(HKL)
    dstar_square = np.dot(np.inner(HKLr, Gstar), HKLr)
    return np.sqrt(dstar_square)


def strain_from_metric_difference(Ginit, Gfinal):
    r"""
    does not seem to work ...
    TODO to repair?
    """
    return 0.5 * (Gfinal - Ginit)


# ---- S. Tardif Part   copy from dhkl module  -----------
from numpy.linalg.linalg import norm

# import xray_tools as xrt
"""
boa  means  b over a!!!!
"""

def S11p(boa, coa, alpha, beta, _):
    return boa ** 2 * coa ** 2 * np.sin(alpha) ** 2


def S22p(_, coa, alpha, beta, gamma):
    return coa ** 2 * np.sin(beta) ** 2


def S33p(boa, _, alpha, beta, gamma):
    return boa ** 2 * np.sin(gamma) ** 2


def S23p(boa, coa, alpha, beta, gamma):
    return boa * coa * (np.cos(beta) * np.cos(gamma) - np.cos(alpha))


def S13p(boa, coa, alpha, beta, gamma):
    return boa ** 2 * coa * (np.cos(gamma) * np.cos(alpha) - np.cos(beta))


def S12p(boa, coa, alpha, beta, gamma):
    return boa * coa ** 2 * (np.cos(alpha) * np.cos(beta) - np.cos(gamma))


def V2p(boa, coa, alpha, beta, gamma):
    return (boa ** 2 * coa ** 2
            * (1.0 - np.cos(alpha) ** 2
                - np.cos(beta) ** 2
                - np.cos(gamma) ** 2
                + 2.0 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)))


def fhkl(boa, coa, alpha, beta, gamma, h, k, l):
    return (1.0 / V2p(boa, coa, alpha, beta, gamma)
        * (S11p(boa, coa, alpha, beta, gamma) * h ** 2
            + S22p(boa, coa, alpha, beta, gamma) * k ** 2
            + S33p(boa, coa, alpha, beta, gamma) * l ** 2
            + 2.0 * S23p(boa, coa, alpha, beta, gamma) * k * l
            + 2.0 * S13p(boa, coa, alpha, beta, gamma) * l * h
            + 2.0 * S12p(boa, coa, alpha, beta, gamma) * h * k))


def dhkl(a, b, c, alpha, beta, gamma, h, k, l):
    """
    return lattice spacing in the unit of a

    a,b,c must have the same unit
    alpha, beta, gamma   in radians
    h,k,l Miller indices
    """
    boa, coa = 1.0 * b / a, 1.0 * c / a
    return 1.0 * a / np.sqrt(fhkl(boa, coa, alpha, beta, gamma, h, k, l))


def E2L(energy):
    """
    energy (ev) to wavelength in meter !!!
    """
    return 12398.0 / energy * 1e-10


def L2E(wavel):
    """
     wavelength in meter to energy (ev)
    """
    return 12398.0 / wavel * 1e10


def calculate_a(fitfile, energy, h, k, l):
    """
    compute lattice parameter from lattice spacing measured from energy for hkl reflection
    and deviatoric strain

    :param fitfile:  fitfile object
    :param energy: energy  in eV

    :returns: scalar a
    """
    print("calculate_a function in CrystalParameters")
    # b over a
    boa = fitfile.boa
    # c over a
    coa = fitfile.coa
    alpha = fitfile.alpha * np.pi / 180.0
    beta = fitfile.beta * np.pi / 180.0
    gamma = fitfile.gamma * np.pi / 180.0
    # reciprocal space basis vectors lengthes
    astar = fitfile.astar_prime
    bstar = fitfile.bstar_prime
    cstar = fitfile.cstar_prime
    #  qxoverq = (h * astar + k * bstar + l * cstar)[0] / norm((h * astar + k * bstar + l * cstar))
    # Bragg angle
    print("astar,bstar,cstar", astar, bstar, cstar)
    theta = (np.arccos((h * astar + k * bstar + l * cstar)[0]
            / norm((h * astar + k * bstar + l * cstar))) - np.pi / 2.0)

    print("theta in rad", theta)
    print("theta in degree", theta / np.pi * 180.0)
    normG = norm((h * astar + k * bstar + l * cstar))
    print("normG par norm linalg", normG)
    normGclassic = np.sqrt(np.sum((h * astar + k * bstar + l * cstar) ** 2))
    print("normGclassic", normGclassic)

    a = (E2L(energy) * np.sqrt(fhkl(boa, coa, alpha, beta, gamma, h, k, l))
        / 2.0 / np.sin(theta))
    #     a = E2L(energy) * np.sqrt(fhkl(boa, coa, alpha, beta, gamma, h, k, l)) / 2. / qxoverq
    return a


def scale_fitfile(fitfileObject, energy, denergy, h, k, l, a0=5.6575):
    """
    compute hydrostatic strain and full strain tensor

    from:
    fitfileObject

    energy in eV
    error on energy in eV
    h,k,l Miller indices
    a0  lattice parameter in Angstrom (10-10 m or 0.1 nm)

    Do:
    set fitfileObjectObject strain attributes
    """
    a = 1e10 * calculate_a(fitfileObject, energy, h, k, l)
    print("calculated lattice parameter (Angstr)", a)
    print("unstrained lattice parameter (Angstr)", a0)

    # calculate the hydrostatic strain:
    fitfileObject.hydrostatic_measured = 3 * (
        (a - a0) / a0 - fitfileObject.deviatoric[0, 0])
    fitfileObject.full_strain_measured = (
        fitfileObject.deviatoric + np.eye(3) * fitfileObject.hydrostatic_measured / 3.0)

    if denergy != 0:
        a_lower = 1e10 * calculate_a(fitfileObject, energy + denergy, h, k, l)
        hydrostatic_lower = 3 * ((a_lower - a0) / a0 - fitfileObject.deviatoric[0, 0])
        a_upper = 1e10 * calculate_a(fitfileObject, energy - denergy, h, k, l)
        hydrostatic_upper = 3 * ((a_upper - a0) / a0 - fitfileObject.deviatoric[0, 0])

        fitfileObject.hydrostatic_measured_error = (
            abs(fitfileObject.hydrostatic_measured - hydrostatic_lower)
            + abs(fitfileObject.hydrostatic_measured - hydrostatic_upper)
        ) / 2.0
        fitfileObject.full_strain_measured_error = (
            np.eye(3) * fitfileObject.hydrostatic_measured_error / 3.0)


def calculate_from_UB(UBB0, energy, h, k, l):
    astar_prime = UBB0[:, 0]
    bstar_prime = UBB0[:, 1]
    cstar_prime = UBB0[:, 2]

    a_prime = np.cross(bstar_prime, cstar_prime) / np.dot(
        astar_prime, np.cross(bstar_prime, cstar_prime))
    b_prime = np.cross(cstar_prime, astar_prime) / np.dot(
        bstar_prime, np.cross(cstar_prime, astar_prime))
    c_prime = np.cross(astar_prime, bstar_prime) / np.dot(
        cstar_prime, np.cross(astar_prime, bstar_prime))

    boa = norm(b_prime) / norm(a_prime)
    coa = norm(c_prime) / norm(a_prime)

    alpha = np.arccos(np.dot(b_prime, c_prime) / norm(b_prime) / norm(c_prime))
    beta = np.arccos(np.dot(c_prime, a_prime) / norm(c_prime) / norm(a_prime))
    gamma = np.arccos(np.dot(a_prime, b_prime) / norm(a_prime) / norm(b_prime))

    theta = (np.arccos((h * astar_prime + k * bstar_prime + l * cstar_prime)[0]
            / norm((h * astar_prime + k * bstar_prime + l * cstar_prime))
        ) - np.pi / 2.0)
    a = (E2L(energy) * np.sqrt(fhkl(boa, coa, alpha, beta, gamma, h, k, l)) / 2.0 / np.sin(theta))
    return a


def calculate_energy_from_UB(UBB0, a0, h, k, l):
    """
    calculate energy from UB.B0 matrix, reference a0 direct lattice parameter length

    """
    astar_prime = UBB0[:, 0]
    bstar_prime = UBB0[:, 1]
    cstar_prime = UBB0[:, 2]

    a_prime = np.cross(bstar_prime, cstar_prime) / np.dot(
        astar_prime, np.cross(bstar_prime, cstar_prime))
    b_prime = np.cross(cstar_prime, astar_prime) / np.dot(
        bstar_prime, np.cross(cstar_prime, astar_prime))
    c_prime = np.cross(astar_prime, bstar_prime) / np.dot(
        cstar_prime, np.cross(astar_prime, bstar_prime))

    boa = norm(b_prime) / norm(a_prime)
    coa = norm(c_prime) / norm(a_prime)

    alpha = np.arccos(np.dot(b_prime, c_prime) / norm(b_prime) / norm(c_prime))
    beta = np.arccos(np.dot(c_prime, a_prime) / norm(c_prime) / norm(a_prime))
    gamma = np.arccos(np.dot(a_prime, b_prime) / norm(a_prime) / norm(b_prime))

    theta = (np.arccos((h * astar_prime + k * bstar_prime + l * cstar_prime)[0]
            / norm((h * astar_prime + k * bstar_prime + l * cstar_prime))
        ) - np.pi / 2.0)
    energy = L2E(a0
        / (np.sqrt(fhkl(boa, coa, alpha, beta, gamma, h, k, l)) / 2.0 / np.sin(theta)))
    return energy


def scale_UB(UBB0, energy, h, k, l, a0=5.6575):
    a = 1e10 * calculate_a(UBB0, energy, h, k, l)
    # calculate the hydrostatic strain:
    return (a - a0) / a0

def norme(vec1):
    r"""
    computes norm of a single vector

    .. note::
        from O Robach

    .. todo:: use better generaltools module
    """
    nvec = np.sqrt(np.inner(vec1, vec1))
    return nvec
