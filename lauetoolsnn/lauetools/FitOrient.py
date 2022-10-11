# -*- coding: utf-8 -*-
"""
Module of Lauetools project

JS Micha Feb 2012

module to fit orientation and strain

http://sourceforge.net/projects/lauetools/
"""

__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"

from scipy.optimize import leastsq, least_squares
import numpy as np

np.set_printoptions(precision=15)
from scipy.linalg import qr

try:
    from lauetools import CrystalParameters as CP
    from lauetools import generaltools as GT
    from lauetools import LaueGeometry as F2TC
    from lauetools import dict_LaueTools as DictLT
    from lauetools.dict_LaueTools import DEG
except:
    import lauetoolsnn.lauetools.CrystalParameters as CP
    import lauetoolsnn.lauetools.generaltools as GT
    import lauetoolsnn.lauetools.LaueGeometry as F2TC
    import lauetoolsnn.lauetools.dict_LaueTools as DictLT
    from lauetoolsnn.lauetools.dict_LaueTools import DEG
    
RAD = 1.0 / DEG
IDENTITYMATRIX = np.eye(3)

def remove_harmonic(hkl, uflab, yz):

    # print "removing harmonics from theoretical peak list"
    nn = len(uflab[:, 0])
    isbadpeak = np.zeros(nn, dtype=np.int)
    toluf = 0.05

    for i in list(range(nn)):
        if isbadpeak[i] == 0:
            for j in list(range(i + 1, nn)):
                if isbadpeak[j] == 0:
                    if GT.norme_vec(uflab[j, :] - uflab[i, :]) < toluf:
                        isbadpeak[j] = 1
                        # print "harmonics :"
                        # print hkl[i,:]
                        # print hkl[j,:]

    # print "isbadpeak = ", isbadpeak
    index_goodpeak = np.where(isbadpeak == 0)
    # print "index_goodpeak =", index_goodpeak
    hkl2 = hkl[index_goodpeak]
    uflab2 = uflab[index_goodpeak]
    yz2 = yz[index_goodpeak]
    nspots2 = len(hkl2[:, 0])

    return (hkl2, uflab2, yz2, nspots2, isbadpeak)


def xy_from_Quat(varying_parameter_values, DATA_Q, nspots, varying_parameter_indices,
                                                                        allparameters,
                                                                        initrot=None,
                                                                        vecteurref=IDENTITYMATRIX,
                                                                        pureRotation=0,
                                                                        labXMAS=0,
                                                                        verbose=0,
                                                                        pixelsize=165.0 / 2048,
                                                                        dim=(2048, 2048),
                                                                        kf_direction="Z>0"):
    """
    compute x and y pixel positions of Laue spots given hkl list

    DATA_Q: array of all 3 elements miller indices
    nspots: indices of selected spots of DATA_Q
    initrot: initial orientation matrix (rotation and distorsion)

    varying_parameter_values: array of value that will be taken into account
    varying_parameter_indices: list of indices (element position) of
                            varying parameters in allparameters array
    allparameters: array of 8 elements: 5 first of calibration parameters
                                        and 3 of angles defining quaternion

    WARNING: All miller indices must be entered in DATA_Q, selection is done in xy_from_Quat
    WARNING2: len(varying_parameter_values)=len(varying_parameter_indices)
    returns:

    array of x y pixel positions of Laue peaks
    """

    allparameters.put(varying_parameter_indices, varying_parameter_values)

    calibration_parameters = allparameters[:5]

    # selecting nspots of DATA_Q
    DATAQ = np.take(DATA_Q, nspots, axis=0)
    trQ = np.transpose(DATAQ)  # np.array(Hs, Ks,Ls) for further computations

    if initrot is not None:

        # R is a pure rotation
        # dot(R,Q)=initrot
        # Q may be viewed as lattice distortion
        if pureRotation:  # extract pure rotation matrix from  UB matrix
            R, Q = qr(initrot)
            R = R / np.sign(np.diag(Q))
        else:  # keep UB matrix rotation + distorsion
            R = initrot

        # initial lattice rotation and distorsion (/ cubic structure)  q = U*B * Q
        trQ = np.dot(np.dot(R, vecteurref), trQ)
        # results are qx,qy,qz

    else:
        print("I DONT LIKE INITROT == None")
        print("this must mean that INITROT = Identity ?...")

    if 0:
        angle_Quat = allparameters[5:8]  # three angles of quaternion
        # with sample rotation
        # print "3 angles representation of quaternion",angle_Quat
        Quat = GT.from3rotangles_toQuat(angle_Quat)
        # print "Quat",Quat
        matfromQuat = np.array(GT.fromQuat_to_MatrixRot(Quat))
        #     print "matfromQuat", matfromQuat
    else:
        matfromQuat = np.eye(3)

    Qrot = np.dot(matfromQuat, trQ)  # lattice rotation due to quaternion
    Qrotn = np.sqrt(np.sum(Qrot ** 2, axis=0))  # norms of Q vectors

    twthe, chi = F2TC.from_qunit_to_twchi(1.*Qrot / Qrotn)
    # if verbose:
    #     print("matfromQuat", matfromQuat)
    #     print("tDATA_Q", np.transpose(DATA_Q))
    #     print("Qrot", Qrot)
    #     print("Qrotn", Qrotn)
    #     print("Qrot/Qrotn", Qrot / Qrotn)
    #     print("twthe,chi", twthe, chi)

    X, Y, theta = F2TC.calc_xycam_from2thetachi(twthe,
                                                chi,
                                                calibration_parameters,
                                                verbose=0,
                                                pixelsize=pixelsize,
                                                kf_direction=kf_direction)

    return X, Y, theta, R


def calc_XY_pixelpositions(calibration_parameters, DATA_Q, nspots, UBmatrix=None,
                                                            B0matrix=IDENTITYMATRIX,
                                                            offset=0,
                                                            pureRotation=0,
                                                            labXMAS=0,
                                                            verbose=0,
                                                            pixelsize=0.079,
                                                            dim=(2048, 2048),
                                                            kf_direction="Z>0"):
    """

    must: len(varying_parameter_values)=len(varying_parameter_indices)

    DATA_Q: array of all 3 elements miller indices
    nspots: indices of selected spots of DATA_Q
    UBmatrix:

    WARNING: All miller indices must be entered in DATA_Q, selection is done in xy_from_Quat
    returns:
    """

    # selecting nspots of DATA_Q
    #     print "DATA_Q in calc_XY_pixelpositions", DATA_Q
    #     print "nspots", nspots
    #     print "len(DATA_Q)", len(DATA_Q)
    DATAQ = np.take(DATA_Q, nspots, axis=0)
    trQ = np.transpose(DATAQ)  # np.array(Hs, Ks,Ls) for further computations

    #     print "DATAQ in xy_from_Quat", DATAQ
    if UBmatrix is not None:

        R = UBmatrix

        #  q = UB * B0 * Q
        trQ = np.dot(np.dot(R, B0matrix), trQ)
        # results are qx,qy,qz
    else:
        print("I DON'T LIKE INITROT == None")
        print("this must mean that INITROT = Identity ?...")

    Qrot = trQ  # lattice rotation due to quaternion
    Qrotn = np.sqrt(np.sum(Qrot ** 2, axis=0))  # norms of Q vectors

    twthe, chi = F2TC.from_qunit_to_twchi(Qrot / Qrotn, labXMAS=labXMAS)

    #     print "twthe, chi", twthe, chi

    if verbose:
        print("tDATA_Q", np.transpose(DATA_Q))
        print("Qrot", Qrot)
        print("Qrotn", Qrotn)
        print("Qrot/Qrotn", Qrot / Qrotn)
        print("twthe,chi", twthe, chi)

    X, Y, theta = F2TC.calc_xycam_from2thetachi(
                                            twthe,
                                            chi,
                                            calibration_parameters,
                                            offset=offset,
                                            verbose=0,
                                            pixelsize=pixelsize,
                                            kf_direction=kf_direction)

    return X, Y, theta, R


def error_function_on_demand_calibration(param_calib,
                                        DATA_Q,
                                        allparameters,
                                        arr_indexvaryingparameters,
                                        nspots,
                                        pixX,
                                        pixY,
                                        initrot=IDENTITYMATRIX,
                                        vecteurref=IDENTITYMATRIX,
                                        pureRotation=1,
                                        verbose=0,
                                        pixelsize=165.0 / 2048,
                                        dim=(2048, 2048),
                                        weights=None,
                                        allspots_info=0,
                                        kf_direction="Z>0"):
    """
    #All miller indices must be entered in DATA_Q,
    selection is done in xy_from_Quat with nspots (array of indices)
    # param_orient is three elements array representation of quaternion
    """
    mat1, mat2, mat3 = IDENTITYMATRIX, IDENTITYMATRIX, IDENTITYMATRIX

    invsq2 = 1 / np.sqrt(2)
    AXIS1,AXIS2, AXIS3 = np.array([[invsq2,-.5,.5],[invsq2,.5,-.5],[0,invsq2,invsq2]])

    if 5 in arr_indexvaryingparameters:
        ind1 = np.where(arr_indexvaryingparameters == 5)[0][0]
        if len(arr_indexvaryingparameters) > 1:
            a1 = param_calib[ind1] * DEG
        else:
            a1 = param_calib[0] * DEG
        # print "a1 (rad)= ",a1
        mat1 = np.array([[np.cos(a1), 0, np.sin(a1)],               
                            [0, 1, 0],
                            [-np.sin(a1), 0, np.cos(a1)]])

        mat1 = GT.matRot(AXIS1, a1/DEG)

    if 6 in arr_indexvaryingparameters:
        ind2 = np.where(arr_indexvaryingparameters == 6)[0][0]
        if len(arr_indexvaryingparameters) > 1:
            a2 = param_calib[ind2] * DEG
        else:
            a2 = param_calib[0] * DEG
        # print "a2 (rad)= ",a2
        mat2 = np.array([[1, 0, 0],
                        [0, np.cos(a2), np.sin(a2)],
                        [0, np.sin(-a2), np.cos(a2)]])

        mat2 = GT.matRot(AXIS2, a2/DEG)

    if 7 in arr_indexvaryingparameters:
        ind3 = np.where(arr_indexvaryingparameters == 7)[0][0]
        if len(arr_indexvaryingparameters) > 1:
            a3 = param_calib[ind3] * DEG
        else:
            a3 = param_calib[0] * DEG
        mat3 = np.array([[np.cos(a3), -np.sin(a3), 0],
                        [np.sin(a3), np.cos(a3), 0],
                        [0, 0, 1]])

        mat3 = GT.matRot(AXIS3, a3/DEG)

    deltamat = np.dot(mat3, np.dot(mat2, mat1))
    newmatrix = np.dot(deltamat, initrot)

    # three last parameters are orientation angles in quaternion expression
    onlydetectorindices = arr_indexvaryingparameters[arr_indexvaryingparameters < 5]

    X, Y, theta, _ = xy_from_Quat(param_calib,
                                    DATA_Q,
                                    nspots,
                                    onlydetectorindices,
                                    allparameters,
                                    initrot=newmatrix,
                                    vecteurref=vecteurref,
                                    pureRotation=pureRotation,
                                    labXMAS=0,
                                    verbose=verbose,
                                    pixelsize=pixelsize,
                                    dim=dim,
                                    kf_direction=kf_direction)

    distanceterm = np.sqrt((X - pixX) ** 2 + (Y - pixY) ** 2)

    if (weights is not None):  # take into account the exp. spots intensity as weight in cost distance function
        allweights = np.sum(weights)
        distanceterm = distanceterm * weights / allweights
        # print "**mean weighted distanceterm   ",mean(distanceterm),"    ********"
        # print "**mean distanceterm   ",mean(distanceterm),"    ********"

    if allspots_info == 0:
        if verbose:
            # print "X",X
            # print "pixX",pixX
            # print "Y",Y
            # print "pixY",pixY
            # print "param_orient",param_calib
            # print "distanceterm",distanceterm
            # print "*****************mean distanceterm   ",mean(distanceterm),"    ********"
            # print "newmatrix", newmatrix
            return distanceterm, deltamat, newmatrix

        else:
            return distanceterm

    elif allspots_info == 1:
        Xtheo = X
        Ytheo = Y
        Xexp = pixX
        Yexp = pixY
        Xdev = Xtheo - Xexp
        Ydev = Ytheo - Yexp

        theta_theo = theta

        spotsData = [Xtheo, Ytheo, Xexp, Yexp, Xdev, Ydev, theta_theo]

        return distanceterm, deltamat, newmatrix, spotsData


def fit_on_demand_calibration(starting_param, miller, allparameters,
                                _error_function_on_demand_calibration,
                                arr_indexvaryingparameters,
                                nspots,
                                pixX,
                                pixY,
                                initrot=IDENTITYMATRIX,
                                vecteurref=IDENTITYMATRIX,
                                pureRotation=1,
                                verbose=0,
                                pixelsize=165.0 / 2048,
                                dim=(2048, 2048),
                                weights=None,
                                kf_direction="Z>0",
                                **kwd):
    """
    #All miller indices must be entered in miller,
    selection is done in xy_from_Quat with nspots (array of indices)
    """
    # parameters = ["distance (mm)",
    #             "Xcen (pixel)",
    #             "Ycen (pixel)",
    #             "Angle1 (deg)",
    #             "Angle2 (deg)",
    #             "theta1",
    #             "theta2",
    #             "theta3"]

    # parameters_being_fitted = [parameters[k] for k in arr_indexvaryingparameters]
    param_calib_0 = starting_param
    if verbose:
        # print(
        #     "\n\n***************************\nfirst error with initial values of:",
        #     parameters_being_fitted, " \n\n***************************\n")

        _error_function_on_demand_calibration(param_calib_0,
                                                miller,
                                                allparameters,
                                                arr_indexvaryingparameters,
                                                nspots,
                                                pixX,
                                                pixY,
                                                initrot=initrot,
                                                vecteurref=vecteurref,
                                                pureRotation=pureRotation,
                                                verbose=1,
                                                pixelsize=pixelsize,
                                                dim=dim,
                                                weights=weights,
                                                kf_direction=kf_direction)

        # print("\n\n***************************\nFitting parameters:  ", parameters_being_fitted,
        #     "\n\n***************************\n")
        # # NEEDS AT LEAST 5 spots (len of nspots)
        # print("With initial values", param_calib_0)

    # setting  keywords of _error_function_on_demand_calibration during the fitting because leastsq handle only *args but not **kwds
    _error_function_on_demand_calibration.__defaults__ = (initrot,
                                                        vecteurref,
                                                        pureRotation,
                                                        0,
                                                        pixelsize,
                                                        dim,
                                                        weights,
                                                        0,
                                                        kf_direction)

    # For transmission geometry , changing gam scale is useful
    # x_scale = [1,1,1,1,.1,1,1,1]  1 except for xgam .1
    xscale = np.ones(len(arr_indexvaryingparameters))
    try:
        posgam = arr_indexvaryingparameters.tolist().index(4)
        xscale[posgam] = .1
    except ValueError:
        pass
    #------------------------
    calib_sol2 = least_squares(_error_function_on_demand_calibration,
                                param_calib_0,
                                args=(miller, allparameters, arr_indexvaryingparameters, nspots, pixX, pixY),
                              tr_solver = 'exact',
                              x_scale=xscale, max_nfev=None)

    # print("\nLEAST_SQUARES")
    # #print("calib_sol2", calib_sol2['x'])
    # print(calib_sol2['x'])
    # print('mean residues', np.mean(calib_sol2['fun']))
    
    return calib_sol2['x']

    # LEASTSQUARE
    calib_sol = leastsq(_error_function_on_demand_calibration,
                            param_calib_0,
                            args=(miller, allparameters, arr_indexvaryingparameters, nspots, pixX, pixY),
                            maxfev=5000,
                            **kwd)  # args=(rre,ertetr,) last , is important!

    if calib_sol[-1] in (1, 2, 3, 4, 5):
        if verbose:
            # print("\n\n **************  End of Fitting  -  Final errors  ****************** \n\n")
            _error_function_on_demand_calibration(calib_sol[0],
                                                    miller,
                                                    allparameters,
                                                    arr_indexvaryingparameters,
                                                    nspots,
                                                    pixX,
                                                    pixY,
                                                    initrot=initrot,
                                                    pureRotation=pureRotation,
                                                    verbose=verbose,
                                                    pixelsize=pixelsize,
                                                    dim=dim,
                                                    weights=weights,
                                                    kf_direction=kf_direction)
        return calib_sol[0]  # 5 detector parameters + deltaangles
    else:
        return None


def error_function_on_demand_strain(param_strain,
                                        DATA_Q,
                                        allparameters,
                                        arr_indexvaryingparameters,
                                        nspots,
                                        pixX,
                                        pixY,
                                        initrot=IDENTITYMATRIX,
                                        Bmat=IDENTITYMATRIX,
                                        pureRotation=0,
                                        verbose=0,
                                        pixelsize=165.0 / 2048.,
                                        dim=(2048, 2048),
                                        weights=None,
                                        kf_direction="Z>0"):
    """
    #All miller indices must be entered in DATA_Q, selection is done in xy_from_Quat with nspots (array of indices)
    # allparameters must contain 5 detector calibration parameters + 5 parameters of strain + 3 angles of elementary rotation
    # param_strain must contain values of one or many parameters of allparameters
    #
    #   strain = param_strain[:5]
    #   deltaangles = param_strain[5:8]
    #   arr_indexvaryingparameters = array of position of parameters whose values are in param_strain
    #    e.g.: arr_indexvaryingparameters = array([5,6,7,8,9]) for only fit strain without orientation refinement
    #    e.g.: arr_indexvaryingparameters = array([5,6,7,8,9, 10,11,12]) for strain AND orientation refinement
    #   in this function calibration is not refined (but values are needed!), arr_indexvaryingparameters must only contain index >= 5
    Bmat=  B0 matrix

    """

    #print('param_strain in error_function_on_demand_strain', param_strain)

    mat1, mat2, mat3 = IDENTITYMATRIX, IDENTITYMATRIX, IDENTITYMATRIX

    # arr_indexvaryingparameters =  [5,6,7,8,9,10,11,12]  first 5 params for strain and 3 last for rotation
    index_of_rot_in_arr_indexvaryingparameters = [10, 11, 12]

    if index_of_rot_in_arr_indexvaryingparameters[0] in arr_indexvaryingparameters:
        ind1 = np.where(
            arr_indexvaryingparameters == index_of_rot_in_arr_indexvaryingparameters[0]
        )[0][0]
        if len(arr_indexvaryingparameters) > 1:
            a1 = param_strain[ind1] * DEG
        else:
            a1 = param_strain[0] * DEG
        # print "a1 (rad)= ",a1
        mat1 = np.array([[np.cos(a1), 0, np.sin(a1)], [0, 1, 0], [-np.sin(a1), 0, np.cos(a1)]])

    if index_of_rot_in_arr_indexvaryingparameters[1] in arr_indexvaryingparameters:
        ind2 = np.where(arr_indexvaryingparameters == index_of_rot_in_arr_indexvaryingparameters[1])[0][0]
        if len(arr_indexvaryingparameters) > 1:
            a2 = param_strain[ind2] * DEG
        else:
            a2 = param_strain[0] * DEG
        # print "a2 (rad)= ",a2
        mat2 = np.array([[1, 0, 0], [0, np.cos(a2), np.sin(a2)], [0, np.sin(-a2), np.cos(a2)]])

    if index_of_rot_in_arr_indexvaryingparameters[2] in arr_indexvaryingparameters:
        ind3 = np.where(
            arr_indexvaryingparameters == index_of_rot_in_arr_indexvaryingparameters[2])[0][0]
        if len(arr_indexvaryingparameters) > 1:
            a3 = param_strain[ind3] * DEG
        else:
            a3 = param_strain[0] * DEG
        mat3 = np.array([[np.cos(a3), -np.sin(a3), 0],
                            [np.sin(a3), np.cos(a3), 0],
                            [0, 0, 1]])

    deltamat = np.dot(mat3, np.dot(mat2, mat1))

    # building B mat
    varyingstrain = np.array([[1.0, param_strain[2], param_strain[3]],
                                [0, param_strain[0], param_strain[4]],
                                [0, 0, param_strain[1]]])

    newmatrix = np.dot(np.dot(deltamat, initrot), varyingstrain)

    # # three last parameters are orientation angles in quaternion expression and are here not used
    # varying_parameter_value = array(allparameters[:5])
    # arr_indexvaryingparameters =  arr_indexvaryingparameters [arr_indexvaryingparameters < 5]

    # varying_parameter_value: array of value that will be taken into account
    # xy_from_Quat  only uses 5 detector calibration parameter
    # fitting_param: index of position of varying parameters in allparameters array
    # allparameters: array of 8 elements: 5 first of calibration parameters and 3 of angles defining quaternion

    patchallparam = allparameters.tolist()

    #    5  detector parameters  +   3  angles   +  5 strain components
    ally = np.array(patchallparam[:5] + [0, 0, 0] + patchallparam[5:])

    if 2 in arr_indexvaryingparameters:
        ally[2]=param_strain[-1]

    # because elem 5 to 7 are used in quaternion calculation
    # TODO : correct also strain calib in the same manner
    X, Y, _, _ = xy_from_Quat(allparameters[:5],
                                DATA_Q,
                                nspots,
                                np.arange(5),
                                ally,
                                initrot=newmatrix,
                                vecteurref=Bmat,
                                pureRotation=0,
                                labXMAS=0,
                                verbose=0,
                                pixelsize=pixelsize,
                                dim=dim,
                                kf_direction=kf_direction)

    distanceterm = np.sqrt((X - pixX) ** 2 + (Y - pixY) ** 2)

    if weights is not None:
        allweights = np.sum(weights)
        distanceterm = distanceterm * weights / allweights

    if verbose:
        # if weights is not None:
        #     print("***********mean weighted pixel deviation   ", np.mean(distanceterm), "    ********")
        # else:
        #     print("***********mean pixel deviation   ", np.mean(distanceterm), "    ********")
        #        print "newmatrix", newmatrix
        return distanceterm, deltamat, newmatrix

    else:
        return distanceterm


def error_function_strain_with_two_orientations(param_strain, DATA_Q, allparameters,
                                                    arr_indexvaryingparameters, nspots, pixX, pixY,
                                                initrot=IDENTITYMATRIX,
                                                Bmat=IDENTITYMATRIX,
                                                pureRotation=0,
                                                verbose=0,
                                                pixelsize=165.0 / 2048,
                                                dim=(2048, 2048),
                                                weights=None):
    """
    #All miller indices must be entered in DATA_Q, selection is done in xy_from_Quat with nspots (array of indices)
    # allparameters must contain 5 detector calibration parameters + 5 parameters of strain + 3 angles of elementary rotation
    # param_strain must contain values of one or many parameters of allparameters
    #
    #   strain = param_strain[:5]
    #   deltaangles = param_strain[5:8]
    #   arr_indexvaryingparameters = array of position of parameters whose values are in param_strain
    #    e.g.: arr_indexvaryingparameters = array([5,6,7,8,9]) for only fit strain without orientation refinement
    #    e.g.: arr_indexvaryingparameters = array([5,6,7,8,9, 10,11,12, 13,14,15]) for strain AND orientation refinement
    #   in this function calibration is not refined (but values are needed!), arr_indexvaryingparameters must only contain index >= 5

    TODO: not implemented for transmission geometry (kf_direction='X>0') and backreflection ('X<0')

    .. warning::
        not completed  !
    """

    mat1, mat2, mat3 = IDENTITYMATRIX, IDENTITYMATRIX, IDENTITYMATRIX

    # arr_indexvaryingparameters =  [5,6,7,8,9,10,11,12]  first 5 params for strain and 6 last for misorientation of two grains
    index_of_rot_in_arr_indexvaryingparameters_1 = [10, 11, 12]
    index_of_rot_in_arr_indexvaryingparameters_2 = [13, 14, 15]

    if index_of_rot_in_arr_indexvaryingparameters_1[0] in arr_indexvaryingparameters:
        ind1 = np.where(
            arr_indexvaryingparameters == index_of_rot_in_arr_indexvaryingparameters_1[0])[0][0]
        if len(arr_indexvaryingparameters) > 1:
            a1 = param_strain[ind1] * DEG
        else:
            a1 = param_strain[0] * DEG
        # print "a1 (rad)= ",a1
        mat1 = np.array([[np.cos(a1), 0, np.sin(a1)],
                        [0, 1, 0],
                        [-np.sin(a1), 0, np.cos(a1)]])

    if index_of_rot_in_arr_indexvaryingparameters_1[1] in arr_indexvaryingparameters:
        ind2 = np.where(
            arr_indexvaryingparameters == index_of_rot_in_arr_indexvaryingparameters_1[1])[0][0]
        if len(arr_indexvaryingparameters) > 1:
            a2 = param_strain[ind2] * DEG
        else:
            a2 = param_strain[0] * DEG
        # print "a2 (rad)= ",a2
        mat2 = np.array([[1, 0, 0],
                        [0, np.cos(a2), np.sin(a2)],
                        [0, np.sin(-a2), np.cos(a2)]])

    if index_of_rot_in_arr_indexvaryingparameters_1[2] in arr_indexvaryingparameters:
        ind3 = np.where(
            arr_indexvaryingparameters == index_of_rot_in_arr_indexvaryingparameters_1[2])[0][0]
        if len(arr_indexvaryingparameters) > 1:
            a3 = param_strain[ind3] * DEG
        else:
            a3 = param_strain[0] * DEG
        mat3 = np.array([[np.cos(a3), -np.sin(a3), 0],
                            [np.sin(a3), np.cos(a3), 0],
                            [0, 0, 1]])

    deltamat_1 = np.dot(mat3, np.dot(mat2, mat1))

    if index_of_rot_in_arr_indexvaryingparameters_2[0] in arr_indexvaryingparameters:
        ind1 = np.where(
            arr_indexvaryingparameters
            == index_of_rot_in_arr_indexvaryingparameters_2[0]
        )[0][0]
        if len(arr_indexvaryingparameters) > 1:
            a1 = param_strain[ind1] * DEG
        else:
            a1 = param_strain[0] * DEG
        # print "a1 (rad)= ",a1
        mat1 = np.array([[np.cos(a1), 0, np.sin(a1)],
                        [0, 1, 0],
                        [-np.sin(a1), 0, np.cos(a1)]])

    if index_of_rot_in_arr_indexvaryingparameters_2[1] in arr_indexvaryingparameters:
        ind2 = np.where(
            arr_indexvaryingparameters
            == index_of_rot_in_arr_indexvaryingparameters_2[1])[0][0]
        if len(arr_indexvaryingparameters) > 1:
            a2 = param_strain[ind2] * DEG
        else:
            a2 = param_strain[0] * DEG
        # print "a2 (rad)= ",a2
        mat2 = np.array([[1, 0, 0],
                            [0, np.cos(a2), np.sin(a2)],
                            [0, np.sin(-a2), np.cos(a2)]])

    if index_of_rot_in_arr_indexvaryingparameters_2[2] in arr_indexvaryingparameters:
        ind3 = np.where(
            arr_indexvaryingparameters
            == index_of_rot_in_arr_indexvaryingparameters_2[2])[0][0]
        if len(arr_indexvaryingparameters) > 1:
            a3 = param_strain[ind3] * DEG
        else:
            a3 = param_strain[0] * DEG
        mat3 = np.array([[np.cos(a3), -np.sin(a3), 0], [np.sin(a3), np.cos(a3), 0], [0, 0, 1]])

    deltamat_2 = np.dot(mat3, np.dot(mat2, mat1))

    # building B mat
    varyingstrain = np.array(
        [[1.0, param_strain[2], param_strain[3]],
            [0, param_strain[0], param_strain[4]],
            [0, 0, param_strain[1]]])

    newmatrix_1 = np.dot(np.dot(deltamat_1, initrot), varyingstrain)

    newmatrix_2 = np.dot(np.dot(deltamat_2, initrot), varyingstrain)

    # # three last parameters are orientation angles in quaternion expression and are here not used
    # varying_parameter_value = array(allparameters[:5])
    # arr_indexvaryingparameters =  arr_indexvaryingparameters [arr_indexvaryingparameters < 5]

    # varying_parameter_value: array of value that will be taken into account
    # xy_from_Quat  only uses 5 detector calibration parameter
    # fitting_param: index of position of varying parameters in allparameters array
    # allparameters: array of 8 elements: 5 first of calibration parameters and 3 of angles defining quaternion

    patchallparam = allparameters.tolist()

    #                5 det parameters    +  3 small rotations     +   5 strain parameters
    ally_1 = np.array(patchallparam[:5] + [0, 0, 0] + patchallparam[5:])
    # because elem 5 to 7 are used in quaternion calculation
    # TODO : correct also strain calib in the same manner
    X1, Y1, _, _ = xy_from_Quat(allparameters[:5],
                                        DATA_Q,
                                        nspots,
                                        np.arange(5),
                                        ally_1,
                                        initrot=newmatrix_1,
                                        vecteurref=Bmat,
                                        pureRotation=0,
                                        labXMAS=0,
                                        verbose=0,
                                        pixelsize=pixelsize,
                                        dim=dim)

    # distanceterm1 = np.sqrt((X1 - pixX) ** 2 + (Y1 - pixY) ** 2)

    #                5 det parameters    +  3 small rotations     +   5 strain parameters
    ally_2 = np.array(patchallparam[:5] + [0, 0, 0] + patchallparam[5:])
    # because elem 5 to 7 are used in quaternion calculation
    # TODO : correct also strain calib in the same manner
    X2, Y2, _, _ = xy_from_Quat(allparameters[:5],
                                    DATA_Q,
                                    nspots,
                                    np.arange(5),
                                    ally_2,
                                    initrot=newmatrix_2,
                                    vecteurref=Bmat,
                                    pureRotation=0,
                                    labXMAS=0,
                                    verbose=0,
                                    pixelsize=pixelsize,
                                    dim=dim)

    distanceterm2 = np.sqrt((X2 - pixX) ** 2 + (Y2 - pixY) ** 2)

    if weights is not None:
        allweights = np.sum(weights)
        distanceterm = distanceterm2 * weights / allweights
        # print "**mean weighted distanceterm   ",mean(distanceterm),"    ********"
    # print "**mean distanceterm   ",mean(distanceterm),"    ********"

    if verbose:
        # if weights is not None:
        #     print("***********mean weighted pixel deviation   ", np.mean(distanceterm), "    ********")
        # else:
        #     print("***********mean pixel deviation   ", np.mean(distanceterm), "    ********")
        return distanceterm2, (deltamat_1, deltamat_2), (newmatrix_1, newmatrix_2)

    else:
        return distanceterm


def fit_on_demand_strain(starting_param,
                                miller,
                                allparameters,
                                _error_function_on_demand_strain,
                                arr_indexvaryingparameters,
                                nspots,
                                pixX,
                                pixY,
                                initrot=IDENTITYMATRIX,
                                Bmat=IDENTITYMATRIX,
                                pureRotation=0,
                                verbose=0,
                                pixelsize=165.0 / 2048,
                                dim=(2048, 2048),
                                weights=None,
                                kf_direction="Z>0",
                                fitycen=False,
                                **kwd):
    """
    To use it:
    allparameters = 5calibdetectorparams + fivestrainparameter + 3deltaangles of orientations
    starting_param = [fivestrainparameter + 3deltaangles of orientations] = [1,1,0,0,0,0,0,0]  typically
    arr_indexvaryingparameters = range(5,13)
    """

    # All miller indices must be entered in miller, selection is done in xy_from_Quat with nspots (array of indices)
    # parameters = ["dd", "xcen", "ycen", "angle1", "angle2", "b/a", "c/a",
    #                     "a12", "a13", "a23", "theta1", "theta2", "theta3", ]

    # parameters_being_fitted = [parameters[k] for k in arr_indexvaryingparameters]

    param_strain_0 = starting_param

    # print('\n\nstarting_param',starting_param)
    if verbose:
        # print("\n\n***************************\nfirst error with initial values of:",
        #     parameters_being_fitted, " \n\n***************************\n")

        _error_function_on_demand_strain(param_strain_0,
                                        miller,
                                        allparameters,
                                        arr_indexvaryingparameters,
                                        nspots,
                                        pixX,
                                        pixY,
                                        initrot=initrot,
                                        Bmat=Bmat,
                                        pureRotation=pureRotation,
                                        verbose=0,
                                        pixelsize=pixelsize,
                                        dim=dim,
                                        weights=weights,
                                        kf_direction=kf_direction)

        # print("\n\n***************************\nFitting parameters:  ",
        #     parameters_being_fitted,
        #     "\n\n***************************\n")
        # # NEEDS AT LEAST 5 spots (len of nspots)
        # print("With initial values", param_strain_0)

    # setting  keywords of _error_function_on_demand_strain during the fitting because leastsq handle only *args but not **kwds
    _error_function_on_demand_strain.__defaults__ = (initrot,
                                                    Bmat,
                                                    pureRotation,
                                                    0,
                                                    pixelsize,
                                                    dim,
                                                    weights,
                                                    kf_direction)

    # LEASTSQUARE
    res = leastsq(_error_function_on_demand_strain,
                    param_strain_0,
                    args=(miller, allparameters, arr_indexvaryingparameters, nspots, pixX, pixY),
                    maxfev=5000,
                    full_output=1,
                    xtol=1.0e-11,
                    epsfcn=0.0,
                    **kwd)


    #---------------------  other least square  ------------------                
    # For ycen fitting together strain component, changing ycen scale is useful
    # x_scale = [1,1,1,1,.1,1,1,1]  1 except for xgam .1
    xscale = np.ones(len(arr_indexvaryingparameters))
    try:
        xscale[-1] = 100
    except ValueError:
        pass
    
    # if 0:
    #     #------------------------
    #     # from scipy.optimize import leastsq, least_squares

    #     calib_sol2 = least_squares(_error_function_on_demand_strain,
    #                                 param_strain_0,
    #                                 args=(miller, allparameters, arr_indexvaryingparameters, nspots, pixX, pixY),
    #                             tr_solver = 'exact',
    #                             x_scale=xscale, max_nfev=None)

    #     # print("\nLEAST_SQUARES")
    #     # #print("calib_sol2", calib_sol2['x'])
    #     # print(calib_sol2['x'])
    #     # print('mean residues', np.mean(calib_sol2['fun']))
        
    #     #return calib_sol2['x']
    #     #---------------------  other least square  ------------------                


    strain_sol = res[0]

    # print("code results", res[-1])
    # print("nb iterations", res[2]["nfev"])
    # print("mesg", res[-2])

    # if verbose:
    #     print("strain_sol", strain_sol)

    if res[-1] not in (1, 2, 3, 4, 5):
        return None
    else:
        if verbose:
            # print("\n\n **************  End of Fitting  -  Final errors  ****************** \n\n")
            _error_function_on_demand_strain(strain_sol,
                                            miller,
                                            allparameters,
                                            arr_indexvaryingparameters,
                                            nspots,
                                            pixX,
                                            pixY,
                                            initrot=initrot,
                                            Bmat=Bmat,
                                            pureRotation=pureRotation,
                                            verbose=verbose,
                                            pixelsize=pixelsize,
                                            dim=dim,
                                            weights=weights,
                                            kf_direction=kf_direction)
        return strain_sol


def plot_refinement_oneparameter(starting_param,
                                miller,
                                allparameters,
                                _error_function_on_demand_calibration,
                                arr_indexvaryingparameters,
                                nspots,
                                pixX,
                                pixY,
                                param_range,
                                initrot=IDENTITYMATRIX,
                                vecteurref=IDENTITYMATRIX,
                                pureRotation=1,
                                verbose=0,
                                pixelsize=165.0 / 2048,
                                dim=(2048, 2048),
                                weights=None,
                                kf_direction="Z>0",
                                **kwd):

    """
    All miller indices must be entered in miller,
    selection is done in xy_from_Quat with nspots (array of indices)
    """
    # parameters = ["distance (mm)", "Xcen (pixel)", "Ycen (pixel)",
    #                 "Angle1 (deg)", "Angle2 (deg)", "theta1", "theta2", "theta3"]

    # parameters_being_fitted = [parameters[k] for k in arr_indexvaryingparameters]
    param_calib_0 = starting_param

    mini, maxi, nbsteps = param_range

    # setting  keywords of _error_function_on_demand_calibration during the fitting because leastsq handle only *args but not **kwds
    _error_function_on_demand_calibration.__defaults__ = (initrot,
                                                        vecteurref,
                                                        pureRotation,
                                                        0,
                                                        pixelsize,
                                                        dim,
                                                        weights,
                                                        kf_direction)

    # designed for rotation angle
    res = []
    for angle in np.linspace(mini, maxi, nbsteps) + param_calib_0:
        residues = _error_function_on_demand_calibration(np.array([angle]),
                                                        miller,
                                                        allparameters,
                                                        arr_indexvaryingparameters,
                                                        nspots,
                                                        pixX,
                                                        pixY,
                                                        initrot=initrot,
                                                        vecteurref=vecteurref,
                                                        pureRotation=pureRotation,
                                                        verbose=0,
                                                        pixelsize=pixelsize,
                                                        weights=weights,
                                                        kf_direction=kf_direction)
        # print "mean(residues)",mean(residues)
        res.append([angle, np.mean(residues)])

    return res


def error_function_XCEN(param_calib,
                        DATA_Q,
                        allparameters,
                        nspots,
                        pixX,
                        pixY,
                        initrot=IDENTITYMATRIX,
                        pureRotation=1,
                        verbose=0,
                        pixelsize=165.0 / 2048):
    """
    seems to be useless ?
    """
    # All miller indices must be entered in DATA_Q, selection is done in xy_from_Quat with nspots (array of indices)
    # param_orient is three elements array representation of quaternion

    X, Y, _, R = xy_from_Quat(param_calib,
                                DATA_Q,
                                nspots,
                                np.arange(8)[1],
                                allparameters,
                                initrot=initrot,
                                pureRotation=pureRotation,
                                labXMAS=0,
                                verbose=verbose,
                                pixelsize=pixelsize)

    distanceterm = np.sqrt((X - pixX) ** 2 + (Y - pixY) ** 2)

    # print "**mean distanceterm   ",mean(distanceterm),"    ********"

    if verbose:
        # print("X", X)
        # print("pixX", pixX)
        # print("Y", Y)
        # print("pixY", pixY)
        # print("param_orient", param_calib)
        # print("distanceterm", distanceterm)
        # print("\n*****************\n\nmean distanceterm   ", np.mean(distanceterm), "    ********\n")
        return distanceterm, R
    else:
        return distanceterm


def fitXCEN(starting_param,
                    miller,
                    allparameters,
                    _error_function_XCEN,
                    nspots,
                    pixX,
                    pixY,
                    initrot=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.0]]),
                    pureRotation=1,
                    verbose=0,
                    pixelsize=165.0 / 2048,
                    **kwd):
    """
    #All miller indices must be entered in miller,
    selection is done in xy_from_Quat with nspots (array of indices)
    """
    param_calib_0 = starting_param
    if verbose:
        # print("\n\n***************************\nfirst error XCEN************************\n")

        _error_function_XCEN(param_calib_0,
                            miller,
                            allparameters,
                            nspots,
                            pixX,
                            pixY,
                            initrot=initrot,
                            pureRotation=pureRotation,
                            verbose=1,
                            pixelsize=pixelsize)

        # print("\n\n***************************\nFitting XCEN ...\n\n***************************\n")

        # print("Starting parameters", param_calib_0)

    # setting  keywords of _error_function_XCEN during the fitting because leastsq handle only *args but not **kwds
    _error_function_XCEN.__defaults__ = (initrot, pureRotation, 0, pixelsize)

    calib_sol = leastsq(_error_function_XCEN,
                        param_calib_0,
                        args=(miller, allparameters, nspots, pixX, pixY),
                        **kwd)  # args=(rre,ertetr,) last , is important!

    # print("calib_sol", calib_sol)
    if calib_sol[-1] in (1, 2, 3, 4, 5):
        if verbose:
            # print("\n\n **************  End of Fitting  -  Final errors  ****************** \n\n")
            _error_function_XCEN(calib_sol[0],
                                    miller,
                                    allparameters,
                                    nspots,
                                    pixX,
                                    pixY,
                                    initrot=initrot,
                                    pureRotation=pureRotation,
                                    verbose=verbose,
                                    pixelsize=pixelsize)
        return calib_sol[0]  # 5 detector parameters
    else:
        return None


def fit_on_demand_strain_2grains(starting_param,
                                miller,
                                allparameters,
                                _error_function_on_demand_strain_2grains,
                                arr_indexvaryingparameters,
                                absolutespotsindices,
                                pixX,
                                pixY,
                                initrot=IDENTITYMATRIX,
                                B0matrix=IDENTITYMATRIX,
                                nb_grains=1,
                                pureRotation=0,
                                verbose=0,
                                pixelsize=165.0 / 2048,
                                dim=(2048, 2048),
                                weights=None,
                                kf_direction="Z>0",
                                **kwd):
    """
    Fit a model of two grains of the same material
    Initial orientation matrices are the same (only strain state differs)

    To use it:
    allparameters = 5calibdetectorparams + fivestrainparameters_g1 + 3deltaangles_g1 of orientations
                    + fivestrainparameters_g2 + 3deltaangles_g2 of orientations
    starting_param = [fivestrainparameter + 3deltaangles of orientations] = [1,1,0,0,0,0,0,0]+[1,1,0,0,0,0,0,0]  typically
    arr_indexvaryingparameters = range(5,21)

    B0matrix   : B0 matrix defining a*,b*,c* basis vectors (in columns) in initial orientation / LT frame
    """
    # All miller indices must be entered in miller
    # selection is done in xy_from_Quat with absolutespotsindices (array of indices)
    # parameterscalib = ["dd", "xcen", "ycen", "angle1", "angle2"]
    # strain_g1 = ["b/a", "c/a", "a12", "a13", "a23"]
    # rot_g1 = ["theta1", "theta2", "theta3"]
    # strain_g2 = ["b/a", "c/a", "a12", "a13", "a23"]
    # parameters = parameterscalib + strain_g1 + rot_g1 + strain_g2
    # parameters_being_fitted = [parameters[k] for k in arr_indexvaryingparameters]

    init_strain_values = starting_param
    if verbose:
        # print("\n\n***************************\nfirst error with initial values of:",
        #     parameters_being_fitted, " \n\n***************************\n")

        _error_function_on_demand_strain_2grains(init_strain_values,
                                                miller,
                                                allparameters,
                                                arr_indexvaryingparameters,
                                                absolutespotsindices,
                                                pixX,
                                                pixY,
                                                initrot=initrot,
                                                B0matrix=B0matrix,
                                                nb_grains=nb_grains,
                                                pureRotation=pureRotation,
                                                verbose=1,
                                                pixelsize=pixelsize,
                                                dim=dim,
                                                weights=weights,
                                                kf_direction=kf_direction)

        # print("\n\n***************************\nFitting parameters:  ",
        #     parameters_being_fitted, "\n\n***************************\n")
        # # NEEDS AT LEAST 5 spots (len of nspots)
        # print("With initial values", init_strain_values)

    # setting  keywords of _error_function_on_demand_strain during the fitting because leastsq handle only *args but not **kwds
    _error_function_on_demand_strain_2grains.__defaults__ = (initrot,
                                                            B0matrix,
                                                            nb_grains,
                                                            pureRotation,
                                                            0,
                                                            pixelsize,
                                                            dim,
                                                            weights,
                                                            kf_direction,
                                                            False)

    #     pixX = np.array(pixX, dtype=np.float64)
    #     pixY = np.array(pixY, dtype=np.float64)
    # LEASTSQUARE
    res = leastsq(error_function_on_demand_strain_2grains,
                init_strain_values,
                args=(
                    miller,
                    allparameters,
                    arr_indexvaryingparameters,
                    absolutespotsindices,
                    pixX,
                    pixY),  # args=(rre,ertetr,) last , is important!
                maxfev=5000,
                full_output=1,
                xtol=1.0e-11,
                epsfcn=0.0,
                **kwd)

    strain_sol = res[0]

    #     print "res", res
    #     print "code results", res[-1]
    # print("nb iterations", res[2]["nfev"])

    # if verbose:
    #     print("strain_sol", strain_sol)

    if res[-1] not in (1, 2, 3, 4, 5):
        return None
    else:
        if verbose:
            # print("\n\n **************  End of Fitting  -  Final errors  ****************** \n\n")
            _error_function_on_demand_strain_2grains(strain_sol,
                                                    miller,
                                                    allparameters,
                                                    arr_indexvaryingparameters,
                                                    absolutespotsindices,
                                                    pixX,
                                                    pixY,
                                                    initrot=initrot,
                                                    B0matrix=B0matrix,
                                                    nb_grains=nb_grains,
                                                    pureRotation=pureRotation,
                                                    verbose=verbose,
                                                    pixelsize=pixelsize,
                                                    dim=dim,
                                                    weights=weights,
                                                    kf_direction=kf_direction,
                                                    returnalldata=True)
        return strain_sol


def error_function_on_demand_strain_2grains(varying_parameters_values,
                                            DATA_Q,
                                            allparameters,
                                            arr_indexvaryingparameters,
                                            absolutespotsindices,
                                            pixX,
                                            pixY,
                                            initrot=IDENTITYMATRIX,
                                            B0matrix=IDENTITYMATRIX,
                                            nb_grains=1,
                                            pureRotation=0,
                                            verbose=0,
                                            pixelsize=165.0 / 2048,
                                            dim=(2048, 2048),
                                            weights=None,
                                            kf_direction="Z>0",
                                            returnalldata=False):
    """
    compute array of errors of weight*((Xtheo-pixX)**2+(Ytheo-pixY)**2) for each pears
    Xtheo, Ytheo derived from kf and q vector: q = UB Bmat B0 G* where G* =[h ,k, l] vector

    Bmat is the displacements matrix   strain = Bmat-Id

    #All miller indices must be entered in DATA_Q, selection is done in xy_from_Quat with absolutespotsindices (array of indices)
    # allparameters must contain 5 detector calibration parameters + 5 parameters_g1 of strain + 3 angles_g1 of elementary rotation
    #                             + 5 parameters_g2 of strain
    # varying_parameters_values must contain values of one or many parameters of allparameters
    #
    #   strain_g1 = varying_parameters_values[:5]
        strain_g2 = varying_parameters_values[8:13]
    #   deltaangles_g1 = varying_parameters_values[5:8]

    #   arr_indexvaryingparameters = array of position of parameters whose values are in varying_parameters_values
    #    e.g.: arr_indexvaryingparameters = array([5,6,7,8,9]) for only fit g1's strain without orientation refinement
    #    e.g.: arr_indexvaryingparameters = array([5,6,7,8,9, 10,11,12]) for g1's strain AND orientation refinement
    #   in this function calibration is not refined (but values are needed!), arr_indexvaryingparameters must only contain index >= 5

    DATA_Q   array of hkl vectors
    pixX     arrays of pixels exp. peaks X positions  [Xs g1,Xs g2]
    pixY     arrays of pixels exp. peaks Y positions [Ys g1,Ys g2]
    absolutespotsindices        [absolutespotsindices g1, absolutespotsindices g2]
    weights    None or [weights g1, weight g2]
    initrot   = guessed UB orientation matrix
    B0matrix    B0 matrix defining a*,b*,c* basis vectors (in columns) in initial orientation / LT frame


    TODO: ?? not implemented for transmission geometry (kf_direction='X>0') ? and backreflection ('X<0')
    """

    if isinstance(allparameters, np.ndarray):
        calibrationparameters = (allparameters.tolist())[:5]
    else:
        calibrationparameters = allparameters[:5]

    rotationselements_indices = [[10, 11, 12],[18, 19, 20]]  # with counting 5 calib parameters
    strainelements_indices = [[5, 6, 7, 8, 9], [13, 14, 15, 16, 17]]

    distances_vector_list = []
    all_deltamatrices = []
    all_newmatrices = []
    for grain_index in list(range(nb_grains)):
        mat1, mat2, mat3 = IDENTITYMATRIX, IDENTITYMATRIX, IDENTITYMATRIX

        # arr_indexvaryingparameters =  [5,6,7,8,9,10,11,12]  first 5 params for strain and 3 last fro roatation
        index_of_rot_in_arr_indexvaryingparameters = rotationselements_indices[grain_index]

        if index_of_rot_in_arr_indexvaryingparameters[0] in arr_indexvaryingparameters:
            ind1 = np.where(arr_indexvaryingparameters == index_of_rot_in_arr_indexvaryingparameters[0])[0][0]
            if len(arr_indexvaryingparameters) > 1:
                a1 = varying_parameters_values[ind1] * DEG
            else:
                a1 = varying_parameters_values[0] * DEG
            # print "a1 (rad)= ",a1
            mat1 = np.array(
                [[np.cos(a1), 0, np.sin(a1)], [0, 1, 0], [-np.sin(a1), 0, np.cos(a1)]])

        if index_of_rot_in_arr_indexvaryingparameters[1] in arr_indexvaryingparameters:
            ind2 = np.where(arr_indexvaryingparameters == index_of_rot_in_arr_indexvaryingparameters[1])[0][0]
            if len(arr_indexvaryingparameters) > 1:
                a2 = varying_parameters_values[ind2] * DEG
            else:
                a2 = varying_parameters_values[0] * DEG
            # print "a2 (rad)= ",a2
            mat2 = np.array(
                [[1, 0, 0], [0, np.cos(a2), np.sin(a2)], [0, np.sin(-a2), np.cos(a2)]])

        if index_of_rot_in_arr_indexvaryingparameters[2] in arr_indexvaryingparameters:
            ind3 = np.where(arr_indexvaryingparameters == index_of_rot_in_arr_indexvaryingparameters[2])[0][0]
            if len(arr_indexvaryingparameters) > 1:
                a3 = varying_parameters_values[ind3] * DEG
            else:
                a3 = varying_parameters_values[0] * DEG
            mat3 = np.array([[np.cos(a3), -np.sin(a3), 0], [np.sin(a3), np.cos(a3), 0], [0, 0, 1]])

        deltamat = np.dot(mat3, np.dot(mat2, mat1))

        all_deltamatrices.append(deltamat)

        # print("all_deltamatrices", all_deltamatrices)

        # building Bmat ------------(triangular up matrix)
        index_of_strain_in_arr_indexvaryingparameters = strainelements_indices[grain_index]

        # print("arr_indexvaryingparameters", arr_indexvaryingparameters)
        # print("varying_parameters_values", varying_parameters_values)

        # default parameters
        s_list = [1, 1, 0, 0, 0]
        for s_index in list(range(5)):
            if (
                index_of_strain_in_arr_indexvaryingparameters[s_index]
                in arr_indexvaryingparameters):
                ind1 = np.where(
                    arr_indexvaryingparameters
                    == index_of_strain_in_arr_indexvaryingparameters[s_index]
                )[0][0]
                if len(arr_indexvaryingparameters) > 1:
                    s_list[s_index] = varying_parameters_values[ind1]
                else:  # handling fit with single fitting parameter
                    s_list[s_index] = varying_parameters_values[0]

        s0, s1, s2, s3, s4 = s_list
        varyingstrain = np.array([[1.0, s2, s3], [0, s0, s4], [0, 0, s1]])

        newmatrix = np.dot(np.dot(deltamat, initrot), varyingstrain)
        all_newmatrices.append(newmatrix)

        #         print "varyingstrain", varyingstrain
        #         print 'all_newmatrices', all_newmatrices

        Xmodel, Ymodel, _, _ = calc_XY_pixelpositions(calibrationparameters,
                                                            DATA_Q,
                                                            absolutespotsindices[grain_index],
                                                            UBmatrix=newmatrix,
                                                            B0matrix=B0matrix,
                                                            pureRotation=0,
                                                            labXMAS=0,
                                                            verbose=0,
                                                            pixelsize=pixelsize,
                                                            dim=dim,
                                                            kf_direction=kf_direction)


        Xexp = pixX[grain_index]
        Yexp = pixY[grain_index]

        distanceterm = np.sqrt((Xmodel - Xexp) ** 2 + (Ymodel - Yexp) ** 2)

        if weights is not None:
            allweights = np.sum(weights[grain_index])
            distanceterm = distanceterm * weights[grain_index] / allweights

        # if verbose:
        #     print("**   grain %d   distance residues = " % grain_index,
        #         distanceterm, "    ********")
        #     print("**   grain %d   mean distance residue = " % grain_index,
        #         np.mean(distanceterm), "    ********")
        #             print "twthe, chi", twthe, chi
        distances_vector_list.append(distanceterm)

    #     print 'len(distances_vector_list)', len(distances_vector_list)

    if nb_grains == 2:
        alldistances_array = np.hstack((distances_vector_list[0], distances_vector_list[1]))
    if nb_grains == 1:
        alldistances_array = distances_vector_list[0]

    if verbose:
        pass
        # if weights is not None:
        #     print("***********mean weighted pixel deviation   ",
        #         np.mean(alldistances_array), "    ********")
        # else:
        #     print("***********mean pixel deviation   ",
        #         np.mean(alldistances_array), "    ********")
    #        print "newmatrix", newmatrix
    if returnalldata:
        # concatenated all pairs distances, all UB matrices, all UB.B0matrix matrices
        return alldistances_array, all_deltamatrices, all_newmatrices

    else:
        return alldistances_array


def error_function_general(varying_parameters_values_array,
                            varying_parameters_keys,
                            Miller_indices,
                            allparameters,
                            absolutespotsindices,
                            Xexp,
                            Yexp,
                            initrot=IDENTITYMATRIX,
                            B0matrix=IDENTITYMATRIX,
                            pureRotation=0,
                            verbose=0,
                            pixelsize=165.0 / 2048,
                            dim=(2048, 2048),
                            weights=None,
                            kf_direction="Z>0",
                            returnalldata=False):
    """
    q = T_LT  UzUyUz Ustart  T_c B0 G*

    Interface error function to return array of pair (exp. - model) distances
    Sum_i [weights_i((Xmodel_i-Xexp_i)**2+(Ymodel_i-Yexp_i)**2) ]

    Xmodel,Ymodel comes from G*=ha*+kb*+lc*

    q = T_LT  UzUyUz Ustart  T_c B0 G*

    B0   reference structure reciprocal space frame (a*,b*,c*) a* // ki  b* perp to a*  and perp to z (z belongs to the plane of ki and detector normal vector n)
            i.e.   columns of B0 are components of a*,b* and c*   expressed in x,y,z LT frame

    possible keys for parameters to be refined are:

    five detector frame calibration parameters:
    detectordistance,xcen,ycen,beta, gamma

    three misorientation angles with respect to LT orthonormal frame (x, y, z) matrices Ux, Uy,Uz:
    anglex,angley,anglez

    5 independent elements of a distortion operator

    -[[Tc00,Tc01,Tc02],[Tc10,Tc11,Tc12],[Tc20,Tc21,Tc22]]
    each column is the transformed reciprocal unit cell vector a*',b*' or c*' expressed in a*,b*,c* frame (reference reciprocal unit cell)

    Usually Tc11, Tc22, Tc01,Tc02,Tc12  with Tc00=1 and the all others = 0 (matrix triangular up)

    # TODO :- [[Td00,Td01,Td02],[Td10,Td11,Td12],[Td20,Td21,Td22]]
    #
    #each column is the transformed direct crystal unit cell vector a',b' or c' expressed in a,b,c frame (reference unit cell)

    -[[T00,T01,T02],[T10,T11,T12],[T20,T21,T22]]
    each column is the transformed LT frame vector x',y' or z' expressed in x,y,z frame

    -[[Ts00,Ts01,Ts02],[Ts10,Ts11,Ts12],[Ts20,Ts21,Ts22]]
    each column is the transformed sample frame vector xs',ys' or zs' expressed in xs,ys,zs frame
    """

    if isinstance(allparameters, np.ndarray):
        calibrationparameters = (allparameters.tolist())[:5]
    else:
        calibrationparameters = allparameters[:5]

    #     print 'allparameters',allparameters

    Uy, Ux, Uz = IDENTITYMATRIX, IDENTITYMATRIX, IDENTITYMATRIX
    Tc = np.array(allparameters[8:17]).reshape((3, 3))
    T = np.array(allparameters[17:26]).reshape((3, 3))
    Ts = np.array(allparameters[26:35]).reshape((3, 3))
    latticeparameters = np.array(allparameters[35:41])
    sourcedepth = allparameters[41]

    #     print "Tc before", Tc

    T_has_elements = False
    Ts_has_elements = False
    Tc_has_elements = False
    latticeparameters_has_elements = False

    nb_varying_parameters = len(varying_parameters_keys)

    for varying_parameter_index, parameter_name in enumerate(varying_parameters_keys):
        #         print "varying_parameter_index,parameter_name", varying_parameter_index, parameter_name

        if parameter_name in ("anglex", "angley", "anglez"):
            #             print "got angles!"
            if nb_varying_parameters > 1:
                anglevalue = (varying_parameters_values_array[varying_parameter_index] * DEG)
            else:
                anglevalue = varying_parameters_values_array[0] * DEG
            # print "anglevalue (rad)= ",anglevalue
            ca = np.cos(anglevalue)
            sa = np.sin(anglevalue)
            if parameter_name == "angley":
                Uy = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
            elif parameter_name == "anglex":
                Ux = np.array([[1.0, 0, 0], [0, ca, sa], [0, -sa, ca]])

            elif parameter_name == "anglez":
                Uz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1.0]])

        elif ((not T_has_elements) and (not Ts_has_elements) and parameter_name
            in ("Tc00", "Tc01", "Tc02", "Tc10", "Tc11", "Tc12", "Tc20", "Tc21", "Tc22")):
            #             print 'got Tc elements: ', parameter_name
            for i in list(range(3)):
                for j in list(range(3)):
                    if parameter_name == "Tc%d%d" % (i, j):
                        #                         print "got parameter_name", parameter_name
                        if nb_varying_parameters > 1:
                            Tc[i, j] = varying_parameters_values_array[varying_parameter_index]
                        else:
                            Tc[i, j] = varying_parameters_values_array[0]
                        Tc_has_elements = True

        elif (not Tc_has_elements and not Ts_has_elements and parameter_name
            in ("T00", "T01", "T02", "T10", "T11", "T12", "T20", "T21", "T22")):
            for i in list(range(3)):
                for j in list(range(3)):
                    if parameter_name is "T%d%d" % (i, j):
                        if nb_varying_parameters > 1:
                            T[i, j] = varying_parameters_values_array[varying_parameter_index]
                        else:
                            T[i, j] = varying_parameters_values_array[0]
                        T_has_elements = True

        elif (not Tc_has_elements and not T_has_elements and parameter_name
            in ("Ts00", "Ts01", "Ts02", "Ts10", "Ts11", "Ts12", "Ts20", "Ts21", "Ts22")):
            for i in list(range(3)):
                for j in list(range(3)):
                    if parameter_name is "Ts%d%d" % (i, j):
                        if nb_varying_parameters > 1:
                            Ts[i, j] = varying_parameters_values_array[varying_parameter_index]
                        else:
                            Ts[i, j] = varying_parameters_values_array[0]
                        Ts_has_elements = True

        elif parameter_name in ("a", "b", "c", "alpha", "beta", "gamma"):
            indparam = dict_lattice_parameters[parameter_name]

            #             if nb_varying_parameters > 1:
            #                 latticeparameters[indparam] = latticeparameters[0] * np.exp(varying_parameters_values_array[varying_parameter_index] / factorscale)
            #             else:
            #                 latticeparameters[indparam] = latticeparameters[0] * np.exp(varying_parameters_values_array[0] / factorscale)

            if nb_varying_parameters > 1:
                latticeparameters[indparam] = varying_parameters_values_array[varying_parameter_index]
            else:
                latticeparameters[indparam] = varying_parameters_values_array[0]
            latticeparameters_has_elements = True

        elif parameter_name in ("distance",):
            calibrationparameters[0] = varying_parameters_values_array[varying_parameter_index]
        elif parameter_name in ("xcen",):
            calibrationparameters[1] = varying_parameters_values_array[varying_parameter_index]
        elif parameter_name in ("ycen",):
            calibrationparameters[2] = varying_parameters_values_array[varying_parameter_index]
        elif parameter_name in ("beta",):
            calibrationparameters[3] = varying_parameters_values_array[varying_parameter_index]
        elif parameter_name in ("gamma",):
            calibrationparameters[4] = varying_parameters_values_array[varying_parameter_index]

        elif parameter_name in ("depth",):
            sourcedepth = varying_parameters_values_array[varying_parameter_index]

    Uxyz = np.dot(Uz, np.dot(Ux, Uy))

    # if verbose:
    #     print("Uxyz", Uxyz)
    #     print("varying_parameters_keys", varying_parameters_keys)
    #     print("varying_parameters_values_array", varying_parameters_values_array)

    #     print("Tc_has_elements", Tc_has_elements)
    #     print("T_has_elements", T_has_elements)
    #     print("Ts_has_elements", Ts_has_elements)
    #     print("latticeparameters_has_elements", latticeparameters_has_elements)

    #     print "Tc after", Tc
    #     print "T", T
    #     print 'Ts', Ts

    # DictLT.RotY40 such as   X=DictLT.RotY40 Xsample  (xs,ys,zs =columns expressed in x,y,z frame)
    # transform in sample frame   Ts
    # same transform in x,y,z LT frame T
    # Ts = DictLT.RotY40-1 T DictLT.RotY40
    # T = DictLT.RotY40 Ts DictLT.RotY40-1

    newmatrix = np.dot(Uxyz, initrot)

    if Tc_has_elements:
        newmatrix = np.dot(newmatrix, Tc)
    elif T_has_elements:
        newmatrix = np.dot(T, newmatrix)
    elif Ts_has_elements:
        T = np.dot(np.dot(DictLT.RotY40, Ts), DictLT.RotYm40)
        newmatrix = np.dot(T, newmatrix)
    elif latticeparameters_has_elements:
        B0matrix = CP.calc_B_RR(latticeparameters, directspace=1, setvolume=False)
    # if verbose:
    #     print("newmatrix", newmatrix)
    #     print("B0matrix", B0matrix)

    Xmodel, Ymodel, _, _ = calc_XY_pixelpositions(calibrationparameters,
                                                        Miller_indices,
                                                        absolutespotsindices,
                                                        UBmatrix=newmatrix,
                                                        B0matrix=B0matrix,
                                                        offset=sourcedepth,
                                                        pureRotation=0,
                                                        labXMAS=0,
                                                        verbose=0,
                                                        pixelsize=pixelsize,
                                                        dim=dim,
                                                        kf_direction=kf_direction)


    distanceterm = np.sqrt((Xmodel - Xexp) ** 2 + (Ymodel - Yexp) ** 2)

    if weights is not None:
        allweights = np.sum(weights)
        distanceterm = distanceterm * weights / allweights

    # if verbose:
    #     #         print "**      distance residues = " , distanceterm, "    ********"
    #     print("**    mean distance residue = ", np.mean(distanceterm), "    ********")
    #             print "twthe, chi", twthe, chi

    alldistances_array = distanceterm
    if verbose:
        # print "varying_parameters_values in error_function_on_demand_strain",varying_parameters_values
        # print "arr_indexvaryingparameters",arr_indexvaryingparameters
        # print "Xmodel",Xmodel
        # print "pixX",pixX
        # print "Ymodel",Ymodel
        # print "pixY",pixY
        # print "newmatrix",newmatrix
        # print "B0matrix",B0matrix
        # print "deltamat",deltamat
        # print "initrot",initrot
        # print "param_orient",param_calib
        # print "distanceterm",distanceterm
        pass
        # if weights is not None:
        #     print("***********mean weighted pixel deviation   ",
        #         np.mean(alldistances_array), "    ********")
        # else:
        #     print("***********mean pixel deviation   ", np.mean(alldistances_array), "    ********")
    #        print "newmatrix", newmatrix
    if returnalldata:
        # concatenated all pairs distances, all UB matrices, all UB.B0matrix matrices
        return alldistances_array, Uxyz, newmatrix, Tc, T, Ts

    else:
        return alldistances_array


def fit_function_general(varying_parameters_values_array,
                                varying_parameters_keys,
                                Miller_indices,
                                allparameters,
                                absolutespotsindices,
                                Xexp,
                                Yexp,
                                UBmatrix_start=IDENTITYMATRIX,
                                B0matrix=IDENTITYMATRIX,
                                nb_grains=1,
                                pureRotation=0,
                                verbose=0,
                                pixelsize=165.0 / 2048,
                                dim=(2048, 2048),
                                weights=None,
                                kf_direction="Z>0",
                                **kwd):
    """
    
    """

    if verbose:
        # print("\n\n******************\nfirst error with initial values of:",
        #     varying_parameters_keys, " \n\n***************************\n")

        error_function_general(varying_parameters_values_array,
                                varying_parameters_keys,
                                Miller_indices,
                                allparameters,
                                absolutespotsindices,
                                Xexp,
                                Yexp,
                                initrot=UBmatrix_start,
                                B0matrix=B0matrix,
                                pureRotation=pureRotation,
                                verbose=1,
                                pixelsize=pixelsize,
                                dim=dim,
                                weights=weights,
                                kf_direction=kf_direction)
        # print("\n\n********************\nFitting parameters:  ",
        #     varying_parameters_keys, "\n\n***************************\n")
        # print("With initial values", varying_parameters_values_array)

    # setting  keywords of _error_function_on_demand_strain during the fitting because leastsq handle only *args but not **kwds
    error_function_general.__defaults__ = (UBmatrix_start,
                                            B0matrix,
                                            pureRotation,
                                            0,
                                            pixelsize,
                                            dim,
                                            weights,
                                            kf_direction,
                                            False)

    #     pixX = np.array(pixX, dtype=np.float64)
    #     pixY = np.array(pixY, dtype=np.float64)
    # LEASTSQUARE
    res = leastsq(error_function_general,
                    varying_parameters_values_array,
                    args=(
                        varying_parameters_keys,
                        Miller_indices,
                        allparameters,
                        absolutespotsindices,
                        Xexp,
                        Yexp,
                    ),  # args=(rre,ertetr,) last , is important!
                    maxfev=5000,
                    full_output=1,
                    xtol=1.0e-11,
                    epsfcn=0.0,
                    **kwd)

    refined_values = res[0]

    #     print "res fit in fit function general", res
    # print("code results", res[-1])
    # print("nb iterations", res[2]["nfev"])
    # print("refined_values", refined_values)

    if res[-1] not in (1, 2, 3, 4, 5):
        return None
    else:
        if verbose:
            # print("\n\n **************  End of Fitting  -  Final errors (general fit function) ****************** \n\n"
            # )
            alldata = error_function_general(refined_values,
                                                varying_parameters_keys,
                                                Miller_indices,
                                                allparameters,
                                                absolutespotsindices,
                                                Xexp,
                                                Yexp,
                                                initrot=UBmatrix_start,
                                                B0matrix=B0matrix,
                                                pureRotation=pureRotation,
                                                verbose=1,
                                                pixelsize=pixelsize,
                                                dim=dim,
                                                weights=weights,
                                                kf_direction=kf_direction,
                                                returnalldata=True)

            # alldistances_array, Uxyz, newmatrix, Tc, T, Ts
            alldistances_array, Uxyz, refinedUB, refinedTc, refinedT, refinedTs = alldata

            # for k, param_key in enumerate(varying_parameters_keys):
            #     print("%s  : start %.4f   --->   refined %.4f"
            #         % (param_key, varying_parameters_values_array[k], refined_values[k]))
            # print("results:\n q= refinedT UBstart refinedTc B0 G*\nq = refinedUB B0 G*")
            # print("refined UBmatrix", refinedUB)
            # print("Uxyz", Uxyz)
            # print("refinedTc, refinedT, refinedTs", refinedTc, refinedT, refinedTs)
            # print("final mean pixel residues : %f with %d spots"
            #     % (np.mean(alldistances_array), len(absolutespotsindices)))

        return refined_values


dict_lattice_parameters = {"a": 0, "b": 1, "c": 2, "alpha": 3, "beta": 4, "gamma": 5}


def fit_function_latticeparameters(varying_parameters_values_array,
                                    varying_parameters_keys,
                                    Miller_indices,
                                    allparameters,
                                    absolutespotsindices,
                                    Xexp,
                                    Yexp,
                                    UBmatrix_start=IDENTITYMATRIX,
                                    nb_grains=1,
                                    pureRotation=0,
                                    verbose=0,
                                    pixelsize=165.0 / 2048,
                                    dim=(2048, 2048),
                                    weights=None,
                                    kf_direction="Z>0",
                                    additional_expression="none",
                                    **kwd):
    """
    fit direct (real) unit cell lattice parameters  (in refinedB0)
    and orientation

    q =   refinedUzUyUz Ustart   refinedB0 G*

    with error function to return array of pair (exp. - model) distances
    Sum_i [weights_i((Xmodel_i-Xexp_i)**2+(Ymodel_i-Yexp_i)**2) ]

    Xmodel,Ymodel comes from G*=ha*+kb*+lc*


    """
    if verbose:
        # print("\n\n******************\nfirst error with initial values of:",
        #     varying_parameters_keys, " \n\n***************************\n",)

        error_function_latticeparameters(varying_parameters_values_array,
                                        varying_parameters_keys,
                                        Miller_indices,
                                        allparameters,
                                        absolutespotsindices,
                                        Xexp,
                                        Yexp,
                                        initrot=UBmatrix_start,
                                        pureRotation=pureRotation,
                                        verbose=1,
                                        pixelsize=pixelsize,
                                        dim=dim,
                                        weights=weights,
                                        kf_direction=kf_direction,
                                        additional_expression=additional_expression)

        # print("\n\n********************\nFitting parameters:  ",
        #     varying_parameters_keys, "\n\n***************************\n")
        # print("With initial values", varying_parameters_values_array)

    #     print '*************** UBmatrix_start before fit************'
    #     print UBmatrix_start
    #     print '*******************************************'

    # setting  keywords of _error_function_on_demand_strain during the fitting because leastsq handle only *args but not **kwds
    error_function_latticeparameters.__defaults__ = (UBmatrix_start,
                                                    pureRotation,
                                                    0,
                                                    pixelsize,
                                                    dim,
                                                    weights,
                                                    kf_direction,
                                                    False,
                                                    additional_expression)

    #     pixX = np.array(pixX, dtype=np.float64)
    #     pixY = np.array(pixY, dtype=np.float64)
    # LEASTSQUARE
    res = leastsq(error_function_latticeparameters,
                        varying_parameters_values_array,
                        args=(
                            varying_parameters_keys,
                            Miller_indices,
                            allparameters,
                            absolutespotsindices,
                            Xexp,
                            Yexp,
                        ),  # args=(rre,ertetr,) last , is important!
                        maxfev=5000,
                        full_output=1,
                        xtol=1.0e-11,
                        epsfcn=0.0,
                        **kwd)

    refined_values = res[0]

    #     print "res fit in fit function general", res
    # print("code results", res[-1])
    # print("nb iterations", res[2]["nfev"])
    # print("refined_values", refined_values)

    if res[-1] not in (1, 2, 3, 4, 5):
        return None
    else:
        if 1:
            # print(
            #     "\n\n **************  End of Fitting  -  Final errors (general fit function) ****************** \n\n"
            # )
            alldata = error_function_latticeparameters(refined_values,
                                                    varying_parameters_keys,
                                                    Miller_indices,
                                                    allparameters,
                                                    absolutespotsindices,
                                                    Xexp,
                                                    Yexp,
                                                    initrot=UBmatrix_start,
                                                    pureRotation=pureRotation,
                                                    verbose=1,
                                                    pixelsize=pixelsize,
                                                    dim=dim,
                                                    weights=weights,
                                                    kf_direction=kf_direction,
                                                    returnalldata=True,
                                                    additional_expression=additional_expression)

            # alldistances_array, Uxyz, newmatrix, Tc, T, Ts
            alldistances_array, Uxyz, refinedUB, refinedB0matrix, refinedLatticeparameters = (
                alldata)

            # print("\n--------------------\nresults:\n------------------")
            # for k, param_key in enumerate(varying_parameters_keys):
            #     print("%s  : start %f   --->   refined %f"
            #         % (param_key, varying_parameters_values_array[k], refined_values[k]))
            # print("q= refinedT UBstart refinedTc B0 G*\nq = refinedUB B0 G*")
            # print("refined UBmatrix", refinedUB.tolist())
            # print("Uxyz", Uxyz.tolist())
            # print("refinedB0matrix", refinedB0matrix.tolist())
            # print("refinedLatticeparameters", refinedLatticeparameters)
            # print("final mean pixel residues : %f with %d spots"
            #     % (np.mean(alldistances_array), len(absolutespotsindices)))

        return refined_values


def error_function_latticeparameters(varying_parameters_values_array,
                                        varying_parameters_keys,
                                        Miller_indices,
                                        allparameters,
                                        absolutespotsindices,
                                        Xexp,
                                        Yexp,
                                        initrot=IDENTITYMATRIX,
                                        pureRotation=0,
                                        verbose=0,
                                        pixelsize=165.0 / 2048,
                                        dim=(2048, 2048),
                                        weights=None,
                                        kf_direction="Z>0",
                                        returnalldata=False,
                                        additional_expression="none"):
    """
    q =   UzUyUz Ustart   B0 G*

    Interface error function to return array of pair (exp. - model) distances
    Sum_i [weights_i((Xmodel_i-Xexp_i)**2+(Ymodel_i-Yexp_i)**2) ]

    Xmodel,Ymodel comes from G*=ha*+kb*+lc*

    q =   refinedUzUyUz Ustart   refinedB0 G*

    B0   reference structure reciprocal space frame (a*,b*,c*) a* // ki  b* perp to a*  and perp to z (z belongs to the plane of ki and detector normal vector n)
            i.e.   columns of B0 are components of a*,b* and c*   expressed in x,y,z LT frame

    refinedB0 is obtained by refining the 5 /6  lattice parameters

    possible keys for parameters to be refined are:

    five detector frame calibration parameters:
    det_distance,det_xcen,det_ycen,det_beta, det_gamma

    three misorientation angles with respect to LT orthonormal frame (x, y, z) matrices Ux, Uy,Uz:
    anglex,angley,anglez

    5 lattice parameters among 6 (a,b,c,alpha, beta,gamma)

    """
    # reading default parameters
    # CCD plane calibration parameters
    if isinstance(allparameters, np.ndarray):
        calibrationparameters = (allparameters.tolist())[:5]
    else:
        calibrationparameters = allparameters[:5]

    # allparameters[5:8]  = 0,0,0
    Uy, Ux, Uz = IDENTITYMATRIX, IDENTITYMATRIX, IDENTITYMATRIX

    latticeparameters = np.array(allparameters[8:14])

    nb_varying_parameters = len(varying_parameters_keys)

    #     factorscale = 1.

    for varying_parameter_index, parameter_name in enumerate(varying_parameters_keys):
        #         print "varying_parameter_index,parameter_name", varying_parameter_index, parameter_name

        if parameter_name in ("anglex", "angley", "anglez"):
            #             print "got angles!"
            if nb_varying_parameters > 1:
                anglevalue = varying_parameters_values_array[varying_parameter_index] * DEG
            else:
                anglevalue = varying_parameters_values_array[0] * DEG
            # print "anglevalue (rad)= ",anglevalue
            ca = np.cos(anglevalue)
            sa = np.sin(anglevalue)
            if parameter_name == "angley":
                Uy = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
            elif parameter_name == "anglex":
                Ux = np.array([[1.0, 0, 0], [0, ca, sa], [0, -sa, ca]])

            elif parameter_name == "anglez":
                Uz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1.0]])

        elif parameter_name in ("alpha", "beta", "gamma"):
            #             print 'got Tc elements: ', parameter_name
            indparam = dict_lattice_parameters[parameter_name]
            #             if nb_varying_parameters > 1:
            #                 latticeparameters[indparam] = latticeparameters[3] * np.exp(varying_parameters_values_array[varying_parameter_index] / factorscale)
            #             else:
            #                 latticeparameters[indparam] = latticeparameters[3] * np.exp(varying_parameters_values_array[0] / factorscale)

            if nb_varying_parameters > 1:
                latticeparameters[indparam] = varying_parameters_values_array[varying_parameter_index]
            else:
                latticeparameters[indparam] = varying_parameters_values_array[0]

        elif parameter_name in ("a", "b", "c"):
            #             print 'got Tc elements: ', parameter_name
            indparam = dict_lattice_parameters[parameter_name]

            #             if nb_varying_parameters > 1:
            #                 latticeparameters[indparam] = latticeparameters[0] * np.exp(varying_parameters_values_array[varying_parameter_index] / factorscale)
            #             else:
            #                 latticeparameters[indparam] = latticeparameters[0] * np.exp(varying_parameters_values_array[0] / factorscale)

            if nb_varying_parameters > 1:
                latticeparameters[indparam] = varying_parameters_values_array[varying_parameter_index]
            else:
                latticeparameters[indparam] = varying_parameters_values_array[0]

    Uxyz = np.dot(Uz, np.dot(Ux, Uy))
    
    if additional_expression == "a==b":
        indparam = dict_lattice_parameters["b"]
        indparam1 = dict_lattice_parameters["a"]
        latticeparameters[indparam] = latticeparameters[indparam1]
        
    newB0matrix = CP.calc_B_RR(latticeparameters, directspace=1, setvolume=False)

    # if verbose:
    #     print("\n-------\nvarying_parameters_keys", varying_parameters_keys)
    #     print("varying_parameters_values_array", varying_parameters_values_array)
    #     print("Uxyz", Uxyz)
    #     print("latticeparameters", latticeparameters)
    #     print("newB0matrix", newB0matrix)

    # DictLT.RotY40 such as   X=DictLT.RotY40 Xsample  (xs,ys,zs =columns expressed in x,y,z frame)
    # transform in sample frame   Ts
    # same transform in x,y,z LT frame T
    # Ts = DictLT.RotY40-1 T DictLT.RotY40
    # T = DictLT.RotY40 Ts DictLT.RotY40-1

    newmatrix = np.dot(Uxyz, initrot)

    # if 0:  # verbose:
    #     print("initrot", initrot)
    #     print("newmatrix", newmatrix)

    Xmodel, Ymodel, _, _ = calc_XY_pixelpositions(calibrationparameters,
                                                    Miller_indices,
                                                    absolutespotsindices,
                                                    UBmatrix=newmatrix,
                                                    B0matrix=newB0matrix,
                                                    pureRotation=0,
                                                    labXMAS=0,
                                                    verbose=0,
                                                    pixelsize=pixelsize,
                                                    dim=dim,
                                                    kf_direction=kf_direction)
 

    if 0:  # verbose:
        print("Xmodel, Ymodel", Xmodel, Ymodel)
    if 0:  # verbose:
        print("Xexp, Yexp", Xexp, Yexp)

    distanceterm = np.sqrt((Xmodel - Xexp) ** 2 + (Ymodel - Yexp) ** 2)

    if weights is not None:
        allweights = np.sum(weights)
        distanceterm = distanceterm * weights / allweights

    # if verbose:
    #     #         print "**      distance residues = " , distanceterm, "    ********"
    #     print("**    mean distance residue = ", np.mean(distanceterm), "    ********")
    #             print "twthe, chi", twthe, chi

    alldistances_array = distanceterm
    if verbose:
        # print "varying_parameters_values in error_function_on_demand_strain",varying_parameters_values
        # print "arr_indexvaryingparameters",arr_indexvaryingparameters
        # print "Xmodel",Xmodel
        # print "pixX",pixX
        # print "Ymodel",Ymodel
        # print "pixY",pixY
        # print "newmatrix",newmatrix
        # print "newB0matrix",newB0matrix
        # print "deltamat",deltamat
        # print "initrot",initrot
        # print "param_orient",param_calib
        # print "distanceterm",distanceterm
        pass
        # if weights is not None:
        #     print("***********mean weighted pixel deviation   ",
        #         np.mean(alldistances_array), "    ********")
        # else:
        #     print(
        #         "***********mean pixel deviation   ", np.mean(alldistances_array),
        #         "    ********")
    #        print "newmatrix", newmatrix
    if returnalldata:
        # concatenated all pairs distances, all UB matrices, all UB.newB0matrix matrices
        return alldistances_array, Uxyz, newmatrix, newB0matrix, latticeparameters

    else:
        return alldistances_array


def error_function_strain(varying_parameters_values_array,
                        varying_parameters_keys,
                        Miller_indices,
                        allparameters,
                        absolutespotsindices,
                        Xexp,
                        Yexp,
                        initrot=IDENTITYMATRIX,
                        B0matrix=IDENTITYMATRIX,
                        pureRotation=0,
                        verbose=0,
                        pixelsize=165.0 / 2048,
                        dim=(2048, 2048),
                        weights=None,
                        kf_direction="Z>0",
                        returnalldata=False):
    """
    q =   refinedStrain refinedUzUyUz Ustart   B0 G*

    Interface error function to return array of pair (exp. - model) distances
    Sum_i [weights_i((Xmodel_i-Xexp_i)**2+(Ymodel_i-Yexp_i)**2) ]

    Xmodel,Ymodel comes from G*=ha*+kb*+lc*


    B0   reference structure reciprocal space frame (a*,b*,c*) a* // ki  b* perp to a*  and perp to z (z belongs to the plane of ki and detector normal vector n)
            i.e.   columns of B0 are components of a*,b* and c*   expressed in x,y,z LT frame

    Strain of reciprocal vectors   : 6 compenents of triangular up matrix ( T00  T01 T02)
                                                                          ( 0    T11 T12)
                                                                         ( 0    0   T22)
                one must be set (usually T00 = 1)

    Algebra:
    X=PX'        e'1    e'2    e'3
                  |      |      |
                  v      v      v
            e1 (  .      .      .  )
        P=  e2 (  .      .      .  )
            e3 (  .      .      .  )

    If A  transform expressed in (e1,e2,e3) basis
    and A' same transform but expressed in (e'1,e'2,e'3) basis
    then  A'=P-1 A P

    X_LT=P X_sample
    P=(cos40, 0 -sin40)
      (0      1    0  )
      (sin40  0  cos40)

    Strain_sample=P-1 Strain_LT P
    Strain_LT    = P  Strain_Sample P-1

    """
    # reading default parameters
    # CCD plane calibration parameters
    if isinstance(allparameters, np.ndarray):
        calibrationparameters = (allparameters.tolist())[:5]
    else:
        calibrationparameters = allparameters[:5]
    #     print 'calibrationparameters', calibrationparameters

    # allparameters[5:8]  = 0,0,0
    Uy, Ux, Uz = IDENTITYMATRIX, IDENTITYMATRIX, IDENTITYMATRIX

    straincomponents = np.array(allparameters[8:14])

    Ts = np.array([straincomponents[:3],
            [0.0, straincomponents[3], straincomponents[4]],
            [0, 0, straincomponents[5]]])

    #     print 'Ts before', Ts

    nb_varying_parameters = len(varying_parameters_keys)

    for varying_parameter_index, parameter_name in enumerate(varying_parameters_keys):
        #         print "varying_parameter_index,parameter_name", varying_parameter_index, parameter_name

        if parameter_name in ("anglex", "angley", "anglez"):
            #             print "got angles!"
            if nb_varying_parameters > 1:
                anglevalue = varying_parameters_values_array[varying_parameter_index] * DEG
            else:
                anglevalue = varying_parameters_values_array[0] * DEG
            # print "anglevalue (rad)= ",anglevalue
            ca = np.cos(anglevalue)
            sa = np.sin(anglevalue)
            if parameter_name == "angley":
                Uy = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
            elif parameter_name == "anglex":
                Ux = np.array([[1.0, 0, 0], [0, ca, sa], [0, -sa, ca]])
            elif parameter_name == "anglez":
                Uz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1.0]])
        elif parameter_name in ("Ts00", "Ts01", "Ts02", "Ts11", "Ts12", "Ts22"):
            #             print 'got Ts elements: ', parameter_name
            for i in list(range(3)):
                for j in list(range(3)):
                    if parameter_name == "Ts%d%d" % (i, j):
                        #                         print "got parameter_name", parameter_name
                        if nb_varying_parameters > 1:
                            Ts[i, j] = varying_parameters_values_array[varying_parameter_index]
                        else:
                            Ts[i, j] = varying_parameters_values_array[0]

    #     print 'Ts after', Ts

    Uxyz = np.dot(Uz, np.dot(Ux, Uy))

    newmatrix = np.dot(Uxyz, initrot)

    #     print 'Uxyz', Uxyz
    #     print 'newmatrix', newmatrix

    # DictLT.RotY40 such as   X=DictLT.RotY40 Xsample  (xs,ys,zs =columns expressed in x,y,z frame)
    # transform in sample frame   Ts
    # same transform in x,y,z LT frame T
    # Ts = DictLT.RotY40-1 T DictLT.RotY40
    # T = DictLT.RotY40 Ts DictLT.RotY40-1

    T = np.dot(np.dot(DictLT.RotY40, Ts), DictLT.RotYm40)
    #     T = np.dot(np.dot(DictLT.RotYm40, Ts), DictLT.RotY40)

    #     print 'T', T

    newmatrix = np.dot(T, newmatrix)

    if 0:  # verbose:
        print("initrot", initrot)
        print("newmatrix", newmatrix)
        print("Miller_indices", Miller_indices)
        print("absolutespotsindices", absolutespotsindices)

    Xmodel, Ymodel, _, _ = calc_XY_pixelpositions(calibrationparameters,
                                                        Miller_indices,
                                                        absolutespotsindices,
                                                        UBmatrix=newmatrix,
                                                        B0matrix=B0matrix,
                                                        pureRotation=0,
                                                        labXMAS=0,
                                                        verbose=0,
                                                        pixelsize=pixelsize,
                                                        dim=dim,
                                                        kf_direction=kf_direction)


    distanceterm = np.sqrt((Xmodel - Xexp) ** 2 + (Ymodel - Yexp) ** 2)

    if weights not in (None, False, "None", "False", 0, "0"):
        allweights = np.sum(weights)
        distanceterm = distanceterm * weights / allweights

    # if verbose:
    #     #         print "**      distance residues = " , distanceterm, "    ********"
    #     print("**    mean distance residue = ", np.mean(distanceterm), "    ********")
    #             print "twthe, chi", twthe, chi

    alldistances_array = distanceterm
    if verbose:
        # print "varying_parameters_values in error_function_on_demand_strain",varying_parameters_values
        # print "arr_indexvaryingparameters",arr_indexvaryingparameters
        # print "Xmodel",Xmodel
        # print "pixX",pixX
        # print "Ymodel",Ymodel
        # print "pixY",pixY
        # print "newmatrix",newmatrix
        # print "newB0matrix",newB0matrix
        # print "deltamat",deltamat
        # print "initrot",initrot
        # print "param_orient",param_calib
        # print "distanceterm",distanceterm
        pass
        # if weights is not None:
        #     print("***********mean weighted pixel deviation   ",
        #         np.mean(alldistances_array), "    ********")
        # else:
        #     print("***********mean pixel deviation   ",
        #         np.mean(alldistances_array), "    ********")
    #        print "newmatrix", newmatrix
    if returnalldata:
        # concatenated all pairs distances, all UB matrices, all UB.newB0matrix matrices
        return alldistances_array, Uxyz, newmatrix, Ts, T

    else:
        return alldistances_array


def fit_function_strain(varying_parameters_values_array,
                    varying_parameters_keys,
                    Miller_indices,
                    allparameters,
                    absolutespotsindices,
                    Xexp,
                    Yexp,
                    UBmatrix_start=IDENTITYMATRIX,
                    B0matrix=IDENTITYMATRIX,
                    nb_grains=1,
                    pureRotation=0,
                    verbose=0,
                    pixelsize=165.0 / 2048,
                    dim=(2048, 2048),
                    weights=None,
                    kf_direction="Z>0",
                    **kwd):
    """
    fit strain components in sample frame
    and orientation

    q =   refinedT refinedUzUyUz Ustart   refinedB0 G*

    with error function to return array of pair (exp. - model) distances
    Sum_i [weights_i((Xmodel_i-Xexp_i)**2+(Ymodel_i-Yexp_i)**2) ]

    Xmodel,Ymodel comes from G*=ha*+kb*+lc*

    where T comes from Ts
    """
    if verbose:
        # print("\n\n******************\nfirst error with initial values of:",
        #     varying_parameters_keys, " \n\n***************************\n")

        error_function_strain(varying_parameters_values_array,
                            varying_parameters_keys,
                            Miller_indices,
                            allparameters,
                            absolutespotsindices,
                            Xexp,
                            Yexp,
                            initrot=UBmatrix_start,
                            B0matrix=B0matrix,
                            pureRotation=pureRotation,
                            verbose=1,
                            pixelsize=pixelsize,
                            dim=dim,
                            weights=weights,
                            kf_direction=kf_direction)

        # print("\n\n********************\nFitting parameters:  ",
        #     varying_parameters_keys, "\n\n***************************\n")
        # print("With initial values", varying_parameters_values_array)

    #     print '*************** UBmatrix_start before fit************'
    #     print UBmatrix_start
    #     print '*******************************************'

    # setting  keywords of _error_function_on_demand_strain during the fitting because leastsq handle only *args but not **kwds
    error_function_strain.__defaults__ = (UBmatrix_start,
                                        B0matrix,
                                        pureRotation,
                                        0,
                                        pixelsize,
                                        dim,
                                        weights,
                                        kf_direction,
                                        False)

    #     pixX = np.array(pixX, dtype=np.float64)
    #     pixY = np.array(pixY, dtype=np.float64)
    # LEASTSQUARE
    res = leastsq(error_function_strain,
                varying_parameters_values_array,
                args=(
                    varying_parameters_keys,
                    Miller_indices,
                    allparameters,
                    absolutespotsindices,
                    Xexp,
                    Yexp,
                ),  # args=(rre,ertetr,) last , is important!
                maxfev=5000,
                full_output=1,
                xtol=1.0e-11,
                epsfcn=0.0,
                **kwd)

    refined_values = res[0]

    #     print "res fit in fit function general", res
    # print("code results", res[-1])
    # print("mesg", res[-2])
    # print("nb iterations", res[2]["nfev"])
    # print("refined_values", refined_values)

    if res[-1] not in (1, 2, 3, 4, 5):
        return None
    else:
        if 1:
            # print("\n\n **************  End of Fitting  -  Final errors (general fit function) ****************** \n\n")
            alldata = error_function_strain(refined_values,
                                            varying_parameters_keys,
                                            Miller_indices,
                                            allparameters,
                                            absolutespotsindices,
                                            Xexp,
                                            Yexp,
                                            initrot=UBmatrix_start,
                                            B0matrix=B0matrix,
                                            pureRotation=pureRotation,
                                            verbose=0,
                                            pixelsize=pixelsize,
                                            dim=dim,
                                            weights=weights,
                                            kf_direction=kf_direction,
                                            returnalldata=True)

            # alldistances_array, Uxyz, newmatrix, Ts, T
            alldistances_array, Uxyz, newmatrix, refinedTs, refinedT = alldata

            # print("\n--------------------\nresults:\n------------------")
            # for k, param_key in enumerate(varying_parameters_keys):
            #     print("%s  : start %f   --->   refined %f"
            #         % (param_key, varying_parameters_values_array[k], refined_values[k]))
            # print("q= refinedT UBstart B0 G*\nq = refinedUB B0 G*")
            # print("refined UBmatrix", newmatrix.tolist())
            # print("Uxyz", Uxyz.tolist())
            # print("refinedT", refinedT.tolist())
            # print("refinedTs", refinedTs.tolist())
            # print("refined_values", refined_values)
            # print("final mean pixel residues : %f with %d spots"
            #     % (np.mean(alldistances_array), len(absolutespotsindices)))

        return refined_values


def error_strain_from_elongation(varying_parameters_values_array,
                                varying_parameters_keys,
                                Miller_indices,
                                allparameters,
                                absolutespotsindices,
                                Xexp,
                                Yexp,
                                initrot=IDENTITYMATRIX,
                                B0matrix=IDENTITYMATRIX,
                                pureRotation=0,
                                verbose=0,
                                pixelsize=165.0 / 2048,
                                dim=(2048, 2048),
                                weights=None,
                                kf_direction="Z>0",
                                returnalldata=False):
    """
    calculate array of the sum of 3 distances from aligned points composing one single Laue spot

    Each elongated spot is composed by 3 points: P1 Pc P2 (Pc at the center et P1, P2 at the ends)

    error = sum (P1-P1exp)**2 +  (P2-P2exp)**2 +(Pc-Pcexp)**2

    But since P1exp end could be wrongly assign to simulated P2 end

    error = sum (P1-P1exp)**2 +  (P1-P2exp)**2 -P1P2exp**2 +
                  (P2-P2exp)**2 + (P2-P1exp)**2 -P1P2exp**2
                  +(Pc-Pcexp)**2

    strain axis in sample frame:
    axis_angle_1, axis_angle_2,minstrainamplitude,zerostrain,maxstrainamplitude
    example: minstrainamplitude=0.98, maxstrainamplitude=1.05, zerostrain=1

    u= (cos angle1, sin angle 1 cos angle 2, sin angle1 sin angle 2)

    X1Model, Y1Model, XcModel,YcModel
    tensile_along_u(v, tensile, u='zsample')


    q =   refinedStrain refinedUzUyUz Ustart   B0 G*

    Xmodel,Ymodel comes from G*=ha*+kb*+lc*


    B0   reference structure reciprocal space frame (a*,b*,c*) a* // ki  b* perp to a*  and perp to z (z belongs to the plane of ki and detector normal vector n)
            i.e.   columns of B0 are components of a*,b* and c*   expressed in x,y,z LT frame

    Strain   : 6 compenents of triangular up matrix ( T00  T01 T02)
                                                   ( 0    T11 T12)
                                                   ( 0    0   T22)
                one must be set (usually T00 = 1)

    Algebra:
    X=PX'        e'1    e'2    e'3
                  |      |      |
                  v      v      v
            e1 (  .      .      .  )
        P=  e2 (  .      .      .  )
            e3 (  .      .      .  )

    If A  transform expressed in (e1,e2,e3) basis
    and A' same transform but expressed in (e'1,e'2,e'3) basis
    then  A'=P-1 A P

    X_LT=P X_sample
    P=(cos40, 0 -sin40)
      (0      1    0  )
      (sin40  0  cos40)

    Strain_sample=P-1 Strain_LT P
    Strain_LT    = P  Strain_Sample P-1

    """
    # reading default parameters
    # CCD plane calibration parameters
    if isinstance(allparameters, np.ndarray):
        calibrationparameters = (allparameters.tolist())[:5]
    else:
        calibrationparameters = allparameters[:5]
    #     print 'calibrationparameters', calibrationparameters

    # allparameters[5:8]  = 0,0,0
    Uy, Ux, Uz = IDENTITYMATRIX, IDENTITYMATRIX, IDENTITYMATRIX

    straincomponents = np.array(allparameters[8:14])

    Ts = np.array([straincomponents[:3],
            [0.0, straincomponents[3], straincomponents[4]],
            [0, 0, straincomponents[5]]])

    #     print 'Ts before', Ts

    nb_varying_parameters = len(varying_parameters_keys)

    for varying_parameter_index, parameter_name in enumerate(varying_parameters_keys):
        #         print "varying_parameter_index,parameter_name", varying_parameter_index, parameter_name

        if parameter_name in ("anglex", "angley", "anglez"):
            #             print "got angles!"
            if nb_varying_parameters > 1:
                anglevalue = varying_parameters_values_array[varying_parameter_index] * DEG
            else:
                anglevalue = varying_parameters_values_array[0] * DEG
            # print "anglevalue (rad)= ",anglevalue
            ca = np.cos(anglevalue)
            sa = np.sin(anglevalue)
            if parameter_name == "angley":
                Uy = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
            elif parameter_name == "anglex":
                Ux = np.array([[1.0, 0, 0], [0, ca, sa], [0, -sa, ca]])
            elif parameter_name == "anglez":
                Uz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1.0]])
        elif parameter_name in ("Ts00", "Ts01", "Ts02", "Ts11", "Ts12", "Ts22"):
            #             print 'got Ts elements: ', parameter_name
            for i in list(range(3)):
                for j in list(range(3)):
                    if parameter_name == "Ts%d%d" % (i, j):
                        #                         print "got parameter_name", parameter_name
                        if nb_varying_parameters > 1:
                            Ts[i, j] = varying_parameters_values_array[varying_parameter_index]
                        else:
                            Ts[i, j] = varying_parameters_values_array[0]

    #     print 'Ts after', Ts

    Uxyz = np.dot(Uz, np.dot(Ux, Uy))

    newmatrix = np.dot(Uxyz, initrot)

    #     print 'Uxyz', Uxyz
    #     print 'newmatrix', newmatrix

    # DictLT.RotY40 such as   X=DictLT.RotY40 Xsample  (xs,ys,zs =columns expressed in x,y,z frame)
    # transform in sample frame   Ts
    # same transform in x,y,z LT frame T
    # Ts = DictLT.RotY40-1 T DictLT.RotY40
    # T = DictLT.RotY40 Ts DictLT.RotY40-1

    T = np.dot(np.dot(DictLT.RotY40, Ts), DictLT.RotYm40)

    #     print 'T', T

    newmatrix = np.dot(T, newmatrix)

    if 0:  # verbose:
        print("initrot", initrot)
        print("newmatrix", newmatrix)
        print("Miller_indices", Miller_indices)
        print("absolutespotsindices", absolutespotsindices)

    Xmodel, Ymodel, _, _ = calc_XY_pixelpositions(calibrationparameters,
                                                Miller_indices,
                                                absolutespotsindices,
                                                UBmatrix=newmatrix,
                                                B0matrix=B0matrix,
                                                pureRotation=0,
                                                labXMAS=0,
                                                verbose=0,
                                                pixelsize=pixelsize,
                                                dim=dim,
                                                kf_direction=kf_direction)


    distanceterm = np.sqrt((Xmodel - Xexp) ** 2 + (Ymodel - Yexp) ** 2)

    if weights is not None:
        allweights = np.sum(weights)
        distanceterm = distanceterm * weights / allweights

    # if verbose:
    #     #         print "**      distance residues = " , distanceterm, "    ********"
    #     print("**    mean distance residue = ", np.mean(distanceterm), "    ********")
    #             print "twthe, chi", twthe, chi

    alldistances_array = distanceterm
    if verbose:
        # print "varying_parameters_values in error_function_on_demand_strain",varying_parameters_values
        # print "arr_indexvaryingparameters",arr_indexvaryingparameters
        # print "Xmodel",Xmodel
        # print "pixX",pixX
        # print "Ymodel",Ymodel
        # print "pixY",pixY
        # print "newmatrix",newmatrix
        # print "newB0matrix",newB0matrix
        # print "deltamat",deltamat
        # print "initrot",initrot
        # print "param_orient",param_calib
        # print "distanceterm",distanceterm
        pass
        # if weights is not None:
        #     print("***********mean weighted pixel deviation   ",
        #         np.mean(alldistances_array), "    ********")
        # else:
        #     print("***********mean pixel deviation   ",
        #         np.mean(alldistances_array), "    ********")
    #        print "newmatrix", newmatrix
    if returnalldata:
        # concatenated all pairs distances, all UB matrices, all UB.newB0matrix matrices
        return alldistances_array, Uxyz, newmatrix, Ts, T

    else:
        return alldistances_array


# --- -----   TESTS & DEMOS  ----------------------


def test_generalfitfunction():
    # Ge example unstrained
    pixX = np.array([1027.1099965580365, 1379.1700028337193, 1288.1100055910788, 926.219994375393, 595.4599989710869, 1183.2699986884652, 1672.670001029018, 1497.400007802548, 780.2700069727559, 819.9099991880139, 873.5600007021501, 1579.39000403102, 1216.4900044928474, 1481.199997684615, 399.87000836895436, 548.2499911593322, 1352.760007116035, 702.5200057620646, 383.7700117705855, 707.2000052800154, 1140.9300043834062, 1730.3299981313016, 289.68999155533413, 1274.8600008806216, 1063.2499947675371, 1660.8600022917144, 1426.670005812432])
    pixY = np.array([1293.2799953573963, 1553.5800003037994, 1460.1599988550274, 872.0599978043742, 876.4400033114814, 598.9200007214372, 1258.6199918206175, 1224.7000037967478, 1242.530005349013, 552.8399954684833, 706.9700021553684, 754.63000554209, 1042.2800069222762, 364.8400055136739, 1297.1899933698528, 1260.320007366279, 568.0299942819768, 949.8800073732916, 754.580011319991, 261.1099917270594, 748.3999917806088, 1063.319998717625, 945.9700059216573, 306.9500110237749, 497.7900029269757, 706.310001700921, 858.780004244009])
    miller_indices = np.array([[3.0, 3.0, 3.0], [2.0, 4.0, 2.0], [3.0, 5.0, 3.0], [5.0, 3.0, 3.0], [6.0, 2.0, 4.0], [6.0, 4.0, 2.0], [3.0, 5.0, 1.0], [4.0, 6.0, 2.0], [5.0, 3.0, 5.0], [7.0, 3.0, 3.0], [4.0, 2.0, 2.0], [5.0, 5.0, 1.0], [5.0, 5.0, 3.0], [7.0, 5.0, 1.0], [5.0, 1.0, 5.0], [3.0, 1.0, 3.0], [8.0, 6.0, 2.0], [7.0, 3.0, 5.0], [5.0, 1.0, 3.0], [9.0, 3.0, 3.0], [7.0, 5.0, 3.0], [5.0, 7.0, 1.0], [7.0, 1.0, 5.0], [5.0, 3.0, 1.0], [9.0, 5.0, 3.0], [7.0, 7.0, 1.0], [3.0, 3.0, 1.0]])
    starting_orientmatrix = np.array([[-0.9727538909589738, -0.21247913537718385, 0.09274958034159074],
            [0.22567394392094073, -0.7761682018781203, 0.5887564805829774],
            [-0.053107604650232926, 0.593645098498364, 0.8029726516869564]])
    #         B0matrix = np.array([[0.17675651789659746, -2.8424615990749217e-17, -2.8424615990749217e-17],
    #                            [0.0, 0.17675651789659746, -1.0823215193524997e-17],
    #                            [0.0, 0.0, 0.17675651789659746]])
    pixelsize = 0.08057
    calibparameters = [69.196, 1050.78, 1116.22, 0.152, -0.251]

    absolutespotsindices = np.arange(len(pixY))
    #
    varying_parameters_keys = ["anglex", "angley", "anglez", "a", "b", "alpha", "beta", "gamma", "depth"]
    varying_parameters_values_array = [0.0, -0, 0.0, 5.678, 5.59, 89.999, 90, 90.0001, 0.02]

    #     varying_parameters_keys = ['distance','xcen','ycen','beta','gamma',
    #                                'anglex', 'angley', 'anglez',
    #                                     'a', 'b', 'alpha', 'beta', 'gamma']
    #     varying_parameters_values_array = [68.5, 1049,1116,0,0,
    #                                        0., -0, 0.,
    #                                        5.678, 5.59, 89.999, 90, 90.0001]

    #     varying_parameters_keys = ['distance','xcen','ycen',
    #                                'anglex', 'angley', 'anglez',
    #                                     'a', 'b', 'alpha', 'beta', 'gamma']
    #     varying_parameters_values_array = [68.9, 1050,1116,
    #                                        0., -0, 0.,
    #                                        5.678, 5.59, 89.999, 90, 90.0001]

    #     varying_parameters_keys = ['distance','ycen',
    #                                'anglex', 'angley', 'anglez',
    #                                     'a', 'b', 'alpha', 'beta', 'gamma']
    #     varying_parameters_values_array = [68.9,1116,
    #                                        0., -0, 0.,
    #                                        5.675, 5.65, 89.999, 90, 90.0001]

    latticeparameters = DictLT.dict_Materials["Ge"][1]
    B0 = CP.calc_B_RR(latticeparameters)

    transformparameters = [0, 0, 0,  # 3 misorientation / initial UB matrix
                            1.0, 0, 0, 0, 1.0, 0, 0, -0.0, 1,  # Tc
                            1, 0, 0, 0, 1, 0, 0, 0, 1,  # T
                            1, 0, 0, 0, 1, 0, 0, 0, 1, ]  # Ts
    sourcedepth = [0]
    allparameters = (calibparameters + transformparameters + latticeparameters + sourcedepth)

    pureUmatrix, residualdistortion = GT.UBdecomposition_RRPP(starting_orientmatrix)

    # print("len(allparameters)", len(allparameters))
    # print("starting_orientmatrix", starting_orientmatrix)
    # print("pureUmatrix", pureUmatrix)

    refined_values = fit_function_general(varying_parameters_values_array,
                                        varying_parameters_keys,
                                        miller_indices,
                                        allparameters,
                                        absolutespotsindices,
                                        pixX,
                                        pixY,
                                        UBmatrix_start=pureUmatrix,
                                        B0matrix=B0,
                                        nb_grains=1,
                                        pureRotation=0,
                                        verbose=0,
                                        pixelsize=pixelsize,
                                        dim=(2048, 2048),
                                        weights=None,
                                        kf_direction="Z>0")

    dictRes = {}
    # print("\n****** Refined Values *********\n")
    for paramname, val in zip(varying_parameters_keys, refined_values):
        dictRes[paramname] = val
    #     print("%s     =>  %.6f" % (paramname, val))
    # print("\n*******************************\n")

    return dictRes
