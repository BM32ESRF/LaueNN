# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 09:56:13 2020

@author: Ravi

Strain functions of Laue
"""
import numpy as np
# import os
from lauetoolsnn.utils_lauenn import resource_path

print("LaueNN base path" , resource_path(''))

# import lauetoolsnn.lauetools.dict_LaueTools as dictLT
import lauetoolsnn.lauetools.generaltools as GT
import lauetoolsnn.lauetools.CrystalParameters as CP
import lauetoolsnn.lauetools.lauecore as LT
# import lauetoolsnn.lauetools.LaueGeometry as Lgeo
# import lauetoolsnn.lauetools.readmccd as RMCCD
# import lauetoolsnn.lauetools.FitOrient as FitO
import lauetoolsnn.lauetools.LaueGeometry as F2TC

## Different functions
def getProximity(TwicethetaChi, data_theta, data_chi, data_hkl, angtol=0.5):
    """This functions gives the indices of all the experimetnal spots that are close to the simulated spots"""
    # theo simul data
    theodata = np.array([TwicethetaChi[0] / 2.0, TwicethetaChi[1]]).T
    # exp data
    sorted_data = np.array([data_theta, data_chi]).T
    table_dist = GT.calculdist_from_thetachi(sorted_data, theodata)
    prox_table = np.argmin(table_dist, axis=1)
    allresidues = np.amin(table_dist, axis=1)
    very_close_ind = np.where(allresidues < angtol)[0]
    List_Exp_spot_close = []
    Miller_Exp_spot = []
    if len(very_close_ind) > 0:
        for theospot_ind in very_close_ind:  # loop over theo spots index
            List_Exp_spot_close.append(prox_table[theospot_ind])
            Miller_Exp_spot.append(data_hkl[theospot_ind])
    else:
        return [],[],[]
    # removing exp spot which appears many times(close to several simulated spots of one grain)--------------
    arrayLESC = np.array(List_Exp_spot_close, dtype=float)
    sorted_LESC = np.sort(arrayLESC)
    diff_index = sorted_LESC - np.array(list(sorted_LESC[1:]) + [sorted_LESC[0]])
    toremoveindex = np.where(diff_index == 0)[0]
    if len(toremoveindex) > 0:
        # index of exp spot in arrayLESC that are duplicated
        ambiguous_exp_ind = GT.find_closest(np.array(sorted_LESC[toremoveindex], dtype=float), arrayLESC, 0.1)[1]
        for ind in ambiguous_exp_ind:
            Miller_Exp_spot[ind] = None
    ProxTablecopy = np.copy(prox_table)
    for theo_ind, exp_ind in enumerate(prox_table):
        where_th_ind = np.where(ProxTablecopy == exp_ind)[0]
        if len(where_th_ind) > 1:
            for indy in where_th_ind:
                ProxTablecopy[indy] = -prox_table[indy]
            closest = np.argmin(allresidues[where_th_ind])
            ProxTablecopy[where_th_ind[closest]] = -ProxTablecopy[where_th_ind[closest]]
    singleindices = []
    refine_indexed_spots = {}
    # loop over close exp. spots
    for k in range(len(List_Exp_spot_close)):
        exp_index = List_Exp_spot_close[k]
        if not singleindices.count(exp_index):
            singleindices.append(exp_index)
            theo_index = np.where(ProxTablecopy == exp_index)[0]
            if (len(theo_index) == 1):  # only one theo spot close to the current exp. spot
                refine_indexed_spots[exp_index] = [exp_index, theo_index, Miller_Exp_spot[k]]
            else:  # recent PATCH:
                closest_theo_ind = np.argmin(allresidues[theo_index])
                if allresidues[theo_index][closest_theo_ind] < angtol:
                    refine_indexed_spots[exp_index] = [exp_index, theo_index[closest_theo_ind], Miller_Exp_spot[k]]
    listofpairs = []
    linkExpMiller = []
    selectedAbsoluteSpotIndices = np.arange(len(data_theta))
    for val in list(refine_indexed_spots.values()):
        if val[2] is not None:
            localspotindex = val[0]
            if not isinstance(val[1], (list, np.ndarray)):
                closetheoindex = val[1]
            else:
                closetheoindex = val[1][0]
            absolute_spot_index = selectedAbsoluteSpotIndices[localspotindex]
            listofpairs.append([absolute_spot_index, closetheoindex])  # Exp, Theo,  where -1 for specifying that it came from automatic linking
            linkExpMiller.append([float(absolute_spot_index)] + [float(elem) for elem in val[2]])  # float(val) for further handling as floats array
    linkedspots_link = np.array(listofpairs)
    linkExpMiller_link = linkExpMiller
    return linkedspots_link, linkExpMiller_link

def error_function_on_demand_strain(param_strain, DATA_Q, nspots,
                                    pixX, pixY, initrot=np.eye(3), Bmat=np.eye(3),
                                    verbose=0, detectorparameters=None, pixelsize=165.0 / 2048., weights=None,
                                    kf_direction="Z>0"):
    DEG = np.pi / 180.0
    ## Building rotation matrix along X, Y and Z
    a1 = param_strain[5] * DEG
    mat1 = np.array([[np.cos(a1), 0, np.sin(a1)], [0, 1, 0], [-np.sin(a1), 0, np.cos(a1)]])
    a2 = param_strain[6] * DEG
    mat2 = np.array([[1, 0, 0], [0, np.cos(a2), np.sin(a2)], [0, np.sin(-a2), np.cos(a2)]])
    a3 = param_strain[7] * DEG
    mat3 = np.array([[np.cos(a3), -np.sin(a3), 0], [np.sin(a3), np.cos(a3), 0], [0, 0, 1]])
    deltamat = np.dot(mat3, np.dot(mat2, mat1))

    # building B mat with proposed lattice parameters (except 'a' is fixed to 1)
    varyingstrain = np.array([[1.0, param_strain[2], param_strain[3]], [0, param_strain[0], param_strain[4]],  [0, 0, param_strain[1]]])
    newmatrix = np.dot(np.dot(deltamat, initrot), varyingstrain)
    
    X, Y, _, _ = xy_from_Quat(DATA_Q, nspots,
                                initrot=newmatrix, vecteurref=Bmat,
                                pixelsize=pixelsize, detectorparameters=detectorparameters, kf_direction=kf_direction)
    distanceterm = np.sqrt((X - pixX) ** 2 + (Y - pixY) ** 2)
    if weights is not None:
        allweights = np.sum(weights)
        distanceterm = distanceterm * weights / allweights
    if verbose:
        return distanceterm, deltamat, newmatrix
    else:
        return distanceterm
    
def xy_from_Quat(DATA_Q, nspots, initrot=None,  vecteurref=np.eye(3),
                 pixelsize=165.0 / 2048, detectorparameters=None, kf_direction="Z>0"):
    """
    compute x and y pixel positions of Laue spots given hkl list
    """
    # selecting nspots of DATA_Q
    DATAQ = np.take(DATA_Q, nspots, axis=0)
    trQ = np.transpose(DATAQ)  # np.array(Hs, Ks, Ls) for further computations
    # R is a pure rotation
    # dot(R,Q)=initrot # Q may be viewed as lattice distortion
    R = initrot # keep UB matrix rotation + distorsion
    # initial lattice rotation and distorsion (/ cubic structure)  q = U*B * Q
    trQ = np.dot(np.dot(R, vecteurref), trQ)
    # results are qx,qy,qz
    matfromQuat = np.eye(3)
    Qrot = np.dot(matfromQuat, trQ)  # lattice rotation due to quaternion
    Qrotn = np.sqrt(np.sum(Qrot ** 2, axis=0))  # norms of Q vectors
    twthe, chi = F2TC.from_qunit_to_twchi(1.*Qrot / Qrotn)
    X, Y, theta = F2TC.calc_xycam_from2thetachi(twthe, chi, detectorparameters, verbose=0, pixelsize=pixelsize, kf_direction=kf_direction)
    return X, Y, theta, R

class strain_estimator():
    def __init__(self, mat, detectorparameters, pixelsize, UBmat, angtol, s_tth, s_chi, exp_posx, exp_posy, Bmat):
        self.detectorparameters = detectorparameters
        self.pixelsize = pixelsize
        self.kf_direction = "Z>0"
        self.mat = mat
        self.UBmat = UBmat
        self.s_tth = s_tth
        self.s_chi = s_chi
        self.angtol = angtol
        self.exp_posx = exp_posx
        self.exp_posy = exp_posy
        self.Bmat = Bmat
        self.expXY = np.hstack((self.exp_posx,self.exp_posy))
        self.reupdate()
    
    def reupdate(self):
        self.grain = CP.Prepare_Grain(self.mat, self.UBmat)
        Twicetheta, Chi, Miller_ind, _, _, _ = LT.SimulateLaue_full_np(self.grain, 5, 22,
                                                                        self.detectorparameters,
                                                                        pixelsize=self.pixelsize,
                                                                        dim=(2018,2016),
                                                                        detectordiameter=self.pixelsize*2018*1.3,
                                                                        removeharmonics=1)
        ## get proximity for exp and theo spots
        linkedspots_link, linkExpMiller_link = getProximity(np.array([Twicetheta, Chi]),  # warning array(2theta, chi) ## Simulated
                                                            self.s_tth/2.0, self.s_chi, # warning theta, chi for exp
                                                            Miller_ind, angtol=self.angtol)
        arraycouples = np.array(linkedspots_link)
        self.exp_indices = np.array(arraycouples[:, 0], dtype=np.int)
        self.sim_indices = np.array(arraycouples[:, 1], dtype=np.int)

        nb_pairs = len(self.exp_indices)
        self.Data_Q = np.array(linkExpMiller_link)[:, 1:]
        self.sim_indices = np.arange(nb_pairs)  # for fitting function this must be an arange...

        self.pixX = np.take(self.exp_posx, self.exp_indices)
        self.pixY = np.take(self.exp_posy, self.exp_indices)
        self.expXY = np.hstack((self.pixX,self.pixY))
    
    def process_args(self, args, kwargs):
        '''
        Converts args or kwargs into a dictionary mapping input names to their
        values.
        :return: input map (dictionary)
        '''
        if not args and kwargs:
            return kwargs
        elif args and type(args[0]) is dict:
            return args[0]
        else:
            self._raise_error_processing_args()
        return None
    
    def _raise_error_processing_args(self):
        msg = '%s.evaluate() accepts a single dictionary or keyword args.' \
            % self.__class__.__name__
        raise TypeError(msg)
        
    def evaluate(self, *args, **kwargs):
        self.reupdate()
        params = self.process_args(args, kwargs)
        b = params['b']
        c = params['c']
        alpha = params['alpha']
        beta = params['beta']
        gamma = params['gamma']
        angx = params['angx']
        angy = params['angy']
        angz = params['angz']
        
        a = 1 ## one constant length
        #print( b, c, alpha, beta, gamma, angx, angy, angz)
        initrot = self.UBmat
        Bmat = self.Bmat
        ## Building rotation matrix along X, Y and Z
        a1 = angx * np.pi / 180.0
        mat1 = np.array([[np.cos(a1), 0, np.sin(a1)], [0, 1, 0], [-np.sin(a1), 0, np.cos(a1)]])
        a2 = angy * np.pi / 180.0
        mat2 = np.array([[1, 0, 0], [0, np.cos(a2), np.sin(a2)], [0, np.sin(-a2), np.cos(a2)]])
        a3 = angz * np.pi / 180.0
        mat3 = np.array([[np.cos(a3), -np.sin(a3), 0], [np.sin(a3), np.cos(a3), 0], [0, 0, 1]])
        deltamat = np.dot(mat3, np.dot(mat2, mat1))

        # building B mat with proposed lattice parameters (except 'a' is fixed to 1)
        varyingstrain = np.array([[1.0, alpha, beta], [0, b, gamma],  [0, 0, c]])
        newmatrix = np.dot(np.dot(deltamat, initrot), varyingstrain)

        # selecting nspots of DATA_Q
        DATAQ = np.take(self.Data_Q, self.sim_indices, axis=0)
        trQ = np.transpose(DATAQ)  # np.array(Hs, Ks, Ls) for further computations
        # R is a pure rotation
        R = newmatrix # keep UB matrix rotation + distorsion
        # initial lattice rotation and distorsion (/ cubic structure)  q = U*B * Q
        trQ = np.dot(np.dot(R, Bmat), trQ)
        # results are qx,qy,qz
        matfromQuat = np.eye(3)
        Qrot = np.dot(matfromQuat, trQ)  # lattice rotation due to quaternion
        Qrotn = np.sqrt(np.sum(Qrot ** 2, axis=0))  # norms of Q vectors
        twthe, chi = F2TC.from_qunit_to_twchi(1.*Qrot / Qrotn)
        X, Y, _ = F2TC.calc_xycam_from2thetachi(twthe, chi, self.detectorparameters, 
                                                verbose=0, pixelsize=self.pixelsize, 
                                                kf_direction=self.kf_direction)
        simXY =  np.hstack((X,Y))
        return simXY
    
    def evaluate1(self, b,c,alpha,beta,gamma,angx,angy,angz):
        self.reupdate()        
        a = 1 ## one constant length
        #print( b, c, alpha, beta, gamma, angx, angy, angz)
        initrot = self.UBmat
        Bmat = self.Bmat
        ## Building rotation matrix along X, Y and Z
        a1 = angx * np.pi / 180.0
        mat1 = np.array([[np.cos(a1), 0, np.sin(a1)], [0, 1, 0], [-np.sin(a1), 0, np.cos(a1)]])
        a2 = angy * np.pi / 180.0
        mat2 = np.array([[1, 0, 0], [0, np.cos(a2), np.sin(a2)], [0, np.sin(-a2), np.cos(a2)]])
        a3 = angz * np.pi / 180.0
        mat3 = np.array([[np.cos(a3), -np.sin(a3), 0], [np.sin(a3), np.cos(a3), 0], [0, 0, 1]])
        deltamat = np.dot(mat3, np.dot(mat2, mat1))

        # building B mat with proposed lattice parameters (except 'a' is fixed to 1)
        varyingstrain = np.array([[1.0, alpha, beta], [0, b, gamma],  [0, 0, c]])
        newmatrix = np.dot(np.dot(deltamat, initrot), varyingstrain)

        # selecting nspots of DATA_Q
        DATAQ = np.take(self.Data_Q, self.sim_indices, axis=0)
        trQ = np.transpose(DATAQ)  # np.array(Hs, Ks, Ls) for further computations
        # R is a pure rotation
        R = newmatrix # keep UB matrix rotation + distorsion
        # initial lattice rotation and distorsion (/ cubic structure)  q = U*B * Q
        trQ = np.dot(np.dot(R, Bmat), trQ)
        # results are qx,qy,qz
        matfromQuat = np.eye(3)
        Qrot = np.dot(matfromQuat, trQ)  # lattice rotation due to quaternion
        Qrotn = np.sqrt(np.sum(Qrot ** 2, axis=0))  # norms of Q vectors
        twthe, chi = F2TC.from_qunit_to_twchi(1.*Qrot / Qrotn)
        X, Y, _ = F2TC.calc_xycam_from2thetachi(twthe, chi, self.detectorparameters, 
                                                verbose=0, pixelsize=self.pixelsize, 
                                                kf_direction=self.kf_direction)
        distanceterm = (X - self.pixX) ** 2 + (Y - self.pixY) ** 2
        return distanceterm, len(X)
    
    def strain_simulator(self, b, c, alpha, beta, gamma, angx, angy, angz):
        self.reupdate()
        a = 1 ## one constant length
        print( b, c, alpha, beta, gamma, angx, angy, angz)
        initrot = self.UBmat
        Bmat = self.Bmat
        ## Building rotation matrix along X, Y and Z
        a1 = angx * np.pi / 180.0
        mat1 = np.array([[np.cos(a1), 0, np.sin(a1)], [0, 1, 0], [-np.sin(a1), 0, np.cos(a1)]])
        a2 = angy * np.pi / 180.0
        mat2 = np.array([[1, 0, 0], [0, np.cos(a2), np.sin(a2)], [0, np.sin(-a2), np.cos(a2)]])
        a3 = angz * np.pi / 180.0
        mat3 = np.array([[np.cos(a3), -np.sin(a3), 0], [np.sin(a3), np.cos(a3), 0], [0, 0, 1]])
        deltamat = np.dot(mat3, np.dot(mat2, mat1))

        # building B mat with proposed lattice parameters (except 'a' is fixed to 1)
        varyingstrain = np.array([[1.0, alpha, beta], [0, b, gamma],  [0, 0, c]])
        newmatrix = np.dot(np.dot(deltamat, initrot), varyingstrain)

        # selecting nspots of DATA_Q
        DATAQ = np.take(self.Data_Q, self.sim_indices, axis=0)
        trQ = np.transpose(DATAQ)  # np.array(Hs, Ks, Ls) for further computations
        # R is a pure rotation
        R = newmatrix # keep UB matrix rotation + distorsion
        # initial lattice rotation and distorsion (/ cubic structure)  q = U*B * Q
        trQ = np.dot(np.dot(R, Bmat), trQ)
        # results are qx,qy,qz
        matfromQuat = np.eye(3)
        Qrot = np.dot(matfromQuat, trQ)  # lattice rotation due to quaternion
        Qrotn = np.sqrt(np.sum(Qrot ** 2, axis=0))  # norms of Q vectors
        twthe, chi = F2TC.from_qunit_to_twchi(1.*Qrot / Qrotn)
        X, Y, _ = F2TC.calc_xycam_from2thetachi(twthe, chi, self.detectorparameters, 
                                                verbose=0, pixelsize=self.pixelsize, 
                                                kf_direction=self.kf_direction)
        simXY =  np.hstack((X,Y))
        return simXY, self.expXY
        