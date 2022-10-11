# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 21:05:54 2022

@author: PURUSHOT

A method for writing the data in pickle file to hdf5 format for later query

"""
import _pickle as cPickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

with open(r"C:\Users\purushot\Desktop\Laue_Zr_HT\model_crg4\ZrO2_1250C_jan2022_1250C_trial\1250_pfoc_normal\results.pickle", "rb") as input_file:
    best_match, \
    mat_global, rotation_matrix1, strain_matrix, strain_matrixs,\
        col, colx, coly, match_rate, files_treated,\
            lim_x, lim_y, spots_len, iR_pix, fR_pix, material_, \
                material1_, _, _, _, _,  _, _ = cPickle.load(input_file)
#best_match array contains all summary 

filenames = list(np.unique(files_treated))
filenames.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

ub_matricies = len(rotation_matrix1)

def reshape(a):
    m,n,r = a.shape
    return a.reshape((m,n*r))

dataframe_grains = []
for i in range(ub_matricies):
    ## Rotation matrix components
    a = reshape(rotation_matrix1[i][0])
    columnsa=['R11', 'R12', 'R13', 'R21', 'R22', 'R23', 'R31', 'R32', 'R33']
    
    ## Strain crystal frame components
    b = reshape(strain_matrix[i][0])
    columnsb=['Ceps11', 'Ceps12', 'Ceps13', 'Ceps21', 'Ceps22', 'Ceps23', 'Ceps31', 'Ceps32', 'Ceps33']
    
    ## Strain sample frame components
    c = reshape(strain_matrixs[i][0])
    columnsc=['Seps11', 'Seps12', 'Seps13', 'Seps21', 'Seps22', 'Seps23', 'Seps31', 'Seps32', 'Seps33']
    
    ## matching rate
    d = match_rate[i][0]
    columnsd=['MatchinRate']
    
    ## filename
    e = np.chararray(d.shape, itemsize=1000)
    for jj, ii in enumerate(filenames):
        e[jj] = ii
    columnse=['filename']
    
    ## spots_len
    f = spots_len[i][0]
    columnsf=['NbofSpots']
    
    ## initial residues
    g = iR_pix[i][0]
    columnsg=['initialResidue']
    
    ## final residues
    h = fR_pix[i][0]
    columnsh=['finalResidue']
    
    ## colx
    f1 = colx[i][0]
    columnsf1=['ipfXr', 'ipfXg', 'ipfXb']
    
    ## coly
    g1 = coly[i][0]
    columnsg1=['ipfYr', 'ipfYg', 'ipfYb']
    
    ## colz
    h1 = col[i][0]
    columnsh1=['ipfZr', 'ipfZg', 'ipfZb']
    
    ## convert matglobal to string of material label
    mat_id = mat_global[i][0]
    mat_label = np.chararray(mat_id.shape, itemsize=1000)
    for jj, ii in enumerate(mat_id):
        if int(ii) == 1:
            mat_label[jj] = material_
        elif int(ii) == 2:
            mat_label[jj] = material1_
    columns11=['material_label']
    
    
    columns = columnsa+columnsb+columnsc+columnsd+columnse+columnsf+\
                columnsg+columnsh+columnsf1+columnsg1+columnsh1+columns11
    
    out_array = np.hstack((a,b,c,d,e,f,g,h,f1,g1,h1,mat_label))
    
    columnstype={'R11':'float64', 'R12':'float64', 'R13':'float64', 'R21':'float64', 
                   'R22':'float64', 'R23':'float64', 'R31':'float64', 'R32':'float64', 
                   'R33':'float64', 'Ceps11':'float64', 'Ceps12':'float64', 'Ceps13':'float64', 'Ceps21':'float64', 
                   'Ceps22':'float64', 'Ceps23':'float64', 'Ceps31':'float64', 'Ceps32':'float64', 
                   'Ceps33':'float64', 'Seps11':'float64', 'Seps12':'float64', 
                   'Seps13':'float64', 'Seps21':'float64', 'Seps22':'float64', 'Seps23':'float64', 
                   'Seps31':'float64', 'Seps32':'float64', 'Seps33':'float64', 'MatchinRate':'float64',
                   'filename':'object', 'NbofSpots':'float64', 'initialResidue':'float64','finalResidue':'float64', 
                   'ipfXr':'float64', 'ipfXg':'float64', 'ipfXb':'float64',
                   'ipfYr':'float64', 'ipfYg':'float64', 'ipfYb':'float64', 'ipfZr':'float64', 
                   'ipfZg':'float64', 'ipfZb':'float64', 'material_label':'object'}
    
    # out_df = pd.DataFrame(out_array, columns=columns)
    # out_df = out_df.astype(dtype= columnstype)
    # out_df.to_hdf('grain_all.h5', key='grain'+str(i))
    # dataframe_grains.append(out_df)
    
    for ij in range(len(columns)):
        temp_ = out_array[:,ij]
        temp_ = temp_.reshape((lim_x, lim_y))
        out_df = pd.DataFrame(temp_)
        dtype_ = columnstype[columns[ij]]
        out_df = out_df.astype(dtype=dtype_)
        out_df.to_hdf('grain_all.h5', key='grain'+str(i)+"/"+columns[ij])


























