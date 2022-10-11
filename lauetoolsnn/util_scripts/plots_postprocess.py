# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 20:09:37 2021

@author: PURUSHOT

Post process the results on Cu Si pads
"""
import _pickle as cPickle
import matplotlib.pyplot as plt
import numpy as np

with open(r"results.pickle", "rb") as input_file:
    best_match, \
    mat_global, rotation_matrix1, strain_matrix, strain_matrixs,\
        col, colx, coly, match_rate, files_treated,\
            lim_x, lim_y, spots_len, iR_pix, fR_pix, material_, \
                material1_, lattice, lattice1, symmetry0, symmetry1,\
                    crystal, crystal1 = cPickle.load(input_file)

rangeval = len(rotation_matrix1)
match_tol = 1

rotation_matrix = [[] for i in range(len(rotation_matrix1))]

for i in range(len(rotation_matrix1)):
    rotation_matrix[i].append(np.zeros((lim_x*lim_y,3,3)))

for i in range(len(rotation_matrix1)):
    temp_mat = rotation_matrix1[i][0]
    temp_matglobal = mat_global[i][0]
    
    for j in range(len(temp_mat)):
        orientation_matrix = temp_mat[j,:,:]

        ## rotate orientation by 40degrees to bring in Sample RF
        omega = np.deg2rad(-40.0)
        # rotation de -omega autour de l'axe x (or Y?) pour repasser dans Rsample
        cw = np.cos(omega)
        sw = np.sin(omega)
        mat_from_lab_to_sample_frame = np.array([[cw, 0.0, sw], [0.0, 1.0, 0.0], [-sw, 0, cw]])
        orientation_matrix = np.dot(mat_from_lab_to_sample_frame, orientation_matrix)
    
        if np.linalg.det(orientation_matrix) < 0:
            orientation_matrix = -orientation_matrix
            
        rotation_matrix[i][0][j,:,:] = orientation_matrix
#%% IPF MAPS
for index in range(rangeval):
    for ii in range(len(np.unique(mat_global[index][0]))):
        mat_index1 = mat_global[index][0]
        mask_ = np.where(mat_index1 != ii+1)[0]
        
        col_plot = np.copy(col[index][0])
        col_plot[mask_,:] = 0,0,0
        col_plot = col_plot.reshape((lim_x, lim_y, 3))
    
        colx_plot = np.copy(colx[index][0])
        colx_plot[mask_,:] = 0,0,0
        colx_plot = colx_plot.reshape((lim_x, lim_y,3))
        
        coly_plot = np.copy(coly[index][0])
        coly_plot[mask_,:] = 0,0,0
        coly_plot = coly_plot.reshape((lim_x, lim_y,3))
        
        fig = plt.figure(figsize=(11.69,8.27), dpi=100)
        bottom, top = 0.1, 0.9
        left, right = 0.1, 0.8
        fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
    
        axs = fig.subplots(1, 3)
        axs[0].set_title(r"IPF Z map", loc='center', fontsize=8)
        axs[0].imshow(col_plot, origin='lower')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        
        axs[1].set_title(r"IPF Y map", loc='center', fontsize=8)
        axs[1].imshow(coly_plot, origin='lower', vmin=0, vmax=2)
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        
        axs[2].set_title(r"IPF X map", loc='center', fontsize=8)
        im = axs[2].imshow(colx_plot, origin='lower', cmap=plt.cm.jet, vmin=0, vmax=100)
        axs[2].set_xticks([])
        axs[2].set_yticks([])

        for ax in axs.flat:
            ax.label_outer()
    
        plt.savefig('IPF_map_'+str(index)+"_"+str(ii+1)+'.png', bbox_inches='tight',format='png', dpi=1000) 
        plt.close(fig)
#%% Plot histograms
rangeval = len(match_rate)
for index in range(rangeval):
    bins = 20
    nan_index = np.where(match_rate[index][0] < match_tol)[0]
    
    spots_len_plot = np.copy(spots_len[index][0])
    spots_len_plot[nan_index] = np.nan 
    mr_plot = np.copy(match_rate[index][0])
    mr_plot[nan_index] = np.nan 
    try:
        title = "Number of spots and matching rate"
        fig = plt.figure()
        axs = fig.subplots(1, 2)
        axs[0].set_title("Number of spots", loc='center', fontsize=8)
        axs[0].hist(spots_len_plot, bins=bins)
        axs[0].set_ylabel('Frequency', fontsize=8)
        axs[0].tick_params(axis='both', which='major', labelsize=8)
        axs[0].tick_params(axis='both', which='minor', labelsize=8)
        axs[1].set_title("matching rate", loc='center', fontsize=8)
        axs[1].hist(mr_plot, bins=bins)
        axs[1].set_ylabel('Frequency', fontsize=8)
        axs[1].tick_params(axis='both', which='major', labelsize=8)
        axs[1].tick_params(axis='both', which='minor', labelsize=8)
        plt.tight_layout()
        plt.savefig(title+"_"+str(index)+'.png', format='png', dpi=1000) 
        plt.close(fig)
    except:
        pass
    
    iR_pix_plot = np.copy(iR_pix[index][0])
    fR_pix_plot = np.copy(fR_pix[index][0])
    iR_pix_plot[nan_index] = np.nan 
    fR_pix_plot[nan_index] = np.nan 
    try:
        title = "Initial and Final residues"
        fig = plt.figure()
        axs = fig.subplots(1, 2)
        axs[0].set_title("Initial residues", loc='center', fontsize=8)
        axs[0].hist(iR_pix_plot, bins=bins)
        axs[0].set_ylabel('Frequency', fontsize=8)
        axs[0].tick_params(axis='both', which='major', labelsize=8)
        axs[0].tick_params(axis='both', which='minor', labelsize=8)
        axs[1].set_title("Final residues", loc='center', fontsize=8)
        axs[1].hist(fR_pix_plot, bins=bins)
        axs[1].set_ylabel('Frequency', fontsize=8)
        axs[1].tick_params(axis='both', which='major', labelsize=8)
        axs[1].tick_params(axis='both', which='minor', labelsize=8)
        plt.tight_layout()
        plt.savefig(title+"_"+str(index)+'.png', format='png', dpi=1000) 
        plt.close(fig)
    except:
        pass
    strain_matrix_plot = np.copy(strain_matrix[index][0])
    e11c = strain_matrix_plot[:,0,0]#.reshape((lim_x, lim_y))
    e22c = strain_matrix_plot[:,1,1]#.reshape((lim_x, lim_y))
    e33c = strain_matrix_plot[:,2,2]#.reshape((lim_x, lim_y))
    e12c = strain_matrix_plot[:,0,1]#.reshape((lim_x, lim_y))
    e13c = strain_matrix_plot[:,0,2]#.reshape((lim_x, lim_y))
    e23c = strain_matrix_plot[:,1,2]#.reshape((lim_x, lim_y))
    e11c[nan_index] = np.nan 
    e22c[nan_index] = np.nan 
    e33c[nan_index] = np.nan 
    e12c[nan_index] = np.nan 
    e13c[nan_index] = np.nan 
    e23c[nan_index] = np.nan 
    try:
        title = "strain Crystal reference"
        fig = plt.figure()
        axs = fig.subplots(2, 3)
        axs[0, 0].set_title(r"$\epsilon_{11}$", loc='center', fontsize=8)
        axs[0, 0].hist(e11c, bins=bins)
        axs[0, 0].set_ylabel('Frequency', fontsize=8)
        axs[0, 0].tick_params(axis='both', which='major', labelsize=8)
        axs[0, 0].tick_params(axis='both', which='minor', labelsize=8)
        
        axs[0, 1].set_title(r"$\epsilon_{22}$", loc='center', fontsize=8)
        axs[0, 1].hist(e22c, bins=bins)
        axs[0, 1].set_ylabel('Frequency', fontsize=8)
        axs[0, 1].tick_params(axis='both', which='major', labelsize=8)
        axs[0, 1].tick_params(axis='both', which='minor', labelsize=8)
        
        axs[0, 2].set_title(r"$\epsilon_{33}$", loc='center', fontsize=8)
        axs[0, 2].hist(e33c, bins=bins)
        axs[0, 2].set_ylabel('Frequency', fontsize=8)
        axs[0, 2].tick_params(axis='both', which='major', labelsize=8)
        axs[0, 2].tick_params(axis='both', which='minor', labelsize=8)
        
        axs[1, 0].set_title(r"$\epsilon_{12}$", loc='center', fontsize=8)
        axs[1, 0].hist(e12c, bins=bins)
        axs[1, 0].set_ylabel('Frequency', fontsize=8)
        axs[1, 0].tick_params(axis='both', which='major', labelsize=8)
        axs[1, 0].tick_params(axis='both', which='minor', labelsize=8)
        
        axs[1, 1].set_title(r"$\epsilon_{13}$", loc='center', fontsize=8)
        axs[1, 1].hist(e13c, bins=bins)
        axs[1, 1].set_ylabel('Frequency', fontsize=8)
        axs[1, 1].tick_params(axis='both', which='major', labelsize=8)
        axs[1, 1].tick_params(axis='both', which='minor', labelsize=8)
        
        axs[1, 2].set_title(r"$\epsilon_{23}$", loc='center', fontsize=8)
        axs[1, 2].hist(e23c, bins=bins)
        axs[1, 2].set_ylabel('Frequency', fontsize=8)
        axs[1, 2].tick_params(axis='both', which='major', labelsize=8)
        axs[1, 2].tick_params(axis='both', which='minor', labelsize=8)
        plt.tight_layout()
        plt.savefig(title+"_"+str(index)+'.png', format='png', dpi=1000) 
        plt.close(fig)
    except:
        pass
    strain_matrixs_plot = np.copy(strain_matrixs[index][0])
    e11s = strain_matrixs_plot[:,0,0]#.reshape((lim_x, lim_y))
    e22s = strain_matrixs_plot[:,1,1]#.reshape((lim_x, lim_y))
    e33s = strain_matrixs_plot[:,2,2]#.reshape((lim_x, lim_y))
    e12s = strain_matrixs_plot[:,0,1]#.reshape((lim_x, lim_y))
    e13s = strain_matrixs_plot[:,0,2]#.reshape((lim_x, lim_y))
    e23s = strain_matrixs_plot[:,1,2]#.reshape((lim_x, lim_y))
    e11s[nan_index] = np.nan 
    e22s[nan_index] = np.nan 
    e33s[nan_index] = np.nan 
    e12s[nan_index] = np.nan 
    e13s[nan_index] = np.nan 
    e23s[nan_index] = np.nan 
    try:
        title = "strain Sample reference"
        fig = plt.figure()
        axs = fig.subplots(2, 3)
        axs[0, 0].set_title(r"$\epsilon_{11}$", loc='center', fontsize=8)
        axs[0, 0].hist(e11s, bins=bins)
        axs[0, 0].set_ylabel('Frequency', fontsize=8)
        axs[0, 0].tick_params(axis='both', which='major', labelsize=8)
        axs[0, 0].tick_params(axis='both', which='minor', labelsize=8)
        
        axs[0, 1].set_title(r"$\epsilon_{22}$", loc='center', fontsize=8)
        axs[0, 1].hist(e22s, bins=bins)
        axs[0, 1].set_ylabel('Frequency', fontsize=8)
        axs[0, 1].tick_params(axis='both', which='major', labelsize=8)
        axs[0, 1].tick_params(axis='both', which='minor', labelsize=8)
        
        axs[0, 2].set_title(r"$\epsilon_{33}$", loc='center', fontsize=8)
        axs[0, 2].hist(e33s, bins=bins)
        axs[0, 2].set_ylabel('Frequency', fontsize=8)
        axs[0, 2].tick_params(axis='both', which='major', labelsize=8)
        axs[0, 2].tick_params(axis='both', which='minor', labelsize=8)
        
        axs[1, 0].set_title(r"$\epsilon_{12}$", loc='center', fontsize=8)
        axs[1, 0].hist(e12s, bins=bins)
        axs[1, 0].set_ylabel('Frequency', fontsize=8)
        axs[1, 0].tick_params(axis='both', which='major', labelsize=8)
        axs[1, 0].tick_params(axis='both', which='minor', labelsize=8)
        
        axs[1, 1].set_title(r"$\epsilon_{13}$", loc='center', fontsize=8)
        axs[1, 1].hist(e13s, bins=bins)
        axs[1, 1].set_ylabel('Frequency', fontsize=8)
        axs[1, 1].tick_params(axis='both', which='major', labelsize=8)
        axs[1, 1].tick_params(axis='both', which='minor', labelsize=8)
        
        axs[1, 2].set_title(r"$\epsilon_{23}$", loc='center', fontsize=8)
        axs[1, 2].hist(e23s, bins=bins)
        axs[1, 2].set_ylabel('Frequency', fontsize=8)
        axs[1, 2].tick_params(axis='both', which='major', labelsize=8)
        axs[1, 2].tick_params(axis='both', which='minor', labelsize=8)
        plt.tight_layout()
        plt.savefig(title+"_"+str(index)+'.png', format='png', dpi=1000) 
        plt.close(fig)
    except:
        pass



if material_ == material1_:
    count = 0
    for index in range(rangeval):
        bins = 20
        ### index for nans
        nan_index = np.where(match_rate[index][0] < match_tol)[0]
        if count == 0:
            spots_len_plot = np.copy(spots_len[index][0])
            mr_plot = np.copy(match_rate[index][0])
            iR_pix_plot = np.copy(iR_pix[index][0])
            fR_pix_plot = np.copy(fR_pix[index][0])
            strain_matrix_plot = np.copy(strain_matrix[index][0])
            e11c = strain_matrix_plot[:,0,0]#.reshape((lim_x, lim_y))
            e22c = strain_matrix_plot[:,1,1]#.reshape((lim_x, lim_y))
            e33c = strain_matrix_plot[:,2,2]#.reshape((lim_x, lim_y))
            e12c = strain_matrix_plot[:,0,1]#.reshape((lim_x, lim_y))
            e13c = strain_matrix_plot[:,0,2]#.reshape((lim_x, lim_y))
            e23c = strain_matrix_plot[:,1,2]#.reshape((lim_x, lim_y))
            strain_matrixs_plot = np.copy(strain_matrixs[index][0])
            e11s = strain_matrixs_plot[:,0,0]#.reshape((lim_x, lim_y))
            e22s = strain_matrixs_plot[:,1,1]#.reshape((lim_x, lim_y))
            e33s = strain_matrixs_plot[:,2,2]#.reshape((lim_x, lim_y))
            e12s = strain_matrixs_plot[:,0,1]#.reshape((lim_x, lim_y))
            e13s = strain_matrixs_plot[:,0,2]#.reshape((lim_x, lim_y))
            e23s = strain_matrixs_plot[:,1,2]#.reshape((lim_x, lim_y))
            spots_len_plot[nan_index] = np.nan 
            mr_plot[nan_index] = np.nan 
            iR_pix_plot[nan_index] = np.nan 
            fR_pix_plot[nan_index] = np.nan 
            e11c[nan_index] = np.nan 
            e22c[nan_index] = np.nan 
            e33c[nan_index] = np.nan 
            e12c[nan_index] = np.nan 
            e13c[nan_index] = np.nan 
            e23c[nan_index] = np.nan 
            e11s[nan_index] = np.nan 
            e22s[nan_index] = np.nan 
            e33s[nan_index] = np.nan 
            e12s[nan_index] = np.nan 
            e13s[nan_index] = np.nan 
            e23s[nan_index] = np.nan 
            count = 1
            
        else:
            temp = np.copy(spots_len[index][0])
            temp[nan_index] = np.nan
            spots_len_plot = np.vstack((spots_len_plot,temp))
            
            temp = np.copy(match_rate[index][0])
            temp[nan_index] = np.nan
            mr_plot = np.vstack((mr_plot,temp))
            
            temp = np.copy(iR_pix[index][0])
            temp[nan_index] = np.nan
            iR_pix_plot = np.vstack((iR_pix_plot,temp))
    
            temp = np.copy(fR_pix[index][0])
            temp[nan_index] = np.nan
            fR_pix_plot = np.vstack((fR_pix_plot,temp))
            
            strain_matrix_plot = np.copy(strain_matrix[index][0])
            temp = np.copy(strain_matrix_plot[:,0,0])
            temp[nan_index] = np.nan
            e11c = np.vstack((e11c,temp))
            temp = np.copy(strain_matrix_plot[:,1,1])
            temp[nan_index] = np.nan
            e22c = np.vstack((e22c,temp))
            temp = np.copy(strain_matrix_plot[:,2,2])
            temp[nan_index] = np.nan
            e33c = np.vstack((e33c,temp))
            temp = np.copy(strain_matrix_plot[:,0,1])
            temp[nan_index] = np.nan
            e12c = np.vstack((e12c,temp))
            temp = np.copy(strain_matrix_plot[:,0,2])
            temp[nan_index] = np.nan
            e13c = np.vstack((e13c,temp))
            temp = np.copy(strain_matrix_plot[:,1,2])
            temp[nan_index] = np.nan
            e23c = np.vstack((e23c,temp))
            ##
            strain_matrixs_plot = np.copy(strain_matrixs[index][0])
            temp = np.copy(strain_matrixs_plot[:,0,0])
            temp[nan_index] = np.nan
            e11s = np.vstack((e11s,temp))
            temp = np.copy(strain_matrixs_plot[:,1,1])
            temp[nan_index] = np.nan
            e22s = np.vstack((e22s,temp))
            temp = np.copy(strain_matrixs_plot[:,2,2])
            temp[nan_index] = np.nan
            e33s = np.vstack((e33s,temp))
            temp = np.copy(strain_matrixs_plot[:,0,1])
            temp[nan_index] = np.nan
            e12s = np.vstack((e12s,temp))
            temp = np.copy(strain_matrixs_plot[:,0,2])
            temp[nan_index] = np.nan
            e13s = np.vstack((e13s,temp))
            temp = np.copy(strain_matrixs_plot[:,1,2])
            temp[nan_index] = np.nan
            e23s = np.vstack((e23s,temp))
    
    spots_len_plot = spots_len_plot.flatten()
    mr_plot = mr_plot.flatten()
    iR_pix_plot = iR_pix_plot.flatten()
    fR_pix_plot = fR_pix_plot.flatten() 
    e11c = e11c.flatten()
    e22c = e22c.flatten()
    e33c = e33c.flatten()
    e12c = e12c.flatten()
    e13c = e13c.flatten()
    e23c = e23c.flatten()
    e11s = e11s.flatten()
    e22s = e22s.flatten()
    e33s = e33s.flatten()
    e12s = e12s.flatten()
    e13s = e13s.flatten()
    e23s = e23s.flatten()
    
    try:
        title = "Number of spots and matching rate"
        fig = plt.figure()
        axs = fig.subplots(1, 2)
        axs[0].set_title("Number of spots", loc='center', fontsize=8)
        axs[0].hist(spots_len_plot, bins=bins)
        axs[0].set_ylabel('Frequency', fontsize=8)
        axs[0].tick_params(axis='both', which='major', labelsize=8)
        axs[0].tick_params(axis='both', which='minor', labelsize=8)
        axs[1].set_title("matching rate", loc='center', fontsize=8)
        axs[1].hist(mr_plot, bins=bins)
        axs[1].set_ylabel('Frequency', fontsize=8)
        axs[1].tick_params(axis='both', which='major', labelsize=8)
        axs[1].tick_params(axis='both', which='minor', labelsize=8)
        plt.tight_layout()
        plt.savefig(title+'.png', format='png', dpi=1000) 
        plt.close(fig)
    except:
        pass
    try:
        title = "Initial and Final residues"
        fig = plt.figure()
        axs = fig.subplots(1, 2)
        axs[0].set_title("Initial residues", loc='center', fontsize=8)
        axs[0].hist(iR_pix_plot, bins=bins)
        axs[0].set_ylabel('Frequency', fontsize=8)
        axs[0].tick_params(axis='both', which='major', labelsize=8)
        axs[0].tick_params(axis='both', which='minor', labelsize=8)
        axs[1].set_title("Final residues", loc='center', fontsize=8)
        axs[1].hist(fR_pix_plot, bins=bins)
        axs[1].set_ylabel('Frequency', fontsize=8)
        axs[1].tick_params(axis='both', which='major', labelsize=8)
        axs[1].tick_params(axis='both', which='minor', labelsize=8)
        plt.tight_layout()
        plt.savefig(title+'.png',format='png', dpi=1000) 
        plt.close(fig)
    except:
        pass
    try:
        title = "strain Crystal reference"
        fig = plt.figure()
        axs = fig.subplots(2, 3)
        axs[0, 0].set_title(r"$\epsilon_{11}$", loc='center', fontsize=8)
        axs[0, 0].hist(e11c, bins=bins)
        axs[0, 0].set_ylabel('Frequency', fontsize=8)
        axs[0, 0].tick_params(axis='both', which='major', labelsize=8)
        axs[0, 0].tick_params(axis='both', which='minor', labelsize=8)
        
        axs[0, 1].set_title(r"$\epsilon_{22}$", loc='center', fontsize=8)
        axs[0, 1].hist(e22c, bins=bins)
        axs[0, 1].set_ylabel('Frequency', fontsize=8)
        axs[0, 1].tick_params(axis='both', which='major', labelsize=8)
        axs[0, 1].tick_params(axis='both', which='minor', labelsize=8)
        
        axs[0, 2].set_title(r"$\epsilon_{33}$", loc='center', fontsize=8)
        axs[0, 2].hist(e33c, bins=bins)
        axs[0, 2].set_ylabel('Frequency', fontsize=8)
        axs[0, 2].tick_params(axis='both', which='major', labelsize=8)
        axs[0, 2].tick_params(axis='both', which='minor', labelsize=8)
        
        axs[1, 0].set_title(r"$\epsilon_{12}$", loc='center', fontsize=8)
        axs[1, 0].hist(e12c, bins=bins)
        axs[1, 0].set_ylabel('Frequency', fontsize=8)
        axs[1, 0].tick_params(axis='both', which='major', labelsize=8)
        axs[1, 0].tick_params(axis='both', which='minor', labelsize=8)
        
        axs[1, 1].set_title(r"$\epsilon_{13}$", loc='center', fontsize=8)
        axs[1, 1].hist(e13c, bins=bins)
        axs[1, 1].set_ylabel('Frequency', fontsize=8)
        axs[1, 1].tick_params(axis='both', which='major', labelsize=8)
        axs[1, 1].tick_params(axis='both', which='minor', labelsize=8)
        
        axs[1, 2].set_title(r"$\epsilon_{23}$", loc='center', fontsize=8)
        axs[1, 2].hist(e23c, bins=bins)
        axs[1, 2].set_ylabel('Frequency', fontsize=8)
        axs[1, 2].tick_params(axis='both', which='major', labelsize=8)
        axs[1, 2].tick_params(axis='both', which='minor', labelsize=8)
        plt.tight_layout()
        plt.savefig(title+'.png', format='png', dpi=1000) 
        plt.close(fig)
    except:
        pass
    try:
        title = "strain Sample reference"
        fig = plt.figure()
        axs = fig.subplots(2, 3)
        axs[0, 0].set_title(r"$\epsilon_{11}$", loc='center', fontsize=8)
        axs[0, 0].hist(e11s, bins=bins)
        axs[0, 0].set_ylabel('Frequency', fontsize=8)
        axs[0, 0].tick_params(axis='both', which='major', labelsize=8)
        axs[0, 0].tick_params(axis='both', which='minor', labelsize=8)
        
        axs[0, 1].set_title(r"$\epsilon_{22}$", loc='center', fontsize=8)
        axs[0, 1].hist(e22s, bins=bins)
        axs[0, 1].set_ylabel('Frequency', fontsize=8)
        axs[0, 1].tick_params(axis='both', which='major', labelsize=8)
        axs[0, 1].tick_params(axis='both', which='minor', labelsize=8)
        
        axs[0, 2].set_title(r"$\epsilon_{33}$", loc='center', fontsize=8)
        axs[0, 2].hist(e33s, bins=bins)
        axs[0, 2].set_ylabel('Frequency', fontsize=8)
        axs[0, 2].tick_params(axis='both', which='major', labelsize=8)
        axs[0, 2].tick_params(axis='both', which='minor', labelsize=8)
        
        axs[1, 0].set_title(r"$\epsilon_{12}$", loc='center', fontsize=8)
        axs[1, 0].hist(e12s, bins=bins)
        axs[1, 0].set_ylabel('Frequency', fontsize=8)
        axs[1, 0].tick_params(axis='both', which='major', labelsize=8)
        axs[1, 0].tick_params(axis='both', which='minor', labelsize=8)
        
        axs[1, 1].set_title(r"$\epsilon_{13}$", loc='center', fontsize=8)
        axs[1, 1].hist(e13s, bins=bins)
        axs[1, 1].set_ylabel('Frequency', fontsize=8)
        axs[1, 1].tick_params(axis='both', which='major', labelsize=8)
        axs[1, 1].tick_params(axis='both', which='minor', labelsize=8)
        
        axs[1, 2].set_title(r"$\epsilon_{23}$", loc='center', fontsize=8)
        axs[1, 2].hist(e23s, bins=bins)
        axs[1, 2].set_ylabel('Frequency', fontsize=8)
        axs[1, 2].tick_params(axis='both', which='major', labelsize=8)
        axs[1, 2].tick_params(axis='both', which='minor', labelsize=8)
        plt.tight_layout()
        plt.savefig(title+'.png', format='png', dpi=1000) 
        plt.close(fig)  
    except:
        pass
    
else:
    material_id = [material_, material1_]
    for matid in range(2):
        count = 0
        for index in range(rangeval):
            bins = 20
            ### index for nans
            nan_index1 = np.where(match_rate[index][0] < match_tol)[0]
            mat_id_index = np.where(mat_global[index][0] != matid+1)[0]
            nan_index = np.hstack((mat_id_index,nan_index1))
            nan_index = np.unique(nan_index)
            
            if count == 0:
                spots_len_plot = np.copy(spots_len[index][0])
                mr_plot = np.copy(match_rate[index][0])
                iR_pix_plot = np.copy(iR_pix[index][0])
                fR_pix_plot = np.copy(fR_pix[index][0])
                strain_matrix_plot = np.copy(strain_matrix[index][0])
                e11c = strain_matrix_plot[:,0,0]#.reshape((lim_x, lim_y))
                e22c = strain_matrix_plot[:,1,1]#.reshape((lim_x, lim_y))
                e33c = strain_matrix_plot[:,2,2]#.reshape((lim_x, lim_y))
                e12c = strain_matrix_plot[:,0,1]#.reshape((lim_x, lim_y))
                e13c = strain_matrix_plot[:,0,2]#.reshape((lim_x, lim_y))
                e23c = strain_matrix_plot[:,1,2]#.reshape((lim_x, lim_y))
                strain_matrixs_plot = np.copy(strain_matrixs[index][0])
                e11s = strain_matrixs_plot[:,0,0]#.reshape((lim_x, lim_y))
                e22s = strain_matrixs_plot[:,1,1]#.reshape((lim_x, lim_y))
                e33s = strain_matrixs_plot[:,2,2]#.reshape((lim_x, lim_y))
                e12s = strain_matrixs_plot[:,0,1]#.reshape((lim_x, lim_y))
                e13s = strain_matrixs_plot[:,0,2]#.reshape((lim_x, lim_y))
                e23s = strain_matrixs_plot[:,1,2]#.reshape((lim_x, lim_y))
                spots_len_plot[nan_index] = np.nan 
                mr_plot[nan_index] = np.nan 
                iR_pix_plot[nan_index] = np.nan 
                fR_pix_plot[nan_index] = np.nan 
                e11c[nan_index] = np.nan 
                e22c[nan_index] = np.nan 
                e33c[nan_index] = np.nan 
                e12c[nan_index] = np.nan 
                e13c[nan_index] = np.nan 
                e23c[nan_index] = np.nan 
                e11s[nan_index] = np.nan 
                e22s[nan_index] = np.nan 
                e33s[nan_index] = np.nan 
                e12s[nan_index] = np.nan 
                e13s[nan_index] = np.nan 
                e23s[nan_index] = np.nan 
                count = 1
                
            else:
                temp = np.copy(spots_len[index][0])
                temp[nan_index] = np.nan
                spots_len_plot = np.vstack((spots_len_plot,temp))
                
                temp = np.copy(match_rate[index][0])
                temp[nan_index] = np.nan
                mr_plot = np.vstack((mr_plot,temp))
                
                temp = np.copy(iR_pix[index][0])
                temp[nan_index] = np.nan
                iR_pix_plot = np.vstack((iR_pix_plot,temp))
        
                temp = np.copy(fR_pix[index][0])
                temp[nan_index] = np.nan
                fR_pix_plot = np.vstack((fR_pix_plot,temp))
                
                strain_matrix_plot = np.copy(strain_matrix[index][0])
                temp = np.copy(strain_matrix_plot[:,0,0])
                temp[nan_index] = np.nan
                e11c = np.vstack((e11c,temp))
                temp = np.copy(strain_matrix_plot[:,1,1])
                temp[nan_index] = np.nan
                e22c = np.vstack((e22c,temp))
                temp = np.copy(strain_matrix_plot[:,2,2])
                temp[nan_index] = np.nan
                e33c = np.vstack((e33c,temp))
                temp = np.copy(strain_matrix_plot[:,0,1])
                temp[nan_index] = np.nan
                e12c = np.vstack((e12c,temp))
                temp = np.copy(strain_matrix_plot[:,0,2])
                temp[nan_index] = np.nan
                e13c = np.vstack((e13c,temp))
                temp = np.copy(strain_matrix_plot[:,1,2])
                temp[nan_index] = np.nan
                e23c = np.vstack((e23c,temp))
                ##
                strain_matrixs_plot = np.copy(strain_matrixs[index][0])
                temp = np.copy(strain_matrixs_plot[:,0,0])
                temp[nan_index] = np.nan
                e11s = np.vstack((e11s,temp))
                temp = np.copy(strain_matrixs_plot[:,1,1])
                temp[nan_index] = np.nan
                e22s = np.vstack((e22s,temp))
                temp = np.copy(strain_matrixs_plot[:,2,2])
                temp[nan_index] = np.nan
                e33s = np.vstack((e33s,temp))
                temp = np.copy(strain_matrixs_plot[:,0,1])
                temp[nan_index] = np.nan
                e12s = np.vstack((e12s,temp))
                temp = np.copy(strain_matrixs_plot[:,0,2])
                temp[nan_index] = np.nan
                e13s = np.vstack((e13s,temp))
                temp = np.copy(strain_matrixs_plot[:,1,2])
                temp[nan_index] = np.nan
                e23s = np.vstack((e23s,temp))
        
        spots_len_plot = spots_len_plot.flatten()
        mr_plot = mr_plot.flatten()
        iR_pix_plot = iR_pix_plot.flatten()
        fR_pix_plot = fR_pix_plot.flatten() 
        e11c = e11c.flatten()
        e22c = e22c.flatten()
        e33c = e33c.flatten()
        e12c = e12c.flatten()
        e13c = e13c.flatten()
        e23c = e23c.flatten()
        e11s = e11s.flatten()
        e22s = e22s.flatten()
        e33s = e33s.flatten()
        e12s = e12s.flatten()
        e13s = e13s.flatten()
        e23s = e23s.flatten()
        
        try:
            title = "Number of spots and matching rate"
            fig = plt.figure()
            axs = fig.subplots(1, 2)
            axs[0].set_title("Number of spots", loc='center', fontsize=8)
            axs[0].hist(spots_len_plot, bins=bins)
            axs[0].set_ylabel('Frequency', fontsize=8)
            axs[0].tick_params(axis='both', which='major', labelsize=8)
            axs[0].tick_params(axis='both', which='minor', labelsize=8)
            axs[1].set_title("matching rate", loc='center', fontsize=8)
            axs[1].hist(mr_plot, bins=bins)
            axs[1].set_ylabel('Frequency', fontsize=8)
            axs[1].tick_params(axis='both', which='major', labelsize=8)
            axs[1].tick_params(axis='both', which='minor', labelsize=8)
            plt.tight_layout()
            plt.savefig(title+"_"+material_id[matid]+'.png', format='png', dpi=1000) 
            plt.close(fig)
        except:
            pass
        try:
            title = "Initial and Final residues"
            fig = plt.figure()
            axs = fig.subplots(1, 2)
            axs[0].set_title("Initial residues", loc='center', fontsize=8)
            axs[0].hist(iR_pix_plot, bins=bins)
            axs[0].set_ylabel('Frequency', fontsize=8)
            axs[0].tick_params(axis='both', which='major', labelsize=8)
            axs[0].tick_params(axis='both', which='minor', labelsize=8)
            axs[1].set_title("Final residues", loc='center', fontsize=8)
            axs[1].hist(fR_pix_plot, bins=bins)
            axs[1].set_ylabel('Frequency', fontsize=8)
            axs[1].tick_params(axis='both', which='major', labelsize=8)
            axs[1].tick_params(axis='both', which='minor', labelsize=8)
            plt.tight_layout()
            plt.savefig(title+"_"+material_id[matid]+'.png',format='png', dpi=1000) 
            plt.close(fig)
        except:
            pass
        try:
            title = "strain Crystal reference"
            fig = plt.figure()
            axs = fig.subplots(2, 3)
            axs[0, 0].set_title(r"$\epsilon_{11}$", loc='center', fontsize=8)
            axs[0, 0].hist(e11c, bins=bins)
            axs[0, 0].set_ylabel('Frequency', fontsize=8)
            axs[0, 0].tick_params(axis='both', which='major', labelsize=8)
            axs[0, 0].tick_params(axis='both', which='minor', labelsize=8)
            
            axs[0, 1].set_title(r"$\epsilon_{22}$", loc='center', fontsize=8)
            axs[0, 1].hist(e22c, bins=bins)
            axs[0, 1].set_ylabel('Frequency', fontsize=8)
            axs[0, 1].tick_params(axis='both', which='major', labelsize=8)
            axs[0, 1].tick_params(axis='both', which='minor', labelsize=8)
            
            axs[0, 2].set_title(r"$\epsilon_{33}$", loc='center', fontsize=8)
            axs[0, 2].hist(e33c, bins=bins)
            axs[0, 2].set_ylabel('Frequency', fontsize=8)
            axs[0, 2].tick_params(axis='both', which='major', labelsize=8)
            axs[0, 2].tick_params(axis='both', which='minor', labelsize=8)
            
            axs[1, 0].set_title(r"$\epsilon_{12}$", loc='center', fontsize=8)
            axs[1, 0].hist(e12c, bins=bins)
            axs[1, 0].set_ylabel('Frequency', fontsize=8)
            axs[1, 0].tick_params(axis='both', which='major', labelsize=8)
            axs[1, 0].tick_params(axis='both', which='minor', labelsize=8)
            
            axs[1, 1].set_title(r"$\epsilon_{13}$", loc='center', fontsize=8)
            axs[1, 1].hist(e13c, bins=bins)
            axs[1, 1].set_ylabel('Frequency', fontsize=8)
            axs[1, 1].tick_params(axis='both', which='major', labelsize=8)
            axs[1, 1].tick_params(axis='both', which='minor', labelsize=8)
            
            axs[1, 2].set_title(r"$\epsilon_{23}$", loc='center', fontsize=8)
            axs[1, 2].hist(e23c, bins=bins)
            axs[1, 2].set_ylabel('Frequency', fontsize=8)
            axs[1, 2].tick_params(axis='both', which='major', labelsize=8)
            axs[1, 2].tick_params(axis='both', which='minor', labelsize=8)
            plt.tight_layout()
            plt.savefig(title+"_"+material_id[matid]+'.png', format='png', dpi=1000) 
            plt.close(fig)
        except:
            pass
        try:
            title = "strain Sample reference"
            fig = plt.figure()
            axs = fig.subplots(2, 3)
            axs[0, 0].set_title(r"$\epsilon_{11}$", loc='center', fontsize=8)
            axs[0, 0].hist(e11s, bins=bins)
            axs[0, 0].set_ylabel('Frequency', fontsize=8)
            axs[0, 0].tick_params(axis='both', which='major', labelsize=8)
            axs[0, 0].tick_params(axis='both', which='minor', labelsize=8)
            
            axs[0, 1].set_title(r"$\epsilon_{22}$", loc='center', fontsize=8)
            axs[0, 1].hist(e22s, bins=bins)
            axs[0, 1].set_ylabel('Frequency', fontsize=8)
            axs[0, 1].tick_params(axis='both', which='major', labelsize=8)
            axs[0, 1].tick_params(axis='both', which='minor', labelsize=8)
            
            axs[0, 2].set_title(r"$\epsilon_{33}$", loc='center', fontsize=8)
            axs[0, 2].hist(e33s, bins=bins)
            axs[0, 2].set_ylabel('Frequency', fontsize=8)
            axs[0, 2].tick_params(axis='both', which='major', labelsize=8)
            axs[0, 2].tick_params(axis='both', which='minor', labelsize=8)
            
            axs[1, 0].set_title(r"$\epsilon_{12}$", loc='center', fontsize=8)
            axs[1, 0].hist(e12s, bins=bins)
            axs[1, 0].set_ylabel('Frequency', fontsize=8)
            axs[1, 0].tick_params(axis='both', which='major', labelsize=8)
            axs[1, 0].tick_params(axis='both', which='minor', labelsize=8)
            
            axs[1, 1].set_title(r"$\epsilon_{13}$", loc='center', fontsize=8)
            axs[1, 1].hist(e13s, bins=bins)
            axs[1, 1].set_ylabel('Frequency', fontsize=8)
            axs[1, 1].tick_params(axis='both', which='major', labelsize=8)
            axs[1, 1].tick_params(axis='both', which='minor', labelsize=8)
            
            axs[1, 2].set_title(r"$\epsilon_{23}$", loc='center', fontsize=8)
            axs[1, 2].hist(e23s, bins=bins)
            axs[1, 2].set_ylabel('Frequency', fontsize=8)
            axs[1, 2].tick_params(axis='both', which='major', labelsize=8)
            axs[1, 2].tick_params(axis='both', which='minor', labelsize=8)
            plt.tight_layout()
            plt.savefig(title+"_"+material_id[matid]+'.png', format='png', dpi=1000) 
            plt.close(fig)  
        except:
            pass
#%%    Plot some data    
from mpl_toolkits.axes_grid1 import make_axes_locatable

for index in range(rangeval):
    strain_matrix_plot = strain_matrix[index][0]
    
    # vmin, vmax= np.min(strain_matrix_plot), np.max(strain_matrix_plot)
        
    fig = plt.figure(figsize=(11.69,8.27), dpi=100)
    bottom, top = 0.1, 0.9
    left, right = 0.1, 0.8
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)

    axs = fig.subplots(2, 3)
    axs[0, 0].set_title(r"$\epsilon_{11}$", loc='center', fontsize=8)
    im=axs[0, 0].imshow(strain_matrix_plot[:,0,0].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet)
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    divider = make_axes_locatable(axs[0,0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    
    axs[0, 1].set_title(r"$\epsilon_{22}$", loc='center', fontsize=8)
    im=axs[0, 1].imshow(strain_matrix_plot[:,1,1].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet)
    divider = make_axes_locatable(axs[0,1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    
    axs[0, 2].set_title(r"$\epsilon_{33}$", loc='center', fontsize=8)
    im=axs[0, 2].imshow(strain_matrix_plot[:,2,2].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet)
    divider = make_axes_locatable(axs[0,2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    
    axs[1, 0].set_title(r"$\epsilon_{12}$", loc='center', fontsize=8)
    im=axs[1, 0].imshow(strain_matrix_plot[:,0,1].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet)
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    divider = make_axes_locatable(axs[1,0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    
    axs[1, 1].set_title(r"$\epsilon_{13}$", loc='center', fontsize=8)
    im=axs[1, 1].imshow(strain_matrix_plot[:,0,2].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet)
    axs[1, 1].set_xticks([])
    divider = make_axes_locatable(axs[1,1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    
    axs[1, 2].set_title(r"$\epsilon_{23}$", loc='center', fontsize=8)
    im = axs[1, 2].imshow(strain_matrix_plot[:,1,2].reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet)
    axs[1, 2].set_xticks([]) 
    divider = make_axes_locatable(axs[1,2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    for ax in axs.flat:
        ax.label_outer()
        
    plt.savefig('figure_strain_UB_'+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
    plt.close(fig)
#%%
from mpl_toolkits.axes_grid1 import make_axes_locatable

for index in range(rangeval):
    col_plot = col[index][0]
    col_plot = col_plot.reshape((lim_x, lim_y, 3))

    mr_plot = match_rate[index][0]
    mr_plot = mr_plot.reshape((lim_x, lim_y))
    
    mat_glob = mat_global[index][0]
    mat_glob = mat_glob.reshape((lim_x, lim_y))
    
    fig = plt.figure(figsize=(11.69,8.27), dpi=100)
    bottom, top = 0.1, 0.9
    left, right = 0.1, 0.8
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)

    axs = fig.subplots(1, 3)
    axs[0].set_title(r"IPF Z map", loc='center', fontsize=8)
    axs[0].imshow(col_plot[:-1,:], origin='lower')
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    
    axs[1].set_title(r"Material Index", loc='center', fontsize=8)
    im = axs[1].imshow(mat_glob[:-1,:], origin='lower', vmin=0, vmax=2)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    
    axs[2].set_title(r"Matching rate", loc='center', fontsize=8)
    im = axs[2].imshow(mr_plot[:-1,:], origin='lower', cmap=plt.cm.jet, vmin=0, vmax=100)
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    for ax in axs.flat:
        ax.label_outer()

    plt.savefig('figure_strain_global_'+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
    plt.close(fig)
    
for index in range(rangeval):
    spots_len_plot = spots_len[index][0]
    spots_len_plot = spots_len_plot.reshape((lim_x, lim_y))
    
    iR_pix_plot = iR_pix[index][0]
    iR_pix_plot = iR_pix_plot.reshape((lim_x, lim_y))
    
    fR_pix_plot = fR_pix[index][0]
    fR_pix_plot = fR_pix_plot.reshape((lim_x, lim_y))
    
    fig = plt.figure(figsize=(11.69,8.27), dpi=100)
    bottom, top = 0.1, 0.9
    left, right = 0.1, 0.8
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)

    axs = fig.subplots(1, 3)
    axs[0].set_title(r"Number of spots detected", loc='center', fontsize=8)
    im = axs[0].imshow(spots_len_plot, origin='lower', cmap=plt.cm.jet)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    axs[1].set_title(r"Initial pixel residues", loc='center', fontsize=8)
    im = axs[1].imshow(iR_pix_plot, origin='lower', cmap=plt.cm.jet)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    axs[2].set_title(r"Final pixel residues", loc='center', fontsize=8)
    im = axs[2].imshow(fR_pix_plot, origin='lower', cmap=plt.cm.jet)
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    for ax in axs.flat:
        ax.label_outer()

    plt.savefig('figure_strain_global_'+str(index)+'.png', bbox_inches='tight',format='png', dpi=1000) 
    plt.close(fig)

                      