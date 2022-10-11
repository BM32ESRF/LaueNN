# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 20:09:37 2021

@author: PURUSHOT

Post process the results on Cu Si pads
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\lauetools")
import CrystalParameters as CP
import dict_LaueTools as dictLT

### Please enter the path of the results.npz file (which is extracted from the LauetoolsNN results directory)
result_directory = r"C:\Users\purushot\Desktop\Al2TiO5_laue_Dec2021_June2022\Dec2021\results_Daniel\VF_sample\VF_M1_R3_1"
save_direc = result_directory + "\\" + "additional_plots"

if not os.path.exists(save_direc):
    os.makedirs(save_direc)
    
results_obj = np.load(result_directory + "//results.npz", allow_pickle=True)
print("Results loaded successfully")

best_match = results_obj["arr_0"] # contains all the details regarding the indexation
mat_global = results_obj["arr_1"] # material id of each pixel (to identify if it is Si or Cu)
rotation_matrix1 = results_obj["arr_2"] # 3x3 rotation matrix for each pixel (in Lauetools frame)
strain_matrix = results_obj["arr_3"] # 3x3 strain matrix of each UB matrix for each pixel (crystal reference frame)
strain_matrixs = results_obj["arr_4"] # 3x3 strain matrix of each UB matrix for each pixel (sample reference frame)
col = results_obj["arr_5"] # Color in sample Z direction confroming the IPF color (be careful)
colx = results_obj["arr_6"] # Color in sample X direction confroming the IPF color (be careful)
coly = results_obj["arr_7"] # Color in sample Y direction confroming the IPF color (be careful)
match_rate = results_obj["arr_8"] # Matching rate of each indexation
files_treated = results_obj["arr_9"] # File names of each pixel
lim_x = results_obj["arr_10"] # Limit X of raster scan
lim_y = results_obj["arr_11"] # Limit Y of raster scan
spots_len = results_obj["arr_12"] # Number of spots per pattern indexed
iR_pix = results_obj["arr_13"] # initial pixel residues before strain refinement
fR_pix = results_obj["arr_14"] # final residue pixel after strain refinement
ub_matricies = len(rotation_matrix1) # Total UB matricies
material_ = str(results_obj["arr_15"]) # Material name for mat id 1
material1_ = str(results_obj["arr_16"]) # Material name for mat id 2
material_id = [material_, material1_]

match_tol = 50
pixel_residues = 3
# NbofSpots = 20
constantlength = "a" ## For strain analysis
mat = 0
bins = 30

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

#%% Plot histograms
for index in range(ub_matricies):
    nan_index0 = np.where(match_rate[index][0] < match_tol)[0]
    nan_index1 = np.where(fR_pix[index][0].flatten() > pixel_residues)[0]
    # nan_index2 = np.where(spots_len[index][0] < NbofSpots)[0]
    nan_index = np.hstack((nan_index0,nan_index1))
    nan_index = np.unique(nan_index)
    
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
        plt.savefig(save_direc+"\\"+title+"_"+str(index)+'.png', format='png', dpi=1000) 
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
        plt.savefig(save_direc+"\\"+title+"_"+str(index)+'.png', format='png', dpi=1000) 
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
        plt.savefig(save_direc+"\\"+title+"_"+str(index)+'.png', format='png', dpi=1000) 
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
        plt.savefig(save_direc+"\\"+title+"_"+str(index)+'.png', format='png', dpi=1000) 
        plt.close(fig)
    except:
        pass
    
    try:
        a,b,c,alp,bet,gam = [],[],[],[],[],[]
        for irot in range(len(rotation_matrix1[index][0])):
            if (match_rate[index][0][irot] < match_tol) or \
                fR_pix[index][0][irot] > pixel_residues:
                continue
            
            lattice_parameter_direct_strain = CP.computeLatticeParameters_from_UB(rotation_matrix1[index][0][irot,:,:], 
                                                                                  material_id[mat], 
                                                                                  constantlength, 
                                                                                  dictmaterials=dictLT.dict_Materials)
            a.append(lattice_parameter_direct_strain[0])
            b.append(lattice_parameter_direct_strain[1])
            c.append(lattice_parameter_direct_strain[2])
            alp.append(lattice_parameter_direct_strain[3])
            bet.append(lattice_parameter_direct_strain[4])
            gam.append(lattice_parameter_direct_strain[5])
        
        title = "Refined unit cell"+" "+material_id[mat]+ " "+str(index)
        fig = plt.figure()
        axs = fig.subplots(2, 3)
        axs[0, 0].set_title(r"a", loc='center', fontsize=8)
        logdata = np.array(a)
        logdata = logdata[~np.isnan(logdata)]
        rangemin, rangemax = np.min(logdata)-0.01, np.max(logdata)+0.01
        axs[0, 0].hist(logdata, bins=bins, density=True, alpha=0.8, range=(rangemin, rangemax))
        axs[0, 0].set_ylabel('Frequency', fontsize=8)
        axs[0, 0].tick_params(axis='both', which='major', labelsize=8)
        axs[0, 0].tick_params(axis='both', which='minor', labelsize=8)
    
        axs[0, 1].set_title(r"b", loc='center', fontsize=8)
        logdata = np.array(b)
        logdata = logdata[~np.isnan(logdata)]
        rangemin, rangemax = np.min(logdata)-0.01, np.max(logdata)+0.01
        axs[0, 1].hist(logdata, bins=bins, density=True, alpha=0.8, range=(rangemin, rangemax))
        axs[0, 1].set_ylabel('Frequency', fontsize=8)
        axs[0, 1].tick_params(axis='both', which='major', labelsize=8)
        axs[0, 1].tick_params(axis='both', which='minor', labelsize=8)
        
        axs[0, 2].set_title(r"c", loc='center', fontsize=8)
        logdata = np.array(c)
        logdata = logdata[~np.isnan(logdata)]
        rangemin, rangemax = np.min(logdata)-0.01, np.max(logdata)+0.01
        axs[0, 2].hist(logdata, bins=bins, density=True, alpha=0.8, range=(rangemin, rangemax))
        axs[0, 2].set_ylabel('Frequency', fontsize=8)
        axs[0, 2].tick_params(axis='both', which='major', labelsize=8)
        axs[0, 2].tick_params(axis='both', which='minor', labelsize=8)
        
        axs[1, 0].set_title(r"$\alpha$", loc='center', fontsize=8)
        logdata = np.array(alp)
        logdata = logdata[~np.isnan(logdata)]
        rangemin, rangemax = np.min(logdata)-0.01, np.max(logdata)+0.01
        axs[1, 0].hist(logdata, bins=bins, density=True, alpha=0.8, range=(rangemin, rangemax))
        axs[1, 0].set_ylabel('Frequency', fontsize=8)
        axs[1, 0].tick_params(axis='both', which='major', labelsize=8)
        axs[1, 0].tick_params(axis='both', which='minor', labelsize=8)
        
        axs[1, 1].set_title(r"$\beta$", loc='center', fontsize=8)
        logdata = np.array(bet)
        logdata = logdata[~np.isnan(logdata)]
        rangemin, rangemax = np.min(logdata)-0.01, np.max(logdata)+0.01
        axs[1, 1].hist(logdata, bins=bins, density=True, alpha=0.8, range=(rangemin, rangemax))
        axs[1, 1].set_ylabel('Frequency', fontsize=8)
        axs[1, 1].tick_params(axis='both', which='major', labelsize=8)
        axs[1, 1].tick_params(axis='both', which='minor', labelsize=8)
        
        axs[1, 2].set_title(r"$\gamma$", loc='center', fontsize=8)
        logdata = np.array(gam)
        logdata = logdata[~np.isnan(logdata)]
        rangemin, rangemax = np.min(logdata)-0.01, np.max(logdata)+0.01
        axs[1, 2].hist(logdata, bins=bins, density=True, alpha=0.8, range=(rangemin, rangemax))
        axs[1, 2].set_ylabel('Frequency', fontsize=8)
        axs[1, 2].tick_params(axis='both', which='major', labelsize=8)
        axs[1, 2].tick_params(axis='both', which='minor', labelsize=8)
    
        plt.tight_layout()
        plt.savefig(save_direc+"\\"+title+'.png', format='png', dpi=1000) 
        plt.close(fig)
    except:
        continue
    
    
count = 0
for index in range(ub_matricies):
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
    title = "Number of spots and matching rate-ALL UB"
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
    plt.savefig(save_direc+"\\"+title+'.png', format='png', dpi=1000) 
    plt.close(fig)
except:
    pass
try:
    title = "Initial and Final residues-ALL UB"
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
    plt.savefig(save_direc+"\\"+title+'.png',format='png', dpi=1000) 
    plt.close(fig)
except:
    pass
try:
    title = "strain Crystal reference-ALL UB"
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
    plt.savefig(save_direc+"\\"+title+'.png', format='png', dpi=1000) 
    plt.close(fig)
except:
    pass
try:
    title = "strain Sample reference-ALL UB"
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
    plt.savefig(save_direc+"\\"+title+'.png', format='png', dpi=1000) 
    plt.close(fig)  
except:
    pass
    