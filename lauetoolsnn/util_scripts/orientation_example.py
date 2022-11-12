# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 19:04:03 2022

@author: PURUSHOT

Script to test misorientation functions

"""
import numpy as np
import os, copy
import _pickle as cPickle
from tqdm import trange
from itertools import compress
from lauetoolsnn.lauetools.quaternions import Orientation, Quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import spatial

folder = os.getcwd()
with open(r"C:\Users\purushot\Desktop\SiC\results_triangle_5UBs\results.pickle", "rb") as input_file:
    best_match, \
    mat_global, rotation_matrix1, strain_matrix, strain_matrixs,\
        col, colx, coly, match_rate, files_treated,\
            lim_x, lim_y, spots_len, iR_pix, fR_pix, material_, \
                material1_, lattice, lattice1, symmetry0, symmetry1,\
                    crystal, crystal1 = cPickle.load(input_file)

material_id = [material_, material1_]
#%% convert UB to proper convention
new_rot_mat = [[np.zeros_like(rotation_matrix1[0][0])] for _ in range(len(rotation_matrix1))]

for index in range(len(rotation_matrix1)):
    for om_ind in trange(len(rotation_matrix1[index][0])):
        ## UB matrix in Laue reference frame (or crystal reference frame?)
        orientation_matrix = rotation_matrix1[index][0][om_ind]
        val = mat_global[index][0][om_ind]
        if val == 0 or np.all(orientation_matrix==0):
            continue
        if val == 1:
            symmetry = symmetry0.name
        elif val == 2:
            symmetry = symmetry1.name
        ## convert to orientation object to inherit all properties of Orientation class
        om = Orientation(matrix=orientation_matrix, symmetry=symmetry).reduced()
        new_rot_mat[index][0][om_ind] = om.asMatrix()

#%%



print("Number of Phases present (includes non indexed phase zero also)", len(np.unique(np.array(mat_global))))

average_UB = []
nb_pixels = []
mat_index = []
average_UB_object = []

# =============================================================================
# To categorize grains, some parameters
# =============================================================================
pixel_grain_definition = 30
radius = 10
misorientation = 5 #in degrees
cos_disorientation = np.cos(np.radians(misorientation/2.))      
make_position_pixel = np.indices((lim_x,lim_y))
make_position_pixel = make_position_pixel.reshape((2, lim_x*lim_y)).T
print('building KD tree...')
kdtree = spatial.KDTree(copy.deepcopy(make_position_pixel)) ##pixel position data?

for index in range(len(rotation_matrix1)):
    for val in np.unique(mat_global[index][0]):
        if val == 0:
            continue ##skipping no indexed patterns
        if val == 1:
            symmetry = symmetry0.name
        elif val == 2:
            symmetry = symmetry1.name
        mat_index1 = mat_global[index][0]
        mask_ = np.where(mat_index1 != val)[0]
        om_object = []
        print("step 1")
        for om_ind in trange(len(rotation_matrix1[index][0])):
            ## UB matrix in Laue reference frame (or crystal reference frame?)
            orientation_matrix = rotation_matrix1[index][0][om_ind]
            ##b convert to orientation object to inherit all properties of Orientation class
            om = Orientation(matrix=orientation_matrix, symmetry=symmetry).reduced()
            # reduced() return the FZ OM
            om_object.append(om)
        mr = match_rate[index][0]
        mr = mr.flatten()
        if len(mask_) > 0:
            mr[mask_] = 0
        if np.max(mr) == 0:
            continue
        
        print("step 2")
        # =============================================================================
        # ### Categorize into grains        
        # =============================================================================
        grainID = -np.ones(len(make_position_pixel),dtype=int)
        orientations = []  # quaternions found for grain
        memberCounts = []  # number of voxels in grain
        p = 0  # point counter
        g = 0  # grain counter
        matchedID = -1

        while p < len(make_position_pixel): # read next data point
            if p in mask_:
                p += 1       # increment point and continue
                continue
            if p > 0 and p % 100 == 0:
                print('Processing point %i of %i (grain count %i)...' % (p,len(grainID),np.count_nonzero(memberCounts)))
            o = om_object[p]
            matched        = False
            alreadyChecked = {}
            candidates     = []
            bestDisorientation = Quaternion([0,0,0,1])  # initialize to 180 deg rotation as worst case
            for i in kdtree.query_ball_point(kdtree.data[p], radius):  # check all neighboring points
                gID = grainID[i]
                if (gID != -1) and (gID not in alreadyChecked): # indexed point belonging to a grain not yet tested?
                    alreadyChecked[gID] = True    # remember not to check again
                    disorientation = o.disorientation(orientations[gID], SST = False)[0]# compare against other orientation
                    if disorientation.quaternion.w >  cos_disorientation:  # within threshold ...
                        candidates.append(gID)   # remember as potential candidate
                        if disorientation.quaternion.w >= bestDisorientation.w: # ... and better than current best? 
                            matched = True
                            matchedID = gID    # remember that grain
                            bestDisorientation = disorientation.quaternion
            if matched:     # did match existing grain
                memberCounts[matchedID] += 1
                if len(candidates) > 1:     # ambiguity in grain identification?
                    largestGrain = sorted(candidates,key=lambda x:memberCounts[x])[-1]  
                    # find largest among potential candidate grains
                    matchedID    = largestGrain
                    for c in [c for c in candidates if c != largestGrain]:   # loop over smaller candidates
                        memberCounts[largestGrain] += memberCounts[c]   # reassign member count of smaller to largest
                        memberCounts[c] = 0
                    grainID = np.where(np.in1d(grainID,candidates), largestGrain, grainID)   
                    # relabel grid points of smaller candidates as largest one
            else:       # no match -> new grain found
                orientations += [o]     # initialize with current orientation
                memberCounts += [1]      # start new membership counter
                matchedID = g
                g += 1               # increment grain counter
            grainID[p] = matchedID    # remember grain index assigned to point
            p += 1       # increment point
            
        print("step 3")
        grain_map = grainID.reshape((lim_x,lim_y))
        
        print('step 4')
        # =============================================================================
        # Let us compute the average orientation of grain definition
        # =============================================================================
        ####Average UB per grains
        grainIDs = np.where(np.array(memberCounts) >= pixel_grain_definition)[0]   # identify "live" grain identifiers
        for gi_ in grainIDs:
            pixel_indices = np.where(grainID==gi_)[0]
            om_object_mod = []
            for hi_ in pixel_indices:
                om_object_mod.append(om_object[hi_])
            avg_om_object = Orientation.average(om_object_mod)
            average_UB_object.append(avg_om_object)
            average_UB.append(avg_om_object.asMatrix())
            nb_pixels.append(len(pixel_indices))
            mat_index.append(val)        

        fig = plt.figure(figsize=(11.69,8.27), dpi=100)
        bottom, top = 0.1, 0.9
        left, right = 0.1, 0.8
        fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
        axs = fig.subplots(1, 1)
        axs.set_title(r"Grain map", loc='center', fontsize=8)
        im=axs.imshow(grain_map, origin='lower', cmap=plt.cm.jet)
        axs.set_xticks([])
        axs.set_yticks([])
        divider = make_axes_locatable(axs)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.ax.tick_params(labelsize=8) 
        axs.label_outer()        
        plt.show()
        # plt.savefig(folder+ "//"+'figure_misorientation_'+str(matid)+"_"+str(index)+'.png', 
        #             bbox_inches='tight',format='png', dpi=1000) 
        # plt.close(fig)
        

average_UB = np.array(average_UB)
nb_pixels = np.array(nb_pixels)
mat_index = np.array(mat_index)

s_ix = np.argsort(nb_pixels)[::-1]
average_UB = average_UB[s_ix]
nb_pixels = nb_pixels[s_ix]
mat_index = mat_index[s_ix]
#############################
save_directory_ = r"C:\Users\purushot\Desktop\SiC\results_triangle_5UBs"
## save a average_rot_mat.txt file
text_file = open(os.path.join(save_directory_,"average_rot_mat.txt"), "w")
text_file.write("# ********** Average UB matrix from Misorientation computation *************\n")

for imat in range(len(average_UB)):
    local_ub = average_UB[imat,:,:].flatten()
    string_ = ",".join(map(str, local_ub))
    string_ = string_ + ","+str(mat_index[imat])
    text_file.write("# ********** UB MATRIX "+str(imat+1)+" ********** \n")
    text_file.write("# Nb of pixel occupied "+str(nb_pixels[imat])+"/"+str(lim_x*lim_y)+" ********** \n")
    text_file.write(string_ + " \n")
text_file.close() 
        
#%% Old part

# ref_index = np.where(mr == np.max(mr))[0][0] ##choose from best matching rate
# ##Quats approach
# reference_orientation = om_object[ref_index]
# misori_object = []
# misori_ang = []
# for ii in trange(len(om_object)):
#     ##Make different material object as zero to avoid symmetry problem during misorientation
#     moveon = False
#     if len(mask_)>0:
#         if ii in mask_:
#             misori_object.append(-1)
#             misori_ang.append(-1)
#             moveon = True
#     if not moveon:
#         int_object = reference_orientation.disorientation(om_object[ii], SST=True)
#         misori_object.append(int_object)
#         misori_ang.append(np.rad2deg(int_object[0].angleAxis[0]))
# misori = np.array(misori_ang)
# print()
# print("maximum misorientation angle (degrees)", np.max(misori))

# max_ang = np.max(misori) + 5
# grain_tol = 3
# bins = np.arange(0, max_ang, grain_tol) ##every 3deg
# zz, binzz = np.histogram(misori, bins=bins) #3°bin
# bin_index = np.where(zz> 0.1*np.max(zz))[0]
# bin_angles = binzz[bin_index]
# grains = np.copy(misori)
# grains_segmentation = np.zeros_like(misori)
# for kk, jj in enumerate(bin_angles):
#     if jj == 0:
#         cond = (grains<=jj+grain_tol/2.) * (grains>=0)
#     else:
#         cond = (grains<=jj+grain_tol/2.) * (grains>=jj-grain_tol/2.)            
#     ### Apply more filters to select good data only here
#     if np.all(cond == False):
#         continue
#     om_object_mod = list(compress(om_object, cond))
#     avg_om_object = Orientation.average(om_object_mod)
#     average_UB_object.append(avg_om_object)
#     average_UB.append(avg_om_object.asMatrix())
#     nb_pixels.append(len(cond[cond]))
#     mat_index.append(val)
#     ## mask the grain pixels
#     for ll in range(len(cond)):
#         if cond[ll]:
#             if grains_segmentation[ll] == 0:
#                 grains_segmentation[ll] = kk+1
# grain_map = grains_segmentation.reshape((lim_x,lim_y))

# ####Average UBs misorientation with each other
# average_UB = np.array(average_UB)
# nb_pixels = np.array(nb_pixels)
# mat_index = np.array(mat_index)

# s_ix = np.argsort(nb_pixels)[::-1]
# average_UB = average_UB[s_ix]
# nb_pixels = nb_pixels[s_ix]
# #############################
# save_directory_ = r"D:\some_projects\GaN\Si_GaN_nanowires\results_Si_2022-07-08_18-22-15"
# ## save a average_rot_mat.txt file
# text_file = open(os.path.join(save_directory_,"average_rot_mat.txt"), "w")
# text_file.write("# ********** Average UB matrix from Misorientation computation *************\n")

# for imat in range(len(average_UB)):
#     local_ub = average_UB[imat,:,:].flatten()
#     string_ = ",".join(map(str, local_ub))
#     string_ = string_ + ","+str(mat_index[imat])
#     text_file.write("# ********** UB MATRIX "+str(imat+1)+" ********** \n")
#     text_file.write("# Nb of pixel occupied "+str(nb_pixels[imat])+"/"+str(lim_x*lim_y)+" ********** \n")
#     text_file.write(string_ + " \n")
# text_file.close()

        
        
        
        
        
        
        
        
        
        