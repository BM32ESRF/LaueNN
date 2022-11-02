# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 10:35:30 2021

@author: PURUSHOT

Mayavi version of Neural network visualization

A simple FFNN cubic Cu classifier with maxHKL upto 5

"""
import os
import numpy as np
from tqdm import tqdm
from net import GenData, read_hdf5, predict_DNN
mlab_exists = True
try:
    from mayavi import mlab
    mlab_exists = True
except:
    mlab_exists = False
    print("mayavi import error, not plotting anything")
    
## Generate Testing Laue pattern
material_model = GenData(key_material = 'Cu',
                        nbUBs=1,
                        sec_mat=False,
                        seed = 10,
                        key_material1 = 'Si',
                        nbUBs1 = 1,
                        seed1 = 15,
                        ang_maxx = 20,
                        step = 0.1)
img_array, codebars, l_tth, l_chi, l_miller_ind, l_intensity, tab_angulardist = material_model.get_data()
l_tth = l_tth-200 ##shift outside to the left side

## Load Keras trained model
model_direc = r"C:\Users\purushot\Desktop\LaueNN_script\python_animation\mayavi_version\LaueNN_model\Cu"

wb = read_hdf5(model_direc + "//model_Cu.h5")
classhkl = np.load(model_direc+"//MOD_grain_classhkl_angbin.npz")["arr_0"]
temp_key = list(wb.keys())
print("weights from Keras model are extracted")

## predict on the generated dataset
fc1, fc2, fc3, output = predict_DNN(codebars, wb, temp_key)
max_pred = np.max(output, axis = 1)
class_predicted = np.argmax(output, axis = 1)
predicted_hkl = classhkl[class_predicted]
predicted_hkl = predicted_hkl.astype(int)
np.savez('activity.npz', input=codebars, fc1=fc1, fc2=fc2, fc3=fc3, fc4=output, output=output,)

print("Input layer shape", codebars.shape)
print("First layer shape", fc1.shape)
print("Second layer shape", fc2.shape)
print("Third layer shape", fc3.shape)
print("Output layer shape", output.shape)

## Load activation function results from the predicted dataset
activity = np.load('activity.npz')

# Layer units
in_z, in_x = np.indices((codebars.shape[1], 1))
fc1_z, fc1_x = np.indices((fc1.shape[1], 1))
fc2_z, fc2_x = np.indices((fc2.shape[1], 1))
fc3_z, fc3_x = np.indices((fc3.shape[1], 1))
out_z, out_x = np.indices((output.shape[1], 1))

in_x = in_x.ravel()
in_z = in_z.ravel() - len(in_z.ravel())//2
in_y = np.zeros_like(in_x)

fc1_x = fc1_x.ravel()
fc1_z = fc1_z.ravel() - len(fc1_z.ravel())//2
fc1_y = np.ones_like(fc1_x) + 40

fc2_x = fc2_x.ravel()
fc2_z = fc2_z.ravel() - len(fc2_z.ravel())//2
fc2_y = np.ones_like(fc2_x) + 80

fc3_x = fc3_x.ravel()
fc3_z = fc3_z.ravel() - len(fc3_z.ravel())//2
fc3_y = np.ones_like(fc3_x) + 120

out_x = out_x.ravel()
out_z = out_z.ravel() - len(out_z.ravel())//2
out_z = out_z * 6
out_y = np.ones_like(out_x) + 160

# Connections between layers
fc1 = wb[temp_key[1]]
fc2 = wb[temp_key[3]]
fc3 = wb[temp_key[5]]
out = wb[temp_key[7]]

fr_in, to_fc1 = (np.abs(fc1) > 0.1).nonzero()
fr_fc1, to_fc2 = (np.abs(fc2) > 0.05).nonzero()
fr_fc2, to_fc3 = (np.abs(fc3) > 0.05).nonzero()
fr_fc3, to_out = (np.abs(out) > 0.1).nonzero()

fr_fc1 += len(in_x)
to_fc1 += len(in_x)
fr_fc2 += len(in_x) + len(fc1_x)
to_fc2 += len(in_x) + len(fc1_x)
fr_fc3 += len(in_x) + len(fc1_x) + len(fc2_x)
to_fc3 += len(in_x) + len(fc1_x) + len(fc2_x)
to_out += len(in_x) + len(fc1_x) + len(fc2_x) + len(fc3_x)

# Create the points
x = np.hstack((in_x, fc1_x, fc2_x, fc3_x, out_x))
y = np.hstack((in_y, fc1_y, fc2_y, fc3_y, out_y))
z = np.hstack((in_z, fc1_z, fc2_z, fc3_z, out_z))

save_directory = os.path.join(model_direc,"frames")
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
    
frame=0
for i in tqdm(list(range(len(codebars)))):
    # if i != 0:
    #     continue
    if mlab_exists:
        fig = mlab.figure(bgcolor=(13 / 255, 21 / 255, 44 / 255), size=(1920, 1080))
    # =============================================================================
    #     ###Input image draw and make connections
    #     # Input angle scattering map
    # =============================================================================
    spot_plot = i
    list_ = np.where(tab_angulardist[spot_plot,:] > 20)[0]
    ind_del = np.where(list_==spot_plot)[0]
    list_ = np.delete(list_, ind_del, axis=0)
    xstack = []
    ystack = []
    conn_stackANG = []
    conn_stackNN = []
    angbins = np.arange(0, 20+0.1, 0.1) 
    for ijk in range(len(l_tth)):
        if ijk in list_:
            continue
        if ijk == spot_plot:
            continue
        xstack.append(spot_plot)
        ystack.append(ijk)
        indd = np.digitize(tab_angulardist[spot_plot,ijk], angbins)
        if indd >= 201:
            continue
        else:
            conn_stackNN.append(indd-1)
            conn_stackANG.append(ijk)
    
    if mlab_exists:
        mlab.points3d(np.zeros_like(l_tth), l_tth, l_chi, color=(1,1,0),
                                        mode='cube', scale_factor=2.0, scale_mode='none')
        mlab.points3d(0, l_tth[spot_plot], l_chi[spot_plot], color=(1,0,0),
                                        mode='cube', scale_factor=2.0, scale_mode='none')
        mlab.points3d(np.zeros_like(l_tth[list_]), l_tth[list_], l_chi[list_], color=(0,0,1),
                                        mode='cube', scale_factor=2.0, scale_mode='none')
        ## add ground truth class label
        for j, ii in enumerate(zip(np.zeros_like(l_tth), l_tth, l_chi)):
            mlab.text3d(x=ii[0], y=ii[1]+1, z=ii[2], text=str(l_miller_ind[j]), scale=2)
        
        mlab.text3d(x=0, y=-110, z=60, text="Ground Truth "+str(l_miller_ind[i]), color=(1,0,0), scale=4)
        ##connect scatter points with lines
        src_img = mlab.pipeline.scalar_scatter(np.zeros_like(l_tth), l_tth, l_chi, l_intensity)
        src_img.mlab_source.dataset.lines = np.vstack((
                                                        np.array(xstack),
                                                        np.array(ystack),
                                                      )).T
        src_img.update()
        lines_img = mlab.pipeline.stripper(src_img)
        connections_img = mlab.pipeline.surface(lines_img, color=(1,0,0), line_width=0.5, opacity=0.5)
        
        x1 = np.hstack((np.zeros_like(l_tth), in_x))
        y1 = np.hstack((l_tth, in_y))
        z1 = np.hstack((l_chi, -in_z))
        ##connect scatter points with lines
        src_img1 = mlab.pipeline.scalar_scatter(x1, y1, z1)
        src_img1.mlab_source.dataset.lines = np.vstack((
                                                        np.array(conn_stackANG),
                                                        np.array(conn_stackNN)+len(l_tth),
                                                      )).T
        src_img1.update()
        lines_img1 = mlab.pipeline.stripper(src_img1)
        connections_img1 = mlab.pipeline.surface(lines_img1, color=(1,1,1), line_width=0.5, opacity=0.5)
    # =============================================================================
    #     ### NN draw and make connections
    # =============================================================================
    act_input = activity['input'][i]
    act_fc1 = activity['fc1'][i]
    act_fc2 = activity['fc2'][i]
    act_fc3 = activity['fc3'][i]
    act_out = activity['fc4'][i]
    s = np.hstack((
                    act_input.ravel() / act_input.max(),
                    act_fc1 / act_fc1.max(),
                    act_fc2 / act_fc2.max(),
                    act_fc3 / act_fc3.max(),
                    act_out / act_out.max(),
                  ))
    if mlab_exists:
        # Layer activation
        acts = mlab.points3d(x, y, -z, s, mode='cube', scale_factor=0.5, scale_mode='none', colormap='gray')
        # Connections
        src = mlab.pipeline.scalar_scatter(x, y, -z, s)
        src.mlab_source.dataset.lines = np.vstack((
                                                    np.hstack((fr_in, fr_fc1, fr_fc2, fr_fc3)),
                                                    np.hstack((to_fc1, to_fc2, to_fc3, to_out)),
                                                  )).T
        src.update()
        lines = mlab.pipeline.stripper(src)
        connections = mlab.pipeline.surface(lines, colormap='gray', line_width=0.1, opacity=0.1)
        ## add output class label
        counter = len(classhkl)-1
        pred_ = output[i,:]
        for j, ii in enumerate(zip(out_x, out_y, out_z)):
            if pred_[counter-j] > 0.75:
                mlab.text3d(x=ii[0], y=ii[1]+1, z=ii[2], text=str(classhkl[counter-j])+" Cu", scale =3, color=(1,0,0))
            else:
                mlab.text3d(x=ii[0], y=ii[1]+1, z=ii[2], text=str(classhkl[counter-j])+" Cu", scale =3)
        ## Write the prediction
        if max_pred[i] > 0.95:
            mlab.text3d(x=out_x[-1], y=out_y[-1]-15, z=np.max(out_z)+15, color=(1,0,0), 
                        text="Prediction :"+str(predicted_hkl[i])+" Cu", scale =4)
        else:
            mlab.text3d(x=out_x[-1], y=out_y[-1]-15, z=np.max(out_z)+15, color=(1,0,0), 
                        text="No prediction above 95%", scale =4)
            
            
        mlab.view(azimuth=0, elevation=90, distance=400, focalpoint=[0, 45, 0], reset_roll=False)
        mlab.savefig(save_directory + '\\frame_'+str(frame)+'.png')
        mlab.close(all=True)
    frame += 1
# mlab.show()