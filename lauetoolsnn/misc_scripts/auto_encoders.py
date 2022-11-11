# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 23:04:13 2022

@author: PURUSHOT

Auto encoders attempts with Laue images

Can we extract single grain related peaks by removing all the other peaks

Sort of like denoising 

Idea is to train an encoder with simulated pattern images
and from experiments extract only these peaks from an image 
of lot of superimposed Laue patterns

Nees more training and more variability in data

Come back to it later!

"""
if __name__ == "__main__":
    
    
    import json
    
    ## Load the json of material and extinctions
    with open(r'C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\lauetools\material.json','r') as f:
        dict_Materials = json.load(f)
    
    with open(r'C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\lauetools\extinction.json','r') as f:
        extinction_json = json.load(f)
        
    ## Modify the dictionary values to add new entries
    dict_Materials["ZrO2_tet_1250C"] = ["ZrO2_tet_1250C", [3.642395418, 3.642395418, 5.28113, 90, 90, 90], "VO2_mono2tet"]

    extinction_json["VO2_mono2tet"] = "VO2_mono2tet"
    
    ## dump the json back with new values
    with open(r'C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\lauetools\material.json', 'w') as fp:
        json.dump(dict_Materials, fp)
    
    with open(r'C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\lauetools\extinction.json', 'w') as fp:
        json.dump(extinction_json, fp)
        
    import numpy as np
    import matplotlib.pyplot as plt
    from random import random as rand1
    from math import acos
    from tqdm import trange
    try:
        import lauetoolsnn.lauetools.lauecore as LT
        import lauetoolsnn.lauetools.CrystalParameters as CP
        import lauetoolsnn.lauetools.imageprocessing as ImProc
    except:
        from lauetools import lauecore as LT
        from lauetools import CrystalParameters as CP
        from lauetools import imageprocessing as ImProc
    
    import cv2
    
    DEG = np.pi / 180.0
    
    # =============================================================================
    # Plot the binning angular distribution
    # =============================================================================
    def Euler2OrientationMatrix(euler):
        """Compute the orientation matrix :math:`\mathbf{g}` associated with
        the 3 Euler angles :math:`(\phi_1, \Phi, \phi_2)`.
        :param euler: The triplet of the Euler angles (in degrees).
        :return g: The 3x3 orientation matrix.
        """
        (rphi1, rPhi, rphi2) = np.radians(euler)
        c1 = np.cos(rphi1)
        s1 = np.sin(rphi1)
        c = np.cos(rPhi)
        s = np.sin(rPhi)
        c2 = np.cos(rphi2)
        s2 = np.sin(rphi2)
        # rotation matrix g
        g11 = c1 * c2 - s1 * s2 * c
        g12 = s1 * c2 + c1 * s2 * c
        g13 = s2 * s
        g21 = -c1 * s2 - s1 * c2 * c
        g22 = -s1 * s2 + c1 * c2 * c
        g23 = c2 * s
        g31 = s1 * s
        g32 = -c1 * s
        g33 = c
        g = np.array([[g11, g12, g13], [g21, g22, g23], [g31, g32, g33]])
        return g

    key_material = 'ZrO2_tet_1250C'
    r = 1  # pixel radius for filling
    detectorparameters = [79.57100, 1950.4400, 1858.9900, 0.3280000, 0.4500000]
    pixelsize =  0.03670000
    nbUBs = 1
    
    img_x, img_y = 4036, 4032
    bin_img_x, bin_img_y = 12, 12
    dataset_size = 5000
    
    training_data = np.zeros((dataset_size, img_x//bin_img_x, img_y//bin_img_y), dtype=np.uint8)
    for jj in trange(dataset_size):
        UBelemagnles = np.random.random((nbUBs,3))*360-180.
        phi1 = rand1() * 360.
        phi = 180. * acos(2 * rand1() - 1) / np.pi
        phi2 = rand1() * 360.
        UBmatrix = Euler2OrientationMatrix((phi1, phi, phi2))
        
        ###########################################################    
        grain = CP.Prepare_Grain(key_material, UBmatrix)
        s_tth, s_chi, s_miller_ind, s_posx, s_posy, s_E= LT.SimulateLaue_full_np(grain, 5, 22,
                                                                                  detectorparameters,
                                                                                  detectordiameter = pixelsize*img_x,
                                                                                  pixelsize=pixelsize,
                                                                                  removeharmonics=1)
        img_arr = np.zeros((img_x//bin_img_x, img_y//bin_img_y), dtype=np.uint8)
        for i in range(len(s_posx)):
            X_pix = int(s_posx[i]/bin_img_x)
            Y_pix = int(s_posy[i]/bin_img_y)
            # img_arr[X_pix-r:X_pix+r, Y_pix-r:Y_pix+r] = 255 #int(1./s_E[i])
            img_arr[X_pix, Y_pix-r] = 255
    
        # img_arr = cv2.resize(img_arr, (0,0), fx = 1/bin_img_x, fy = 1/bin_img_y)
        training_data[jj, :, :] = img_arr
    ###########################################################
    # fig = plt.figure()
    # plt.imshow(training_data[0,:,:])
    # plt.show()
    
    
    #%% NN library and code
    ## Inspired from https://blog.keras.io/building-autoencoders-in-keras.html
    if 1:
        import keras
        from keras import layers
        
        input_img = keras.Input(shape=(img_x//bin_img_x, img_y//bin_img_y, 1))
        
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        
        decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        
        autoencoder = keras.Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        autoencoder.summary()
        
        len_mi = np.array([iq for iq in range(len(training_data))])
        indices_testing = np.random.choice(len_mi, int(len(training_data)*0.10), replace=False)
        testing_data = np.copy(training_data[indices_testing, :, :])
        if len(indices_testing) !=0:
            training_data = np.delete(training_data, indices_testing, axis=0)
    
        training_data = training_data.astype('float32')/255
        testing_data = testing_data.astype('float32')/255        
        training_data = np.reshape(training_data, (len(training_data), img_x//bin_img_x, img_y//bin_img_y, 1))
        testing_data = np.reshape(testing_data, (len(testing_data), img_x//bin_img_x, img_y//bin_img_y, 1))

        noise_factor = 0.03
        x_train_noisy = training_data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=training_data.shape) 
        x_test_noisy = testing_data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=testing_data.shape) 
        x_train_noisy = np.clip(x_train_noisy, 0., 1.)
        x_test_noisy = np.clip(x_test_noisy, 0., 1.)
        
        autoencoder.fit(x_train_noisy, training_data,
                        epochs=100,
                        batch_size=30,
                        shuffle=True,
                        validation_data=(x_test_noisy, testing_data))

    if 0:
        ## lets see if we can clear off Be peaks from HT Laue experiments
        
        ## Load experimental image
        image = plt.imread(r"C:\Users\purushot\Desktop\Laue_Zr_HT\1250Cfriday\Zr3_1250_0000.tif")
        
        backgroundimage = ImProc.compute_autobackground_image(image, boxsizefilter=10)
        # basic substraction
        data_8bit_rawtiff = ImProc.computefilteredimage(image, 
                                                        backgroundimage, 
                                                        "sCMOS_16M", 
                                                        usemask=True, 
                                                        formulaexpression="A-B")
        
        bg_threshold = 50
        data_8bit_rawtiff[data_8bit_rawtiff < bg_threshold] = 0
        data_8bit_rawtiff[data_8bit_rawtiff > 0] = 255
        data_8bit_rawtiff = data_8bit_rawtiff.astype(np.uint8)
        
        img_arr = cv2.resize(data_8bit_rawtiff, (0,0), fx = 1/bin_img_x, fy = 1/bin_img_y)

        img_arr = img_arr.astype('float32')/255
        plt.figure(figsize=(20, 4))
        plt.imshow(img_arr)
        plt.show()
        plt.close()
        
        decoded_imgs = autoencoder.predict(img_arr.reshape((1,168*2,168*2,1)))
        plt.figure(figsize=(20, 4))
        plt.imshow(decoded_imgs[0,:,:,0])
        plt.show()
        plt.close()




