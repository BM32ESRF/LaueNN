# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 21:40:11 2022

@author: PURUSHOT

Template matching with open CV

COllection of useful opencv function for object detection and template matching
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to fill all the bounding box
def fill_rects(image, stats):
    for i,stat in enumerate(stats):
        if i > 0:
            p1 = (stat[0],stat[1])
            p2 = (stat[0] + stat[2],stat[1] + stat[3])
            cv2.rectangle(image,p1,p2,255,-1)

# Load image file
img1 = plt.imread(r'C:\Users\purushot\Desktop\Guillou_Laue\scan_0012\ech1_map2D_3_0000.tif')
img2 = plt.imread(r'C:\Users\purushot\Desktop\Guillou_Laue\scan_0012\ech1_map2D_3_0004.tif')

# Subtract the 2 image to get the difference region
img3 = cv2.subtract(img1,img2)

# Make it smaller to speed up everything and easier to cluster
small_img = img3 #◘cv2.resize(img3,(0,0),fx = 0.25, fy = 0.25)

# Morphological close process to cluster nearby objects
fat_img = cv2.dilate(small_img, None,iterations = 3)
fat_img = cv2.erode(fat_img, None,iterations = 3)

fat_img = cv2.dilate(fat_img, None,iterations = 3)
fat_img = cv2.erode(fat_img, None,iterations = 3)

# Threshold strong signals
_, bin_img = cv2.threshold(fat_img,20,255,cv2.THRESH_BINARY)

bin_img = bin_img.astype("uint8")
# Analyse connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)

# Cluster all the intersected bounding box together
rsmall, csmall = np.shape(small_img)
new_img1 = np.zeros((rsmall, csmall), dtype=np.uint8)

fill_rects(new_img1,stats)

# Analyse New connected components to get final regions
num_labels_new, labels_new, stats_new, centroids_new = cv2.connectedComponentsWithStats(new_img1)

labels_disp = np.uint8(200*labels/np.max(labels)) + 50
labels_disp2 = np.uint8(200*labels_new/np.max(labels_new)) + 50

cv2.imshow('new_img1',new_img1)
cv2.imshow('diff',img3)
# cv2.imshow('small_img',small_img)
# cv2.imshow('fat_img',fat_img)
cv2.imshow('bin_img',bin_img)
# cv2.imshow("labels",labels_disp)
# cv2.imshow("labels_disp2",labels_disp2)
cv2.waitKey(0)
#%%
import numpy as np
from scipy.ndimage import generate_binary_structure, binary_erosion, maximum_filter
import matplotlib.pyplot as plt

try:
    from lauetools import imageprocessing as ImProc
except:
    import lauetoolsnn.lauetools.imageprocessing as ImProc

def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)
    neighborhood = np.ones((5,5), dtype=bool)
    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.
    #we create the mask of the background
    background = (image==0)
    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background
    return detected_peaks

#applying the detection and plotting results
data_8bit_raw = plt.imread(r'C:\Users\purushot\Desktop\Guillou_Laue\scan_0012\ech1_map2D_3_0000.tif')
# framedim = data_8bit_raw.shape
# CCDLabel = "sCMOS"
# backgroundimage = ImProc.compute_autobackground_image(data_8bit_raw, boxsizefilter=10)
# data_8bit_raw_bg = ImProc.computefilteredimage(data_8bit_raw, backgroundimage, 
#                                             CCDLabel, usemask=True, formulaexpression="A-B")
# ## simple thresholding
# bg_threshold = 100
# data_8bit_raw[data_8bit_raw_bg<bg_threshold] = 0

detected_peaks = detect_peaks(data_8bit_raw)
plt.subplot(1,2,(1))
plt.imshow(data_8bit_raw)
plt.subplot(1,2,(2))
detected_peaks = detected_peaks.astype('int8')
detected_peaks[detected_peaks==1] = 255
plt.imshow(detected_peaks, cmap='gray')
plt.show()

#%%
import cv2
import cv2
import os
import glob

img_dir = "C:/Images"  # Enter Directory of all images 
data_path = os.path.join(img_dir, "*.tif") #Assume images are in tiff format
img_files = glob.glob(data_path)

# Set up tracker.
# Instead of MIL, you can also use

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[2]

# if int(minor_ver) < 3:
#     tracker = cv2.Tracker_create(tracker_type)
# else:
if tracker_type == 'BOOSTING':
    tracker = cv2.TrackerBoosting_create()
if tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
if tracker_type == 'TLD':
    tracker = cv2.TrackerTLD_create()
if tracker_type == 'MEDIANFLOW':
    tracker = cv2.TrackerMedianFlow_create()
if tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()
if tracker_type == 'MOSSE':
    tracker = cv2.TrackerMOSSE_create()
if tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()

# Read first image
frame = cv2.imread(img_files[0])
# Define an initial bounding box
bbox = (287, 23, 86, 320)
# Uncomment the line below to select a different bounding box
bbox = cv2.selectROI(frame, False)
# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

#while True:

# Iterate image files instead of reading from a video file
for f1 in img_files:
    frame = cv2.imread(f1)
    # Start timer
    timer = cv2.getTickCount()
    # Update tracker
    ok, bbox = tracker.update(frame)
    # Calculate Frames per second (FPS)
    #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    fps = 30 # We don't know the fps from the set of images
    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
    # Display result
    cv2.imshow("Tracking", frame)
    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff 
    if k == 27:
        break
