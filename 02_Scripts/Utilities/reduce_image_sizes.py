# reduce_image_sizes.py
#   by: C Weinschenk
# ***************************** Run Notes ***************************** #
# - Looks through all of 01_Data materials and creates 600x400 image    #
#       for the front end website                                       #
# ********************************************************************* #

# Import Packages
import os
import cv2
import numpy as np

# Define figure dir location
img_dir = '../../01_Data/'

#Define new image size
width = 600
height = 400
dim = (width,height)

for dirpath, dirs, files in os.walk(img_dir):    
    path = dirpath.split('/')
    for f in files:
        # Skip if file doesn't end with .JPG
        if not f.lower().endswith('.jpg'):
            continue
        img = cv2.imread(dirpath + '/' + f.lower())
        print('Original Dimensions : ',img.shape)

        img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        cv2.imwrite(dirpath + '/' + f[:-4]+'600x400.jpg', img_resized)
        print('Resized ' + f)