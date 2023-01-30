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
img_dir = '../../01_Data/Polyester_Bed_Skirt/'

#Define new image size
width = 2000
height = 1333
dim = (width,height)

for dirpath, dirs, files in os.walk(img_dir):    
    path = dirpath.split('/')
    for f in files:
        # Skip if file doesn't end with .JPG
        if not f.lower().endswith('.jpg'):
            continue
        if '600x400' in f:
            continue
        
        # skip file if size already less than 1 MB
        if os.path.getsize(f'{dirpath}{f}') < 1000000:
            continue

        print(f)

        img = cv2.imread(dirpath + '/' + f.lower())
        
        print('Original Dimensions : ',img.shape)

        img_resized = cv2.resize(img, (2000,1333), interpolation=cv2.INTER_AREA)

        cv2.imwrite(dirpath + '/' + f, img_resized)

        # create smaller version of the primary picture for thumbnail on front-end
        if f.split('.')[0] == path[-2]:

            img_resized = cv2.resize(img, (600,400), interpolation=cv2.INTER_AREA)

            cv2.imwrite(dirpath + '/' + f[:-4]+'600x400.jpg', img_resized)

            print('Resized ' + f)
        
        print()