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

from PIL import Image

def convert_png_to_jpeg(png_filename, jpeg_filename):
    """
    Converts a PNG file to a JPEG file.

    Args:
        png_filename (str): The path to the input PNG file.
        jpeg_filename (str): The path for the output JPEG file.
    """
    try:
        # Open the image file
        with Image.open(png_filename) as im:
            # Convert the image to RGB mode if it has an alpha channel (transparency)
            # JPEG does not support transparency, so a white background is typically used
            if im.mode in ('RGBA', 'P'):
                im = im.convert('RGB')
            
            # Save the image in JPEG format
            im.save(jpeg_filename, 'JPEG')
            print(f"Successfully converted {png_filename} to {jpeg_filename}")

    except FileNotFoundError:
        print(f"Error: The file '{png_filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Define figure dir location
img_dir = '../../01_Data/SCBA_Bracket/'

#Define new image size
width = 2000
height = 1333
dim = (width,height)

for dirpath, dirs, files in os.walk(img_dir):    
    path = dirpath.split('/')
    for f in files:
        # Skip if file doesn't end with .JPG
        if f.lower().endswith('.png'):
            print('*** CONVERT ***')
            output_fid = f.replace('.png','.jpg')
            convert_png_to_jpeg(f'{img_dir}/{f}', f'{img_dir}/{output_fid}')
            f = output_fid
        if not f.lower().endswith('.jpg'):
            continue
        if '600x400' in f:
            continue

        # # skip file if size already less than 1 MB
        # if os.path.getsize(f'{dirpath}{f}') < 1000000:
        #     continue

        print(f)

        img = cv2.imread(dirpath + '/' + f.lower())
        
        print('Original Dimensions : ',img.shape)

        img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        cv2.imwrite(dirpath + '/' + f, img_resized)

        # create smaller version of the primary picture for thumbnail on front-end
        if f.split('.')[0] == path[-2]:

            img_resized = cv2.resize(img, (600,400), interpolation=cv2.INTER_AREA)

            cv2.imwrite(dirpath + '/' + f[:-4]+'600x400.jpg', img_resized)

            print('Resized ' + f)
        
        print()