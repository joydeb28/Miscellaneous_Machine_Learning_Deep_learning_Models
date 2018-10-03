#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 10:04:29 2018

@author: joy
"""

import cv2
import numpy as np
from PIL import Image
from pytesseract import image_to_string
import glob
# Path of working folder on Disk
src_path = "The_Alchemist_Dataset/"#if got error give full path

def get_images(path):
    
    file_name_list = []
    
    for file_name in glob.glob(path+'*.jpg'):
        file_name_list.append(file_name)
        
    return file_name_list

def get_string(img_path):
    # Read image with opencv
    img = cv2.imread(img_path)

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    # Write image after removed noise
    cv2.imwrite(img_path + "removed_noise.png", img)

    #  Apply threshold to get image with only black and white
    #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # Write the image after apply opencv to do some ...
    cv2.imwrite(img_path + "thres.png", img)

    # Recognize text with tesseract for python
    result = image_to_string(Image.open(img_path+"thres.png"))

    # Remove template file
    #os.remove(temp)

    return Image.open(img_path),result


def run(file_name):
    images_list = get_images(file_name)
    
    for i in images_list:
        imge,res = get_string(i)
        print(imge,res)

run(src_path)
