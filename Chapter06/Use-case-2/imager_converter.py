#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code will convert the dataset's pixel values to Images
@author: raz
"""

import os
import csv
import argparse
import numpy as np 
import scipy.misc

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', required=True, help="path of the csv file")
parser.add_argument('-o', '--output', required=True, help="path of the output directory")
args = parser.parse_args()

w, h = 48, 48
image = np.zeros((h, w), dtype=np.uint8)
id = 1

with open(args.file) as csvfile:
    datareader = csv.reader(csvfile, delimiter =',')
    next(datareader,None)
    
    for row in datareader:
        
        emotion = row[0]
        pixels = row[1].split()
        usage = row[2]
        pixels_array = np.asarray(pixels, dtype=np.int)

        image = pixels_array.reshape(w, h)
        stacked_image = np.dstack((image,) * 3)

        image_folder = os.path.join(args.output, usage)
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        image_file =  os.path.join(image_folder , emotion +'_'+ str(id) +'.jpg')
        scipy.misc.imsave(image_file, stacked_image)
        id += 1 
        if id % 100 == 0:
            print('Processed {} images'.format(id))

print("Finished conversion to {} images".format(id))
