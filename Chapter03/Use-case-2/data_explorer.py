

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 21:55:34 2019

@author: raz
"""

# Five classes (manhole, pavement, pothole, roadmarkings, shadow)
# Import the necessary modules

import cv2
import glob
import numpy as np
import os
import random
import matplotlib.pyplot as plt

# List to store the images of each class

images =[]

# Function to read image file one by one and append to the images list

def data_explorer(image_dir, images_to_explore):  
        
    image_files =os.path.join(image_dir, "*.jpg") 
    files = glob.glob (image_files) 
        
    for myFile in files:
        image = cv2.imread (myFile)
        images.append (image)

    data = np.array(images) 
    
 
 # random indexes for the images to be explored
 
    r =len(data)
    print (r)
    my_randoms = random.sample(range(r), images_to_explore)
    # list of random indexes 
    data_toexplore = my_randoms

    # Plot the images to explore 

    for i in range(len(data_toexplore)):
        plt.subplot(1, 4, i+1)
        plt.axis('off')
        #print(i)
        plt.imshow(data[data_toexplore[i]])
        plt.subplots_adjust(wspace=0.5)

    plt.show()
   
# Only replace with a directory of yours    

#DATASET_PATH = "/home/raz/anaconda3/chapter3/dataset/shadow"    
DATASET_PATH = "dataset-garbage/cardboard"    

#Call the function with 4 random images to be explored

data_explorer(DATASET_PATH, 4)

#if __name__ == "__main__":
#    a = sys.argv[1]
#    #b = sys.argv[2]
#    #print(b)
#    data_explorer(a)
#    
#    r =len(data)
#    my_randoms = random.sample(range(r), b)




# Determine the (random) indexes of the images that you want to see 
    
# only replace with a directory of yours

#data_explorer(pothole)
## generate random integer values
#from random import seed
#from random import randint
## seed random number generator
#seed(1)
## generate some integers
#r =len(data)
#print(r)
#for _ in range(5):
#	value = randint(0, r)
#	print(value)