#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:48:33 2019

@author: raz
"""

# Import the module
import Augmentor
#da = Augmentor.Pipeline("fer-dataset-created/angry")
#da = Augmentor.Pipeline("fer-dataset-created/happy")
#da = Augmentor.Pipeline("fer-dataset-created/neutral")
#da = Augmentor.Pipeline("fer-dataset-created/sad")
da = Augmentor.Pipeline("fer-dataset-created/scared")

# Define the augmentation
da.rotate90(probability=0.5)
da.rotate270(probability=0.5)
da.flip_left_right(probability=0.8)
da.flip_top_bottom(probability=0.3)
da.crop_random(probability=1, percentage_area=0.5)
da.resize(probability=1.0, width=120, height=120)

# Do the augmentation operation
da.sample(1500)