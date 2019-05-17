
"""
Chapter 9: Healthcare IoT
Code for ECG matlab signal exploration
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
 #Import to a python dictionary
Class1 = scipy.io.loadmat('dataset/ECG/A00001.mat') # Normal Rhythm
Class2 = scipy.io.loadmat('dataset/ECG/A00004.mat') # Atrial Fibrillation
Class3 = scipy.io.loadmat('dataset/ECG/A00008.mat') # Other Rhythm
Class4 = scipy.io.loadmat('dataset/ECG/A00022.mat') # Noisy measurement

# Scale Conversion (mV)
y1 = Class1["val"]/1000
y2 = Class2["val"]/1000
y3 = Class3["val"]/1000
y4 = Class4["val"]/1000

x =np.arange(0, 15, 1/300) # Number of sampling times to be displayed based on 300Hz sampling rate 
l=len(x)

# Plotting
fig, ax = plt.subplots(4, 1)
ax[0].plot(x, y1[0, 0:l], 'r') #row=0, col=0

ax[1].plot(x, y2[0, 0:l], 'b') #row=1, col=0

ax[2].plot(x, y3[0, 0:l], 'g') #row=0, col=1
ax[3].plot(x, y4[0, 0:l], 'k') #row=1, col=1
plt.subplots_adjust(hspace=.9)
fig.text(.5, 0.04, 'Sampling Time (sec)', ha='center', va='center')
fig.text(0.06, 0.5, 'Amplitude', ha='center', va='center', rotation='vertical')
ax[0].set_title('Normal Rhythm')
ax[1].set_title('Atrial Fibrillation')
ax[2].set_title('Other Rhythm')
ax[3].set_title('Noisy Measurement')
plt.savefig('ecg_data_exploration.png', dpi=300)
plt.show()
