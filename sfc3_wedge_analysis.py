import matplotlib.pyplot as plt
import sys
import os
import h5py as hp
import numpy as np
import scipy.ndimage as nd
import cv2
import matplotlib.patches as patch
from scipy.optimize import curve_fit
from scipy.special import erf

file = sys.argv[1]
prefix = file[:-3]
print(prefix)
data = hp.File(file,'r')
image_raw = np.array(data.get('pds_image'))


# plot filtered image:
fig, ax = plt.subplots(4, 1)
image_filt = nd.median_filter(image_raw, 5)
image_filt = nd.gaussian_filter(image_filt, 5)
lineout = np.average(image_filt, 0)
gradient = np.gradient(lineout)
ax[0].imshow(image_filt)
ax[1].plot(lineout)
ax[2].plot(gradient)
ax[3].plot(lineout * (np.abs(gradient)<1))


indices = []
step_vals = []


for ind in range(len(lineout)):
	if np.abs(gradient[ind])<1:
		indices.append(float(ind))
		step_vals.append(lineout[ind])


indices = np.array(indices)
step_vals = np.array(step_vals)

plt.figure()
plt.scatter(np.log(indices*float(max(indices))**-1), step_vals)

def response_curve(x, A, x0, d, c):
	return A *(1- erf((x-x0)/d))+c

param, cov = curve_fit(response_curve, (indices), step_vals, p0 = [3000, 600,200, 1000])
xfin = np.linspace(min(indices), max(indices), 1000)

print(param)
plt.figure()
plt.scatter((indices), step_vals)
plt.plot((xfin), response_curve((xfin), param[0], param[1], param[2], param[3]), c='red')

with open(prefix + '_analysis.txt', 'w') as data:
    for element in param:
        data.write(str(element)+'\n')

plt.show()
