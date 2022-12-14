import numpy as np
import matplotlib.pyplot as plt
import h5py as hp
import sys
from scipy import ndimage
import numpy.linalg as lin

def roll_average(vector_input, half_window):
	vector_averaged = vector_input
	for i in range(len(vector_averaged)):
		num_elements = 0
		average_sum = 0
		for w in range(i - half_window, i+half_window):
			if w >= 0 and w<len(vector_averaged):
				average_sum += vector_averaged[w]
				num_elements += 1
		vector_averaged[i] = average_sum/num_elements
	return vector_averaged

file = sys.argv[1]
print('analyzing ', file)
shot_num = file[5:11]
data = hp.File(file, 'r')
image_raw = np.array(data.get('Streak_array'))
image_clean = image_raw[0,:,:]

image_rot = ndimage.rotate(image_clean, .0297*180/3.1415)
bg = image_rot[40:50,200:300]
bg_average = np.average(np.average(bg,1),0)
image_rot = image_rot - bg_average

image_rot = ndimage.median_filter(image_rot, size = 3)

c1_lineout = np.average(image_rot[175:200,:], 0)
c2_lineout = np.average(image_rot[145:170,:], 0)
fid_lineout = np.average(image_rot[70:90,:], 0)

c1_lineout_av = roll_average(c1_lineout, 5)
c2_lineout_av = roll_average(c2_lineout, 5)

plt.figure()
plt.pcolor(image_rot)
plt.colorbar()

plt.savefig(file[:-3] + '_pcolor.png')

fig, ax = plt.subplots(3,1)

ax[0].plot(c1_lineout)
ax[0].plot(c1_lineout_av)
ax[1].plot(c2_lineout)
ax[1].plot(c2_lineout_av)
ax[2].plot(fid_lineout)
ax[2].set_xlabel('time --->')
ax[0].set_ylabel('ch1 signal')
ax[1].set_ylabel('ch2 signal')
ax[2].set_ylabel('fiducial signal')
ax[0].set_ylim(ymin = 0)
ax[1].set_ylim(ymin = 0)
ax[2].set_ylim(ymin = 0)

title_form = 'Channel lineouts, shot#: '
plt.suptitle(title_form + shot_num)
plt.savefig(file[:-3] + '_lineouts.png')

#determining the relative timing of the fiducial peaks
time_between = .547 ### check this VALUE!!!! 

#finding peaks in the fiducial lineout:
fid_cutoff = np.max(fid_lineout)*.55
fid_filter = fid_lineout >= fid_cutoff
#establish index ranges:
in_peak = False
left_sides = []
right_sides = []
for element in range(len(fid_filter)):
	
	if fid_filter[element] == 1:
		if in_peak:
			pass
		else:
			in_peak = True
			left_sides.append(element)
	else:
		if in_peak:
			in_peak = False
			right_sides.append(element)
		else:
			pass
print('Peak bounds detected: ' + str(len(left_sides)))
print(left_sides)
print(right_sides)

#trying to fit a gaussian to each segment:
peak_segments = []
for element in range(len(left_sides)):
	peak_segments.append(fid_lineout[left_sides[element]:right_sides[element]])

#creating the matrix of gaussians that we will compare to fiducial train:

peak_centers = []
for peak in peak_segments:
	seg_len = len(peak)
	peak_norm = peak*(np.max(peak))**-1
	G = np.zeros(shape = (seg_len, seg_len))
	for i in range(seg_len):
		for j in range(seg_len):
			G[i,j] = np.exp(-((j-i)**2)/50)
	Gt = np.transpose(G)
	x = np.matmul(np.matmul(lin.inv(np.matmul(Gt, G)), Gt), peak)
	center_ind = list(x).index(np.max(x))
	peak_centers.append(center_ind)
	#plt.figure()
	#plt.plot(peak)
	#plt.plot(np.matmul(G, x))
print(peak_centers) 
peak_centers = np.array(peak_centers)
left_sides = np.array(left_sides)
centers = peak_centers + left_sides

print(centers)
for center in centers:
	ax[2].axvline(center)

#finding the average distance between the timing fiducial peaks: 
peak_diffs = []
for element in range(1,len(centers)):
	peak_diffs.append(centers[element] - centers[element-1])

average_peak_diff = np.average(peak_diffs)

time_per_pixel = 548/average_peak_diff

#defining the shape of the response function
c2_lineout = c2_lineout[50:-50]
c1_lineout = c1_lineout[50:-50]


#background subtraction for lineouts:
c2_lineout = c2_lineout - np.average(c2_lineout[0:50])
c1_lineout = c1_lineout - np.average(c1_lineout[0:50])

#going through the inversion: 
fit_len = len(c2_lineout)
print(fit_len)

G = np.zeros(shape = (fit_len, fit_len))
for i in range(fit_len):
	for j in range(fit_len):
		if i<=j:
			G[i,j]= np.exp(-(j-i)*time_per_pixel/15)
		else:
			G[i,j] = np.exp((j-i)*time_per_pixel/1200)
#Gt = np.transpose(G)
#x = np.matmul(np.matmul(lin.inv(np.matmul(Gt, G)), Gt), c2_lineout)
xls1 = lin.lstsq(G,c1_lineout)
xls2 = lin.lstsq(G,c2_lineout)

xls1 = roll_average(xls1[0], 5)
xls2 = roll_average(xls2[0], 5)
fs = 15

fig, ax = plt.subplots(2,1, figsize = (6, 6), dpi=400)
time = np.linspace(1,len(xls1), len(xls1))*time_per_pixel

ax[0].plot(time, 10*xls1, c='r')
ax[0].plot(time, c1_lineout, zorder = 0, linestyle = '--', c='b')
ax[1].plot(time, 10*xls2, c='r')
ax[1].plot(time, c2_lineout, linestyle = '--', zorder=0, c='b')
#plt.suptitle('PXTD Signal Deconvolution')
ax[0].set_ylabel('Proton Signal', fontsize = fs)
ax[1].set_ylabel('Neutron Signal', fontsize = fs)
ax[1].set_xlabel('time (ps)', fontsize=fs)
#ax[0].set_xticklabels(fontsize=16)
#ax[1].set_xticklabels(fontsize=16)
plt.savefig('PXTD_Decon_'+shot_num+'.png')
plt.show()



