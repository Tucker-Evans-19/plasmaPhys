import numpy as np
import matplotlib.pyplot as plt
import h5py as hp
import sys
from scipy import ndimage


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
shot_num = file[7:13]
data = hp.File(file, 'r')
image_raw = np.array(data.get('Streak_array'))
image_clean = image_raw[0,:,:] - image_raw[1,:,:]

image_rot = ndimage.rotate(image_clean, .0297*180/3.1415)
bg = image_rot[40:50,200:300]
bg_average = np.average(np.average(bg,1),0)
image_rot = image_rot - bg_average

image_rot = ndimage.median_filter(image_rot, size = 3)

c1_lineout = np.average(image_rot[175:200,:], 0)
c2_lineout = np.average(image_rot[145:170,:], 0)
fid_lineout = np.average(image_rot[70:90,:], 0)

c1_lineout_av = roll_average(c1_lineout, 3)
c2_lineout_av = roll_average(c2_lineout, 3)

plt.figure(dpi = 500)
plt.pcolor(image_rot)
plt.colorbar()

plt.savefig(file[:-3] + '_pcolor.png')

fig, ax = plt.subplots(3,1, dpi = 500)

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

plt.figure()
im = (np.squeeze(image_raw[0,100:460, 110:240] - image_raw[1,100:460, 110:240]))
plt.pcolor(im,vmax = 2000)
plt.colorbar()

while True:
	click = plt.ginput(1)
	pos = click[0]
	x = int(pos[1])
	y = int(pos[0])
	print(np.average(im[x-10:x+10,y-10:y+10]))








plt.show()
