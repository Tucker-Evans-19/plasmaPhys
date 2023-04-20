import numpy as np
import matplotlib.pyplot as plt
import h5py as hp
import sys
from scipy import ndimage
import scipy.linalg as lin
from scipy.optimize import curve_fit
from scipy.optimize import nnls
from scipy.optimize import least_squares
import numpy.linalg as linalg
import os
import matplotlib.patches as pat
import pandas as pd


#-------------------------------------------------------------------
# basic function definitions
#-------------------------------------------------------------------

def Gauss(x, A, B, x0):
	y = A*np.exp(-1*(x-x0)**2/(2*B**2))
	return y

def Two_Gauss(x, A1, B1, x1, A2, B2, x2):	
	y = A1*np.exp(-1*(x-x1)**2/(2*B1**2)) + A2*np.exp(-1*(x-x2)**2/(2*B2**2))
	return y


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
#----------------------------------------------------------------
# FILE CONVERSION (if necessary)
#----------------------------------------------------------------
file = sys.argv[1]

#making sure that this is in h5 format and if not, updating it. 
if file[-3:] == 'hdf':
	print('Converting to h5 format')
	os.system(f"./h4toh5 {file}")
	file = file[:-4] + '.h5'
elif file[-3:] == '.h5':
	print('Recognized file format')
else:
	print('Not a recognized file format. Exiting.')
	sys.exit()

print(f'analyzing {file}')

#---------------------------------------------------------------
# USER PEAK INPUTS
#--------------------------------------------------------------

# requesting the correct number of peaks for each channel from the user:
print('Enter the number of peaks on each channel. If this does not apply (you do not want fitting), just enter 0.')

np1 = input('Enter number of peaks on channel 1:')
np2 = input('Enter number of peaks on channel 2:')
np3 = input('Enter number of peaks on channel 3:')

np1 = int(np1)
np2 = int(np2)
np3 = int(np3)

#---------------------------------------------------------------
# FILE INTERPRETATION
#---------------------------------------------------------------

#breaking out file information:
shot_num = file[-13:-7]
data = hp.File(file, 'r')
image_raw = np.array(data.get('Streak_array'))
image_clean = image_raw[0,:,:] - image_raw[1,:,:]

#rotating the image (needed for all PXTD images)
image_rot = ndimage.rotate(image_clean, .0297*180/3.1415)
	
#defining regions of interest for the channels (assuming 3 channels)
c1low = 210
c1high = 220

c2low = 175
c2high = 185

c3low = 140
c3high = 150

fidlow = 80
fidhigh = 90

med_filt2 = 3
med_filt1 = 10

#taking blocks of full image that will be the basis of the lineouts once averaged
c1_lineout_block = ndimage.median_filter(image_rot[c1low:c1high,100:-50], size = (med_filt1, med_filt2))
c2_lineout_block = ndimage.median_filter(image_rot[c2low:c2high,100:-50], size = (med_filt1, med_filt2))
c3_lineout_block = ndimage.median_filter(image_rot[c3low:c3high,100:-50], size = (med_filt1, med_filt2))

#--------------------------------------------------
# PCOLOR IMAGE - lineout blocks
#--------------------------------------------------

#Plotting pcolor of each lineout block
fig, ax = plt.subplots(3,1)
ax[0].pcolor(c1_lineout_block)
ax[1].pcolor(c2_lineout_block)
ax[2].pcolor(c3_lineout_block)


#averaging to find lineout of each channel
c1_lineout = np.average(c1_lineout_block, 0)
c2_lineout = np.average(c2_lineout_block, 0)
c3_lineout = np.average(c3_lineout_block, 0)

c1_lineout = c1_lineout - c1_lineout[0]
c2_lineout = c2_lineout - c2_lineout[0]
c3_lineout = c3_lineout - c3_lineout[0]

fid_lineout = np.average(image_rot[fidlow:fidhigh,:], 0)


#---------------------------------------------------
# PCOLOR IMAGE - full
#--------------------------------------------------


#plotting pcolor of full image and regions used to create the channel lineouts:
fig, ax = plt.subplots()
pplot = ax.pcolor(image_rot, vmin = 0, vmax = 3300, cmap = 'Greys')
ax.set_xlim([100,400])
ax.set_ylim([50,250])
plt.colorbar(pplot)


ax.text(110, c1high + 5 , 'channel 1', color = 'r')
ax.axhline(c1low, c='r')
ax.axhline(c1high, c='r')

ax.text(110, c2high+5, 'channel 2', color = 'g') 
ax.axhline(c2low, c='g')
ax.axhline(c2high, c='g')


ax.text(110, c3high+5, 'channel 3', color = 'b')
ax.axhline(c3low, c='b')
ax.axhline(c3high, c='b')

ax.text(110, fidhigh+5, 'channel 4')
ax.axhline(fidlow, c='white', linestyle = '--')
ax.axhline(fidhigh, c='white', linestyle = '--')

plt.savefig(file[:-3] + '_pcolor.png')


#----------------------------------------------------------
# FIDUCIAL PEAK TIMING 
#----------------------------------------------------------

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

peak_centers = np.array(peak_centers)
left_sides = np.array(left_sides)
centers = peak_centers + left_sides

#finding the average distance between the timing fiducial peaks: 
peak_diffs = []
for element in range(1,len(centers)):
	peak_diffs.append(centers[element] - centers[element-1])

average_peak_diff = np.average(peak_diffs)

time_per_pixel = 548/average_peak_diff



#-----------------------------------------------
# DECONVOLUTION 
#-----------------------------------------------


#defining the shape of the response function
c2_lineout = c2_lineout[50:-50]
c1_lineout = c1_lineout[50:-50]
c3_lineout = c3_lineout[50:-50]

#INVERSION OF CHANNEL DATA:
#going through the inversion: 
fit_len = len(c2_lineout)
#print(fit_len)
risetime = 15
falltime = 1200

G = np.zeros(shape = (fit_len, fit_len))
for i in range(fit_len):
	for j in range(fit_len):
		if i<=j:
			G[i,j]= np.exp(-(j-i)*time_per_pixel/risetime)
		else:
			G[i,j] = np.exp((j-i)*time_per_pixel/falltime)
#Gt = np.transpose(G)
#x = np.matmul(np.matmul(lin.inv(np.matmul(Gt, G)), Gt), c2_lineout)
#xls1 = lin.solve(G,c1_lineout, assume_a='pos')
#xls2 = lin.solve(G,c2_lineout, assume_a='pos')
#xls3 = lin.solve(G,c3_lineout, assume_a='pos')

#xls1 = linalg.lstsq(G,c1_lineout, rcond = None)
#xls2 = linalg.lstsq(G,c2_lineout, rcond = None)
#xls3 = linalg.lstsq(G,c3_lineout, rcond = None)

#xls1 = roll_average(xls1[0], 5)
#xls2 = roll_average(xls2[0], 5)
#xls3 = roll_average(xls3[0], 5)


xls1, r1 = nnls(G, c1_lineout, maxiter = None)
xls2, r2 = nnls(G, c2_lineout, maxiter = None) 
xls3, r3 = nnls(G, c3_lineout, maxiter = None)

Grad2 = np.zeros(shape = (fit_len, fit_len))
sums = np.ones(shape = (fit_len,))
for i in range(2,fit_len-1):
	Grad2[i, i-1] = 1
	Grad2[i, i] = -2
	Grad2[i, i+1] = 1
if False:
	def Gox1(x):
		return np.abs(np.matmul(G,x) - c1_lineout) + np.sum(np.gradient(np.gradient(x))**2)
	def Gox2(x):
		return np.abs(np.matmul(G,x) - c2_lineout) + np.sum(np.gradient(np.gradient(x))**2)
	def Gox3(x):
		return np.abs(np.matmul(G,x) - c3_lineout) + np.sum(np.gradient(np.gradient(x))**2)
if False:
	weights = np.ones(shape = (fit_len,))
	
	def Gox1(x):
		#return sums @ ((np.matmul((np.matmul(G,x) - c1_lineout), weights)**2) + (np.matmul(Grad2, x)**2))
		return ((np.matmul((np.matmul(G,x) - c1_lineout), weights)**2))
	def Gox2(x):
		return ((np.matmul((np.matmul(G,x) - c2_lineout), weights)**2))
	def Gox3(x):
		return ((np.matmul((np.matmul(G,x) - c3_lineout), weights)**2))

	
	xls1_fit = least_squares(Gox1, roll_average(np.abs(np.gradient(c1_lineout)), 10), bounds = (0, np.inf))
	xls2_fit = least_squares(Gox2, np.abs(np.gradient(c2_lineout)), bounds = (0, np.inf))
	xls3_fit = least_squares(Gox3, np.abs(np.gradient(c3_lineout)), bounds = (0, np.inf))





	xls1 = xls1_fit.x
	xls2 = xls2_fit.x
	xls3 = xls3_fit.x


#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------
# PLOTTING DECONVOLUTION LINEOUTS
#----------------------------------------------------------------------------------------------------------


fs = 15

fig, ax = plt.subplots(3,1, dpi=150)
time = np.linspace(1,len(xls1), len(xls1))*time_per_pixel

ax[0].plot(time, xls1, c='r')
ax[0].plot(time, c1_lineout*.1, zorder = 0, linestyle = '--', c='b')

ax[1].plot(time, xls2, c='r')
ax[1].plot(time, c2_lineout*.1, linestyle = '--', zorder=0, c='b')

ax[2].plot(time, xls3, c='r')
ax[2].plot(time, c3_lineout*.1, linestyle = '--', zorder=0, c='b')

#plt.suptitle('PXTD Signal Deconvolution')
ax[0].set_ylabel('channel 1', fontsize = fs)
ax[1].set_ylabel('channel 2', fontsize = fs)
ax[2].set_ylabel('channel 3', fontsize = fs)

ax[2].set_xlabel('time (ps)', fontsize=fs)



print('Pick peaks on each channel from left to right, top to bottom.')
peak_clicks1 = plt.ginput(np1)
peak_clicks2 = plt.ginput(np2)
peak_clicks3 = plt.ginput(np3)
	
peak_clicks = [peak_clicks1, peak_clicks2, peak_clicks3]

pgs = [[], [], []]
heights = [[],[],[]]

for ind in range(len(peak_clicks)):
	for peak in peak_clicks[ind]:
		pgs[ind].append(int(peak[0]/time_per_pixel))


for ind in range(len(peak_clicks)):
	for peak in peak_clicks[ind]:
		heights[ind].append(int(peak[1]))

print(pgs)

# setting ylimits 
#ax[0].set_ylim([0, 3500])
#ax[1].set_ylim([0, 3500])
#ax[2].set_ylim([0, 3500])


#ax[0].set_xticklabels(fontsize=16)
#ax[1].set_xticklabels(fontsize=16)
# general setup for the following taken from : 
# https://www.geeksforgeeks.org/python-gaussian-fit/

nps = [np1, np2, np3]
params = [] 
covs = [] 
scaled_time = np.linspace(1, len(xls1), len(xls1))
fits = []
xs = [xls1, xls2, xls3]
for channel in range(3):
	
	if nps[channel] ==0:
		pass
	elif nps[channel] == 1:
		
		param, cov = curve_fit(Gauss, scaled_time, xs[channel], p0=[heights[channel][0],10, pgs[channel][0]], bounds = ([0,[np.inf, np.inf,np.inf]]))
		fits.append(Gauss(scaled_time, param[0], param[1], param[2]))
		params.append(param)
		covs.append(cov)
		
	elif nps[channel] == 2:	
		param, cov = curve_fit(Two_Gauss, scaled_time, xs[channel], p0 = [heights[channel][0], 10, pgs[channel][0], heights[channel][1], 10, pgs[channel][1]])
		fits.append(Two_Gauss(scaled_time, param[0], param[1], param[2], param[3], param[4], param[5]))
		params.append(param)
		covs.append(cov)
	else:
		print('Currently unsupported number of peaks for this channel')
		sys.exit()
if False:
	# Fitting the outputs to gaussians:
	if np1 ==0:
		pass
	elif np1 ==1:
		param1, cov1 = curve_fit(Gauss, np.linspace(1,len(xls1), len(xls1)), xls1, p0=[500,100, peak_guess1], bounds = ([0,[5000, 200, 200]]))
		fit_A1 = param1[0]
		fit_B1 = param1[1]
		fit_x1 = param1[2]
			
		fit1 = Gauss(np.linspace(1,len(xls1), len(xls1)), fit_A1, fit_B1, fit_x1)
		err1 = np.sqrt(np.diag(cov1))
		
	elif np1 ==2:	
		param1, cov1 = curve_fit(Two_Gauss, np.linspace(1,len(xls1), len(xls1)), xls1, p0 = [500, 100, 100, 700, 100, 100])
		fit1 = Two_Gauss(np.linspace(1, len(xls1), len(xls1)), param1[0], param1[1], param1[2], param1[3], param1[4], param1[5])
		err1 = np.sqrt(np.diag(cov1))
	else:
		print('Currently unsupported number of peaks for this channel')
		sys.exit()




	if np2 ==0:
		pass
	elif np2 ==1:
		param2a, cov2a = curve_fit(Gauss, np.linspace(1,len(xls2), len(xls2)), xls2, p0=[500,100, 100], bounds = ([0,[5000, 200, 200]]))	
		fit2 = Gauss(np.linspace(1,len(xls2), len(xls2)), param2a[0], param2a[1], param2a[2])
		err2 = np.sqrt(np.diag(cov2a))

	elif np2 ==2:	
		
		#try:
		#	param2, cov2 = curve_fit(Two_Gauss, np.linspace(1,len(xls2), len(xls2)), xls2, p0 = [1500, 100, 100, 500, 100, 120], maxfev = 10000)
		#	fit2 = Two_Gauss(np.linspace(1, len(xls2), len(xls2)), param2[0], param2[1], param2[2], param2[3], param2[4], param2[5])
		#except(RuntimeError):
		#	print('We were not able to find both peaks on channel 2. Now fitting for one peak.')
		#	param2, cov2 = curve_fit(Gauss, np.linspace(1,len(xls2), len(xls2)), xls2, p0=[500,100, 4000/time_per_pixel ])	
		#	fit2 = Gauss(np.linspace(1,len(xls2), len(xls2)), param2[0], param2[1], param2[2])
		
		#print('We were not able to find both peaks on channel 2. Now fitting for one peak.')
		param2a, cov2a = curve_fit(Gauss, np.linspace(1,len(xls2), len(xls2)), xls2, p0=[500,100, 100 ],  bounds = ([0,[5000, 200, 200]]))	
		fit2_temp = Gauss(np.linspace(1,len(xls2), len(xls2)), param2a[0], param2a[1], param2a[2])
		xls2_temp = xls2 - fit2_temp
		param2b, cov2b = curve_fit(Gauss, np.linspace(1, len(xls2_temp), len(xls2_temp)), xls2_temp, p0 = [500, 100, 100],  bounds = ([0,[5000, 200, 200]]))
		
		fit2 = fit2_temp +  Gauss(np.linspace(1,len(xls2), len(xls2)), param2b[0], param2b[1], param2b[2])
		err2a = np.sqrt(np.diag(cov2a))
		err2b = np.sqrt(np.diag(cov2b))
		err2 = err2a + err2b
	else:
		print('Currently unsupported number of peaks for this channel')
		sys.exit()

fit1, fit2, fit3 = fits


# if you did the fitting for gaussian peaks, they will be plotted here: 
if np1 > 0:
	ax[0].plot(time, fit1, c='orange')
	#ax[0].fill_between(time, fit1*10 + 10*np.sqrt(err1), fit1*10 - 10*np.sqrt(err1))


if np2>0:
	ax[1].plot(time, fit2, c='orange')
	#ax[0].fill_between(time, fit2*10 + 10*err2, fit2*10 - 10*err2)

if np3>0:
	ax[2].plot(time, fit3, c='orange')




#print(param1)
#print(param2a)
try:
	print('\nChannel 1 peak info:\n')
	print(f'\tpeak height: {param1[0]}')
	print(f'\tpeak width: {param1[1]*time_per_pixel}')
	print(f'\tpeak center: {param1[2]*time_per_pixel}')

	print('\nChannel 2 peak info:\n')
	print(f'\tpeak height: {param2a[0]}')
	print(f'\tpeak width: {param2a[1]*time_per_pixel}')
	print(f'\tpeak center: {param2a[2]*time_per_pixel}')



#	print(param2b)

	print('\nChannel 2b peak info:\n')
	print(f'\tpeak height: {param2b[0]}')
	print(f'\tpeak width: {param2b[1]*time_per_pixel}')
	print(f'\tpeak center: {param2b[2]*time_per_pixel}')

except(NameError):
	pass



ax[0].grid()
ax[1].grid()
ax[2].grid()
plt.suptitle('Time Resolved Emission History (PXTD): '+shot_num)
plt.savefig('./PXTD_Decon_'+shot_num+'.png')

line_outputs = pd.DataFrame(
	{
		"time":time,
		"channel1_raw":c1_lineout,
		"channel2_raw":c2_lineout,
		"channel3_raw":c3_lineout,
		"channel1_decon":xls1, 
		"channel2_decon":xls2, 
		"channel3_decon":xls3,
		"channel1_fit":fit1,
		"channel2_fit":fit2,
		"channel3_fit":fit3
	}

)

#converting the fits to be all of the same length
for ind in range(len(params)):
	param = params[ind]
	if param.shape[0] ==3:
		param = np.concatenate((param, [0,0,0]))
	params[ind] = param


print(params)

fit_outputs = pd.DataFrame(
	{
		"peak_params1":params[0],
		"peak_params2":params[1],
		"peak_params3":params[2]
	}	
)

line_outputs.to_csv('PXTD_lineout_output_'+shot_num+'.csv')
fit_outputs.to_csv('PXTD_peakFit_output_'+shot_num+'.csv')




print('--------------------Analysis complete--------------------')

plt.show()

