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
import scipy.special as sp
from scipy.linalg import convolution_matrix

#------------------------------#
# Parameter definitions:
# -----------------------------#

#scintillator response
risetime = 20
falltime = 1200

# window on PXTD lineout that we are considering
fidlow = 82
fidhigh = 92

left_side = 100
right_side = -100

# defining the shape of the median filter
med_filt2 = 1
med_filt1 = 10

#-------------------------------------------------------------------
# basic function definitions
#-------------------------------------------------------------------
def timeAxis(x,a,b,c):
    # we need to have a fit to the timing fiducials.
    # according to Neel and Hong's work this can just be a quadratic
    return a*x**2 + b*x + c

def Gauss(x, A, B, x0):
	y = A*np.exp(-1*(x-x0)**2/(2*B**2))
	return y

def Two_Gauss(x, A1, B1, x1, A2, B2, x2):	
	y = A1*np.exp(-1*(x-x1)**2/(2*B1**2)) + A2*np.exp(-1*(x-x2)**2/(2*B2**2))
	return y

def skew_gaussian(x, A, B, x0, alpha):
    y = (1/np.sqrt(2*np.pi)) * Gauss(x, A, B, x0)*.5*(1 + sp.erf(alpha * (x-x0)/(sqrt(2)*2*B)))
    return y

def scint_IRF(t, t_rise, t_fall):
    irf_unnorm = (1-np.exp(-t/t_rise)) * np.exp(-t/t_fall)
    irf_norm = irf_unnorm/np.max(irf_unnorm)
    return irf_norm

def pxtd_IRF(t, t_rise, t_fall):
    return scint_IRF(t, t_rise, t_fall)


def pxtd_IRF_conv_mat(t, t_rise, t_fall, length):
    return convolution_matrix(pxtd_IRF(t, t_rise, t_fall), length, mode = 'same')

def convolved_signal(conv_mat, emission_history):
    return np.matmul(conv_mat, emission_history)

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


#--------------------------------------------------
# MAIN FUNCTIONS FOR ANALYZING A RAW PXTD H5 FILE: 
#--------------------------------------------------

def pxtd_lineouts_2channel(file):
    c1low = 181
    c1high = 189

    c2low = 161
    c2high = 168
    #----------------------------------------------------------------
    # FILE CONVERSION (if necessary)
    #----------------------------------------------------------------

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
    # FILE INTERPRETATION
    #---------------------------------------------------------------

    #breaking out file information:
    shot_num = file[-13:-7]
    data = hp.File(file, 'r')
    image_raw = np.array(data.get('Streak_array'))
    image_clean = image_raw[0,:,:] - image_raw[1,:,:]
    #breaking out the energy information 

    try:
        energy_file = sys.argv[2]

        with open(energy_file, 'r') as energy_data:
            header = energy_data.readline()
            spec_line = energy_data.readline()
            spec_info = spec_line.split()
            d3hep_e_mean, d3hep_e_err, d3hep_sigma, d3hep_sigma_err, T_D, T_T = spec_info
    except(IndexError):
        pass 

    #rotating the image (needed for all PXTD images)
    image_rot = ndimage.rotate(image_clean, .0297*180/3.1415)

    #taking blocks of full image that will be the basis of the lineouts once averaged
    c1_lineout_block = ndimage.median_filter(image_rot[c1low:c1high,left_side:right_side], size = (med_filt1, med_filt2))
    c2_lineout_block = ndimage.median_filter(image_rot[c2low:c2high,left_side:right_side], size = (med_filt1, med_filt2))
    fid_lineout_block = ndimage.median_filter(image_rot[fidlow:fidhigh, left_side:right_side], size = (med_filt1, med_filt2))
    #--------------------------------------------------
    # PCOLOR IMAGE - lineout blocks
    #--------------------------------------------------

    #Plotting pcolor of each lineout block
    #averaging to find lineout of each channel
    c1_lineout = np.average(c1_lineout_block, 0)
    c2_lineout = np.average(c2_lineout_block, 0)
    fid_lineout = np.average(fid_lineout_block, 0)

    c1_lineout -= min(c1_lineout) 
    c2_lineout -= min(c2_lineout)

    c1_lineout = c1_lineout/ np.max(c1_lineout)
    c2_lineout = c2_lineout/ np.max(c2_lineout)

    #----------------------------------------------------------
    # FIDUCIAL PEAK TIMING 
    #----------------------------------------------------------
    #finding peaks in the fiducial lineout:
    fid_cutoff = np.max(fid_lineout)*.5
    fid_filter = (fid_lineout >= fid_cutoff)
    norm_fid = fid_lineout/ np.max(fid_lineout)

    #establish index ranges:
    in_peak = False
    left_sides = []
    right_sides = []

    for element in range(round(len(fid_filter)*.8)):
        
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

    peak_segments = []
    #for element in range(len(left_sides)):
    #	peak_segments.append(fid_lineout[left_sides[element]:right_sides[element]])

    peak_centers = []
    peak_centers = .5*(np.array(left_sides) + np.array(right_sides))
    
    print('peak fiducial information: ')
    print(peak_centers)
    centers = peak_centers
    #finding the average distance between the timing fiducial peaks: 
    peak_diffs = []
    for element in range(1,len(centers)):
        peak_diffs.append(centers[element] - centers[element-1])

    average_peak_diff = np.average(peak_diffs)

    t_params, t_cov = curve_fit(timeAxis, centers, 548*np.arange(len(centers)))
    print(t_params)
    a,b,c = t_params
    time_axis = timeAxis(np.arange(len(c2_lineout)),a,b,c)
    time_axis -= min(time_axis)
    time_per_pixel = 548/average_peak_diff

    return time_axis, c1_lineout, c2_lineout, fid_lineout

def pxtd_lineouts_3channel(file):

    #defining regions of interest for the channels (assuming 3 channels)
    c1low = 181
    c1high = 189

    c2low = 161
    c2high = 168

    c3low = 140
    c3high = 150
    #----------------------------------------------------------------
    # FILE CONVERSION (if necessary)
    #----------------------------------------------------------------

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
    # FILE INTERPRETATION
    #---------------------------------------------------------------

    #breaking out file information:
    shot_num = file[-13:-7]
    data = hp.File(file, 'r')
    image_raw = np.array(data.get('Streak_array'))
    image_clean = image_raw[0,:,:] - image_raw[1,:,:]
    #breaking out the energy information 

    try:
        energy_file = sys.argv[2]

        with open(energy_file, 'r') as energy_data:
            header = energy_data.readline()
            spec_line = energy_data.readline()
            spec_info = spec_line.split()
            d3hep_e_mean, d3hep_e_err, d3hep_sigma, d3hep_sigma_err, T_D, T_T = spec_info
    except(IndexError):
        pass 

    #rotating the image (needed for all PXTD images)
    image_rot = ndimage.rotate(image_clean, .0297*180/3.1415)

    #taking blocks of full image that will be the basis of the lineouts once averaged
    c1_lineout_block = ndimage.median_filter(image_rot[c1low:c1high,left_side:right_side], size = (med_filt1, med_filt2))
    c2_lineout_block = ndimage.median_filter(image_rot[c2low:c2high,left_side:right_side], size = (med_filt1, med_filt2))
    c3_lineout_block = ndimage.median_filter(image_rot[c3low:c3high,left_side:right_side], size = (med_filt1, med_filt2))
    fid_lineout_block = ndimage.median_filter(image_rot[fidlow:fidhigh, left_side:right_side], size = (med_filt1, med_filt2))
    #--------------------------------------------------
    # PCOLOR IMAGE - lineout blocks
    #--------------------------------------------------

    #Plotting pcolor of each lineout block
    #averaging to find lineout of each channel
    c1_lineout = np.average(c1_lineout_block, 0)
    c2_lineout = np.average(c2_lineout_block, 0)
    c3_lineout = np.average(c3_lineout_block, 0)
    fid_lineout = np.average(fid_lineout_block, 0)

    c1_lineout -= min(c1_lineout) 
    c2_lineout -= min(c2_lineout)

    c1_lineout = c1_lineout/ np.max(c1_lineout)
    c2_lineout = c2_lineout/ np.max(c2_lineout)

    #----------------------------------------------------------
    # FIDUCIAL PEAK TIMING 
    #----------------------------------------------------------
    #finding peaks in the fiducial lineout:
    fid_cutoff = np.max(fid_lineout)*.5
    fid_filter = (fid_lineout >= fid_cutoff)
    norm_fid = fid_lineout/ np.max(fid_lineout)

    #establish index ranges:
    in_peak = False
    left_sides = []
    right_sides = []

    for element in range(round(len(fid_filter)*.8)):
        
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

    peak_segments = []
    #for element in range(len(left_sides)):
    #	peak_segments.append(fid_lineout[left_sides[element]:right_sides[element]])

    peak_centers = []
    peak_centers = .5*(np.array(left_sides) + np.array(right_sides))
    
    print('peak fiducial information: ')
    print(peak_centers)
    centers = peak_centers
    #finding the average distance between the timing fiducial peaks: 
    peak_diffs = []
    for element in range(1,len(centers)):
        peak_diffs.append(centers[element] - centers[element-1])

    average_peak_diff = np.average(peak_diffs)

    t_params, t_cov = curve_fit(timeAxis, centers, 548*np.arange(len(centers)))
    print(t_params)
    a,b,c = t_params
    time_axis = timeAxis(np.arange(len(c2_lineout)),a,b,c)
    time_axis -= min(time_axis)
    time_per_pixel = 548/average_peak_diff

    return time_axis, c1_lineout, c2_lineout, c3_lineout, fid_lineout
'''
def analyzePXTD(file, num_channels = 2):
    #--------------------------------------------------------------
    #SETTING NUMBER OF CHANNELS:
    #--------------------------------------------------------------

    if num_channels == 2:
        #defining regions of interest for the channels (assuming 2 channels)
        c1low = 181
        c1high = 189

        c2low = 161
        c2high = 168

    elif num_channels ==3:
        #defining regions of interest for the channels (assuming 3 channels)
        c1low = 181
        c1high = 189

        c2low = 161
        c2high = 168

        c3low = 140
        c3high = 150
    else:
        print('Incompatible number of channels. Exiting...') 
        time_axis = []
        
        xls1 = []
        xls2 = []
        fit1 = []
        fit2 = []
        
        return time_axis, xls1, xls2, fit1, fit2

    #----------------------------------------------------------------
    # FILE CONVERSION (if necessary)
    #----------------------------------------------------------------

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
    # FILE INTERPRETATION
    #---------------------------------------------------------------

    #breaking out file information:
    shot_num = file[-13:-7]
    data = hp.File(file, 'r')
    image_raw = np.array(data.get('Streak_array'))
    image_clean = image_raw[0,:,:] - image_raw[1,:,:]
    #breaking out the energy information 

    try:
        energy_file = sys.argv[2]

        with open(energy_file, 'r') as energy_data:
            header = energy_data.readline()
            spec_line = energy_data.readline()
            spec_info = spec_line.split()
            d3hep_e_mean, d3hep_e_err, d3hep_sigma, d3hep_sigma_err, T_D, T_T = spec_info
    except(IndexError):
        pass 

    #rotating the image (needed for all PXTD images)
    image_rot = ndimage.rotate(image_clean, .0297*180/3.1415)

    #taking blocks of full image that will be the basis of the lineouts once averaged
    c1_lineout_block = ndimage.median_filter(image_rot[c1low:c1high,left_side:right_side], size = (med_filt1, med_filt2))
    c2_lineout_block = ndimage.median_filter(image_rot[c2low:c2high,left_side:right_side], size = (med_filt1, med_filt2))
    if num_channels == 3: 
        c3_lineout_block = ndimage.median_filter(image_rot[c3low:c3high,left_side:right_side], size = (med_filt1, med_filt2))
    fid_lineout_block = ndimage.median_filter(image_rot[fidlow:fidhigh, left_side:right_side], size = (med_filt1, med_filt2))
    #--------------------------------------------------
    # PCOLOR IMAGE - lineout blocks
    #--------------------------------------------------

    #Plotting pcolor of each lineout block
    #averaging to find lineout of each channel
    c1_lineout = np.average(c1_lineout_block, 0)
    c2_lineout = np.average(c2_lineout_block, 0)
    if num_channels ==3:
        c3_lineout = np.average(c3_lineout_block, 0)

    fid_lineout = np.average(fid_lineout_block, 0)


    c1_lineout -= min(c1_lineout) 
    c2_lineout -= min(c2_lineout)

    c1_lineout = c1_lineout/ np.max(c1_lineout)
    c2_lineout = c2_lineout/ np.max(c2_lineout)

    #----------------------------------------------------------
    # FIDUCIAL PEAK TIMING 
    #----------------------------------------------------------
    #finding peaks in the fiducial lineout:
    fid_cutoff = np.max(fid_lineout)*.5
    fid_filter = (fid_lineout >= fid_cutoff)
    norm_fid = fid_lineout/ np.max(fid_lineout)

    #establish index ranges:
    in_peak = False
    left_sides = []
    right_sides = []

    for element in range(round(len(fid_filter)*.8)):
        
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

    peak_segments = []
    fid_lineout  = fid_lineout[left_side:right_side]
    #for element in range(len(left_sides)):
    #	peak_segments.append(fid_lineout[left_sides[element]:right_sides[element]])

    peak_centers = []
    peak_centers = .5*(np.array(left_sides) + np.array(right_sides))
    
    print('peak fiducial information: ')
    print(peak_centers)
    centers = peak_centers
    #finding the average distance between the timing fiducial peaks: 
    peak_diffs = []
    for element in range(1,len(centers)):
        peak_diffs.append(centers[element] - centers[element-1])

    average_peak_diff = np.average(peak_diffs)

    t_params, t_cov = curve_fit(timeAxis, centers, 548*np.arange(len(centers)))
    print(t_params)
    a,b,c = t_params
    time_axis = timeAxis(np.arange(len(c2_lineout)),a,b,c)
    time_axis -= min(time_axis)
    time_per_pixel = 548/average_peak_diff
    #-----------------------------------------------
    # DECONVOLUTION 
    #-----------------------------------------------

    #INVERSION OF CHANNEL DATA:
    #going through the inversion: 
    fit_len = len(c2_lineout)

    #convolution_matrix = np.transpose(pxtd_IRF_conv_mat(time_axis, risetime, falltime, fit_len))
    G = np.zeros(shape = (fit_len, fit_len))
    for i in range(fit_len):
        for j in range(fit_len):
            if j >= i:
                G[i,j] = (1-np.exp(-(time_axis[j] - time_axis[i])/risetime))*np.exp(-(time_axis[j] - time_axis[i])/falltime)
            else:
                G[i,j] = 0

    convolution_matrix = np.transpose(G)

    def convolved_signal1(emission_history):
        return (np.matmul(convolution_matrix, np.transpose(emission_history)) - c1_lineout)

    def convolved_signal2(emission_history):
        return (np.matmul(convolution_matrix, np.transpose(emission_history)) - c2_lineout)

    c1_grad = np.gradient(c1_lineout) 
    c1_grad_norm = (c1_grad >0)*c1_grad/np.max(c1_grad)
    c2_grad = np.gradient(c2_lineout)
    c2_grad_norm = (c2_grad>0) * c2_grad/np.max(c2_grad)

    res1 = least_squares(convolved_signal1,c1_grad_norm,bounds = (0, np.inf),  method = 'trf') 
    res2 = least_squares(convolved_signal2,c2_grad_norm,bounds = (0, np.inf),  method = 'trf')

    xls1 = res1.x
    xls2 = res2.x

    fit1 = np.matmul(convolution_matrix, np.transpose(xls1)) 
    fit2 = np.matmul(convolution_matrix, np.transpose(xls2))
    xls1[-1] = 0
    xls2[-1] = 0

    return time_axis, xls1, xls2, fit1, fit2, shot_num, image_rot




def analyzePXTD_energy(file, num_channels = 2):
    #--------------------------------------------------------------
    #SETTING NUMBER OF CHANNELS:
    #--------------------------------------------------------------

    if num_channels == 2:
        #defining regions of interest for the channels (assuming 2 channels)
        c1low = 181
        c1high = 189

        c2low = 161
        c2high = 168

        c3low = 140
        c3high = 150
    elif num_channels ==3:
        #defining regions of interest for the channels (assuming 3 channels)
        c1low = 181
        c1high = 189

        c2low = 161
        c2high = 168

        c3low = 140
        c3high = 150
    else:
        print('Incompatible number of channels. Exiting...') 
        time_axis = []
        
        xls1 = []
        xls2 = []
        fit1 = []
        fit2 = []
        
        return time_axis, xls1, xls2, fit1, fit2

    #----------------------------------------------------------------
    # FILE CONVERSION (if necessary)
    #----------------------------------------------------------------

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
    # FILE INTERPRETATION
    #---------------------------------------------------------------

    #breaking out file information:
    shot_num = file[-13:-7]
    data = hp.File(file, 'r')
    image_raw = np.array(data.get('Streak_array'))
    image_clean = image_raw[0,:,:] - image_raw[1,:,:]
    #breaking out the energy information 

    try:
        energy_file = sys.argv[2]

        with open(energy_file, 'r') as energy_data:
            header = energy_data.readline()
            spec_line = energy_data.readline()
            spec_info = spec_line.split()
            d3hep_e_mean, d3hep_e_err, d3hep_sigma, d3hep_sigma_err, T_D, T_T = spec_info
    except(IndexError):
        pass 

    #rotating the image (needed for all PXTD images)
    image_rot = ndimage.rotate(image_clean, .0297*180/3.1415)

    #taking blocks of full image that will be the basis of the lineouts once averaged
    c1_lineout_block = ndimage.median_filter(image_rot[c1low:c1high,left_side:right_side], size = (med_filt1, med_filt2))
    c2_lineout_block = ndimage.median_filter(image_rot[c2low:c2high,left_side:right_side], size = (med_filt1, med_filt2))
    c3_lineout_block = ndimage.median_filter(image_rot[c3low:c3high,left_side:right_side], size = (med_filt1, med_filt2))
    fid_lineout_block = ndimage.median_filter(image_rot[fidlow:fidhigh, left_side:right_side], size = (med_filt1, med_filt2))
    #--------------------------------------------------
    # PCOLOR IMAGE - lineout blocks
    #--------------------------------------------------

    #Plotting pcolor of each lineout block
    #averaging to find lineout of each channel
    c1_lineout = np.average(c1_lineout_block, 0)
    c2_lineout = np.average(c2_lineout_block, 0)
    c3_lineout = np.average(c3_lineout_block, 0)

    fid_lineout = np.average(fid_lineout_block, 0)


    c1_lineout -= min(c1_lineout) 
    c2_lineout -= min(c2_lineout)

    c1_lineout = c1_lineout/ np.max(c1_lineout)
    c2_lineout = c2_lineout/ np.max(c2_lineout)

    #----------------------------------------------------------
    # FIDUCIAL PEAK TIMING 
    #----------------------------------------------------------
    #finding peaks in the fiducial lineout:
    fid_cutoff = np.max(fid_lineout)*.5
    fid_filter = (fid_lineout >= fid_cutoff)
    norm_fid = fid_lineout/ np.max(fid_lineout)

    #establish index ranges:
    in_peak = False
    left_sides = []
    right_sides = []


    for element in range(round(len(fid_filter)*.8)):
        
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

    peak_segments = []
    fid_lineout  = fid_lineout[left_side:right_side]
    #for element in range(len(left_sides)):
    #	peak_segments.append(fid_lineout[left_sides[element]:right_sides[element]])

    peak_centers = []
    peak_centers = .5*(np.array(left_sides) + np.array(right_sides))
    
    print('peak fiducial information: ')
    print(peak_centers)
    centers = peak_centers
    #finding the average distance between the timing fiducial peaks: 
    peak_diffs = []
    for element in range(1,len(centers)):
        peak_diffs.append(centers[element] - centers[element-1])

    average_peak_diff = np.average(peak_diffs)

    t_params, t_cov = curve_fit(timeAxis, centers, 548*np.arange(len(centers)))
    print(t_params)
    a,b,c = t_params
    time_axis = timeAxis(np.arange(len(c2_lineout)),a,b,c)
    time_axis -= min(time_axis)
    time_per_pixel = 548/average_peak_diff
    #-----------------------------------------------
    # DECONVOLUTION 
    #-----------------------------------------------

    #INVERSION OF CHANNEL DATA:
    #going through the inversion: 
    fit_len = len(c2_lineout)

    #convolution_matrix = np.transpose(pxtd_IRF_conv_mat(time_axis, risetime, falltime, fit_len))
    G = np.zeros(shape = (fit_len, fit_len))
    for i in range(fit_len):
        for j in range(fit_len):
            if j >= i:
                G[i,j] = (1-np.exp(-(time_axis[j] - time_axis[i])/risetime))*np.exp(-(time_axis[j] - time_axis[i])/falltime)
            else:
                G[i,j] = 0

    convolution_matrix = np.transpose(G)

    def convolved_signal1(emission_history):
        return (np.matmul(convolution_matrix, np.transpose(emission_history)) - c1_lineout)

    def convolved_signal2(emission_history):
        return (np.matmul(convolution_matrix, np.transpose(emission_history)) - c2_lineout)

    c1_grad = np.gradient(c1_lineout) 
    c1_grad_norm = (c1_grad >0)*c1_grad/np.max(c1_grad)
    c2_grad = np.gradient(c2_lineout)
    c2_grad_norm = (c2_grad>0) * c2_grad/np.max(c2_grad)

    res1 = least_squares(convolved_signal1,c1_grad_norm,bounds = (0, np.inf),  method = 'trf') 
    res2 = least_squares(convolved_signal2,c2_grad_norm,bounds = (0, np.inf),  method = 'trf')

    xls1 = res1.x
    xls2 = res2.x

    fit1 = np.matmul(convolution_matrix, np.transpose(xls1)) 
    fit2 = np.matmul(convolution_matrix, np.transpose(xls2))
    xls1[-1] = 0
    xls2[-1] = 0

    return time_axis, xls1, xls2, fit1, fit2, shot_num, image_rot

'''
