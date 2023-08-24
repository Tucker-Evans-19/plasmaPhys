import numpy as np
import matplotlib.pyplot as plt
import spectraProp as sp
import pxtd_analysis as pa
import scipy.optimize as so



# file to be analyzed:
file = './data_files/PTD_s103740_ptd.h5'

#number of fits to average together:
num_rounds = 5
D3Hep_Tion =  9.3
DTn_Tion = 9
DDn_Tion = 11.11

# reading out the time axis and lineouts of the data channels that we care about:
image_time_axis, c1_lineout, c2_lineout, fid_lineout = pa.analyzePXTD_2channel(file)
image_time_axis = image_time_axis*.001 # converting the time scale to be in nanoseconds
image_time_axis += .7
# plotting the normalized lineouts from the two channels (neutrons and protons)
#plt.figure()
#plt.plot(image_time_axis, c1_lineout, c='red') 
#plt.plot(image_time_axis, c2_lineout, c='green') 

plt.figure()
plt.plot(c1_lineout, c='red')
plt.plot(c2_lineout, c='blue')

# interpolating data to an even grid for the fitting

time_axis = np.linspace(0, 3, 450)
data_interp1 = np.interp(time_axis, image_time_axis, c1_lineout, left=0) #interpolating data to intermediate grid
data_interp2 = np.interp(time_axis, image_time_axis, c2_lineout, left=0) 

#plt.figure() #plotting interpolated data
#plt.plot(time_axis, data_interp1)
#plt.plot(time_axis, data_interp2)
plt.figure()
plt.plot(data_interp1)
plt.plot(data_interp2)

print('click to the left and then to the right of the DTn peak')
DTn_left_click, DTn_right_click = plt.ginput(2)
print('click to the left and then to the right of the DDn peak')
DDn_left_click, DDn_right_click = plt.ginput(2)
print('click to the left and then to the right of the D3Hep peak')
D3Hep_left_click, D3Hep_right_click = plt.ginput(2)
print('click to the left and then to the right of a falling tail region with no signal expected')
falling_left_click, falling_right_click = plt.ginput(2)

# analyzing the falling tail region to fit a fall time:
def exponential_decay(x, fall_time):
    return np.exp(-x/fall_time)

def exponential_decay2(x, fall_time1, fall_time2):
    return np.exp(-x/fall_time1 - x/fall_time2)

falling_time = time_axis[int(falling_left_click[0]):int(falling_right_click[0])]
falling_lineout = data_interp2[int(falling_left_click[0]):int(falling_right_click[0])]

falling_fit, falling_cov = so.curve_fit(exponential_decay, falling_time-falling_time[0], falling_lineout/falling_lineout[0])
print(f'fall_time: {falling_fit}')

falling_fit2, falling_cov2 = so.curve_fit(exponential_decay2, falling_time-falling_time[0], falling_lineout/falling_lineout[0])
print(f'fall_times: {falling_fit2}')

fall_time = float(falling_fit[0])

# generating a filter based on the selection of left and right clicks above
DTn_filter = np.zeros(time_axis.shape)
DTn_filterLeft = int(DTn_left_click[0])
DTn_filterRight = int(DTn_right_click[0])

DTn_filter[DTn_filterLeft:DTn_filterRight] = np.ones(DTn_filterRight - DTn_filterLeft)

# generating a filter based on the selection of left and right clicks above
D3Hep_filter = np.zeros(time_axis.shape)
D3Hep_filterLeft = int(D3Hep_left_click[0])
D3Hep_filterRight = int(D3Hep_right_click[0])

D3Hep_filter[D3Hep_filterLeft:D3Hep_filterRight] = np.ones(D3Hep_filterRight - D3Hep_filterLeft)

# generating a filter based on the selection of left and right clicks above
DDn_filter = np.zeros(time_axis.shape)
DDn_filterLeft = int(DDn_left_click[0])
DDn_filterRight = int(DDn_right_click[0])

DDn_filter[DDn_filterLeft:DDn_filterRight] = np.ones(DDn_filterRight - DDn_filterLeft)


#fitting to the D3Hep signal:
plt.figure()
D3Hep_emission_fits = []
D3Hep_response_fits = []

plt.figure()
plt.plot(time_axis) 
plt.show()



# running through 30 fitting rounds with randomly generate spectra (same mean and spread)
pxtd_conv_mat = sp.pxtd_conv_matrix(time_axis)
D3Hep_offset = data_interp1[D3Hep_filterLeft]
for i in range(1, num_rounds+1):    
    e_spread_mat = sp.energy_spread_matrix(time_axis, 'D3Hep', D3Hep_Tion, 3.1)
    A = np.matmul((pxtd_conv_mat), (e_spread_mat))

    def objective_D3Hep(x):
        return (np.matmul(A, x) - data_interp1- D3Hep_offset)*(D3Hep_filter)

    result1 = so.least_squares(objective_D3Hep, np.zeros(time_axis.shape), bounds = (0, np.inf), method = 'dogbox')
    #result1 = so.least_squares(objective1, np.zeros(time_axis.shape), method = 'lm')
    
    emission_fit = result1.x
    D3Hep_emission_fits.append(emission_fit)
    D3Hep_response_fits.append(np.matmul(A, emission_fit)/np.max(np.matmul(A, emission_fit)))

for emission_fit in D3Hep_emission_fits:
    plt.plot(time_axis, emission_fit/np.max(emission_fit), c = 'blue')
for response_fit in D3Hep_response_fits:
    plt.plot(time_axis, response_fit/np.max(response_fit), c='k') 

emission_fit_matrix = np.asarray(D3Hep_emission_fits)
D3Hep_average_emission = np.average(emission_fit_matrix, axis =0)
D3Hep_var_emission = np.var(emission_fit_matrix, axis=0)


plt.plot(time_axis, data_interp1)
plt.plot(time_axis, D3Hep_average_emission/np.max(D3Hep_average_emission), c = 'red')


#fitting to the DTn signal:
plt.figure()
DTn_emission_fits = []
DTn_response_fits = []

# running through 30 fitting rounds with randomly generate spectra (same mean and spread)
pxtd_conv_mat = sp.pxtd_conv_matrix(time_axis)
DTn_offset = data_interp2[DTn_filterLeft]
for i in range(1, num_rounds+1):    
    e_spread_mat = sp.energy_spread_matrix(time_axis, 'DTn', DTn_Tion, 3.1)
    A = np.matmul((pxtd_conv_mat), (e_spread_mat))
    
    def objective_DTn(x):
        return (np.matmul(A, x) - data_interp2 - DTn_offset)*(DTn_filter)

    result1 = so.least_squares(objective_DTn, np.zeros(time_axis.shape), bounds = (0, np.inf), method = 'dogbox')
    #result1 = so.least_squares(objective1, np.zeros(time_axis.shape), method = 'lm')
    
    emission_fit = result1.x
    DTn_emission_fits.append(emission_fit)
    DTn_response_fits.append(np.matmul(A, emission_fit)/np.max(np.matmul(A, emission_fit)))

for emission_fit in DTn_emission_fits:
    plt.plot(time_axis, emission_fit/np.max(emission_fit), c = 'blue')
for response_fit in DTn_response_fits:
    plt.plot(time_axis, response_fit/np.max(response_fit), c='k') 

emission_fit_matrix = np.asarray(DTn_emission_fits)
DTn_average_emission = np.average(emission_fit_matrix, axis =0)
DTn_var_emission = np.var(emission_fit_matrix, axis=0)

# getting the first response fit to subtract from DDn signal as background
DTn_response_fit = DTn_response_fits[0]

plt.plot(time_axis, data_interp2)
plt.plot(time_axis, DTn_average_emission/np.max(DTn_average_emission), c = 'red')

#fitting to the DDn signal:
plt.figure()
DDn_emission_fits = []
DDn_response_fits = []

# running through 30 fitting rounds with randomly generate spectra (same mean and spread)
pxtd_conv_mat = sp.pxtd_conv_matrix(time_axis)
DDn_offset = data_interp2[DDn_filterLeft]

DDn_response_noDTn = data_interp2 - DTn_response_fit
DDn_response_noDTn -= np.min(DDn_response_noDTn)

for i in range(1, num_rounds+1):    
    e_spread_mat = sp.energy_spread_matrix(time_axis, 'DDn', DDn_Tion, 3.1)
    A = np.matmul((pxtd_conv_mat), (e_spread_mat))
    
    def objective_DDn(x):
        return (np.matmul(A, x) - DDn_response_noDTn)*(DDn_filter)

    result1 = so.least_squares(objective_DDn, np.zeros(time_axis.shape), bounds = (0, np.inf), method = 'dogbox')
    #result1 = so.least_squares(objective1, np.zeros(time_axis.shape), method = 'lm')
    
    emission_fit = result1.x
    DDn_emission_fits.append(emission_fit)
    DDn_response_fits.append(np.matmul(A, emission_fit)/np.max(np.matmul(A, emission_fit)))

for emission_fit in DDn_emission_fits:
    plt.plot(time_axis, emission_fit/np.max(emission_fit), c = 'blue')
for response_fit in DDn_response_fits:
    plt.plot(time_axis, response_fit/np.max(response_fit), c='k') 

emission_fit_matrix = np.asarray(DDn_emission_fits)
DDn_average_emission = np.average(emission_fit_matrix, axis =0)
DDn_var_emission = np.var(emission_fit_matrix, axis=0)

plt.plot(time_axis, DDn_response_noDTn/np.max(DDn_response_noDTn))
plt.plot(time_axis, DDn_average_emission/np.max(DDn_average_emission), c = 'red')

plt.figure()
plt.plot(time_axis, DDn_average_emission/np.max(DDn_average_emission), c = 'blue', linestyle = '--') 
plt.plot(time_axis, DTn_average_emission/np.max(DTn_average_emission), c = 'red', linestyle = '--')
plt.plot(time_axis, D3Hep_average_emission/np.max(D3Hep_average_emission), c = 'green', linestyle = '--')

# fitting a gaussian to each of the signals to get bang time and burn width
print('fitting gaussians to emission history fits')

def gaussian(x, a, b, c):
    return a*np.exp(-(x-b)**2/(2*c**2))

D3Hep_gauss_fit, cov = so.curve_fit(gaussian, time_axis, D3Hep_average_emission, p0 = [1, .8, .1])
DTn_gauss_fit, cov = so.curve_fit(gaussian, time_axis, DTn_average_emission, p0 = [1, .8, .1])
DDn_gauss_fit, cov = so.curve_fit(gaussian, time_axis, DDn_average_emission, p0= [1, .8, .1])

plt.figure()
plt.plot(time_axis, DDn_average_emission/np.max(DDn_average_emission), c = 'blue', linestyle = '--') 
plt.plot(time_axis, DTn_average_emission/np.max(DTn_average_emission), c = 'red', linestyle = '--')
plt.plot(time_axis, D3Hep_average_emission/np.max(D3Hep_average_emission), c = 'green', linestyle = '--')

D3Hep_gaussian = gaussian(time_axis, D3Hep_gauss_fit[0], D3Hep_gauss_fit[1], D3Hep_gauss_fit[2])
DTn_gaussian = gaussian(time_axis, DTn_gauss_fit[0], DTn_gauss_fit[1], DTn_gauss_fit[2])
DDn_gaussian = gaussian(time_axis, DDn_gauss_fit[0], DDn_gauss_fit[1], DDn_gauss_fit[2])

plt.plot(time_axis, D3Hep_gaussian/np.max(D3Hep_gaussian), c = 'green', linestyle = '-')
plt.plot(time_axis, DTn_gaussian/np.max(DTn_gaussian) , c = 'red', linestyle = '-')
plt.plot(time_axis, DDn_gaussian/np.max(DDn_gaussian), c = 'blue', linestyle = '-')

plt.xlim([.5, 1.2])

plt.figure()

plt.plot(time_axis, D3Hep_gaussian/np.max(D3Hep_gaussian), c = 'green', linestyle = '-')
plt.plot(time_axis, DTn_gaussian/np.max(DTn_gaussian) , c = 'red', linestyle = '-')
plt.plot(time_axis, DDn_gaussian/np.max(DDn_gaussian), c = 'blue', linestyle = '-')

plt.xlim([.5, 1.2])
print(D3Hep_gauss_fit)
print(DTn_gauss_fit)
print(DDn_gauss_fit)


plt.show()



