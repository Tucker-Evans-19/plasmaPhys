import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import pandas as pd
import sys
import os
from numpy.matlib import repmat
from scipy.optimize import curve_fit
from scipy.special import erf






#reading in Bosch-Hale coefficients:
with open('BH_coeffs.txt', 'r') as bh:
    cddn = []
    cddp = []
    cd3hep = []
    cdtn = []
    lines = bh.readlines()
    lnum = 0    
    for line in lines:
        if lnum > 0:
            vals = line.split()
            cddn.append(float(vals[0]))
            cddp.append(float(vals[1]))
            cd3hep.append(float(vals[2]))
            cdtn.append(float(vals[3]))
    
        lnum += 1

    Cddn = np.array(cddn)
    Cddp = np.array(cddp)
    Cd3hep = np.array(cd3hep)
    Cdtn = np.array(cdtn)
    Cmat = [Cddn, Cddp, Cd3hep, Cdtn]
    
    Cmat = np.array(Cmat)
    Bs = Cmat[:,0]
    Ms = Cmat[:,1]

    c1 = Cmat[:,2]
    c2 = Cmat[:,3]
    c3 = Cmat[:,4]
    c4 = Cmat[:,5]
    c5 = Cmat[:,6]
    c6 = Cmat[:,7]
    c7 = Cmat[:,8]


def bh_react_ddn(T):
    index = 0 
    theta = T/(1-T*(c2[index] + T*(c4[index]+T*c6[index]))/(1 + T*(c3[index] + T*(c5[index] + T*c7[index]))))
    zeta =((Bs[index]**2)/(4*theta))**(1/3)
    sigmav = c1[index]*theta*np.sqrt(zeta/(Ms[index]*T**3))*np.exp(-3*zeta)

    return sigmav

def bh_react_ddp(T):
    index = 1 
    theta = T/(1-T*(c2[index] + T*(c4[index]+T*c6[index]))/(1 + T*(c3[index] + T*(c5[index] + T*c7[index]))))
    zeta =((Bs[index]**2)/(4*theta))**(1/3)
    sigmav = c1[index]*theta*np.sqrt(zeta/(Ms[index]*T**3))*np.exp(-3*zeta)

    return sigmav

def bh_react_d3hep(T):
    index = 2 
    theta = T/(1-T*(c2[index] + T*(c4[index]+T*c6[index]))/(1 + T*(c3[index] + T*(c5[index] + T*c7[index]))))
    zeta =((Bs[index]**2)/(4*theta))**(1/3)
    sigmav = c1[index]*theta*np.sqrt(zeta/(Ms[index]*T**3))*np.exp(-3*zeta)

    return sigmav


def bh_react_dtn(T):
    index = 3 
    theta = T/(1-T*(c2[index] + T*(c4[index]+T*c6[index]))/(1 + T*(c3[index] + T*(c5[index] + T*c7[index]))))
    zeta =((Bs[index]**2)/(4*theta))**(1/3)
    sigmav = c1[index]*theta*np.sqrt(zeta/(Ms[index]*T**3))*np.exp(-3*zeta)

    return sigmav



# reading in filename and pulling up the hyades output file that we are interested in:

filename = sys.argv[1]
file_prefix = filename[0:-4]

ds = nc.Dataset(filename, mode='r')
atmnum = ds.variables['AtmNum']
atmfrc = ds.variables['AtmFrc']
fD = atmfrc[0,0]
f3He = atmfrc[0,1]


#reading in all of the variables that we care about 
nitot = np.array(ds.variables['Deni'])
netot = ds.variables['Dene']
Rs = ds.variables['R']
Rshell = Rs[:,200]
time = np.array(ds.variables['DumpTimes'])
rhos = np.array(ds.variables['Rho'])
vols = ds.variables['Vol']
tion = np.array(ds.variables['Ti'])
dt = np.array(ds.variables['Dtave'])
Dt = np.average(dt)




'''
fig, ax = plt.subplots()

ax.plot(time, Rs[:,200:300], c='g')
ax.plot(time, Rs[:,0:200], c='b')
ax.plot(time, Rshell, c = 'r')

plt.ylim([0,0.05])
'''
Time = repmat(time, 301, 1) # making a matrix version of the time variable for plotting mesh


#plotting temperature
logT = np.log(tion/np.max(tion))
'''
plt.figure()
plt.pcolormesh(np.transpose(Time), Rs, logT, cmap = 'bone', shading = 'auto')
plt.ylim(0, 0.0440)
plt.colorbar()
plt.plot(time, Rshell, c = 'r', linestyle = ':')
plt.title('Tion vs. Time')

'''
#Reaction histories
reactMat_ddn = bh_react_ddn(tion) 
reactMat_ddp = bh_react_ddp(tion) 
reactMat_d3hep = bh_react_d3hep(tion) 
reactMat_dtn = bh_react_dtn(tion) 

reaction_rate_ddp = .5*fD**2 * np.sum(nitot[:, 0:200]**2 * reactMat_ddp[:,0:200]*vols[:, 0:200], axis = 1)
reaction_rate_d3hep = fD*f3He * np.sum(nitot[:, 0:200]**2 * reactMat_d3hep[:,0:200]*vols[:,0:200], axis = 1)


ddp_profile = reaction_rate_ddp
d3hep_profile = reaction_rate_d3hep

#Defining two gaussian function for fitting to signal
#NOTE: This is probably not the best thing to fit to the signal 
#TODO: Find more appropriate functional form for the output


def two_gaussian(x, a1, b1, c1, a2, b2, c2):
    return a1*np.exp(-(x - b1)**2/(.5*c1**2)) + a2 * np.exp(-(x-b2)**2/(.5*c2**2))

def gaussian(x, a, b, c):
    return a*np.exp(-(x-b)**2/(.5*c**2))

def skew_gaussian(x, a, b, c, alpha):
    return 2*gaussian(x, a, b, c)*.5*(1+erf(alpha*x/np.sqrt(2)))

def two_skew(x, a1, b1, c1, alpha1, a2, b2, c2, alpha2):
    return skew_gaussian(x, a1, b1, c1, alpha1) + skew_gaussian(x, a2, b2, c2, alpha2)

def three_gaussian(x, a,b,c,d,e,f,g,h,i):
    return gaussian(x, a,b,c) + gaussian(x,d,e,f) + gaussian(x, g, h, i)


'''
param, cov = curve_fit(two_gaussian, time, d3hep_profile, p0 = [np.max(d3hep_profile), .8*10**-9, .1*10**-9, np.max(d3hep_profile), .9*10**-9, .1*10**-9])
a1, b1, c1, a2, b2, c2 = param
plt.plot(time, two_gaussian(time, a1, b1, c1, a2, b2, c2))
print(param)

param, cov = curve_fit(two_gaussian, time, ddp_profile, p0 = [np.max(ddp_profile), .8*10**-9, .1*10**-9, np.max(ddp_profile), .9*10**-9, .1*10**-9])
a1, b1, c1, a2, b2, c2 = param
plt.plot(time, two_gaussian(time, a1, b1, c1, a2, b2, c2))
print(param)

'''
#finding the shock trajectory:
rho_gas = rhos[:,0:200]
rho_gas_grad = np.diff(rho_gas, axis = 1)
rho0 = np.average(rhos[0,0:200])
rhor = np.trapz(rhos[:,0:200], x = Rs[:, 0:200], axis = 1)

shock_rad = []
print(rho_gas.shape)

for ind in range(rho_gas.shape[0]):
    radial_count = 0
    while radial_count < rho_gas.shape[1]-1 and np.abs(rho_gas[ind, radial_count]-rho_gas[ind, 0]) < 2*np.abs(rho_gas[ind,0]):
        radial_count += 1
    shock_rad.append(Rs[ind, radial_count])

plt.figure()

#plt.plot(time, shock_rad, c = 'red', linewidth = 2, linestyle = ':')
#plt.plot(time, Rshell, c = 'black', linestyle = ':', linewidth = 2)
#plt.plot(time, rhor/np.max(rhor), c='purple', linestyle = ':', linewidth = 2)

plt.xlim([.6*10**-9, 1*10**-9])
plt.xlabel('time (s)')
plt.ylabel('shell and shock traj.')

plt.twinx()
plt.plot(time, ddp_profile, c = 'green', linewidth = 2)
plt.plot(time, d3hep_profile, c = 'blue', linewidth = 2)

#finding shock convergence time:
shock_rad_list = list(shock_rad)
shock_conv_ind = shock_rad_list.index(min(shock_rad_list))
shock_conv_time = time[shock_conv_ind]

#finding peak compression time:
shell_rad_list = list(Rshell)
peak_comp_ind = shell_rad_list.index(min(shell_rad_list))
peak_comp_time = time[peak_comp_ind]

#finding the difference between peak compression and shock convergence times:
print('shock conv. and peak comp. times: ')
print(f'shock conv: {shock_conv_time}')
print(f'peak comp: {peak_comp_time}')

#finding the density in the capsule at shock convergence:
rho_shock_conv = np.average(rhos[shock_conv_ind, 0:200])

shock_strength = rho_shock_conv/rho0
print(f'shock strength: {shock_strength}')



#finding the time of maximum emission:
ddp_profile = list(ddp_profile)
d3hep_profile = list(d3hep_profile)

idx_max_ddp = ddp_profile.index(max(ddp_profile))
idx_max_d3hep = d3hep_profile.index(max(d3hep_profile))

idx_rise50_ddp = list(ddp_profile > .5*max(ddp_profile)).index(1)
idx_rise50_d3hep = list(d3hep_profile > .5*max(d3hep_profile)).index(1)

idx_rise90_ddp = list(ddp_profile >.9*max(ddp_profile)).index(1)
idx_rise90_d3hep = list(d3hep_profile > .9*max(d3hep_profile)).index(1)

bt_max_ddp = time[idx_max_ddp]
bt_max_d3hep = time[idx_max_d3hep]

bt_rise50_ddp = time[idx_rise50_ddp]
bt_rise50_d3hep = time[idx_rise50_d3hep]

bt_rise90_ddp = time[idx_rise90_ddp]
bt_rise90_d3hep = time[idx_rise90_d3hep]

plt.axvline(bt_max_ddp, c = 'g', linestyle = ':')
#plt.axvline(bt_rise50_ddp,c = 'g', linestyle = ':')
#plt.axvline(bt_rise90_ddp,c = 'g', linestyle = ':')

plt.axvline(bt_max_d3hep, c= 'b', linestyle = ':')
#plt.axvline(bt_rise50_d3hep, c ='b', linestyle = ':')
#plt.axvline(bt_rise90_d3hep, c = 'b', linestyle = ':')

plt.xlim([.6*10**-9, 1*10**-9])
plt.xlabel('time (s)')
plt.ylabel('emission/time')


#printing out key times in implosion that we can evaluate for separation

print(round(bt_max_ddp*10**12))
print(round(bt_max_d3hep*10**12))
plt.savefig(file_prefix +'.png')



param, cov = curve_fit(two_skew, time, d3hep_profile, p0 = [np.max(d3hep_profile), .8*10**-9, .1*10**-9,0,  np.max(d3hep_profile), .9*10**-9, .1*10**-9, 0], bounds =(-np.inf, np.inf), maxfev=10000)
a1, b1, c1, alpha1, a2, b2, c2, alpha2 = param
plt.plot(time, two_skew(time, a1, b1, c1,alpha1,  a2, b2, c2, alpha2))
plt.plot(time, skew_gaussian(time, a1, b1, c1, alpha1))
plt.plot(time, skew_gaussian(time, a2, b2, c2, alpha2))

print('shock/compression components: ')
sk1 = list(skew_gaussian(time, a1, b1, c1, alpha1))
sk2 = list(skew_gaussian(time, a2, b2, c2, alpha2))
print(a1/a2)
print(max(sk1)/max(sk2))

#contributions from shock and compresion
print('shock and compression yields')
print(f'shock component: {np.trapz(sk1, x=time)}')
print(f'compression component: {np.trapz(sk2, x = time)}')

print(f'total yield: {np.trapz(d3hep_profile, x=time)}')


print(f'ddp bang time: {bt_max_ddp}')
print(f'd3hep shock bt: {b1}')
print(f'd3hep comp bt: {b2}')

'''
param3, cov3 = curve_fit(three_gaussian, time, d3hep_profile, p0 = [np.max(d3hep_profile), .8*10**-9, .1*10**-9,  np.max(d3hep_profile), .9*10**-9, .1*10**-9, np.max(d3hep_profile), .85*10**-9, .1*10**-9 ], bounds =(-np.inf, np.inf), maxfev = 10000)

a, b,c,d,e,f,g,h,i= param3
plt.plot(time, three_gaussian(time, a, b, c, d, e, f, g, h, i))
plt.plot(time, gaussian(time, a, b, c))
plt.plot(time, gaussian(time, d,e,f))
plt.plot(time, gaussian(time, g, h, i))
'''
# convolving emission curves above with the time of flight broadening response

import scipy.ndimage as nd
plt.figure()

blur_val = round(((5*10**-12)/np.average(dt))**.5)
#blur_val = 10
pxtd_hist_ddp =nd.gaussian_filter(ddp_profile, blur_val)
pxtd_hist_d3hep = nd.gaussian_filter(d3hep_profile, blur_val)
plt.plot(time, pxtd_hist_ddp, linestyle = '-.', c = 'g')
plt.plot(time, pxtd_hist_d3hep, linestyle = '-.', c = 'b')
print(blur_val)
print(np.average(dt))

ddp_profile = list(pxtd_hist_ddp)
d3hep_profile = list(pxtd_hist_d3hep)

#finding the time of maximum emission:
ddp_profile = list(ddp_profile)
d3hep_profile = list(d3hep_profile)

idx_max_ddp = ddp_profile.index(max(ddp_profile))
idx_max_d3hep = d3hep_profile.index(max(d3hep_profile))

idx_rise50_ddp = list(ddp_profile > .5*max(ddp_profile)).index(1)
idx_rise50_d3hep = list(d3hep_profile > .5*max(d3hep_profile)).index(1)

idx_rise90_ddp = list(ddp_profile >.9*max(ddp_profile)).index(1)
idx_rise90_d3hep = list(d3hep_profile > .9*max(d3hep_profile)).index(1)

bt_max_ddp = time[idx_max_ddp]
bt_max_d3hep = time[idx_max_d3hep]

bt_rise50_ddp = time[idx_rise50_ddp]
bt_rise50_d3hep = time[idx_rise50_d3hep]

bt_rise90_ddp = time[idx_rise90_ddp]
bt_rise90_d3hep = time[idx_rise90_d3hep]

plt.axvline(bt_max_ddp, c = 'g', linestyle = ':')
#plt.axvline(bt_rise50_ddp,c = 'g', linestyle = ':')
#plt.axvline(bt_rise90_ddp,c = 'g', linestyle = ':')

plt.axvline(bt_max_d3hep, c= 'b', linestyle = ':')
#plt.axvline(bt_rise50_d3hep, c ='b', linestyle = ':')
#plt.axvline(bt_rise90_d3hep, c = 'b', linestyle = ':')

plt.xlim([.6*10**-9, 1*10**-9])
plt.xlabel('time (s)')
plt.ylabel('emission/time')


#printing out key times in implosion that we can evaluate for separation

print(round(bt_max_ddp*10**12))
print(round(bt_max_d3hep*10**12))
print(round(bt_rise90_ddp*10**12))
print(round(bt_rise90_d3hep*10**12))

plt.savefig(file_prefix + '_smooth.png')

param, cov = curve_fit(two_gaussian, time, d3hep_profile, p0 = [np.max(d3hep_profile), .8*10**-9, .1*10**-9,  np.max(d3hep_profile), .9*10**-9, .1*10**-9], bounds = (0, np.inf))
a1, b1, c1, a2, b2, c2 = param
plt.plot(time, two_gaussian(time, a1, b1, c1,  a2, b2, c2))
plt.plot(time, gaussian(time, a1, b1, c1))
plt.plot(time, gaussian(time, a2, b2, c2))
print(param)


plt.figure()

plt.plot(time, rhor)
plt.plot(time, np.average(tion[:,0:200], axis=1))







plt.show()



































