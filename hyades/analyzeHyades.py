import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import pandas as pd
import sys
import os
from numpy.matlib import repmat
from scipy.optimize import curve_fit
from scipy.special import erf

def getHyadesEmissionHistory(hyades_file):
    #reading in Bosch-Hale coefficients:
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

    # reading in filename and pulling up the hyades output file that we are interested in:
    filename = hyades_file
    file_prefix = filename[0:-4]

    ds = nc.Dataset(filename, mode='r')
    atmnum = ds.variables['AtmNum']
    atmfrc = ds.variables['AtmFrc']
    fD = atmfrc[0,0]
    f3He = atmfrc[0,1]
    fT = .005
    
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

    Time = repmat(time, 301, 1) # making a matrix version of the time variable for plotting mesh

    #plotting temperature
    logT = np.log(tion/np.max(tion))

    #Reaction histories
    reactMat_ddn = bh_react_ddn(tion) 
    reactMat_ddp = bh_react_ddp(tion) 
    reactMat_d3hep = bh_react_d3hep(tion) 
    reactMat_dtn = bh_react_dtn(tion) 

    
    rate_ddn = .5*fD**2 * np.sum(nitot[:, 0:200]**2 * reactMat_ddn[:,0:200]*vols[:, 0:200], axis = 1)
    rate_ddp = .5*fD**2 * np.sum(nitot[:, 0:200]**2 * reactMat_ddp[:,0:200]*vols[:, 0:200], axis = 1)
    rate_d3hep = fD*f3He * np.sum(nitot[:, 0:200]**2 * reactMat_d3hep[:,0:200]*vols[:,0:200], axis = 1)
    rate_dtn = fD*fT * np.sum(nitot[:, 0:200]**2 * reactMat_dtn[:, 0:200]*vols[:, 0:200], axis = 1)

    return rate_ddn, rate_ddp, rate_d3hep, rate_dtn 
    
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

#finding shock convergence time:
shock_rad_list = list(shock_rad)
shock_conv_ind = shock_rad_list.index(min(shock_rad_list))
shock_conv_time = time[shock_conv_ind]

#finding peak compression time:
shell_rad_list = list(Rshell)
peak_comp_ind = shell_rad_list.index(min(shell_rad_list))
peak_comp_time = time[peak_comp_ind]

rho_shock_conv = np.average(rhos[shock_conv_ind, 0:200])

with open(file_prefix, '.txt') as output:
    for element in range(len(time)):
        output.writelines(time[element] + ', ' + 


'''
