import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import pandas as pd
import sys
import os
from numpy.matlib import repmat
from scipy.optimize import curve_fit

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


