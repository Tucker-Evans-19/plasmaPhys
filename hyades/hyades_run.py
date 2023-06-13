import numpy as np
import matplotlib.pyplot
import netCDF4 as nc

class hyades_run:
    def __init__(self, filename, name=''):

        # reading in filename and pulling up the hyades output file that we are interested in:
        self.file_prefix = filename[0:-4]
        self.name = name
        
        if self.name == '':
            self.name = self.file_prefix
        
        ds = nc.Dataset(filename, mode='r')
        self.atmnum = ds.variables['AtmNum']
        atmfrc = ds.variables['AtmFrc']
        self.fD = atmfrc[0,0]
        self.f3He = atmfrc[0,1]
        self.fT = 0.005
        #NOTE: Fix this to be not a random number
        #reading in all of the variables that we care about 
        self.nitot = np.array(ds.variables['Deni'])
        self.netot = ds.variables['Dene']
        self.Rs = ds.variables['R']
        self.Rshell = self.Rs[:,200]
        self.time = np.array(ds.variables['DumpTimes'])
        self.rhos = np.array(ds.variables['Rho'])
        self.vols = ds.variables['Vol']
        self.tion = np.array(ds.variables['Ti'])
        self.dt = np.array(ds.variables['Dtave'])

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
    
    def get_emission_histories(self):
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

        #Reaction histories
        reactMat_ddn = bh_react_ddn(self.tion) 
        reactMat_ddp = bh_react_ddp(self.tion) 
        reactMat_d3hep = bh_react_d3hep(self.tion) 
        reactMat_dtn = bh_react_dtn(self.tion) 
        
        rate_ddn = .5*self.fD**2 * np.sum(self.nitot[:, 0:200]**2 * reactMat_ddn[:,0:200]*self.vols[:, 0:200], axis = 1)
        rate_ddp = .5*self.fD**2 * np.sum(self.nitot[:, 0:200]**2 * reactMat_ddp[:,0:200]*self.vols[:, 0:200], axis = 1)
        rate_d3hep = self.fD*self.f3He * np.sum(self.nitot[:, 0:200]**2 * reactMat_d3hep[:,0:200]*self.vols[:,0:200], axis = 1)
        rate_dtn = self.fD*self.fT * np.sum(self.nitot[:, 0:200]**2 * reactMat_dtn[:, 0:200]*self.vols[:, 0:200], axis = 1)
        return rate_ddn, rate_ddp, rate_d3hep, rate_dtn 
