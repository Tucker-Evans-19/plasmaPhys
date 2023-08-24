# This library is for the creation of synthetic spectra and for their propagation in
# time to recreate observed spectra at a detector.
'''
# UNITS:
 * energy, MeV
 * velocity, cm/s
 * temperature, keV

'''

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import ballabio as ba

# DEFINING SOME IMPORTANT CONSTANTS:

rm_proton = 938.272 # MeV
rm_neutron  = 939.565 # MeV
c = 29.98 # cm/ns

# from Hong's thesis
#rise_time = .089
#fall_time = 1.4

# from discussion with Neel
rise_time = .02
fall_time = 1.2

def plot_spectrum(bin_centers, counts, ax = None):
    if ax == None:
        fig, ax = plt.subplots()
    ax.plot(bin_center, counts)
    ax.set_xlabel('Energy')
    ax.set_ylabel('Counts')


def synth_spec_gauss(particle_type, temperature, num_particles=5000, birth_time = 0):
    #ballabio calculation of the mean energy and standard deviation
    mean_energy, sigma_energy =  ba.ballabio_mean_std(particle_type, temperature)

    # creating synthetic population
    pop = rd.normal(mean_energy, sigma_energy, num_particles)


    return pop, mean_energy, sigma_energy
def get_pop_velocities(particle_type, pop_energies):
    
    # getting the velocity distribution:
    if particle_type == 'DTn':
        mass = rm_neutron
    elif particle_type == 'DDn':
        mass = rm_neutron
    elif particle_type == 'D3Hep':
        mass = rm_proton
    else:
        particle_type = 'None'
        mass = 0 
        print('invalid particle type')

    ER = pop_energies/mass
    #print(ER)
    beta2 = 1 - (ER + 1)**-2
    #print(beta2)
    beta = beta2**.5
    #print(beta)
    velocities = beta*c

    return velocities


def time_to_dist(dist, velocities):
    times = dist * velocities**-1
    return times
        
def pop_histogram(pop, ax = None):
    num_numparticles = pop.size
    
    if ax ==None:
        fig, ax = plt.subplots()
    energy_bin_edges = np.linspace(0, 20, min([num_particles/2, 200]))
    counts, energy_bins = ax.hist(pop, energy_bin_edges)
    ax.set_xlim([10, 20])
    return counts, energy_bins

def time_trace_at_dist(dist, particle_type, population_energies, birth_time, time_bins):
    velocities  = get_pop_velocities(particle_type, population_energies)
    times = time_to_dist(dist, velocities) + birth_time
    counts_per_time, time_bins = np.histogram(times, bins = time_bins)
    return time_bins, counts_per_time

def energy_spread_matrix(time_axis, particle_type, temperature, distance):
    # for a given type of particle at a given temperature, return a matrix A such that Ax is
    # the fluence(time) through a detector placed at a distance d from the source 
    
    # instantiating an empty version of our matrix
    num_steps = time_axis.size
    energy_spread_matrix = np.zeros((num_steps, num_steps))
    time_list = list(time_axis)
    time_step = time_axis[2] - time_axis[1]
    time_list.append(time_axis[-1] + time_step) 
    time_bin_edges = np.asarray(time_list)

    # allowing for energies 3 sigma to either side of the mean_energy
    #energy_axis = np.linspace(mean_energy - 4*sigma_energy, mean_energy + 3*sigma_energy, num_energy_bins)

    # calculating number of particles to have the energy specified (normalized spectrum)
    #num_per_energy = np.exp(-(energy_axis - mean_energy)**2/(2*sigma_energy**2))
    #num_per_energy = num_per_energy*np.sum(num_per_energy)**-1

    population_energies, mean_energy, sigma_energy = synth_spec_gauss(particle_type, temperature)
    time_bin_edges, time_trace = time_trace_at_dist(distance, particle_type, population_energies, 0, time_bin_edges)
    time_trace = time_trace/np.sum(time_trace)
    for i in range(num_steps):
        for j in range(num_steps):
            if j>=i:
                energy_spread_matrix[i,j] = time_trace[j-i]
            else:
                energy_spread_matrix[i,j] = 0

    return np.transpose(energy_spread_matrix)
    

def pxtd_conv_matrix(bin_centers):
    # creates a matrix G such that Gy is the recorded history of the PXTD detector given a fluence at time t of y
    num_counts = bin_centers.size
    G = np.zeros(shape = (num_counts, num_counts))
    for i in range(num_counts):
        for j in range(num_counts):
            if j>=i:
                G[i,j] = (1-np.exp(-(bin_centers[j] - bin_centers[i])/rise_time))*np.exp(-(bin_centers[j] - bin_centers[i])/fall_time)
            else:
                G[i,j] = 0
    G=G*np.sum(G[0, :])**-1
    
    return np.transpose(G) 

###################################################################################
'''
def synth_pxtd_trace2(dist, population_velocities, G, num_counts, bin_edges, birth_time = 0):
    # generates a trace as from one time step and a full spectrum of particles leaving at that time step
    time_to_dist(dist,population_velocities) 
    times = time_to_dist(dist, population_velocities) + birth_time
    counts, bins = np.histogram(times, bins = bin_edges)
    pxtd_trace = np.matmul(np.transpose(G), counts)
    return pxtd_trace 

def fitting_pxtd_trace_1peak(time_axis, bang_time, burn_width, G, total_particles = 100):
    # assuming that you have a gaussian emission profile generates the trace at the PXTD detector (spectra at each time estep as well)
    trace_total = np.zeros(time_bins.size-1)
    gaussian_emission = np.exp(-(time_axis - bang_time)**2/(.5*(.5*burn_width)**2))/(np.sqrt(2*np.pi)*burn_width/2)
    yield_at_time = total_particles*gaussian_emission
    
    trace_total = np.zeros(time_axis.size-1)
    
    for time_step in range(time_axis.size-1):

        time = time_axis[time_step]
        particle_yield = int(yield_at_time[time_step])
 
        pop, mean_energy, sigma_energy = sp.synth_spec_gauss('DTn', 5, num_particles = particle_yield, birth_time = time)
        velocities = sp.get_pop_velocities('DTn', pop)
        ptrace = sp.synth_pxtd_trace2(3.1, velocities, G, num_counts, time_axis, birth_time=time)
        trace_total += ptraceA
    return trace_total/np.max(trace_total)
def synthetic_pxtd_trace(dist, population_velocities, birth_time =0, bin_edges = np.linspace(0,2,num = 200)):
    time_to_dist(dist,population_velocities) 
    times = time_to_dist(dist, population_velocities) + birth_time
    counts, bins = np.histogram(times, bins = bin_edges)
    num_counts = len(bin_edges) - 1
    bin_centers = []
    for element in range(num_counts):
        bin_centers.append((bin_edges[element] + bin_edges[element+1])*.5)
    G = np.zeros(shape = (num_counts, num_counts))
    for i in range(num_counts):
        for j in range(num_counts):
            if j>=i:
                G[i,j] = (1-np.exp(-(bin_centers[j] - bin_centers[i])/rise_time))*np.exp(-(bin_centers[j] - bin_centers[i])/fall_time)
            else:
                G[i,j] = 0
    pxtd_trace = np.matmul(np.transpose(G), counts)
    return bin_centers, pxtd_trace

def fitting_pxtd_trace(time_axis, emission_profile, ion_temperature, G):
    total_particles = 100
    trace_total = np.zeros(time_axis.size-1)
    yield_at_time = total_particles*emission_profile # multiplying the normalized emission profile by the number of particles that we want.
    
    for time_step in range(time_axis.size-1):

        time = time_axis[time_step]
        particle_yield = int(yield_at_time[time_step])
 
        pop, mean_energy, sigma_energy = sp.synth_spec_gauss('DTn', ion_temperature, num_particles = particle_yield, birth_time = time)
        velocities = sp.get_pop_velocities('DTn', pop)
        ptrace = sp.synth_pxtd_trace2(3.1, velocities, G, num_counts, time_axis, birth_time=time)
        trace_total += ptrace
    return trace_total/np.max(trace_total)
'''


