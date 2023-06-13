import numpy as np
import matplotlib.pyplot as plt
from hyades_run import hyades_run
import os

files = os.listdir()
runs =[]
for file in files:
    if file[-4:] == '.cdf':
        runs.append(hyades_run(file))


plt.figure()
legends = []

for run in runs:
    average_temp = np.average(run.tion[:, 0:200], axis =1)
    plt.plot(run.time, average_temp)
    legends.append(run.file_prefix)

plt.title('Average Temperature vs. Time') 
plt.xlabel('Time (s)')
plt.ylabel('Temperature (keV)')


plt.figure()
legends = []

for run in runs: 
    plt.plot(run.time, run.Rs[:,200])
    legends.append(run.file_prefix)

plt.title('Shell radius vs. Time') 
plt.xlabel('Time (s)')
plt.ylabel('Shell Radius (cm)')


plt.figure()
legends = []

for run in runs: 
    plt.plot(run.time, np.average(run.rhos[:,0:200], axis = 1))
    legends.append(run.file_prefix)

plt.title('Average Density vs. Time') 
plt.xlabel('Time (s)')
plt.ylabel('Average Density (g/cm**2)')


plt.show()
