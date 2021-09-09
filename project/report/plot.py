import numpy as np
import matplotlib.pyplot as plt

omp = np.genfromtxt('runtime.csv', skip_header=1, delimiter=',')[1:10,1:6] * 10**-6
omp_coarse = np.genfromtxt('runtime.csv', skip_header=1, delimiter=',')[1:10,6:10] * 10**-6
serial = np.genfromtxt('runtime.csv', skip_header=1, delimiter=',')[0,1:6] * 10**-6
serial_coarse = np.genfromtxt('runtime.csv', skip_header=1, delimiter=',')[0,6:10] * 10**-6
acc = np.genfromtxt('runtime.csv', skip_header=1, delimiter=',')[10,1:6] * 10**-6
acc_coarse = np.genfromtxt('runtime.csv', skip_header=1, delimiter=',')[10,6:10] * 10**-6
x = np.array((1,2,3,5,10,20,30,40,50))

type = 'Speedup'
model = 'coarse'

title = np.array(('Step 1', 'Step 2', 'Step 3', 'Step 4'))
if model == 'coarse':
    title[3] = 'Total'

'''    
for i in range(4):
    if model == 'coarse':
        omp = omp_coarse
        serial = serial_coarse
        acc = acc_coarse
        
    plt.subplot(2, 2, i+1)
    
    if type == 'Speedup':
        plt.plot(x, serial[i]/omp[:,i], marker='o', color='k')  
        plt.axhline(1, linewidth=1, color='b', linestyle='--', label='Serial')
        plt.axhline(serial[i]/acc[i], linewidth=1, color='r', linestyle='--', label='OpenACC') 
    else:
        plt.plot(x, omp[:,i], marker='o', color='k') 
        plt.axhline(serial[i], linewidth=1, color='b', linestyle='--', label='Serial')
        plt.axhline(acc[i], linewidth=1, color='r', linestyle='--', label='OpenACC') 
      
    plt.title(title[i])
    if i//2 == 1:
        plt.xlabel('Number of OpenMP threads')
    if i%2 == 0:
        plt.ylabel(type)
    if i == 0:
        plt.legend()    
plt.subplots_adjust(hspace=0.3)
plt.subplots_adjust(wspace=0.3)
#plt.savefig(type+'_'+model+'.pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()

'''
for i in range(4):
    plt.subplot(2, 2, i+1)
    if i == 3:
        plt.plot(x, serial[4]/omp[:,4], marker='o', color='tab:orange', label='N = 24149')
        plt.axhline(serial[4]/acc[4], linewidth=1, color='tab:orange', linestyle='--')
    else:
        plt.plot(x, serial[i]/omp[:,i], marker='o', color='tab:orange', label='N = 24149')
        plt.axhline(serial[i]/acc[i], linewidth=1, color='tab:orange', linestyle='--')
    plt.plot(x, serial_coarse[i]/omp_coarse[:,i], marker='o', color='tab:blue', label='N = 6384')
    plt.axhline(serial_coarse[i]/acc_coarse[i], linewidth=1, color='tab:blue', linestyle='--')
    
    plt.title(title[i])
    if i//2 == 1:
        plt.xlabel('Number of OpenMP threads')
    if i%2 == 0:
        plt.ylabel(type)
    if i == 0:
        plt.legend()
plt.subplots_adjust(hspace=0.3)
plt.subplots_adjust(wspace=0.3)
plt.savefig(type+'_'+model+'.pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()

    