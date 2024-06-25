import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.transforms as mtransforms
import h5py
import numpy as np
from matplotlib.ticker import FuncFormatter
import sys, os, shutil

from sklearn.linear_model import LinearRegression

plt.rcParams.update({'font.size': 8})


def main():
    filename = "input_output.h5"

    print("Plotting " + filename)
    dic = h5py.File(filename, 'r')
    folder = "."

    vesicle_per_spike_1_f = dic.get('vesicle_per_spike_1_f')[()]
    vesicle_per_spike_1_f_single = dic.get('vesicle_per_spike_1_f_single')[()]
    spike_rate_1_f = dic.get('spike_rate_1_f')[()]
    vesicle_per_spike_regular = dic.get('vesicle_per_spike_regular')[()]
    vesicle_per_spike_regular_single = dic.get('vesicle_per_spike_regular_single')[()]
    spike_rate_regular = dic.get('spike_rate_regular')[()]
    vesicle_per_spike_poisson = dic.get('vesicle_per_spike_poisson')[()]
    vesicle_per_spike_poisson_single = dic.get('vesicle_per_spike_poisson_single')[()]
    spike_rate_poisson = dic.get('spike_rate_poisson')[()]

    sizes = dic.get('sizes')[()]
    nSizes = len(sizes)
    nExperiments = vesicle_per_spike_1_f.shape[2]

    fig, axs = plt.subplots(1, nSizes, figsize=(nSizes*2.1,2.1))#, sharey=True)
    
    for i_size, size in enumerate(sizes):   
        
        axs[i_size].plot(spike_rate_1_f[0,i_size,:], vesicle_per_spike_1_f[0,i_size, :], color='black')
        axs[i_size].plot(spike_rate_regular[0,i_size,:], vesicle_per_spike_regular[0,i_size, :], color='black', linestyle='--')
        axs[i_size].plot(spike_rate_poisson[0,i_size,:], vesicle_per_spike_poisson[0,i_size, :], color='black', linestyle=':')
        axs[i_size].set_ylabel(r'vesicles per spike')
        axs[i_size].set_xlabel(r'input rate [Hz]')
        axs[i_size].set_title(r'size {}x'.format(size))
        
        # axs[0].set_yscale('log')
        # axs[0].set_xscale('log')
        axs[i_size].spines["top"].set_visible(False)
        axs[i_size].spines["right"].set_visible(False)
        
    axs[2].legend(['1F', 'regular', 'poisson'], loc='upper right', frameon=False)

    plt.tight_layout()
    plt.savefig("input_output.pdf")
    

    fig, axs = plt.subplots(1, nSizes, figsize=(nSizes*2.1,2.1))#, sharey=True)
    
    for i_size, size in enumerate(sizes):   
        
        axs[i_size].plot(spike_rate_1_f[0,i_size,:], vesicle_per_spike_1_f_single[0,i_size, :], color='black')
        axs[i_size].plot(spike_rate_regular[0,i_size,:], vesicle_per_spike_regular_single[0,i_size, :], color='black', linestyle='--')
        axs[i_size].plot(spike_rate_poisson[0,i_size,:], vesicle_per_spike_poisson_single[0,i_size, :], color='black', linestyle=':')
        axs[i_size].set_ylabel(r'vesicles per spike')
        axs[i_size].set_xlabel(r'input rate [Hz]')
        axs[i_size].set_title(r'size {}x'.format(size))
        
        # axs[0].set_yscale('log')
        # axs[0].set_xscale('log')
        axs[i_size].spines["top"].set_visible(False)
        axs[i_size].spines["right"].set_visible(False)
        
    axs[2].legend(['1F', 'regular', 'poisson'], loc='upper right', frameon=False)

    plt.tight_layout()
    plt.savefig("input_output_single.pdf")



main()
