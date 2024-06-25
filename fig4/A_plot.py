import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.transforms as mtransforms
import h5py
import numpy as np
from matplotlib.ticker import FuncFormatter
import sys, os, shutil

from sklearn.linear_model import LinearRegression

plt.rcParams.update({'font.size': 8})


def compute_noise_spectrum_characteristics(cluster_rate, within_cluster_rate, max_events_in_cluster, z):

    norm = sum([m**z for m in range(1,max_events_in_cluster+1)])
    m_average = sum([m * m**z / norm for m in range(1,max_events_in_cluster+1)])
    m2_average = sum([m**2 * m**z / norm for m in range(1,max_events_in_cluster+1)])

    total_rate = cluster_rate * m_average

    shot_noise_level = 2 * cluster_rate * m_average
    upper_noise_level = 2 * cluster_rate * m2_average

    f_1=0.3/(m_average * within_cluster_rate)
    f_0=0.5/(max_events_in_cluster * within_cluster_rate)

    return total_rate, upper_noise_level, shot_noise_level, f_0, f_1



filename = "whitening_test.h5"

print("Plotting " + filename)
dic = h5py.File(filename, 'r')
folder = "."

try:
    os.mkdir(folder)
except:
    pass


experiments = [dic[k] for k in sorted(list(dic.keys()))]

print(experiments)

for experiment in experiments:

    input_rate = experiment.get('input_rate')[()] 
    within_cluster_rate = experiment.get('within_cluster_rate')[()] 
    max_events_in_cluster = experiment.get('max_events_in_cluster')[()]
    z = -2.0
    norm = sum([m**z for m in range(1,max_events_in_cluster)])
    rate_correction_factor = sum([m * m**z for m in range(1,max_events_in_cluster)]) / norm

    sizes = experiment.get('sizes')[()]
    nSizes = len(sizes)
    print(sizes)
    
    for i_size in range(nSizes):
    #i_size = np.where(sizes == 1.0)[0][0]
        size = sizes[i_size]
        
        omegas = experiment.get('omegas')[()]

        vesicle_release_trains = experiment.get('vesicle_release_trains')[()][:,:,i_size] 
        vesicle_release_trains_single = experiment.get('vesicle_release_trains_single')[()][:,:,i_size]  
        spike_trains = experiment.get('spike_trains')[()][:,:,i_size]  

        vesicle_power_spectrums = experiment.get('vesicle_power_spectrums')[()][:,:,i_size]  
        vesicle_power_spectrums_mean = np.mean(vesicle_power_spectrums, axis=1) 
        vesicle_power_spectrums_mean /= np.dot(vesicle_power_spectrums_mean[:-1], omegas[1:] - omegas[:-1])
        vesicle_power_spectrums_error = np.std(vesicle_power_spectrums, axis=1)
        
        vesicle_power_spectrums_single = experiment.get('vesicle_power_spectrums_single')[()][:,:,i_size]  
        vesicle_power_spectrums_single_mean = np.mean(vesicle_power_spectrums_single, axis=1) 
        vesicle_power_spectrums_single_mean /= np.dot(vesicle_power_spectrums_single_mean[:-1], omegas[1:] - omegas[:-1])
        vesicle_power_spectrums_single_error = np.std(vesicle_power_spectrums_single, axis=1)

        spike_power_spectrums = experiment.get('spike_power_spectrums')[()][:,:,i_size]  
        spike_power_spectrums_mean = np.mean(spike_power_spectrums, axis=1) 
        spike_power_spectrums_mean /= np.dot(spike_power_spectrums_mean[:-1], omegas[1:] - omegas[:-1])
        spike_power_spectrums_error = np.std(spike_power_spectrums, axis=1)

        one_f_spectrum = 1.0 / omegas
        one_f_spectrum = one_f_spectrum / np.dot(one_f_spectrum[:-1], omegas[1:] - omegas[:-1])


        fig, axs = plt.subplots(1,1, figsize=(2.1,1.6))
        axs = np.reshape(axs,-1)
        
        
        # spectrum

        cutoff_ind = 15 # don't show the low frequency part of the spectrum
        omegas = omegas[cutoff_ind:]

        axs[0].plot(omegas, one_f_spectrum[cutoff_ind:], color='black', linewidth=0.3)

        axs[0].plot(omegas, spike_power_spectrums_mean[cutoff_ind:], color='lightcoral', label="input spikes")
        axs[0].plot(omegas, vesicle_power_spectrums_mean[cutoff_ind:], color='cornflowerblue', label="full model")
        axs[0].plot(omegas, vesicle_power_spectrums_single_mean[cutoff_ind:], color='gray', linestyle='dashed', label="single timescale")
        
        # axs[0].legend(frameon=False)


        axs[0].set_ylabel(r'spectral density')
        axs[0].set_xlabel(r'frequency [Hz]')
        axs[0].set_yscale('log')
        axs[0].set_xscale('log')
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)

        vesicle_rate = 0.0
        spike_rate = 0.0
        nRuns = vesicle_release_trains.shape[1]
        for i in range(nRuns):
            vesicle_rate += len(vesicle_release_trains[:,i])/vesicle_release_trains[-1,i]/nRuns
            spike_rate += len(spike_trains[:,i])/spike_trains[-1,i]/nRuns

        vesicle_rate_single = 0.0
        nRuns = vesicle_release_trains.shape[1]
        for i in range(nRuns):
            vesicle_rate_single += len(vesicle_release_trains_single[:,i])/vesicle_release_trains_single[-1,i]/nRuns
        
        print("Spike rate: {}Hz".format(spike_rate))
        print("Vesicle release rate: {}Hz".format(vesicle_rate))
        print("Vesicle release rate single timescale: {}Hz".format(vesicle_rate_single))

        plt.tight_layout()
        plt.savefig("whitening_size={}_input_rate={:.3f}_Hz.pdf".format(size,spike_rate))
        

        fig, axs = plt.subplots(2,1, figsize=(2.1,1.6))
        axs = np.reshape(axs,-1)


        # spikes

        min_t = 100.0
        max_t = 300.0
        ind_run = 0

        train1 = vesicle_release_trains[vesicle_release_trains[:,ind_run]>min_t,ind_run]
        train1_single = vesicle_release_trains_single[vesicle_release_trains_single[:,ind_run]>min_t,ind_run]
        data1 = train1[train1<max_t]
        
        train2 = spike_trains[spike_trains[:,ind_run]>min_t,ind_run]
        data2 = train2[train2<max_t]
        axs[0].eventplot([data1,[],data2], colors=["cornflowerblue","gray","lightcoral"], linewidths=[0.5,0.5,0.5])
        
        axs[0].plot([min_t,min_t+30],[-1.5,-1.5],color="black")
        axs[0].annotate("30 s", (min_t, -1.3))
        axs[0].set_ylabel('ves. and sp.')
        axs[0].tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
        axs[0].spines["bottom"].set_visible(False)
        axs[0].spines["right"].set_visible(False)
        axs[0].spines["top"].set_visible(False)
        axs[0].set_xlim([min_t,max_t])
        
        
        # pools

        labels = ["Primed", "Docked", "Loaded", "Recovery", "Fused"]

        import matplotlib.cm as cm

        cmap = cm.get_cmap('viridis')
        colors = [cmap(1 - i/5) for i in range(5)]

        pools = experiment.get('pool_levels')[()][(3,2,1,0,5),:,ind_run,i_size]
        ts = experiment.get('pool_levels_ts')[()][:,ind_run,i_size]
        
        axs[1].step(ts,100 * pools.T, where='post')

        for i,j in enumerate(axs[1].lines):
            j.set_color(colors[i])
            j.set_label(labels[i])

        axs[1].set_ylabel("Pool levels")
        axs[1].spines["bottom"].set_visible(False)
        axs[1].spines["right"].set_visible(False)
        axs[1].spines["top"].set_visible(False)
        axs[1].tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
        axs[1].set_xlim([min_t,max_t])
        axs[1].sharex(axs[0])

        #axs[1].legend(frameon=False)

        import matplotlib.ticker as mtick
        axs[1].yaxis.set_major_formatter(mtick.PercentFormatter())
        
        
        plt.tight_layout()
        plt.savefig("whitening_spikes_and_pools_size={}_input_rate={:.3f}_Hz.pdf".format(size,spike_rate))


