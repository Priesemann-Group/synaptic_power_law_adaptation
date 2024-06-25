import matplotlib.pyplot as plt
import h5py
import numpy as np
import sys, os, shutil


plt.rcParams.update({'font.size': 8})


filename = "average_rate.h5"

print("Plotting " + filename)
dic = h5py.File(filename, 'r')
folder = "."
#shutil.rmtree(folder, ignore_errors=True)

try:
    os.mkdir(folder)
except:
    pass

experiments = [dic[k] for k in sorted(list(dic.keys()))]

for experiment in experiments:

    sizes = experiment.get('sizes')[()]
    nSizes = len(sizes)
    print(sizes)
    
    i_size = np.where(sizes == 1.0)[0][0]
    size = sizes[i_size]
    

    vesicle_release_trains = experiment.get('vesicle_release_trains')[()][:,:,i_size] 
    vesicle_release_trains_single = experiment.get('vesicle_release_trains_single')[()][:,:,i_size]  
    spike_trains = experiment.get('spike_trains')[()][:,:,i_size]  

    print("release rate full: {}".format(np.sum(vesicle_release_trains[:,0] > 0)/np.max(vesicle_release_trains[:,0])))

    print("release rate single: {}".format(np.sum(vesicle_release_trains_single[:,0] > 0)/np.max(vesicle_release_trains_single[:,0])))


    fig, axs = plt.subplots(2,1, figsize=(2.1,1.6))
    axs = np.reshape(axs,-1)


    # spikes

    min_t = 4000.0
    max_t = min_t + 15 * 1000.0
    ind_run = 0

    train1 = vesicle_release_trains[vesicle_release_trains[:,ind_run]>min_t,ind_run]
    train1_single = vesicle_release_trains_single[vesicle_release_trains_single[:,ind_run]>min_t,ind_run]
    data1 = train1[train1<max_t]
    data1_single = train1_single[train1_single<max_t]
    
    train2 = spike_trains[spike_trains[:,ind_run]>min_t,ind_run]
    data2 = train2[train2<max_t]
    axs[0].eventplot([data1,[],data1_single,[],data2], colors=["cornflowerblue","gray","gray","gray","lightcoral"], linewidths=0.1)
    
    axs[0].set_ylabel('ves. and sp.')
    axs[0].tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
    axs[0].spines["bottom"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[0].spines["top"].set_visible(False)
    axs[0].set_xlim([min_t,max_t])
    
    
    rates

    # gaussian kernel
    # compute average rates with gaussian kernel
    n_runs = vesicle_release_trains.shape[1]
    window = 300
    gauss_kernel = lambda x, t: np.exp(-(x-t)**2/2.0/window**2) / np.sqrt(2*np.pi) / window
    # gauss_kernel = lambda x, t: (np.abs(x-t) < window/2) / window
    step = 50
    nBins = int((max_t - min_t) / step)
    rates = np.zeros((n_runs,nBins))
    rates_single = np.zeros((n_runs,nBins))
    rates_spikes = np.zeros((n_runs,nBins))
    ts = np.zeros(nBins)

    for ind_run in range(n_runs):
        print("{} of {}".format(ind_run, n_runs), end="\r", flush=True)
        train1 = vesicle_release_trains[vesicle_release_trains[:,ind_run]>min_t,ind_run]
        train1_single = vesicle_release_trains_single[vesicle_release_trains_single[:,ind_run]>min_t,ind_run]
        data1 = train1[train1<max_t]
        data1_single = train1_single[train1_single<max_t]
        
        train2 = spike_trains[spike_trains[:,ind_run]>min_t,ind_run]
        data2 = train2[train2<max_t]
        for i in range(nBins):
            ts[i] = min_t + i * step
            rates[ind_run,i] += np.sum(gauss_kernel(train1, ts[i]))
            rates_single[ind_run,i] += np.sum(gauss_kernel(train1_single, ts[i]))
            rates_spikes[ind_run,i] += np.sum(gauss_kernel(train2, ts[i]))
            ts[i] += 0.5 * window

    rates_avg = np.mean(rates, axis=0)
    rates_single_avg = np.mean(rates_single, axis=0)
    rates_spikes_avg = np.mean(rates_spikes, axis=0)

    rates_spikes_pred = rates_spikes_avg / np.mean(rates_spikes_avg) * np.mean(rates_avg)

    axs[1].plot(ts, rates_spikes_pred, color="lightcoral", label="input spikes", linewidth=0.5)
    axs[1].plot(ts, rates_avg, color="cornflowerblue", label="full model")
    axs[1].plot(ts, rates_single_avg, color="gray", linestyle='dashed', label="single timescale")
    axs[1].set_ylabel('rate [Hz]')
    axs[1].tick_params(left = True, right = False , labelleft = True, labelbottom = False, bottom = False)
    axs[1].spines["right"].set_visible(False)
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["bottom"].set_visible(False)
    # axs[1].legend(frameon=False)
    axs[1].plot([ts[0],ts[0]+30*60],[0.01,0.01],color="black")
    axs[1].annotate("30 min", (ts[0], 0.01))
    axs[1].set_xlim([ts[0],max_t])
    axs[0].set_xlim([ts[0],max_t])
    # axs[1].set_ylim([5*10**-3,10])
    # axs[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig("fig4C.pdf")
    
    




    # plot spike rate vs vesicle release rate
    min_t = 0.0
    max_t = np.max(vesicle_release_trains)

    # compute average rates with window kernel
    n_runs = vesicle_release_trains.shape[1]
    window = 50
    step = 50
    nBins = int((max_t - min_t) / step)
    print(nBins)
    rates = np.zeros((n_runs,nBins))
    rates_single = np.zeros((n_runs,nBins))
    rates_spikes = np.zeros((n_runs,nBins))
    ts = np.zeros(nBins)

    for ind_run in range(n_runs):
        train1 = vesicle_release_trains[vesicle_release_trains[:,ind_run]>min_t,ind_run]
        train1_single = vesicle_release_trains_single[vesicle_release_trains_single[:,ind_run]>min_t,ind_run]
        data1 = train1[train1<max_t]
        data1_single = train1_single[train1_single<max_t]
        
        train2 = spike_trains[spike_trains[:,ind_run]>min_t,ind_run]
        data2 = train2[train2<max_t]
        for i in range(nBins):
            if i % 100 == 0:
                print("Run {} of {} \t {}%".format(ind_run, n_runs,100*i/nBins), end="\r", flush=True)
            ts[i] = min_t + i * step
            rates[ind_run,i] += np.sum((train1 > ts[i]) & (train1 < ts[i] + window)) / window
            rates_single[ind_run,i] += np.sum((train1_single > ts[i]) & (train1_single < ts[i] + window)) / window
            rates_spikes[ind_run,i] += np.sum((train2 > ts[i]) & (train2 < ts[i] + window)) / window
            ts[i] += 0.5 * window

    rates_avg = np.mean(rates, axis=0)
    rates_single_avg = np.mean(rates_single, axis=0)
    rates_spikes_avg = np.mean(rates_spikes, axis=0)


    fig, ax = plt.subplots(1,1, figsize=(2.1,1.6))

    X = np.logspace(np.log10(0.2),  np.log10(50), num=40)# np.log10(np.max(rates_spikes_avg)), num=50)
    print(X)
    Y = np.zeros(X.size)
    Y_std = np.zeros(X.size)
    Y_single = np.zeros(X.size)
    Y_single_std = np.zeros(X.size)
    for i in range(X.size-1):
        inds = np.where((rates_spikes_avg > X[i]) & (rates_spikes_avg < X[i+1]))
        Y[i] = np.mean(rates_avg[inds] / rates_spikes_avg[inds])
        Y_std[i] = np.std(rates_avg[inds] / rates_spikes_avg[inds])
        Y_single[i] = np.mean(rates_single_avg[inds] / rates_spikes_avg[inds])
        Y_single_std[i] = np.std(rates_single_avg[inds] / rates_spikes_avg[inds])
    inds = np.where(rates_spikes_avg > X[-1])
    Y[-1] = np.std(rates_avg[inds] / rates_spikes_avg[inds])
    Y_single[-1] = np.mean(rates_single_avg[inds] / rates_spikes_avg[inds])
    Y_single[-1] = np.mean(rates_single_avg[inds] / rates_spikes_avg[inds])
    Y_single_std[-1] = np.std(rates_single_avg[inds] / rates_spikes_avg[inds])
    print(Y)
    # print firing rates at bin closest to 1 Hz
    ind_1 = np.argmin((X - 1.0)**2)
    print("full model release rate at {} Hz: ".format(X[ind_1]))
    print(Y[ind_1])
    print("single timescale release rate at {} Hz: ".format(X[ind_1]))
    print(Y_single[ind_1])
    ax.plot(X, Y, color="cornflowerblue", label="full model")
    ax.fill_between(X, Y-Y_std, Y+Y_std, color="cornflowerblue", linewidth=0, alpha=0.3)
    ax.plot(X, Y_single, color="gray", linestyle='dashed', label="single timescale")
    ax.fill_between(X, Y_single-Y_single_std, Y_single+Y_single_std, color="gray", linewidth=0, alpha=0.3)
    # ax.legend(frameon=False)            
    ax.set_ylabel('ves. per spike')
    ax.set_xlabel('spike rate [Hz]')
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig("fig4C_rate_rate.pdf".format(size))

