import matplotlib.pyplot as plt
import h5py
import numpy as np
import sys, os, shutil


plt.rcParams.update({'font.size': 8})

filename = "../fig4/whitening_test.h5"

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
    
    vesicle_release_trains = experiment.get('vesicle_release_trains')[()]
    vesicle_release_trains_single = experiment.get('vesicle_release_trains_single')[()]
    spike_trains = experiment.get('spike_trains')[()] 



    fig = plt.figure(figsize=(2*2.0,2*1.0))
    axs = []
    axs.append(plt.subplot(221))
    axs.append(plt.subplot(223, sharex=axs[0]))
    axs.append(plt.subplot(222))
    axs.append(plt.subplot(224, sharex=axs[2]))
    





    # 1

    min_t = 26
    max_t = 26.7
    ind_run = 0
    ind_size = 1

    train1 = vesicle_release_trains[vesicle_release_trains[:,ind_run,ind_size]>min_t,ind_run,ind_size]
    train1_single = vesicle_release_trains_single[vesicle_release_trains_single[:,ind_run,ind_size]>min_t,ind_run,ind_size]
    data1 = train1[train1<max_t]
    print(data1)
    train2 = spike_trains[spike_trains[:,ind_run,ind_size]>min_t,ind_run,ind_size]
    data2 = train2[train2<max_t]
    axs[0].eventplot(np.array([data1,[],data2]), colors=["cornflowerblue","gray","lightcoral"])
    
    
    axs[0].plot([min_t,min_t+0.1],[-1.5,-1.5],color="black")
    axs[0].annotate("100 ms", (min_t, -1.3))
    axs[0].set_ylabel('ves. and spikes')
    axs[0].tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
    axs[0].spines["bottom"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[0].spines["top"].set_visible(False)
    axs[0].set_xlim([min_t,max_t])
    
    # 2

    labels = ["Primed", "Docked", "Loaded", "Recovery", "Fused"]

    import matplotlib.cm as cm

    cmap = cm.get_cmap('viridis')
    colors = [cmap(1 - i/5) for i in range(5)]

    pools = experiment.get('pool_levels')[()][(3,2,1,0,5),:,ind_run,ind_size]
    ts = experiment.get('pool_levels_ts')[()][:,ind_run,ind_size]
    
    #inds = np.where(np.logical_and(ts > min_t, ts < max_t))[0]

    #axs[2].step(ts[inds],100 * pools[:,inds].T)
    axs[1].step(ts,100 * pools.T, where='post')


    for i,j in enumerate(axs[1].lines):
        j.set_color(colors[i])
        j.set_label(labels[i])

    axs[1].set_ylabel("Pool levels")
    axs[1].spines["bottom"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].spines["top"].set_visible(False)
    axs[1].tick_params(left = True, right = False , labelleft = True, labelbottom = False, bottom = False)
    axs[1].set_xlim([min_t,max_t])

    axs[1].legend(frameon=False)

    import matplotlib.ticker as mtick
    axs[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # 3
    
    min_t = 20.0
    max_t = 100.0
    ind_run = 0

    train1 = vesicle_release_trains[vesicle_release_trains[:,ind_run,ind_size]>min_t,ind_run,ind_size]
    train1_single = vesicle_release_trains_single[vesicle_release_trains_single[:,ind_run,ind_size]>min_t,ind_run,ind_size]
    data1 = train1[train1<max_t]
    print(data1)
    train2 = spike_trains[spike_trains[:,ind_run,ind_size]>min_t,ind_run,ind_size]
    data2 = train2[train2<max_t]
    axs[2].eventplot(np.array([data1,[],data2]), colors=["cornflowerblue","gray","lightcoral"])
    
    
    axs[2].plot([min_t,min_t+10],[-1.5,-1.5],color="black")
    axs[2].annotate("10 s", (min_t, -1.3))
    axs[2].set_ylabel('ves. and spikes')
    axs[2].tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
    axs[2].spines["bottom"].set_visible(False)
    axs[2].spines["right"].set_visible(False)
    axs[2].spines["top"].set_visible(False)
    axs[2].set_xlim([min_t,max_t])
    
    # 4

    labels = ["Primed", "Docked", "Loaded", "Recovery", "Fused"]

    import matplotlib.cm as cm

    cmap = cm.get_cmap('viridis')
    colors = [cmap(1 - i/5) for i in range(5)]

    pools = experiment.get('pool_levels')[()][(3,2,1,0,5),:,ind_run,ind_size]
    ts = experiment.get('pool_levels_ts')[()][:,ind_run,ind_size]
    
    #inds = np.where(np.logical_and(ts > min_t, ts < max_t))[0]

    #axs[2].step(ts[inds],100 * pools[:,inds].T)
    axs[3].step(ts,100 * pools.T, where='post')


    for i,j in enumerate(axs[3].lines):
        j.set_color(colors[i])
        j.set_label(labels[i])

    axs[3].set_ylabel("Pool levels")
    axs[3].spines["bottom"].set_visible(False)
    axs[3].spines["right"].set_visible(False)
    axs[3].spines["top"].set_visible(False)
    axs[3].tick_params(left = True, right = False , labelleft = True, labelbottom = False, bottom = False)
    axs[3].set_xlim([min_t,max_t])

    axs[3].legend(frameon=False)

    import matplotlib.ticker as mtick
    axs[3].yaxis.set_major_formatter(mtick.PercentFormatter())
    
    vesicle_rate = 0.0
    spike_rate = 0.0
    nRuns = vesicle_release_trains.shape[1]
    for i in range(nRuns):
        vesicle_rate += len(vesicle_release_trains[:,i])/vesicle_release_trains[-1,i]/nRuns
        spike_rate += len(spike_trains[:,i])/spike_trains[-1,i]/nRuns
    
    print("Spike rate: {}Hz".format(spike_rate))
    print("Vesicle release rate: {}Hz".format(vesicle_rate))
    print("Vesicle release rate single: {}Hz".format(len(train1_single)/train1_single[-1]))

    plt.tight_layout()
    plt.savefig("pool_levels_rate={:.3f}_Hz.pdf".format(spike_rate[0]))
    #plt.show() 


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
