import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.transforms as mtransforms
import h5py
import numpy as np
from matplotlib.ticker import FuncFormatter
import sys, os, shutil

from sklearn.linear_model import LinearRegression

plt.rcParams.update({'font.size': 8})

n_boot = 10000
cil = 0.99
def conf(x):
    mn = np.mean(x)
    samples = np.random.choice(x, size=(len(x), n_boot), replace=True)
    means = np.sort(np.apply_along_axis(np.mean, 0, samples))
    ind1 = int((1 - cil)/2*n_boot)
    ind2 = int((1 - (1 - cil)/2)*n_boot)
    return (mn - means[ind1], means[ind2] - mn)

def autoscale_based_on(ax, lines):
    ax.dataLim = mtransforms.Bbox.unit()
    for line in lines:
        xy = np.vstack(line.get_data()).T
        ax.dataLim.update_from_data_xy(xy, ignore=False)
    ax.autoscale_view()

def main():

    filename = "../fig2/depression_test_50Hz.h5"

    print("Plotting " + filename)
    dic = h5py.File(filename, 'r')
    folder = "."
    

    ves_per_spike = dic.get('ves_per_spike')[()] 
    ves_per_spike_single = dic.get('ves_per_spike_single')[()] 
    ps = dic.get('ps')[()]
    sizes = dic.get('sizes')[()]
    test_times = dic.get('test_times')[()]


    fig, axs = plt.subplots(len(ps),len(sizes), figsize=(len(sizes)*2.1,len(ps)*1.6))#, sharey=True)
    axs = np.reshape(axs,(len(ps),len(sizes)))

    for i_size in range(len(sizes)):
        for i_p in range(len(ps)):

            ves_per_spike_single_mean = np.mean(ves_per_spike_single[:,:,i_p,i_size], axis=1)
            ves_per_spike_single_error = np.apply_along_axis(conf, 1, ves_per_spike_single[:,:,i_p,i_size])
            
            axs[i_p,i_size].errorbar(test_times, ves_per_spike_single_mean, ves_per_spike_single_error.T, 
                                     linestyle='None', marker='o', zorder=1, markersize=2.0, color='gray')
 
            ves_per_spike_mean = np.mean(ves_per_spike[:,:,i_p,i_size], axis=1)
            ves_per_spike_error = np.apply_along_axis(conf, 1, ves_per_spike[:,:,i_p,i_size])

            axs[i_p,i_size].errorbar(test_times, ves_per_spike_mean, ves_per_spike_error.T, 
                                     linestyle='None', marker='o', zorder=1, markersize=2.0, color='cornflowerblue')
            
            axs[i_p,i_size].set_ylabel(r'vesicles per spike')
            axs[i_p,i_size].set_xlabel(r'stimulation time [s]')
            axs[i_p,i_size].set_yscale('log')
            axs[i_p,i_size].set_xscale('log')
            axs[i_p,i_size].set_title('size={}x, p={}'.format(sizes[i_size], ps[i_p]))
            #axs[i_p].set_xticks([0.1,1,10])


            #Y = np.log(ves_per_spike_mean[0:-4,i_p].reshape(-1,1))
            #X = np.log(test_times[0:-4].reshape(-1,1))
            
            #axs[i_p].plot(np.exp(X), np.exp(LinearRegression().fit(X, Y).predict(X)), color='black', linewidth=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig("depression_test.pdf")
    #plt.show()



main()
