import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import h5py
import numpy as np
import sys, os, shutil


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

def get_euler_solution():
    # Define the parameters
    TAU_RECOVER = 3.0
    TAU_LOAD = 0.5
    TAU_DOCK = 1.0
    TAU_PRIME = 0.1
    MAX_RECOVERY = 40
    MAX_LOADED = 40
    MAX_DOCKED = 20
    MAX_PRIMED = 4
    MAX_FUSED = 104
    RELEASE_PROBABILITY = 0.2


    # Define the differential equations
    def equations(y, t):
        recovery, loaded, docked, primed, fused = y

        fused_to_recovery = 1 / TAU_RECOVER * (fused > 0) * (recovery <= MAX_RECOVERY)
        recovery_to_loaded = 4 / TAU_LOAD * (recovery / MAX_RECOVERY) * (1 - loaded / MAX_LOADED)
        loaded_to_docked = 4 / TAU_DOCK * (loaded / MAX_LOADED) * (1 - docked / MAX_DOCKED)
        docked_to_primed = 4 / TAU_PRIME * (docked / MAX_DOCKED) * (1 - primed / MAX_PRIMED)

        drecovery = fused_to_recovery - recovery_to_loaded
        dloaded = recovery_to_loaded - loaded_to_docked
        ddocked = loaded_to_docked - docked_to_primed
        dprimed = docked_to_primed
        dfused = - fused_to_recovery
        
        return [drecovery, dloaded, ddocked, dprimed, dfused]

    # define spikes
    input_rate = 50 # Hz

    # Define the initial conditions and time interval
    y0 = [40, 40, 20, 4, 0]
    t0 = 1.0 / input_rate
    tf = 500
    dt = 0.0005

    # Solve the differential equations using the Euler method
    t = np.arange(t0, tf+dt, dt)
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0, :] = y0


    vesicles_per_spike = np.zeros(len(t))
    
    for i in range(n-1):
        y[i+1, :] = y[i, :] + dt * np.array(equations(y[i, :], t[i]))
        vesicles_per_spike[i] = RELEASE_PROBABILITY * y[i,3]
        y[i+1, :] += dt * input_rate * vesicles_per_spike[i] * np.array([0, 0, 0, -1, 1])

    ind_start, ind_end = int(0.1/dt), -2
    return t[ind_start:ind_end], y[ind_start:ind_end,:], vesicles_per_spike[ind_start:ind_end]

def autoscale_based_on(ax, lines):
    ax.dataLim = mtransforms.Bbox.unit()
    for line in lines:
        xy = np.vstack(line.get_data()).T
        ax.dataLim.update_from_data_xy(xy, ignore=False)
    ax.autoscale_view()


filename = "compare_discrete_continuous.h5"

print("Plotting " + filename)
dic = h5py.File(filename, 'r')
folder = "."
#shutil.rmtree(folder, ignore_errors=True)

try:
    os.mkdir(folder)
except:
    pass


# 1

fig, axs = plt.subplots(1,1, figsize=(2.1,1.6))
axs = np.reshape(axs,-1)


ves_per_spike = dic.get('ves_per_spike')[()] 
ves_per_spike_mean = np.mean(ves_per_spike, axis=1)
ves_per_spike_single = dic.get('ves_per_spike_single')[()] 
ves_per_spike_single_mean = np.mean(ves_per_spike_single, axis=1)
ves_per_spike_single_slow = dic.get('ves_per_spike_single_slow')[()] 
ves_per_spike_single_slow_mean = np.mean(ves_per_spike_single_slow, axis=1)
ps = dic.get('ps')[()]
sizes = dic.get('sizes')[()]
test_times = dic.get('test_times')[()]

i_p = np.where(ps == 0.2)[0][0]
i_size = np.where(sizes == 1.0)[0][0]

ves_per_spike_error = np.apply_along_axis(conf, 1, ves_per_spike[:,:,i_p,i_size])

axs[0].errorbar(test_times, ves_per_spike_mean[:,i_p,i_size], ves_per_spike_error.T, 
                linestyle='None', marker='o', zorder=2, markersize=2.0, label="full model", 
                color='cornflowerblue')

ves_per_spike_single_error = np.apply_along_axis(conf, 1, ves_per_spike_single[:,:,i_p,i_size])

axs[0].errorbar(test_times, ves_per_spike_single_mean[:,i_p,i_size], ves_per_spike_single_error.T, 
                linestyle='None', marker='o', zorder=1, markersize=2.0, label="single timescale (fast)",
                color='gray')

ves_per_spike_single_slow_error = np.apply_along_axis(conf, 1, ves_per_spike_single_slow[:,:,i_p,i_size])

axs[0].errorbar(test_times, ves_per_spike_single_slow_mean[:,i_p,i_size], ves_per_spike_single_slow_error.T, 
                linestyle='None', marker='s', zorder=1, markersize=2.0, label="single timescale (slow)",
                color='gray')

#t, y, vesicles_per_spike = get_euler_solution()
#axs[0].plot(t, vesicles_per_spike, zorder=0, color='black', linewidth=0.3, label="deterministic model")

axs[0].set_ylabel(r'ves. per spike')
axs[0].set_xlabel(r'stimulation time [s]')
axs[0].set_yscale('log')
axs[0].set_xscale('log')
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
axs[0].legend(frameon=False)

axins = axs[0].inset_axes([0.6, 0.6, 0.4, 0.4])

# plot some data on the inset plot
axins.plot(test_times, ves_per_spike_mean[:,i_p,i_size], 
                linestyle='-', marker='', zorder=2, markersize=1.0, label="full model", 
                color='cornflowerblue')
axins.plot(test_times, ves_per_spike_single_mean[:,i_p,i_size], 
                linestyle='-', marker='', zorder=1, markersize=1.0, label="single timescale (fast)",
                color='gray')
axins.plot(test_times, ves_per_spike_single_slow_mean[:,i_p,i_size], 
                linestyle='-', marker='', zorder=1, markersize=1.0, label="single timescale (slow)",
                color='gray')

# turn off the ticks and labels on the inset plot
axins.set_xticks([])
axins.set_yticks([])
axins.set_xticklabels([])
axins.set_yticklabels([])
axins.set_xlim(-0.3, 5)
axins.set_ylim(-0.07, 0.6)
axins.spines["top"].set_visible(False)
axins.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("depression_test_single_p.pdf")




fig, axs = plt.subplots(2,1, figsize=(2.1,1.6))
axs = np.reshape(axs,-1)

# 2

run_id = 1
max_t = 0.4
min_t = 0.07

spikes = dic.get('input_spikes')[()]
vesicles = dic.get('output_ves')[()][:,run_id,i_p,i_size]

train1 = vesicles
data1 = train1[train1<max_t]
train2 = spikes
data2 = train2[train2<max_t]
axs[0].eventplot([data1,[],data2], colors=["cornflowerblue","gray","lightcoral"])

axs[0].plot([min_t+0.01,min_t+0.01+0.05],[-1.5,-1.5],color="black")
axs[0].annotate("50 ms", (min_t+0.01, -1.4))
axs[0].set_ylabel('ves. and sp.')
axs[0].tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
axs[0].spines["bottom"].set_visible(False)
axs[0].spines["right"].set_visible(False)
axs[0].spines["top"].set_visible(False)
axs[0].set_xlim((min_t,max_t))


# 3

labels = ["Primed", "Docked", "Loaded", "Recovery", "Fused"]

import matplotlib.cm as cm

cmap = cm.get_cmap('viridis')
colors = [cmap(1 - i/5) for i in range(5)]

pools = dic.get('pool_levels')[()][(3,2,1,0,5),:,run_id,i_p,i_size]
ts = dic.get('pool_levels_ts')[()][:,run_id,i_p,i_size]

# print(dic.get('pool_levels')[()][6,:30,run_id,i_p])
# print(dic.get('pool_levels_ts')[()][:30,run_id,i_p])

inds = np.where(ts < max_t)[0]

axs[1].sharex(axs[0])
axs[1].step(ts[inds],100 * pools[:,inds].T, where='post')

for i,j in enumerate(axs[1].lines):
    j.set_color(colors[i])
    j.set_label(labels[i])

axs[1].set_ylabel("Pool levels")
axs[1].spines["bottom"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].spines["top"].set_visible(False)
axs[1].tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)

#axs[1].legend(frameon=False)

import matplotlib.ticker as mtick
axs[1].yaxis.set_major_formatter(mtick.PercentFormatter())

plt.tight_layout()
plt.savefig("depression_test_single_p_spikes.pdf")
#plt.show()


fig, axs = plt.subplots(1,1, figsize=(2.1,1.6))
axs = np.reshape(axs,-1)

# compute cumulative vesicles per bin

input_rate = 50  # Hz
minTestTimes = 0.1  # s
maxTestTimes = 200.0  # s

nTestTimes = 20
test_times = np.exp(np.linspace(np.log(minTestTimes), np.log(maxTestTimes - nTestTimes * minTestTimes), nTestTimes))
test_times += np.arange(1, nTestTimes+1) * minTestTimes

maxT = maxTestTimes
nSpikes = int(maxT * input_rate) + 1
spikes = np.linspace(0, maxT, nSpikes) + minTestTimes

spikes_per_bin = np.zeros(nTestTimes)
spikes_per_bin[0] = np.sum(np.logical_and(spikes >= 0, spikes < test_times[0]))
for i in range(nTestTimes-1):
    spikes_per_bin[i+1] = np.sum(np.logical_and(spikes >= test_times[i], spikes < test_times[i+1]))
    

axs[0].plot(test_times, np.cumsum(ves_per_spike_mean[:,i_p,i_size] * spikes_per_bin),
                linestyle='None', marker='o', zorder=2, markersize=2.0, label="full model", 
                color='cornflowerblue')

axs[0].plot(test_times, np.cumsum(ves_per_spike_single_mean[:,i_p,i_size] * spikes_per_bin), 
                linestyle='None', marker='o', zorder=1, markersize=2.0, label="single timescale",
                color='gray')

axs[0].plot(test_times, np.cumsum(ves_per_spike_single_slow_mean[:,i_p,i_size] * spikes_per_bin), 
                linestyle='None', marker='s', zorder=1, markersize=2.0, label="single timescale",
                color='gray')

axs[0].set_ylabel(r'used resources')
axs[0].set_xlabel(r'stimulation time [s]')
axs[0].set_yscale('log')
axs[0].set_xscale('log')
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)

axins = axs[0].inset_axes([0.1, 0.65, 0.4, 0.4])

# plot some data on the inset plot
axins.plot(test_times, np.cumsum(ves_per_spike_mean[:,i_p,i_size] * spikes_per_bin),
                linestyle='-', marker='', zorder=2, markersize=1.0, label="full model", 
                color='cornflowerblue')

axins.plot(test_times, np.cumsum(ves_per_spike_single_mean[:,i_p,i_size] * spikes_per_bin), 
                linestyle='-', marker='', zorder=1, markersize=1.0, label="single timescale",
                color='gray')

axins.plot(test_times, np.cumsum(ves_per_spike_single_slow_mean[:,i_p,i_size] * spikes_per_bin), 
                linestyle='-', marker='', zorder=1, markersize=1.0, label="single timescale",
                color='gray')

# turn off the ticks and labels on the inset plot
axins.set_xticks([])
axins.set_yticks([])
axins.set_xticklabels([])
axins.set_yticklabels([])
axins.set_xlim(-20.0, 230)
axins.set_ylim(-40, 400)
axins.spines["top"].set_visible(False)
axins.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("depression_test_single_p_energy.pdf")


