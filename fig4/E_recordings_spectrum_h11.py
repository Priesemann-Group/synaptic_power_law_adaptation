import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
plt.rcParams.update({'font.size': 8})

def power_spectrum(events, min_omega, max_omega, n_bins):
    omegas = np.exp(np.linspace(np.log(min_omega), np.log(max_omega), n_bins))
    omega_exp = np.outer(events, omegas)
    complex_exp = np.exp(-1j * 2 * np.pi * omega_exp)
    spectrum = complex_exp.sum(axis=0)
    T = events[-1]
    power_spectrum = np.abs(spectrum)**2 / T
    return power_spectrum, omegas

def full_power_spectrum(events, length, n_bins):
    n_adjust = 300
    omegas1 = np.array([i/length for i in range(n_adjust)])
    omegas2 = np.exp(np.linspace(np.log(np.max(omegas1)), np.log(300.0), n_bins - n_adjust))
    omegas = np.concatenate((omegas1, omegas2))
    omega_exp = np.outer(events, omegas)
    complex_exp = np.exp(-1j * 2 * np.pi * omega_exp)
    spectrum = complex_exp.sum(axis=0)
    power_spectrum = np.abs(spectrum)**2 / length
    return power_spectrum, omegas

def coeff_of_variation(events):
    events = events[events > 0]
    diffs = np.diff(events)
    return np.std(diffs) / np.mean(diffs)
        
def split_times(times, n, T=None):
    if T is None:
        split_dur = times[-1] / n
    else:
        split_dur = T / n
    splits = []
    for i in range(n):
        inds = (times > i * split_dur) * (times < (i+1) * split_dur)
        if (np.sum(inds) > 0):
            splits.append(times[inds] - times[max(0, inds[0]-1)])
    return splits

def process_file(file_path, n_splits=5, min_rate=0.7, max_rate=5, min_omega=0.003, max_omega=1, bins_omega=100, max_T = None):

    print("Processing file " + file_path)

    times = []
    clusters = []
    res = []

    # read csv
    for i in range(1,9):
        times.append(pd.read_csv(f'{file_path}.res.{i}', engine='pyarrow', header=None).values.flatten())
        c = pd.read_csv(f'{file_path}.clu.{i}', engine='pyarrow', header=None).values.flatten()
        c = c[1:] # remove number of clusters
        clusters.append(c * (100**i))

    times = np.concatenate(times)
    clusters = np.concatenate(clusters)

    times = times / 20000 # convert to seconds
    if max_T is not None:
        # full spectrum, truncate to max_T to avoid edge effects
        T = max_T / n_splits
        clusters = clusters[times < max_T]
        times = times[times < max_T]
    else:
        # main spectrum
        T = np.max(times) / n_splits

    ids = np.unique(clusters)

    CVs = []
    
    for id in ids:
        spikes = times[clusters == id]

        rate = len(spikes)/spikes[-1]

        if rate > max_rate or rate < min_rate:
            continue

        CVs.append(coeff_of_variation(spikes))
    
        splits = split_times(spikes, n_splits, T=T * n_splits)
        if min_omega is None:
            f = lambda x: full_power_spectrum(x, T, bins_omega)
        else:
            f = lambda x: power_spectrum(x, min_omega, max_omega, bins_omega)

        spectrum, omegas = f(splits[0])
        spectrum = spectrum * 0
        for split in splits:
            s, _ = f(split)
            spectrum += s
    
        spectrum /= sum(spectrum[:-1] * (omegas[1:] - omegas[:-1]))

        res.append((omegas, spectrum, rate))

    return res, CVs

def process_sim_results(releases, min_omega=0.003, max_omega=1, bins_omega=100, n_splits=5):
    releases = releases[releases > 0]
    splits = split_times(releases, n_splits)
    res = []
    for split in splits:
        spectrum, omegas = power_spectrum(split, min_omega, max_omega, bins_omega)
        spectrum /= sum(spectrum[:-1] * (omegas[1:] - omegas[:-1]))
        res.append(spectrum)
    return np.mean(np.array(res), axis=0)


def main(min_rate=0.7, max_rate=5, n_splits=5):
    min_omega = 0.003
    max_omega = 1
    bins_omega = 100

    recording_file_path1 = 'h11_data/Achilles_11012013'
    recording_file_path2 = 'h11_data/Buddy_06272013'
    recording_file_path3 = 'h11_data/Cicero_09102014'
    recording_file_path4 = 'h11_data/Gatsby_08022013'
    simulation_file_path = f"response_h11_{min_rate}Hz_{max_rate}Hz.h5"

    # get simulation spectrum
    dic = h5py.File(simulation_file_path, 'r')

    sizes = dic.get('sizes')[()]
    i_size = np.where(sizes == 1.0)[0][0]

    releases = dic.get('releases')[()]
    releases_single = dic.get('releases_single')[()]

    max_spikes, nRepetitions, nSizes, nExperiments = releases.shape
    process = lambda x: process_sim_results(x, min_omega=min_omega, max_omega=max_omega, bins_omega=bins_omega, n_splits=n_splits)
    # remove experiments with inconsistent input patterns
    filter = lambda x: np.all(np.diff(x[x > 0]) > 0)
    filtered_releases_inds = [(i,j) for i in range(nRepetitions) for j in range(nExperiments) if filter(releases[:, i, i_size, j])]
    filtered_releases_single_inds = [(i,j) for i in range(nRepetitions) for j in range(nExperiments) if filter(releases_single[:, i, i_size, j])]
    sim_spectrums = [process(releases[:,i, i_size, j]) for i,j in filtered_releases_inds]
    sim_spectrums_single = [process(releases_single[:,i, i_size, j]) for i,j in filtered_releases_single_inds]
    
    # get CVs
    CVS = [coeff_of_variation(releases[:,i, i_size, j]) for i,j in filtered_releases_inds]
    CVS_single = [coeff_of_variation(releases_single[:,i, i_size, j]) for i,j in filtered_releases_single_inds]
    print(f"CV full: mean={np.mean(CVS)}, min,max=({np.min(CVS)}, {np.max(CVS)})")
    print(f"CV single: mean={np.mean(CVS_single)}, min,max=({np.min(CVS_single)}, {np.max(CVS_single)})")

    # get data spectrum
    s1, CVs1 = process_file(recording_file_path1, n_splits=n_splits, min_rate=min_rate, max_rate=max_rate, min_omega=min_omega, max_omega=max_omega, bins_omega=bins_omega)
    s2, CVs2 = process_file(recording_file_path2, n_splits=n_splits, min_rate=min_rate, max_rate=max_rate, min_omega=min_omega, max_omega=max_omega, bins_omega=bins_omega)
    s3, CVs3 = process_file(recording_file_path3, n_splits=n_splits, min_rate=min_rate, max_rate=max_rate, min_omega=min_omega, max_omega=max_omega, bins_omega=bins_omega)
    s4, CVs4 = process_file(recording_file_path4, n_splits=n_splits, min_rate=min_rate, max_rate=max_rate, min_omega=min_omega, max_omega=max_omega, bins_omega=bins_omega)
    s = s1 + s2 + s3 + s4
    CVs = CVs1 + CVs2 + CVs3 + CVs4
    print(f"CV data: mean={np.mean(CVs)}, min,max=({np.min(CVs)}, {np.max(CVs)})")

    fig, axs = plt.subplots(1, 1, figsize=(2.1, 1.6))
    axs = [axs]

    # plot 1/f^alpha
    if max_rate == 0.7:
        alpha = -0.35
    else:
        alpha = -0.5
    omegas = np.exp(np.linspace(np.log(min_omega), np.log(max_omega), bins_omega))
    one_f = np.power(omegas, alpha)
    one_f /= sum(one_f[:-1] * (omegas[1:] - omegas[:-1]))
    axs[0].plot(omegas, one_f, color='black', linewidth=0.5)

    # plot data spectrum
    specs = []
    for omegas, spec, rate in s:
        axs[0].plot(omegas, spec, color='lightcoral', alpha=0.05, linewidth=0.3)
        specs.append(spec)
            
    mean_spec = np.mean(np.array(specs), axis=0)

    axs[0].plot(omegas, mean_spec, color='lightcoral')

    # plot simulation spectrum
    mean_sim_spec = np.mean(np.array(sim_spectrums), axis=0)
    axs[0].plot(omegas, mean_sim_spec, color='cornflowerblue')

    mean_sim_spec_single = np.mean(np.array(sim_spectrums_single), axis=0)
    axs[0].plot(omegas, mean_sim_spec_single, color='gray', linestyle='dashed')

    axs[0].spines["right"].set_visible(False)
    axs[0].spines["top"].set_visible(False)
    axs[0].set_yscale('log')
    axs[0].set_xscale('log')
    axs[0].set_xlabel('frequency [Hz]')
    axs[0].set_ylabel('spectral density')
    plt.tight_layout()
    plt.savefig(f'h11_{min_rate}Hz_{max_rate}Hz.pdf')


    # # plot spikes and pools

    pool_examples = dic.get('pool_examples')[()]
    times_examples = dic.get('times_examples')[()]
    response_examples = dic.get('response_examples')[()]
    response_examples_single = dic.get('response_examples_single')[()]
    input_examples = dic.get('input_examples')[()]

    fig, axs = plt.subplots(2,1, figsize=(2.1,1.6))
    axs = np.reshape(axs,-1)


    min_t = 200.0
    max_t = 400.0
    example_ind = 6
    ind_size = 0

    train1 = response_examples[response_examples[:,ind_size,example_ind]>min_t,ind_size,example_ind]
    train1_single = response_examples_single[response_examples_single[:,ind_size,example_ind]>min_t,ind_size,example_ind]
    data1 = train1[train1<max_t]

    train2 = input_examples[input_examples[:,example_ind]>min_t,example_ind]
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

    pools = pool_examples[(3,2,1,0,5),:,ind_size,example_ind]
    ts = times_examples[:,ind_size,example_ind]
    inds = (ts > min_t) * (ts < max_t)
    pools = pools[:,inds]
    ts = ts[inds]

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

    import matplotlib.ticker as mtick
    axs[1].yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.tight_layout()
    plt.savefig(f"h11_spikes_and_pools_{min_rate}Hz_{max_rate}Hz.pdf")



    # plot full spectrum
    bins_omega = 1000
    max_T = 8.8 * 60 * 60


    # get data spectrum
    s1, _ = process_file(recording_file_path1, n_splits=n_splits, min_rate=min_rate, max_rate=max_rate, min_omega=None, max_omega=None, bins_omega=bins_omega, max_T=max_T)
    # exclude for full spectrum because its only 6 hours
    # s2, _ = process_file(recording_file_path2, n_splits=n_splits, min_rate=min_rate, max_rate=max_rate, min_omega=None, max_omega=None, bins_omega=bins_omega, max_T=max_T)
    s3, _ = process_file(recording_file_path3, n_splits=n_splits, min_rate=min_rate, max_rate=max_rate, min_omega=None, max_omega=None, bins_omega=bins_omega, max_T=max_T)
    s4, _ = process_file(recording_file_path4, n_splits=n_splits, min_rate=min_rate, max_rate=max_rate, min_omega=None, max_omega=None, bins_omega=bins_omega, max_T=max_T)
    s = s1 + s3 + s4

    fig, axs = plt.subplots(1, 1, figsize=(2.1, 1.6))
    axs = [axs]

    # plot 1/f^alpha
    omegas = np.exp(np.linspace(np.log(0.005), np.log(3.0), bins_omega))
    one_f = np.power(omegas, alpha)
    one_f /= sum(one_f[:-1] * (omegas[1:] - omegas[:-1]))
    axs[0].plot(omegas, one_f, color='black', linewidth=0.5)

    # plot data spectrum
    specs = []
    for omegas, spec, rate in s:
        axs[0].plot(omegas, spec, color='lightcoral', alpha=0.01, linewidth=0.3)
        specs.append(spec)
            
    mean_spec = np.mean(np.array(specs), axis=0)

    axs[0].plot(omegas, mean_spec, color='lightcoral')

    axs[0].vlines(7.0, 0, 20, color='red', linestyle='dashed', linewidth=0.2) # theta
    axs[0].vlines(200.0, 0, 20, color='red', linestyle='dashed', linewidth=0.2) # gamma?

    axs[0].set_ylim([0.001, 20])
    axs[0].spines["right"].set_visible(False)
    axs[0].spines["top"].set_visible(False)
    axs[0].set_yscale('log')
    axs[0].set_xscale('log')
    axs[0].set_xlabel('frequency [Hz]')
    axs[0].set_ylabel('spectral density')
    plt.tight_layout()
    plt.savefig(f'h11_full_spectrum_{min_rate}Hz_{max_rate}Hz.pdf')


main(min_rate=0.7, max_rate=5, n_splits=5)
main(min_rate=0.1, max_rate=0.7, n_splits=5)