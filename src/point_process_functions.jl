using Random
using StatsBase

"""
An Intermittent Poisson Process Generating 1/f Noise with Possible Application to Fluorescence Intermittency
"""
function intermittent_poisson_process(cluster_rate::Float64, intermission_rate::Float64;
    rng=MersenneTwister(1)::AbstractRNG, n_total_events::Int=Int(1e4), max_events_in_cluster::Int=100,
    max_events_in_intermission::Int=100, z_on::Float64=-2.0, z_off::Float64=-2.0)::Array{Float64,1}

    events = zeros(Float64, n_total_events)

    n_distribution = Weights([n^z_on for n in 1:max_events_in_cluster])
    m_distribution = Weights([m^z_off for m in 1:max_events_in_intermission])

    t = 0.0
    i = 1
    state = true # true = ON, false = OFF
    event_counter = sample(rng, n_distribution)
    while i <= n_total_events
        if state
            # create next event
            dt = randexp(rng) / cluster_rate
            t += dt
            event_counter -= 1

            events[i] = t
            i += 1

            if event_counter == 0
                state = false
                event_counter = sample(rng, n_distribution)
            end
        else
            # create next event
            dt = randexp(rng) / intermission_rate
            t += dt
            event_counter -= 1

            if event_counter == 0
                state = true
                event_counter = sample(rng, m_distribution)
            end
        end
    end

    return events
end

function poisson_process(rate::Float64, n_total_events::Int; rng=MersenneTwister(1)::AbstractRNG)::Array{Float64,1}
    events = zeros(Float64, n_total_events)
    t = 0.0
    i = 1
    while i <= n_total_events
        dt = randexp(rng) / rate
        t += dt
        events[i] = t
        i += 1
    end
    return events
end

"""

# Python implementation

import numpy as np
from numpy.random import Generator, PCG64, exponential, choice


def intermittent_poisson_process(cluster_rate, intermission_rate, rng=Generator(PCG64(1)), n_total_events=int(1e4),
                                  max_events_in_cluster=100, max_events_in_intermission=100, z_on=-2.0, z_off=-2.0):

    events = np.zeros(n_total_events)

    n_distribution = np.power(np.arange(1, max_events_in_cluster+1), z_on)
    n_distribution = n_distribution / np.sum(n_distribution)

    m_distribution = np.power(np.arange(1, max_events_in_intermission+1), z_off)
    m_distribution = m_distribution / np.sum(m_distribution)

    t = 0.0
    i = 0
    state = True  # True = ON, False = OFF

    event_counter = choice(np.arange(1, max_events_in_cluster+1), p=n_distribution)

    while i < n_total_events:
        if state:
            # create next event
            dt = exponential(1.0/cluster_rate, size=1)
            t += dt
            events[i] = t
            i += 1
            event_counter -= 1

            if event_counter == 0:
                state = False
                event_counter = choice(np.arange(1, max_events_in_intermission+1), p=m_distribution)
        else:
            # create next event
            dt = exponential(1.0/intermission_rate, size=1)
            t += dt
            event_counter -= 1

            if event_counter == 0:
                state = True
                event_counter = choice(np.arange(1, max_events_in_cluster+1), p=n_distribution)

    return events

"""



"""
Computes power spectral density for a point process.

- events: list of event times of point process
- min_omega: minimum measured frequency of spectrum
- max_omega: maximum measured frequency of spectrum
- n_bins: number of bins for spectrum between min_omega and max_omega
"""
function power_spectrum(events::Array{Float64,1}, min_omega::Float64, max_omega::Float64,
    n_bins::Int)::Tuple{Array{Float64,1},Array{Float64,1}}

    n_events = length(events)
    spectrum = zeros(Complex, n_bins)
    omegas = map(exp, collect(range(log(min_omega), stop=log(max_omega), length=n_bins)))

    for i in 1:n_bins
        for j in 1:n_events
            spectrum[i] += exp(-1im * 2 * pi * omegas[i] * events[j])
        end
    end

    T = events[end]
    return Array{Float64,1}(map(abs2, spectrum) ./ T), omegas
end


"""
### For testing

using ProgressMeter
using Plots
n = 30
spectra = zeros(n,50)
@showprogress for i in 1:n
    s = intermittent_poisson_process(15.0,3.0,n_total_events=1000000,max_events_in_cluster=10000, max_events_in_intermission=10000,rng=MersenneTwister(i), z_on=-2.0, z_off=-2.0)
    power, omegas = power_spectrum(s,0.0001,10.0,50)
    spectra[i,:] .= power 
end
s = intermittent_poisson_process(15.0,3.0,n_total_events=1000000,max_events_in_cluster=10000, max_events_in_intermission=10000,rng=MersenneTwister(1), z_on=-2.0, z_off=-2.0)
power, omegas = power_spectrum(s,0.0001,10.0,50)
plot(omegas, mean(spectra, dims=1)[:], xscale=:log10, yscale=:log10)
"""

"""
using ProgressMeter
using Plots
n = 20
N = 30
spectra = zeros(n,N)
@showprogress for i in 1:n
    s = intermittent_poisson_process(50.0,3.0,n_total_events=1000000,max_events_in_cluster=100000, max_events_in_intermission=10000,rng=MersenneTwister(i), z_on=-2.0, z_off=-2.0)
    power, omegas = power_spectrum(s,0.0001,10.0,N)
    spectra[i,:] .= power 
end
s = intermittent_poisson_process(15.0,3.0,n_total_events=1000000,max_events_in_cluster=10000, max_events_in_intermission=10000,rng=MersenneTwister(1), z_on=-2.0, z_off=-2.0)
power, omegas = power_spectrum(s,0.0001,10.0,N)
plot(omegas, mean(spectra, dims=1)[:], xscale=:log10, yscale=:log10)
plot!(omegas, 1 ./ omegas, xscale=:log10, yscale=:log10)
"""


### Helper functions

"""
Measures the number of vesicles released in a given time window.

- n_released: number of released vesicles at each time point
- ts: time points
- test_times: time windows
- spikes: spike times
"""
function get_released_vesicles_at_test_times(n_released, ts, test_times, spikes)
    released_vesicles = zeros(length(test_times))

    for i in 1:length(test_times)
        if i == 1
            inds = (ts .>= 0) .& (ts .< test_times[1])
            n_spikes = sum(spikes .< test_times[1])
        else
            # have to get n of releases from before test starts
            spike_before_test_time = spikes[spikes.<test_times[i-1]][end]
            # and compare to end of test
            inds = (ts .>= spike_before_test_time) .& (ts .< test_times[i])
            n_spikes = sum((spikes .>= test_times[i-1]) .& (spikes .< test_times[i]))
        end
        n_released_slice = n_released[inds]
        released_vesicles[i] = (n_released_slice[end] - n_released_slice[1]) / n_spikes
    end

    return released_vesicles
end

function get_vesicle_release_times(n_released, ts, max_releases)
    vesicle_release_times = -ones(max_releases)

    ind = 1
    for i in 2:length(ts)
        if n_released[i] - n_released[i-1] > 0
            vesicle_release_times[ind] = ts[i]
            ind += 1
        end
    end

    return vesicle_release_times
end

""" same as get_vesicle_release_times but inserts multiple times for multi-releases """
function get_multiple_vesicle_release_times(n_released, ts, max_releases)
    vesicle_release_times = -ones(max_releases)

    ind = 1
    for i in 2:length(ts)
        for j in 1:n_released[i]-n_released[i-1]
            vesicle_release_times[ind] = ts[i]
            ind += 1
        end
    end

    return vesicle_release_times
end

function get_vesicle_releases_at_spikes(n_released, ts, spikes)::Array{Int}
    vesicle_releases = zeros(length(spikes))

    n_released_last_t = n_released[1]

    for i in 1:length(spikes)
        n_released_t = n_released[ts.==spikes[i]][1]
        vesicle_releases[i] = n_released_t - n_released_last_t
        n_released_last_t = n_released_t
    end

    return vesicle_releases
end


"""
Inserts spikes into array while respecting the max length of the array.

- train: 2D array to insert spikes into
- index: index of the array to insert spikes into
- spikes: spikes to insert
"""
function insert_spikes!(train, index, spikes)
    max_spikes = length(train[index, :])
    n_spikes = length(spikes)
    for i in 1:min(max_spikes, n_spikes)
        train[index, i] = spikes[i]
    end
end


function convert_timebased_binbased(inputs, input_ts, bin_size, max_time)
    n_bins = Int(max_time / bin_size)
    ts = collect(range(0.0, stop=max_time + bin_size, length=n_bins + 1))
    bin_inputs = zeros(n_bins)
    for j in 1:n_bins
        inds = (input_ts .>= ts[j]) .& (input_ts .< ts[j+1])
        if sum(inds) == 0
            bin_inputs[j] = inputs[input_ts.<ts[j+1]][end]
        else
            bin_inputs[j] = mean(inputs[inds])
        end
    end
    return bin_inputs
end

function convert_timebased_binbased_spikes(spikes, bin_size, max_time)
    n_bins = Int(max_time / bin_size)
    ts = collect(range(0.0, stop=max_time + bin_size, length=n_bins + 1))
    bin_spikes = zeros(n_bins)
    for j in 1:n_bins
        bin_spikes[j] = sum((spikes .>= ts[j]) .& (spikes .< ts[j+1]))
    end
    return bin_spikes
end