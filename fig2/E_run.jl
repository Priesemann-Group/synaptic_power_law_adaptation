using Random
using HDF5
using ProgressMeter
using Plots
include("../src/Gillespie.jl")
include("../src/params.jl")
include("../src/point_process_functions.jl")
include("../src/vesicle_system.jl")
include("../src/vesicle_system_single.jl")


function model_release_probability(t, past_releases, b, kappa, timebins)
    time_since_spikes = t .- past_releases
    n_spikes_in_bins = zeros(length(timebins))
    n_spikes_in_bins[1] = sum(time_since_spikes .< timebins[1])
    for i in 2:length(timebins)
        n_spikes_in_bins[i] = sum(time_since_spikes .< timebins[i] .&& time_since_spikes .>= timebins[i-1])
    end

    a = b
    for i in 1:length(timebins)
        a += kappa[i] * n_spikes_in_bins[i]
    end
    p = 1 / (1 + exp(-a))
    return p, n_spikes_in_bins
end

function fit_model(spikes, releases)

    nGradientSteps = 20

    # parameters
    b = zeros(nGradientSteps)
    minBinTimes = 0.001 #s
    maxBinTimes = 500.0 #s
    nBinTimes = 40
    nPastSpikes = 5000 # consider only past n spikes to speed up computation
    timebins = map(exp, collect(range(log(minBinTimes), stop=log(maxBinTimes - nBinTimes * minBinTimes), length=nBinTimes)))
    kappa = zeros(nGradientSteps, nBinTimes)

    loglikelihood = zeros(nGradientSteps)

    # fit
    eta_b = 5 * 0.07
    eta_kappa = 5 * 0.017 ./ [timebins[i] - [0; timebins][i] for i in 1:length(timebins)] # normalize learning rate by bin size
    @showprogress for i in 1:nGradientSteps-1
        dkappa = zeros(nBinTimes)
        db = 0.0
        for i_spike in 1:length(spikes)
            i_spike1 = max(1, i_spike - nPastSpikes)
            i_spike2 = i_spike - 1
            inds = @view releases[i_spike1:i_spike2]
            past_releases = @view spikes[i_spike1:i_spike2][inds]
            p_release, n_spikes_in_bins = model_release_probability(spikes[i_spike], past_releases, b[i], kappa[i, :], timebins)
            loglikelihood[i] += releases[i_spike] * log(p_release) + (1 - releases[i_spike]) * log(1 - p_release)
            dkappa .+= (releases[i_spike] - p_release) * n_spikes_in_bins
            db += releases[i_spike] - p_release
        end
        kappa[i+1, :] .= kappa[i, :] .+ eta_kappa .* dkappa / length(spikes)
        b[i+1] = b[i] + eta_b * db / length(spikes)
        loglikelihood[i] /= length(spikes)

    end

    return b, kappa, timebins, loglikelihood
end

function main()


    #input_rates = [0.3,0.5,0.75]
    input_rates = [0.5]
    size = 1.0
    p_fuse = 0.1
    nExperiments = length(input_rates)

    for ind_exp in 1:nExperiments

        # synapse tesing settings
        cluster_rate = 50.0 * input_rates[ind_exp]
        intermittence_rate = 1.0 * input_rates[ind_exp]
        max_events_in_cluster = 10000
        max_events_in_intermission = 10000
        z = -2.0

        max_spikes = 50000

        rng = MersenneTwister(10)

        spikes = intermittent_poisson_process(cluster_rate, intermittence_rate,
            n_total_events=max_spikes,
            max_events_in_cluster=max_events_in_cluster,
            max_events_in_intermission=max_events_in_intermission,
            z_on=z,
            z_off=z,
            rng=rng)

        params_copy = copy(params)
        (xs, ts) = run_vesicle_cycle(params_copy, spikes, rng, size=size, p_fuse=p_fuse)

        vesicle_releases = get_vesicle_releases_at_spikes(xs[:, 7], ts, spikes)

        # fit model
        b, kappa, timebins, loglikelihood = fit_model(spikes, vesicle_releases)

        params_copy = copy(params)
        (xs, ts) = run_single_pool_vesicle_cycle(params_copy, spikes, rng, size=size, p_fuse=p_fuse)

        vesicle_releases = get_vesicle_releases_at_spikes(xs[:, 7], ts, spikes)

        # fit model
        b_single, kappa_single, timebins_single, loglikelihood_single = fit_model(spikes, vesicle_releases)

        file = "model_fit.h5"
        rm(file, force=true)
        # save data
        h5write(file, "kappa", kappa)
        h5write(file, "b", b)
        h5write(file, "timebins", timebins)
        h5write(file, "loglikelihood", loglikelihood)
        h5write(file, "kappa_single", kappa_single)
        h5write(file, "b_single", b_single)
        h5write(file, "timebins_single", timebins_single)
        h5write(file, "loglikelihood_single", loglikelihood_single)
    end
end

main()

