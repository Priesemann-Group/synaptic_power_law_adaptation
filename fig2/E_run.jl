using Random
using HDF5
using ProgressMeter
using Plots
include("../src/Gillespie.jl")
include("../src/params.jl")
include("../src/point_process_functions.jl")
include("../src/vesicle_system.jl")
include("../src/vesicle_system_single.jl")


function model_release_probability(t, past_releases, past_release_times, b, kappa, timebins)
    time_since_spikes = t .- past_release_times
    n_spikes_in_bins = zeros(length(timebins))
    inds = time_since_spikes .< timebins[1]
    n_spikes_in_bins[1] = sum(past_releases[inds])
    for i in 2:length(timebins)
        inds = (time_since_spikes .< timebins[i]) .& (time_since_spikes .>= timebins[i-1])
        n_spikes_in_bins[i] = sum(past_releases[inds])
    end

    a = b
    for i in 1:length(timebins)
        a += kappa[i] * n_spikes_in_bins[i]
    end
    p = 1 / (1 + exp(-a))
    return p, a, n_spikes_in_bins
end

function log_binomial(k, n, p)
    # we can ignore the binomial coefficient
    return k * log(p) + (n - k) * log(1 - p)
end

function compute_parameter_gradients(spikes, releases, b, kappa, timebins, nPastSpikes)
    dkappa = zeros(length(timebins))
    db = 0.0
    for i_spike in 1:length(spikes)
        i_spike1 = max(1, i_spike - nPastSpikes)
        i_spike2 = i_spike - 1
        past_events = @view releases[i_spike1:i_spike2]
        inds = (past_events) .> 0
        past_releases = past_events[inds]
        past_release_times = @view spikes[i_spike1:i_spike2][inds]
        p, a, n_spikes_in_bins = model_release_probability(spikes[i_spike], past_releases, past_release_times, b, kappa, timebins)
        k = releases[i_spike]
        dp_da = p / (1 + exp(a))
        dl_dp = (k - 4 * p) / (p - p * p)
        dkappa .+= dl_dp * dp_da * n_spikes_in_bins
        db += dl_dp * dp_da * 1
    end
    return db, dkappa
end

function compute_loglikelihood(spikes, releases, b, kappa, timebins, nPastSpikes)
    loglikelihood = 0.0
    for i_spike in 1:length(spikes)
        i_spike1 = max(1, i_spike - nPastSpikes)
        i_spike2 = i_spike - 1
        past_events = @view releases[i_spike1:i_spike2]
        inds = (past_events) .> 0
        past_releases = past_events[inds]
        past_release_times = @view spikes[i_spike1:i_spike2][inds]
        p, _, _ = model_release_probability(spikes[i_spike], past_releases, past_release_times, b, kappa, timebins)
        k = releases[i_spike]
        loglikelihood += log_binomial(k, 4, p)
    end
    return loglikelihood / length(spikes)
end

function fit_model(spikes, releases, spikes_test, releases_test)

    nGradientSteps = 20

    # parameters
    b = zeros(nGradientSteps)
    minBinTimes = 0.001 #s
    maxBinTimes = 500.0 #s
    nBinTimes = 40 # number of time bins in kernel
    nPastSpikes = 5000 # consider only past n spikes to speed up computation
    timebins = map(exp, collect(range(log(minBinTimes), stop=log(maxBinTimes - nBinTimes * minBinTimes), length=nBinTimes)))
    kappa = zeros(nGradientSteps, nBinTimes)

    loglikelihood = zeros(nGradientSteps)
    loglikelihood_test = zeros(nGradientSteps)

    # fit
    eta_b = 3 * 0.4
    eta_kappa = 3 * 0.017 ./ [timebins[i] - [0; timebins][i] for i in 1:length(timebins)] # normalize learning rate by bin size
    @showprogress for i in 1:nGradientSteps-1
        loglikelihood[i] = compute_loglikelihood(spikes, releases, b[i], kappa[i, :], timebins, nPastSpikes)
        loglikelihood_test[i] = compute_loglikelihood(spikes_test, releases_test, b[i], kappa[i, :], timebins, nPastSpikes)

        db, dkappa = compute_parameter_gradients(spikes, releases, b[i], kappa[i, :], timebins, nPastSpikes)

        kappa[i+1, :] .= kappa[i, :] .+ eta_kappa .* dkappa / length(spikes)
        b[i+1] = b[i] + eta_b * db / length(spikes)
    end

    return b, kappa, timebins, loglikelihood, loglikelihood_test
end

function main()

    input_rate = 0.5
    size = 1.0
    p_fuse = 0.2

    # synapse tesing settings
    cluster_rate = 50.0 * input_rate
    intermittence_rate = 1.0 * input_rate
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

    # print firing rate
    println("Firing rate: ", length(spikes) / spikes[end])

    spikes_test = intermittent_poisson_process(cluster_rate, intermittence_rate,
        n_total_events=max_spikes,
        max_events_in_cluster=max_events_in_cluster,
        max_events_in_intermission=max_events_in_intermission,
        z_on=z,
        z_off=z,
        rng=rng)


    ## full model

    params_copy = copy(params)
    (xs, ts) = run_vesicle_cycle(params_copy, spikes, rng, size=size, p_fuse=p_fuse)
    vesicle_releases = get_vesicle_releases_at_spikes(xs[:, 7], ts, spikes)

    params_copy = copy(params)
    (xs_test, ts_test) = run_vesicle_cycle(params_copy, spikes_test, rng, size=size, p_fuse=p_fuse)
    vesicle_releases_test = get_vesicle_releases_at_spikes(xs_test[:, 7], ts_test, spikes_test)

    # fit model
    b, kappa, timebins, loglikelihood_train, loglikelihood_test =
        fit_model(spikes, vesicle_releases, spikes_test, vesicle_releases_test)


    ## single pool model

    params_copy = copy(params)
    (xs, ts) = run_single_pool_vesicle_cycle(params_copy, spikes, rng, size=size, p_fuse=p_fuse)
    vesicle_releases = get_vesicle_releases_at_spikes(xs[:, 7], ts, spikes)

    params_copy = copy(params)
    (xs_test, ts_test) = run_single_pool_vesicle_cycle(params_copy, spikes_test, rng, size=size, p_fuse=p_fuse)
    vesicle_releases_test = get_vesicle_releases_at_spikes(xs_test[:, 7], ts_test, spikes_test)

    # fit model
    b_single, kappa_single, timebins_single, loglikelihood_train_single, loglikelihood_test_single =
        fit_model(spikes, vesicle_releases, spikes_test, vesicle_releases_test)


    file = "model_fit.h5"
    rm(file, force=true)
    # save data
    h5write(file, "kappa", kappa)
    h5write(file, "b", b)
    h5write(file, "timebins", timebins)
    h5write(file, "loglikelihood", loglikelihood_train)
    h5write(file, "loglikelihood_test", loglikelihood_test)
    h5write(file, "kappa_single", kappa_single)
    h5write(file, "b_single", b_single)
    h5write(file, "timebins_single", timebins_single)
    h5write(file, "loglikelihood_single", loglikelihood_train_single)
    h5write(file, "loglikelihood_test_single", loglikelihood_test_single)

end

main()

