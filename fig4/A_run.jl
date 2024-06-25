using Random
using HDF5
using ProgressMeter
using Plots
include("../src/Gillespie.jl")
include("../src/params.jl")
include("../src/point_process_functions.jl")
include("../src/vesicle_system.jl")
include("../src/vesicle_system_single.jl")

function main()

    folder = ""
    file = "$(folder)whitening_test.h5"

    rm(file, force=true)

    input_rates = [0.5, 0.75, 1.0]
    nExperiments = length(input_rates)
    sizes = [0.5, 1.0, 2.0]
    nSizes = length(sizes)

    pfuse_single = 0.08

    for ind_exp in 1:nExperiments

        # power spectrum settings
        min_freq = 0.0005 #Hz
        max_freq = 3.0 #Hz
        n_bins = 100
        _, omegas = power_spectrum(zeros(10), min_freq, max_freq, n_bins)

        # synapse tesing settings
        cluster_rate = 50.0 * input_rates[ind_exp]
        intermittence_rate = 1.0 * input_rates[ind_exp]
        max_events_in_cluster = 10000
        max_events_in_intermission = 10000
        z = -2.0

        nRepetitions = 100
        nRepetitionsSaved = 10
        nTransitionsSaved = 10000
        max_spikes = 1000000

        vesicle_release_trains = -ones(nSizes, nRepetitions, max_spikes)
        vesicle_power_spectrums = zeros(nSizes, nRepetitions, n_bins)
        vesicle_release_trains_single = -ones(nSizes, nRepetitions, max_spikes)
        vesicle_power_spectrums_single = zeros(nSizes, nRepetitions, n_bins)
        spike_trains = -ones(nSizes, nRepetitions, max_spikes)
        spike_power_spectrums = zeros(nSizes, nRepetitions, n_bins)
        pool_levels = zeros(nSizes, nRepetitionsSaved, nTransitionsSaved, 8)
        pool_levels_ts = zeros(nSizes, nRepetitionsSaved, nTransitionsSaved)

        for i_size in 1:nSizes
            size = sizes[i_size]

            println("Run $((ind_exp - 1) * nSizes + i_size) of $(nSizes * nExperiments)")
            @showprogress for i in 1:nRepetitions

                rng = MersenneTwister(10 * i)

                spikes = intermittent_poisson_process(cluster_rate, intermittence_rate,
                    n_total_events=max_spikes,
                    max_events_in_cluster=max_events_in_cluster,
                    max_events_in_intermission=max_events_in_intermission,
                    z_on=z,
                    z_off=z,
                    rng=rng)

                (xs, ts) = run_vesicle_cycle(copy(params), spikes, rng, size=size)
                vesicle_release_trains[i_size, i, :] = get_vesicle_release_times(xs[:, 7], ts, max_spikes)
                train = vesicle_release_trains[i_size, i, vesicle_release_trains[i_size, i, :].>0.0]
                power_vesicles, _ = power_spectrum(train, min_freq, max_freq, n_bins)
                vesicle_power_spectrums[i_size, i, :] = power_vesicles

                if i <= nRepetitionsSaved
                    pool_levels[i_size, i, :, :] = xs[1:nTransitionsSaved, :]
                    pool_levels_ts[i_size, i, :] = ts[1:nTransitionsSaved]
                end

                (xs_single, ts_single) = run_single_pool_vesicle_cycle(copy(params), spikes, rng, size=size, p_fuse=pfuse_single)
                vesicle_release_trains_single[i_size, i, :] = get_vesicle_release_times(xs_single[:, 7], ts_single, max_spikes)
                train_single = vesicle_release_trains_single[i_size, i, vesicle_release_trains_single[i_size, i, :].>0.0]
                power_vesicles_single, _ = power_spectrum(train_single, min_freq, max_freq, n_bins)
                vesicle_power_spectrums_single[i_size, i, :] = power_vesicles_single

                spike_trains[i_size, i, :] = spikes
                power_spikes, _ = power_spectrum(spikes, min_freq, max_freq, n_bins)
                spike_power_spectrums[i_size, i, :] = power_spikes
            end

            pool_levels[i_size, :, :, 1] ./= size * Int(params["rec_max"])
            pool_levels[i_size, :, :, 2] ./= size * Int(params["load_max"])
            pool_levels[i_size, :, :, 3] ./= size * Int(params["dock_max"])
            pool_levels[i_size, :, :, 4] ./= size * Int(params["prime_max"])
            pool_levels[i_size, :, :, 5] ./= size * Int(params["sprime_max"])
            pool_levels[i_size, :, :, 6] ./= size * Int(params["fuse_max"])
        end


        h5write(file, "experiment$(ind_exp)/pool_levels", pool_levels)
        h5write(file, "experiment$(ind_exp)/pool_levels_ts", pool_levels_ts)

        h5write(file, "experiment$(ind_exp)/omegas", omegas)
        maxind = findfirst(x -> x > 500 * 60, spike_trains[1, 1, :])
        h5write(file, "experiment$(ind_exp)/spike_trains", spike_trains[:, :, 1:maxind])
        h5write(file, "experiment$(ind_exp)/spike_power_spectrums", spike_power_spectrums)
        maxind = findfirst(x -> x > 500 * 60, vesicle_release_trains[1, 1, :])
        h5write(file, "experiment$(ind_exp)/vesicle_release_trains", vesicle_release_trains[:, :, 1:maxind])
        maxind = findfirst(x -> x > 500 * 60, vesicle_release_trains_single[1, 1, :])
        h5write(file, "experiment$(ind_exp)/vesicle_release_trains_single", vesicle_release_trains_single[:, :, 1:maxind])
        h5write(file, "experiment$(ind_exp)/vesicle_power_spectrums", vesicle_power_spectrums)
        h5write(file, "experiment$(ind_exp)/vesicle_power_spectrums_single", vesicle_power_spectrums_single)
        h5write(file, "experiment$(ind_exp)/input_rate", cluster_rate)
        h5write(file, "experiment$(ind_exp)/within_cluster_rate", cluster_rate)
        h5write(file, "experiment$(ind_exp)/max_events_in_cluster", max_events_in_cluster)
        h5write(file, "experiment$(ind_exp)/sizes", sizes)
    end
end

main()

