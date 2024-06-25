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

    input_rates = [4.0]
    #input_rates = [0.3]
    nExperiments = length(input_rates)
    sizes = [1.0]
    nSizes = length(sizes)

    pfuse_single = 0.04 # similar release rates


    for ind_exp in 1:nExperiments

        # synapse tesing settings
        cluster_rate = 50.0 * input_rates[ind_exp]
        intermittence_rate = 1.0 * input_rates[ind_exp]
        max_events_in_cluster = 10000
        max_events_in_intermission = 10000
        z = -2.0

        nRepetitions = 10
        max_spikes = 10000000

        vesicle_release_trains = -ones(nSizes, nRepetitions, max_spikes)
        vesicle_release_trains_single = -ones(nSizes, nRepetitions, max_spikes)
        spike_trains = -ones(nSizes, nRepetitions, max_spikes)

        for i_size in 1:nSizes
            size = sizes[i_size]

            rng = MersenneTwister(20)


            spikes = intermittent_poisson_process(cluster_rate, intermittence_rate,
                n_total_events=max_spikes,
                max_events_in_cluster=max_events_in_cluster,
                max_events_in_intermission=max_events_in_intermission,
                z_on=z,
                z_off=z,
                rng=rng)

            println("Run $((ind_exp - 1) * nSizes + i_size) of $(nSizes * nExperiments)")
            for i in 1:nRepetitions
                println("Repetition $i of $nRepetitions")
                println("Spike rate ", max_spikes / spikes[end])

                (xs, ts) = run_vesicle_cycle(copy(params), spikes, rng, size=size)
                vesicle_release_trains[i_size, i, :] = get_vesicle_release_times(xs[:, 7], ts, max_spikes)
                println("Release rate full model: ", xs[end, 7] / spikes[end])

                (xs_single, ts_single) = run_single_pool_vesicle_cycle(copy(params), spikes, rng, size=size, p_fuse=pfuse_single)
                vesicle_release_trains_single[i_size, i, :] = get_vesicle_release_times(xs_single[:, 7], ts_single, max_spikes)
                println("Release rate single model: ", xs_single[end, 7] / spikes[end])

                spike_trains[i_size, i, :] = spikes
            end
        end


        folder = ""
        file = "$(folder)average_rate.h5"

        rm(file, force=true)

        h5write(file, "experiment$(ind_exp)/spike_trains", spike_trains)
        h5write(file, "experiment$(ind_exp)/vesicle_release_trains", vesicle_release_trains)
        h5write(file, "experiment$(ind_exp)/vesicle_release_trains_single", vesicle_release_trains_single)
        h5write(file, "experiment$(ind_exp)/input_rate", cluster_rate)
        h5write(file, "experiment$(ind_exp)/within_cluster_rate", cluster_rate)
        h5write(file, "experiment$(ind_exp)/max_events_in_cluster", max_events_in_cluster)
        h5write(file, "experiment$(ind_exp)/sizes", sizes)
    end
end

main()

