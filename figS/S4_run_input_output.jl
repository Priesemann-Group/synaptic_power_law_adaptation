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
    file = "$(folder)input_output.h5"

    rm(file, force=true)

    input_rates = collect(0.1:0.1:2.0)
    nExperiments = length(input_rates)
    sizes = [0.5, 1.0, 2.0]
    nSizes = length(sizes)
    nRepetitions = 1
    max_spikes = 100000

    vesicle_per_spike_1_f = zeros(nExperiments, nSizes, nRepetitions)
    vesicle_per_spike_1_f_single = zeros(nExperiments, nSizes, nRepetitions)
    spike_rate_1_f = zeros(nExperiments, nSizes, nRepetitions)
    vesicle_per_spike_regular = zeros(nExperiments, nSizes, nRepetitions)
    vesicle_per_spike_regular_single = zeros(nExperiments, nSizes, nRepetitions)
    spike_rate_regular = zeros(nExperiments, nSizes, nRepetitions)
    vesicle_per_spike_poisson = zeros(nExperiments, nSizes, nRepetitions)
    vesicle_per_spike_poisson_single = zeros(nExperiments, nSizes, nRepetitions)
    spike_rate_poisson = zeros(nExperiments, nSizes, nRepetitions)

    @showprogress for ind_exp in 1:nExperiments

        # synapse tesing settings
        cluster_rate = 50.0 * input_rates[ind_exp]
        intermittence_rate = 1.0 * input_rates[ind_exp]
        max_events_in_cluster = 10000
        max_events_in_intermission = 10000
        z = -2.0

        for i_size in 1:nSizes
            size = sizes[i_size]

            for i in 1:nRepetitions

                rng = MersenneTwister(10 * i)

                # 1/f
                spikes = intermittent_poisson_process(cluster_rate, intermittence_rate,
                    n_total_events=max_spikes,
                    max_events_in_cluster=max_events_in_cluster,
                    max_events_in_intermission=max_events_in_intermission,
                    z_on=z,
                    z_off=z,
                    rng=rng)

                (xs, ts) = run_vesicle_cycle(copy(params), spikes, size, rng)
                vesicle_release_train = get_vesicle_release_times(xs[:, 7], ts, length(spikes))

                vesicle_per_spike_1_f[ind_exp, i_size, i] = mean(vesicle_release_train .> 0)
                spike_rate_1_f[ind_exp, i_size, i] = length(spikes) / spikes[end]


                (xs, ts) = run_single_pool_vesicle_cycle(copy(params), spikes, size, rng)
                vesicle_release_train = get_vesicle_release_times(xs[:, 7], ts, length(spikes))

                vesicle_per_spike_1_f_single[ind_exp, i_size, i] = mean(vesicle_release_train .> 0)


                # regular
                spikes = collect(range(0.0, stop=spikes[end], length=length(spikes))) # same rate as 1/f

                (xs, ts) = run_vesicle_cycle(copy(params), spikes, size, rng)
                vesicle_release_train = get_vesicle_release_times(xs[:, 7], ts, length(spikes))

                vesicle_per_spike_regular[ind_exp, i_size, i] = mean(vesicle_release_train .> 0)
                spike_rate_regular[ind_exp, i_size, i] = length(spikes) / spikes[end]

                (xs, ts) = run_single_pool_vesicle_cycle(copy(params), spikes, size, rng)
                vesicle_release_train = get_vesicle_release_times(xs[:, 7], ts, length(spikes))

                vesicle_per_spike_regular_single[ind_exp, i_size, i] = mean(vesicle_release_train .> 0)

                # poisson
                rate = length(spikes) / spikes[end]
                spikes = poisson_process(rate, length(spikes), rng=rng)

                (xs, ts) = run_vesicle_cycle(copy(params), spikes, size, rng)
                vesicle_release_train = get_vesicle_release_times(xs[:, 7], ts, length(spikes))

                vesicle_per_spike_poisson[ind_exp, i_size, i] = mean(vesicle_release_train .> 0)
                spike_rate_poisson[ind_exp, i_size, i] = length(spikes) / spikes[end]

                (xs, ts) = run_single_pool_vesicle_cycle(copy(params), spikes, size, rng)
                vesicle_release_train = get_vesicle_release_times(xs[:, 7], ts, length(spikes))

                vesicle_per_spike_poisson_single[ind_exp, i_size, i] = mean(vesicle_release_train .> 0)

            end
        end

    end
    h5write(file, "vesicle_per_spike_1_f", vesicle_per_spike_1_f)
    h5write(file, "vesicle_per_spike_1_f_single", vesicle_per_spike_1_f_single)
    h5write(file, "spike_rate_1_f", spike_rate_1_f)
    h5write(file, "vesicle_per_spike_regular", vesicle_per_spike_regular)
    h5write(file, "vesicle_per_spike_regular_single", vesicle_per_spike_regular_single)
    h5write(file, "spike_rate_regular", spike_rate_regular)
    h5write(file, "vesicle_per_spike_poisson", vesicle_per_spike_poisson)
    h5write(file, "vesicle_per_spike_poisson_single", vesicle_per_spike_poisson_single)
    h5write(file, "spike_rate_poisson", spike_rate_poisson)

    h5write(file, "sizes", sizes)
end

main()

