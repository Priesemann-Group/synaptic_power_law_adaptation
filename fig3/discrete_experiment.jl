using Random
using HDF5
using ProgressMeter
include("../src/Gillespie.jl")
include("../src/params.jl")
include("../src/vesicle_system.jl")
include("../src/vesicle_system_single.jl")
include("../src/point_process_functions.jl")

function main()


    input_rate = 20 #Hz
    testTime = 2.0 #s

    params["spiking_rate"] = input_rate

    nRepetitions = 100

    nSpikesSaved = 1000
    nTransitionsSaved = 3000

    sizes = [1.0]
    nSizes = length(sizes)
    ps = [0.37]
    nPs = length(ps)
    conditions = [(T_exh, T_pause) for T_exh in [0.4, 4, 40.0] for T_pause in [1, 10, 40, 100]]
    nConditions = length(conditions)

    input_spikes = zeros(nSpikesSaved)
    output_ves = -ones(nSizes, nPs, nConditions, nRepetitions, nSpikesSaved)
    pool_levels = zeros(nSizes, nPs, nConditions, nRepetitions, nTransitionsSaved, 8)
    pool_levels_ts = zeros(nSizes, nPs, nConditions, nRepetitions, nTransitionsSaved)


    for i_size in 1:nSizes
        size = sizes[i_size]

        @showprogress for i_cond in 1:nConditions

            (T_exh, T_pause) = conditions[i_cond]

            params["T_exh"] = T_exh
            params["T_pause"] = T_pause

            maxT = 150.0
            nSpikes = Int(maxT * input_rate) + 1
            spikes = collect(range(0.0, stop=maxT, length=nSpikes))
            #remove spikes in pause time
            spikes = [spikes[spikes.<T_exh]; spikes[spikes.>T_exh+T_pause]]

            input_spikes[1:min(nSpikesSaved, length(spikes))] .= spikes[1:min(nSpikesSaved, length(spikes))]

            for i_p in 1:nPs

                for i in 1:nRepetitions

                    # full model                                
                    rng = MersenneTwister(10 * i)
                    (xs, ts) = run_vesicle_cycle(copy(params), spikes, rng, p_fuse=ps[i_p], size=size)

                    output_ves[i_size, i_p, i_cond, i, :] = get_multiple_vesicle_release_times(xs[:, 7], ts, nSpikesSaved)

                    inds = 1:min(nTransitionsSaved, length(ts))
                    pool_levels[i_size, i_p, i_cond, i, inds, :] = xs[inds, :]
                    pool_levels_ts[i_size, i_p, i_cond, i, inds] = ts[inds]

                    xs = 0
                    ts = 0
                end
            end

        end

        pool_levels[i_size, :, :, :, :, 1] ./= Int(params["rec_max"] * size)
        pool_levels[i_size, :, :, :, :, 2] ./= Int(params["load_max"] * size)
        pool_levels[i_size, :, :, :, :, 3] ./= Int(params["dock_max"] * size)
        pool_levels[i_size, :, :, :, :, 4] ./= Int(params["prime_max"] * size)
        pool_levels[i_size, :, :, :, :, 5] ./= Int(params["sprime_max"] * size)
        pool_levels[i_size, :, :, :, :, 6] ./= Int(params["fuse_max"] * size)
    end


    bin_size = 0.1
    max_time = 200.0
    bins_ts = collect(range(0.0, stop=max_time, length=Int(max_time / bin_size)))
    binned_pool_levels = zeros(nSizes, nPs, nConditions, nRepetitions, Int(max_time / bin_size), 8)
    binned_vesicle_releases = zeros(nSizes, nPs, nConditions, nRepetitions, Int(max_time / bin_size))
    for i_size in 1:nSizes
        for i_cond in 1:nConditions
            for i_p in 1:nPs
                for i in 1:nRepetitions
                    for i_pool in 1:8
                        binned_pool_levels[i_size, i_p, i_cond, i, :, i_pool] =
                            convert_timebased_binbased(pool_levels[i_size, i_p, i_cond, i, :, i_pool], pool_levels_ts[i_size, i_p, i_cond, i, :], bin_size, max_time)
                    end
                    binned_vesicle_releases[i_size, i_p, i_cond, i, :] =
                        convert_timebased_binbased_spikes(output_ves[i_size, i_p, i_cond, i, :], bin_size, max_time)
                end
            end
        end
    end




    folder = ""
    file = "$(folder)discrete_experiment.h5"

    rm(file, force=true)

    h5write(file, "input_spikes", input_spikes)
    h5write(file, "pool_levels", pool_levels)
    h5write(file, "pool_levels_ts", pool_levels_ts)
    h5write(file, "output_ves", output_ves)
    h5write(file, "ps", ps)
    h5write(file, "sizes", sizes)
    h5write(file, "binned_pool_levels", binned_pool_levels)
    h5write(file, "binned_vesicle_releases", binned_vesicle_releases)
    h5write(file, "bins_ts", bins_ts)
    h5write(file, "conditions", conditions)
end

main()

