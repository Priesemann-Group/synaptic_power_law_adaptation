using Random
using HDF5
using ProgressMeter
include("../src/Gillespie.jl")
include("../src/params.jl")
include("../src/vesicle_system.jl")
include("../src/vesicle_system_single.jl")
include("../src/point_process_functions.jl")

function main(condition)


    if condition == 1
        input_rate = 50 #Hz
    else
        input_rate = 10 #Hz
    end
    minTestTimes = 0.1 #s
    maxTestTimes = 200.0 #s

    params["spiking_rate"] = input_rate

    nTestTimes = 20
    if condition == 1
        nRepetitions = 100
    else
        nRepetitions = 500
    end
    test_times = map(exp, collect(range(log(minTestTimes), stop=log(maxTestTimes - nTestTimes * minTestTimes), length=nTestTimes)))
    test_times .+= [i * minTestTimes for i in 1:nTestTimes]
    nSpikesSaved = 1000
    nTransitionsSaved = 1000

    sizes = [0.5, 1.0, 2.0]
    nSizes = length(sizes)
    ps = [0.1, 0.2, 0.5]
    nPs = length(ps)

    ves_per_spike = zeros(nSizes, nPs, nRepetitions, nTestTimes)
    ves_per_spike_single = zeros(nSizes, nPs, nRepetitions, nTestTimes)
    ves_per_spike_single_slow = zeros(nSizes, nPs, nRepetitions, nTestTimes)
    input_spikes = zeros(nSpikesSaved)
    output_ves = zeros(nSizes, nPs, nRepetitions, nSpikesSaved)
    pool_levels = zeros(nSizes, nPs, nRepetitions, nTransitionsSaved, 8)
    pool_levels_ts = zeros(nSizes, nPs, nRepetitions, nTransitionsSaved)

    maxT = maxTestTimes
    nSpikes = Int(maxT * input_rate) + 1
    spikes = collect(range(0.0, stop=maxT, length=nSpikes)) .+ minTestTimes

    input_spikes .= spikes[1:nSpikesSaved]

    for i_size in 1:nSizes

        size = sizes[i_size]

        for i_p in 1:nPs

            println("Run $((i_size - 1) * nPs + i_p) of $(nSizes * nPs)")
            @showprogress for i in 1:nRepetitions

                # full model                                
                rng = MersenneTwister(i)
                (xs, ts) = run_vesicle_cycle(copy(params), spikes, rng, p_fuse=ps[i_p], size=size)

                ves_per_spike[i_size, i_p, i, :] = get_released_vesicles_at_test_times(xs[:, 7], ts, test_times, spikes)

                output_ves[i_size, i_p, i, :] = get_vesicle_release_times(xs[:, 7], ts, nSpikesSaved)

                pool_levels[i_size, i_p, i, :, :] = xs[1:nTransitionsSaved, :]
                pool_levels_ts[i_size, i_p, i, :] = ts[1:nTransitionsSaved]

                # single pool model fast
                params_copy = copy(params)
                params_copy["tau_fuse_prime"] = 0.2 # s
                params_copy["rate_fuse_prime"] = 1 / params_copy["tau_fuse_prime"] # 1/s 

                rng = MersenneTwister(i)
                (xs, ts) = run_single_pool_vesicle_cycle(params_copy, spikes, rng, size=size, p_fuse=ps[i_p])

                ves_per_spike_single[i_size, i_p, i, :] = get_released_vesicles_at_test_times(xs[:, 7], ts, test_times, spikes)

                # single pool model slow
                params_copy = copy(params)
                params_copy["tau_fuse_prime"] = 2.0 # s
                params_copy["rate_fuse_prime"] = 1 / params_copy["tau_fuse_prime"] # 1/s 

                rng = MersenneTwister(i)
                (xs, ts) = run_single_pool_vesicle_cycle(params_copy, spikes, rng, size=size, p_fuse=ps[i_p])

                ves_per_spike_single_slow[i_size, i_p, i, :] = get_released_vesicles_at_test_times(xs[:, 7], ts, test_times, spikes)

                xs = 0
                ts = 0
            end
        end

        pool_levels[i_size, :, :, :, 1] ./= Int(params["rec_max"] * size)
        pool_levels[i_size, :, :, :, 2] ./= Int(params["load_max"] * size)
        pool_levels[i_size, :, :, :, 3] ./= Int(params["dock_max"] * size)
        pool_levels[i_size, :, :, :, 4] ./= Int(params["prime_max"] * size)
        pool_levels[i_size, :, :, :, 5] ./= Int(params["sprime_max"] * size)
        pool_levels[i_size, :, :, :, 6] ./= Int(params["fuse_max"] * size)
    end


    folder = ""
    if condition == 1
        file = "$(folder)depression_test_50Hz.h5"
    else
        file = "$(folder)depression_test_10Hz.h5"
    end

    rm(file, force=true)

    h5write(file, "input_spikes", input_spikes)
    h5write(file, "pool_levels", pool_levels)
    h5write(file, "pool_levels_ts", pool_levels_ts)
    h5write(file, "output_ves", output_ves)
    h5write(file, "ves_per_spike", ves_per_spike)
    h5write(file, "ves_per_spike_single", ves_per_spike_single)
    h5write(file, "ves_per_spike_single_slow", ves_per_spike_single_slow)
    h5write(file, "test_times", test_times)
    h5write(file, "ps", ps)
    h5write(file, "sizes", sizes)
end

main(1) # 50 Hz
main(2) # 10 Hz

