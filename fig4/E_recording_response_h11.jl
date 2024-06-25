using Random
using HDF5
using CSV
using DataFrames
using ProgressMeter
using Plots
include("../src/Gillespie.jl")
include("../src/params.jl")
include("../src/point_process_functions.jl")
include("../src/vesicle_system.jl")
include("../src/vesicle_system_single.jl")

function insert_truncated!(a, b)
    n = length(a)
    if length(b) > n
        a .= b[1:n]
    else
        a[1:length(b)] .= b
        a[length(b)+1:end] .= 0
    end
end

function insert_truncated2d!(a, b)
    n = size(a, 1)
    if size(b, 1) > n
        a .= b[1:n, :]
    else
        a[1:size(b, 1), :] .= b
        a[size(b, 1)+1:end, :] .= 0
    end
end

function get_data(filenames, min_rate=0.7, max_rate=5.0)
    spikes_data = []
    cluster_data = []
    for i in range(1, 8)
        for filename in filenames
            data = Matrix(CSV.read("h11_data/$filename.res.$i", DataFrame, header=false)) ./ 20000
            data = data[:]
            append!(spikes_data, data)
            clusters = Matrix(CSV.read("h11_data/$filename.clu.$i", DataFrame))
            clusters = clusters[:]
            append!(cluster_data, clusters * 100^(i - 1))
        end
    end

    ids = unique(cluster_data)
    neurons_ids = []
    neurons_spikes = []
    for i in 1:length(ids)
        data = spikes_data[cluster_data.==ids[i]]
        rate = 1 / mean(diff(data))
        println("Rate: ", rate)
        if rate > max_rate || rate < min_rate # skip if rate is too high or too low
            continue
        end
        append!(neurons_spikes, [data])
        append!(neurons_ids, [ids[i]])
    end

    return neurons_ids, neurons_spikes
end

function main(; min_rate=0.7, max_rate=5.0)

    # load data from csv
    filenames = ["Achilles_11012013", "Buddy_06272013", "Cicero_09102014", "Gatsby_08022013"]
    neurons_ids, neurons_spikes = get_data(filenames, min_rate, max_rate)

    nExperiments = length(neurons_spikes)
    sizes = [1.0]
    nSizes = length(sizes)
    nRepetitions = 1
    max_spikes = maximum([length(neurons_spikes[i]) for i in 1:nExperiments])

    releases = zeros(nExperiments, nSizes, nRepetitions, max_spikes)
    releases_single = zeros(nExperiments, nSizes, nRepetitions, max_spikes)

    pool_examples = zeros(nExperiments, nSizes, max_spikes, 6)
    times_examples = zeros(nExperiments, nSizes, max_spikes)
    response_examples = zeros(nExperiments, nSizes, max_spikes)
    response_examples_single = zeros(nExperiments, nSizes, max_spikes)
    input_examples = zeros(nExperiments, max_spikes)

    @showprogress for ind_exp in 1:nExperiments
        spikes = Array{Float64}(neurons_spikes[ind_exp])

        insert_truncated!((@view input_examples[ind_exp, :]), spikes)

        for i_size in 1:nSizes
            synapse_size = sizes[i_size]

            for i in 1:nRepetitions

                rng = MersenneTwister(10 * i)

                (xs, ts) = run_vesicle_cycle(copy(params), spikes, rng, size=synapse_size)
                rs = get_vesicle_release_times(xs[:, 7], ts, max_spikes)
                releases[ind_exp, i_size, i, :] = rs

                if i == 1
                    insert_truncated2d!((@view pool_examples[ind_exp, i_size, :, :]), xs[:, 1:6])
                    insert_truncated!((@view response_examples[ind_exp, i_size, :]), rs)
                    insert_truncated!((@view times_examples[ind_exp, i_size, :]), ts)
                end

                (xs, ts) = run_single_pool_vesicle_cycle(copy(params), spikes, rng, size=synapse_size)
                rs = get_vesicle_release_times(xs[:, 7], ts, max_spikes)
                releases_single[ind_exp, i_size, i, :] = rs

                if i == 1
                    insert_truncated!((@view response_examples_single[ind_exp, i_size, :]), rs)
                end

            end

            pool_examples[ind_exp, i_size, :, 1] ./= synapse_size * Int(params["rec_max"])
            pool_examples[ind_exp, i_size, :, 2] ./= synapse_size * Int(params["load_max"])
            pool_examples[ind_exp, i_size, :, 3] ./= synapse_size * Int(params["dock_max"])
            pool_examples[ind_exp, i_size, :, 4] ./= synapse_size * Int(params["prime_max"])
            pool_examples[ind_exp, i_size, :, 5] ./= synapse_size * Int(params["sprime_max"])
            pool_examples[ind_exp, i_size, :, 6] ./= synapse_size * Int(params["fuse_max"])
        end
    end

    folder = ""
    file = "$(folder)response_h11_$(min_rate)Hz_$(max_rate)Hz.h5"

    rm(file, force=true)
    h5write(file, "releases", releases)
    h5write(file, "releases_single", releases_single)
    h5write(file, "ids", Array{Float64}(neurons_ids))
    h5write(file, "pool_examples", pool_examples)
    h5write(file, "times_examples", times_examples)
    h5write(file, "response_examples", response_examples)
    h5write(file, "response_examples_single", response_examples_single)
    h5write(file, "input_examples", input_examples)
    h5write(file, "sizes", sizes)
end

main(min_rate=0.7, max_rate=5.0)
main(min_rate=0.1, max_rate=0.7)

