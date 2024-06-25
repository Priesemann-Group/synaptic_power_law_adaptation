
sigmoid(x) = 1.0 / (1.0 + exp(-x))

function F(x, p)
    (x_rec, x_load, x_dock, x_prime, x_sprime, x_fused, x_released, x_aged) = x

    total_ves = sum(x[1:end-2])
    ves_generation = p["rate_generation"] * min(p["max_vesicle_generation"], max(0.0, (p["cycle_capacity"] - total_ves)))

    rec_to_load = p["rate_rec_load"] * x_rec * (p["load_max"] - x_load) * 4
    rec_to_load /= (p["rec_max"] * p["load_max"])

    dock_to_load = p["rate_dock_load"] * x_dock * (p["load_max"] - x_load) * 4
    dock_to_load /= (p["dock_max"] * p["load_max"])

    load_to_dock = p["rate_load_dock"] * x_load * (p["dock_max"] - x_dock) * 4
    load_to_dock /= (p["dock_max"] * p["load_max"])

    dock_to_pri = p["rate_dock_prime"] * x_dock * (p["prime_max"] - x_prime) * 2
    dock_to_pri /= (p["dock_max"] * p["prime_max"])

    fuse_to_rec = p["rate_endo"] * sigmoid(100 * (x_fused - 0.5))
    fuse_to_rec *= sigmoid(100 * (p["rec_max"] - x_rec - 0.5))

    return [rec_to_load, dock_to_load, load_to_dock, dock_to_pri,
        ves_generation, fuse_to_rec]
end

transitions = Array{Float64}([
    # rec_to_load 
    -1 1 0 0 0 0 0 0;
    # dock_to_load 
    0 1 -1 0 0 0 0 0;
    # load_to_dock
    0 -1 1 0 0 0 0 0;
    # dock_to_pri
    0 0 -1 1 0 0 0 0;
    # generation
    0 0 0 0 0 1 0 0;
    # fuse_to_rec
    1 0 0 0 0 -1 0 0])

function spike(x, p, rng)
    (x_rec, x_load, x_dock, x_prime, x_sprime, x_fused, x_released, x_aged) = x

    released = sum(rand(rng, Int(x_prime)) .< p["p_fuse_prime"])
    aged = sum(rand(rng, released) .< p["p_aging"])
    x_prime -= released
    x_released += released
    x_aged += aged
    x_fused += released - aged

    return [x_rec, x_load, x_dock, x_prime, x_sprime, x_fused, x_released, x_aged]
end

function run_vesicle_cycle(params_copy, spikes, rng; nSamples=Int(1e8), p_fuse=0.2, size=1.0, timescale=1.0)

    params_copy["cycle_capacity"] = ceil(size * params_copy["cycle_capacity"])
    params_copy["rec_max"] = ceil(size * params_copy["rec_max"])
    params_copy["load_max"] = ceil(size * params_copy["load_max"])
    params_copy["dock_max"] = ceil(size * params_copy["dock_max"])
    params_copy["prime_max"] = ceil(size * params_copy["prime_max"])
    params_copy["sprime_max"] = ceil(size * params_copy["sprime_max"])

    params_copy["rate_rec_load"] = timescale * params_copy["rate_rec_load"]
    params_copy["rate_load_dock"] = timescale * params_copy["rate_load_dock"]
    params_copy["rate_dock_prime"] = timescale * params_copy["rate_dock_prime"]
    params_copy["rate_dock_load"] = timescale * params_copy["rate_dock_load"]
    params_copy["rate_endo"] = timescale * params_copy["rate_endo"]
    params_copy["rate_generation"] = timescale * params_copy["rate_generation"]

    params_copy["p_fuse_prime"] = p_fuse

    x_rec = Int(params_copy["rec_max"])
    x_load = Int(params_copy["load_max"])
    x_dock = Int(params_copy["dock_max"])
    x_prime = Int(params_copy["prime_max"])
    x_sprime = Int(params_copy["sprime_max"])
    x_fused = 0
    x_released = 0
    x_aged = 0
    x0 = Array{Int}([x_rec, x_load, x_dock, x_prime, x_sprime, x_fused, x_released, x_aged])

    test_time = spikes[end]

    return gillespie(x0, F, transitions, params_copy, nSamples, maxT=test_time,
        external_events=spikes, eventFunction=spike, rng=rng)
end
