
sigmoid(x) = 1.0 / (1.0 + exp(-x))

function F_single(x, p)
    (x_rec, x_load, x_dock, x_prime, x_sprime, x_fused, x_released, x_aged) = x

    total_ves = sum(x[1:end-2])
    ves_generation = p["rate_generation"] * min(p["max_vesicle_generation"], max(0.0, (p["cycle_capacity"] - total_ves)))

    # fuse_to_pri = p["rate_fuse_prime"] * x_fused * (p["prime_max"] - x_prime) * 2
    # fuse_to_pri /= (p["fuse_max"] * p["prime_max"])

    fuse_to_pri = p["rate_fuse_prime"] * (x_fused > 0) * (p["prime_max"] - x_prime > 0)

    return [ves_generation, fuse_to_pri]
end

transitions_single = Array{Float64}([
    # generation
    0 0 0 0 0 1 0 0;
    # fuse_to_pri
    0 0 0 1 0 -1 0 0])

function spike_single(x, p, rng)
    (x_rec, x_load, x_dock, x_prime, x_sprime, x_fused, x_released, x_aged) = x

    released = sum(rand(rng, Int(x_prime)) .< p["p_fuse_prime"])
    aged = sum(rand(rng, released) .< p["p_aging"])
    x_prime -= released
    x_released += released
    x_aged += aged
    x_fused += released - aged

    return [x_rec, x_load, x_dock, x_prime, x_sprime, x_fused, x_released, x_aged]
end

function run_single_pool_vesicle_cycle(params_copy, spikes, rng; nSamples=Int(1e8), p_fuse=0.2, size=1.0, timescale=1.0)

    params_copy["cycle_capacity"] = ceil(size * params_copy["cycle_capacity"])
    params_copy["prime_max"] = ceil(size * params_copy["prime_max"])

    params_copy["rate_fuse_prime"] = timescale * params_copy["rate_fuse_prime"]

    params_copy["p_fuse_prime"] = p_fuse

    x_rec = 0
    x_load = 0
    x_dock = 0
    x_prime = Int(params_copy["prime_max"])
    x_sprime = 0
    x_fused = Int(params_copy["fuse_max"]) - Int(params_copy["prime_max"])
    x_released = 0
    x_aged = 0
    x0 = Array{Int}([x_rec, x_load, x_dock, x_prime, x_sprime, x_fused, x_released, x_aged])

    test_time = spikes[end]

    return gillespie(x0, F_single, transitions_single, params_copy, nSamples, maxT=test_time,
        external_events=spikes, eventFunction=spike_single, rng=rng)
end
