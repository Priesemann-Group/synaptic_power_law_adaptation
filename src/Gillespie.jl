using Random

""" 
Runs the gillespie algorithm for a given system.
Returns concentrations and event times.

- x0: initial conditions [n]
- F: rate-function; [n], [j] -> [m]
- tansitions: transition matrix [m,n]
- params: parameter matrix [j]
- nSamples: maximum number of samples
- maxT: termination time
- externalEvents: (sorted) matrix [k] of k external event-times
- eventFunction: executed when external event appears; [n], [j] -> [m]
- rng: random number generator
"""
function gillespie(
    x0::Array{Int}, F, 
    transitions::Array{Float64,2}, 
    params::Union{Array{Float64},Dict{String,Float64}},
    nSamples::Int; 
    maxT::Float64=Inf, 
    external_events::Array{Float64}=Array{Float64}([]),
    eventFunction=identity,
    rng::AbstractRNG=MersenneTwister(1000))

    current_event = 1
    n_events = length(external_events)
    xs = zeros(nSamples+1, length(x0))
    xs[1,:] = x0
    ts = zeros(nSamples+1)
    t = 0.0

    for i_t in 1:nSamples
        if t > maxT
            break
        end
        if i_t == nSamples
            print("$(nSamples) Gillespie samples exhausted.")
        end
    
        # get reaction rates for current concentration
        rates = F(xs[i_t,:], params)
        # get timestep and index of next reaction
        (dt, ind) = next_sample(rng, t -> rates, sum(rates))

        if current_event <= n_events && t+dt > external_events[current_event]
            # if external event occurs before next sample apply 
            # event function
            t = external_events[current_event]
            xs[i_t+1,:] = eventFunction(xs[i_t,:], params, rng)
            current_event += 1
        elseif dt != Inf 
            # otherwise perform sampled transition
            t += dt
            xs[i_t+1,:] = xs[i_t,:] + transitions[ind,:]
        else
            # no transitions left
            break
        end
        ts[i_t+1] = t
    end
    # only return samples and not the whole array
    inds = ts .> 0.0
    inds[1] = true
    return (xs[inds,:], ts[inds])
end


"""  
Runs the gillespie algorithm for a given system.
Returns rates of reactions, concentrations and event times.

- x0: initial conditions [n]
- F: rate-function; [n], [j] -> [m]
- tansitions: transition matrix [m,n]
- params: parameter matrix [j]
- nSamples: maximum number of samples
- maxT: termination time
- externalEvents: (sorted) matrix [k] of k external event-times
- eventFunction: executed when external event appears; [n], [j] -> [m]
- rng: random number generator
"""
function gillespie_rates(
    x0::Array{Int}, 
    F, 
    transitions::Array{Float64,2}, 
    params::Union{Array{Float64},Dict{String,Float64}},
    nSamples::Int; 
    maxT::Float64=Inf, 
    external_events::Array{Float64}=Array{Float64}([]),
    eventFunction=identity,
    rng::AbstractRNG=MersenneTwister(1000))

    current_event = 1
    n_events = length(external_events)
    xs = zeros(nSamples+1, length(x0))
    xs[1,:] = x0
    ts = zeros(nSamples+1)
    t = 0.0
    rates_out = zeros(nSamples+1, size(transitions,1))
    rates_out[1,:] = F(xs[1,:], params)

    for i_t in 1:nSamples
        if t > maxT
            break
        end
    
        # get reaction rates for current concentration
        rates = rates_out[i_t,:]
        # get timestep and index of next reaction
        (dt, ind) = next_sample(rng, t -> rates, sum(rates))

        if current_event < n_events && t+dt > external_events[current_event]
            # if external event occurs before next sample apply 
            # event function
            t = external_events[current_event]
            xs[i_t+1,:] = eventFunction(xs[i_t,:], params, rng)
            current_event += 1
        elseif dt != Inf 
            # otherwise perform sampled transition
            t += dt
            xs[i_t+1,:] = xs[i_t,:] + transitions[ind,:]
        else
            # no transitions left
            break
        end
        ts[i_t+1] = t
        rates_out[i_t+1,:] = F(xs[i_t+1,:], params)
    end
    # only return samples and not the whole array
    inds = ts .> 0.0
    inds[1] = true
    return (rates_out[inds,:], xs[inds,:], ts[inds])
end

"""
next_event_time(rate::Function, max_rate::Float64, rng::AbstractRNG)::Float64
Generate a new event from an inhomogeneous poisson process with rate Lambda(t).
Based on (Ogata’s Modified Thinning Algorithm: Ogata,  1981,  p.25,  Algorithm  2)
see also https://www.math.fsu.edu/~ychen/research/Thinning%20algorithm.pdf
# Arguments
- `rate`: rate(dt) has to be defined outside (e..g t-> rate(t+t0,args))
- `max_rate`: maximal rate in near future (has to be evaluated externally)
- `rng`: random number generator
# Output
* returns the next event time
"""
function next_time(rng::AbstractRNG, rate::Function, max_rate::Float64)::Float64
    dt = 0.0
    theta = 1.0 / max_rate
    while true
        # generate next event from bounding homogeneous Poisson process with max_rate
        dt += randexp(rng) * theta
        # accept next event with probability rate(t)/rate_max [Thinning algorithm]
        if rand(rng) < rate(dt) / max_rate
            return dt
        end
    end
end

"""
Generate a new event id from a collection of inhomogeneous poisson processes with
rates Lambda(t).
# Arguments
- `rates`: rates(dt); Float -> [Float]
- `max_rate`: maximal rate in near future (has to be evaluated externally)
- `rng`: random number generator
# Output
* returns the next event id
"""
function next_event(rng::AbstractRNG, rates::Array{Float64})::Int
    cumulated_rates = cumsum(rates)
    sum_rate = cumulated_rates[end]

    theta = rand(rng) * sum_rate
    id = 1
    # catch lower-bound case that cannot be reached by binary search
    if theta >= cumulated_rates[1]
        # binary search
        index_l = 1
        index_r = length(cumulated_rates)
        while index_l < index_r - 1
            # index_m = floor(Int,(index_l+index_r)/2)
            index_m = fld(index_l + index_r, 2)
            if cumulated_rates[index_m] < theta
                index_l = index_m
            else
                index_r = index_m
            end
        end
        id = index_r
    end

    return id
end

"""
Generate a new event from a collection of inhomogeneous poisson processes with
rates Lambda(t).
# Arguments
- `rates`: rates(dt); Float -> [Float]
- `max_rate`: maximal rate in near future (has to be evaluated externally)
- `rng`: random number generator
# Output
* returns the next event time and event id as tuple (dt, id)
"""
function next_sample(rng::AbstractRNG, rates::Function, max_rate::Float64)::Tuple{Float64,Int}
    if max_rate == 0.0
        return Inf, -1
    end
    rate(t) = sum(rates(t))
    dt = next_time(rng, rate, max_rate)
    id = next_event(rng, rates(dt))
    return dt, id
end
