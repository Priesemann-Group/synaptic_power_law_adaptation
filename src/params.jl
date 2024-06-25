
params = Dict{String,Float64}()

# capacities 

capacity_multiplier = 1.0
params["cycle_capacity"] = 100.0 * capacity_multiplier
params["rec_max"] = 38.0 * capacity_multiplier
params["load_max"] = 38.0 * capacity_multiplier
params["dock_max"] = 20.0 * capacity_multiplier
params["prime_max"] = 4.0 * capacity_multiplier
params["sprime_max"] = 4.0 * capacity_multiplier
params["fuse_max"] = params["cycle_capacity"]

# time constants

params["tau_rec_load"] = 0.5 # s
params["tau_load_dock"] = 1.0 # s
params["tau_dock_prime"] = 0.1 # s
params["tau_generation"] = 100.0 / 16.0 * 60.0 * 60.0 # s ; slot refilled in 6.25 h
# determines how many vesicles can be generated at a time
params["max_vesicle_generation"] = 100.0

# rates 

params["rate_rec_load"] = 1 / params["tau_rec_load"] # 1/s
params["rate_load_dock"] = 1 / params["tau_load_dock"] # 1/s
params["rate_dock_load"] = 0.0 # 1/s
params["rate_dock_prime"] = 1 / params["tau_dock_prime"] # 1/s
params["rate_endo"] = 10.0 / 30.0 # 1/s
params["rate_generation"] = 1.0 / params["tau_generation"] # 1/s 

params["p_aging"] = 1.0 / 200.0 # probability
params["p_fuse_prime"] = 0.1 # probability
params["p_fuse_sprime"] = 0.5 # probability

# rates for single pool model
params["tau_fuse_prime"] = 0.2 # s
params["rate_fuse_prime"] = 1 / params["tau_fuse_prime"] # 1/s 
