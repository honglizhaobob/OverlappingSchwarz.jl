using Distributed
using Random
Random.seed!(10);
# Total number of Monte Carlo samples
nmc = 100_000_000;

### 1. Non-parallel version ###

# Benchmark the time for single-threaded execution
t1 = @elapsed begin
    sum_uniforms = 0.0
    for i = 1:nmc
        global sum_uniforms += rand()  # Generate random numbers in a single thread
    end
end

average_uniforms = sum_uniforms / nmc
println("Average (without parallel): $average_uniforms")
println("... Elapsed time without parallel computations = $t1")

println("\n");
### 2. Parallel version using Distributed computing ###

# Add 4 worker processes
num_workers = 8;
addprocs(num_workers)
@everywhere using Random  # Load Random module on all workers

# Divide the total number of samples equally across workers
nmc_per_worker = nmc / nworkers()

t2 = @elapsed begin
    # Distributed sum calculation
    sum_random_numbers = @distributed (+) for i in 1:(nmc_per_worker * nworkers())
        rand()  # Each worker generates random numbers
    end
end

average_random_numbers = sum_random_numbers / (nmc_per_worker * nworkers())
println("Average (with parallel): $average_random_numbers")
println("... Elapsed time with 8 parallel workers = $t2")

# Clean up workers (optional if you're done)
rmprocs(workers())
