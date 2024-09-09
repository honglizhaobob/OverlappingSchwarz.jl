# Development mode 
using Revise

# Package in development
using OverlappingSchwarz

# other packages for this file 
using Random;
using Plots; gr();

# fix seed for reproducibility
Random.seed!(1);

##################################################
# Define problem parameters
##################################################
# time grid
tstart, tend = 0, 10.0;
ndims = 2;

nt = 601;
t = collect(range(tstart, tend, nt));
dt = t[2]-t[1];
# control and state targets
xd = zeros(ndims, nt);
ud = zeros(ndims, nt);

# differential constraint matrices
A = [-1. 0; 0 -100.];
B = [2. 0; 0 2.];

# penalty matrices 
D = zeros(ndims,ndims);
D = [1. 0; 2. 0];
L = [1. 0; 0 1.];
# penalty parameter for terminal condition
mu = 1.0;
S = mu * [1. 0; 0 1.];

# initial condition for the full problem
x0 = [-1.0, -2.0];

# create linear quadratic control
prob = OverlappingSchwarz.LinearQuadraticProblem(
    x0, t, xd, ud, A, B, D, L, S
);
x, lambda, u = OverlappingSchwarz.solve(prob);

p = plot(x[1, :], x[2, :], 
    linestyle=:solid, linewidth=2.5, color=:blue, 
    label="",
    dpi=300
);