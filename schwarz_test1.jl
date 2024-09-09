# (05/06-05/09) Test 1: 
# linear quadratic control that is stiff
# ----------------------------------------------------------------------
includet("OverlappingSchwarz.jl");
using .OSP

# import standard libraries
using LinearAlgebra;
using Plots, LaTeXStrings
using Random; Random.seed!(1);
using Interpolations
# define problem parameters

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

# simulate and visualize trajectory 
# x0 = 0.5*ones(ndims);
# x = zeros(ndims, nt);
# x[:, 1] = x0;
# # random control 
# u = (sqrt(dt)/dt)*randn(ndims, nt);
# for n = 2:nt 
#     x[:, n] = x[:, n-1] + dt * (A * x[:, n-1] + B * u[:, n-1]);
# end
# plot(x[1, :], x[2, :], linewidth=2.0)

# solve control with a fixed initial condition
solve_global = true;
# initial condition for the full problem
x0 = [-1.0, -2.0];
if solve_global
    # create linear quadratic control
    prob = OSP.LinearQuadraticProblem(
        x0, t, xd, ud, A, B, D, L, S
    );
    x, lambda, u = OSP.solve(prob);
    global p = plot(x[1, :], x[2, :], 
        linestyle=:solid, linewidth=2.5, color=:blue, 
        label="",
        dpi=300
    );
    plot!(p,title="Global",xlabel=L"$x_1$",ylabel=L"$x_2$");
    scatter([x[1,1]],[x[2,1]],label=L"$t = 0$",markersize=3.0,color=:black); 
    scatter([x[1,end]],[x[2,end]],label=L"$t = T)$",markersize=3.0,color=:black); 
    savefig(p, "./fig/global2.png");
end



#error()
############################################################
# initialize Schwarz problem
knots = [3.0, 7.0];
# extensions 
tau_left = [0.0, 0.3, 0.3];
tau_right = [0.3, 0.3, 0.0];
prob = OSP.SchwarzProblem(
    ndims, tstart, tend, knots, tau_left, tau_right
);
println("* Overlapped intervals: $(prob.doms)");
println("* Maximum overlap size = $(OSP.max_overlap(prob))");

#error()
# initialize all subproblems

# varying time step sizes 
all_dt = [8e-3, 1.5e-2, 1.5e-2];
for n = 1:prob.num_probs
    # initial conditions 
    if n == 1
        # first subproblem has fixed initial condition
        sub_x0 = x0;
    else
        # initialize randomly 
        sub_x0 = randn(ndims);
    end

    # specify time grid and ODE solver 
    sub_tstart = prob.doms[n][1]; sub_tend = prob.doms[n][2];
    sub_nt = Int(ceil((sub_tend-sub_tstart)/all_dt[n]));
    sub_t = collect(range(sub_tstart, sub_tend, sub_nt));
    # initialize state tracking (terminal point randomly initialized)
    sub_xd = zeros(ndims, sub_nt);
    if n < prob.num_probs
        # last interval does not require tracking
        sub_xd[:, end] = randn(ndims);
    else
        # last interval uses original problem terminal condition 
        sub_xd[:, end] = xd[:, end];
    end
    sub_ud = zeros(ndims, sub_nt);
    prob.subproblems[n] = OSP.LinearQuadraticProblem(
        sub_x0, sub_t, sub_xd, sub_ud, A, B, D, L, S
    );
end
#error()
# begin Schwarz iterations 
max_schwarz_iterations = 5;#50;
# solutions to subproblems with overlaps 
x_overlapped = Vector{Any}(undef, prob.num_probs);
lambda_overlapped = Vector{Any}(undef, prob.num_probs);
u_overlapped = Vector{Any}(undef, prob.num_probs);

for n = 1:max_schwarz_iterations
    println("**************************************************")
    println("Schwarz Iter = $(n)");
    println("**************************************************")
    # solve (in parallel) each subproblem to optimality
    for k = 1:prob.num_probs
        global sub_x, sub_lambda, sub_u = OSP.solve(
            prob.subproblems[k], 1e-2
        );
        # store 
        x_overlapped[k] = sub_x;
        u_overlapped[k] = sub_u;
        lambda_overlapped[k] = sub_lambda;
    end
    # update initial conditions (first interval does not require update)
    for k = 2:prob.num_probs
        # left boundary time 
        t_query = prob.subproblems[k].t[1];
        # query previous interval's state

        # find index in the previous subproblem 
        prev_t = prob.subproblems[k-1].t;
        prev_x = x_overlapped[k-1];
        prev_u = u_overlapped[k-1];
        prev_lambda = lambda_overlapped[k-1];

        # index of time to the right of query
        idx = findfirst(x -> x >= t_query, prev_t);
        if isequal(prev_t[idx], t_query)
            x_query = prev_x[:, idx];
            u_query = prev_u[:, idx];
            lambda_query = prev_lambda[:, idx];
        else
            # linear interpolation 
            x_query = OSP.linear_interp(t_query, 
                prev_t[idx-1], prev_t[idx], 
                prev_x[:, idx-1], prev_x[:, idx]
            );
            u_query = OSP.linear_interp(t_query, 
                prev_t[idx-1], prev_t[idx], 
                prev_u[:, idx-1], prev_u[:, idx]
            );
            lambda_query = OSP.linear_interp(t_query, 
                prev_t[idx-1], prev_t[idx], 
                prev_lambda[:, idx-1], prev_lambda[:, idx]
            );
        end
        # update initial condition of current interval 
        OSP.update_initial!(prob.subproblems[k], collect(x_query));
    end
    # update terminal conditions (last interval does not require update) 
    for k = 1:prob.num_probs-1
        # right boundary time 
        t_query = prob.subproblems[k].t[end];
        # query next interval's state and adjoint 
        
        # find index in next interval
        next_t = prob.subproblems[k+1].t;
        next_x = x_overlapped[k+1];
        next_u = u_overlapped[k+1];
        next_lambda = lambda_overlapped[k+1];
        
        idx = findfirst(x -> x >= t_query, next_t);
        
        # linear interpolation including boundary
        x_query = OSP.linear_interp(t_query, 
                next_t[idx-1], next_t[idx], 
                next_x[:, idx-1], next_x[:, idx]
            );
        u_query = OSP.linear_interp(t_query, 
                next_t[idx-1], next_t[idx], 
                next_u[:, idx-1], next_u[:, idx]
            );
        lambda_query = OSP.linear_interp(t_query, 
                next_t[idx-1], next_t[idx], 
                next_lambda[:, idx-1], next_lambda[:, idx]
            );
        # update terminal data as derived 
        terminal_data = collect(x_query)-(prob.subproblems[k].S\collect(lambda_query));
        OSP.update_terminal!(prob.subproblems[k], terminal_data);
    end
end

p = plot(linestyle=:solid, seriestype=:scatter, markersize=2.0);
for k = eachindex(prob.subproblems)
    plot!(p, x_overlapped[k][1, :], 
        x_overlapped[k][2, :], linewidth=2.0,
        linestyle=:dashdot,
        label="Subproblem $(k)"
    );
end

tmp = OSP.schwarz_size(prob);
tmp = maximum(tmp);
plot!(p, title="Local", dpi=300);
plot!(p, xlabel=L"$x_1$", ylabel=L"$x_2$");
savefig(p, "./fig/local3.png");