######################################################################
# Stand-alone optimal control problem (OCP) definitions 
######################################################################
abstract type OptimalControlProblem end;

mutable struct LinearQuadraticProblem <: OptimalControlProblem
    """
        Represents an instance of a time-varying linear quadratic problem. 
        The general form is:

            min (1/2)∫₀ᵀ(u(t)-u_d(t))^TL(t)(u(t)-u_d(t)) + 
                        (x(t)-x_d(t))^TD(t)(x(t)-x_d(t)) dt +
                (1/2)∫₀ᵀ(x(T)-x_d(T))^TS(x(T)-x_d(T))

        subject to:
            dxdt = A(t)x + B(t)u 
            x(0) = x₀

        Log:
            (05/03/2024) Starting with assuming all matrices are constant.
    """
    # initial condition 
    x0 :: Vector
    # temporal grid 
    t :: Vector
    # data for path tracking 
    xd :: Matrix 
    ud :: Matrix

    # differential constraint matrices
    A :: Matrix
    B :: Matrix

    # penalty matrices 
    D :: Matrix
    L :: Matrix 
    S :: Matrix
end

function solve(prob :: LinearQuadraticProblem, alpha=1e-2, 
    maxiter=1e+3, tol=1e-4, verbose=true, print_every=100
);
    """ 
        Solves the minimization of linear quadratic problem. 

        The solution requires solving a two-point boundary 
        value problem, which we do by gradient descent and 
        random initialization of control path.

        By default, uses forward Euler integrator.

        Inputs:
            prob (struct)
                An instance of LinearQuadraticProblem.
            alpha (float)
                gradient descent step size.
            maxiter (int)
                maximum number of gradient descent iterations.
            tol (float)
                gradient norm tolearance parameter for early stopping.
        Outputs:
            x, lambda, u (matrix)
                optimal state, adjoint and control paths.
    """
    nt = length(prob.t);
    dt = prob.t[2]-prob.t[1];
    # state dimensions 
    ndims = length(prob.x0);
    # control initialization 
    u = randn(ndims, nt);
    x = zeros(ndims, nt);
    lambda = zeros(ndims, nt);
    for n = 1:maxiter
        # forward integration 

        # forward Euler
        x[:, 1] .= prob.x0;
        for i = 2:nt
            x[:, i] .= x[:, i-1] .+ dt*(
                prob.A * x[:, i-1] + prob.B * u[:, i-1]
            );
        end

        # evaluate terminal cost
        obs = 0.5*(x[:,end]-prob.xd[:,end])'*prob.S*(x[:,end]-prob.xd[:,end]);
        # backward integration 
        lambda[:, end] .= prob.S * (x[:, end] - prob.xd[:, end]);
        for i = nt-1:-1:1
            lambda[:, i] .= lambda[:, i+1] + 
                dt * (prob.A' * lambda[:, i+1] - prob.D * (x[:, i+1] - prob.xd[:, i+1]));
        end
        grad_u = prob.B'*prob.L * (u - prob.ud) - prob.B' * lambda;
        # gradient descent 
        u[:, :] .= u[:, :] .- alpha * grad_u[:, :];

        # check for early stopping 
        gradnorm = sum(sum(grad_u .^ 2, dims=1))*dt;
        if verbose && iszero(mod(n, print_every))
            println("----------");
            println("** Iter = $(Int(n)), Gradient norm = $(gradnorm)");
            println("** Terminal observation = $(obs)");
            println("----------");
        end
        if gradnorm < tol 
            break;
        end
    end
    return x, lambda, u;
end

function update_initial!(
    prob :: OptimalControlProblem, x0 :: Vector
)
    """ 
        Updates the initial conditions of a given problem.
    """
    prob.x0[:] .= x0[:];
end

function update_terminal!(
    prob :: OptimalControlProblem, xend :: Vector
)
    """
        Updates the terminal time data of a given problem.
    """
    prob.xd[:, end] .= xend[:];
end

# solving nonlinear problems 
mutable struct NonlinearProblem <: OptimalControlProblem
    """
        Represents an instance of general Bolza-type nonlinear cost 
        functional minimization with nonlinear dynamical systems. 
        Due to the genericness of the code definition, exsitence of 
        solution is not guaranteed. Please double check before using the 
        struct that the problem is solvable. 

            min ∫₀ᵀL(t,x,u)dt + Φ(x(T))

        subject to:
            dxdt = f(t,x,u)
            x(0) = x₀

    """
    # initial condition 
    x0 :: Vector
    # temporal grid 
    t :: Vector
    # external problem parameters

    # routines for evaluating dynamics 
    get_f :: Function 
    get_f_grad_u :: Function 
    get_f_grad_x :: Function 

    # routines for evaluating cost functional 
    get_Phi :: Function
    get_Phi_grad_x :: Function 
    get_L :: Function
    get_L_grad_u :: Function
    get_L_grad_x :: Function
end

############################################################
# Other helpers
############################################################
function linear_interp(t, t0, t1, f0, f1)
    """ 
        Interpolates a vector-valued scalar function linearly.
        
        Inputs:
            t (scalar)
                query time, must be in-between t0 and t1
            t0, t1 (scalar)
                left and right 
            f0, f1 (vector)
                function values on boundary left and right
        Returns:
            ft (vector)
                ft = f0 + (f1 - f0)/(t1 - t0) * (t - t0)
    """
    @assert t0 <= t <= t1 
    if isequal(t, t0)
        return f0
    end
    if isequal(t, t1)
        return f1
    end
    df = f1 - f0; 
    dt = t1 - t0; 
    dfdt = df./dt;
    ft = f0 + dfdt .* (t - t0);
    return ft;
end

function plot2d_subproblems(prob :: OptimalControlProblem)
    """ 
        Plot a 2D piecewise solution with overlaps. 
    """
    @assert prob.ndims == 2
    p = plot(linestyle=:solid, seriestype=:scatter, markersize=1.0);
    for k = eachindex(prob.subproblems)
        plot!(p, prob.subproblems[k][1, :], prob.subproblems[k][2, :]);
    end
    return p;
end