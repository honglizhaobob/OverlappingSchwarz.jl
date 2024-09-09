# Development mode 
using Revise

# Package in development
using OverlappingSchwarz

# Main file for testing


# Implementation of gradient-based search method for minimizing the action
# functional of a constrained path. 
#
# We consider the noise-perturbed system:
#
#   dX/dt = b(X) + √ϵ⋅σ⋅η, X(0) = x
# 
# And consider an extreme event at time T: { F(X(T)) > γ } 
# for some performance function F(⋅), and threshold γ.
#
#
# ----------------------------------------------------------------------
# Import libraries
# ----------------------------------------------------------------------
using LinearAlgebra
using Statistics
using Random
using Plots
using JLD2


Random.seed!(20);

## (12/24/23) specific problem instance for the 2d problem in:
# https://arxiv.org/abs/2303.11919
##################################################
# define parameters for SDE
##################################################

# helper functions for the SDE
# 
# [ dx1/dt ] = (-x1 - x1*x2)dt + √ϵ      ⋅ dW1
# [ dx2/dt ] = (-4x2 + x1^2)dt + (1/2)√ϵ ⋅ dW2
#
function get_b(x)
    """ 
        Evaluate drift at a point.
    """
    b = zeros(2);
    b[1] = -x[1]-x[1]*x[2];
    b[2] = -4x[2] + x[1]^2;
    return b
end

function get_sigma()
    """ 
        Evaluate correlation matrix σ
    """
    return diagm([1.0, 0.5]);
end

function get_A()
    """ 
        Evaluate covariance σσᵀ
    """
    return diagm([1.0, 0.25]);
end

function get_A_inv()
    """ 
        Inverse of covariance matrix
    """
    return diagm([1.0, 4.0]);
end

function get_b_grad(x)
    """
        Evaluate gradient of drift (a matrix).

        Each row i is ∇ₓb_i
    """
    return [
        -1-x[2] -x[1]
        2x[1] -4
    ]
end

function get_hess_b_transform(p)
    """
        Evaluates <∇²b(x), p> such that the (i, j)th entry 
        is sum_k=1^n p_k * (∂_i∂_j b_k)
    """
    # ∇_x(∇_xb_1) = 0, ∇_x(∇_xb_2) = [2 0; 0 0] so that
    # <∇²b(x), p> = [2p(2) 0; 0 0]
    return [
        2p[2] 0;
        0 0
    ];
end


function get_f(x)
    """ 
        Evaluate failure criterion.
    """
    return x[1]+2x[2]
end

function get_f_grad(x)
    """ 
        Evaluate gradient of f.
    """
    return [1., 2.];
end

function get_f_hessian_transform()
    """ Hessian of f is 0. """
    return [0.; 0.];
end

## Instanton equations helper
####
function action_gradient(p, phat)
    """ 
        Gradient of action with respect to control, 
        derived from adjoint state method.
    """
    return (get_A()*(p-phat)')';
end

function action_finite_difference_hessian_transform(p, phat, u, alpha=1e-6)
    """ 
        Computes hessian path product by finite differencing 
        on the Jacobian (of cost functional).

        HessJ(u) ≈ (∇J(p + αu) - ∇J(p))/α for α very small.
    """
    return (action_gradient(p + alpha*u, phat) - action_gradient(p, phat))./alpha;
end

function forward_integration!(
    x, xinit, p, dt
)
    """
        Integrate the state governing equations forward
        in time.

        forward Euler scheme is used. Space for the path 
        is pre-allocated. In this routine, control path is 
        considered fixed (modified in place).
    """
    nt = size(x, 1)-1;
    x[1, :] .= xinit;
    for i = 2:nt+1
        x[i, :] .= x[i-1, :] + (get_b(x[i-1, :]) + get_A()*p[i-1, :])*dt;
    end
end

function backward_integration!(
    phat, phatinit, x, dt
)
    """ 
        Integrate conjugate momentum backwards in time. Space for 
        path is pre-allocated. In this routine, state path is considered
        fixed (modified in place).

        Technically, this is backward Euler (more stable), since we are integrating
        backwards in time, that is:

        Discretize:
        dp/dt = -∇bᵀ(x)⋅p => (p_n - p_n-1)/Δt = -∇bᵀ(x_n)⋅p_n
        p_n-1 = p_n + ∇bᵀ(x_n)⋅p_n⋅Δt
    """
    nt = size(x, 1)-1;
    phat[nt+1, :] .= phatinit;
    for i = nt:-1:1
        phat[i, :] .= phat[i+1, :] + get_b_grad(x[i+1, :])'*phat[i+1, :]*dt;
    end
end

function loop_integrate!(
    x, p, phat, xinit, z, lambda, mu, dt
)
    """ 
        Complete 1 round of both a forward and backward pass of integration.
    """
    # forward
    forward_integration!(x, xinit, p, dt);
    # evaluate criterion at last time
    obs = get_f(x[end, :]);

    # initialize backward problem (unconstrained penalty method)
    phatinit = (lambda + mu*(z - obs))*get_f_grad(x[end, :]);
    # backward
    backward_integration!(phat, phatinit, x, dt);
    return obs;
end

function forward_second_integration!(
    xi, xi_init, x, u, dt
)
    """ 
        Integrates the second-order adjoint forward equations with forward Euler

        x is instanton path.
        u is an arbitrary forcing, another path that is input.
    """
    nt = size(x, 1)-1;
    xi[1, :] .= xi_init;
    for i = 2:nt+1
        xi[i, :] .= xi[i-1, :] .+ ( get_b_grad(x[i-1, :])*xi[i-1, :] + get_A()*u[i-1, :] )*dt;
    end
end

function backward_second_integration!(
    eta, eta_init, x, phat, xi, dt
)
    """
        Integrates the second order adjoint backward equations.

        x : instanton path 
        p : control 
        xi : forward equation results
    """
    nt = size(x, 1)-1;
    # terminal condition is always zero in our example.
    eta[nt+1, :] .= eta_init;
    for i = nt:-1:1
        eta[i, :] .= phat[i+1, :] .- (
                        get_hess_b_transform(phat[i+1, :])*xi[i+1, :] 
                        .- get_b_grad(x[i+1, :])'*eta[i+1, :]
                    )*dt;
    end
end

function loop_second_integrate!(
    x, phat, xi, eta, xi_init, u, lambda, dt
)
    """
        Integrates the full second-order adjoint equations, first adjoints 
        should already be computed. 

        (!!!) mu is not yet supported, use mu=0.

        Returns cost functional's hessian transformed input.
    """
    # forward
    forward_second_integration!(xi, xi_init, x, u, dt);
    
    # backward
    eta_init = lambda * get_f_hessian_transform();
    backward_second_integration!(eta, eta_init, x, phat, xi, dt);

    # hessian path product
    hessian_u = (get_A()*(u-eta)')'
    return hessian_u;
end

## Optimization routine
####
function action(x, dt)
    """ 
        Evaluates path action (path integral).
    """
    # first evaluate spatial norm
    norm_t = sum(x.*(get_A()*x')', dims=2);
    # integrate over t, trapezoidal rule
    res = 0.0;
    nt = length(norm_t)-1;
    for i = 2:nt+1
        res = res + (dt/2)*(norm_t[i] + norm_t[i-1]);
    end
    res = (1/2)*res;
    return res;
end

function gradient_norm(dLdp, dt)
    """ 
        Evaluates L^2 norm of action gradient 
        (with respect to control) path.
    """
    norm_t = sum(dLdp .* dLdp, dims=2);
    res = 0.0;
    nt = length(norm_t)-1;
    for i = 2:nt+1
        res = res + (dt/2)*(norm_t[i] + norm_t[i-1]);
    end
    return sqrt(res);
end

function path_innerprod(a, b, dt)
    """ 
        Computes inner product of two paths in L^2.
    """
    innerprod_t = sum(a .* b, dims=2);
    res = 0.0;
    nt = length(innerprod_t)-1;
    for i = 2:nt+1
        res = res + (dt/2)*(innerprod_t[i] + innerprod_t[i-1]);
    end
    return res;
end

function sqp_subproblem_action_finite_difference(
    d, x, p, phat, dt, xinit, z, action, action_grad, alpha=1e-8
)
    """
        Subproblem of sqp, which computes the modified action (of J):

            I(d) = J(p) + <(dJ/dp), d> + (1/2)*<HessJ(d), d>
        The HessJ transform is computed using finite difference instead 
        of exactly using second-order adjoint ODEs.

        Perturbing the Jacobian requires solving the first order equation again
        with input control (p + u).
    """
    # unperturbed gradient
    old_grad = copy(action_grad);
    # evaluate perturbed gradient
    xnew = copy(x);
    # perturb the control
    pnew = copy(p) + alpha*d;
    phatnew = copy(phat);
    # solve first order equation
    forward_integration!(xnew, xinit, pnew, dt);
    # evaluate objective
    obs = get_f(xnew[end, :]);
    # initialize backward problem (unconstrained penalty method)
    phatinit = (lambda + mu*(z - obs))*get_f_grad(xnew[end, :]);
    # backward
    backward_integration!(phatnew, phatinit, xnew, dt);
    # perturbed gradient
    new_grad = action_gradient(pnew, phatnew);
    # Hessian transformed input 
    finite_difference_hessian_d = (new_grad-old_grad)./alpha;
    # compute subproblem objective, also return its gradient at d
    return action + path_innerprod(action_grad, d, dt) + 0.5*path_innerprod(finite_difference_hessian_d, d, dt), action_grad + finite_difference_hessian_d;
end

function sqp_subproblem_action(
    d, action, action_grad, x, phat, xi, eta, xi_init, u, lambda, dt
)
    """ 
        Computes hessian transform exactly.
    """
    error();
end
maxiter = 100;
# noise amplitude for the process
eps = 0.5;
# temporal grid
t_start = 0.0;
t_end = 1.0;
nt = 2001;
tgrid = collect(range(t_start, t_end, nt+1));
dt = tgrid[2]-tgrid[1];
# number of dimenions
d = 2;
# initial condition for state equations
xinit = [0., 0.];
# target observable 
z = 3.0;
# penalty parameter for terminal condition
mu = 0.0;
# penalty parameter for Lagrangian optimality condition
lambda = 1.0;
# initialize space for storing paths
x = zeros(nt+1, d);
# initialize control variables for optimization (random initialization or warm start)
p = randn(nt+1, d);
# initialize adjoint-state control variables
phat = zeros(nt+1, d);

# initial descent steps for armijo line search
sigma_init = 0.9;
sigma_min = 0.1;
beta = 0.9;
c = 0.1

mode = "gradient_descent";
# save gradient norm history to see convergence
all_gradnorm = zeros(maxiter);
#### Gradient descent 
if mode == "bare"
    # repeatedly solving the instanton equations without any 
    # gradient update should correspond to step size 1 
    # gradient descent
    gradnorm_threshold = 1e-8;  # in this bare case, we are only minimizing (1/2)*||p||_a^2-lambda(f(T)-obs)
    lambda = 2.5;
    for i = 1:maxiter
        # repeatedly integrate forward and backward
        forward_integration!(x, xinit, p, dt);
        # get terminal condition
        obs = get_f(x[end, :]);
        pinit = lambda*get_f_grad(x[end, :]);
        # integrate backward
        backward_integration!(p, pinit, x, dt);
        # check gradient convergence
        gradnorm = 0.5*sqrt(path_innerprod(p, p, dt));
        all_gradnorm[i] = gradnorm;
        println(gradnorm)
    end
    #jldsave("./data/bare.jld2", x=all_gradnorm);
end

if mode == "gradient_descent"
    # gradient norm threshold for early stopping
    gradnorm_threshold = 1e-6;
    sigma2 = 0.1;
    reduction = 1;
    for i in 1:maxiter
        println(i)
        
        # complete loop integration
        obs = loop_integrate!(
            x, p, phat, xinit, z, lambda, mu, dt
        );
        # compute action gradient
        dLdp = action_gradient(p, phat);

        # compute action value
        action_value = action(p, dt) + lambda*(obs-z) + mu*(obs-z)^2/2;
        println(action_value)
        
        # break condition from gradient norm
        gradnorm = gradient_norm(dLdp, dt);
        println(gradnorm);
        #if gradnorm < gradnorm_threshold
        #    println("Early stopping ...")
        #    break;
        #end
        # compute search direction (preconditioned by covariance)
        #s = -(get_A()\dLdp')';
        # (not preconditioned)
        s = -dLdp;
        # take a small, fixed descent step
        #p[:, :] .= p[:, :] .+ 0.1*s[:, :];

        # armijo line search
        sigma = sigma_init;
        while sigma > sigma_min 
            # try descent 
            p_copy = copy(p);
            x_copy = copy(x);
            p_copy[:, :] .= p_copy[:, :] .+ sigma*s[:, :];
            # integrate forward and evaluate action
            forward_integration!(x_copy, xinit, p_copy, dt);
            new_obs = get_f(x_copy[end, :]);
            new_action_value = action(p_copy, dt) + lambda*(new_obs - z) + (mu/2)*(new_obs - z)^2;
            if action_value <= new_action_value + c*sigma*path_innerprod(dLdp, s, dt)
                break;
            end
            sigma = sigma * beta;
        end
        println("... Armijo final descent step length = ", sigma);
        p[:, :] .= p[:, :] .+ sigma2*s[:, :];
        global sigma = sigma2 * reduction;

        # save gradient norm history
        all_gradnorm[i] = gradnorm;
    end
    #jldsave("./data/gd.jld2", x=all_gradnorm);
elseif mode == "cg"
    # conjugate gradient method with fixed step length
    # Applied in paper: Nonlinear conjugate gradient methods for the optimal control of laser surface hardening 
    maxiter = 50;
    gradient_norm_threshold = 1e-6;
    counter = 0;
    # initialize descent direction
    loop_integrate!(x, p, phat, xinit, z, lambda, mu, dt);
    # compute action gradient
    dLdp = action_gradient(p, phat);

    # compute action value
    obs = get_f(x[end, :]);
    action_value = action(p, dt) + lambda*(obs-z) + mu*(obs-z)^2/2;

    # break condition from gradient norm
    gradnorm = gradient_norm(dLdp, dt);

    # initial descent direction
    #s = -(get_A()\dLdp')';
    s = -dLdp;
    while counter < maxiter
        # if gradnorm < gradient_norm_threshold
        #     break;
        # end
        global gradnorm;
        println(counter);
        println(gradnorm)
        # take a step 
        p[:, :] .= p[:, :] .+ 0.1 * s[:, :];
        # compute new action value and gradient 
        loop_integrate!(x, p, phat, xinit, z, lambda, mu, dt);
        obs = get_f(x[end, :]);
        dLdp_old = copy(dLdp);
        dLdp[:, :] .= action_gradient(p, phat);
        gradnorm = gradient_norm(dLdp, dt);
        action_value = action(p, dt)+lambda*(obs-z) + mu*(obs-z)^2/2;
        println(action_value)
        # Fletcher-Reeves
        betafr = (gradient_norm(dLdp, dt)/gradient_norm(dLdp_old, dt))^2;
        # next round's descent step
        s[:, :] .= -dLdp[:, :] .+ betafr * s[:, :];

        global counter = counter + 1;
        

        all_gradnorm[counter] = gradnorm;
    end
    jldsave("./data/cg.jld2", x=all_gradnorm);
elseif mode == "sqp"
    #reduction = 0.9;
    sigma = 1.1;
    # sequential quadratic programming
    # gradient norm threshold for early stopping
    gradnorm_threshold = 1e-6;
    for i in 1:maxiter
        println(i)
        
        # complete loop integration
        obs = loop_integrate!(
            x, p, phat, xinit, z, lambda, mu, dt
        );
        # compute action gradient
        dLdp = action_gradient(p, phat);

        # compute action value
        action_value = action(p, dt) + lambda*(obs-z) + mu*(obs-z)^2/2;
        #println(action_value)
        
        # break condition from gradient norm
        gradnorm = gradient_norm(dLdp, dt);
        # println(gradnorm)
        # if gradnorm < gradnorm_threshold
        #     println("Early stopping ...")
        #     break;
        # end
        # compute search direction by running CG on approximate quadratic programming
        # https://andrew.gibiansky.com/blog/machine-learning/hessian-free-optimization/

        # randomly initialize
        global d = randn(nt+1, 2);
        # sqp convergence 
        inner_loss, inner_grad = sqp_subproblem_action_finite_difference(d, x, p, phat, dt, xinit, z, action_value, dLdp);
        inner_maxiter = 5000;
        inner_counter = 0;
        # initial search direction for inner QP
        delta_d = -(get_A()\inner_grad')';
        inner_break_condition = 1e-10;
        while inner_counter <= inner_maxiter
            if inner_counter == 0
                println(">>> In inner QP ... ")
            end
            #println(inner_counter);
            # try a small step
            d[:, :] .= d[:, :] .+ 0.1.*delta_d[:, :];
            # compute new objective and its gradient
            inner_loss_new, inner_grad_new = sqp_subproblem_action_finite_difference(d, x, p, phat, dt, xinit, z, action_value, dLdp);
            # Fletcher-Reeves
            betafr = (path_innerprod(inner_grad_new, inner_grad, dt)/gradient_norm(inner_grad, dt))^2;
            # attempt to solve the inner QP with conjugate gradient 
            inner_grad_norm = gradient_norm(inner_grad_new, dt)^2;
            #println(inner_grad_norm)
            if inner_grad_norm < inner_break_condition
                break;
            end
            # next round's descent step
            delta_d[:, :] .= -inner_grad_new[:, :] .+ betafr * delta_d[:, :];
            inner_counter = inner_counter + 1;
        end
        # after solving inner QP, we have approximate Newton step
        # take Newton step for outer problem 
        p[:, :] .= p[:, :] .+ sigma*d[:, :];
        global sigma = sigma * reduction;

        all_gradnorm[i] = gradnorm;
    end
    jldsave("./data/sqp.jld2", x=all_gradnorm);
else
    error();
end