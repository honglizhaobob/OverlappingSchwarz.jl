# Utility functions for computing path-space optimization
######################################################################
# libraries and modules
######################################################################
using LinearAlgebra
using Statistics
using Random
using Base
using Plots; gr();
######################################################################
mutable struct Path
    """ 
        A path object in the Hilbert space of L^2([0, T] x R^d). 
        For numerical computations, the time dimension is discretized.
        Furthermore, d is finite (infinite case, or PDE constraints, 
        are to be updated). Inner products are defined on this space
        accordingly.
    """
    # positions over time in R^d, effectively a matrix of dimension (nt x d)
    x 
    # number of recorded time steps
    nt
    # time discretization
    dt
    # final time
    t_end
    # temporal grid
    tgrid
    # number of dimensions (implied by x)
    ndims
    function Path(tgrid, x)
        @assert size(x, 1) == length(tgrid)
        @assert iszero(tgrid[1])
        nt = length(tgrid);
        dt = tgrid[2]-tgrid[1];
        t_end = tgrid[end];
        ndims = size(x, 2);
        return new(
            x, nt, dt, t_end, tgrid, ndims
        );
    end
end

function Path(t_start, t_end, nt, d)
    """ 
        Constructor that randomly initializes a path.
        Effectively white noise with noise 1.0 
    """
    # discretize time
    tgrid = LinRange(t_start, t_end, nt);
    nt = length(tgrid);
    # generate random path
    x = randn(nt, d);
    return Path(
        tgrid, x
    );
end

function Base.:zeros(t_start, t_end, nt, d)
    """
        Construct a path of all zeros.
    """
    # discretize time
    tgrid = LinRange(t_start, t_end, nt);
    nt = length(tgrid);
    # generate random path
    x = zeros(nt, d);
    return Path(
        tgrid, x
    );
end

# an operator abstract type that maps a path f to the same space
abstract type PathOperator end

struct OrthogonalProjection <: PathOperator
    """
        Projects a path onto its orthogonal space in L^2 
        
        Namely, it is an operator Proj such that

            g = Proj(f)
            g * f == 0
    """
    f :: Path
    g :: Path 
    OrthogonalProjection(f, g) = new(f, g);
end

struct IdentityOperator <: PathOperator
    """ 
        The identity operator in path space, returns the path itself.
    """
    IdentityOperator() = new();
end

struct ConstantLinearOperator <: PathOperator
    """
        An operator that applies a constant linear transformation 
        to all points on the path.
    """
    mat
    ConstantLinearOperator(mat) = new(mat);
end

struct SymmetricConstantLinearOperator <: PathOperator
    """
        An extension to the constant linear transformation operator 
        by ensuring the transformation is symmetric.
    """
    mat 
    function SymmetricConstantLinearOperator(mat)
        @assert issymmetric(mat);
        return new(mat);
    end
end

mutable struct ParameterizedLinearOperator <: PathOperator
    """
        An extension to the constant linear operator, however,
        the matrix can depend on another path. 
    """
    # base path 
    param :: Path
    # a function that returns a matrix of comparable dimensions given a point
    mat_func :: Function
    ParameterizedLinearOperator(param, mat_func) = new(param, mat_func);
end

mutable struct SpatioTemporalOperator <: PathOperator
    """
        Implements an operator mapping given path x(t):
            x(t) ↦ b(x(t), t) ≡ y(t)
    """
    # pointwise mapper, of form b(x, t)
    mapping :: Function 
    SpatioTemporalOperator(mapping) = new(mapping);
end

mutable struct ParameterizedSpatioTemporalOperator <: PathOperator
    """
        Implements an operator mapping that is parameterized by 
        another path. For instance, Jacobian query of a function 
        of a path:
            x(t) ↦ f(x(t), t; u(t))
    """
    # parameter path associated with the operator
    param :: Path
    # pointwise mapper, of form f(u, x, t)
    mapping :: Function 
    ParameterizedSpatioTemporalOperator(param, mapping) = new(param, mapping);
end

function update!(operator :: ParameterizedSpatioTemporalOperator, new_param :: Path) 
    """
        Updates the parameters stored in the path operator.
    """
    operator.param = copy(new_param);
end

# third order path operators; when taken inner product with a path, returns a path operator 
abstract type ThirdOrderPathOperator end 

mutable struct ParameterizedLinearThirdOrderOperator <: ThirdOrderPathOperator
    """ 
        Represents an order-3 tensor in function space. Each slice of the 
        tensor is a linear path operator. The third order tensor is itself 
        parameterized by a path; tensor diagram:
                                     |
                                --  H(x)  --
        By default, any inner product with the third order tensor will 
        contract along the middle dimension.
    """
    param :: Path

    # function of x that returns a matrix
    #   M(x, f)_ij ↦ ∑_k=1^N C_k(x)_ij * f_k, should be implemented directly
    mapping :: Function 

    function ParameterizedLinearThirdOrderOperator(param, mapping)
        return new(param, mapping);
    end
end

function update!(operator :: ParameterizedLinearThirdOrderOperator, new_param :: Path)
    """
        Updates the parameterization of a third order path operator. 
    """
    operator.param = copy(new_param);
end

mutable struct ParameterizedLinearThirdOrderOperatorContracted <: PathOperator 
    """
        For more details see `ParameterizedLinearThirdOrderOperator`. 
        This struct represents the intermediate result of a third order 
        linear operator contracted with a path (in middle index). However,
        we do not evaluate the weighted sum of matrices / sum of second order operators 
        explicitly; but the contracted path is stored, and only used when using this
        operator to transform another path. That is, this class effectively represents:

                                        f
                                        |
                                   --  H(x)  --
        When computing: 
    
                                        f
                                        |
                                   -- H(x) -- g
        For some path g, this lazy evaluation turns a:
            O(2 * nt * d^2): 
                sum along temporal dimension, nt, of d x d matrices, then multiply with
                length-d vectors at each nt, another nt * d^2
        Into:
            O(nt * d^2 + nt * d)
                Multiply with length-d vectors at each nt, then sum along nt of the vectors.
    """
    x :: Path 

    f :: Path 

    # inherited mapping from third order operator
    mapping :: Function

    function ParameterizedLinearThirdOrderOperatorContracted(x, f, mapping)
        return new(x, f, mapping);
    end
end
# a functional is a linear operator that maps a path f to a number
abstract type PathFunctional end


mutable struct PathOuterProd <: PathOperator
    """ 
        Naive outer product implementation. Given two paths, 
        represents an operator (f ⊗ g) such that:
        (f ⊗ g) h = <g, h>*f
    """
    f :: Path 
    g :: Path
    PathOuterProd(f, g) = new(f, g);
end

function outer(f :: Path, g :: Path)
    """
        Returns outer product f⊗g
    """
    @assert compatible(f, g);
    return PathOuterProd(f, g);
end

function projection(f)
    """ 
        Returns a projection operator onto space f^⟂
    """
    return OrthogonalProjection(f, f);
end

#-----------------------------------------------------------------------------
# Path generations
#-----------------------------------------------------------------------------
abstract type PathGenerator end  # an abstract data type defining a path generator
                                 # where the path is constrained by a differential problem.
mutable struct ForwardODE <: PathGenerator
    """ 
        Generates a forward path by solving the ODE of the following form:

            [ (dx/dt)(t) ] = [ b(x(t)) ] + [ C * p(t)) ]
            [ x(0) ]       = [ input(x) ]

        where the problem is solved from [0, T], in dimension R^d. 
        `b` is a function from R^d to R^d that defines the forcing term.
        `p` is the control path, considered as an input that `x` implicitly
        depends on, also transformed by a constant matrix `C`. The constant 
        matrix needs not be stored explicitly, and is available as matrix-vector 
        multiplication query.
    """
    # inputs
    x_init 
    p :: Path
    tgrid 

    # ODE properties
    C_query :: Function 
    b :: Function
    x :: Path

    function ForwardODE(
        x_init, p, tgrid, C_query, b
    )
        # current path object in memory is initialized as 0.0 (except initial condition)
        x = zeros(tgrid[1], tgrid[end], length(tgrid), length(x_init));
        # input initial condition
        x.x[1, :] .= x_init;
        return new(
            x_init, p, tgrid, C_query, b, x
        );
    end
end

function integrate!(
    prob :: ForwardODE
)
    """ 
        Integrates the forward ODE problem using forward Euler method.

        Modifies the stored path in-space.
    """
    # number of time steps to integrate 
    nt = prob.x.nt;
    dt = prob.x.dt;
    # integrate the path with forward Euler
    prob.x.x[1, :] .= prob.x_init[:];
    for i = 1:nt-1
        prob.x.x[i+1, :] .= prob.x.x[i, :] .+ ( prob.b(prob.x.x[i, :]) .+ prob.C_query(prob.p.x[i, :]) ) * dt;
    end
end

mutable struct BackwardODE <: PathGenerator
    """
        Generates a path by solving the backward ODE (or adjoint ODE) system, 
        as the following:

            [ (dp/dt)(t) ] = [ -∇ₓbᵀ(x(t)) * p(t) ] 
            [ p(T) ]       = [ input(p) ]
    """
    # inputs
    p_term                       # terminal condition of p(t) path
    x :: Path                    # forward path
    tgrid

    # ODE properties
    grad_b_transpose_query :: Function     
                                 # matrix-vector multiplication of ∇ₓbᵀ * v
    p :: Path                    # path for p (also depends on x)
    function BackwardODE(
        p_term, x, tgrid, grad_b_transpose_query, p
    )
        # current path object in memory is initialized as 0.0 (except initial condition)
        p = zeros(tgrid[1], tgrid[end], length(tgrid), length(p_term));
        # input terminal condition
        p.x[end, :] .= p_term;
        return new(
            p_term, x, tgrid, grad_b_transpose_query, p
        );
    end
end

function integrate!(prob :: BackwardODE)
    """
        Integrates the backward ODE problem using (backward Euler).
    """
    # number of time steps to integrate 
    nt = prob.p.nt;
    dt = prob.p.dt;
    # integrate the path with forward Euler
    prob.p.x[end, :] .= prob.p_term[:];
    for i = nt:-1:2
        prob.p.x[i-1, :] .= prob.p.x[i, :] .+ ( 
                prob.grad_b_transpose_query(prob.x.x[i, :], prob.p.x[i, :])
            ) * dt;
    end
end
#-----------------------------------------------------------------------------
# Path arithmetics
#-----------------------------------------------------------------------------
function compatible(x :: Path, y :: Path)
    """
        Check if two paths belong to the same numerical space.
    """
    time_axis_check = ((x.t_end == y.t_end) && (x.dt == y.dt));
    space_dimension_check = (size(x.x) == size(y.x));
    return all([time_axis_check, space_dimension_check]);
end

function compatible(operator :: PathOuterProd, h :: Path)
    """
        Check if outer product operator can be applied on path.
    """
    # enough to check for g ⊗ h the h path
    g = operator.g;
    return compatible(g, h);
end

function Base.:+(f :: Path, g :: Path)
    """
        Add two paths together
    """
    @assert compatible(f, g);
    data = copy(f.x);
    data[:, :] .= data[:, :] .+ g.x[:, :];
    return Path(f.tgrid, data);
end

function Base.:+(f :: Path, a :: Real)
    """ 
        Add a scalar to the path.
    """
    @assert !isnan(a);
    data = copy(f.x);
    data[:, :] .= data[:, :] .+ a;
    return Path(
        f.tgrid, 
        data
    );
end

function Base.:-(f :: Path, a :: Real)
    """
        Subtract a scalar to the path.
    """
    @assert !isnan(a);
    data = copy(f.x);
    data[:, :] .= data[:, :] .- a;
    return Path(
        f.tgrid, 
        data
    );
end

function Base.:+(f :: Path, x :: Union{Vector, Matrix})
    """ 
        Shift the path by a point in the space.
    """
    if isa(x, Vector)
        x = reshape(x, 1, length(x));
    end
    @assert !any(isnan.(x));
    @assert f.ndims == size(x, 2);
    data = copy(f.x);
    data[:, :] .= data[:, :] .+ x;
    return Path(
        f.tgrid, 
        data
    );
end

function Base.:-(f :: Path, x :: Union{Vector, Matrix})
    """ 
        Negative of shift by a point.
    """
    return f + (-x);
end

function Base.:-(f :: Path)
    """ 
        Negate a path.
    """
    data = -copy(f.x);
    return Path(f.tgrid, data);
end

function Base.:-(f :: Path, g :: Path)
    """
        Subtract path y from x
    """
    return f + (-g);
end

function Base.:*(f :: Path, g :: Path) 
    """ 
        Computes inner product between two paths.
        Implemented using trapz.
    """
    @assert compatible(f, g);
    res = 0.0;
    nt = f.nt;
    dt = f.dt;
    for i = 2:nt
        res += 0.5 * dt * (f.x[i, :]'g.x[i, :] + f.x[i-1, :]'g.x[i-1, :]);
    end
    return res;
end

function Base.:*(f :: Path, a :: Real)
    """ 
        Scale the path.
    """
    data = a .* copy(f.x);
    return Path(
        f.tgrid, data
    );
end

function Base.:*(a :: Real, f :: Path)
    return f * a;
end

function Base.:*(operator :: PathOuterProd, h :: Path)
    """ 
        Apply outer product operator on path. 
        
        Should return a path.
    """
    @assert compatible(operator, h);
    return (operator.g * h) * operator.f;
end

function Base.:*(operator :: OrthogonalProjection, h :: Path)
    """ 
        Apply orthogonal projection operator on path h. 

        Effectively:
            (I - (f ⊗ f)/norm(f)) * h
    """
    f = operator.f;
    angle = (f * h)/(norm(f)^2);
    return h - angle * f;
end

function Base.:*(operator :: IdentityOperator, h :: Path)
    """ 
        Apply identity transformation to the input path h. 
    """
    return copy(h);
end

function Base.:*(operator :: SpatioTemporalOperator, h :: Path)
    """ 
        Transforms a path by applying a function b(h(t), t) 
        at each point along the path h(t).
    """
    # create a copy of path
    f = copy(h);
    nt = f.nt;
    for i = 1:nt 
        f.x[i, :] .= operator.mapping(f.x[i, :], f.tgrid[i]);
    end
    return f;
end

function Base.:*(operator :: ParameterizedSpatioTemporalOperator, h :: Path)
    """
        Transforms a path by applying a parameterized function f(h(t), t; x(t))
        at each point along the path h(t).
    """
    @assert compatible(operator.param, h);
    # create a copy of input path 
    f = copy(h);
    nt = f.nt;
    for i = 1:nt
        f.x[i, :] = operator.mapping(operator.param.x[i, :], h.x[i, :], h.tgrid[i]);
    end
    return f;
end

function Base.:*(operator :: ConstantLinearOperator, h :: Path)
    # create a copy of input path 
    f = copy(h);
    # apply linear transformation to all spatial points on the path 
    f.x[:, :] .= (operator.mat * f.x')';
    return f;
end

function Base.:*(operator :: ParameterizedLinearOperator, h :: Path)
    @assert compatible(operator.param, h);
    # create a copy of input path 
    f = copy(h);
    nt = f.nt;
    for i = 1:nt
        f.x[i, :] = operator.mat_func(operator.param.x[i, :]) * h.x[i, :];
    end
    return f;
end

function Base.:*(operator :: ParameterizedLinearThirdOrderOperator, f :: Path)
    @assert compatible(operator.param, f);
    # returns a contracted second order operator 
    return ParameterizedLinearThirdOrderOperatorContracted(
        operator.param, f, operator.mapping
    );
end

function Base.:*(
    operator :: ParameterizedLinearThirdOrderOperatorContracted, 
    g :: Path
)
    @assert compatible(operator.x, g);
    _h = copy(g) * 0.0;
    nt = _h.nt;
    for i = 1:nt 
        _h.x[i, :] .= operator.mapping(
            operator.x.x[i, :], operator.f.x[i, :]
        ) * g.x[i, :];
    end
    return _h;
end

function Base.:/(f :: Path, a :: Real)
    """ 
        Scale the path by 1/a, errors if a=0.
    """
    @assert !iszero(a) && !isnan(a);
    data = copy(f.x) ./ a;
    return Path(
        f.tgrid, data
    );
end

function Base.:\(operator :: ConstantLinearOperator, h :: Path)
    """
        Applies invertible constant linear operator to path.
    """
    # create a copy of input path 
    f = copy(h);
    # apply linear transformation to all spatial points on the path 
    f.x[:, :] .= (operator.mat \ f.x')';
    return f;
end

function Base.:sqrt(operator :: ConstantLinearOperator)
    """ 
        Computes Cholesky decomposition of the underlying matrix 
        transformation. Throws an error if Cholesky fails.
    """
    # requires A = σσ^T; the lower triangular form
    _mat_sqrt = cholesky(operator.mat).L;
    return ConstantLinearOperator(_mat_sqrt);
end

function Base.:adjoint(operator :: ConstantLinearOperator)
    """ 
        Returns adjoint operator of the linear transformation, 
        effectively the underlying matrix is transposed. 
    """
    _mat_adjoint = Matrix(transpose(operator.mat));
    return ConstantLinearOperator(_mat_adjoint);
end

function LinearAlgebra.:norm(f :: Path)
    """ 
        Computes L^2 norm in [0, T] x R^d space.
    """
    return sqrt(f * f);
end


function normalize(f :: Path)
    """
        Normalize the path.
    """
    return f / LinearAlgebra.norm(f);
end

function hadamard(f :: Path, g :: Path)
    """ 
        Perform Hadamard product.
    """
    @assert compatible(f, g);
    data = copy(f.x);
    data[:, :] .= data[:, :] .* g.x[:, :];
    return Path(
        f.tgrid, data
    );
end


function Base.:copy(h :: Path)
    return Path(copy(h.tgrid), copy(h.x));
end


## Operator eigenvalue problems
function arnoldi(operator :: PathOperator, f :: Path, n :: Int, eps=1e-8)
    """ 
        An iterative eigenvalue process for eigen-pairs in Hilbert space.

        Returns a set of (n+1) eigen-functions, along with a size 
        (n+1) x n upper-Hessenberg matrix, whose eigenvalues from the 
        first n rows are the eigenvalues of the Krylov vectors; and approximate 
        the largest n eigenvalues of the operator.
    """
    # Upper Hessenberg matrix
    H = zeros(n+1, n);
    Q = Vector{Path}(undef, n+1);
    # normalize input path 
    f = normalize(f);
    Q[1] = f;
    for i = 2:n+1
        println("Computing Krylov vector $(i) ...");
        # candidate eigen path 
        v = operator * Q[i-1];
        for j = 1:i-1
            # subtract projection onto previous paths 
            H[j, i-1] = Q[j] * v;
            v = v - H[j, i-1] * Q[j];
        end
        H[i, i-1] = norm(v);
        if H[i, i-1] > eps
            # significant eigenpath, add 
            Q[i] = v / H[i, i-1];
        else
            break;
        end
    end
    return Q, H;
end

mutable struct ExampleIntegralOperator <: PathOperator
    """
        An example integral operator for which we can compute the 
        eigenvalues exactly:

            K(f) = ∫_0^1 G(t, s)f(s)ds 
        where:
            G(t, s) = 
                s * (1-t),  0 <= s <= t <= 1
                t * (1-s),  0 <= t <= s <= 1
        See:
        https://math.stackexchange.com/questions/689617/eigenvalues-and-eigenvectors-of-an-integral-operator/689750
    """
    # must be in [0, 1]
    tgrid 

    # G(t, s)
    kernel 

    function ExampleIntegralOperator(tgrid)
        @assert iszero(tgrid[1]) && isone(tgrid[end])
        function kernel(t, s)
            @assert 0 <= t <= 1 && 0 <= s <= 1
            if s <= t
                return s * (1-t);
            else
                return t * (1-s);
            end
        end
        return new(tgrid, kernel);
    end
end

function Base.:*(operator :: ExampleIntegralOperator, f :: Path)
    # only works in 1d
    @assert f.ndims == 1
    @assert length(operator.tgrid) == length(f.tgrid)
    @assert iszero(f.tgrid[1]) && isone(f.tgrid[end])
    # apply integral operator with the specific kernel 
    nt = length(operator.tgrid);
    dt = operator.tgrid[2]-operator.tgrid[1];
    Af = copy(f);
    for i = 1:nt
        # fix t 
        t = Af.tgrid[i];
        # integrate with fixed t 
        integral_t = 0.0;
        # trapz
        for j = 2:nt 
            integral_t = integral_t + 0.5 * dt * (
                operator.kernel(t, tgrid[j-1])*f.x[j-1] + operator.kernel(t, tgrid[j])*f.x[j]
            )
        end
        # store integral for this t 
        Af.x[i] = integral_t;
    end
    return Af;
end