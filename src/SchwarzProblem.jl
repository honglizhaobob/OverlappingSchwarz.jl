

mutable struct SchwarzProblem
    """
        An overlapping Schwarz time domain scheme to solve 
        a continuous optimal control problem, based on 
        perturbation of data at initial and terminal boundaries. 

        The problem only store primal and dual data at the interfaces.
        Given correctly defined subproblems, the full trajectory can 
        be reproduced given the interface data only. 
    """
    # problem dimensions 
    ndims
    # domain boundaries 
    t_start
    t_end

    # overlaps on the left and right 
    tau_left
    tau_right

    # all overlapped subintervals
    domsn  # no overlap
    doms   # overlapped

    # number of subproblems 
    num_probs

    # vector of subproblems
    subproblems

    function SchwarzProblem(
        ndims, t_start, t_end, knots, tau_left, tau_right
    )
        """ 
            Constructor that performs data validation. 
            The overlaps cannot extend beyond left and right 
            domain boundaries. 
        """
        # number of problems
        m = length(knots)+1;
        num_probs = m;
        @assert length(tau_left) == length(tau_right) == m
        # cannot extend left and right ends
        @assert iszero(tau_left[1]) & iszero(tau_right[end]);
        # sort knots if not already sorted 
        knots = sort!(knots);
        @assert all(knots .< t_end) & all(knots .> t_start);
        # create all subdomains (no overlap)
        domsn = partition([t_start, t_end], knots);

        # create all subdomains (with overlap)
        doms = deepcopy(domsn);
        # extend left and right boundaries for each subproblem 
        for i = 1:m
            # extend left boundary
            doms[i][1] = domsn[i][1]-tau_left[i];
            # extend right boundary 
            doms[i][2] = domsn[i][2]+tau_right[i];
        end

        # placeholder for subproblems 
        subproblems = Vector{OptimalControlProblem}(undef, num_probs);
        
        return new(
            ndims, t_start, t_end,
            tau_left, tau_right,
            domsn, doms, num_probs, subproblems
        );
    end
end

function partition(dom, knots)
    """ 
        Partitions an interval at fixed points. 

        Inputs:
            dom: (vector)
                Domain boundaries [a, b].
            knots: (vector)
                Vector of points between a and b. 
        Returns:
            doms: (vector of vectors)
                Vector of domain boundaries for subintervals.
    """
    sort!(knots);
    t_start, t_end = dom[1], dom[2];
    # validate knots 
    @assert all(knots .> t_start) & all(knots .< t_end);
    # number of problems 
    m = length(knots)+1;
    doms = Vector{Vector}(undef, m);
    for i = 1:m
        if i == 1
            doms[i] = [t_start, knots[i]];
        elseif i == m
            doms[i] = [knots[i-1], t_end];
        else
            doms[i] = [knots[i-1], knots[i]];
        end
    end
    return doms;
end

function max_overlap(prob :: SchwarzProblem)
    """ 
        Returns maximum overlap size of the Schwarz problem. 
    """
    return maximum(
        prob.tau_left + prob.tau_right
    );
end

function schwarz_size(prob :: SchwarzProblem)
    """
        Returns a vector of sizes of each subproblem.
    """
    b = prob.num_probs;
    res = Vector{Any}(undef, b);
    for k = eachindex(prob.subproblems)
        res[k] = length(prob.subproblems[k].t);
    end
    return res;
end


function greet()
    println("Hello world");
end