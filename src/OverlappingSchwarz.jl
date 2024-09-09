######################################################################
# Overlapping Schwarz time decomposition in continuous space, 
# supports adaptive mesh and nonuniform overlaps. 
######################################################################

module OverlappingSchwarz

    # dependency
    using LinearAlgebra;
    # random
    using Statistics, Random;
    # plotting 
    using Plots; gr();
    # overriding the base library 
    using Base;
    # interpolating paths 
    using Interpolations;
    # parallel processes
    using Distributed;

    # exports
    export Random;
    export LinearQuadraticProblem, NonlinearProblem;
    # main functionality
    include("schwarz.jl");
    include("ocp.jl");
    # utility functions
    include("path_utils.jl");
end
