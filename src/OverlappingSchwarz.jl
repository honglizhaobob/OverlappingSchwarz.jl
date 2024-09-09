######################################################################
# Overlapping Schwarz time decomposition in continuous space, 
# supports adaptive mesh and nonuniform overlaps. 
######################################################################

module OverlappingSchwarz

    # dependency
    using LinearAlgebra
    # random
    using Statistics, Random
    # plotting 
    using Plots; gr();
    # overriding the base library 
    using Base

    # exports 
    export greet

    # include other code files 
    include("SchwarzProblem.jl");
    include("path_utils.jl");
end
