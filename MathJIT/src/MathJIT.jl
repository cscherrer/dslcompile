module MathJIT

include("optimal_rational.jl")

# Export the main functions for finding optimal rational approximations
export find_optimal_rational, evaluate_rational, test_approximation_error

greet() = print("Hello World!")

end # module MathJIT
