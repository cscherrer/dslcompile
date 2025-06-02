using DSLCompile

# Example 1: Approximate exp(x) on [0, 1] with tolerance 1e-6
println("Example 1: Approximating exp(x) on [0, 1]")
println("=" ^ 50)

f1(x) = exp(x)
result1 = find_optimal_rational(f1, (0.0, 1.0), 1e-6)

println("Function: exp(x)")
println("Interval: [0, 1]")
println("Tolerance: 1e-6")
println("Optimal degrees: numerator=$(result1.degree_n), denominator=$(result1.degree_d)")
println("Total degree: $(result1.total_degree)")
println("Achieved error: $(result1.error)")
println("Numerator coefficients: $(result1.N)")
println("Denominator coefficients: $(result1.D)")

# Test the approximation
test1 = test_approximation_error(f1, result1, (0.0, 1.0))
println("Verification - Max error: $(test1.max_error) at x=$(test1.max_error_point)")
println()

# Example 2: Approximate sin(x) on [0, π/2] with tolerance 1e-8
println("Example 2: Approximating sin(x) on [0, π/2]")
println("=" ^ 50)

f2(x) = sin(x)
result2 = find_optimal_rational(f2, (0.0, BigFloat(π)/2), 1e-8)

println("Function: sin(x)")
println("Interval: [0, π/2]")
println("Tolerance: 1e-8")
println("Optimal degrees: numerator=$(result2.degree_n), denominator=$(result2.degree_d)")
println("Total degree: $(result2.total_degree)")
println("Achieved error: $(result2.error)")

# Test the approximation
test2 = test_approximation_error(f2, result2, (0.0, BigFloat(π)/2))
println("Verification - Max error: $(test2.max_error) at x=$(test2.max_error_point)")
println()

# Example 3: Approximate 1/(1+x^2) on [-1, 1] with custom weight function
println("Example 3: Approximating 1/(1+x²) on [-1, 1] with custom weighting")
println("=" ^ 70)

f3(x) = 1 / (1 + x^2)
# Weight function that emphasizes accuracy near the endpoints
weight_fn(x, y) = 1 + abs(x)  # Higher weight near x = ±1

result3 = find_optimal_rational(f3, (-1.0, 1.0), 1e-6, w=weight_fn)

println("Function: 1/(1+x²)")
println("Interval: [-1, 1]")
println("Tolerance: 1e-6")
println("Weight function: 1 + |x| (emphasizes endpoints)")
println("Optimal degrees: numerator=$(result3.degree_n), denominator=$(result3.degree_d)")
println("Total degree: $(result3.total_degree)")
println("Achieved weighted error: $(result3.error)")

# Test the approximation (unweighted error for comparison)
test3 = test_approximation_error(f3, result3, (-1.0, 1.0))
println("Verification - Max unweighted error: $(test3.max_error) at x=$(test3.max_error_point)")
println()

# Example 4: Compare different tolerance levels for the same function
println("Example 4: Tolerance comparison for exp(x) on [0, 1]")
println("=" ^ 50)

tolerances = [1e-3, 1e-6, 1e-9, 1e-12]
println("Tolerance\tTotal Degree\tNumerator\tDenominator\tActual Error")
println("-" ^ 70)

for tol in tolerances
    try
        result = find_optimal_rational(f1, (0.0, 1.0), tol)
        println("$(tol)\t\t$(result.total_degree)\t\t$(result.degree_n)\t\t$(result.degree_d)\t\t$(result.error)")
    catch e
        println("$(tol)\t\tFailed: $e")
    end
end 