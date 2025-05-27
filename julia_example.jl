#!/usr/bin/env julia

# Simple example demonstrating optimal rational function approximation
using MathJIT

println("🔬 MathJIT: Optimal Rational Function Approximation")
println("=" ^ 55)
println()

# Example: Find the fastest rational approximation for exp(x) on [0,1]
println("📊 Finding optimal rational approximation for exp(x) on [0,1]")
println("   Target tolerance: 1e-6")
println()

f(x) = exp(x)
result = find_optimal_rational(f, (0.0, 1.0), 1e-6)

println("✅ Results:")
println("   • Optimal degrees: numerator=$(result.degree_n), denominator=$(result.degree_d)")
println("   • Total degree: $(result.total_degree)")
println("   • Achieved error: $(result.error)")
println()

# Test the approximation at a few points
println("🧪 Testing approximation accuracy:")
test_points = [0.0, 0.25, 0.5, 0.75, 1.0]
for x in test_points
    actual = f(x)
    approx = evaluate_rational(x, result.N, result.D)
    error = abs(actual - approx)
    println("   • f($x) = $(round(actual, digits=6)), approx = $(round(approx, digits=6)), error = $(error)")
end
println()

# Compare with different tolerance levels
println("📈 Tolerance vs. Degree comparison:")
tolerances = [1e-3, 1e-6, 1e-9]
for tol in tolerances
    try
        res = find_optimal_rational(f, (0.0, 1.0), tol)
        println("   • Tolerance $tol → Total degree: $(res.total_degree)")
    catch e
        println("   • Tolerance $tol → Failed: $e")
    end
end
println()

println("🎯 Key insight: Lower degree rational functions are faster to evaluate!")
println("   This package automatically finds the minimal degree needed for your tolerance.") 