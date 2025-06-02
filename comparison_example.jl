#!/usr/bin/env julia

# Comparison example showing the improvement from the algorithm change
using DSLCompile

println("🔬 Algorithm Improvement Demonstration")
println("=" ^ 45)
println()

f(x) = exp(x)
tolerance = 1e-6
interval = (0.0, 1.0)

println("📊 Finding optimal rational approximation for exp(x) on [0,1]")
println("   Target tolerance: $tolerance")
println()

result = find_optimal_rational(f, interval, tolerance)

println("✅ Current Algorithm Results:")
println("   • Degrees: numerator=$(result.degree_n), denominator=$(result.degree_d)")
println("   • Total degree: $(result.total_degree)")
println("   • Achieved error: $(result.error)")
println()

println("📈 Key Improvement:")
println("   The algorithm now completes each total degree round and selects")
println("   the approximation with the LOWEST ERROR within that degree level,")
println("   rather than just taking the first one that meets the tolerance.")
println()

println("🎯 Benefits:")
println("   • Better accuracy for the same computational cost")
println("   • More consistent results")
println("   • Optimal use of the available polynomial degrees")
println()

# Show all valid approximations at this degree level for comparison
println("🔍 All degree-5 combinations that meet tolerance $tolerance:")
println("   (This shows why choosing the best one matters)")

for n in 0:5
    d = 5 - n
    try
        N, D, E, X = Remez.ratfn_minimax(f, interval, n, d, (x, y) -> BigFloat(1))
        if E <= tolerance
            println("   • ($n, $d): error = $(Float64(E))")
        end
    catch e
        # Skip failed combinations
    end
end 