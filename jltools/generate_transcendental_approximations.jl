#!/usr/bin/env julia

# Generate optimal rational approximations for transcendental functions
# This script generates Rust code with polynomial/rational approximations

using Dates
include("src/optimal_rational.jl")

"""
Generate Rust code for a rational function approximation.
"""
function generate_rust_rational(name, coeffs_num, coeffs_den, interval, error)
    rust_code = """
/// $name approximation using optimal rational function
/// Interval: $interval, Max error: $error
#[inline]
fn $(name)_approx(x: f64) -> f64 {
    // Numerator coefficients (constant term first)
    let num_coeffs = [$(join(map(c -> "$(Float64(c))", coeffs_num), ", "))];
    
    // Denominator coefficients (constant term first)  
    let den_coeffs = [$(join(map(c -> "$(Float64(c))", coeffs_den), ", "))];
    
    // Evaluate numerator using Horner's method
    let mut num = num_coeffs[$(length(coeffs_num)-1)];
    for i in (0..$(length(coeffs_num)-1)).rev() {
        num = num * x + num_coeffs[i];
    }
    
    // Evaluate denominator using Horner's method
    let mut den = den_coeffs[$(length(coeffs_den)-1)];
    for i in (0..$(length(coeffs_den)-1)).rev() {
        den = den * x + den_coeffs[i];
    }
    
    num / den
}
"""
    return rust_code
end

"""
Try to find a rational approximation with fallback to polynomial if needed.
"""
function find_approximation_with_fallback(f, interval, tolerance, name)
    println("\n📊 Computing $name approximation on $interval...")
    
    # First try rational approximation with conservative max degree
    try
        result = find_optimal_rational(f, interval, tolerance, max_degree=12)
        println("   Optimal degrees: ($(result.degree_n), $(result.degree_d))")
        println("   Achieved error: $(Float64(result.error))")
        return result
    catch e
        println("   Rational approximation failed: $e")
        
        # Fallback: try pure polynomial approximation using Remez directly
        println("   Trying polynomial approximation...")
        try
            # Try polynomial degrees from 4 to 12
            for degree in 4:12
                try
                    N, D, E, X = Remez.ratfn_minimax(f, interval, degree, 0)
                    if E <= tolerance
                        println("   Found polynomial degree $degree with error $(Float64(E))")
                        return (
                            N = N,
                            D = D,
                            error = E,
                            degree_n = degree,
                            degree_d = 0,
                            total_degree = degree,
                            alternation_points = X
                        )
                    end
                catch poly_e
                    continue
                end
            end
            println("   Could not find suitable polynomial approximation")
            return nothing
        catch fallback_e
            println("   Polynomial fallback also failed: $fallback_e")
            return nothing
        end
    end
end

"""
Generate optimal approximations for common transcendental functions.
"""
function generate_transcendental_approximations()
    println("🔬 Generating optimal rational approximations for transcendental functions...")
    println("=" ^ 70)
    
    # More conservative tolerance
    tolerance = 1e-10
    
    approximations = []
    
    # 1. Natural logarithm ln(1+x) on [0, 1]
    ln_result = find_approximation_with_fallback(x -> log(1 + x), (0.0, 1.0), tolerance, "ln(1+x)")
    if ln_result !== nothing
        push!(approximations, (
            name = "ln_1plus",
            result = ln_result,
            interval = "[0, 1]",
            description = "ln(1+x) for x ∈ [0,1]"
        ))
    end
    
    # 2. Exponential function exp(x) on [-0.5, 0.5] (smaller interval)
    exp_result = find_approximation_with_fallback(x -> exp(x), (-0.5, 0.5), tolerance, "exp(x)")
    if exp_result !== nothing
        push!(approximations, (
            name = "exp",
            result = exp_result,
            interval = "[-0.5, 0.5]", 
            description = "exp(x) for x ∈ [-0.5, 0.5]"
        ))
    end
    
    # 3. Sine function sin(x) on [-π/4, π/4]
    sin_result = find_approximation_with_fallback(x -> sin(x), (-π/4, π/4), tolerance, "sin(x)")
    if sin_result !== nothing
        push!(approximations, (
            name = "sin",
            result = sin_result,
            interval = "[-π/4, π/4]",
            description = "sin(x) for x ∈ [-π/4, π/4]"
        ))
    end
    
    # 4. Cosine function cos(x) on [0, π/4] (smaller interval)
    cos_result = find_approximation_with_fallback(x -> cos(x), (0.0, π/4), tolerance, "cos(x)")
    if cos_result !== nothing
        push!(approximations, (
            name = "cos",
            result = cos_result,
            interval = "[0, π/4]",
            description = "cos(x) for x ∈ [0, π/4]"
        ))
    end
    
    return approximations
end

"""
Generate complete Rust module with all approximations.
"""
function generate_rust_module(approximations)
    rust_code = """
//! Optimal rational approximations for transcendental functions
//! 
//! This module contains automatically generated optimal rational function
//! approximations for common transcendental functions, computed using the
//! Remez exchange algorithm in Julia.
//!
//! Generated on: $(Dates.now())

use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;
use crate::error::{MathCompileError, Result};

"""
    
    for approx in approximations
        result = approx.result
        rust_code *= generate_rust_rational(
            approx.name,
            result.N,
            result.D, 
            approx.interval,
            Float64(result.error)
        )
        rust_code *= "\n"
    end
    
    # Add helper functions for generating Cranelift IR
    rust_code *= """
/// Generate Cranelift IR for evaluating a polynomial using Horner's method
pub fn generate_polynomial_ir(
    builder: &mut FunctionBuilder,
    x: Value,
    coeffs: &[f64]
) -> Value {
    if coeffs.is_empty() {
        return builder.ins().f64const(0.0);
    }
    
    // Start with the highest degree coefficient
    let mut result = builder.ins().f64const(coeffs[coeffs.len() - 1]);
    
    // Apply Horner's method: result = result * x + coeff[i]
    for &coeff in coeffs.iter().rev().skip(1) {
        result = builder.ins().fmul(result, x);
        let coeff_val = builder.ins().f64const(coeff);
        result = builder.ins().fadd(result, coeff_val);
    }
    
    result
}

/// Generate Cranelift IR for evaluating a rational function
pub fn generate_rational_ir(
    builder: &mut FunctionBuilder,
    x: Value,
    num_coeffs: &[f64],
    den_coeffs: &[f64]
) -> Value {
    let numerator = generate_polynomial_ir(builder, x, num_coeffs);
    let denominator = generate_polynomial_ir(builder, x, den_coeffs);
    builder.ins().fdiv(numerator, denominator)
}

"""
    
    # Add specific IR generation functions for each transcendental function
    for approx in approximations
        result = approx.result
        num_coeffs = map(Float64, result.N)
        den_coeffs = map(Float64, result.D)
        
        rust_code *= """
/// Generate Cranelift IR for $(approx.description)
/// Max error: $(Float64(result.error))
pub fn generate_$(approx.name)_ir(builder: &mut FunctionBuilder, x: Value) -> Value {
    let num_coeffs = [$(join(num_coeffs, ", "))];
    let den_coeffs = [$(join(den_coeffs, ", "))];
    generate_rational_ir(builder, x, &num_coeffs, &den_coeffs)
}

"""
    end
    
    return rust_code
end

# Main execution
function main()
    println("🚀 MathCompile Transcendental Function Approximation Generator")
    println("=" ^ 60)
    
    # Generate approximations
    approximations = generate_transcendental_approximations()
    
    # Generate Rust code
    println("\n🔧 Generating Rust code...")
    rust_code = generate_rust_module(approximations)
    
    # Write to file
    output_file = "../src/transcendental.rs"
    open(output_file, "w") do f
        write(f, rust_code)
    end
    
    println("✅ Generated Rust module: $output_file")
    
    # Print summary
    println("\n📈 Summary:")
    println("=" ^ 30)
    for approx in approximations
        result = approx.result
        println("• $(approx.description)")
        println("  Degrees: ($(result.degree_n), $(result.degree_d)), Error: $(Float64(result.error))")
    end
    
    println("\n🎯 Next steps:")
    println("1. Include the generated module in src/jit.rs")
    println("2. Replace placeholder implementations with calls to these functions")
    println("3. Test the new implementations!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 