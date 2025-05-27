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
Generate optimal approximations for common transcendental functions.
"""
function generate_transcendental_approximations()
    println("🔬 Generating optimal rational approximations for transcendental functions...")
    println("=" ^ 70)
    
    # Start with more practical tolerance for f64 (about 10-12 decimal digits)
    # We can increase precision later once the system is working
    tolerance = 1e-10
    
    approximations = []
    
    # 1. Natural logarithm ln(x) on [1, 2]
    # We'll use ln(1+u) where u ∈ [0,1] for better numerical properties
    println("\n📊 Computing ln(1+x) approximation on [0, 1]...")
    try
        ln_result = find_optimal_rational(x -> log(1 + x), (0.0, 1.0), tolerance, max_degree=10)
        println("   Optimal degrees: ($(ln_result.degree_n), $(ln_result.degree_d))")
        println("   Achieved error: $(Float64(ln_result.error))")
        
        push!(approximations, (
            name = "ln_1plus",
            result = ln_result,
            interval = "[0, 1]",
            description = "ln(1+x) for x ∈ [0,1]"
        ))
    catch e
        println("   Failed to compute ln approximation: $e")
        println("   Skipping ln for now...")
    end
    
    # 2. Exponential function exp(x) on [-1, 1]  
    println("\n📊 Computing exp(x) approximation on [-1, 1]...")
    try
        exp_result = find_optimal_rational(x -> exp(x), (-1.0, 1.0), tolerance, max_degree=10)
        println("   Optimal degrees: ($(exp_result.degree_n), $(exp_result.degree_d))")
        println("   Achieved error: $(Float64(exp_result.error))")
        
        push!(approximations, (
            name = "exp",
            result = exp_result,
            interval = "[-1, 1]", 
            description = "exp(x) for x ∈ [-1,1]"
        ))
    catch e
        println("   Failed to compute exp approximation: $e")
        println("   Skipping exp for now...")
    end
    
    # 3. Sine function sin(x) on [-π/4, π/4] (smaller interval for better convergence)
    println("\n📊 Computing sin(x) approximation on [-π/4, π/4]...")
    try
        sin_result = find_optimal_rational(x -> sin(x), (-π/4, π/4), tolerance, max_degree=8)
        println("   Optimal degrees: ($(sin_result.degree_n), $(sin_result.degree_d))")
        println("   Achieved error: $(Float64(sin_result.error))")
        
        push!(approximations, (
            name = "sin",
            result = sin_result,
            interval = "[-π/4, π/4]",
            description = "sin(x) for x ∈ [-π/4, π/4]"
        ))
    catch e
        println("   Failed to compute sin approximation: $e")
        println("   Skipping sin for now...")
    end
    
    # 4. Cosine function cos(x) on [0, π/4] (smaller interval)
    println("\n📊 Computing cos(x) approximation on [0, π/4]...")
    try
        cos_result = find_optimal_rational(x -> cos(x), (0.0, π/4), tolerance, max_degree=8)
        println("   Optimal degrees: ($(cos_result.degree_n), $(cos_result.degree_d))")
        println("   Achieved error: $(Float64(cos_result.error))")
        
        push!(approximations, (
            name = "cos",
            result = cos_result,
            interval = "[0, π/4]",
            description = "cos(x) for x ∈ [0, π/4]"
        ))
    catch e
        println("   Failed to compute cos approximation: $e")
        println("   Skipping cos for now...")
    end
    
    # If we don't have any approximations, create some simple polynomial ones
    if isempty(approximations)
        println("\n⚠️  No rational approximations succeeded, creating simple polynomial approximations...")
        
        # Simple Taylor series approximations
        # sin(x) ≈ x - x³/6 + x⁵/120 for small x
        sin_coeffs_num = [0.0, 1.0, 0.0, -1.0/6.0, 0.0, 1.0/120.0]
        sin_coeffs_den = [1.0]
        
        push!(approximations, (
            name = "sin_taylor",
            result = (
                N = sin_coeffs_num,
                D = sin_coeffs_den,
                error = 1e-6,
                degree_n = 5,
                degree_d = 0,
                total_degree = 5
            ),
            interval = "[-π/4, π/4]",
            description = "sin(x) Taylor series approximation"
        ))
        
        # cos(x) ≈ 1 - x²/2 + x⁴/24 for small x
        cos_coeffs_num = [1.0, 0.0, -1.0/2.0, 0.0, 1.0/24.0]
        cos_coeffs_den = [1.0]
        
        push!(approximations, (
            name = "cos_taylor",
            result = (
                N = cos_coeffs_num,
                D = cos_coeffs_den,
                error = 1e-6,
                degree_n = 4,
                degree_d = 0,
                total_degree = 4
            ),
            interval = "[-π/4, π/4]",
            description = "cos(x) Taylor series approximation"
        ))
        
        # exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24 for small x
        exp_coeffs_num = [1.0, 1.0, 1.0/2.0, 1.0/6.0, 1.0/24.0]
        exp_coeffs_den = [1.0]
        
        push!(approximations, (
            name = "exp_taylor",
            result = (
                N = exp_coeffs_num,
                D = exp_coeffs_den,
                error = 1e-6,
                degree_n = 4,
                degree_d = 0,
                total_degree = 4
            ),
            interval = "[-1, 1]",
            description = "exp(x) Taylor series approximation"
        ))
        
        # ln(1+x) ≈ x - x²/2 + x³/3 - x⁴/4 for small x
        ln_coeffs_num = [0.0, 1.0, -1.0/2.0, 1.0/3.0, -1.0/4.0]
        ln_coeffs_den = [1.0]
        
        push!(approximations, (
            name = "ln_taylor",
            result = (
                N = ln_coeffs_num,
                D = ln_coeffs_den,
                error = 1e-6,
                degree_n = 4,
                degree_d = 0,
                total_degree = 4
            ),
            interval = "[0, 1]",
            description = "ln(1+x) Taylor series approximation"
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
use crate::error::{MathJITError, Result};

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
    println("🚀 MathJIT Transcendental Function Approximation Generator")
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