//! Optimal rational approximations for transcendental functions
//!
//! This module contains automatically generated optimal rational function
//! approximations for common transcendental functions, computed using the
//! Remez exchange algorithm in Julia.
//!
//! Generated on: 2025-05-27T17:50:14.453

#[cfg(feature = "cranelift")]
use cranelift_codegen::ir::{InstBuilder, Value};
#[cfg(feature = "cranelift")]
use cranelift_frontend::FunctionBuilder;

/// `ln_1plus` approximation using optimal rational function
/// Interval: [0, 1], Max error: 6.248044858924071e-12
#[inline]
fn ln_1plus_approx(x: f64) -> f64 {
    // Numerator coefficients (constant term first)
    let num_coeffs = [
        0.0,
        0.9999999985585198,
        1.3031632785795166,
        0.4385659053064146,
        0.03085953976409006,
    ];

    // Denominator coefficients (constant term first)
    let den_coeffs = [
        1.0,
        1.8031632248969947,
        1.0068149572238094,
        0.18320686065538652,
        0.0068149572238094085,
    ];

    // Evaluate numerator using Horner's method
    let mut num = num_coeffs[4];
    for i in (0..4).rev() {
        num = num * x + num_coeffs[i];
    }

    // Evaluate denominator using Horner's method
    let mut den = den_coeffs[4];
    for i in (0..4).rev() {
        den = den * x + den_coeffs[i];
    }

    num / den
}

/// exp approximation using optimal rational function
/// Interval: [-0.5, 0.5], Max error: 2.1580070170984572e-11
#[inline]
fn exp_approx(x: f64) -> f64 {
    // Numerator coefficients (constant term first)
    let num_coeffs = [
        0.9999999999817503,
        0.42917796261241653,
        0.07164583021340264,
        0.004783741506421553,
    ];

    // Denominator coefficients (constant term first)
    let den_coeffs = [
        1.0,
        -0.5708220372139153,
        0.1424678650066201,
        -0.018939778254233708,
        0.001176234397980246,
    ];

    // Evaluate numerator using Horner's method
    let mut num = num_coeffs[3];
    for i in (0..3).rev() {
        num = num * x + num_coeffs[i];
    }

    // Evaluate denominator using Horner's method
    let mut den = den_coeffs[4];
    for i in (0..4).rev() {
        den = den * x + den_coeffs[i];
    }

    num / den
}

/// sin approximation using optimal rational function
/// Interval: [-π/4, π/4], Max error: 8.974715606466886e-13
#[inline]
fn sin_approx(x: f64) -> f64 {
    // Numerator coefficients (constant term first)
    let num_coeffs = [
        0.0,
        1.0000000000125895,
        0.0,
        -0.15266388542550388,
        0.0,
        0.005999540118150915,
        0.0,
        -8.173670882958005e-5,
    ];

    // Denominator coefficients (constant term first)
    let den_coeffs = [1.0, 0.0, 0.014002781650496685];

    // Evaluate numerator using Horner's method
    let mut num = num_coeffs[7];
    for i in (0..7).rev() {
        num = num * x + num_coeffs[i];
    }

    // Evaluate denominator using Horner's method
    let mut den = den_coeffs[2];
    for i in (0..2).rev() {
        den = den * x + den_coeffs[i];
    }

    num / den
}

/// cos approximation using optimal rational function
/// Interval: [0, π/4], Max error: 8.492520741606233e-11
#[inline]
fn cos_approx(x: f64) -> f64 {
    // Numerator coefficients (constant term first)
    let num_coeffs = [
        1.0000000000849252,
        -0.04419808517009371,
        -0.468545034572871,
        0.022095248245365844,
        0.025958373239365604,
        -0.0018934016585943506,
    ];

    // Denominator coefficients (constant term first)
    let den_coeffs = [1.0, -0.04419807131962928, 0.03145459448704991];

    // Evaluate numerator using Horner's method
    let mut num = num_coeffs[5];
    for i in (0..5).rev() {
        num = num * x + num_coeffs[i];
    }

    // Evaluate denominator using Horner's method
    let mut den = den_coeffs[2];
    for i in (0..2).rev() {
        den = den * x + den_coeffs[i];
    }

    num / den
}

/// Generate Cranelift IR for evaluating a polynomial using Horner's method
#[cfg(feature = "cranelift")]
pub fn generate_polynomial_ir(builder: &mut FunctionBuilder, x: Value, coeffs: &[f64]) -> Value {
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
#[cfg(feature = "cranelift")]
pub fn generate_rational_ir(
    builder: &mut FunctionBuilder,
    x: Value,
    num_coeffs: &[f64],
    den_coeffs: &[f64],
) -> Value {
    let numerator = generate_polynomial_ir(builder, x, num_coeffs);
    let denominator = generate_polynomial_ir(builder, x, den_coeffs);
    builder.ins().fdiv(numerator, denominator)
}

/// Generate Cranelift IR for ln(1+x) for x ∈ [0,1]
/// Max error: 6.248044858924071e-12
#[cfg(feature = "cranelift")]
pub fn generate_ln_1plus_ir(builder: &mut FunctionBuilder, x: Value) -> Value {
    let num_coeffs = [
        0.0,
        0.9999999985585198,
        1.3031632785795166,
        0.4385659053064146,
        0.03085953976409006,
    ];
    let den_coeffs = [
        1.0,
        1.8031632248969947,
        1.0068149572238094,
        0.18320686065538652,
        0.0068149572238094085,
    ];
    generate_rational_ir(builder, x, &num_coeffs, &den_coeffs)
}

/// Generate Cranelift IR for exp(x) for x ∈ [-0.5, 0.5]
/// Max error: 2.1580070170984572e-11
#[cfg(feature = "cranelift")]
pub fn generate_exp_ir(builder: &mut FunctionBuilder, x: Value) -> Value {
    let num_coeffs = [
        0.9999999999817503,
        0.42917796261241653,
        0.07164583021340264,
        0.004783741506421553,
    ];
    let den_coeffs = [
        1.0,
        -0.5708220372139153,
        0.1424678650066201,
        -0.018939778254233708,
        0.001176234397980246,
    ];
    generate_rational_ir(builder, x, &num_coeffs, &den_coeffs)
}

/// Generate Cranelift IR for sin(x) for x ∈ [-π/4, π/4]
/// Max error: 8.974715606466886e-13
#[cfg(feature = "cranelift")]
pub fn generate_sin_ir(builder: &mut FunctionBuilder, x: Value) -> Value {
    let num_coeffs = [
        0.0,
        1.0000000000125895,
        0.0,
        -0.15266388542550388,
        0.0,
        0.005999540118150915,
        0.0,
        -8.173670882958005e-5,
    ];
    let den_coeffs = [1.0, 0.0, 0.014002781650496685];
    generate_rational_ir(builder, x, &num_coeffs, &den_coeffs)
}

/// Generate Cranelift IR for cos(x) for x ∈ [0, π/4]
/// Max error: 8.492520741606233e-11
#[cfg(feature = "cranelift")]
pub fn generate_cos_ir(builder: &mut FunctionBuilder, x: Value) -> Value {
    let num_coeffs = [
        1.0000000000849252,
        -0.04419808517009371,
        -0.468545034572871,
        0.022095248245365844,
        0.025958373239365604,
        -0.0018934016585943506,
    ];
    let den_coeffs = [1.0, -0.04419807131962928, 0.03145459448704991];
    generate_rational_ir(builder, x, &num_coeffs, &den_coeffs)
}
