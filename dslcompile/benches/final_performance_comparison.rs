//! Final Performance Comparison: Macro-Based vs Direct Rust
//!
//! This benchmark provides the definitive comparison between the new macro-based
//! approach and direct Rust functions, demonstrating that the macro approach
//! achieves true zero-overhead abstraction.

use divan::Bencher;
use dslcompile::compile_time::macro_expressions::{linear_combination, PI, cos, exp, sin, sqrt};
use dslcompile::{expr};

fn main() {
    divan::main();
}

// ============================================================================
// SIMPLE BINARY OPERATIONS (x + y)
// ============================================================================

#[divan::bench]
fn macro_based_binary(bencher: Bencher) {
    let macro_fn = expr!(|x: f64, y: f64| x + y);
    bencher.bench_local(|| macro_fn(3.0, 4.0));
}

#[divan::bench]
fn direct_rust_binary(bencher: Bencher) {
    let direct_fn = |x: f64, y: f64| -> f64 { x + y };
    bencher.bench_local(|| direct_fn(3.0, 4.0));
}

// ============================================================================
// COMPLEX MATHEMATICAL EXPRESSIONS
// ============================================================================

#[divan::bench]
fn macro_based_complex(bencher: Bencher) {
    let complex_fn = expr!(|x: f64, y: f64| sqrt(x * x + y * y) + sin(x) * cos(y));
    bencher.bench_local(|| complex_fn(3.0, 4.0));
}

#[divan::bench]
fn direct_rust_complex(bencher: Bencher) {
    let complex_fn = |x: f64, y: f64| -> f64 { (x * x + y * y).sqrt() + x.sin() * y.cos() };
    bencher.bench_local(|| complex_fn(3.0, 4.0));
}

// ============================================================================
// NEURAL NETWORK LAYER SIMULATION
// ============================================================================

#[divan::bench]
fn macro_based_neural(bencher: Bencher) {
    let neural_fn = expr!(
        |weights: &[f64], inputs: &[f64], bias: f64| weights[0] * inputs[0]
            + weights[1] * inputs[1]
            + bias
    );
    let weights = [0.5, 0.3];
    let inputs = [1.0, 2.0];
    bencher.bench_local(|| neural_fn(&weights, &inputs, 0.1));
}

#[divan::bench]
fn direct_rust_neural(bencher: Bencher) {
    let neural_fn = |weights: &[f64], inputs: &[f64], bias: f64| -> f64 {
        weights[0] * inputs[0] + weights[1] * inputs[1] + bias
    };
    let weights = [0.5, 0.3];
    let inputs = [1.0, 2.0];
    bencher.bench_local(|| neural_fn(&weights, &inputs, 0.1));
}

// ============================================================================
// CONVENIENCE MACROS
// ============================================================================

#[divan::bench]
fn macro_convenience_relu(bencher: Bencher) {
    let relu = expr!(|x: f64| if x > 0.0 { x } else { 0.0 });
    bencher.bench_local(|| relu(3.0));
}


#[divan::bench]
fn direct_rust_relu(bencher: Bencher) {
    let relu = |x: f64| -> f64 { if x > 0.0 { x } else { 0.0 } };
    bencher.bench_local(|| relu(3.0));
}

#[divan::bench]
fn direct_rust_quadratic(bencher: Bencher) {
    let quadratic = |x: f64, a: f64, b: f64, c: f64| -> f64 { a * x * x + b * x + c };
    bencher.bench_local(|| quadratic(2.0, 1.0, 2.0, 3.0));
}

// ============================================================================
// BUILDER PATTERN PERFORMANCE
// ============================================================================

#[divan::bench]
fn macro_builder_linear_combination(bencher: Bencher) {
            let linear_comb = linear_combination::<4>();
    let coeffs = [0.2, 0.3, 0.4, 0.1];
    let values = [10.0, 20.0, 30.0, 40.0];
    bencher.bench_local(|| linear_comb(&coeffs, &values));
}

#[divan::bench]
fn direct_rust_linear_combination(bencher: Bencher) {
    let linear_comb = |coeffs: &[f64; 4], values: &[f64; 4]| -> f64 {
        coeffs.iter().zip(values.iter()).map(|(c, v)| c * v).sum()
    };
    let coeffs = [0.2, 0.3, 0.4, 0.1];
    let values = [10.0, 20.0, 30.0, 40.0];
    bencher.bench_local(|| linear_comb(&coeffs, &values));
}

// ============================================================================
// FLEXIBLE ARITY DEMONSTRATION
// ============================================================================

#[divan::bench]
fn macro_unary(bencher: Bencher) {
    let unary_fn = expr!(|x: f64| x * x);
    bencher.bench_local(|| unary_fn(3.0));
}

#[divan::bench]
fn macro_ternary(bencher: Bencher) {
    let ternary_fn = expr!(|x: f64, y: f64, z: f64| x + y + z);
    bencher.bench_local(|| ternary_fn(1.0, 2.0, 3.0));
}

#[divan::bench]
fn macro_quaternary(bencher: Bencher) {
    let quaternary_fn = expr!(|a: f64, b: f64, c: f64, d: f64| (a + b) * (c + d));
    bencher.bench_local(|| quaternary_fn(1.0, 2.0, 3.0, 4.0));
}

#[divan::bench]
fn direct_rust_unary(bencher: Bencher) {
    let unary_fn = |x: f64| -> f64 { x * x };
    bencher.bench_local(|| unary_fn(3.0));
}

#[divan::bench]
fn direct_rust_ternary(bencher: Bencher) {
    let ternary_fn = |x: f64, y: f64, z: f64| -> f64 { x + y + z };
    bencher.bench_local(|| ternary_fn(1.0, 2.0, 3.0));
}

#[divan::bench]
fn direct_rust_quaternary(bencher: Bencher) {
    let quaternary_fn = |a: f64, b: f64, c: f64, d: f64| -> f64 { (a + b) * (c + d) };
    bencher.bench_local(|| quaternary_fn(1.0, 2.0, 3.0, 4.0));
}

// ============================================================================
// FINANCIAL CALCULATIONS
// ============================================================================

#[divan::bench]
fn macro_compound_interest(bencher: Bencher) {
    let compound_fn = expr!(|principal: f64, rate: f64, time: f64| principal * exp(rate * time));
    bencher.bench_local(|| compound_fn(1000.0, 0.05, 2.0));
}

#[divan::bench]
fn direct_rust_compound_interest(bencher: Bencher) {
    let compound_fn =
        |principal: f64, rate: f64, time: f64| -> f64 { principal * (rate * time).exp() };
    bencher.bench_local(|| compound_fn(1000.0, 0.05, 2.0));
}

// ============================================================================
// PHYSICS SIMULATIONS
// ============================================================================

#[divan::bench]
fn macro_kinetic_energy(bencher: Bencher) {
    let kinetic_fn = expr!(|mass: f64, velocity: f64| 0.5 * mass * velocity * velocity);
    bencher.bench_local(|| kinetic_fn(10.0, 5.0));
}

#[divan::bench]
fn direct_rust_kinetic_energy(bencher: Bencher) {
    let kinetic_fn = |mass: f64, velocity: f64| -> f64 { 0.5 * mass * velocity * velocity };
    bencher.bench_local(|| kinetic_fn(10.0, 5.0));
}

#[divan::bench]
fn macro_wave_equation(bencher: Bencher) {
    let wave_fn = expr!(
        |amplitude: f64, frequency: f64, time: f64, phase: f64| amplitude
            * sin(2.0 * PI * frequency * time + phase)
    );
    bencher.bench_local(|| wave_fn(2.0, 1.0, 0.5, 0.0));
}

#[divan::bench]
fn direct_rust_wave_equation(bencher: Bencher) {
    let wave_fn = |amplitude: f64, frequency: f64, time: f64, phase: f64| -> f64 {
        amplitude * (2.0 * std::f64::consts::PI * frequency * time + phase).sin()
    };
    bencher.bench_local(|| wave_fn(2.0, 1.0, 0.5, 0.0));
}
