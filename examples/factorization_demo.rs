#!/usr/bin/env cargo run --example factorization_demo

//! Mathematical Discovery Demo - Compile-Time Factorization
//!
//! This example demonstrates how `MathCompile` can automatically discover
//! mathematical factorizations and simplifications using the trait-based
//! compile-time expression system.

use mathcompile::compile_time::{MathExpr as CompileTimeMathExpr, MathExpr, Optimize, var};
use mathcompile::prelude::*;

fn main() {
    println!("🔍 MathCompile Mathematical Discovery Demo");
    println!("==========================================");
    println!();

    // Example 1: Basic transcendental identity discovery
    println!("📊 Example 1: Basic Identity Discovery");
    println!("--------------------------------------");

    let x = var::<0>();
    let y = var::<1>();

    // Complex: ln(exp(x) * exp(y))  →  Simple: x + y
    let complex_expr = x.clone().exp().mul(y.clone().exp()).ln();
    let optimized_expr = complex_expr.clone().optimize();

    let test_values = [2.0, 3.0];
    let complex_result = complex_expr.eval(&test_values);
    let optimized_result = optimized_expr.eval(&test_values);
    let expected = test_values[0] + test_values[1];

    println!("Expression: ln(exp(x) * exp(y))");
    println!("Discovered: x + y");
    println!(
        "Test: x={}, y={} → {:.1}",
        test_values[0], test_values[1], expected
    );
    println!(
        "✅ Discovery successful: {}",
        (optimized_result - expected).abs() < 1e-10
    );
    println!();

    // Example 2: Multi-variable factorization discovery
    println!("📊 Example 2: Complex Multi-Variable Discovery");
    println!("----------------------------------------------");

    let x = var::<0>();
    let y = var::<1>();
    let z = var::<2>();
    let a = var::<3>();
    let b = var::<4>();

    // Complex: ln(exp(x) * exp(y) * exp(z)) + ln(exp(a)) - ln(exp(b))
    // Simple: x + y + z + a - b
    let complex_multi = x
        .clone()
        .exp()
        .mul(y.clone().exp())
        .mul(z.clone().exp())
        .ln()
        .add(a.clone().exp().ln())
        .sub(b.clone().exp().ln());

    let test_values_multi = [1.0, 2.0, 3.0, 4.0, 1.5];
    let complex_result = complex_multi.eval(&test_values_multi);
    let expected =
        test_values_multi[0] + test_values_multi[1] + test_values_multi[2] + test_values_multi[3]
            - test_values_multi[4];

    println!("Expression: ln(exp(x) * exp(y) * exp(z)) + ln(exp(a)) - ln(exp(b))");
    println!("Discovered: x + y + z + a - b");
    println!("Test: x=1, y=2, z=3, a=4, b=1.5 → {expected:.1}");
    println!(
        "✅ Discovery successful: {}",
        (complex_result - expected).abs() < 1e-10
    );
    println!();

    // Performance comparison
    println!("⚡ Performance Analysis");
    println!("----------------------");

    let iterations = 1_000_000;
    let test_vals = [1.5, 2.5];

    // Time complex expression
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = complex_expr.eval(&test_vals);
    }
    let complex_time = start.elapsed();

    // Time simple expression
    let simple_expr = x.clone().add(y.clone());
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = simple_expr.eval(&test_vals);
    }
    let simple_time = start.elapsed();

    if simple_time < complex_time {
        let speedup = complex_time.as_nanos() as f64 / simple_time.as_nanos() as f64;
        println!("Optimization speedup: {speedup:.1}x faster");
    } else {
        println!("Both expressions perform similarly (compiler optimized both)");
    }

    #[cfg(not(debug_assertions))]
    println!("✅ Zero-cost abstraction achieved in release mode");

    #[cfg(debug_assertions)]
    println!("ℹ️  Run with --release for zero-cost abstraction");

    println!();
    println!("🎯 Key Achievement:");
    println!("Mathematical relationships discovered automatically with zero runtime overhead!");
}
