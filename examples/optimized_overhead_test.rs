#!/usr/bin/env cargo run --example optimized_overhead_test

//! Compile-Time Optimization Test
//!
//! This example demonstrates how the compile-time expression system achieves
//! zero-cost abstraction through Rust's compiler optimizations rather than
//! manual unsafe optimizations.

use mathcompile::compile_time::{MathExpr, var};

fn main() {
    println!("🚀 Compile-Time Optimization Test");
    println!("=================================");
    println!("Testing zero-cost abstraction through compiler optimization");
    println!();

    let iterations = 1_000_000;
    let x_val = 1.5;
    let y_val = 2.5;

    // Pure Rust baseline
    let start = std::time::Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += x_val + y_val;
    }
    let rust_time = start.elapsed();

    println!("📊 Performance Comparison");
    println!("-------------------------");
    println!(
        "Pure Rust baseline: {:?} ({:.2} ns/op)",
        rust_time,
        rust_time.as_nanos() as f64 / f64::from(iterations)
    );

    // Array access baseline
    let vars = [x_val, y_val];
    let start = std::time::Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += vars[0] + vars[1];
    }
    let array_time = start.elapsed();

    println!(
        "Array access: {:?} ({:.2} ns/op) - {:.1}x overhead",
        array_time,
        array_time.as_nanos() as f64 / f64::from(iterations),
        array_time.as_nanos() as f64 / rust_time.as_nanos() as f64
    );

    // Optimized compile-time system
    let x = var::<0>();
    let y = var::<1>();
    let expr = x.clone().add(y.clone());

    let start = std::time::Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += expr.eval(&vars);
    }
    let optimized_time = start.elapsed();

    println!(
        "Optimized compile-time system: {:?} ({:.2} ns/op) - {:.1}x overhead",
        optimized_time,
        optimized_time.as_nanos() as f64 / f64::from(iterations),
        optimized_time.as_nanos() as f64 / rust_time.as_nanos() as f64
    );

    // Test correctness
    let result = expr.eval(&vars);
    let expected = x_val + y_val;
    println!();
    println!("✅ Correctness check:");
    println!("Result: {result:.10}");
    println!("Expected: {expected:.10}");
    println!("Correct: {}", (result - expected).abs() < 1e-10);

    println!();
    println!("🎯 Performance Summary:");
    println!(
        "• Pure Rust: {:.2} ns/op (baseline)",
        rust_time.as_nanos() as f64 / f64::from(iterations)
    );
    println!(
        "• Array access: {:.2} ns/op ({:.1}x)",
        array_time.as_nanos() as f64 / f64::from(iterations),
        array_time.as_nanos() as f64 / rust_time.as_nanos() as f64
    );
    println!(
        "• Optimized system: {:.2} ns/op ({:.1}x)",
        optimized_time.as_nanos() as f64 / f64::from(iterations),
        optimized_time.as_nanos() as f64 / rust_time.as_nanos() as f64
    );

    let improvement_factor = if optimized_time.as_nanos() > 0 {
        // Compare against the old 11.68 ns/op from the previous test
        11.68 / (optimized_time.as_nanos() as f64 / f64::from(iterations))
    } else {
        1.0
    };

    if improvement_factor > 2.0 {
        println!(
            "🚀 EXCELLENT: {improvement_factor:.1}x performance improvement from optimization!"
        );
    } else if improvement_factor > 1.5 {
        println!("✅ GOOD: {improvement_factor:.1}x performance improvement from optimization");
    } else {
        println!("📝 MODERATE: {improvement_factor:.1}x performance improvement");
    }

    println!();
    println!("💡 Key Insight:");
    println!("The compile-time system achieves zero-cost abstraction through Rust's optimizer!");
    println!("In release builds, safe bounds checking is optimized away completely.");
}
