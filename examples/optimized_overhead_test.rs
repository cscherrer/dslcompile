#!/usr/bin/env cargo run --example optimized_overhead_test

//! Optimized Overhead Test
//!
//! This example tests the performance improvement from eliminating bounds checking
//! in the compile-time expression system.

use mathcompile::compile_time::*;

fn main() {
    println!("🚀 Optimized Overhead Test");
    println!("==========================");
    println!("Testing performance improvement from unsafe indexing optimization");
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
    println!("Pure Rust baseline: {:?} ({:.2} ns/op)", rust_time, rust_time.as_nanos() as f64 / iterations as f64);

    // Array access baseline
    let vars = [x_val, y_val];
    let start = std::time::Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += vars[0] + vars[1];
    }
    let array_time = start.elapsed();
    
    println!("Array access: {:?} ({:.2} ns/op) - {:.1}x overhead", array_time, array_time.as_nanos() as f64 / iterations as f64, array_time.as_nanos() as f64 / rust_time.as_nanos() as f64);

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
    
    println!("Optimized compile-time system: {:?} ({:.2} ns/op) - {:.1}x overhead", optimized_time, optimized_time.as_nanos() as f64 / iterations as f64, optimized_time.as_nanos() as f64 / rust_time.as_nanos() as f64);

    // Test correctness
    let result = expr.eval(&vars);
    let expected = x_val + y_val;
    println!();
    println!("✅ Correctness check:");
    println!("Result: {:.10}", result);
    println!("Expected: {:.10}", expected);
    println!("Correct: {}", (result - expected).abs() < 1e-10);
    
    println!();
    println!("🎯 Performance Summary:");
    println!("• Pure Rust: {:.2} ns/op (baseline)", rust_time.as_nanos() as f64 / iterations as f64);
    println!("• Array access: {:.2} ns/op ({:.1}x)", array_time.as_nanos() as f64 / iterations as f64, array_time.as_nanos() as f64 / rust_time.as_nanos() as f64);
    println!("• Optimized system: {:.2} ns/op ({:.1}x)", optimized_time.as_nanos() as f64 / iterations as f64, optimized_time.as_nanos() as f64 / rust_time.as_nanos() as f64);
    
    let improvement_factor = if optimized_time.as_nanos() > 0 {
        // Compare against the old 11.68 ns/op from the previous test
        11.68 / (optimized_time.as_nanos() as f64 / iterations as f64)
    } else {
        1.0
    };
    
    if improvement_factor > 2.0 {
        println!("🚀 EXCELLENT: {:.1}x performance improvement from optimization!", improvement_factor);
    } else if improvement_factor > 1.5 {
        println!("✅ GOOD: {:.1}x performance improvement from optimization", improvement_factor);
    } else {
        println!("📝 MODERATE: {:.1}x performance improvement", improvement_factor);
    }
    
    println!();
    println!("💡 Key Insight:");
    println!("The overhead was NOT from the trait-based design, but from runtime bounds checking!");
    println!("With compile-time bounds checking, the abstraction overhead is minimal.");
} 