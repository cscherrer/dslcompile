#!/usr/bin/env cargo run --example release_mode_test --release

//! Release Mode Performance Test
//!
//! This example tests whether the overhead is due to debug builds
//! lacking compiler optimizations.

use dslcompile::compile_time::{MathExpr, var};

fn main() {
    println!("🚀 Release Mode Performance Test");
    println!("================================");

    #[cfg(debug_assertions)]
    println!("⚠️  Running in DEBUG mode - optimizations disabled");

    #[cfg(not(debug_assertions))]
    println!("✅ Running in RELEASE mode - optimizations enabled");

    println!();

    let iterations = 10_000_000; // More iterations for release mode
    let x_val = 1.5;
    let y_val = 2.5;

    // Pure Rust baseline
    let start = std::time::Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += x_val + y_val;
    }
    let rust_time = start.elapsed();

    println!("📊 Performance Comparison ({iterations} iterations)");
    println!("------------------------------------------");
    println!(
        "Pure Rust baseline: {:?} ({:.3} ns/op)",
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
        "Array access: {:?} ({:.3} ns/op) - {:.2}x overhead",
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
        "Compile-time system: {:?} ({:.3} ns/op) - {:.2}x overhead",
        optimized_time,
        optimized_time.as_nanos() as f64 / f64::from(iterations),
        optimized_time.as_nanos() as f64 / rust_time.as_nanos() as f64
    );

    // Test with inline function to see if that helps
    #[inline(always)]
    fn inline_add(a: f64, b: f64) -> f64 {
        a + b
    }

    let start = std::time::Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += inline_add(vars[0], vars[1]);
    }
    let inline_time = start.elapsed();

    println!(
        "Inline function: {:?} ({:.3} ns/op) - {:.2}x overhead",
        inline_time,
        inline_time.as_nanos() as f64 / f64::from(iterations),
        inline_time.as_nanos() as f64 / rust_time.as_nanos() as f64
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
    println!("🎯 Performance Analysis:");

    let overhead = optimized_time.as_nanos() as f64 / rust_time.as_nanos() as f64;

    #[cfg(debug_assertions)]
    {
        println!("• Running in DEBUG mode - expect higher overhead");
        println!("• Compiler optimizations are disabled");
        println!("• Function inlining is limited");
        if overhead > 5.0 {
            println!("• {overhead:.1}x overhead is expected in debug builds");
        } else {
            println!("• {overhead:.1}x overhead is reasonable for debug builds");
        }
        println!("• Try running with --release flag for optimized performance");
    }

    #[cfg(not(debug_assertions))]
    {
        println!("• Running in RELEASE mode - optimizations enabled");
        if overhead < 1.5 {
            println!(
                "• {:.2}x overhead - EXCELLENT! Near zero-cost abstraction achieved",
                overhead
            );
        } else if overhead < 2.0 {
            println!(
                "• {:.2}x overhead - VERY GOOD! Minimal abstraction cost",
                overhead
            );
        } else if overhead < 3.0 {
            println!(
                "• {:.2}x overhead - GOOD! Reasonable abstraction cost",
                overhead
            );
        } else {
            println!("• {:.2}x overhead - Still room for improvement", overhead);
        }
    }

    println!();
    println!("💡 Key Insights:");
    println!("• Compiler optimization level significantly affects performance");
    println!("• Release builds enable function inlining and other optimizations");
    println!("• The trait-based system should perform much better in release mode");
}
