#!/usr/bin/env cargo run --example compile_time_performance_test

//! Compile-Time Performance Investigation
//!
//! This example investigates the performance characteristics of the compile-time
//! expression system to understand why simple operations are slower than expected.

use mathcompile::compile_time::{Add, MathExpr, Optimize, Var, constant, var, zero};

fn main() {
    println!("🔍 Compile-Time Performance Investigation");
    println!("========================================");
    println!();

    // Test 1: Raw Rust performance baseline
    println!("📊 Test 1: Raw Rust Performance Baseline");
    println!("----------------------------------------");

    let iterations = 1_000_000;
    let x_val = 1.5;
    let y_val = 2.5;

    // Pure Rust addition
    let start = std::time::Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += x_val + y_val;
    }
    let rust_time = start.elapsed();

    println!("Pure Rust addition ({iterations} iterations): {rust_time:?}");
    println!("Result: {:.6}", sum / f64::from(iterations));
    println!();

    // Test 2: Compile-time system with variables
    println!("📊 Test 2: Compile-Time System with Variables");
    println!("---------------------------------------------");

    let x = var::<0>();
    let y = var::<1>();
    let expr = x.clone().add(y.clone());
    let test_vals = [x_val, y_val];

    let start = std::time::Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += expr.eval(&test_vals);
    }
    let compile_time_var_time = start.elapsed();

    println!("Compile-time variables ({iterations} iterations): {compile_time_var_time:?}");
    println!("Result: {:.6}", sum / f64::from(iterations));

    if rust_time.as_nanos() > 0 {
        let slowdown = compile_time_var_time.as_nanos() as f64 / rust_time.as_nanos() as f64;
        println!("Slowdown vs pure Rust: {slowdown:.1}x");
    }
    println!();

    // Test 3: Compile-time system with constants
    println!("📊 Test 3: Compile-Time System with Constants");
    println!("---------------------------------------------");

    let const_x = constant(x_val);
    let const_y = constant(y_val);
    let const_expr = const_x.add(const_y);

    let start = std::time::Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += const_expr.eval(&[]);
    }
    let compile_time_const_time = start.elapsed();

    println!("Compile-time constants ({iterations} iterations): {compile_time_const_time:?}");
    println!("Result: {:.6}", sum / f64::from(iterations));

    if rust_time.as_nanos() > 0 {
        let slowdown = compile_time_const_time.as_nanos() as f64 / rust_time.as_nanos() as f64;
        println!("Slowdown vs pure Rust: {slowdown:.1}x");
    }
    println!();

    // Test 4: Single variable access overhead
    println!("📊 Test 4: Variable Access Overhead");
    println!("-----------------------------------");

    let x = var::<0>();
    let test_vals = [x_val];

    let start = std::time::Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += x.eval(&test_vals);
    }
    let var_access_time = start.elapsed();

    println!("Single variable access ({iterations} iterations): {var_access_time:?}");
    println!("Result: {:.6}", sum / f64::from(iterations));

    if rust_time.as_nanos() > 0 {
        let slowdown = var_access_time.as_nanos() as f64 / rust_time.as_nanos() as f64;
        println!("Slowdown vs pure Rust: {slowdown:.1}x");
    }
    println!();

    // Test 5: Array access overhead
    println!("📊 Test 5: Array Access Overhead");
    println!("---------------------------------");

    let test_vals = [x_val, y_val];

    let start = std::time::Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += test_vals[0] + test_vals[1];
    }
    let array_access_time = start.elapsed();

    println!("Direct array access ({iterations} iterations): {array_access_time:?}");
    println!("Result: {:.6}", sum / f64::from(iterations));

    if rust_time.as_nanos() > 0 {
        let slowdown = array_access_time.as_nanos() as f64 / rust_time.as_nanos() as f64;
        println!("Slowdown vs pure Rust: {slowdown:.1}x");
    }
    println!();

    // Test 6: Function call overhead
    println!("📊 Test 6: Function Call Overhead");
    println!("---------------------------------");

    fn simple_add(vars: &[f64]) -> f64 {
        vars[0] + vars[1]
    }

    let start = std::time::Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += simple_add(&test_vals);
    }
    let function_call_time = start.elapsed();

    println!("Function call overhead ({iterations} iterations): {function_call_time:?}");
    println!("Result: {:.6}", sum / f64::from(iterations));

    if rust_time.as_nanos() > 0 {
        let slowdown = function_call_time.as_nanos() as f64 / rust_time.as_nanos() as f64;
        println!("Slowdown vs pure Rust: {slowdown:.1}x");
    }
    println!();

    // Test 7: Trait method call overhead
    println!("📊 Test 7: Trait Method Call Overhead");
    println!("-------------------------------------");

    fn eval_trait_method<T: MathExpr>(expr: &T, vars: &[f64]) -> f64 {
        expr.eval(vars)
    }

    let start = std::time::Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += eval_trait_method(&expr, &test_vals);
    }
    let trait_method_time = start.elapsed();

    println!("Trait method call ({iterations} iterations): {trait_method_time:?}");
    println!("Result: {:.6}", sum / f64::from(iterations));

    if rust_time.as_nanos() > 0 {
        let slowdown = trait_method_time.as_nanos() as f64 / rust_time.as_nanos() as f64;
        println!("Slowdown vs pure Rust: {slowdown:.1}x");
    }
    println!();

    // Test 8: Complex expression type analysis
    println!("📊 Test 8: Expression Type Analysis");
    println!("-----------------------------------");

    // Let's see what the actual type looks like
    let x = var::<0>();
    let y = var::<1>();
    let simple_add = x.clone().add(y.clone());

    println!("Expression type: Add<Var<0>, Var<1>>");
    println!(
        "Size of expression: {} bytes",
        std::mem::size_of_val(&simple_add)
    );
    println!("Size of Var<0>: {} bytes", std::mem::size_of::<Var<0>>());
    println!(
        "Size of Add<Var<0>, Var<1>>: {} bytes",
        std::mem::size_of::<Add<Var<0>, Var<1>>>()
    );
    println!();

    // Test 9: Optimized expression performance
    println!("📊 Test 9: Optimized Expression Performance");
    println!("-------------------------------------------");

    // Create an expression that can be optimized
    let x = var::<0>();
    let zero_const = zero();
    let optimizable = x.clone().add(zero_const);
    let optimized = optimizable.clone().optimize();

    let start = std::time::Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += optimizable.eval(&[x_val]);
    }
    let unoptimized_time = start.elapsed();

    let start = std::time::Instant::now();
    let mut sum_opt = 0.0;
    for _ in 0..iterations {
        sum_opt += optimized.eval(&[x_val]);
    }
    let optimized_time = start.elapsed();

    println!("Unoptimized x + 0 ({iterations} iterations): {unoptimized_time:?}");
    println!("Optimized x + 0 → x ({iterations} iterations): {optimized_time:?}");

    if optimized_time.as_nanos() > 0 {
        let speedup = unoptimized_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;
        println!("Optimization speedup: {speedup:.2}x");
    }

    println!(
        "Results match: {}",
        (sum / f64::from(iterations) - sum_opt / f64::from(iterations)).abs() < 1e-10
    );
    println!();

    // Summary
    println!("🎓 Performance Analysis Summary");
    println!("==============================");
    println!();

    println!("Key findings:");
    println!(
        "• Pure Rust addition: {:?} ({:.2} ns/op)",
        rust_time,
        rust_time.as_nanos() as f64 / f64::from(iterations)
    );
    println!(
        "• Compile-time variables: {:?} ({:.2} ns/op)",
        compile_time_var_time,
        compile_time_var_time.as_nanos() as f64 / f64::from(iterations)
    );
    println!(
        "• Array access: {:?} ({:.2} ns/op)",
        array_access_time,
        array_access_time.as_nanos() as f64 / f64::from(iterations)
    );
    println!(
        "• Function calls: {:?} ({:.2} ns/op)",
        function_call_time,
        function_call_time.as_nanos() as f64 / f64::from(iterations)
    );
    println!();

    let var_overhead = compile_time_var_time.as_nanos() as f64 / rust_time.as_nanos() as f64;
    let array_overhead = array_access_time.as_nanos() as f64 / rust_time.as_nanos() as f64;
    let function_overhead = function_call_time.as_nanos() as f64 / rust_time.as_nanos() as f64;

    println!("Overhead analysis:");
    println!("• Variable system overhead: {var_overhead:.1}x");
    println!("• Array access overhead: {array_overhead:.1}x");
    println!("• Function call overhead: {function_overhead:.1}x");
    println!();

    if var_overhead > 100.0 {
        println!("🚨 CRITICAL: Variable system is >100x slower than pure Rust!");
        println!("   This indicates a fundamental performance issue.");
    } else if var_overhead > 10.0 {
        println!("⚠️  WARNING: Variable system is >10x slower than pure Rust");
        println!("   This suggests significant optimization opportunities.");
    } else if var_overhead > 2.0 {
        println!("📝 MODERATE: Variable system has some overhead");
        println!("   This is expected for abstraction layers.");
    } else {
        println!("✅ EXCELLENT: Variable system overhead is minimal");
        println!("   The abstraction cost is very reasonable.");
    }

    println!();
    println!("💡 Recommendations:");
    if var_overhead > 10.0 {
        println!("• Consider inline/const evaluation for hot paths");
        println!("• Investigate array bounds checking overhead");
        println!("• Profile trait method dispatch costs");
        println!("• Consider procedural macros for zero-cost abstractions");
    } else {
        println!("• Current performance is reasonable for the abstraction level");
        println!("• Focus on algorithmic optimizations rather than micro-optimizations");
    }
}
