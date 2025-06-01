#!/usr/bin/env cargo run --example compile_time_demo

//! Compile-Time Expression Demo
//!
//! This example demonstrates the trait-based compile-time expression system
//! that provides zero-overhead mathematical composition and optimization.
//! All optimizations happen at compile time through trait resolution.

use mathcompile::compile_time::{MathExpr, Optimize, var};

fn main() {
    println!("🚀 MathCompile Compile-Time Expression Demo");
    println!("==========================================");
    println!();

    println!("🎯 Demonstrating Zero-Overhead Compile-Time Expressions");
    println!("--------------------------------------------------------");
    println!("All composition and optimization happens at compile time!");
    println!("Runtime evaluation is just direct function calls with no overhead.");
    println!();

    // Example 1: Basic composition
    println!("📊 Example 1: Basic Expression Composition");
    println!("------------------------------------------");

    let x = var::<0>();
    let y = var::<1>();
    let z = var::<2>();

    // Build expression: (x + y) * z
    let expr1 = x.clone().add(y.clone()).mul(z.clone());

    let test_values = [2.0, 3.0, 4.0];
    let result1 = expr1.eval(&test_values);

    println!("Expression: (x + y) * z");
    println!(
        "Values: x={}, y={}, z={}",
        test_values[0], test_values[1], test_values[2]
    );
    println!("Result: {result1}");
    println!(
        "Expected: {}",
        (test_values[0] + test_values[1]) * test_values[2]
    );
    println!(
        "✅ Matches: {}",
        result1 == (test_values[0] + test_values[1]) * test_values[2]
    );
    println!();

    // Example 2: Transcendental functions
    println!("📊 Example 2: Transcendental Functions");
    println!("--------------------------------------");

    let a = var::<0>();
    let b = var::<1>();

    // Build expression: exp(a) * exp(b)
    let expr2 = a.clone().exp().mul(b.clone().exp());

    let test_values2 = [1.0, 2.0];
    let result2 = expr2.eval(&test_values2);
    let expected2 = test_values2[0].exp() * test_values2[1].exp();

    println!("Expression: exp(a) * exp(b)");
    println!("Values: a={}, b={}", test_values2[0], test_values2[1]);
    println!("Result: {result2:.6}");
    println!("Expected: {expected2:.6}");
    println!("✅ Matches: {}", (result2 - expected2).abs() < 1e-10);
    println!();

    // Example 3: Compile-time optimization - ln(exp(x)) → x
    println!("🔧 Example 3: Compile-Time Optimization - ln(exp(x)) → x");
    println!("----------------------------------------------------------");

    let x = var::<0>();

    // Original expression: ln(exp(x))
    let original_expr = x.clone().exp().ln();

    // Optimized expression (compile-time transformation)
    let optimized_expr = original_expr.clone().optimize();

    let test_value = 2.5;
    let original_result = original_expr.eval(&[test_value]);
    let optimized_result = optimized_expr.eval(&[test_value]);

    println!("Original expression: ln(exp(x))");
    println!("Optimized expression: x  (compile-time transformation!)");
    println!("Test value: x={test_value}");
    println!("Original result: {original_result:.10}");
    println!("Optimized result: {optimized_result:.10}");
    println!("Expected (x): {test_value:.10}");
    println!(
        "✅ Optimization preserves correctness: {}",
        (optimized_result - test_value).abs() < 1e-10
    );
    println!(
        "✅ Results match: {}",
        (original_result - optimized_result).abs() < 1e-10
    );
    println!();

    // Example 4: Complex nested optimization
    println!("🔧 Example 4: Complex Nested Expression with Optimization");
    println!("----------------------------------------------------------");

    let x = var::<0>();
    let y = var::<1>();
    let z = var::<2>();

    // Build the complex expression from our factorization demo:
    // ln(exp(x) * exp(y) * exp(z)) + ln(exp(a)) - ln(exp(b))
    // But we'll use a simpler version: ln(exp(x) * exp(y)) + ln(exp(z))
    let complex_expr = x
        .clone()
        .exp()
        .mul(y.clone().exp())
        .ln()
        .add(z.clone().exp().ln());

    let test_values3 = [1.0, 2.0, 3.0];
    let complex_result = complex_expr.eval(&test_values3);
    let expected_simple = test_values3[0] + test_values3[1] + test_values3[2]; // Should be x + y + z

    println!("Complex expression: ln(exp(x) * exp(y)) + ln(exp(z))");
    println!("Hidden simple form: x + y + z");
    println!(
        "Values: x={}, y={}, z={}",
        test_values3[0], test_values3[1], test_values3[2]
    );
    println!("Complex result: {complex_result:.6}");
    println!("Expected simple: {expected_simple:.6}");
    println!(
        "✅ Mathematical equivalence: {}",
        (complex_result - expected_simple).abs() < 1e-10
    );
    println!();

    // Example 5: Function composition
    println!("🔗 Example 5: Function Composition");
    println!("----------------------------------");

    // Define reusable expression functions
    fn quadratic_expr() -> impl MathExpr {
        let x = var::<0>();
        let a = var::<1>();
        let b = var::<2>();
        let c = var::<3>();

        // ax² + bx + c
        a.clone()
            .mul(x.clone().mul(x.clone()))
            .add(b.clone().mul(x.clone()))
            .add(c.clone())
    }

    fn exponential_decay_expr() -> impl MathExpr {
        let initial = var::<0>();
        let rate = var::<1>();
        let time = var::<2>();

        // initial * exp(-rate * time)
        initial
            .clone()
            .mul(rate.clone().neg().mul(time.clone()).exp())
    }

    // Use the composed expressions
    let quad_expr = quadratic_expr();
    let decay_expr = exponential_decay_expr();

    let quad_result = quad_expr.eval(&[2.0, 1.0, -1.0, 3.0]); // x=2, a=1, b=-1, c=3 → 1*4 + (-1)*2 + 3 = 5
    let decay_result = decay_expr.eval(&[10.0, 0.5, 2.0]); // initial=10, rate=0.5, time=2 → 10*exp(-1) ≈ 3.68

    println!("Quadratic expression: ax² + bx + c");
    println!("  Values: x=2, a=1, b=-1, c=3");
    println!("  Result: {quad_result:.6}");
    println!("  Expected: {:.6}", 5.0);

    println!("Exponential decay: initial * exp(-rate * time)");
    println!("  Values: initial=10, rate=0.5, time=2");
    println!("  Result: {decay_result:.6}");
    println!("  Expected: {:.6}", 10.0_f64 * (-0.5_f64 * 2.0_f64).exp());
    println!();

    // Example 6: Performance comparison
    println!("⚡ Example 6: Performance Analysis");
    println!("---------------------------------");

    let x = var::<0>();
    let y = var::<1>();

    // Complex expression that could be optimized
    let complex_perf_expr = x.clone().exp().mul(y.clone().exp()).ln();

    // Simple equivalent expression
    let simple_perf_expr = x.clone().add(y.clone());

    let iterations = 1_000_000;
    let test_vals = [1.5, 2.5];

    // Time complex expression
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = complex_perf_expr.eval(&test_vals);
    }
    let complex_time = start.elapsed();

    // Time simple expression
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = simple_perf_expr.eval(&test_vals);
    }
    let simple_time = start.elapsed();

    println!("Performance test ({iterations} iterations):");
    println!("Complex expression ln(exp(x) * exp(y)): {complex_time:?}");
    println!("Simple expression (x + y): {simple_time:?}");

    if simple_time < complex_time {
        let speedup = complex_time.as_nanos() as f64 / simple_time.as_nanos() as f64;
        println!("🚀 Simple form is {speedup:.2}x faster!");
    } else {
        println!("📊 Performance difference within measurement noise");
    }

    // Verify they produce the same result
    let complex_result = complex_perf_expr.eval(&test_vals);
    let simple_result = simple_perf_expr.eval(&test_vals);
    println!(
        "✅ Results match: {}",
        (complex_result - simple_result).abs() < 1e-10
    );
    println!();

    println!("🎓 Key Benefits of Compile-Time Expressions:");
    println!("--------------------------------------------");
    println!("✅ Zero runtime overhead - all composition resolved at compile time");
    println!("✅ Type safety - invalid expressions caught at compile time");
    println!("✅ Automatic optimization - mathematical simplifications applied automatically");
    println!("✅ Perfect inlining - compiler can optimize across expression boundaries");
    println!("✅ Composability - expressions can be built from other expressions");
    println!("✅ No allocations - all evaluation is stack-based");
    println!();

    println!("🔮 Limitations and Future Work:");
    println!("-------------------------------");
    println!("• Limited optimization patterns (only what's encoded in traits)");
    println!("• Complex type signatures for deep expressions");
    println!("• Compilation time may increase with complex expressions");
    println!("• Need more sophisticated constant handling (f64 const generics)");
    println!("• Could benefit from procedural macros for better syntax");
    println!();

    println!("✨ Demo Complete!");
    println!("This demonstrates compile-time mathematical expression composition");
    println!("with zero runtime overhead and automatic optimization!");
}
