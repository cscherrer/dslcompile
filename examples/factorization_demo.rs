#!/usr/bin/env cargo run --example factorization_demo

//! Mathematical Discovery Demo - Compile-Time Factorization with Egglog
//!
//! This example demonstrates how `MathCompile` can automatically discover
//! mathematical factorizations and simplifications using the breakthrough
//! compile-time egglog + macro optimization system.

use mathcompile::compile_time::{MathExpr, constant, var};
use mathcompile_macros::optimize_compile_time;
use std::time::Instant;

fn main() {
    println!("🚀 MathCompile Zero-Cost Optimization Demo");
    println!("============================================");
    println!();

    // Define test variables
    let x: f64 = 2.5;
    let y: f64 = 1.5;
    let z: f64 = 0.8;

    println!("📊 Performance Comparison: Zero-Cost vs Runtime Optimization");
    println!("Variables: x = {x}, y = {y}, z = {z}");
    println!();

    // Test 1: Simple trigonometric expression
    println!("🧮 Test 1: sin(x) + cos(y)");
    println!("---------------------------");

    // Zero-cost optimization (generates direct code)
    let start = Instant::now();
    let result1_optimized = optimize_compile_time!(var::<0>().sin().add(var::<1>().cos()), [x, y]);
    let time1_optimized = start.elapsed();

    // Manual calculation for comparison
    let start = Instant::now();
    let result1_manual = x.sin() + y.cos();
    let time1_manual = start.elapsed();

    println!("Zero-cost result: {result1_optimized:.10}");
    println!("Manual result:    {result1_manual:.10}");
    println!(
        "Difference:       {:.2e}",
        (result1_optimized - result1_manual).abs()
    );
    println!("Zero-cost time:   {time1_optimized:?}");
    println!("Manual time:      {time1_manual:?}");
    println!();

    // Test 2: Complex expression with optimization opportunities
    println!("🔬 Test 2: ln(exp(x)) + y * 1 + 0 * z (should optimize to x + y)");
    println!("------------------------------------------------------------------");

    let start = Instant::now();
    let result2_optimized = optimize_compile_time!(
        var::<0>()
            .exp()
            .ln()
            .add(var::<1>().mul(constant(1.0)))
            .add(var::<2>().mul(constant(0.0))),
        [x, y, z]
    );
    let time2_optimized = start.elapsed();

    let start = Instant::now();
    let result2_manual = x + y; // Expected optimized result
    let time2_manual = start.elapsed();

    println!("Zero-cost result: {result2_optimized:.10}");
    println!("Expected (x + y): {result2_manual:.10}");
    println!(
        "Difference:       {:.2e}",
        (result2_optimized - result2_manual).abs()
    );
    println!("Zero-cost time:   {time2_optimized:?}");
    println!("Manual time:      {time2_manual:?}");
    println!();

    // Test 3: Nested expression with multiple optimization opportunities
    println!("🎯 Test 3: exp(ln(x) + ln(y)) (should optimize to x * y)");
    println!("----------------------------------------------------------");

    let start = Instant::now();
    let result3_optimized =
        optimize_compile_time!(var::<0>().ln().add(var::<1>().ln()).exp(), [x, y]);
    let time3_optimized = start.elapsed();

    let start = Instant::now();
    let result3_manual = x * y; // Expected optimized result
    let time3_manual = start.elapsed();

    println!("Zero-cost result: {result3_optimized:.10}");
    println!("Expected (x * y): {result3_manual:.10}");
    println!(
        "Difference:       {:.2e}",
        (result3_optimized - result3_manual).abs()
    );
    println!("Zero-cost time:   {time3_optimized:?}");
    println!("Manual time:      {time3_manual:?}");
    println!();

    // Test 4: Performance benchmark with repeated evaluations
    println!("⚡ Test 4: Performance Benchmark (1M evaluations)");
    println!("--------------------------------------------------");

    // Benchmark zero-cost optimization
    let start = Instant::now();
    let mut sum_optimized = 0.0;
    for i in 0..1_000_000 {
        let xi = x + f64::from(i) * 0.000001;
        let yi = y + f64::from(i) * 0.000001;
        sum_optimized += optimize_compile_time!(
            var::<0>().sin().add(var::<1>().cos().pow(constant(2.0))),
            [xi, yi]
        );
    }
    let time4_optimized = start.elapsed();

    // Benchmark manual calculation
    let start = Instant::now();
    let mut sum_manual = 0.0;
    for i in 0..1_000_000 {
        let xi = x + f64::from(i) * 0.000001;
        let yi = y + f64::from(i) * 0.000001;
        sum_manual += xi.sin() + yi.cos().powf(2.0);
    }
    let time4_manual = start.elapsed();

    println!("Zero-cost sum:    {sum_optimized:.6}");
    println!("Manual sum:       {sum_manual:.6}");
    println!(
        "Difference:       {:.2e}",
        (sum_optimized - sum_manual).abs()
    );
    println!(
        "Zero-cost time:   {:?} ({:.2} ns/eval)",
        time4_optimized,
        time4_optimized.as_nanos() as f64 / 1_000_000.0
    );
    println!(
        "Manual time:      {:?} ({:.2} ns/eval)",
        time4_manual,
        time4_manual.as_nanos() as f64 / 1_000_000.0
    );

    let speedup = time4_manual.as_nanos() as f64 / time4_optimized.as_nanos() as f64;
    println!("Speedup:          {speedup:.2}x");
    println!();

    // Test 5: Code generation demonstration
    println!("🔧 Test 5: Generated Code Inspection");
    println!("-------------------------------------");

    use mathcompile::compile_time::optimized::{ToAst, equality_saturation, generate_direct_code};

    let expr5 = var::<0>().sin().add(var::<1>().cos().pow(constant(2.0)));
    let ast = expr5.to_ast();
    let optimized_ast = equality_saturation(&ast, 10);
    let generated_code = generate_direct_code(&optimized_ast, &["x", "y"]);

    println!("Original expression: sin(x) + cos(y)^2");
    println!("Generated code:      {generated_code}");
    println!();

    println!("✅ Demo completed successfully!");
    println!();
    println!("🎉 Key Achievements:");
    println!("   • Zero runtime dispatch (no Box<dyn Fn>, no enums)");
    println!("   • Compile-time egglog optimization");
    println!("   • Direct Rust code generation");
    println!("   • Performance matching hand-written code");
    println!("   • Complete mathematical reasoning via equality saturation");
}
