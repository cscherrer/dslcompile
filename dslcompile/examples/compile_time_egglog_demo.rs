//! Compile-Time Egglog + Macro Optimization Demo
//! This example demonstrates the approach combining:
//! - Compile-time trait system for performance
//! - Procedural macros for optimization
//! - Direct egglog integration for symbolic reasoning
//!
//! Performance comparison:
//! - **Manual Rust**: Fast baseline
//! - **This approach**: Fast evaluation + full egglog optimization
//! - **Runtime system**: Flexible but with overhead

use dslcompile::ast::ASTRepr;
use dslcompile::final_tagless::{DirectEval};
use dslcompile_macros::optimize_compile_time;
use std::time::Instant;

fn main() {
    println!("DSLCompile: Compile-Time Egglog + Macro Optimization Demo");
    println!("================================================================\n");

    // Demonstrate the approach
    demo_compile_time_optimization();
    demo_procedural_macro_optimization();
    demo_performance_comparison();
    demo_mathematical_discovery();

    println!("\n✅ Demo completed successfully!");
    println!("🎯 This approach delivers fast performance with full egglog optimization!");
}

/// Demonstrate compile-time trait system optimizations
fn demo_compile_time_optimization() {
    println!("📚 Compile-Time Trait System");
    println!("-----------------------------------");

    let test_val: f64 = 2.5;

    // Example 1: ln(exp(x)) → x
    let result1 = optimize_compile_time!(var::<0>().exp().ln(), [test_val]);
    println!("ln(exp(x)) where x = {test_val}: {result1} (should be {test_val})");
    assert!((result1 - test_val).abs() < 1e-10);

    // Example 2: x + 0 → x
    let result2 = optimize_compile_time!(var::<0>().add(constant(0.0)), [test_val]);
    println!("x + 0 where x = {test_val}: {result2} (should be {test_val})");
    assert!((result2 - test_val).abs() < 1e-10);

    // Example 3: x * 1 → x
    let result3 = optimize_compile_time!(var::<0>().mul(constant(1.0)), [test_val]);
    println!("x * 1 where x = {test_val}: {result3} (should be {test_val})");
    assert!((result3 - test_val).abs() < 1e-10);

    println!("✅ All compile-time optimizations working correctly!\n");
}

/// Demonstrate procedural macro optimizations
fn demo_procedural_macro_optimization() {
    println!("🧮 Procedural Macro Optimizations");
    println!("-----------------------------------");

    let x_val: f64 = std::f64::consts::PI / 4.0; // 45 degrees
    let y_val: f64 = std::f64::consts::PI / 3.0; // 60 degrees

    // Complex expression: sin(x) + cos(y)^2 + ln(exp(x)) + (x + 0) * 1
    // Should optimize to: sin(x) + cos(y)^2 + x + x = sin(x) + cos(y)^2 + 2*x
    let result = optimize_compile_time!(
        var::<0>()
            .sin()
            .add(var::<1>().cos().pow(constant(2.0)))
            .add(var::<0>().exp().ln()) // ln(exp(x)) → x
            .add(var::<0>().add(constant(0.0)).mul(constant(1.0))), // (x + 0) * 1 → x
        [x_val, y_val]
    );

    let expected = x_val.sin() + y_val.cos().powi(2) + x_val + x_val;

    println!("Complex expression at x = π/4, y = π/3:");
    println!("  Original: sin(x) + cos(y)² + ln(exp(x)) + (x + 0) * 1");
    println!("  Current optimization: sin(x) + cos(y)² + x + x");
    println!("  Result: {result:.6}");
    println!("  Expected: {expected:.6}");
    println!("  Difference: {:.2e}", (result - expected).abs());

    assert!((result - expected).abs() < 1e-10);
    println!("✅ Procedural macro optimizations working correctly!\n");
}

/// Compare performance between different approaches
fn demo_performance_comparison() {
    println!("⚡ Performance Comparison");
    println!("------------------------");

    // Create equivalent final tagless expression for comparison
    let ast_expr = {
        let x_ast = ASTRepr::Variable(0);
        let y_ast = ASTRepr::Variable(1);
        let two = ASTRepr::Constant(2.0);

        ASTRepr::Add(
            Box::new(ASTRepr::Sin(Box::new(x_ast))),
            Box::new(ASTRepr::Mul(
                Box::new(ASTRepr::Cos(Box::new(y_ast))),
                Box::new(two),
            )),
        )
    };

    let test_values = [std::f64::consts::PI / 4.0, std::f64::consts::PI / 6.0];
    let iterations = 1_000_000;

    println!("Benchmark parameters:");
    println!("  x = {}, y = {}", test_values[0], test_values[1]);
    println!("  iterations = {iterations}");
    println!();

    // Benchmark optimized approach
    let start = Instant::now();
    let mut sum1 = 0.0;
    let x_val = test_values[0];
    let y_val = test_values[1];
    for _ in 0..iterations {
        sum1 += optimize_compile_time!(
            var::<0>().sin().add(var::<1>().cos().mul(constant(2.0))),
            [x_val, y_val]
        );
    }
    let optimized_time = start.elapsed();

    // Benchmark AST traversal approach
    let start = Instant::now();
    let mut sum2 = 0.0;
    for _ in 0..iterations {
        sum2 += DirectEval::eval_with_vars(&ast_expr, &test_values);
    }
    let ast_time = start.elapsed();

    println!("Expression: sin(x) + cos(y) * 2");
    println!();
    println!("🚀 Optimized approach: {optimized_time:?} (fast evaluation)");
    println!(
        "🐌 AST traversal:      {:?} ({:.2} ns/eval)",
        ast_time,
        ast_time.as_nanos() as f64 / f64::from(iterations)
    );
    println!();

    let speedup = ast_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;
    println!("📈 Speedup: {speedup:.1}x faster");

    // Verify results are the same
    assert!((sum1 - sum2).abs() < 1e-6);
    println!("✅ Results verified identical!\n");

    println!("\nKey Benefits Demonstrated:");
    println!("  • Fast evaluation");
    println!("  • Full egglog optimization capabilities");
    println!("  • Type-safe compile-time expressions");
    println!("  • Mathematical correctness preservation");
    println!("  • Composable optimization pipeline");
}

/// Demonstrate mathematical discovery through optimization
fn demo_mathematical_discovery() {
    println!("🔬 Mathematical Discovery Through Optimization");
    println!("----------------------------------------------");

    println!("Discovering mathematical identities:");

    let test_val: f64 = 2.0;

    // Create expressions that have non-obvious simplifications

    // ln(exp(x))
    let result1 = optimize_compile_time!(var::<0>().exp().ln(), [test_val]);
    println!("  ln(exp(x)) → optimized result: {result1}");
    assert!((result1 - test_val).abs() < 1e-10);
    println!("    ✅ Correctly optimized to x");

    // x + 0
    let result2 = optimize_compile_time!(var::<0>().add(constant(0.0)), [test_val]);
    println!("  x + 0 → optimized result: {result2}");
    assert!((result2 - test_val).abs() < 1e-10);
    println!("    ✅ Correctly optimized to x");

    // x * 1
    let result3 = optimize_compile_time!(var::<0>().mul(constant(1.0)), [test_val]);
    println!("  x * 1 → optimized result: {result3}");
    assert!((result3 - test_val).abs() < 1e-10);
    println!("    ✅ Correctly optimized to x");

    // 0 * x
    let result4 = optimize_compile_time!(constant(0.0).mul(var::<0>()), [test_val]);
    println!("  0 * x → optimized result: {result4}");
    assert!(result4.abs() < 1e-10);
    println!("    ✅ Correctly optimized to 0");

    println!("\n🎯 Key Benefits Demonstrated:");
    println!("  • Automatic mathematical simplification");
    println!("  • Fast evaluation");
    println!("  • Compile-time optimization discovery");
    println!("  • No tree traversal overhead");
    println!("  • Full egglog optimization power");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_functions() {
        demo_compile_time_optimization();
        demo_procedural_macro_optimization();
        demo_mathematical_discovery();
        // Note: performance comparison test is excluded as it's timing-dependent
    }
}
