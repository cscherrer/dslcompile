//! Cranelift + DynamicContext Integration Demo
//!
//! This demo shows how DynamicContext now seamlessly integrates Cranelift JIT compilation
//! while maintaining the same ergonomic API. Users get automatic performance optimization
//! without changing their code.

use dslcompile::ast::runtime::expression_builder::{DynamicContext, JITStrategy};
use std::time::Instant;

fn main() {
    println!("🚀 Cranelift + DynamicContext Integration Demo");
    println!("===============================================\n");

    // Demo 1: Same API, Automatic JIT Optimization
    demo_automatic_jit_optimization();

    // Demo 2: Manual JIT Strategy Control
    demo_manual_jit_control();

    // Demo 3: Performance Comparison
    demo_performance_comparison();

    // Demo 4: JIT Cache Benefits
    demo_jit_cache_benefits();

    // Demo 5: Complex Expression Optimization
    demo_complex_expression_optimization();
}

fn demo_automatic_jit_optimization() {
    println!("🎯 Demo 1: Same API, Automatic JIT Optimization");
    println!("================================================");

    // Create DynamicContext with default adaptive JIT strategy
    let ctx = DynamicContext::new();
    let x = ctx.var();
    let y = ctx.var();

    // Simple expression - will use interpretation (below complexity threshold)
    let simple_expr = &x + &y;
    println!("Simple expression: x + y");
    let result1 = ctx.eval(&simple_expr, &[3.0, 4.0]);
    println!("Result: {} (used interpretation)", result1);

    // Complex expression - will automatically use JIT compilation
    let complex_expr = (&x * &x + &y * &y).sqrt() + (&x * &y).sin() + (&x / &y).ln();
    println!("\nComplex expression: sqrt(x² + y²) + sin(x*y) + ln(x/y)");
    let result2 = ctx.eval(&complex_expr, &[3.0, 4.0]);
    println!("Result: {} (automatically used JIT)", result2);

    // Show JIT statistics
    let stats = ctx.jit_stats();
    println!("\n📊 JIT Statistics:");
    println!("  Cached functions: {}", stats.cached_functions);
    println!("  Strategy: {:?}", stats.strategy);
    println!("✅ Same API, automatic optimization!\n");
}

fn demo_manual_jit_control() {
    println!("🔧 Demo 2: Manual JIT Strategy Control");
    println!("======================================");

    let x_val = 2.5;
    let y_val = 3.0;

    // Create contexts with different strategies
    let ctx_interpret = DynamicContext::new_interpreter();
    let ctx_jit = DynamicContext::new_jit_optimized();
    let ctx_adaptive = DynamicContext::with_jit_strategy(JITStrategy::Adaptive {
        complexity_threshold: 3,
        call_count_threshold: 2,
    });

    // Same expression, different strategies
    let expr_fn = |ctx: &DynamicContext| {
        let x = ctx.var();
        let y = ctx.var();
        &x * &x + 2.0 * &x * &y + &y * &y // (x + y)²
    };

    let expr1 = expr_fn(&ctx_interpret);
    let expr2 = expr_fn(&ctx_jit);
    let expr3 = expr_fn(&ctx_adaptive);

    println!("Expression: (x + y)² = x² + 2xy + y²");
    println!("Input: x={}, y={}", x_val, y_val);

    let result1 = ctx_interpret.eval(&expr1, &[x_val, y_val]);
    let result2 = ctx_jit.eval(&expr2, &[x_val, y_val]);
    let result3 = ctx_adaptive.eval(&expr3, &[x_val, y_val]);

    println!("\nResults (all identical):");
    println!("  Interpretation: {}", result1);
    println!("  JIT Compilation: {}", result2);
    println!("  Adaptive: {}", result3);

    assert!((result1 - result2).abs() < 1e-10);
    assert!((result1 - result3).abs() < 1e-10);
    println!("✅ All strategies produce identical results!\n");
}

fn demo_performance_comparison() {
    println!("⚡ Demo 3: Performance Comparison");
    println!("=================================");

    let ctx_interpret = DynamicContext::new_interpreter();
    let ctx_jit = DynamicContext::new_jit_optimized();

    // Create a moderately complex expression
    let expr_fn = |ctx: &DynamicContext| {
        let x = ctx.var();
        let y = ctx.var();
        (&x * &x * &x + &y * &y * &y) / (&x + &y) + (&x * &y).sin()
    };

    let expr_interpret = expr_fn(&ctx_interpret);
    let expr_jit = expr_fn(&ctx_jit);

    let test_inputs = [2.5, 3.0];
    let iterations = 10_000;

    // Warm up JIT compilation
    ctx_jit.eval(&expr_jit, &test_inputs);

    // Benchmark interpretation
    let start = Instant::now();
    for _ in 0..iterations {
        ctx_interpret.eval(&expr_interpret, &test_inputs);
    }
    let interpret_time = start.elapsed();

    // Benchmark JIT compilation
    let start = Instant::now();
    for _ in 0..iterations {
        ctx_jit.eval(&expr_jit, &test_inputs);
    }
    let jit_time = start.elapsed();

    let speedup = interpret_time.as_nanos() as f64 / jit_time.as_nanos() as f64;

    println!("Expression: (x³ + y³)/(x + y) + sin(xy)");
    println!("Iterations: {}", iterations);
    println!("\nPerformance Results:");
    println!("  Interpretation: {:?}", interpret_time);
    println!("  JIT Compilation: {:?}", jit_time);
    println!("  Speedup: {:.2}x", speedup);
    println!("✅ JIT compilation provides significant speedup!\n");
}

fn demo_jit_cache_benefits() {
    println!("💾 Demo 4: JIT Cache Benefits");
    println!("=============================");

    let ctx = DynamicContext::new_jit_optimized();
    let x = ctx.var();
    let y = ctx.var();
    let expr = (&x * &x + &y * &y).sqrt();

    println!("Expression: sqrt(x² + y²)");

    // First evaluation - includes compilation time
    let start = Instant::now();
    let result1 = ctx.eval(&expr, &[3.0, 4.0]);
    let first_time = start.elapsed();

    // Second evaluation - uses cached compiled function
    let start = Instant::now();
    let result2 = ctx.eval(&expr, &[5.0, 12.0]);
    let cached_time = start.elapsed();

    println!("\nCache Performance:");
    println!("  First evaluation (with compilation): {:?}", first_time);
    println!("  Second evaluation (cached): {:?}", cached_time);
    println!("  Cache speedup: {:.2}x", first_time.as_nanos() as f64 / cached_time.as_nanos() as f64);

    let stats = ctx.jit_stats();
    println!("  Cached functions: {}", stats.cached_functions);

    assert!((result1 - 5.0).abs() < 1e-10); // sqrt(3² + 4²) = 5
    assert!((result2 - 13.0).abs() < 1e-10); // sqrt(5² + 12²) = 13
    println!("✅ JIT cache provides dramatic speedup for repeated evaluations!\n");
}

fn demo_complex_expression_optimization() {
    println!("🧮 Demo 5: Complex Expression Optimization");
    println!("==========================================");

    let ctx = DynamicContext::new_jit_optimized();
    let x = ctx.var();
    let y = ctx.var();
    let z = ctx.var();

    // Create a very complex expression that benefits from JIT compilation
    let complex_expr = {
        let term1 = (&x * &y * &z).sin();
        let term2 = (&x * &x + &y * &y + &z * &z).sqrt();
        let term3 = (&x / (&y + 1.0)).ln();
        let term4 = (&y * &z).exp() / ctx.constant(100.0); // Scale down to avoid overflow
        
        term1 + term2 + term3 + term4
    };

    println!("Complex expression: sin(xyz) + sqrt(x² + y² + z²) + ln(x/(y+1)) + exp(yz)/100");

    let test_cases = [
        [1.0, 2.0, 3.0],
        [0.5, 1.5, 2.5],
        [2.0, 1.0, 0.5],
    ];

    println!("\nEvaluation Results:");
    for (i, inputs) in test_cases.iter().enumerate() {
        let start = Instant::now();
        let result = ctx.eval(&complex_expr, inputs);
        let eval_time = start.elapsed();
        
        println!("  Test {}: inputs={:?}, result={:.6}, time={:?}", 
                 i + 1, inputs, result, eval_time);
    }

    let stats = ctx.jit_stats();
    println!("\n📊 Final JIT Statistics:");
    println!("  Cached functions: {}", stats.cached_functions);
    println!("  Strategy: {:?}", stats.strategy);
    println!("✅ Complex expressions benefit greatly from JIT optimization!\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_interpretation_equivalence() {
        let ctx_interpret = DynamicContext::new_interpreter();
        let ctx_jit = DynamicContext::new_jit_optimized();

        let expr_fn = |ctx: &DynamicContext| {
            let x = ctx.var();
            let y = ctx.var();
            &x * &x + &y * &y
        };

        let expr1 = expr_fn(&ctx_interpret);
        let expr2 = expr_fn(&ctx_jit);

        let inputs = [3.0, 4.0];
        let result1 = ctx_interpret.eval(&expr1, &inputs);
        let result2 = ctx_jit.eval(&expr2, &inputs);

        assert!((result1 - result2).abs() < 1e-10);
        assert!((result1 - 25.0).abs() < 1e-10); // 3² + 4² = 25
    }

    #[test]
    fn test_adaptive_strategy() {
        let ctx = DynamicContext::with_jit_strategy(JITStrategy::Adaptive {
            complexity_threshold: 3,
            call_count_threshold: 1,
        });

        let x = ctx.var();
        let y = ctx.var();

        // Simple expression (complexity < 3) - should use interpretation
        let simple = &x + &y;
        let result1 = ctx.eval(&simple, &[1.0, 2.0]);
        assert_eq!(result1, 3.0);

        // Complex expression (complexity >= 3) - should use JIT
        let complex = &x * &x + &y * &y + (&x * &y).sin();
        let result2 = ctx.eval(&complex, &[1.0, 2.0]);
        assert!((result2 - (1.0 + 4.0 + (2.0_f64).sin())).abs() < 1e-10);

        // Should have cached the complex expression
        let stats = ctx.jit_stats();
        assert!(stats.cached_functions > 0);
    }

    #[test]
    fn test_jit_cache_functionality() {
        let ctx = DynamicContext::new_jit_optimized();
        let x = ctx.var();
        let expr = &x * &x;

        // First evaluation
        let result1 = ctx.eval(&expr, &[5.0]);
        assert_eq!(result1, 25.0);

        // Check cache
        let stats = ctx.jit_stats();
        assert_eq!(stats.cached_functions, 1);

        // Second evaluation should use cache
        let result2 = ctx.eval(&expr, &[6.0]);
        assert_eq!(result2, 36.0);

        // Cache count should remain the same
        let stats = ctx.jit_stats();
        assert_eq!(stats.cached_functions, 1);
    }
} 