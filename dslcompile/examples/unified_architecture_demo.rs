use dslcompile::ast::{ASTRepr, DynamicContext};
use dslcompile::symbolic::symbolic::{OptimizationConfig, SymbolicOptimizer};
use std::time::Instant;

fn main() {
    println!("🚀 Unified Architecture Demo: Expression Building → Strategy Selection");
    println!("====================================================================");

    // ============================================================================
    // STEP 1: BUILD EXPRESSIONS NATURALLY (Same API Always)
    // ============================================================================
    println!("\n📝 STEP 1: Building expressions using natural syntax");

    // Build expressions using the existing DynamicContext API
    let ctx = DynamicContext::new();
    let x = ctx.var();
    let y = ctx.var();

    // Create a complex expression: sin(x) + cos(y) * 2.0 + 1.0
    let sin_x = x.sin();
    let cos_y = y.cos();
    let two = ctx.constant(2.0);
    let one = ctx.constant(1.0);
    let cos_y_times_two = &cos_y * &two;
    let complex_expr: dslcompile::ast::TypedBuilderExpr<f64> = &sin_x + &cos_y_times_two + &one;

    // Convert to AST for optimization
    let ast_expr: ASTRepr<f64> = complex_expr.into();
    println!("  Built expression: sin(x) + cos(y) * 2.0 + 1.0");
    println!("  AST: {ast_expr:?}");

    let test_values = [std::f64::consts::PI / 2.0, 0.0]; // sin(π/2) = 1, cos(0) = 1
    let expected = 1.0 + 1.0 * 2.0 + 1.0; // = 4.0

    // ============================================================================
    // STEP 2: STRATEGY SELECTION (Configuration-Based)
    // ============================================================================
    println!("\n⚙️ STEP 2: Choosing optimization strategies via configuration");

    // Strategy 1: Zero-Overhead (Maximum Performance)
    println!("\n🔥 Strategy 1: Zero-Overhead (Maximum Performance)");
    let zero_overhead_config = OptimizationConfig::zero_overhead();
    println!("  Config: {zero_overhead_config:?}");

    let start = Instant::now();
    let mut zero_optimizer = SymbolicOptimizer::with_config(zero_overhead_config).unwrap();
    let zero_optimized = zero_optimizer.optimize(&ast_expr).unwrap();
    let zero_time = start.elapsed();

    println!("  Optimized AST: {zero_optimized:?}");
    println!("  Optimization time: {zero_time:?}");

    let zero_result = zero_optimized.eval_with_vars(&test_values);
    println!("  Result: {zero_result} (expected: {expected})");
    println!(
        "  ✅ Correctness: {}",
        if (zero_result - expected).abs() < 1e-10 {
            "PASS"
        } else {
            "FAIL"
        }
    );

    // Strategy 2: Dynamic Flexible (Interpretation)
    println!("\n🌊 Strategy 2: Dynamic Flexible (Interpretation)");
    let flexible_config = OptimizationConfig::dynamic_flexible();
    println!("  Config: {flexible_config:?}");

    let start = Instant::now();
    let mut flexible_optimizer = SymbolicOptimizer::with_config(flexible_config).unwrap();
    let flexible_optimized = flexible_optimizer.optimize(&ast_expr).unwrap();
    let flexible_time = start.elapsed();

    println!("  Optimized AST: {flexible_optimized:?}");
    println!("  Optimization time: {flexible_time:?}");

    let flexible_result = flexible_optimized.eval_with_vars(&test_values);
    println!("  Result: {flexible_result} (expected: {expected})");
    println!(
        "  ✅ Correctness: {}",
        if (flexible_result - expected).abs() < 1e-10 {
            "PASS"
        } else {
            "FAIL"
        }
    );

    // Strategy 3: Dynamic Performance (Codegen)
    println!("\n🚀 Strategy 3: Dynamic Performance (Codegen)");
    let performance_config = OptimizationConfig::dynamic_performance();
    println!("  Config: {performance_config:?}");

    let start = Instant::now();
    let mut performance_optimizer = SymbolicOptimizer::with_config(performance_config).unwrap();
    let performance_optimized = performance_optimizer.optimize(&ast_expr).unwrap();
    let performance_time = start.elapsed();

    println!("  Optimized AST: {performance_optimized:?}");
    println!("  Optimization time: {performance_time:?}");

    let performance_result = performance_optimized.eval_with_vars(&test_values);
    println!("  Result: {performance_result} (expected: {expected})");
    println!(
        "  ✅ Correctness: {}",
        if (performance_result - expected).abs() < 1e-10 {
            "PASS"
        } else {
            "FAIL"
        }
    );

    // Strategy 4: Adaptive (Smart Selection)
    println!("\n🧠 Strategy 4: Adaptive (Smart Selection)");
    let adaptive_config = OptimizationConfig::adaptive();
    println!("  Config: {adaptive_config:?}");

    let start = Instant::now();
    let mut adaptive_optimizer = SymbolicOptimizer::with_config(adaptive_config).unwrap();
    let adaptive_optimized = adaptive_optimizer.optimize(&ast_expr).unwrap();
    let adaptive_time = start.elapsed();

    println!("  Optimized AST: {adaptive_optimized:?}");
    println!("  Optimization time: {adaptive_time:?}");

    let adaptive_result = adaptive_optimized.eval_with_vars(&test_values);
    println!("  Result: {adaptive_result} (expected: {expected})");
    println!(
        "  ✅ Correctness: {}",
        if (adaptive_result - expected).abs() < 1e-10 {
            "PASS"
        } else {
            "FAIL"
        }
    );

    // ============================================================================
    // STEP 3: PERFORMANCE COMPARISON
    // ============================================================================
    println!("\n📊 STEP 3: Performance comparison");

    let iterations = 100_000;
    println!("  Running {iterations} evaluations for each strategy...");

    // Benchmark zero-overhead
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = zero_optimized.eval_with_vars(&test_values);
    }
    let zero_eval_time = start.elapsed();

    // Benchmark flexible
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = flexible_optimized.eval_with_vars(&test_values);
    }
    let flexible_eval_time = start.elapsed();

    // Benchmark performance
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = performance_optimized.eval_with_vars(&test_values);
    }
    let performance_eval_time = start.elapsed();

    // Benchmark adaptive
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = adaptive_optimized.eval_with_vars(&test_values);
    }
    let adaptive_eval_time = start.elapsed();

    println!("\n🏆 Performance Results ({iterations} iterations):");
    println!(
        "  Zero-Overhead:     {:?} ({:.2}ns per eval)",
        zero_eval_time,
        zero_eval_time.as_nanos() as f64 / f64::from(iterations)
    );
    println!(
        "  Dynamic Flexible:  {:?} ({:.2}ns per eval)",
        flexible_eval_time,
        flexible_eval_time.as_nanos() as f64 / f64::from(iterations)
    );
    println!(
        "  Dynamic Performance: {:?} ({:.2}ns per eval)",
        performance_eval_time,
        performance_eval_time.as_nanos() as f64 / f64::from(iterations)
    );
    println!(
        "  Adaptive:          {:?} ({:.2}ns per eval)",
        adaptive_eval_time,
        adaptive_eval_time.as_nanos() as f64 / f64::from(iterations)
    );

    // ============================================================================
    // STEP 4: ZERO-OVERHEAD DEMONSTRATION
    // ============================================================================
    println!("\n⚡ STEP 4: Zero-Overhead demonstration with constant expressions");

    // Create an expression with only constants: 3.0 + 4.0 * 2.0
    let const_expr = ASTRepr::Add(
        Box::new(ASTRepr::Constant(3.0)),
        Box::new(ASTRepr::Mul(
            Box::new(ASTRepr::Constant(4.0)),
            Box::new(ASTRepr::Constant(2.0)),
        )),
    );

    println!("  Constant expression: 3.0 + 4.0 * 2.0");
    println!("  Original AST: {const_expr:?}");

    // Zero-overhead should fold this to a single constant
    let mut zero_optimizer =
        SymbolicOptimizer::with_config(OptimizationConfig::zero_overhead()).unwrap();
    let const_optimized = zero_optimizer.optimize(&const_expr).unwrap();

    println!("  Zero-overhead optimized: {const_optimized:?}");

    match const_optimized {
        ASTRepr::Constant(val) => {
            println!("  ✅ SUCCESS: Folded to constant {val} (expected: 11.0)");
            assert!((val - 11.0).abs() < 1e-10, "Expected 11.0, got {val}");
        }
        _ => {
            println!("  ❌ FAILED: Not folded to constant");
        }
    }

    // ============================================================================
    // SUMMARY
    // ============================================================================
    println!("\n🎯 SUMMARY: Unified Architecture Benefits");
    println!("==========================================");
    println!("✅ Natural expression building (same API for all strategies)");
    println!("✅ Configuration-based optimization (no granular per-operation methods)");
    println!("✅ Zero-overhead available when needed");
    println!("✅ Full flexibility for dynamic expressions");
    println!("✅ Performance options for critical code");
    println!("✅ Smart adaptive selection");
    println!("✅ Backward compatible with existing code");
    println!("✅ Incremental migration path");

    println!("\n🚀 The unified architecture gives users the best of all worlds!");
}
