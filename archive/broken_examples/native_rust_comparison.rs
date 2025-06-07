use dslcompile::ast::{ASTRepr, DynamicContext};
use dslcompile::symbolic::symbolic::{OptimizationConfig, SymbolicOptimizer};
// Note: zero_overhead_core removed - using Enhanced Scoped System instead
use std::time::Instant;

fn main() -> Result<()> {
    println!("🏁 DSL vs Native Rust Performance Comparison");
    println!("============================================");

    // Test values
    let x = std::f64::consts::PI / 2.0; // sin(π/2) = 1
    let y = 0.0; // cos(0) = 1
    let expected = 1.0 + 1.0 * 2.0 + 1.0; // = 4.0

    println!("Test expression: sin(x) + cos(y) * 2.0 + 1.0");
    println!("Test values: x = π/2, y = 0");
    println!("Expected result: {expected}");
    println!();

    // ============================================================================
    // NATIVE RUST BASELINES
    // ============================================================================

    println!("🦀 NATIVE RUST BASELINES");
    println!("========================");

    // Native Rust - Direct computation
    fn native_direct(x: f64, y: f64) -> f64 {
        x.sin() + y.cos() * 2.0 + 1.0
    }

    // Native Rust - With intermediate variables (more realistic)
    fn native_intermediate(x: f64, y: f64) -> f64 {
        let sin_x = x.sin();
        let cos_y = y.cos();
        let cos_y_times_two = cos_y * 2.0;
        sin_x + cos_y_times_two + 1.0
    }

    // Native Rust - Function pointer (dynamic dispatch overhead)
    fn native_function_ptr(x: f64, y: f64, f: fn(f64, f64) -> f64) -> f64 {
        f(x, y)
    }

    let iterations = 1_000_000;

    // Benchmark native direct
    let start = Instant::now();
    let mut result = 0.0;
    for _ in 0..iterations {
        result = native_direct(x, y);
    }
    let native_direct_time = start.elapsed();
    println!(
        "Native Direct:        {:?} ({:.2}ns per call) → {result}",
        native_direct_time,
        native_direct_time.as_nanos() as f64 / f64::from(iterations)
    );

    // Benchmark native intermediate
    let start = Instant::now();
    for _ in 0..iterations {
        result = native_intermediate(x, y);
    }
    let native_intermediate_time = start.elapsed();
    println!(
        "Native Intermediate:  {:?} ({:.2}ns per call) → {result}",
        native_intermediate_time,
        native_intermediate_time.as_nanos() as f64 / f64::from(iterations)
    );

    // Benchmark native function pointer
    let start = Instant::now();
    for _ in 0..iterations {
        result = native_function_ptr(x, y, native_direct);
    }
    let native_fn_ptr_time = start.elapsed();
    println!(
        "Native Function Ptr:  {:?} ({:.2}ns per call) → {result}",
        native_fn_ptr_time,
        native_fn_ptr_time.as_nanos() as f64 / f64::from(iterations)
    );

    println!();

    // ============================================================================
    // ZERO OVERHEAD CORE COMPARISON
    // ============================================================================

    println!("⚡ ZERO OVERHEAD CORE");
    println!("====================");

    // Test our zero-overhead implementations
    // Note: DirectComputeContext and SmartContext removed - using Enhanced Scoped System instead
    // TODO: Migrate to Enhanced Scoped System
    println!("DirectComputeContext and SmartContext removed - demo needs migration to Enhanced Scoped System");
    return Ok(());

    // Benchmark direct compute context
    let start = Instant::now();
    for _ in 0..iterations {
        // Simulate the complex expression using direct operations
        let sin_x = x.sin();
        let cos_y = y.cos();
        let cos_y_times_two = direct_ctx.mul_direct(cos_y, 2.0);
        let temp = direct_ctx.add_direct(sin_x, cos_y_times_two);
        result = direct_ctx.add_direct(temp, 1.0);
    }
    let direct_compute_time = start.elapsed();
    println!(
        "DirectComputeContext: {:?} ({:.2}ns per call) → {result}",
        direct_compute_time,
        direct_compute_time.as_nanos() as f64 / f64::from(iterations)
    );

    // Benchmark smart context
    let start = Instant::now();
    for _ in 0..iterations {
        let sin_x = x.sin();
        let cos_y = y.cos();
        let cos_y_times_two = smart_ctx.mul_smart(cos_y, 2.0);
        let temp = smart_ctx.add_smart(sin_x, cos_y_times_two);
        result = smart_ctx.add_smart(temp, 1.0);
    }
    let smart_context_time = start.elapsed();
    println!(
        "SmartContext:         {:?} ({:.2}ns per call) → {result}",
        smart_context_time,
        smart_context_time.as_nanos() as f64 / f64::from(iterations)
    );

    println!();

    // ============================================================================
    // UNIFIED ARCHITECTURE COMPARISON
    // ============================================================================

    println!("🚀 UNIFIED ARCHITECTURE");
    println!("=======================");

    // Build the expression once
    let ctx = DynamicContext::new();
    let x_var = ctx.var();
    let y_var = ctx.var();
    let sin_x = x_var.sin();
    let cos_y = y_var.cos();
    let two = ctx.constant(2.0);
    let one = ctx.constant(1.0);
    let cos_y_times_two = &cos_y * &two;
    let complex_expr: dslcompile::ast::TypedBuilderExpr<f64> = &sin_x + &cos_y_times_two + &one;
    let ast_expr: ASTRepr<f64> = complex_expr.into();

    let test_values = [x, y];

    // Test all strategies
    let strategies = [
        ("StaticCodegen", OptimizationConfig::zero_overhead()),
        ("DynamicCodegen", OptimizationConfig::dynamic_performance()),
        ("Interpretation", OptimizationConfig::dynamic_flexible()),
        ("Adaptive", OptimizationConfig::adaptive()),
    ];

    for (name, config) in strategies {
        let mut optimizer = SymbolicOptimizer::with_config(config).unwrap();
        let optimized = optimizer.optimize(&ast_expr).unwrap();

        let start = Instant::now();
        for _ in 0..iterations {
            result = optimized.eval_with_vars(&test_values);
        }
        let strategy_time = start.elapsed();
        println!(
            "{:<17} {:?} ({:.2}ns per call) → {result}",
            format!("{}:", name),
            strategy_time,
            strategy_time.as_nanos() as f64 / f64::from(iterations)
        );
    }

    println!();

    // ============================================================================
    // SIMPLE ARITHMETIC COMPARISON
    // ============================================================================

    println!("🔢 SIMPLE ARITHMETIC (x + y)");
    println!("=============================");

    let simple_iterations = 10_000_000; // More iterations for simple ops

    // Native addition
    let start = Instant::now();
    for _ in 0..simple_iterations {
        result = x + y;
    }
    let native_add_time = start.elapsed();
    println!(
        "Native Addition:      {:?} ({:.2}ns per call)",
        native_add_time,
        native_add_time.as_nanos() as f64 / f64::from(simple_iterations)
    );

    // Zero overhead addition
    let start = Instant::now();
    for _ in 0..simple_iterations {
        result = direct_ctx.add_direct(x, y);
    }
    let zero_add_time = start.elapsed();
    println!(
        "Zero Overhead Add:    {:?} ({:.2}ns per call)",
        zero_add_time,
        zero_add_time.as_nanos() as f64 / f64::from(simple_iterations)
    );

    // AST-based addition
    let simple_ast = ASTRepr::Add(
        Box::new(ASTRepr::Variable(0)),
        Box::new(ASTRepr::Variable(1)),
    );
    let start = Instant::now();
    for _ in 0..simple_iterations {
        result = simple_ast.eval_with_vars(&test_values);
    }
    let ast_add_time = start.elapsed();
    println!(
        "AST Addition:         {:?} ({:.2}ns per call)",
        ast_add_time,
        ast_add_time.as_nanos() as f64 / f64::from(simple_iterations)
    );

    println!();

    // ============================================================================
    // OVERHEAD ANALYSIS
    // ============================================================================

    println!("📊 OVERHEAD ANALYSIS");
    println!("===================");

    let native_baseline = native_direct_time.as_nanos() as f64 / f64::from(iterations);

    println!("Overhead vs Native Direct:");
    println!(
        "  Zero Overhead Core:   {:.1}x",
        (direct_compute_time.as_nanos() as f64 / f64::from(iterations)) / native_baseline
    );
    println!(
        "  Smart Context:        {:.1}x",
        (smart_context_time.as_nanos() as f64 / f64::from(iterations)) / native_baseline
    );

    // Calculate overhead for unified strategies
    let mut optimizer =
        SymbolicOptimizer::with_config(OptimizationConfig::zero_overhead()).unwrap();
    let optimized = optimizer.optimize(&ast_expr).unwrap();

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = optimized.eval_with_vars(&test_values);
    }
    let unified_time = start.elapsed();

    println!(
        "  Unified StaticCodegen: {:.1}x",
        (unified_time.as_nanos() as f64 / f64::from(iterations)) / native_baseline
    );

    println!();

    // ============================================================================
    // CONSTANT FOLDING DEMONSTRATION
    // ============================================================================

    println!("🎯 CONSTANT FOLDING POWER");
    println!("=========================");

    // Create a complex constant expression
    let const_expr = ASTRepr::Add(
        Box::new(ASTRepr::Mul(
            Box::new(ASTRepr::Constant(3.0)),
            Box::new(ASTRepr::Constant(4.0)),
        )),
        Box::new(ASTRepr::Add(
            Box::new(ASTRepr::Constant(5.0)),
            Box::new(ASTRepr::Mul(
                Box::new(ASTRepr::Constant(2.0)),
                Box::new(ASTRepr::Constant(6.0)),
            )),
        )),
    );

    println!("Complex constant: (3*4) + (5 + (2*6)) = 29");
    println!("Original AST: {const_expr:?}");

    // Zero-overhead should fold this completely
    let mut zero_optimizer =
        SymbolicOptimizer::with_config(OptimizationConfig::zero_overhead()).unwrap();
    let folded = zero_optimizer.optimize(&const_expr).unwrap();

    println!("Folded AST: {folded:?}");

    match folded {
        ASTRepr::Constant(val) => {
            println!("✅ SUCCESS: Completely folded to {val}");

            // Benchmark the folded constant vs native
            let start = Instant::now();
            for _ in 0..simple_iterations {
                result = folded.eval_with_vars(&[]);
            }
            let folded_time = start.elapsed();

            let start = Instant::now();
            for _ in 0..simple_iterations {
                result = 29.0; // Native constant
            }
            let native_const_time = start.elapsed();

            println!(
                "Folded constant eval: {:?} ({:.2}ns per call)",
                folded_time,
                folded_time.as_nanos() as f64 / f64::from(simple_iterations)
            );
            println!(
                "Native constant:      {:?} ({:.2}ns per call)",
                native_const_time,
                native_const_time.as_nanos() as f64 / f64::from(simple_iterations)
            );
            println!(
                "Overhead: {:.1}x",
                (folded_time.as_nanos() as f64) / (native_const_time.as_nanos() as f64)
            );
        }
        _ => {
            println!("❌ Not fully folded");
        }
    }

    println!();

    // ============================================================================
    // SUMMARY
    // ============================================================================

    println!("🏆 PERFORMANCE SUMMARY");
    println!("=====================");
    println!("For complex expressions (sin(x) + cos(y) * 2.0 + 1.0):");
    println!("  Native Rust:          {native_baseline:.1}ns (baseline)");
    println!(
        "  Zero Overhead Core:   {:.1}ns ({:.1}x overhead)",
        direct_compute_time.as_nanos() as f64 / f64::from(iterations),
        (direct_compute_time.as_nanos() as f64 / f64::from(iterations)) / native_baseline
    );
    println!(
        "  Unified Architecture: {:.1}ns ({:.1}x overhead)",
        unified_time.as_nanos() as f64 / f64::from(iterations),
        (unified_time.as_nanos() as f64 / f64::from(iterations)) / native_baseline
    );

    println!();
    println!("🎯 KEY INSIGHTS:");
    println!("• Zero-overhead approaches native Rust performance");
    println!("• Unified architecture adds minimal overhead (~1-2x)");
    println!("• Constant folding achieves true zero-overhead for constants");
    println!("• Expression building cost is amortized over many evaluations");
    println!("• DSL provides flexibility with acceptable performance cost");
}
