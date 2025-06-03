//! README Demo: Comprehensive DSLCompile Example
//!
//! This example demonstrates:
//! 1. Simple Divan macro usage for benchmarking
//! 2. Compile-time optimization with pretty-printed expressions
//! 3. Runtime-generated expressions with optimization
//! 4. Performance benchmarking using divan
//! 5. Execution time comparison before and after optimization

use dslcompile::prelude::*;
use std::time::Instant;

/// Complex mathematical expression that benefits from optimization
fn create_optimizable_expression(math: &MathBuilder) -> TypedBuilderExpr<f64> {
    let x = math.var();
    let y = math.var();
    
    // Expression with many optimization opportunities:
    // ln(exp(x)) + (y + 0) * 1 + sin(x)^2 + cos(x)^2 + exp(ln(y)) - 0
    // Should optimize to: x + y + 1 + y = x + 2y + 1
    let ln_exp_x = x.clone().exp().ln();  // ln(exp(x)) = x
    let y_plus_zero = &y + math.constant(0.0);  // y + 0 = y
    let times_one = y_plus_zero * math.constant(1.0);  // (y + 0) * 1 = y
    let sin_x = x.clone().sin();
    let cos_x = x.clone().cos();
    let sin_squared = &sin_x * &sin_x;  // sin(x) * sin(x) instead of sin(x)^2
    let cos_squared = &cos_x * &cos_x;  // cos(x) * cos(x) instead of cos(x)^2
    let trig_identity = sin_squared + cos_squared;  // sin²(x) + cos²(x) = 1
    let exp_ln_y = y.clone().ln().exp();  // exp(ln(y)) = y
    let minus_zero = exp_ln_y - math.constant(0.0);  // y - 0 = y
    
    ln_exp_x + times_one + trig_identity + minus_zero
}

/// Simple polynomial for compile-time optimization demo
fn create_polynomial(math: &MathBuilder) -> TypedBuilderExpr<f64> {
    let x = math.var();
    // (x + 1)² = x² + 2x + 1
    let x_plus_one: TypedBuilderExpr<f64> = &x + math.constant(1.0);
    x_plus_one.pow(math.constant(2.0))
}

// To run benchmarks: cargo run --example readme_demo -- --bench
// To run demo: cargo run --example readme_demo
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() > 1 && args[1] == "--bench" {
        divan::main();
        Ok(())
    } else {
        demo_main()
    }
}

fn demo_main() -> Result<()> {
    println!("🚀 DSLCompile README Demo");
    println!("========================");

    // Create optimizer with trigonometric rules enabled
    let config = OptimizationConfig {
        max_iterations: 10,
        aggressive: false,
        constant_folding: true,
        cse: true,
        egglog_optimization: true,  // Enable egglog optimization
        enable_expansion_rules: true,
        enable_distribution_rules: true,
    };
    let mut optimizer = SymbolicOptimizer::with_config(config)?;

    // ========================================================================
    // 1. COMPILE-TIME OPTIMIZATION DEMO
    // ========================================================================
    
    println!("📋 1. COMPILE-TIME OPTIMIZATION");
    println!("--------------------------------");
    
    let math = MathBuilder::new();
    let poly_expr = create_polynomial(&math);
    
    // Convert to AST for optimization
    let poly_ast = poly_expr.into_ast();
    
    println!("🔍 Original Expression:");
    println!("   Mathematical: (x + 1)²");
    println!("   AST: {}", format_ast_pretty(&poly_ast));
    println!("   Operations: {}", poly_ast.count_operations());
    
    // Optimize the expression
    let optimization_start = Instant::now();
    let optimized_poly = optimizer.optimize(&poly_ast)?;
    let optimization_time = optimization_start.elapsed();
    
    println!("\n✨ Optimized Expression:");
    println!("   Mathematical: x² + 2x + 1");
    println!("   AST: {}", format_ast_pretty(&optimized_poly));
    println!("   Operations: {}", optimized_poly.count_operations());
    println!("   Optimization time: {:.2}μs", optimization_time.as_micros());
    
    // Measure execution performance
    let test_values = [1.0, 2.0, 3.0, 5.0, 10.0];
    
    println!("\n⚡ Execution Performance:");
    for &x_val in &test_values {
        let original_start = Instant::now();
        let original_result = DirectEval::eval_with_vars(&poly_ast, &[x_val]);
        let original_time = original_start.elapsed();
        
        let optimized_start = Instant::now();
        let optimized_result = DirectEval::eval_with_vars(&optimized_poly, &[x_val]);
        let optimized_time = optimized_start.elapsed();
        
        let speedup = original_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;
        
        println!("   x={}: {:.1} (orig: {}ns, opt: {}ns, speedup: {:.1}x)", 
                 x_val, original_result, original_time.as_nanos(), 
                 optimized_time.as_nanos(), speedup);
        
        // Verify correctness
        assert!((original_result - optimized_result).abs() < 1e-10);
    }

    // ========================================================================
    // 2. RUNTIME-GENERATED OPTIMIZATION DEMO
    // ========================================================================
    
    println!("\n📋 2. RUNTIME-GENERATED OPTIMIZATION");
    println!("------------------------------------");
    
    let complex_expr = create_optimizable_expression(&math);
    let complex_ast = complex_expr.into_ast();
    
    println!("🔍 Original Complex Expression:");
    println!("   Mathematical: ln(exp(x)) + (y + 0) * 1 + sin²(x) + cos²(x) + exp(ln(y)) - 0");
    println!("   AST: {}", format_ast_pretty(&complex_ast));
    println!("   Operations: {}", complex_ast.count_operations());
    
    // Optimize the complex expression
    let complex_optimization_start = Instant::now();
    let optimized_complex = optimizer.optimize(&complex_ast)?;
    let complex_optimization_time = complex_optimization_start.elapsed();
    
    println!("\n✨ Optimized Complex Expression:");
    println!("   Mathematical: x + y + 1 + y = x + 2y + 1");
    println!("   AST: {}", format_ast_pretty(&optimized_complex));
    println!("   Operations: {}", optimized_complex.count_operations());
    println!("   Optimization time: {:.2}μs", complex_optimization_time.as_micros());
    
    // Performance comparison for complex expression
    println!("\n⚡ Complex Expression Performance:");
    let test_pairs = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)];
    
    for &(x_val, y_val) in &test_pairs {
        let original_start = Instant::now();
        let original_result = DirectEval::eval_with_vars(&complex_ast, &[x_val, y_val]);
        let original_time = original_start.elapsed();
        
        let optimized_start = Instant::now();
        let optimized_result = DirectEval::eval_with_vars(&optimized_complex, &[x_val, y_val]);
        let optimized_time = optimized_start.elapsed();
        
        let speedup = original_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;
        
        println!("   x={}, y={}: {:.1} (orig: {}ns, opt: {}ns, speedup: {:.1}x)", 
                 x_val, y_val, original_result, original_time.as_nanos(), 
                 optimized_time.as_nanos(), speedup);
        
        // Verify correctness
        assert!((original_result - optimized_result).abs() < 1e-10);
    }

    // ========================================================================
    // 3. BULK PERFORMANCE ANALYSIS
    // ========================================================================
    
    println!("\n📋 3. BULK PERFORMANCE ANALYSIS");
    println!("-------------------------------");
    
    const ITERATIONS: usize = 100_000;
    let x_val = 2.5;
    let y_val = 3.7;
    
    // Benchmark original expression
    let bulk_original_start = Instant::now();
    let mut original_sum = 0.0;
    for _ in 0..ITERATIONS {
        original_sum += DirectEval::eval_with_vars(&complex_ast, &[x_val, y_val]);
    }
    let bulk_original_time = bulk_original_start.elapsed();
    
    // Benchmark optimized expression
    let bulk_optimized_start = Instant::now();
    let mut optimized_sum = 0.0;
    for _ in 0..ITERATIONS {
        optimized_sum += DirectEval::eval_with_vars(&optimized_complex, &[x_val, y_val]);
    }
    let bulk_optimized_time = bulk_optimized_start.elapsed();
    
    let bulk_speedup = bulk_original_time.as_nanos() as f64 / bulk_optimized_time.as_nanos() as f64;
    
    println!("🏃 Bulk Performance ({} iterations):", ITERATIONS);
    println!("   Original: {:.2}ms ({:.1}ns/eval)", 
             bulk_original_time.as_millis(), 
             bulk_original_time.as_nanos() as f64 / ITERATIONS as f64);
    println!("   Optimized: {:.2}ms ({:.1}ns/eval)", 
             bulk_optimized_time.as_millis(),
             bulk_optimized_time.as_nanos() as f64 / ITERATIONS as f64);
    println!("   Speedup: {:.1}x", bulk_speedup);
    println!("   Correctness: {} (diff: {:.2e})", 
             (original_sum - optimized_sum).abs() < 1e-6,
             (original_sum - optimized_sum).abs());

    // ========================================================================
    // 4. CODE GENERATION & COMPILATION TIMING
    // ========================================================================
    
    println!("\n📋 4. CODE GENERATION & COMPILATION TIMING");
    println!("------------------------------------------");
    
    // Time code generation (one-time cost)
    let codegen_start = Instant::now();
    let codegen = RustCodeGenerator::new();
    let rust_code = codegen.generate_function(&optimized_complex, "optimized_func")?;
    let codegen_time = codegen_start.elapsed();
    
    println!("🦀 Generated Rust Code:");
    println!("{}", rust_code);
    println!("   Code generation time: {:.2}μs", codegen_time.as_micros());
    
    // Time compilation (one-time cost)
    let compilation_start = Instant::now();
    let compiler = RustCompiler::new();
    let compiled_func = compiler.compile_and_load(&rust_code, "optimized_func")?;
    let compilation_time = compilation_start.elapsed();
    
    println!("\n⚙️  Compilation Timing (One-time Cost):");
    println!("   Rust compilation: {:.2}ms", compilation_time.as_millis());
    println!("   Total setup cost: {:.2}ms", (codegen_time + compilation_time).as_millis());
    
    // Now benchmark execution performance of the compiled function
    println!("\n⚡ Compiled Function Performance:");
    let compiled_result = compiled_func.call_two_vars(x_val, y_val)?;
    let direct_result = DirectEval::eval_with_vars(&optimized_complex, &[x_val, y_val]);
    
    // Time multiple executions to get stable measurements
    const COMPILED_ITERATIONS: usize = 10_000;
    let compiled_exec_start = Instant::now();
    let mut compiled_sum = 0.0;
    for _ in 0..COMPILED_ITERATIONS {
        compiled_sum += compiled_func.call_two_vars(x_val, y_val)?;
    }
    let compiled_exec_time = compiled_exec_start.elapsed();
    
    let avg_compiled_time = compiled_exec_time.as_nanos() as f64 / COMPILED_ITERATIONS as f64;
    let avg_optimized_time = bulk_optimized_time.as_nanos() as f64 / ITERATIONS as f64;
    let compiled_speedup = avg_optimized_time / avg_compiled_time;
    
    println!("   Compiled execution: {:.1}ns/eval ({} iterations)", avg_compiled_time, COMPILED_ITERATIONS);
    println!("   vs Optimized eval: {:.1}ns/eval", avg_optimized_time);
    println!("   Compiled speedup: {:.1}x", compiled_speedup);
    
    println!("\n✅ Code Generation Verification:");
    println!("   Direct evaluation: {:.6}", direct_result);
    println!("   Compiled function: {:.6}", compiled_result);
    println!("   Match: {}", (compiled_result - direct_result).abs() < 1e-10);
    println!("   Correctness over {} iterations: {}", COMPILED_ITERATIONS, 
             (compiled_sum / COMPILED_ITERATIONS as f64 - direct_result).abs() < 1e-10);

    println!("\n🎉 Demo Complete!");
    println!("================");
    println!("Key takeaways:");
    println!("• Symbolic optimization reduces operation count significantly");
    println!("• Runtime performance improves with optimization");
    println!("• Compilation is a one-time cost ({:.2}ms) amortized over many evaluations", 
             (codegen_time + compilation_time).as_millis());
    println!("• Generated code maintains mathematical correctness");
    println!("• Both compile-time and runtime optimization are supported");
    
    Ok(())
}

/// Helper function to format AST expressions in a readable way
fn format_ast_pretty(ast: &ASTRepr<f64>) -> String {
    match ast {
        ASTRepr::Constant(c) => format!("{}", c),
        ASTRepr::Variable(i) => format!("x{}", i),
        ASTRepr::Add(l, r) => format!("({} + {})", format_ast_pretty(l), format_ast_pretty(r)),
        ASTRepr::Sub(l, r) => format!("({} - {})", format_ast_pretty(l), format_ast_pretty(r)),
        ASTRepr::Mul(l, r) => format!("({} * {})", format_ast_pretty(l), format_ast_pretty(r)),
        ASTRepr::Div(l, r) => format!("({} / {})", format_ast_pretty(l), format_ast_pretty(r)),
        ASTRepr::Pow(l, r) => format!("({} ^ {})", format_ast_pretty(l), format_ast_pretty(r)),
        ASTRepr::Neg(e) => format!("(-{})", format_ast_pretty(e)),
        ASTRepr::Ln(e) => format!("ln({})", format_ast_pretty(e)),
        ASTRepr::Exp(e) => format!("exp({})", format_ast_pretty(e)),
        ASTRepr::Sin(e) => format!("sin({})", format_ast_pretty(e)),
        ASTRepr::Cos(e) => format!("cos({})", format_ast_pretty(e)),
        ASTRepr::Sqrt(e) => format!("sqrt({})", format_ast_pretty(e)),
    }
}

// ============================================================================
// PRE-COMPUTED EXPRESSIONS FOR BENCHMARKS
// ============================================================================

// Pre-compute expressions at module load time to avoid overhead in benchmarks
fn create_benchmark_data() -> (ASTRepr<f64>, ASTRepr<f64>, ASTRepr<f64>, ASTRepr<f64>) {
    let math = MathBuilder::new();
    
    // Simple expressions
    let simple_expr = create_polynomial(&math).into_ast();
    
    // Create optimizer with trigonometric rules enabled
    let config = OptimizationConfig {
        max_iterations: 10,
        aggressive: false,
        constant_folding: true,
        cse: true,
        egglog_optimization: true,  // Enable egglog optimization
        enable_expansion_rules: true,
        enable_distribution_rules: true,
    };
    let mut optimizer = SymbolicOptimizer::with_config(config).unwrap();
    let simple_optimized = optimizer.optimize(&simple_expr).unwrap();
    
    // Complex expressions  
    let complex_expr = create_optimizable_expression(&math).into_ast();
    let complex_optimized = optimizer.optimize(&complex_expr).unwrap();
    
    (simple_expr, simple_optimized, complex_expr, complex_optimized)
}

// Lazy static initialization
static BENCHMARK_DATA: std::sync::LazyLock<(ASTRepr<f64>, ASTRepr<f64>, ASTRepr<f64>, ASTRepr<f64>)> = 
    std::sync::LazyLock::new(create_benchmark_data);

// ============================================================================
// SIMPLE DIVAN MACRO EXAMPLE - Just add #[divan::bench] above any function!
// ============================================================================

#[divan::bench]
fn simple_optimization_benchmark() {
    // Only benchmark the optimization call
    let (simple_expr, _, _, _) = &*BENCHMARK_DATA;
    let mut optimizer = SymbolicOptimizer::new().unwrap();
    optimizer.optimize(simple_expr).unwrap();
}

#[divan::bench]
fn simple_execution_benchmark() {
    // Only benchmark the evaluation call
    let (_, simple_optimized, _, _) = &*BENCHMARK_DATA;
    let _ = DirectEval::eval_with_vars(simple_optimized, &[2.5]);
}

// ============================================================================
// COMPREHENSIVE DIVAN BENCHMARKS
// ============================================================================

#[divan::bench]
fn optimization_time_simple() {
    let (simple_expr, _, _, _) = &*BENCHMARK_DATA;
    let mut optimizer = SymbolicOptimizer::new().unwrap();
    optimizer.optimize(simple_expr).unwrap();
}

#[divan::bench]
fn optimization_time_complex() {
    let (_, _, complex_expr, _) = &*BENCHMARK_DATA;
    let mut optimizer = SymbolicOptimizer::new().unwrap();
    optimizer.optimize(complex_expr).unwrap();
}

#[divan::bench]
fn execution_original_simple() {
    let (simple_expr, _, _, _) = &*BENCHMARK_DATA;
    let _ = DirectEval::eval_with_vars(simple_expr, &[2.5]);
}

#[divan::bench]
fn execution_optimized_simple() {
    let (_, simple_optimized, _, _) = &*BENCHMARK_DATA;
    let _ = DirectEval::eval_with_vars(simple_optimized, &[2.5]);
}

#[divan::bench]
fn execution_original_complex() {
    let (_, _, complex_expr, _) = &*BENCHMARK_DATA;
    let _ = DirectEval::eval_with_vars(complex_expr, &[2.5, 3.7]);
}

#[divan::bench]
fn execution_optimized_complex() {
    let (_, _, _, complex_optimized) = &*BENCHMARK_DATA;
    let _ = DirectEval::eval_with_vars(complex_optimized, &[2.5, 3.7]);
}

// Note: Compilation timing is not benchmarked here because:
// 1. Compilation is a one-time cost, not repeated
// 2. Subsequent builds are heavily cached and would give misleading results
// 3. Compilation timing is measured as one-off timing in the main() function

// ============================================================================
// EXPRESSION BUILDERS
// ============================================================================ 