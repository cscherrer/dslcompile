//! Comprehensive benchmarks comparing expression optimization and compilation performance
//!
//! This benchmark suite demonstrates the performance benefits of:
//! 1. Symbolic optimization (egglog-style algebraic simplification)
//! 2. Different compilation strategies for various expression complexities

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use dslcompile::backends::cranelift::CraneliftCompiler;
use dslcompile::final_tagless::{ASTMathExpr, DirectEval, VariableRegistry};
use dslcompile::prelude::*;
use dslcompile::{OptimizationConfig, SymbolicOptimizer};

/// Complex mathematical expression for benchmarking (using new unified system)
fn create_complex_expression() -> ASTRepr<f64> {
    // Complex expression: sin(x^2 + ln(exp(y))) * cos(sqrt(x + y)) + exp(ln(x * y)) - (x + 0) * 1
    // This expression contains many optimization opportunities:
    // - ln(exp(y)) = y
    // - exp(ln(x * y)) = x * y
    // - (x + 0) * 1 = x
    // - sqrt can be optimized in some cases

    let math = MathBuilder::new();
    let x = math.var();
    let y = math.var();

    let x_squared_plus_ln_exp_y = &x.clone().pow(math.constant(2.0)) + &y.clone().exp().ln();
    let sqrt_x_plus_y = (&x + &y).sqrt();
    let sin_cos_part = x_squared_plus_ln_exp_y.sin() * sqrt_x_plus_y.cos();
    let exp_ln_xy = (&x * &y).ln().exp();
    let x_plus_zero_times_one = (&x + math.constant(0.0)) * math.constant(1.0);

    let result: TypedBuilderExpr<f64> = sin_cos_part + exp_ln_xy - x_plus_zero_times_one;
    result.into_ast()
}

/// Medium complexity expression (using new unified system)
fn create_medium_expression() -> ASTRepr<f64> {
    // Medium expression: x^3 + 2*x^2 + ln(exp(x)) + (y + 0) * 1
    let math = MathBuilder::new();
    let x = math.var();
    let y = math.var();

    let x_cubed = x.clone().pow(math.constant(3.0));
    let two_x_squared = math.constant(2.0) * x.clone().pow(math.constant(2.0));
    let ln_exp_x = x.clone().exp().ln();
    let y_plus_zero_times_one = (&y + math.constant(0.0)) * math.constant(1.0);

    let result: TypedBuilderExpr<f64> = x_cubed + two_x_squared + ln_exp_x + y_plus_zero_times_one;
    result.into_ast()
}

/// Simple expression for baseline comparison (using new unified system)
fn create_simple_expression() -> ASTRepr<f64> {
    // Simple expression: x + y + 1
    let math = MathBuilder::new();
    let x = math.var();
    let y = math.var();

    let result: TypedBuilderExpr<f64> = &x + &y + math.constant(1.0);
    result.into_ast()
}

/// Benchmark direct evaluation (no compilation)
fn bench_direct_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("direct_evaluation");

    let simple_expr = create_simple_expression();
    let medium_expr = create_medium_expression();
    let complex_expr = create_complex_expression();

    // Test values
    let x = 2.5;
    let y = 1.8;

    group.bench_function("simple", |b| {
        b.iter(|| DirectEval::eval_two_vars(black_box(&simple_expr), black_box(x), black_box(y)));
    });

    group.bench_function("medium", |b| {
        b.iter(|| DirectEval::eval_two_vars(black_box(&medium_expr), black_box(x), black_box(y)));
    });

    group.bench_function("complex", |b| {
        b.iter(|| DirectEval::eval_two_vars(black_box(&complex_expr), black_box(x), black_box(y)));
    });

    group.finish();
}

/// Benchmark optimization effects
fn bench_optimization_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_comparison");

    let complex_expr = create_complex_expression();

    // Create optimizers
    let mut basic_optimizer = SymbolicOptimizer::new().unwrap();

    let mut config = OptimizationConfig::default();
    // Use aggressive mode for better optimization but avoid expensive egglog
    config.aggressive = true;
    config.constant_folding = true;
    config.cse = true;
    // Leave egglog_optimization = false (default) for performance
    let mut advanced_optimizer = SymbolicOptimizer::with_config(config).unwrap();

    // Optimize expressions
    let basic_optimized = basic_optimizer.optimize(&complex_expr).unwrap();
    let advanced_optimized = advanced_optimizer.optimize(&complex_expr).unwrap();

    println!(
        "Original expression operations: {}",
        complex_expr.count_operations()
    );
    println!(
        "Basic optimized operations: {}",
        basic_optimized.count_operations()
    );
    println!(
        "Advanced optimized operations: {}",
        advanced_optimized.count_operations()
    );

    let x = 2.5;
    let y = 1.8;

    group.bench_function("original", |b| {
        b.iter(|| DirectEval::eval_two_vars(black_box(&complex_expr), black_box(x), black_box(y)));
    });

    group.bench_function("basic_optimized", |b| {
        b.iter(|| {
            DirectEval::eval_two_vars(black_box(&basic_optimized), black_box(x), black_box(y))
        });
    });

    group.bench_function("advanced_optimized", |b| {
        b.iter(|| {
            DirectEval::eval_two_vars(black_box(&advanced_optimized), black_box(x), black_box(y))
        });
    });

    group.finish();
}

/// Benchmark JIT compilation vs direct evaluation
fn bench_compilation_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("compilation_strategies");
    group.sample_size(50); // Reduce sample size for compilation benchmarks

    let complex_expr = create_complex_expression();

    // Optimize the expression first
    let mut config = OptimizationConfig::default();
    config.egglog_optimization = true;
    config.constant_folding = true;
    let mut optimizer = SymbolicOptimizer::with_config(config).unwrap();
    let optimized_expr = optimizer.optimize(&complex_expr).unwrap();

    println!("Optimized expression: {optimized_expr:?}");

    // Test values
    let x = 2.5;
    let y = 1.8;

    // Benchmark direct evaluation of optimized expression
    group.bench_function("direct_optimized", |b| {
        b.iter(|| {
            DirectEval::eval_two_vars(black_box(&optimized_expr), black_box(x), black_box(y))
        });
    });

    // Benchmark Cranelift JIT compilation
    let jit_compiler = CraneliftCompiler::new_default().unwrap();
    let registry = VariableRegistry::for_expression(&optimized_expr);
    let compiled_func = jit_compiler
        .compile_expression(&optimized_expr, &registry)
        .unwrap();

    group.bench_function("cranelift_jit", |b| {
        b.iter(|| compiled_func.call(&[black_box(x), black_box(y)]));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_direct_evaluation,
    bench_optimization_comparison,
    bench_compilation_strategies
);

criterion_main!(benches);
