//! Comprehensive benchmarks comparing expression optimization and compilation performance
//!
//! This benchmark suite demonstrates the performance benefits of:
//! 1. Symbolic optimization (egglog-style algebraic simplification)
//! 2. Different compilation strategies for various expression complexities

use criterion::{Criterion, criterion_group, criterion_main};
use dslcompile::ast::{ASTRepr, DynamicContext, TypedBuilderExpr, VariableRegistry};

use dslcompile::{OptimizationConfig, SymbolicOptimizer};
use std::hint::black_box;

/// Complex mathematical expression for benchmarking (using new unified system)
fn create_complex_expression() -> ASTRepr<f64> {
    // Complex expression: sin(x^2 + ln(exp(y))) * cos(sqrt(x + y)) + exp(ln(x * y)) - (x + 0) * 1
    // This expression contains many optimization opportunities:
    // - ln(exp(y)) = y
    // - exp(ln(x * y)) = x * y
    // - (x + 0) * 1 = x
    // - sqrt can be optimized in some cases

    let math = DynamicContext::new();
    let x = math.var();
    let y = math.var();

    let x_squared_plus_ln_exp_y = &x.clone().pow(math.constant(2.0)) + &y.clone().exp().ln();
    let sqrt_x_plus_y = (&x + &y).sqrt();
    let sin_cos_part = x_squared_plus_ln_exp_y.sin() * sqrt_x_plus_y.cos();
    let exp_ln_xy = (&x * &y).ln().exp();
    let x_plus_zero_times_one = (&x + math.constant(0.0)) * math.constant(1.0);

    let result: TypedBuilderExpr<f64> = sin_cos_part + exp_ln_xy - x_plus_zero_times_one;
    result.into()
}

/// Medium complexity expression (using new unified system)
fn create_medium_expression() -> ASTRepr<f64> {
    // Medium expression: x^3 + 2*x^2 + ln(exp(x)) + (y + 0) * 1
    let math = DynamicContext::new();
    let x = math.var();
    let y = math.var();

    let x_cubed = x.clone().pow(math.constant(3.0));
    let two_x_squared = math.constant(2.0) * x.clone().pow(math.constant(2.0));
    let ln_exp_x = x.clone().exp().ln();
    let y_plus_zero_times_one = (&y + math.constant(0.0)) * math.constant(1.0);

    let result: TypedBuilderExpr<f64> = x_cubed + two_x_squared + ln_exp_x + y_plus_zero_times_one;
    result.into()
}

/// Simple expression for baseline comparison (using new unified system)
fn create_simple_expression() -> ASTRepr<f64> {
    // Simple expression: x + y + 1
    let math = DynamicContext::new();
    let x = math.var();
    let y = math.var();

    let result: TypedBuilderExpr<f64> = &x + &y + math.constant(1.0);
    result.into()
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
        b.iter(|| simple_expr.eval_two_vars(x, y));
    });

    group.bench_function("medium", |b| {
        b.iter(|| medium_expr.eval_two_vars(x, y));
    });

    group.bench_function("complex", |b| {
        b.iter(|| complex_expr.eval_two_vars(x, y));
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
        b.iter(|| complex_expr.eval_two_vars(x, y));
    });

    group.bench_function("basic_optimized", |b| {
        b.iter(|| basic_optimized.eval_two_vars(x, y));
    });

    group.bench_function("advanced_optimized", |b| {
        b.iter(|| advanced_optimized.eval_two_vars(x, y));
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
        b.iter(|| optimized_expr.eval_two_vars(x, y));
    });

    // Note: Cranelift JIT compilation removed - focusing on compile-time optimization

    group.finish();
}

fn bench_basic_optimization(c: &mut Criterion) {
    c.bench_function("basic_optimization", |b| {
        b.iter(|| {
            // Create a simple expression that can be optimized
            let math = DynamicContext::new();
            let x = math.var();
            let expr = &x + 0.0; // x + 0 should optimize to x
            let ast = expr.into();

            // Optimize it
            let mut optimizer = SymbolicOptimizer::new().unwrap();
            let optimized = optimizer.optimize(&ast).unwrap();

            black_box(optimized);
        });
    });
}

fn bench_complex_optimization(c: &mut Criterion) {
    c.bench_function("complex_optimization", |b| {
        b.iter(|| {
            // Create a more complex expression
            let math = DynamicContext::new();
            let x = math.var();
            let expr = (&x + 0.0) * 1.0 + (&x * 0.0); // (x + 0) * 1 + (x * 0) should optimize to x
            let ast = expr.into();

            let mut optimizer = SymbolicOptimizer::new().unwrap();
            let optimized = optimizer.optimize(&ast).unwrap();

            black_box(optimized);
        });
    });
}

fn bench_transcendental_optimization(c: &mut Criterion) {
    c.bench_function("transcendental_optimization", |b| {
        b.iter(|| {
            // Test optimization of transcendental functions
            let math = DynamicContext::new();
            let x = math.var();
            let sin_x = x.clone().sin();
            let cos_x = x.clone().cos();
            let expr = sin_x.clone() * sin_x.clone() + cos_x.clone() * cos_x.clone(); // sin²(x) + cos²(x) should optimize to 1
            let ast = expr.into();

            let mut optimizer = SymbolicOptimizer::new().unwrap();
            let optimized = optimizer.optimize(&ast).unwrap();

            black_box(optimized);
        });
    });
}

criterion_group!(
    benches,
    bench_direct_evaluation,
    bench_optimization_comparison,
    bench_compilation_strategies,
    bench_basic_optimization,
    bench_complex_optimization,
    bench_transcendental_optimization
);

criterion_main!(benches);
