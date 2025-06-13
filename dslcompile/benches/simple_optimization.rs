//! Simple benchmark comparing optimization performance

use criterion::{Criterion, criterion_group, criterion_main};
use dslcompile::OptimizationConfig;

use dslcompile::{prelude::*, symbolic::symbolic::SymbolicOptimizer};
use std::hint::black_box;

/// Complex mathematical expression for benchmarking
fn create_complex_expression() -> ASTRepr<f64> {
    // Complex expression: sin(x^2 + ln(exp(y))) * cos(sqrt(x + y)) + exp(ln(x * y)) - (x + 0) * 1
    // This expression contains many optimization opportunities:
    // - ln(exp(y)) = y
    // - exp(ln(x * y)) = x * y
    // - (x + 0) * 1 = x

    let mut math = DynamicContext::new();
    let x = math.var();
    let y = math.var();

    // Simple expression: 2x + y
    let _simple_expr = math.constant(2.0) * &x + &y;

    // Medium expression: xy + sin(x)
    let _medium_expr = &x * &y + x.clone().sin();

    // Complex expression: x * x² + exp(y)
    let result: TypedBuilderExpr<f64> = &x * x.clone().pow(math.constant(2.0)) + y.exp();
    result.into()
}

/// Benchmark optimization effects
fn bench_optimization_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_performance");

    let complex_expr = create_complex_expression();

    // Create optimizers
    let mut basic_optimizer = SymbolicOptimizer::new().unwrap();

    let mut config = OptimizationConfig::default();
    config.egglog_optimization = true;
    config.constant_folding = true;
    config.aggressive = true;
    let mut advanced_optimizer = SymbolicOptimizer::with_config(config).unwrap();

    // Optimize expressions
    let basic_optimized = basic_optimizer.optimize(&complex_expr).unwrap();
    let advanced_optimized = advanced_optimizer.optimize(&complex_expr).unwrap();

    println!("\n🔍 Expression Analysis:");
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

    println!("\n📊 Optimization Results:");
    println!("Original: {complex_expr:?}");
    println!("Advanced: {advanced_optimized:?}");

    let x = 2.5;
    let y = 1.8;

    // Verify correctness
    let original_result = complex_expr.eval_two_vars(x, y);
    let optimized_result = advanced_optimized.eval_two_vars(x, y);
    println!("\n✅ Correctness Check:");
    println!("Original result: {original_result}");
    println!("Optimized result: {optimized_result}");
    println!("Difference: {}", (original_result - optimized_result).abs());

    // Direct evaluation (baseline)
    group.bench_function("direct", |b| {
        b.iter(|| complex_expr.eval_two_vars(black_box(x), black_box(y)));
    });
    group.bench_function("direct_simple", |b| {
        b.iter(|| basic_optimized.eval_two_vars(black_box(x), black_box(y)));
    });

    // Note: Cranelift evaluation removed - focusing on compile-time optimization

    group.finish();
}

/// Benchmark optimization time vs execution time tradeoff
fn bench_optimization_tradeoff(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_tradeoff");

    let complex_expr = create_complex_expression();

    // Benchmark optimization time
    group.bench_function("optimization_time", |b| {
        b.iter(|| {
            let mut config = OptimizationConfig::default();
            config.egglog_optimization = true;
            config.constant_folding = true;
            let mut optimizer = SymbolicOptimizer::with_config(config).unwrap();
            optimizer.optimize(black_box(&complex_expr)).unwrap()
        });
    });

    // Pre-optimize for execution benchmarks
    let mut config = OptimizationConfig::default();
    config.egglog_optimization = true;
    config.constant_folding = true;
    let mut optimizer = SymbolicOptimizer::with_config(config).unwrap();
    let optimized = optimizer.optimize(&complex_expr).unwrap();

    let x = 2.5;
    let y = 1.8;

    // Calculate speedup
    let original_time = std::time::Instant::now();
    for _ in 0..10000 {
        let _ = complex_expr.eval_two_vars(x, y);
    }
    let original_duration = original_time.elapsed();

    let optimized_time = std::time::Instant::now();
    for _ in 0..10000 {
        let _ = optimized.eval_two_vars(x, y);
    }
    let optimized_duration = optimized_time.elapsed();

    let speedup_opt = original_duration.as_nanos() as f64 / optimized_duration.as_nanos() as f64;

    println!("\n📈 Performance Analysis:");
    println!("Original (10k evals): {original_duration:?}");
    println!("Optimized (10k evals): {optimized_duration:?}");
    println!("Optimization speedup: {speedup_opt:.2}x");

    group.finish();
}

criterion_group!(
    benches,
    bench_optimization_performance,
    bench_optimization_tradeoff
);

criterion_main!(benches);
