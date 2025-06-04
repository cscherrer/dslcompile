//! Simple benchmark comparing optimization performance

use criterion::{Criterion, criterion_group, criterion_main};
use dslcompile::OptimizationConfig;
use dslcompile::backends::cranelift::CraneliftCompiler;
use dslcompile::final_tagless::{DirectEval, VariableRegistry};
use dslcompile::prelude::*;
use dslcompile::symbolic::symbolic::SymbolicOptimizer;
use std::hint::black_box;

/// Complex mathematical expression for benchmarking
fn create_complex_expression() -> ASTRepr<f64> {
    // Complex expression: sin(x^2 + ln(exp(y))) * cos(sqrt(x + y)) + exp(ln(x * y)) - (x + 0) * 1
    // This expression contains many optimization opportunities:
    // - ln(exp(y)) = y
    // - exp(ln(x * y)) = x * y
    // - (x + 0) * 1 = x

    let math = MathBuilder::new();
    let x = math.var();
    let y = math.var();

    // Simple expression: 2x + y
    let _simple_expr = math.constant(2.0) * &x + &y;

    // Medium expression: xy + sin(x)
    let _medium_expr = &x * &y + x.clone().sin();

    // Complex expression: x * x² + exp(y)
    let result: TypedBuilderExpr<f64> = &x * x.clone().pow(math.constant(2.0)) + y.exp();
    result.into_ast()
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
    let original_result = DirectEval::eval_two_vars(&complex_expr, x, y);
    let optimized_result = DirectEval::eval_two_vars(&advanced_optimized, x, y);
    println!("\n✅ Correctness Check:");
    println!("Original result: {original_result}");
    println!("Optimized result: {optimized_result}");
    println!("Difference: {}", (original_result - optimized_result).abs());

    group.bench_function("original_expression", |b| {
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

    #[cfg(feature = "cranelift")]
    {
        // Benchmark Cranelift JIT compilation
        let mut jit_compiler = CraneliftCompiler::new_default().unwrap();
        let registry = VariableRegistry::for_expression(&advanced_optimized);
        let jit_func = jit_compiler
            .compile_expression(&advanced_optimized, &registry)
            .unwrap();

        group.bench_function("cranelift_jit", |b| {
            b.iter(|| jit_func.call(&[black_box(x), black_box(y)]));
        });

        // Benchmark pre-compiled JIT function (amortized cost)
        let mut jit_compiler = CraneliftCompiler::new_default().unwrap();
        let registry = VariableRegistry::for_expression(&advanced_optimized);
        let jit_func = jit_compiler
            .compile_expression(&advanced_optimized, &registry)
            .unwrap();

        group.bench_function("precompiled_jit", |b| {
            b.iter(|| jit_func.call(&[black_box(x), black_box(y)]));
        });

        println!("\n🔧 JIT Compilation Stats:");
        println!(
            "Compilation time: {} μs",
            jit_func.metadata().compilation_time_ms
        );
        println!(
            "Expression complexity: {} operations",
            jit_func.metadata().expression_complexity
        );
    }

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
        let _ = DirectEval::eval_two_vars(&complex_expr, x, y);
    }
    let original_duration = original_time.elapsed();

    let optimized_time = std::time::Instant::now();
    for _ in 0..10000 {
        let _ = DirectEval::eval_two_vars(&optimized, x, y);
    }
    let optimized_duration = optimized_time.elapsed();

    #[cfg(feature = "cranelift")]
    {
        // Test JIT performance
        let mut jit_compiler = CraneliftCompiler::new_default().unwrap();
        let registry = VariableRegistry::for_expression(&optimized);
        let jit_func = jit_compiler
            .compile_expression(&optimized, &registry)
            .unwrap();

        let jit_time = std::time::Instant::now();
        for _ in 0..10000 {
            let _ = jit_func.call(&[black_box(x), black_box(y)]);
        }
        let jit_duration = jit_time.elapsed();

        let speedup_opt =
            original_duration.as_nanos() as f64 / optimized_duration.as_nanos() as f64;
        let speedup_jit = original_duration.as_nanos() as f64 / jit_duration.as_nanos() as f64;
        let jit_vs_opt = optimized_duration.as_nanos() as f64 / jit_duration.as_nanos() as f64;

        println!("\n📈 Performance Analysis:");
        println!("Original (10k evals): {original_duration:?}");
        println!("Optimized (10k evals): {optimized_duration:?}");
        println!("JIT (10k evals): {jit_duration:?}");
        println!("Optimization speedup: {speedup_opt:.2}x");
        println!("JIT speedup: {speedup_jit:.2}x");
        println!("JIT vs Optimized: {jit_vs_opt:.2}x");
    }

    #[cfg(not(feature = "cranelift"))]
    {
        let speedup_opt =
            original_duration.as_nanos() as f64 / optimized_duration.as_nanos() as f64;

        println!("\n📈 Performance Analysis:");
        println!("Original (10k evals): {original_duration:?}");
        println!("Optimized (10k evals): {optimized_duration:?}");
        println!("Optimization speedup: {speedup_opt:.2}x");
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_optimization_performance,
    bench_optimization_tradeoff
);

criterion_main!(benches);
