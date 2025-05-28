//! Simple benchmark comparing optimization performance

use criterion::{black_box, criterion_group, criterion_main, Criterion};
#[cfg(feature = "cranelift")]
use mathcompile::backends::cranelift::JITCompiler;
use mathcompile::final_tagless::DirectEval;
use mathcompile::prelude::*;
use mathcompile::symbolic::{OptimizationConfig, SymbolicOptimizer};

/// Complex mathematical expression for benchmarking
fn create_complex_expression() -> ASTRepr<f64> {
    // Complex expression: sin(x^2 + ln(exp(y))) * cos(sqrt(x + y)) + exp(ln(x * y)) - (x + 0) * 1
    // This expression contains many optimization opportunities:
    // - ln(exp(y)) = y
    // - exp(ln(x * y)) = x * y
    // - (x + 0) * 1 = x

    let mut math = MathBuilder::new();
    let x = math.var("x");
    let y = math.var("y");

    // Simple expression: 2x + y
    let _simple_expr = math.add(&math.mul(&math.constant(2.0), &x), &y);

    // Medium expression: xy + sin(x)
    let _medium_expr = math.add(&math.mul(&x, &y), &math.sin(&x));

    // Complex expression: x * x² + exp(y)
    math.add(
        &math.mul(&x, &math.pow(&x, &math.constant(2.0))),
        &math.exp(&y),
    )
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
        group.bench_function("cranelift_jit", |b| {
            b.iter(|| {
                let jit_compiler = JITCompiler::new().unwrap();
                let jit_func = jit_compiler
                    .compile_two_vars(&advanced_optimized, "x", "y")
                    .unwrap();
                jit_func.call_two_vars(black_box(x), black_box(y))
            });
        });

        // Benchmark pre-compiled JIT function (amortized cost)
        let jit_compiler = JITCompiler::new().unwrap();
        let jit_func = jit_compiler
            .compile_two_vars(&advanced_optimized, "x", "y")
            .unwrap();

        group.bench_function("precompiled_jit", |b| {
            b.iter(|| jit_func.call_two_vars(black_box(x), black_box(y)));
        });

        println!("\n🔧 JIT Compilation Stats:");
        println!("Code size: {} bytes", jit_func.stats.code_size_bytes);
        println!(
            "Compilation time: {} μs",
            jit_func.stats.compilation_time_us
        );
        println!("Operations compiled: {}", jit_func.stats.operation_count);
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
        DirectEval::eval_two_vars(&complex_expr, x, y);
    }
    let original_duration = original_time.elapsed();

    let optimized_time = std::time::Instant::now();
    for _ in 0..10000 {
        DirectEval::eval_two_vars(&optimized, x, y);
    }
    let optimized_duration = optimized_time.elapsed();

    #[cfg(feature = "cranelift")]
    {
        // Test JIT performance
        let jit_compiler = JITCompiler::new().unwrap();
        let jit_func = jit_compiler.compile_two_vars(&optimized, "x", "y").unwrap();

        let jit_time = std::time::Instant::now();
        for _ in 0..10000 {
            jit_func.call_two_vars(x, y);
        }
        let jit_duration = jit_time.elapsed();

        let speedup_opt =
            original_duration.as_nanos() as f64 / optimized_duration.as_nanos() as f64;
        let speedup_jit = original_duration.as_nanos() as f64 / jit_duration.as_nanos() as f64;
        let jit_vs_opt = optimized_duration.as_nanos() as f64 / jit_duration.as_nanos() as f64;

        println!("\n⚡ Performance Comparison (10k evaluations):");
        println!("Original time: {original_duration:?}");
        println!("Optimized time: {optimized_duration:?}");
        println!("JIT time: {jit_duration:?}");
        println!("Optimization speedup: {speedup_opt:.2}x");
        println!("JIT speedup vs original: {speedup_jit:.2}x");
        println!("JIT speedup vs optimized: {jit_vs_opt:.2}x");
    }

    #[cfg(not(feature = "cranelift"))]
    {
        let speedup_opt =
            original_duration.as_nanos() as f64 / optimized_duration.as_nanos() as f64;

        println!("\n⚡ Performance Comparison (10k evaluations):");
        println!("Original time: {original_duration:?}");
        println!("Optimized time: {optimized_duration:?}");
        println!("Optimization speedup: {speedup_opt:.2}x");
        println!("(JIT benchmarks disabled - enable 'cranelift' feature)");
    }

    group.finish();
}

/// Benchmark Rust code generation
fn bench_rust_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("rust_generation");

    let complex_expr = create_complex_expression();

    // Optimize first
    let mut config = OptimizationConfig::default();
    config.egglog_optimization = true;
    config.constant_folding = true;
    let mut optimizer = SymbolicOptimizer::with_config(config).unwrap();
    let optimized = optimizer.optimize(&complex_expr).unwrap();

    group.bench_function("rust_code_generation", |b| {
        b.iter(|| {
            optimizer
                .generate_rust_source(black_box(&optimized), "bench_func")
                .unwrap()
        });
    });

    // Show generated code
    let rust_code = optimizer
        .generate_rust_source(&optimized, "optimized_func")
        .unwrap();
    println!("\n🦀 Generated Rust Code:");
    println!("{rust_code}");

    group.finish();
}

/// Comprehensive benchmark comparing all execution strategies
fn bench_execution_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("execution_strategies");
    group.sample_size(100); // Reduce sample size for compilation benchmarks

    let complex_expr = create_complex_expression();

    // Optimize the expression
    let mut config = OptimizationConfig::default();
    config.egglog_optimization = true;
    config.constant_folding = true;
    let mut optimizer = SymbolicOptimizer::with_config(config).unwrap();
    let optimized = optimizer.optimize(&complex_expr).unwrap();

    let x = 2.5;
    let y = 1.8;

    println!("\n🚀 Comprehensive Strategy Comparison:");
    println!(
        "Expression operations: {} → {}",
        complex_expr.count_operations(),
        optimized.count_operations()
    );

    // 1. Direct evaluation (baseline)
    group.bench_function("1_direct_evaluation", |b| {
        b.iter(|| DirectEval::eval_two_vars(black_box(&complex_expr), black_box(x), black_box(y)));
    });

    // 2. Optimized direct evaluation
    group.bench_function("2_optimized_evaluation", |b| {
        b.iter(|| DirectEval::eval_two_vars(black_box(&optimized), black_box(x), black_box(y)));
    });

    #[cfg(feature = "cranelift")]
    {
        // 3. JIT compilation + execution (full cost)
        group.bench_function("3_jit_compile_and_run", |b| {
            b.iter(|| {
                let jit_compiler = JITCompiler::new().unwrap();
                let jit_func = jit_compiler.compile_two_vars(&optimized, "x", "y").unwrap();
                jit_func.call_two_vars(black_box(x), black_box(y))
            });
        });

        // 4. Pre-compiled JIT execution (amortized cost)
        let jit_compiler = JITCompiler::new().unwrap();
        let jit_func = jit_compiler.compile_two_vars(&optimized, "x", "y").unwrap();

        group.bench_function("4_precompiled_jit_execution", |b| {
            b.iter(|| jit_func.call_two_vars(black_box(x), black_box(y)));
        });

        // Show when JIT compilation pays off
        let compilation_cost_ns = u128::from(jit_func.stats.compilation_time_us) * 1000;
        let direct_eval_time = std::time::Instant::now();
        DirectEval::eval_two_vars(&optimized, x, y);
        let direct_eval_ns = direct_eval_time.elapsed().as_nanos();

        let jit_eval_time = std::time::Instant::now();
        jit_func.call_two_vars(x, y);
        let jit_eval_ns = jit_eval_time.elapsed().as_nanos();

        if jit_eval_ns > 0 && direct_eval_ns > jit_eval_ns {
            let breakeven_calls = compilation_cost_ns / (direct_eval_ns - jit_eval_ns);
            println!("\n💡 JIT Breakeven Analysis:");
            println!(
                "Compilation cost: {} μs",
                jit_func.stats.compilation_time_us
            );
            println!("Direct eval time: {direct_eval_ns} ns");
            println!("JIT eval time: {jit_eval_ns} ns");
            println!("JIT pays off after ~{breakeven_calls} calls");
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_optimization_performance,
    bench_optimization_tradeoff,
    bench_rust_generation,
    bench_execution_strategies
);

criterion_main!(benches);
