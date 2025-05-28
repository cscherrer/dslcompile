//! Comprehensive benchmarks comparing expression optimization and Rust compilation performance
//!
//! This benchmark suite demonstrates the performance benefits of:
//! 1. Symbolic optimization (egglog-style algebraic simplification)
//! 2. Rust hot-loading compilation vs Cranelift JIT
//! 3. Different compilation strategies for various expression complexities

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mathjit::backends::cranelift::JITCompiler;
use mathjit::final_tagless::{DirectEval, JITEval, JITMathExpr};
use mathjit::symbolic::{CompilationStrategy, OptimizationConfig, RustOptLevel, SymbolicOptimizer};

use libloading::{Library, Symbol};
use std::fs;

/// Complex mathematical expression for benchmarking
fn create_complex_expression() -> mathjit::final_tagless::JITRepr<f64> {
    // Complex expression: sin(x^2 + ln(exp(y))) * cos(sqrt(x + y)) + exp(ln(x * y)) - (x + 0) * 1
    // This expression contains many optimization opportunities:
    // - ln(exp(y)) = y
    // - exp(ln(x * y)) = x * y
    // - (x + 0) * 1 = x
    // - sqrt can be optimized in some cases
    <JITEval as JITMathExpr>::add(
        <JITEval as JITMathExpr>::sub(
            <JITEval as JITMathExpr>::mul(
                <JITEval as JITMathExpr>::sin(<JITEval as JITMathExpr>::add(
                    <JITEval as JITMathExpr>::pow(
                        <JITEval as JITMathExpr>::var("x"),
                        <JITEval as JITMathExpr>::constant(2.0),
                    ),
                    <JITEval as JITMathExpr>::ln(<JITEval as JITMathExpr>::exp(
                        <JITEval as JITMathExpr>::var("y"),
                    )),
                )),
                <JITEval as JITMathExpr>::cos(<JITEval as JITMathExpr>::sqrt(
                    <JITEval as JITMathExpr>::add(
                        <JITEval as JITMathExpr>::var("x"),
                        <JITEval as JITMathExpr>::var("y"),
                    ),
                )),
            ),
            <JITEval as JITMathExpr>::exp(<JITEval as JITMathExpr>::ln(
                <JITEval as JITMathExpr>::mul(
                    <JITEval as JITMathExpr>::var("x"),
                    <JITEval as JITMathExpr>::var("y"),
                ),
            )),
        ),
        <JITEval as JITMathExpr>::mul(
            <JITEval as JITMathExpr>::add(
                <JITEval as JITMathExpr>::var("x"),
                <JITEval as JITMathExpr>::constant(0.0),
            ),
            <JITEval as JITMathExpr>::constant(1.0),
        ),
    )
}

/// Medium complexity expression
fn create_medium_expression() -> mathjit::final_tagless::JITRepr<f64> {
    // Medium expression: x^3 + 2*x^2 + ln(exp(x)) + (y + 0) * 1
    <JITEval as JITMathExpr>::add(
        <JITEval as JITMathExpr>::add(
            <JITEval as JITMathExpr>::add(
                <JITEval as JITMathExpr>::pow(
                    <JITEval as JITMathExpr>::var("x"),
                    <JITEval as JITMathExpr>::constant(3.0),
                ),
                <JITEval as JITMathExpr>::mul(
                    <JITEval as JITMathExpr>::constant(2.0),
                    <JITEval as JITMathExpr>::pow(
                        <JITEval as JITMathExpr>::var("x"),
                        <JITEval as JITMathExpr>::constant(2.0),
                    ),
                ),
            ),
            <JITEval as JITMathExpr>::ln(<JITEval as JITMathExpr>::exp(
                <JITEval as JITMathExpr>::var("x"),
            )),
        ),
        <JITEval as JITMathExpr>::mul(
            <JITEval as JITMathExpr>::add(
                <JITEval as JITMathExpr>::var("y"),
                <JITEval as JITMathExpr>::constant(0.0),
            ),
            <JITEval as JITMathExpr>::constant(1.0),
        ),
    )
}

/// Simple expression for baseline comparison
fn create_simple_expression() -> mathjit::final_tagless::JITRepr<f64> {
    // Simple expression: x + y + 1
    <JITEval as JITMathExpr>::add(
        <JITEval as JITMathExpr>::add(
            <JITEval as JITMathExpr>::var("x"),
            <JITEval as JITMathExpr>::var("y"),
        ),
        <JITEval as JITMathExpr>::constant(1.0),
    )
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
    config.egglog_optimization = true;
    config.constant_folding = true;
    config.aggressive = true;
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

/// Benchmark JIT compilation vs Rust compilation
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
    group.bench_function("cranelift_jit", |b| {
        b.iter(|| {
            let jit_compiler = JITCompiler::new().unwrap();
            let jit_func = jit_compiler
                .compile_two_vars(&optimized_expr, "x", "y")
                .unwrap();
            jit_func.call_two_vars(black_box(x), black_box(y))
        });
    });

    // Benchmark Rust hot-loading compilation (setup once, then execute many times)
    let temp_dir = std::env::temp_dir().join("mathjit_bench");
    let source_dir = temp_dir.join("sources");
    let lib_dir = temp_dir.join("libs");

    // Setup directories
    let _ = fs::create_dir_all(&source_dir);
    let _ = fs::create_dir_all(&lib_dir);

    // Compile Rust version once
    let rust_strategy = CompilationStrategy::HotLoadRust {
        source_dir: source_dir.clone(),
        lib_dir: lib_dir.clone(),
        opt_level: RustOptLevel::O2,
    };

    let rust_optimizer = SymbolicOptimizer::with_strategy(rust_strategy).unwrap();
    let rust_code = rust_optimizer
        .generate_rust_source(&optimized_expr, "bench_func")
        .unwrap();

    let source_path = source_dir.join("bench_func.rs");
    let lib_path = lib_dir.join("libbench_func.so");

    // Compile the Rust library
    if rust_optimizer
        .compile_rust_dylib(&rust_code, &source_path, &lib_path, &RustOptLevel::O2)
        .is_ok()
        && lib_path.exists()
    {
        // Load the library and benchmark execution
        if let Ok(lib) = unsafe { Library::new(&lib_path) } {
            if let Ok(func) = unsafe {
                lib.get::<Symbol<unsafe extern "C" fn(f64, f64) -> f64>>(b"bench_func_two_vars")
            } {
                group.bench_function("rust_compiled", |b| {
                    b.iter(|| unsafe { func(black_box(x), black_box(y)) });
                });
            }
        }
    }

    // Cleanup
    let _ = fs::remove_dir_all(&temp_dir);

    group.finish();
}

/// Benchmark different expression complexities
fn bench_complexity_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("complexity_scaling");

    let expressions = vec![
        ("simple", create_simple_expression()),
        ("medium", create_medium_expression()),
        ("complex", create_complex_expression()),
    ];

    let x = 2.5;
    let y = 1.8;

    for (name, expr) in &expressions {
        // Direct evaluation
        group.bench_with_input(BenchmarkId::new("direct", name), expr, |b, expr| {
            b.iter(|| DirectEval::eval_two_vars(black_box(expr), black_box(x), black_box(y)));
        });

        // Optimized evaluation
        let mut optimizer = SymbolicOptimizer::new().unwrap();
        let optimized = optimizer.optimize(expr).unwrap();

        group.bench_with_input(
            BenchmarkId::new("optimized", name),
            &optimized,
            |b, expr| {
                b.iter(|| DirectEval::eval_two_vars(black_box(expr), black_box(x), black_box(y)));
            },
        );

        // JIT compiled
        group.bench_with_input(BenchmarkId::new("jit", name), &optimized, |b, expr| {
            b.iter(|| {
                let jit_compiler = JITCompiler::new().unwrap();
                let jit_func = jit_compiler.compile_two_vars(expr, "x", "y").unwrap();
                jit_func.call_two_vars(black_box(x), black_box(y))
            });
        });
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

    // Benchmark execution time savings
    group.bench_function("unoptimized_execution", |b| {
        b.iter(|| DirectEval::eval_two_vars(black_box(&complex_expr), black_box(x), black_box(y)));
    });

    group.bench_function("optimized_execution", |b| {
        b.iter(|| DirectEval::eval_two_vars(black_box(&optimized), black_box(x), black_box(y)));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_direct_evaluation,
    bench_optimization_comparison,
    bench_compilation_strategies,
    bench_complexity_scaling,
    bench_optimization_tradeoff
);

criterion_main!(benches);
