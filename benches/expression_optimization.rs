//! Comprehensive benchmarks comparing expression optimization and Rust compilation performance
//!
//! This benchmark suite demonstrates the performance benefits of:
//! 1. Symbolic optimization (egglog-style algebraic simplification)
//! 2. Rust hot-loading compilation vs Cranelift JIT
//! 3. Different compilation strategies for various expression complexities

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
#[cfg(feature = "cranelift")]
use mathjit::backends::cranelift::JITCompiler;
use mathjit::final_tagless::{ASTEval, ASTMathExprf64, DirectEval};
use mathjit::symbolic::{CompilationStrategy, OptimizationConfig, RustOptLevel, SymbolicOptimizer};

use libloading::{Library, Symbol};
use std::fs;

/// Complex mathematical expression for benchmarking (f64 version)
fn create_complex_expression() -> mathjit::final_tagless::ASTRepr<f64> {
    // Complex expression: sin(x^2 + ln(exp(y))) * cos(sqrt(x + y)) + exp(ln(x * y)) - (x + 0) * 1
    // This expression contains many optimization opportunities:
    // - ln(exp(y)) = y
    // - exp(ln(x * y)) = x * y
    // - (x + 0) * 1 = x
    // - sqrt can be optimized in some cases
    <ASTEval as ASTMathExprf64>::add(
        <ASTEval as ASTMathExprf64>::sub(
            <ASTEval as ASTMathExprf64>::mul(
                <ASTEval as ASTMathExprf64>::sin(<ASTEval as ASTMathExprf64>::add(
                    <ASTEval as ASTMathExprf64>::pow(
                        <ASTEval as ASTMathExprf64>::var("x"),
                        <ASTEval as ASTMathExprf64>::constant(2.0),
                    ),
                    <ASTEval as ASTMathExprf64>::ln(<ASTEval as ASTMathExprf64>::exp(
                        <ASTEval as ASTMathExprf64>::var("y"),
                    )),
                )),
                <ASTEval as ASTMathExprf64>::cos(<ASTEval as ASTMathExprf64>::sqrt(
                    <ASTEval as ASTMathExprf64>::add(
                        <ASTEval as ASTMathExprf64>::var("x"),
                        <ASTEval as ASTMathExprf64>::var("y"),
                    ),
                )),
            ),
            <ASTEval as ASTMathExprf64>::exp(<ASTEval as ASTMathExprf64>::ln(
                <ASTEval as ASTMathExprf64>::mul(
                    <ASTEval as ASTMathExprf64>::var("x"),
                    <ASTEval as ASTMathExprf64>::var("y"),
                ),
            )),
        ),
        <ASTEval as ASTMathExprf64>::mul(
            <ASTEval as ASTMathExprf64>::add(
                <ASTEval as ASTMathExprf64>::var("x"),
                <ASTEval as ASTMathExprf64>::constant(0.0),
            ),
            <ASTEval as ASTMathExprf64>::constant(1.0),
        ),
    )
}

/// Medium complexity expression (f64 version)
fn create_medium_expression() -> mathjit::final_tagless::ASTRepr<f64> {
    // Medium expression: x^3 + 2*x^2 + ln(exp(x)) + (y + 0) * 1
    <ASTEval as ASTMathExprf64>::add(
        <ASTEval as ASTMathExprf64>::add(
            <ASTEval as ASTMathExprf64>::add(
                <ASTEval as ASTMathExprf64>::pow(
                    <ASTEval as ASTMathExprf64>::var("x"),
                    <ASTEval as ASTMathExprf64>::constant(3.0),
                ),
                <ASTEval as ASTMathExprf64>::mul(
                    <ASTEval as ASTMathExprf64>::constant(2.0),
                    <ASTEval as ASTMathExprf64>::pow(
                        <ASTEval as ASTMathExprf64>::var("x"),
                        <ASTEval as ASTMathExprf64>::constant(2.0),
                    ),
                ),
            ),
            <ASTEval as ASTMathExprf64>::ln(<ASTEval as ASTMathExprf64>::exp(
                <ASTEval as ASTMathExprf64>::var("x"),
            )),
        ),
        <ASTEval as ASTMathExprf64>::mul(
            <ASTEval as ASTMathExprf64>::add(
                <ASTEval as ASTMathExprf64>::var("y"),
                <ASTEval as ASTMathExprf64>::constant(0.0),
            ),
            <ASTEval as ASTMathExprf64>::constant(1.0),
        ),
    )
}

/// Simple expression for baseline comparison (f64 version)
fn create_simple_expression() -> mathjit::final_tagless::ASTRepr<f64> {
    // Simple expression: x + y + 1
    <ASTEval as ASTMathExprf64>::add(
        <ASTEval as ASTMathExprf64>::add(
            <ASTEval as ASTMathExprf64>::var("x"),
            <ASTEval as ASTMathExprf64>::var("y"),
        ),
        <ASTEval as ASTMathExprf64>::constant(1.0),
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
    #[cfg(feature = "cranelift")]
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
        #[cfg(feature = "cranelift")]
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

    // Benchmark optimization time using default (practical) configuration
    group.bench_function("optimization_time", |b| {
        b.iter(|| {
            // Use default configuration for realistic optimization timing
            let mut optimizer = SymbolicOptimizer::new().unwrap();
            optimizer.optimize(black_box(&complex_expr)).unwrap()
        });
    });

    // Pre-optimize for execution benchmarks using default configuration
    let mut optimizer = SymbolicOptimizer::new().unwrap();
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

/// Benchmark to demonstrate generic type support
fn bench_generic_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("generic_types");

    // Test values
    let x_f64 = 2.5_f64;
    let y_f64 = 1.8_f64;

    // Create expressions for f64
    let complex_f64 = create_complex_expression();

    group.bench_function("f64_complex", |b| {
        b.iter(|| {
            DirectEval::eval_two_vars(black_box(&complex_f64), black_box(x_f64), black_box(y_f64))
        });
    });

    // Demonstrate Rust backend generic code generation capabilities
    use mathjit::backends::rust_codegen::RustCodeGenerator;

    let rust_codegen = RustCodeGenerator::new();

    // Generate Rust code for f64
    if let Ok(f64_code) = rust_codegen.generate_function_generic(&complex_f64, "test_f64", "f64") {
        println!("Generated f64 function length: {} chars", f64_code.len());
    }

    // Generate Rust code for f32 (demonstrating generic backend capability)
    if let Ok(f32_code) = rust_codegen.generate_function_generic(&complex_f64, "test_f32", "f32") {
        println!("Generated f32 function length: {} chars", f32_code.len());
    }

    // Show that we can generate code for other types too
    if let Ok(i32_code) = rust_codegen.generate_function_generic(&complex_f64, "test_i32", "i32") {
        println!("Generated i32 function length: {} chars", i32_code.len());
    }

    group.finish();
}

/// Benchmark compilation pipeline phases separately: codegen → compilation → execution
fn bench_compilation_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("compilation_pipeline");
    group.sample_size(50); // Reduce sample size for compilation benchmarks

    let complex_expr = create_complex_expression();
    let x = 2.5;
    let y = 1.8;

    // Pre-optimize the expression once
    let mut optimizer = SymbolicOptimizer::new().unwrap();
    let optimized_expr = optimizer.optimize(&complex_expr).unwrap();

    // === CODEGEN PHASE BENCHMARKS ===
    
    // Benchmark Rust codegen time
    use mathjit::backends::rust_codegen::RustCodeGenerator;
    let rust_codegen = RustCodeGenerator::new();
    
    group.bench_function("rust_codegen", |b| {
        b.iter(|| {
            rust_codegen.generate_function(black_box(&optimized_expr), "bench_func").unwrap()
        });
    });

    #[cfg(feature = "cranelift")]
    group.bench_function("cranelift_codegen", |b| {
        b.iter(|| {
            // Just measure IR generation time, not compilation
            let compiler = JITCompiler::new().unwrap();
            // Note: This includes some compilation, but it's the closest we can get to pure codegen
            drop(compiler.compile_two_vars(black_box(&optimized_expr), "x", "y").unwrap());
        });
    });

    // === COMPILATION PHASE BENCHMARKS ===
    
    // Pre-generate Rust code for compilation benchmarks
    let rust_code = rust_codegen.generate_function(&optimized_expr, "bench_func").unwrap();
    
    // Setup temp directories
    let temp_dir = std::env::temp_dir().join("mathjit_pipeline_bench");
    let source_dir = temp_dir.join("sources");
    let lib_dir = temp_dir.join("libs");
    let _ = fs::create_dir_all(&source_dir);
    let _ = fs::create_dir_all(&lib_dir);
    
    let source_path = source_dir.join("bench_func.rs");
    let lib_path = lib_dir.join("libbench_func.so");

    // Benchmark Rust compilation time (O0 - debug)
    group.bench_function("rust_compile_o0", |b| {
        let rust_strategy = CompilationStrategy::HotLoadRust {
            source_dir: source_dir.clone(),
            lib_dir: lib_dir.clone(),
            opt_level: RustOptLevel::O0,
        };
        let rust_optimizer = SymbolicOptimizer::with_strategy(rust_strategy).unwrap();
        
        b.iter(|| {
            let _ = rust_optimizer.compile_rust_dylib(
                black_box(&rust_code), 
                &source_path, 
                &lib_path, 
                &RustOptLevel::O0
            );
        });
    });

    // Benchmark Rust compilation time (O2 - optimized)
    group.bench_function("rust_compile_o2", |b| {
        let rust_strategy = CompilationStrategy::HotLoadRust {
            source_dir: source_dir.clone(),
            lib_dir: lib_dir.clone(),
            opt_level: RustOptLevel::O2,
        };
        let rust_optimizer = SymbolicOptimizer::with_strategy(rust_strategy).unwrap();
        
        b.iter(|| {
            let _ = rust_optimizer.compile_rust_dylib(
                black_box(&rust_code), 
                &source_path, 
                &lib_path, 
                &RustOptLevel::O2
            );
        });
    });

    // Benchmark Rust compilation time (O3 - aggressive optimization)
    group.bench_function("rust_compile_o3", |b| {
        let rust_strategy = CompilationStrategy::HotLoadRust {
            source_dir: source_dir.clone(),
            lib_dir: lib_dir.clone(),
            opt_level: RustOptLevel::O3,
        };
        let rust_optimizer = SymbolicOptimizer::with_strategy(rust_strategy).unwrap();
        
        b.iter(|| {
            let _ = rust_optimizer.compile_rust_dylib(
                black_box(&rust_code), 
                &source_path, 
                &lib_path, 
                &RustOptLevel::O3
            );
        });
    });

    // === EXECUTION PHASE BENCHMARKS ===
    
    // Pre-compile for execution benchmarks
    
    // 1. Compile Rust O0 version
    let rust_strategy_o0 = CompilationStrategy::HotLoadRust {
        source_dir: source_dir.clone(),
        lib_dir: lib_dir.clone(),
        opt_level: RustOptLevel::O0,
    };
    let rust_optimizer_o0 = SymbolicOptimizer::with_strategy(rust_strategy_o0).unwrap();
    let lib_path_o0 = lib_dir.join("libbench_func_o0.so");
    
    if rust_optimizer_o0.compile_rust_dylib(&rust_code, &source_path, &lib_path_o0, &RustOptLevel::O0).is_ok() {
        if let Ok(lib) = unsafe { Library::new(&lib_path_o0) } {
            if let Ok(func) = unsafe {
                lib.get::<Symbol<unsafe extern "C" fn(f64, f64) -> f64>>(b"bench_func_two_vars")
            } {
                group.bench_function("rust_execute_o0", |b| {
                    b.iter(|| unsafe { func(black_box(x), black_box(y)) });
                });
            }
        }
    }

    // 2. Compile Rust O2 version
    let rust_strategy_o2 = CompilationStrategy::HotLoadRust {
        source_dir: source_dir.clone(),
        lib_dir: lib_dir.clone(),
        opt_level: RustOptLevel::O2,
    };
    let rust_optimizer_o2 = SymbolicOptimizer::with_strategy(rust_strategy_o2).unwrap();
    let lib_path_o2 = lib_dir.join("libbench_func_o2.so");
    
    if rust_optimizer_o2.compile_rust_dylib(&rust_code, &source_path, &lib_path_o2, &RustOptLevel::O2).is_ok() {
        if let Ok(lib) = unsafe { Library::new(&lib_path_o2) } {
            if let Ok(func) = unsafe {
                lib.get::<Symbol<unsafe extern "C" fn(f64, f64) -> f64>>(b"bench_func_two_vars")
            } {
                group.bench_function("rust_execute_o2", |b| {
                    b.iter(|| unsafe { func(black_box(x), black_box(y)) });
                });
            }
        }
    }

    // 3. Compile Rust O3 version
    let rust_strategy_o3 = CompilationStrategy::HotLoadRust {
        source_dir: source_dir.clone(),
        lib_dir: lib_dir.clone(),
        opt_level: RustOptLevel::O3,
    };
    let rust_optimizer_o3 = SymbolicOptimizer::with_strategy(rust_strategy_o3).unwrap();
    let lib_path_o3 = lib_dir.join("libbench_func_o3.so");
    
    if rust_optimizer_o3.compile_rust_dylib(&rust_code, &source_path, &lib_path_o3, &RustOptLevel::O3).is_ok() {
        if let Ok(lib) = unsafe { Library::new(&lib_path_o3) } {
            if let Ok(func) = unsafe {
                lib.get::<Symbol<unsafe extern "C" fn(f64, f64) -> f64>>(b"bench_func_two_vars")
            } {
                group.bench_function("rust_execute_o3", |b| {
                    b.iter(|| unsafe { func(black_box(x), black_box(y)) });
                });
            }
        }
    }

    // 4. Cranelift execution
    #[cfg(feature = "cranelift")]
    {
        let jit_compiler = JITCompiler::new().unwrap();
        let jit_func = jit_compiler.compile_two_vars(&optimized_expr, "x", "y").unwrap();
        
        group.bench_function("cranelift_execute", |b| {
            b.iter(|| jit_func.call_two_vars(black_box(x), black_box(y)));
        });
    }

    // 5. Direct evaluation (baseline)
    group.bench_function("direct_execute", |b| {
        b.iter(|| DirectEval::eval_two_vars(black_box(&optimized_expr), black_box(x), black_box(y)));
    });

    // Cleanup
    let _ = fs::remove_dir_all(&temp_dir);
    
    group.finish();
}

/// Comprehensive benchmark comparing egglog vs non-egglog optimization strategies
fn bench_egglog_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("egglog_comparison");
    group.sample_size(100); // Increase sample size for execution benchmarks
    
    let complex_expr = create_complex_expression();
    let x = 2.5;
    let y = 1.8;

    // === PRE-OPTIMIZE EXPRESSIONS (DONE ONCE) ===
    
    println!("Pre-optimizing expressions for comparison...");
    
    // Default optimization (fast)
    let start = std::time::Instant::now();
    let mut default_optimizer = SymbolicOptimizer::new().unwrap();
    let default_optimized = default_optimizer.optimize(&complex_expr).unwrap();
    let default_time = start.elapsed();
    
    // Egglog optimization (slow but thorough)
    let start = std::time::Instant::now();
    let mut egglog_config = OptimizationConfig::default();
    egglog_config.egglog_optimization = true;
    egglog_config.constant_folding = true;
    egglog_config.cse = true;
    let mut egglog_optimizer = SymbolicOptimizer::with_config(egglog_config).unwrap();
    let egglog_optimized = egglog_optimizer.optimize(&complex_expr).unwrap();
    let egglog_time = start.elapsed();

    println!("Original expression operations: {}", complex_expr.count_operations());
    println!("Default optimized operations: {} (took {:?})", default_optimized.count_operations(), default_time);
    println!("Egglog optimized operations: {} (took {:?})", egglog_optimized.count_operations(), egglog_time);
    println!("Egglog optimization is {:.1}x slower than default", egglog_time.as_secs_f64() / default_time.as_secs_f64());
    
    // === EXECUTION TIME COMPARISON (MAIN FOCUS) ===
    
    // Baseline - original expression
    group.bench_function("execute_original", |b| {
        b.iter(|| DirectEval::eval_two_vars(black_box(&complex_expr), black_box(x), black_box(y)));
    });

    // Default optimization execution
    group.bench_function("execute_default_optimized", |b| {
        b.iter(|| DirectEval::eval_two_vars(black_box(&default_optimized), black_box(x), black_box(y)));
    });

    // Egglog optimization execution
    group.bench_function("execute_egglog_optimized", |b| {
        b.iter(|| DirectEval::eval_two_vars(black_box(&egglog_optimized), black_box(x), black_box(y)));
    });

    // === CODEGEN TIME COMPARISON (QUICK) ===
    
    use mathjit::backends::rust_codegen::RustCodeGenerator;
    let rust_codegen = RustCodeGenerator::new();
    
    group.bench_function("codegen_default_optimized", |b| {
        b.iter(|| {
            rust_codegen.generate_function(black_box(&default_optimized), "default_func").unwrap()
        });
    });

    group.bench_function("codegen_egglog_optimized", |b| {
        b.iter(|| {
            rust_codegen.generate_function(black_box(&egglog_optimized), "egglog_func").unwrap()
        });
    });

    // === OPTIONAL: SINGLE COMPILATION TIME MEASUREMENT (NOT BENCHMARKED) ===
    
    // Just measure compilation time once for comparison, don't benchmark it
    let default_rust_code = rust_codegen.generate_function(&default_optimized, "default_func").unwrap();
    let egglog_rust_code = rust_codegen.generate_function(&egglog_optimized, "egglog_func").unwrap();
    
    println!("Generated Rust code sizes:");
    println!("  Default optimized: {} chars", default_rust_code.len());
    println!("  Egglog optimized: {} chars", egglog_rust_code.len());
    
    // Quick compilation time test (not benchmarked due to expense)
    let temp_dir = std::env::temp_dir().join("mathjit_egglog_quick");
    let source_dir = temp_dir.join("sources");
    let lib_dir = temp_dir.join("libs");
    let _ = fs::create_dir_all(&source_dir);
    let _ = fs::create_dir_all(&lib_dir);
    
    let rust_strategy = CompilationStrategy::HotLoadRust {
        source_dir: source_dir.clone(),
        lib_dir: lib_dir.clone(),
        opt_level: RustOptLevel::O2,
    };
    let rust_optimizer = SymbolicOptimizer::with_strategy(rust_strategy).unwrap();
    
    // Time compilation once each (not benchmarked)
    let default_source_path = source_dir.join("default_func.rs");
    let egglog_source_path = source_dir.join("egglog_func.rs");
    let default_lib_path = lib_dir.join("libdefault_func.so");
    let egglog_lib_path = lib_dir.join("libegglog_func.so");
    
    let start = std::time::Instant::now();
    let default_compile_result = rust_optimizer.compile_rust_dylib(
        &default_rust_code, 
        &default_source_path, 
        &default_lib_path, 
        &RustOptLevel::O2
    );
    let default_compile_time = start.elapsed();
    
    let start = std::time::Instant::now();
    let egglog_compile_result = rust_optimizer.compile_rust_dylib(
        &egglog_rust_code, 
        &egglog_source_path, 
        &egglog_lib_path, 
        &RustOptLevel::O2
    );
    let egglog_compile_time = start.elapsed();
    
    println!("Rust compilation times (O2):");
    println!("  Default optimized: {:?}", default_compile_time);
    println!("  Egglog optimized: {:?}", egglog_compile_time);
    
    // === COMPILED EXECUTION BENCHMARKS ===
    
    // Benchmark compiled execution if compilation succeeded
    if default_compile_result.is_ok() {
        if let Ok(lib) = unsafe { Library::new(&default_lib_path) } {
            if let Ok(func) = unsafe {
                lib.get::<Symbol<unsafe extern "C" fn(f64, f64) -> f64>>(b"default_func_two_vars")
            } {
                group.bench_function("compiled_execute_default", |b| {
                    b.iter(|| unsafe { func(black_box(x), black_box(y)) });
                });
            }
        }
    }

    if egglog_compile_result.is_ok() {
        if let Ok(lib) = unsafe { Library::new(&egglog_lib_path) } {
            if let Ok(func) = unsafe {
                lib.get::<Symbol<unsafe extern "C" fn(f64, f64) -> f64>>(b"egglog_func_two_vars")
            } {
                group.bench_function("compiled_execute_egglog", |b| {
                    b.iter(|| unsafe { func(black_box(x), black_box(y)) });
                });
            }
        }
    }

    // Cleanup
    let _ = fs::remove_dir_all(&temp_dir);
    
    group.finish();
}

criterion_group!(
    benches,
    bench_direct_evaluation,
    bench_optimization_comparison,
    bench_compilation_strategies,
    bench_complexity_scaling,
    bench_optimization_tradeoff,
    bench_generic_types,
    bench_compilation_pipeline,
    bench_egglog_comparison
);

criterion_main!(benches);
