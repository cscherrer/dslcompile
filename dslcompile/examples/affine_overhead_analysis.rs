use dslcompile::ast::{ASTRepr, DynamicContext};
use dslcompile::backends::{RustCodeGenerator, RustCompiler, RustOptLevel};
use rand::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Affine Overhead Analysis: time = a*n + b");
    println!("============================================");
    println!("Measuring overhead as an affine function across multiple input sizes");
    println!();

    // Test different input sizes for linear regression
    let test_sizes = vec![1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000];

    // Generate random test data
    let mut rng = thread_rng();
    let max_size = *test_sizes.iter().max().unwrap();
    let test_data: Vec<(f64, f64)> = (0..max_size)
        .map(|_| (rng.gen_range(-100.0..100.0), rng.gen_range(-100.0..100.0)))
        .collect();

    println!("Generated {max_size} random test pairs");
    println!("Test sizes: {test_sizes:?}");
    println!();

    // ============================================================================
    // SETUP: Build and compile expression once
    // ============================================================================

    let ctx = DynamicContext::new();
    let x_var = ctx.var();
    let y_var = ctx.var();
    let expr = &x_var * &y_var + 42.0; // x * y + 42
    let ast_expr: ASTRepr<f64> = expr.into();

    println!("🔧 Compiling expression: x * y + 42");
    let codegen = RustCodeGenerator::new();
    let rust_code = codegen.generate_function(&ast_expr, "test_func")?;

    let compiler = RustCompiler::with_opt_level(RustOptLevel::O3)
        .with_extra_flags(vec!["-C".to_string(), "target-cpu=native".to_string()]);

    let temp_dir = std::env::temp_dir();
    let source_path = temp_dir.join("affine_test.rs");
    let lib_path = temp_dir.join("libaffine_test.so");

    std::fs::write(&source_path, &rust_code)?;
    compiler.compile_dylib(&rust_code, &source_path, &lib_path)?;

    use dlopen2::raw::Library;
    let library = Library::open(&lib_path)?;
    let compiled_func: extern "C" fn(f64, f64) -> f64 = unsafe { library.symbol("test_func")? };

    println!("✅ Compilation complete");
    println!();

    // ============================================================================
    // BENCHMARK: Measure across different input sizes
    // ============================================================================

    let mut native_measurements = Vec::new();
    let mut compiled_measurements = Vec::new();

    println!("🏃 Running benchmarks across different input sizes...");
    println!(
        "{:>10} {:>15} {:>15} {:>15} {:>15}",
        "Size", "Native (ns)", "Compiled (ns)", "Native/op", "Compiled/op"
    );
    println!("{:-<75}", "");

    for &size in &test_sizes {
        let data_slice = &test_data[..size];

        // Warm up
        for _ in 0..10 {
            let mut sum = 0.0;
            for &(x, y) in data_slice.iter().take(100) {
                sum += x * y + 42.0;
            }
            std::hint::black_box(sum);
        }

        // Measure native Rust (multiple runs for accuracy)
        let mut native_times = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let mut sum = 0.0;
            for &(x, y) in data_slice {
                sum += x * y + 42.0;
            }
            let elapsed = start.elapsed();
            std::hint::black_box(sum);
            native_times.push(elapsed.as_nanos() as f64);
        }
        let native_time = native_times.iter().sum::<f64>() / native_times.len() as f64;

        // Measure compiled version (multiple runs for accuracy)
        let mut compiled_times = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let mut sum = 0.0;
            for &(x, y) in data_slice {
                sum += compiled_func(x, y);
            }
            let elapsed = start.elapsed();
            std::hint::black_box(sum);
            compiled_times.push(elapsed.as_nanos() as f64);
        }
        let compiled_time = compiled_times.iter().sum::<f64>() / compiled_times.len() as f64;

        let native_per_op = native_time / size as f64;
        let compiled_per_op = compiled_time / size as f64;

        println!(
            "{size:>10} {native_time:>15.0} {compiled_time:>15.0} {native_per_op:>15.3} {compiled_per_op:>15.3}"
        );

        native_measurements.push((size as f64, native_time));
        compiled_measurements.push((size as f64, compiled_time));
    }

    println!();

    // ============================================================================
    // LINEAR REGRESSION: Find affine model time = a*n + b
    // ============================================================================

    fn linear_regression(points: &[(f64, f64)]) -> (f64, f64, f64) {
        let n = points.len() as f64;
        let sum_x: f64 = points.iter().map(|(x, _)| *x).sum();
        let sum_y: f64 = points.iter().map(|(_, y)| *y).sum();
        let sum_xy: f64 = points.iter().map(|(x, y)| x * y).sum();
        let sum_x2: f64 = points.iter().map(|(x, _)| x * x).sum();

        // Linear regression: y = ax + b
        let a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let b = (sum_y - a * sum_x) / n;

        // Calculate R²
        let y_mean = sum_y / n;
        let ss_tot: f64 = points.iter().map(|(_, y)| (y - y_mean).powi(2)).sum();
        let ss_res: f64 = points.iter().map(|(x, y)| (y - (a * x + b)).powi(2)).sum();
        let r_squared = 1.0 - (ss_res / ss_tot);

        (a, b, r_squared)
    }

    println!("📈 LINEAR REGRESSION ANALYSIS");
    println!("=============================");

    let (native_a, native_b, native_r2) = linear_regression(&native_measurements);
    let (compiled_a, compiled_b, compiled_r2) = linear_regression(&compiled_measurements);

    println!(
        "Native Rust model:    time = {native_a:.6}*n + {native_b:.1} ns  (R² = {native_r2:.4})"
    );
    println!(
        "Compiled model:       time = {compiled_a:.6}*n + {compiled_b:.1} ns  (R² = {compiled_r2:.4})"
    );
    println!();

    // ============================================================================
    // OVERHEAD ANALYSIS
    // ============================================================================

    let overhead_per_op = compiled_a - native_a; // Difference in slope (ns per operation)
    let overhead_constant = compiled_b - native_b; // Difference in intercept (constant overhead)

    println!("🎯 AFFINE OVERHEAD ANALYSIS");
    println!("===========================");
    println!("Per-operation overhead:   {overhead_per_op:.6} ns/op");
    println!("Constant overhead:        {overhead_constant:.1} ns");
    println!();

    if overhead_per_op.abs() < 0.001 {
        println!("✅ EXCELLENT: Per-operation overhead < 0.001 ns/op");
    } else if overhead_per_op.abs() < 0.01 {
        println!("✅ GOOD: Per-operation overhead < 0.01 ns/op");
    } else {
        println!("⚠️  MODERATE: Per-operation overhead = {overhead_per_op:.3} ns/op");
    }

    if overhead_constant.abs() < 100.0 {
        println!("✅ EXCELLENT: Constant overhead < 100 ns");
    } else if overhead_constant.abs() < 1000.0 {
        println!("✅ GOOD: Constant overhead < 1μs");
    } else {
        println!("⚠️  HIGH: Constant overhead = {overhead_constant:.0} ns");
    }

    println!();

    // ============================================================================
    // PREDICTIONS AND VALIDATION
    // ============================================================================

    println!("🔮 MODEL PREDICTIONS");
    println!("===================");

    let test_n = 2_000_000.0;
    let native_pred = native_a * test_n + native_b;
    let compiled_pred = compiled_a * test_n + compiled_b;
    let overhead_pred = compiled_pred - native_pred;

    println!("For n = {test_n:.0} operations:");
    println!(
        "  Native prediction:    {:.1} ms",
        native_pred / 1_000_000.0
    );
    println!(
        "  Compiled prediction:  {:.1} ms",
        compiled_pred / 1_000_000.0
    );
    println!(
        "  Overhead prediction:  {:.1} ms ({:.1}%)",
        overhead_pred / 1_000_000.0,
        (overhead_pred / native_pred) * 100.0
    );

    println!();

    // ============================================================================
    // SUMMARY
    // ============================================================================

    println!("📋 SUMMARY");
    println!("==========");
    println!(
        "Runtime codegen overhead follows: overhead(n) = {overhead_per_op:.6}*n + {overhead_constant:.1}"
    );

    if overhead_per_op > 0.0 {
        println!("📈 Compiled version is {overhead_per_op:.3} ns/op SLOWER per operation");
    } else {
        println!(
            "📉 Compiled version is {:.3} ns/op FASTER per operation",
            -overhead_per_op
        );
    }

    if overhead_constant > 0.0 {
        println!("🔧 Compiled version has {overhead_constant:.0} ns constant setup cost");
    } else {
        println!(
            "🚀 Compiled version has {:.0} ns constant speedup",
            -overhead_constant
        );
    }

    // Cleanup
    let _ = std::fs::remove_file(&source_path);
    let _ = std::fs::remove_file(&lib_path);

    Ok(())
}
