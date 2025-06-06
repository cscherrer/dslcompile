use dslcompile::ast::{ASTRepr, DynamicContext};
use dslcompile::backends::{RustCodeGenerator, RustCompiler, RustOptLevel};
use rand::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Runtime Codegen Overhead Test - Pure Evaluation Performance");
    println!("==============================================================");
    println!("Testing ONLY evaluation time with runtime data (no constant propagation)");
    println!();

    // Generate random test data to prevent constant propagation
    let mut rng = thread_rng();
    let iterations = 10_000_000; // Large number for accurate timing
    let test_data: Vec<(f64, f64)> = (0..iterations)
        .map(|_| (rng.gen_range(-100.0..100.0), rng.gen_range(-100.0..100.0)))
        .collect();

    println!("Generated {iterations} random test pairs");
    println!("Data range: x,y ∈ [-100, 100]");
    println!();

    // ============================================================================
    // SIMPLE ADDITION: x + y
    // ============================================================================

    println!("🧮 SIMPLE ADDITION: x + y");
    println!("=========================");

    // 1. Native Rust baseline (pure evaluation)
    let start = Instant::now();
    let mut native_sum = 0.0;
    for &(x, y) in &test_data {
        native_sum += x + y; // Accumulate to prevent optimization
    }
    let native_time = start.elapsed();
    let native_ns_per_op = native_time.as_nanos() as f64 / iterations as f64;

    println!("Native Rust:          {native_ns_per_op:.3}ns per operation (sum: {native_sum:.2})");

    // 2. Build expression for codegen
    let ctx = DynamicContext::new();
    let x_var = ctx.var();
    let y_var = ctx.var();
    let add_expr = &x_var + &y_var;
    let ast_expr: ASTRepr<f64> = add_expr.into();

    // 3. Generate and compile Rust code (EXCLUDE from timing)
    println!("Generating and compiling Rust code...");
    let codegen = RustCodeGenerator::new();
    let rust_code = codegen.generate_function(&ast_expr, "add_func")?;

    let compiler = RustCompiler::with_opt_level(RustOptLevel::O3).with_extra_flags(vec![
        "-C".to_string(),
        "target-cpu=native".to_string(),
        "-C".to_string(),
        "opt-level=3".to_string(),
    ]);

    let temp_dir = std::env::temp_dir();
    let source_path = temp_dir.join("add_test.rs");
    let lib_path = temp_dir.join("libadd_test.so");

    std::fs::write(&source_path, &rust_code)?;
    compiler.compile_dylib(&rust_code, &source_path, &lib_path)?;

    // 4. Load compiled function (EXCLUDE from timing)
    use dlopen2::raw::Library;
    let library = Library::open(&lib_path)?;
    let compiled_add: extern "C" fn(f64, f64) -> f64 = unsafe { library.symbol("add_func")? };

    println!("Compilation complete. Testing pure evaluation performance...");
    println!();

    // 5. Benchmark ONLY evaluation time
    let start = Instant::now();
    let mut compiled_sum = 0.0;
    for &(x, y) in &test_data {
        compiled_sum += compiled_add(x, y); // Direct function call
    }
    let compiled_time = start.elapsed();
    let compiled_ns_per_op = compiled_time.as_nanos() as f64 / iterations as f64;

    println!(
        "Compiled Rust:        {compiled_ns_per_op:.3}ns per operation (sum: {compiled_sum:.2})"
    );

    // Calculate overhead
    let overhead_ns = compiled_ns_per_op - native_ns_per_op;
    let overhead_ratio = compiled_ns_per_op / native_ns_per_op;

    println!();
    println!("📊 OVERHEAD ANALYSIS:");
    println!("  Absolute overhead:  {overhead_ns:.3}ns per operation");
    println!("  Relative overhead:  {overhead_ratio:.2}x native performance");

    // Verify results are identical (within floating point precision)
    let diff = (native_sum - compiled_sum).abs();
    println!("  Result difference:  {diff:.2e} (should be ~0)");

    if overhead_ns < 2.0 {
        println!("  ✅ EXCELLENT: Overhead < 2ns");
    } else if overhead_ns < 5.0 {
        println!("  ⚠️  ACCEPTABLE: Overhead < 5ns");
    } else {
        println!("  ❌ POOR: Overhead > 5ns - investigate!");
    }

    println!();

    // ============================================================================
    // SIMPLE MULTIPLICATION: x * y
    // ============================================================================

    println!("🧮 SIMPLE MULTIPLICATION: x * y");
    println!("================================");

    // Native baseline
    let start = Instant::now();
    let mut native_prod = 1.0;
    for &(x, y) in &test_data[..1000] {
        // Smaller sample to avoid overflow
        native_prod *= (x * y).abs().sqrt(); // Prevent overflow while keeping computation
    }
    let native_mul_time = start.elapsed();
    let native_mul_ns = native_mul_time.as_nanos() as f64 / 1000.0;

    // Build multiplication expression
    let mul_expr = &x_var * &y_var;
    let mul_ast: ASTRepr<f64> = mul_expr.into();

    // Generate and compile
    let mul_code = codegen.generate_function(&mul_ast, "mul_func")?;
    let mul_source_path = temp_dir.join("mul_test.rs");
    let mul_lib_path = temp_dir.join("libmul_test.so");

    std::fs::write(&mul_source_path, &mul_code)?;
    compiler.compile_dylib(&mul_code, &mul_source_path, &mul_lib_path)?;

    let mul_library = Library::open(&mul_lib_path)?;
    let compiled_mul: extern "C" fn(f64, f64) -> f64 = unsafe { mul_library.symbol("mul_func")? };

    // Benchmark multiplication
    let start = Instant::now();
    let mut compiled_prod = 1.0;
    for &(x, y) in &test_data[..1000] {
        compiled_prod *= (compiled_mul(x, y)).abs().sqrt();
    }
    let compiled_mul_time = start.elapsed();
    let compiled_mul_ns = compiled_mul_time.as_nanos() as f64 / 1000.0;

    println!("Native Rust:          {native_mul_ns:.3}ns per operation");
    println!("Compiled Rust:        {compiled_mul_ns:.3}ns per operation");

    let mul_overhead = compiled_mul_ns - native_mul_ns;
    println!("Absolute overhead:    {mul_overhead:.3}ns per operation");

    println!();

    // ============================================================================
    // COMPLEX EXPRESSION: x*x + 2*x*y + y*y
    // ============================================================================

    println!("🧮 COMPLEX EXPRESSION: x² + 2xy + y²");
    println!("=====================================");

    // Native baseline
    let start = Instant::now();
    let mut native_complex = 0.0;
    for &(x, y) in &test_data[..100_000] {
        // Moderate sample size
        native_complex += x * x + 2.0 * x * y + y * y;
    }
    let native_complex_time = start.elapsed();
    let native_complex_ns = native_complex_time.as_nanos() as f64 / 100_000.0;

    // Build complex expression
    let complex_expr = &x_var * &x_var + 2.0 * &x_var * &y_var + &y_var * &y_var;
    let complex_ast: ASTRepr<f64> = complex_expr.into();

    // Generate and compile
    let complex_code = codegen.generate_function(&complex_ast, "complex_func")?;
    let complex_source_path = temp_dir.join("complex_test.rs");
    let complex_lib_path = temp_dir.join("libcomplex_test.so");

    std::fs::write(&complex_source_path, &complex_code)?;
    compiler.compile_dylib(&complex_code, &complex_source_path, &complex_lib_path)?;

    let complex_library = Library::open(&complex_lib_path)?;
    let compiled_complex: extern "C" fn(f64, f64) -> f64 =
        unsafe { complex_library.symbol("complex_func")? };

    // Benchmark complex expression
    let start = Instant::now();
    let mut compiled_complex_result = 0.0;
    for &(x, y) in &test_data[..100_000] {
        compiled_complex_result += compiled_complex(x, y);
    }
    let compiled_complex_time = start.elapsed();
    let compiled_complex_ns = compiled_complex_time.as_nanos() as f64 / 100_000.0;

    println!("Native Rust:          {native_complex_ns:.3}ns per operation");
    println!("Compiled Rust:        {compiled_complex_ns:.3}ns per operation");

    let complex_overhead = compiled_complex_ns - native_complex_ns;
    println!("Absolute overhead:    {complex_overhead:.3}ns per operation");

    println!();

    // ============================================================================
    // SUMMARY
    // ============================================================================

    println!("🏆 RUNTIME CODEGEN OVERHEAD SUMMARY");
    println!("===================================");
    println!("Simple addition:      {overhead_ns:.3}ns overhead");
    println!("Simple multiplication: {mul_overhead:.3}ns overhead");
    println!("Complex expression:   {complex_overhead:.3}ns overhead");
    println!();

    let avg_overhead = (overhead_ns + mul_overhead + complex_overhead) / 3.0;
    println!("Average overhead:     {avg_overhead:.3}ns per operation");

    if avg_overhead < 1.0 {
        println!("✅ EXCELLENT: Runtime codegen has sub-nanosecond overhead!");
    } else if avg_overhead < 3.0 {
        println!("✅ GOOD: Runtime codegen overhead is acceptable");
    } else {
        println!("❌ POOR: Runtime codegen has significant overhead - needs optimization");
    }

    // Cleanup
    let _ = std::fs::remove_file(&source_path);
    let _ = std::fs::remove_file(&lib_path);
    let _ = std::fs::remove_file(&mul_source_path);
    let _ = std::fs::remove_file(&mul_lib_path);
    let _ = std::fs::remove_file(&complex_source_path);
    let _ = std::fs::remove_file(&complex_lib_path);

    Ok(())
}
