use dslcompile::ast::{ASTRepr, DynamicContext, TypedBuilderExpr};
use dslcompile::backends::{RustCodeGenerator, RustCompiler, RustOptLevel};
use rand::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 DynamicContext Runtime Codegen Performance Test");
    println!("==================================================");

    // Build the same expression as Demo 6: x * y + 42
    let ctx = DynamicContext::new();
    let x_var = ctx.typed_var::<f64>();
    let y_var = ctx.typed_var::<f64>();
    let expr: TypedBuilderExpr<f64> =
        ctx.expr_from(x_var) * ctx.expr_from(y_var) + ctx.constant(42.0);

    println!("Expression: x * y + 42");
    println!();

    // Convert to AST for codegen
    let ast_expr = ASTRepr::from(expr.clone());

    // Generate and compile Rust code
    println!("🔧 Generating and compiling Rust code...");
    let codegen = RustCodeGenerator::new();
    let rust_code = codegen.generate_function(&ast_expr, "test_func")?;

    println!("Generated code:");
    println!("{rust_code}");
    println!();

    let compiler = RustCompiler::with_opt_level(RustOptLevel::O3)
        .with_extra_flags(hlist!["-C".to_string(), "target-cpu=native".to_string()]);

    let temp_dir = std::env::temp_dir();
    let source_path = temp_dir.join("dynamic_test.rs");
    let lib_path = temp_dir.join("libdynamic_test.so");

    std::fs::write(&source_path, &rust_code)?;
    compiler.compile_dylib(&rust_code, &source_path, &lib_path)?;

    // Load compiled function
    use dlopen2::raw::Library;
    let library = Library::open(&lib_path)?;
    let compiled_func: extern "C" fn(f64, f64) -> f64 = unsafe { library.symbol("test_func")? };

    println!("✅ Compilation complete!");
    println!();

    // Generate random test data to prevent constant propagation
    let mut rng = thread_rng();
    let iterations = 100_000;
    let test_data: Vec<(f64, f64)> = (0..iterations)
        .map(|_| (rng.gen_range(-10.0..10.0), rng.gen_range(-10.0..10.0)))
        .collect();

    println!("📊 Performance Comparison (with random inputs):");
    println!("===============================================");

    // 1. DynamicContext interpretation
    let start = Instant::now();
    let mut interp_sum = 0.0;
    for &(x, y) in &test_data {
        interp_sum += ctx.eval(&expr, &[x, y]);
    }
    let interp_time = start.elapsed();
    let interp_ns = interp_time.as_nanos() as f64 / iterations as f64;

    println!("DynamicContext (interpreted): {interp_ns:.1}ns per evaluation");

    // 2. Runtime compiled version
    let start = Instant::now();
    let mut compiled_sum = 0.0;
    for &(x, y) in &test_data {
        compiled_sum += compiled_func(x, y);
    }
    let compiled_time = start.elapsed();
    let compiled_ns = compiled_time.as_nanos() as f64 / iterations as f64;

    println!("DynamicContext (compiled):    {compiled_ns:.1}ns per evaluation");

    // 3. Native Rust baseline with same random data
    let start = Instant::now();
    let mut native_sum = 0.0;
    for &(x, y) in &test_data {
        native_sum += x * y + 42.0;
    }
    let native_time = start.elapsed();
    let native_ns = native_time.as_nanos() as f64 / iterations as f64;

    println!("Native Rust:                  {native_ns:.1}ns per evaluation");

    println!();
    println!("📈 Performance Analysis:");
    println!("========================");
    println!(
        "Interpretation overhead: +{:.1}ns ({:.1}x slower)",
        interp_ns - native_ns,
        interp_ns / native_ns
    );
    println!(
        "Compilation overhead:    +{:.1}ns ({:.1}x slower)",
        compiled_ns - native_ns,
        compiled_ns / native_ns
    );
    println!(
        "Compilation speedup:     {:.1}x faster than interpretation",
        interp_ns / compiled_ns
    );

    // Verify correctness (sums should be approximately equal)
    println!();
    println!("🔍 Correctness Check:");
    println!("=====================");
    println!("Interpreted sum:  {interp_sum:.2}");
    println!("Compiled sum:     {compiled_sum:.2}");
    println!("Native sum:       {native_sum:.2}");

    let max_diff = (interp_sum - native_sum)
        .abs()
        .max((compiled_sum - native_sum).abs());
    println!("Max difference:   {max_diff:.2e}");
    println!("Results match:    {}", max_diff < 1e-6);

    // Cleanup
    let _ = std::fs::remove_file(&source_path);
    let _ = std::fs::remove_file(&lib_path);

    Ok(())
}
