//! Cranelift JIT vs Rust Codegen Execution Performance Benchmark
//!
//! This benchmark compares PURE EXECUTION TIME (not compilation) between:
//! 1. Cranelift JIT compiled functions
//! 2. Rust hot-loading compiled functions
//!
//! Both backends are pre-compiled, then we measure only execution performance
//! to determine if Rust codegen consistently outperforms Cranelift for execution.

use divan::Bencher;
use dlopen2::raw::Library;
use dslcompile::ast::{ASTRepr, DynamicContext, VariableRegistry};
use dslcompile::backends::cranelift::{CompiledFunction, CraneliftCompiler};
use dslcompile::backends::{RustCodeGenerator, RustCompiler, RustOptLevel};
use dslcompile::{OptimizationConfig, SymbolicOptimizer};
use std::fs;

/// Compiled Rust function wrapper using dlopen2
struct CompiledRustFunction {
    _library: Library,
    function: extern "C" fn(f64) -> f64,
}

impl CompiledRustFunction {
    fn load(
        lib_path: &std::path::Path,
        func_name: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let library = Library::open(lib_path)?;
        let function: extern "C" fn(f64) -> f64 =
            unsafe { library.symbol::<extern "C" fn(f64) -> f64>(func_name)? };
        Ok(Self {
            _library: library,
            function,
        })
    }

    fn call(&self, x: f64) -> f64 {
        (self.function)(x)
    }
}

/// Test expressions of varying complexity
fn create_simple_expr() -> ASTRepr<f64> {
    // Test expression: x^2 + 2*x + 1
    let math = DynamicContext::new();
    let x = math.var();
    (&x * &x + 2.0 * &x + 1.0).into()
}

fn create_medium_expr() -> ASTRepr<f64> {
    // More complex expression: x^4 + 3*x^3 + 2*x^2 + x + 1
    let math = DynamicContext::new();
    let x = math.var();
    let x2 = x.clone().pow(math.constant(2.0));
    let x3 = x.clone().pow(math.constant(3.0));
    let x4 = x.clone().pow(math.constant(4.0));
    (x4 + 3.0 * x3 + 2.0 * x2 + &x + 1.0).into()
}

fn create_complex_expr() -> ASTRepr<f64> {
    // Transcendental expression: x * exp(cos(x)) + sqrt(x)
    let math = DynamicContext::new();
    let x = math.var();
    let cos_x = x.clone().cos();
    let exp_cos_x = cos_x.exp();
    let sqrt_x = x.clone().sqrt();
    (&x * exp_cos_x + sqrt_x).into()
}

/// Setup compiled functions for benchmarking
fn setup_functions(
    expr: &ASTRepr<f64>,
    func_name: &str,
) -> Result<(CompiledFunction, CompiledRustFunction), Box<dyn std::error::Error>> {
    // Optimize the expression first
    let mut config = OptimizationConfig::default();
    config.egglog_optimization = true;
    config.constant_folding = true;
    let mut optimizer = SymbolicOptimizer::with_config(config)?;
    let optimized = optimizer.optimize(expr)?;

    // Compile with Cranelift
    let mut compiler = CraneliftCompiler::new_default()?;
    let registry = VariableRegistry::for_expression(&optimized);
    let compiled_func = compiler.compile_expression(&optimized, &registry)?;

    // Compile with Rust
    let temp_dir = std::env::temp_dir().join("dslcompile_cranelift_vs_rust_bench");
    let source_dir = temp_dir.join("sources");
    let lib_dir = temp_dir.join("libs");

    fs::create_dir_all(&source_dir)?;
    fs::create_dir_all(&lib_dir)?;

    let codegen = RustCodeGenerator::new();
    let compiler = RustCompiler::with_opt_level(RustOptLevel::O2);

    let rust_source = codegen.generate_function(&optimized, func_name)?;
    let source_path = source_dir.join(format!("{func_name}.rs"));
    let lib_path = lib_dir.join(format!("lib{func_name}.so"));

    compiler.compile_dylib(&rust_source, &source_path, &lib_path)?;
    let rust_func = CompiledRustFunction::load(&lib_path, func_name)?;

    Ok((compiled_func, rust_func))
}

#[divan::bench]
fn simple_cranelift(bencher: Bencher) {
    let (cranelift_func, _) = setup_functions(&create_simple_expr(), "simple_func").unwrap();
    let test_value = 2.5;

    bencher.bench_local(|| cranelift_func.call(&[test_value]).unwrap());
}

#[divan::bench]
fn simple_rust(bencher: Bencher) {
    let (_, rust_func) = setup_functions(&create_simple_expr(), "simple_func_rust").unwrap();
    let test_value = 2.5;

    bencher.bench_local(|| rust_func.call(test_value));
}

#[divan::bench]
fn medium_cranelift(bencher: Bencher) {
    let (cranelift_func, _) = setup_functions(&create_medium_expr(), "medium_func").unwrap();
    let test_value = 2.5;

    bencher.bench_local(|| cranelift_func.call(&[test_value]).unwrap());
}

#[divan::bench]
fn medium_rust(bencher: Bencher) {
    let (_, rust_func) = setup_functions(&create_medium_expr(), "medium_func_rust").unwrap();
    let test_value = 2.5;

    bencher.bench_local(|| rust_func.call(test_value));
}

#[divan::bench]
fn complex_cranelift(bencher: Bencher) {
    let (cranelift_func, _) = setup_functions(&create_complex_expr(), "complex_func").unwrap();
    let test_value = 2.5;

    bencher.bench_local(|| cranelift_func.call(&[test_value]).unwrap());
}

#[divan::bench]
fn complex_rust(bencher: Bencher) {
    let (_, rust_func) = setup_functions(&create_complex_expr(), "complex_func_rust").unwrap();
    let test_value = 2.5;

    bencher.bench_local(|| rust_func.call(test_value));
}

fn main() {
    println!("🔬 Cranelift vs Rust Codegen Execution Performance");
    println!("==================================================");
    println!();

    println!("📊 Expression Stats:");
    println!(
        "  Simple:  {} operations",
        create_simple_expr().count_operations()
    );
    println!(
        "  Medium:  {} operations",
        create_medium_expr().count_operations()
    );
    println!(
        "  Complex: {} operations",
        create_complex_expr().count_operations()
    );
    println!();

    println!("🚀 Running execution benchmarks...");
    println!("   (Compilation costs excluded - measuring pure execution speed)");
    println!();

    divan::main();
}
