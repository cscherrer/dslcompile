//! Cranelift JIT vs Rust Codegen Execution Performance Benchmark
//!
//! This benchmark compares PURE EXECUTION TIME (not compilation) between:
//! 1. Cranelift JIT compiled functions
//! 2. Rust hot-loading compiled functions
//!
//! Both backends are pre-compiled, then we measure only execution performance
//! to determine if Rust codegen consistently outperforms Cranelift for execution.

use divan::Bencher;
use libloading::{Library, Symbol};
use mathjit::backends::cranelift::JITCompiler;
use mathjit::backends::{RustCodeGenerator, RustCompiler, RustOptLevel};
use mathjit::final_tagless::{JITEval, JITMathExpr};
use mathjit::symbolic::{OptimizationConfig, SymbolicOptimizer};
use std::fs;

/// Compiled Rust function wrapper
struct CompiledRustFunction {
    _library: Library,
    function: Symbol<'static, extern "C" fn(f64) -> f64>,
}

impl CompiledRustFunction {
    unsafe fn load(
        lib_path: &std::path::Path,
        func_name: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let library = Library::new(lib_path)?;
        let function: Symbol<extern "C" fn(f64) -> f64> = library.get(func_name.as_bytes())?;
        let function = std::mem::transmute(function);
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
fn create_simple_expr() -> mathjit::final_tagless::JITRepr<f64> {
    // f(x) = x^2 + 2x + 1
    JITEval::add(
        JITEval::add(
            JITEval::pow(JITEval::var("x"), JITEval::constant(2.0)),
            JITEval::mul(JITEval::constant(2.0), JITEval::var("x")),
        ),
        JITEval::constant(1.0),
    )
}

fn create_medium_expr() -> mathjit::final_tagless::JITRepr<f64> {
    // f(x) = x^4 + 3x^3 + 2x^2 + x + 1
    JITEval::add(
        JITEval::add(
            JITEval::add(
                JITEval::add(
                    JITEval::pow(JITEval::var("x"), JITEval::constant(4.0)),
                    JITEval::mul(
                        JITEval::constant(3.0),
                        JITEval::pow(JITEval::var("x"), JITEval::constant(3.0)),
                    ),
                ),
                JITEval::mul(
                    JITEval::constant(2.0),
                    JITEval::pow(JITEval::var("x"), JITEval::constant(2.0)),
                ),
            ),
            JITEval::var("x"),
        ),
        JITEval::constant(1.0),
    )
}

fn create_complex_expr() -> mathjit::final_tagless::JITRepr<f64> {
    // f(x) = sin(x^2) * exp(cos(x)) + ln(x + 1) * sqrt(x)
    JITEval::add(
        JITEval::mul(
            JITEval::sin(JITEval::pow(JITEval::var("x"), JITEval::constant(2.0))),
            JITEval::exp(JITEval::cos(JITEval::var("x"))),
        ),
        JITEval::mul(
            JITEval::ln(JITEval::add(JITEval::var("x"), JITEval::constant(1.0))),
            JITEval::sqrt(JITEval::var("x")),
        ),
    )
}

/// Setup compiled functions for benchmarking
fn setup_functions(
    expr: &mathjit::final_tagless::JITRepr<f64>,
    func_name: &str,
) -> Result<
    (
        mathjit::backends::cranelift::JITFunction,
        CompiledRustFunction,
    ),
    Box<dyn std::error::Error>,
> {
    // Optimize the expression first
    let mut config = OptimizationConfig::default();
    config.egglog_optimization = true;
    config.constant_folding = true;
    let mut optimizer = SymbolicOptimizer::with_config(config)?;
    let optimized = optimizer.optimize(expr)?;

    // Compile with Cranelift
    let jit_compiler = JITCompiler::new()?;
    let cranelift_func = jit_compiler.compile_single_var(&optimized, "x")?;

    // Compile with Rust
    let temp_dir = std::env::temp_dir().join("mathjit_cranelift_vs_rust_bench");
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
    let rust_func = unsafe { CompiledRustFunction::load(&lib_path, func_name)? };

    Ok((cranelift_func, rust_func))
}

#[divan::bench]
fn simple_cranelift(bencher: Bencher) {
    let (cranelift_func, _) = setup_functions(&create_simple_expr(), "simple_func").unwrap();
    let test_value = 2.5;

    bencher.bench_local(|| cranelift_func.call_single(test_value));
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

    bencher.bench_local(|| cranelift_func.call_single(test_value));
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

    bencher.bench_local(|| cranelift_func.call_single(test_value));
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
