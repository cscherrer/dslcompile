//! Cranelift vs Rust Codegen Performance Benchmark
//!
//! This benchmark compares the execution performance of Cranelift JIT compilation
//! vs Rust codegen + dynamic loading for mathematical expressions.

use divan::Bencher;
use mathcompile::backends::{RustCodeGenerator, RustCompiler, CompiledRustFunction};
use mathcompile::final_tagless::{ASTRepr, ASTEval, ASTMathExpr};
use mathcompile::{OptimizationConfig, SymbolicOptimizer};

fn create_simple_expr() -> ASTRepr<f64> {
    // f(x) = x^2 + 2x + 1
    ASTEval::add(
        ASTEval::add(
            ASTEval::pow(ASTEval::var(0), ASTEval::constant(2.0)),
            ASTEval::mul(ASTEval::constant(2.0), ASTEval::var(0)),
        ),
        ASTEval::constant(1.0),
    )
}

fn create_medium_expr() -> ASTRepr<f64> {
    // f(x) = x^4 + 3x^3 + 2x^2 + x + 1
    ASTEval::add(
        ASTEval::add(
            ASTEval::add(
                ASTEval::add(
                    ASTEval::pow(ASTEval::var(0), ASTEval::constant(4.0)),
                    ASTEval::mul(
                        ASTEval::constant(3.0),
                        ASTEval::pow(ASTEval::var(0), ASTEval::constant(3.0)),
                    ),
                ),
                ASTEval::mul(
                    ASTEval::constant(2.0),
                    ASTEval::pow(ASTEval::var(0), ASTEval::constant(2.0)),
                ),
            ),
            ASTEval::var(0),
        ),
        ASTEval::constant(1.0),
    )
}

fn create_complex_expr() -> ASTRepr<f64> {
    // f(x) = sin(x^2) * exp(cos(x)) + ln(x + 1) * sqrt(x)
    ASTEval::add(
        ASTEval::mul(
            ASTEval::sin(ASTEval::pow(ASTEval::var(0), ASTEval::constant(2.0))),
            ASTEval::exp(ASTEval::cos(ASTEval::var(0))),
        ),
        ASTEval::mul(
            ASTEval::ln(ASTEval::add(ASTEval::var(0), ASTEval::constant(1.0))),
            ASTEval::sqrt(ASTEval::var(0)),
        ),
    )
}

fn compile_rust_function(expr: &ASTRepr<f64>) -> Result<CompiledRustFunction, Box<dyn std::error::Error>> {
    // Optimize the expression first
    let mut config = OptimizationConfig::default();
    config.egglog_optimization = true;
    config.constant_folding = true;
    let mut optimizer = SymbolicOptimizer::with_config(config)?;
    let optimized = optimizer.optimize(expr)?;

    // Generate and compile Rust code
    let codegen = RustCodeGenerator::new();
    let compiler = RustCompiler::new();
    
    let rust_source = codegen.generate_function(&optimized, "bench_func")?;
    let compiled_func = compiler.compile_and_load(&rust_source, "bench_func")?;
    
    Ok(compiled_func)
}

#[divan::bench]
fn rust_simple_execution(bencher: Bencher) {
    let expr = create_simple_expr();
    let compiled_func = compile_rust_function(&expr).expect("Failed to compile simple expression");
    let test_value = 2.5;

    bencher.bench_local(|| {
        compiled_func.call(test_value).expect("Function call failed")
    });
}

#[divan::bench]
fn rust_medium_execution(bencher: Bencher) {
    let expr = create_medium_expr();
    let compiled_func = compile_rust_function(&expr).expect("Failed to compile medium expression");
    let test_value = 2.5;

    bencher.bench_local(|| {
        compiled_func.call(test_value).expect("Function call failed")
    });
}

#[divan::bench]
fn rust_complex_execution(bencher: Bencher) {
    let expr = create_complex_expr();
    let compiled_func = compile_rust_function(&expr).expect("Failed to compile complex expression");
    let test_value = 2.5;

    bencher.bench_local(|| {
        compiled_func.call(test_value).expect("Function call failed")
    });
}

// Note: Cranelift benchmarks would be added here when the Cranelift backend is available
// For now, we focus on the Rust codegen performance

fn main() {
    println!("🔬 Rust Codegen Execution Performance Benchmark");
    println!("===============================================");
    println!();

    println!("📊 Expression Stats:");
    println!("  Simple:  Basic polynomial (x² + 2x + 1)");
    println!("  Medium:  Higher-order polynomial (x⁴ + 3x³ + 2x² + x + 1)");
    println!("  Complex: Transcendental functions (sin(x²) * exp(cos(x)) + ln(x+1) * √x)");
    println!();

    println!("🚀 Running execution benchmarks...");
    println!("   (Compilation costs excluded - measuring pure execution speed)");
    println!();

    divan::main();
}
