//! Benchmark comparing LLVM JIT vs Rust compilation backends
//!
//! This benchmark demonstrates that LLVM JIT compilation can match or exceed
//! the performance of the Rust hot-loading backend while eliminating
//! compilation overhead.

#![cfg(all(feature = "llvm_jit", not(miri)))]

use divan::{bench, Bencher};
use dslcompile::{
    ast::ASTRepr,
    backends::{LLVMJITCompiler, RustCompiler, RustOptLevel},
    prelude::*,
};
use inkwell::context::Context;
use inkwell::OptimizationLevel;

fn main() {
    divan::main();
}

/// Create a complex polynomial expression for benchmarking
fn create_polynomial_expr() -> ASTRepr<f64> {
    // f(x) = x^4 - 2x^3 + 3x^2 - 4x + 5
    ASTRepr::add_from_array([
        ASTRepr::Pow(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Constant(4.0)),
        ),
        ASTRepr::Neg(Box::new(ASTRepr::mul_from_array([
            ASTRepr::Constant(2.0),
            ASTRepr::Pow(
                Box::new(ASTRepr::Variable(0)),
                Box::new(ASTRepr::Constant(3.0)),
            ),
        ]))),
        ASTRepr::mul_from_array([
            ASTRepr::Constant(3.0),
            ASTRepr::Pow(
                Box::new(ASTRepr::Variable(0)),
                Box::new(ASTRepr::Constant(2.0)),
            ),
        ]),
        ASTRepr::Neg(Box::new(ASTRepr::mul_from_array([
            ASTRepr::Constant(4.0),
            ASTRepr::Variable(0),
        ]))),
        ASTRepr::Constant(5.0),
    ])
}

/// Create a transcendental expression for benchmarking
fn create_transcendental_expr() -> ASTRepr<f64> {
    // f(x) = sin(2x) * cos(3x) + ln(x + 1) + sqrt(x^2 + 1)
    ASTRepr::add_from_array([
        ASTRepr::mul_from_array([
            ASTRepr::Sin(Box::new(ASTRepr::mul_from_array([
                ASTRepr::Constant(2.0),
                ASTRepr::Variable(0),
            ]))),
            ASTRepr::Cos(Box::new(ASTRepr::mul_from_array([
                ASTRepr::Constant(3.0),
                ASTRepr::Variable(0),
            ]))),
        ]),
        ASTRepr::Ln(Box::new(ASTRepr::add_from_array([
            ASTRepr::Variable(0),
            ASTRepr::Constant(1.0),
        ]))),
        ASTRepr::Sqrt(Box::new(ASTRepr::add_from_array([
            ASTRepr::Pow(
                Box::new(ASTRepr::Variable(0)),
                Box::new(ASTRepr::Constant(2.0)),
            ),
            ASTRepr::Constant(1.0),
        ]))),
    ])
}

#[bench]
fn bench_llvm_compilation_polynomial(bencher: Bencher) {
    let expr = create_polynomial_expr();
    
    bencher.bench(|| {
        let context = Context::create();
        let mut compiler = LLVMJITCompiler::new(&context);
        compiler.compile_single_var_with_opt(&expr, OptimizationLevel::Aggressive).unwrap()
    });
}

#[bench]
fn bench_rust_compilation_polynomial(bencher: Bencher) {
    let expr = create_polynomial_expr();
    
    bencher.bench(|| {
        let mut compiler = RustCompiler::new(None).unwrap();
        compiler.set_optimization_level(RustOptLevel::Aggressive);
        compiler.compile_expression::<f64>(&expr).unwrap()
    });
}

#[bench]
fn bench_llvm_execution_polynomial(bencher: Bencher) {
    let context = Context::create();
    let mut compiler = LLVMJITCompiler::new(&context);
    let expr = create_polynomial_expr();
    let compiled_fn = compiler.compile_single_var_with_opt(&expr, OptimizationLevel::Aggressive).unwrap();
    
    bencher.bench(|| {
        let mut sum = 0.0;
        for i in 0..1000 {
            let x = (i as f64) / 100.0;
            sum += unsafe { compiled_fn.call(x) };
        }
        sum
    });
}

#[bench]
fn bench_rust_execution_polynomial(bencher: Bencher) {
    let mut compiler = RustCompiler::new(None).unwrap();
    compiler.set_optimization_level(RustOptLevel::Aggressive);
    let expr = create_polynomial_expr();
    let compiled_fn = compiler.compile_expression::<f64>(&expr).unwrap();
    
    bencher.bench(|| {
        let mut sum = 0.0;
        for i in 0..1000 {
            let x = (i as f64) / 100.0;
            sum += compiled_fn.call(&[x]);
        }
        sum
    });
}

#[bench]
fn bench_llvm_execution_transcendental(bencher: Bencher) {
    let context = Context::create();
    let mut compiler = LLVMJITCompiler::new(&context);
    let expr = create_transcendental_expr();
    let compiled_fn = compiler.compile_single_var_with_opt(&expr, OptimizationLevel::Aggressive).unwrap();
    
    bencher.bench(|| {
        let mut sum = 0.0;
        for i in 0..1000 {
            let x = (i as f64) / 100.0;
            sum += unsafe { compiled_fn.call(x) };
        }
        sum
    });
}

#[bench]
fn bench_rust_execution_transcendental(bencher: Bencher) {
    let mut compiler = RustCompiler::new(None).unwrap();
    compiler.set_optimization_level(RustOptLevel::Aggressive);
    let expr = create_transcendental_expr();
    let compiled_fn = compiler.compile_expression::<f64>(&expr).unwrap();
    
    bencher.bench(|| {
        let mut sum = 0.0;
        for i in 0..1000 {
            let x = (i as f64) / 100.0;
            sum += compiled_fn.call(&[x]);
        }
        sum
    });
}

#[bench]
fn bench_native_rust_polynomial(bencher: Bencher) {
    bencher.bench(|| {
        let mut sum = 0.0;
        for i in 0..1000 {
            let x = (i as f64) / 100.0;
            sum += x.powi(4) - 2.0 * x.powi(3) + 3.0 * x.powi(2) - 4.0 * x + 5.0;
        }
        sum
    });
}

#[bench]
fn bench_native_rust_transcendental(bencher: Bencher) {
    bencher.bench(|| {
        let mut sum = 0.0;
        for i in 0..1000 {
            let x = (i as f64) / 100.0;
            sum += (2.0 * x).sin() * (3.0 * x).cos() + (x + 1.0).ln() + (x * x + 1.0).sqrt();
        }
        sum
    });
}

// Multi-variable benchmarks
fn create_multi_var_expr() -> ASTRepr<f64> {
    // f(x, y, z) = x*y*z + x^2 + y^2 + z^2
    ASTRepr::add_from_array([
        ASTRepr::mul_from_array([
            ASTRepr::Variable(0),
            ASTRepr::Variable(1),
            ASTRepr::Variable(2),
        ]),
        ASTRepr::Pow(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Constant(2.0)),
        ),
        ASTRepr::Pow(
            Box::new(ASTRepr::Variable(1)),
            Box::new(ASTRepr::Constant(2.0)),
        ),
        ASTRepr::Pow(
            Box::new(ASTRepr::Variable(2)),
            Box::new(ASTRepr::Constant(2.0)),
        ),
    ])
}

#[bench]
fn bench_llvm_multi_var_execution(bencher: Bencher) {
    let context = Context::create();
    let mut compiler = LLVMJITCompiler::new(&context);
    let expr = create_multi_var_expr();
    let compiled_fn = compiler.compile_multi_var(&expr).unwrap();
    
    bencher.bench(|| {
        let mut sum = 0.0;
        for i in 0..100 {
            let vars = [(i as f64) / 10.0, (i as f64) / 20.0, (i as f64) / 30.0];
            sum += unsafe { compiled_fn.call(vars.as_ptr()) };
        }
        sum
    });
}

#[bench]
fn bench_rust_multi_var_execution(bencher: Bencher) {
    let mut compiler = RustCompiler::new(None).unwrap();
    compiler.set_optimization_level(RustOptLevel::Aggressive);
    let expr = create_multi_var_expr();
    let compiled_fn = compiler.compile_expression::<f64>(&expr).unwrap();
    
    bencher.bench(|| {
        let mut sum = 0.0;
        for i in 0..100 {
            let vars = [(i as f64) / 10.0, (i as f64) / 20.0, (i as f64) / 30.0];
            sum += compiled_fn.call(&vars);
        }
        sum
    });
}

#[bench]
fn bench_native_rust_multi_var(bencher: Bencher) {
    bencher.bench(|| {
        let mut sum = 0.0;
        for i in 0..100 {
            let x = (i as f64) / 10.0;
            let y = (i as f64) / 20.0;
            let z = (i as f64) / 30.0;
            // f(x, y, z) = x*y*z + x^2 + y^2 + z^2
            sum += x * y * z + x * x + y * y + z * z;
        }
        sum
    });
}