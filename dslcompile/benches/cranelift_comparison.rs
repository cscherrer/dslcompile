//! Cranelift JIT Backend Benchmarks
//!
//! This benchmark tests the modern Cranelift implementation performance
//! across different optimization levels and expression complexities.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use dslcompile::backends::cranelift::{CraneliftCompiler, OptimizationLevel};
use dslcompile::final_tagless::{ASTEval, VariableRegistry};
use std::time::Instant;

/// Create a simple expression: x^2 + 2*x + 1
fn create_simple_expr() -> (dslcompile::final_tagless::ASTRepr<f64>, VariableRegistry) {
    let mut registry = VariableRegistry::new();
    let x_idx = registry.register_variable();

    let expr = ASTEval::add(
        ASTEval::add(
            ASTEval::pow(ASTEval::var(x_idx), ASTEval::constant(2.0)),
            ASTEval::mul(ASTEval::constant(2.0), ASTEval::var(x_idx)),
        ),
        ASTEval::constant(1.0),
    );

    (expr, registry)
}

/// Create a complex expression: sin(x) * cos(y) + exp(x*y) / sqrt(x^2 + y^2)
fn create_complex_expr() -> (dslcompile::final_tagless::ASTRepr<f64>, VariableRegistry) {
    let mut registry = VariableRegistry::new();
    let x_idx = registry.register_variable();
    let y_idx = registry.register_variable();

    let x = ASTEval::var(x_idx);
    let y = ASTEval::var(y_idx);

    let expr = ASTEval::add(
        ASTEval::mul(ASTEval::sin(x.clone()), ASTEval::cos(y.clone())),
        ASTEval::div(
            ASTEval::exp(ASTEval::mul(x.clone(), y.clone())),
            ASTEval::sqrt(ASTEval::add(
                ASTEval::pow(x, ASTEval::constant(2.0)),
                ASTEval::pow(y, ASTEval::constant(2.0)),
            )),
        ),
    );

    (expr, registry)
}

/// Benchmark compilation time for simple expressions
fn bench_simple_compilation(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_compilation");

    let (expr, registry) = create_simple_expr();

    group.bench_function("cranelift_basic", |b| {
        b.iter(|| {
            let mut compiler = CraneliftCompiler::new(OptimizationLevel::Basic).unwrap();
            let _compiled = compiler
                .compile_expression(black_box(&expr), black_box(&registry))
                .unwrap();
        });
    });

    group.bench_function("cranelift_full", |b| {
        b.iter(|| {
            let mut compiler = CraneliftCompiler::new(OptimizationLevel::Full).unwrap();
            let _compiled = compiler
                .compile_expression(black_box(&expr), black_box(&registry))
                .unwrap();
        });
    });

    group.finish();
}

/// Benchmark compilation time for complex expressions
fn bench_complex_compilation(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_compilation");

    let (expr, registry) = create_complex_expr();

    group.bench_function("cranelift_basic", |b| {
        b.iter(|| {
            let mut compiler = CraneliftCompiler::new(OptimizationLevel::Basic).unwrap();
            let _compiled = compiler
                .compile_expression(black_box(&expr), black_box(&registry))
                .unwrap();
        });
    });

    group.bench_function("cranelift_full", |b| {
        b.iter(|| {
            let mut compiler = CraneliftCompiler::new(OptimizationLevel::Full).unwrap();
            let _compiled = compiler
                .compile_expression(black_box(&expr), black_box(&registry))
                .unwrap();
        });
    });

    group.finish();
}

/// Benchmark execution performance for simple expressions
fn bench_simple_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_execution");

    let (expr, registry) = create_simple_expr();

    // Pre-compile functions
    let mut compiler = CraneliftCompiler::new(OptimizationLevel::Full).unwrap();
    let func = compiler.compile_expression(&expr, &registry).unwrap();

    let test_value = 2.5;

    group.bench_function("cranelift_full", |b| {
        b.iter(|| func.call(black_box(&[test_value])).unwrap());
    });

    group.finish();
}

/// Benchmark execution performance for complex expressions
fn bench_complex_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_execution");

    let (expr, registry) = create_complex_expr();

    // Pre-compile functions
    let mut compiler = CraneliftCompiler::new(OptimizationLevel::Full).unwrap();
    let func = compiler.compile_expression(&expr, &registry).unwrap();

    let test_values = [1.5, 2.0];

    group.bench_function("cranelift_full", |b| {
        b.iter(|| func.call(black_box(&test_values)).unwrap());
    });

    group.finish();
}

/// Benchmark integer power optimization
fn bench_integer_power_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("integer_power_optimization");

    let mut registry = VariableRegistry::new();
    let x_idx = registry.register_variable();

    // Test x^8 - should benefit from binary exponentiation
    let expr = ASTEval::pow(ASTEval::var(x_idx), ASTEval::constant(8.0));

    // Pre-compile functions
    let mut compiler = CraneliftCompiler::new(OptimizationLevel::Full).unwrap();
    let func = compiler.compile_expression(&expr, &registry).unwrap();

    let test_value = 1.5;

    group.bench_function("cranelift_optimized", |b| {
        b.iter(|| func.call(black_box(&[test_value])).unwrap());
    });

    group.finish();
}

/// Demonstrate compilation metadata improvements
fn demonstrate_metadata_improvements() {
    println!("\n=== Cranelift Implementation Performance ===\n");

    let (simple_expr, simple_registry) = create_simple_expr();
    let (_complex_expr, _complex_registry) = create_complex_expr();

    // Modern implementation
    println!("Cranelift Implementation:");
    let start = Instant::now();
    let mut compiler = CraneliftCompiler::new(OptimizationLevel::Full).unwrap();
    let func = compiler
        .compile_expression(&simple_expr, &simple_registry)
        .unwrap();
    let compile_time = start.elapsed();

    println!("  Simple expr compile time: {compile_time:?}");
    println!("  Metadata: {:?}", func.metadata());
    println!("  Signature: {:?}", func.signature());

    // Test correctness
    let test_value = 3.0;
    let result = func.call(&[test_value]).unwrap();

    println!("\nCorrectness Test (x = {test_value}):");
    println!("  Result: {result}");
    println!(
        "  Expected: {}",
        (test_value * test_value + 2.0 * test_value + 1.0)
    );
    println!(
        "  Results match: {}",
        (result - (test_value * test_value + 2.0 * test_value + 1.0)).abs() < 1e-10
    );
}

criterion_group!(
    benches,
    bench_simple_compilation,
    bench_complex_compilation,
    bench_simple_execution,
    bench_complex_execution,
    bench_integer_power_optimization
);

criterion_main!(benches);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_improvements() {
        demonstrate_metadata_improvements();
    }

    #[test]
    fn test_correctness_comparison() {
        let (expr, registry) = create_simple_expr();

        // Compile with V2 implementation
        let mut compiler = CraneliftCompiler::new(OptimizationLevel::Basic).unwrap();
        let func = compiler.compile_expression(&expr, &registry).unwrap();

        // Test multiple values
        for test_value in [0.0, 1.0, 2.5, -1.5, 10.0] {
            let result = func.call(&[test_value]).unwrap();
            let expected = test_value * test_value + 2.0 * test_value + 1.0;

            assert!(
                (result - expected).abs() < 1e-10,
                "V2 result mismatch for x={test_value}: got {result}, expected {expected}"
            );
        }
    }
}
