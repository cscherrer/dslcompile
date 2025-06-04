//! Zero-Overhead Heterogeneous System Benchmark
//!
//! This benchmark compares the old system with the new zero-overhead heterogeneous system
//! to verify we've eliminated HashMap lookups and runtime type dispatch.

use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use dslcompile::compile_time::{
    // Old system
    Context, ScopedVarArray, ScopedMathExpr,
    // Zero-overhead heterogeneous system
    heterogeneous_v3::{ZeroContext, ZeroVar, ZeroInputs, zero_add, zero_array_index, ZeroExpr}
};

// ============================================================================
// OLD SYSTEM BENCHMARKS
// ============================================================================

fn old_system_scalar_addition(c: &mut Criterion) {
    // Pre-build expression
    let mut builder = Context::new_f64();
    let expr = builder.new_scope(|scope| {
        let (x, scope) = scope.auto_var();
        let (y, _scope) = scope.auto_var();
        x.add(y)
    });

    // Pre-create data
    let values = vec![3.0, 4.0];

    c.bench_function("old_system_scalar_addition", |b| {
        b.iter(|| {
            let vars = ScopedVarArray::<f64, 0>::new(black_box(values.clone()));
            let result = black_box(expr.eval(&vars));
            black_box(result)
        });
    });
}

fn old_system_array_indexing(c: &mut Criterion) {
    // The old system can't do native array indexing, so we simulate it
    // by putting array elements in flat Vec<f64> and accessing by index
    let mut builder = Context::new_f64();
    let expr = builder.new_scope(|scope| {
        // Simulate array[2] by accessing the 2nd variable
        let (elem0, scope) = scope.auto_var();
        let (elem1, scope) = scope.auto_var();
        let (target_elem, _scope) = scope.auto_var(); // This is our "indexed" element
        target_elem  // Return the element we want
    });

    // Pre-create data - flattened array
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    c.bench_function("old_system_array_indexing", |b| {
        b.iter(|| {
            let vars = ScopedVarArray::<f64, 0>::new(black_box(values.clone()));
            let result = black_box(expr.eval(&vars));
            black_box(result)
        });
    });
}

// ============================================================================
// ZERO-OVERHEAD SYSTEM BENCHMARKS
// ============================================================================

fn zero_system_scalar_addition(c: &mut Criterion) {
    // Pre-build expression
    let mut ctx: ZeroContext<0> = ZeroContext::new();
    let x: ZeroVar<f64, 0> = ctx.var();
    let y: ZeroVar<f64, 0> = ctx.var();
    let expr = zero_add::<f64, _, _, 0>(x, y);

    c.bench_function("zero_system_scalar_addition", |b| {
        b.iter(|| {
            let mut inputs = ZeroInputs::new();
            inputs.add_f64(0, black_box(3.0));
            inputs.add_f64(1, black_box(4.0));
            
            let result = black_box(expr.eval(&inputs));
            black_box(result)
        });
    });
}

fn zero_system_array_indexing(c: &mut Criterion) {
    // Pre-build expression with NATIVE array indexing
    let mut ctx: ZeroContext<0> = ZeroContext::new();
    let array: ZeroVar<Vec<f64>, 0> = ctx.var();
    let index: ZeroVar<usize, 0> = ctx.var();
    let expr = zero_array_index::<f64, _, _, 0>(array, index);

    // Pre-create data - native Vec<f64>
    let array_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    c.bench_function("zero_system_array_indexing", |b| {
        b.iter(|| {
            let mut inputs = ZeroInputs::new();
            inputs.add_vec_f64(0, black_box(array_data.clone()));
            inputs.add_usize(1, black_box(2));
            
            let result = black_box(expr.eval(&inputs));
            black_box(result)
        });
    });
}

fn zero_system_neural_network(c: &mut Criterion) {
    // Pre-build complex expression: weights[index] + bias
    let mut ctx: ZeroContext<0> = ZeroContext::new();
    let weights: ZeroVar<Vec<f64>, 0> = ctx.var();
    let index: ZeroVar<usize, 0> = ctx.var();
    let bias: ZeroVar<f64, 0> = ctx.var();
    
    let indexed_weight = zero_array_index::<f64, _, _, 0>(weights, index);
    let expr = zero_add::<f64, _, _, 0>(indexed_weight, bias);

    // Pre-create data
    let weights_data = vec![0.1, 0.2, 0.3, 0.4, 0.5];

    c.bench_function("zero_system_neural_network", |b| {
        b.iter(|| {
            let mut inputs = ZeroInputs::new();
            inputs.add_vec_f64(0, black_box(weights_data.clone()));
            inputs.add_usize(1, black_box(2));
            inputs.add_f64(2, black_box(0.5));
            
            let result = black_box(expr.eval(&inputs));
            black_box(result)
        });
    });
}

// ============================================================================
// DIRECT RUST BASELINES
// ============================================================================

fn direct_rust_scalar_addition(c: &mut Criterion) {
    c.bench_function("direct_rust_scalar_addition", |b| {
        b.iter(|| {
            let x = black_box(3.0f64);
            let y = black_box(4.0f64);
            let result = black_box(x + y);
            black_box(result)
        });
    });
}

fn direct_rust_array_indexing(c: &mut Criterion) {
    let array = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    c.bench_function("direct_rust_array_indexing", |b| {
        b.iter(|| {
            let index = black_box(2usize);
            let result = black_box(array[index]);
            black_box(result)
        });
    });
}

fn direct_rust_neural_network(c: &mut Criterion) {
    let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5];

    c.bench_function("direct_rust_neural_network", |b| {
        b.iter(|| {
            let index = black_box(2usize);
            let bias = black_box(0.5f64);
            let result = black_box(weights[index] + bias);
            black_box(result)
        });
    });
}

criterion_group!(
    old_system_benches,
    old_system_scalar_addition,
    old_system_array_indexing
);

criterion_group!(
    zero_system_benches,
    zero_system_scalar_addition,
    zero_system_array_indexing,
    zero_system_neural_network
);

criterion_group!(
    direct_rust_benches,
    direct_rust_scalar_addition,
    direct_rust_array_indexing,
    direct_rust_neural_network
);

criterion_main!(old_system_benches, zero_system_benches, direct_rust_benches); 