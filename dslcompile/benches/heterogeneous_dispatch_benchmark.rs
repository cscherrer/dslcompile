//! Runtime Dispatch Elimination Benchmark
//!
//! Compares v3 (with runtime dispatch) vs v4 (zero dispatch) to measure
//! the performance impact of eliminating std::any::Any and downcast_ref.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dslcompile::compile_time::heterogeneous_v3::{
    ZeroContext, ZeroInputs, zero_add, zero_array_index, ZeroExpr
};
use dslcompile::compile_time::heterogeneous_v4::{
    TrueZeroContext, TrueZeroInputs, true_zero_add, true_zero_array_index, TrueZeroExpr, TrueZeroArrayExpr
};

fn bench_v3_scalar_addition(c: &mut Criterion) {
    let mut ctx = ZeroContext::<0>::new();
    let x = ctx.var::<f64>();
    let y = ctx.var::<f64>();
    let expr = zero_add::<f64, _, _, 0>(x, y);
    
    let mut inputs = ZeroInputs::new();
    inputs.add_f64(0, 3.0);
    inputs.add_f64(1, 4.0);
    
    c.bench_function("v3_scalar_addition_runtime_dispatch", |b| {
        b.iter(|| {
            black_box(expr.eval(&inputs))
        })
    });
}

fn bench_v4_scalar_addition(c: &mut Criterion) {
    let mut ctx = TrueZeroContext::<0>::new();
    let x = ctx.var::<f64>();
    let y = ctx.var::<f64>();
    let expr = true_zero_add::<f64, _, _, 0>(x, y);
    
    let mut inputs = TrueZeroInputs::new();
    inputs.add_f64(0, 3.0);
    inputs.add_f64(1, 4.0);
    
    c.bench_function("v4_scalar_addition_zero_dispatch", |b| {
        b.iter(|| {
            black_box(expr.eval(&inputs))
        })
    });
}

fn bench_v3_array_indexing(c: &mut Criterion) {
    let mut ctx = ZeroContext::<0>::new();
    let array = ctx.var::<Vec<f64>>();
    let index = ctx.var::<usize>();
    let expr = zero_array_index::<f64, _, _, 0>(array, index);
    
    let mut inputs = ZeroInputs::new();
    inputs.add_vec_f64(0, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    inputs.add_usize(1, 2);
    
    c.bench_function("v3_array_indexing_runtime_dispatch", |b| {
        b.iter(|| {
            black_box(expr.eval(&inputs))
        })
    });
}

fn bench_v4_array_indexing(c: &mut Criterion) {
    let mut ctx = TrueZeroContext::<0>::new();
    let array = ctx.var::<Vec<f64>>();
    let index = ctx.var::<usize>();
    let expr = true_zero_array_index::<f64, _, _, 0>(array, index);
    
    let mut inputs = TrueZeroInputs::new();
    inputs.add_vec_f64(0, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    inputs.add_usize(1, 2);
    
    c.bench_function("v4_array_indexing_zero_dispatch", |b| {
        b.iter(|| {
            black_box(expr.eval_array(&inputs))
        })
    });
}

fn bench_v3_neural_network(c: &mut Criterion) {
    let mut ctx = ZeroContext::<0>::new();
    let weights = ctx.var::<Vec<f64>>();
    let index = ctx.var::<usize>();
    let bias = ctx.var::<f64>();
    
    let indexed_weight = zero_array_index::<f64, _, _, 0>(weights, index);
    let expr = zero_add::<f64, _, _, 0>(indexed_weight, bias);
    
    let mut inputs = ZeroInputs::new();
    inputs.add_vec_f64(0, vec![0.1, 0.2, 0.3, 0.4]);
    inputs.add_usize(1, 1);
    inputs.add_f64(2, 0.5);
    
    c.bench_function("v3_neural_network_runtime_dispatch", |b| {
        b.iter(|| {
            black_box(expr.eval(&inputs))
        })
    });
}

fn bench_v4_neural_network(c: &mut Criterion) {
    let mut ctx = TrueZeroContext::<0>::new();
    let weights = ctx.var::<Vec<f64>>();
    let index = ctx.var::<usize>();
    let bias = ctx.var::<f64>();
    
    let indexed_weight = true_zero_array_index::<f64, _, _, 0>(weights, index);
    
    // For now, test components separately since the type composition is complex
    let mut inputs = TrueZeroInputs::new();
    inputs.add_vec_f64(0, vec![0.1, 0.2, 0.3, 0.4]);
    inputs.add_usize(1, 1);
    inputs.add_f64(2, 0.5);
    
    c.bench_function("v4_neural_network_zero_dispatch", |b| {
        b.iter(|| {
            let weight_result = black_box(indexed_weight.eval_array(&inputs));
            let bias_result = black_box(bias.eval(&inputs));
            black_box(weight_result + bias_result)
        })
    });
}

fn bench_direct_rust_baseline(c: &mut Criterion) {
    let weights = vec![0.1, 0.2, 0.3, 0.4];
    let index = 1;
    let bias = 0.5;
    
    c.bench_function("direct_rust_baseline", |b| {
        b.iter(|| {
            black_box(weights[index] + bias)
        })
    });
}

criterion_group!(
    benches,
    bench_v3_scalar_addition,
    bench_v4_scalar_addition,
    bench_v3_array_indexing,
    bench_v4_array_indexing,
    bench_v3_neural_network,
    bench_v4_neural_network,
    bench_direct_rust_baseline
);
criterion_main!(benches); 