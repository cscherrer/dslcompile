//! Vec Lookup Elimination Benchmark
//!
//! Compares v4 (with Vec lookup) vs v5 (O(1) array access) to measure
//! the performance impact of eliminating the O(n) var_map Vec search.
//!
//! Performance target: Match old system's ~5.7ns scalar operations

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dslcompile::compile_time::heterogeneous_v4::{
    TrueZeroContext, TrueZeroInputs, true_zero_add, true_zero_array_index, TrueZeroExpr, TrueZeroArrayExpr
};
use dslcompile::compile_time::heterogeneous_v5::{
    UltimateZeroContext, UltimateZeroInputs, ultimate_zero_add, ultimate_zero_array_index, 
    UltimateZeroExpr, UltimateZeroArrayExpr
};

fn bench_v4_scalar_addition_vec_lookup(c: &mut Criterion) {
    let mut ctx = TrueZeroContext::<0>::new();
    let x = ctx.var::<f64>();
    let y = ctx.var::<f64>();
    let expr = true_zero_add::<f64, _, _, 0>(x, y);
    
    let mut inputs = TrueZeroInputs::new();
    inputs.add_f64(0, 3.0);
    inputs.add_f64(1, 4.0);
    
    c.bench_function("v4_scalar_addition_vec_lookup", |b| {
        b.iter(|| {
            black_box(expr.eval(&inputs))
        })
    });
}

fn bench_v5_scalar_addition_array_access(c: &mut Criterion) {
    let mut ctx = UltimateZeroContext::<0, 8>::new();
    let x = ctx.var::<f64>();
    let y = ctx.var::<f64>();
    let expr = ultimate_zero_add::<f64, _, _, 0>(x, y);
    
    let mut inputs = UltimateZeroInputs::<8>::new();
    inputs.add_f64(0, 3.0);
    inputs.add_f64(1, 4.0);
    
    c.bench_function("v5_scalar_addition_array_access", |b| {
        b.iter(|| {
            black_box(expr.eval(&inputs))
        })
    });
}

fn bench_v4_array_indexing_vec_lookup(c: &mut Criterion) {
    let mut ctx = TrueZeroContext::<0>::new();
    let array = ctx.var::<Vec<f64>>();
    let index = ctx.var::<usize>();
    let expr = true_zero_array_index::<f64, _, _, 0>(array, index);
    
    let mut inputs = TrueZeroInputs::new();
    inputs.add_vec_f64(0, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    inputs.add_usize(1, 2);
    
    c.bench_function("v4_array_indexing_vec_lookup", |b| {
        b.iter(|| {
            black_box(expr.eval_array(&inputs))
        })
    });
}

fn bench_v5_array_indexing_array_access(c: &mut Criterion) {
    let mut ctx = UltimateZeroContext::<0, 8>::new();
    let array = ctx.var::<Vec<f64>>();
    let index = ctx.var::<usize>();
    let expr = ultimate_zero_array_index::<f64, _, _, 0>(array, index);
    
    let mut inputs = UltimateZeroInputs::<8>::new();
    inputs.add_vec_f64(0, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    inputs.add_usize(1, 2);
    
    c.bench_function("v5_array_indexing_array_access", |b| {
        b.iter(|| {
            black_box(expr.eval_array(&inputs))
        })
    });
}

fn bench_v4_neural_network_vec_lookup(c: &mut Criterion) {
    let mut ctx = TrueZeroContext::<0>::new();
    let weights = ctx.var::<Vec<f64>>();
    let index = ctx.var::<usize>();
    let bias = ctx.var::<f64>();
    
    let indexed_weight = true_zero_array_index::<f64, _, _, 0>(weights, index);
    
    let mut inputs = TrueZeroInputs::new();
    inputs.add_vec_f64(0, vec![0.1, 0.2, 0.3, 0.4]);
    inputs.add_usize(1, 1);
    inputs.add_f64(2, 0.5);
    
    c.bench_function("v4_neural_network_vec_lookup", |b| {
        b.iter(|| {
            let weight_result = black_box(indexed_weight.eval_array(&inputs));
            let bias_result = black_box(bias.eval(&inputs));
            black_box(weight_result + bias_result)
        })
    });
}

fn bench_v5_neural_network_array_access(c: &mut Criterion) {
    let mut ctx = UltimateZeroContext::<0, 8>::new();
    let weights = ctx.var::<Vec<f64>>();
    let index = ctx.var::<usize>();
    let bias = ctx.var::<f64>();
    
    let indexed_weight = ultimate_zero_array_index::<f64, _, _, 0>(weights, index);
    
    let mut inputs = UltimateZeroInputs::<8>::new();
    inputs.add_vec_f64(0, vec![0.1, 0.2, 0.3, 0.4]);
    inputs.add_usize(1, 1);
    inputs.add_f64(2, 0.5);
    
    c.bench_function("v5_neural_network_array_access", |b| {
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

// Test scaling behavior with more variables
fn bench_v4_scaling_8_vars(c: &mut Criterion) {
    let mut ctx = TrueZeroContext::<0>::new();
    let vars: Vec<_> = (0..8).map(|_| ctx.var::<f64>()).collect();
    let expr = true_zero_add::<f64, _, _, 0>(vars[6].clone(), vars[7].clone());
    
    let mut inputs = TrueZeroInputs::new();
    for i in 0..8 {
        inputs.add_f64(i, i as f64);
    }
    
    c.bench_function("v4_scaling_8_vars_vec_lookup", |b| {
        b.iter(|| {
            black_box(expr.eval(&inputs))
        })
    });
}

fn bench_v5_scaling_8_vars(c: &mut Criterion) {
    let mut ctx = UltimateZeroContext::<0, 16>::new();
    let vars: Vec<_> = (0..8).map(|_| ctx.var::<f64>()).collect();
    let expr = ultimate_zero_add::<f64, _, _, 0>(vars[6].clone(), vars[7].clone());
    
    let mut inputs = UltimateZeroInputs::<16>::new();
    for i in 0..8 {
        inputs.add_f64(i, i as f64);
    }
    
    c.bench_function("v5_scaling_8_vars_array_access", |b| {
        b.iter(|| {
            black_box(expr.eval(&inputs))
        })
    });
}

criterion_group!(
    benches,
    bench_v4_scalar_addition_vec_lookup,
    bench_v5_scalar_addition_array_access,
    bench_v4_array_indexing_vec_lookup,
    bench_v5_array_indexing_array_access,
    bench_v4_neural_network_vec_lookup,
    bench_v5_neural_network_array_access,
    bench_v4_scaling_8_vars,
    bench_v5_scaling_8_vars,
    bench_direct_rust_baseline
);
criterion_main!(benches); 