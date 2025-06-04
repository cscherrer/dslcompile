//! Fixed Context System Performance Comparison
//!
//! This benchmark uses std::hint::black_box and improved measurement techniques
//! to get accurate performance data for both systems.

use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use dslcompile::compile_time::{
    // Old system
    Context, ScopedVarArray, ScopedMathExpr, compose,
    // New system  
    heterogeneous_v2::{HeteroContext, HeteroVar, HeteroInputs, HeteroEvaluator, array_index, scalar_add}
};

// ============================================================================
// FIXED OLD SYSTEM BENCHMARKS - More Realistic
// ============================================================================

fn old_system_expression_building(c: &mut Criterion) {
    c.bench_function("old_system_expression_building", |b| {
        b.iter(|| {
            let mut builder = Context::new_f64();
            
            // Build f(x, y) = x + y (focus on building, not evaluation)
            let expr = black_box(builder.new_scope(|scope| {
                let (x, scope) = scope.auto_var();
                let (y, _scope) = scope.auto_var();
                x.add(y)
            }));
            
            // Return the expression to prevent optimization
            black_box(expr)
        });
    });
}

fn old_system_expression_evaluation(c: &mut Criterion) {
    // Pre-build the expression outside the benchmark
    let mut builder = Context::new_f64();
    let expr = builder.new_scope(|scope| {
        let (x, scope) = scope.auto_var();
        let (y, _scope) = scope.auto_var();
        x.add(y)
    });

    c.bench_function("old_system_expression_evaluation", |b| {
        b.iter(|| {
            let vars = ScopedVarArray::<f64, 0>::new(vec![3.0, 4.0]);
            let result = black_box(expr.eval(&vars));
            black_box(result)
        });
    });
}

fn old_system_complex_expression_building(c: &mut Criterion) {
    c.bench_function("old_system_complex_expression_building", |b| {
        b.iter(|| {
            let mut builder = Context::new_f64();
            
            // Build quadratic: f(x, y) = x² + xy + y²
            let expr = black_box(builder.new_scope(|scope| {
                let (x, scope) = scope.auto_var();
                let (y, _scope) = scope.auto_var();
                x.clone().mul(x.clone())
                    .add(x.mul(y.clone()))
                    .add(y.clone().mul(y))
            }));
            
            black_box(expr)
        });
    });
}

fn old_system_complex_expression_evaluation(c: &mut Criterion) {
    // Pre-build the expression
    let mut builder = Context::new_f64();
    let expr = builder.new_scope(|scope| {
        let (x, scope) = scope.auto_var();
        let (y, _scope) = scope.auto_var();
        x.clone().mul(x.clone())
            .add(x.mul(y.clone()))
            .add(y.clone().mul(y))
    });

    c.bench_function("old_system_complex_expression_evaluation", |b| {
        b.iter(|| {
            let vars = ScopedVarArray::<f64, 0>::new(vec![3.0, 4.0]);
            let result = black_box(expr.eval(&vars));
            black_box(result)
        });
    });
}

// ============================================================================
// FIXED NEW SYSTEM BENCHMARKS - Equivalent Tests
// ============================================================================

fn new_system_expression_building(c: &mut Criterion) {
    c.bench_function("new_system_expression_building", |b| {
        b.iter(|| {
            let mut ctx: HeteroContext<0> = HeteroContext::new();
            
            // Build f(x, y) = x + y with heterogeneous types
            let x: HeteroVar<f64, 0> = ctx.var();
            let y: HeteroVar<f64, 0> = ctx.var();
            let expr = black_box(scalar_add(x, y));
            
            black_box(expr)
        });
    });
}

fn new_system_expression_evaluation(c: &mut Criterion) {
    // Pre-build the expression
    let mut ctx: HeteroContext<0> = HeteroContext::new();
    let x: HeteroVar<f64, 0> = ctx.var();
    let y: HeteroVar<f64, 0> = ctx.var();
    let expr = scalar_add(x, y);

    c.bench_function("new_system_expression_evaluation", |b| {
        b.iter(|| {
            let mut inputs = HeteroInputs::new();
            inputs.add_scalar_f64(0, 3.0);
            inputs.add_scalar_f64(1, 4.0);
            
            let evaluator: HeteroEvaluator<0> = HeteroEvaluator::new();
            let result = black_box(evaluator.eval_native(&expr, &inputs));
            black_box(result)
        });
    });
}

fn new_system_heterogeneous_building(c: &mut Criterion) {
    c.bench_function("new_system_heterogeneous_building", |b| {
        b.iter(|| {
            let mut ctx: HeteroContext<0> = HeteroContext::new();
            
            // Build heterogeneous expression: array[index] (impossible in old system)
            let array: HeteroVar<Vec<f64>, 0> = ctx.var();
            let index: HeteroVar<usize, 0> = ctx.var();
            let expr = black_box(array_index(array, index));
            
            black_box(expr)
        });
    });
}

fn new_system_heterogeneous_evaluation(c: &mut Criterion) {
    // Pre-build the expression
    let mut ctx: HeteroContext<0> = HeteroContext::new();
    let array: HeteroVar<Vec<f64>, 0> = ctx.var();
    let index: HeteroVar<usize, 0> = ctx.var();
    let expr = array_index(array, index);

    c.bench_function("new_system_heterogeneous_evaluation", |b| {
        b.iter(|| {
            let mut inputs = HeteroInputs::new();
            inputs.add_vec_f64(0, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
            inputs.add_usize(1, 2);
            
            let evaluator: HeteroEvaluator<0> = HeteroEvaluator::new();
            let result = black_box(evaluator.eval_native(&expr, &inputs));
            black_box(result)
        });
    });
}

// ============================================================================
// MEMORY OVERHEAD COMPARISON - The Key Metric
// ============================================================================

fn old_system_input_preparation(c: &mut Criterion) {
    c.bench_function("old_system_input_preparation", |b| {
        b.iter(|| {
            // Simulate the old system's requirement to flatten all inputs
            let inputs = black_box(vec![0.5, 0.8, 0.3, 0.9, 0.1]);
            let weights = black_box(vec![1.2, 0.7, 2.1, 0.4, 1.8]);
            let bias = black_box(0.5f64);
            let threshold = black_box(0.0f64);
            
            // Flatten everything into single Vec<f64> - this is the overhead!
            let mut flattened = Vec::with_capacity(inputs.len() + weights.len() + 2);
            flattened.extend_from_slice(&inputs);
            flattened.extend_from_slice(&weights);
            flattened.push(bias);
            flattened.push(threshold);
            
            black_box(flattened)
        });
    });
}

fn new_system_input_preparation(c: &mut Criterion) {
    c.bench_function("new_system_input_preparation", |b| {
        b.iter(|| {
            // New system: no flattening required, native types
            let inputs = black_box(vec![0.5, 0.8, 0.3, 0.9, 0.1]);
            let weights = black_box(vec![1.2, 0.7, 2.1, 0.4, 1.8]);
            let bias = black_box(0.5f64);
            let threshold = black_box(0.0f64);
            
            // No flattening needed - just return the native types
            black_box((inputs, weights, bias, threshold))
        });
    });
}

// ============================================================================
// DIRECT OPERATION COMPARISON  
// ============================================================================

fn direct_arithmetic_baseline(c: &mut Criterion) {
    c.bench_function("direct_arithmetic_baseline", |b| {
        b.iter(|| {
            let x = black_box(3.0f64);
            let y = black_box(4.0f64);
            let result = black_box(x + y);
            black_box(result)
        });
    });
}

fn direct_array_access_baseline(c: &mut Criterion) {
    c.bench_function("direct_array_access_baseline", |b| {
        b.iter(|| {
            let array = black_box(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
            let index = black_box(2usize);
            let result = black_box(array[index]);
            black_box(result)
        });
    });
}

criterion_group!(
    old_system_benches,
    old_system_expression_building,
    old_system_expression_evaluation,
    old_system_complex_expression_building,
    old_system_complex_expression_evaluation
);

criterion_group!(
    new_system_benches,
    new_system_expression_building,
    new_system_expression_evaluation,
    new_system_heterogeneous_building,
    new_system_heterogeneous_evaluation
);

criterion_group!(
    memory_benches,
    old_system_input_preparation,
    new_system_input_preparation
);

criterion_group!(
    baseline_benches,
    direct_arithmetic_baseline,
    direct_array_access_baseline
);

criterion_main!(old_system_benches, new_system_benches, memory_benches, baseline_benches); 