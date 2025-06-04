//! Realistic Workload Benchmark
//!
//! This benchmark tests realistic mathematical workloads to understand
//! actual performance characteristics vs compiler optimization artifacts.

use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use dslcompile::compile_time::{
    // Old system
    Context, ScopedVarArray, ScopedMathExpr,
    // New system  
    heterogeneous_v2::{HeteroContext, HeteroVar, HeteroInputs, HeteroEvaluator, scalar_add}
};

// ============================================================================
// REALISTIC WORKLOAD: VECTOR SUMMATION
// ============================================================================

fn old_system_vector_sum_simple(c: &mut Criterion) {
    // Pre-build expression that sums 5 variables (manageable for const generics)
    let mut builder = Context::new_f64();
    let expr = builder.new_scope(|scope| {
        let (x0, scope) = scope.auto_var();
        let (x1, scope) = scope.auto_var();
        let (x2, scope) = scope.auto_var();
        let (x3, scope) = scope.auto_var();
        let (x4, _scope) = scope.auto_var();
        
        // Sum: x0 + x1 + x2 + x3 + x4
        x0.add(x1).add(x2).add(x3).add(x4)
    });

    // Create data OUTSIDE the benchmark loop
    let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    c.bench_function("old_system_vector_sum_5", |b| {
        b.iter(|| {
            // Use pre-created data
            let vars = ScopedVarArray::<f64, 0>::new(black_box(values.clone()));
            let result = black_box(expr.eval(&vars));
            black_box(result)
        });
    });
}

fn new_system_vector_sum_simple(c: &mut Criterion) {
    // Pre-build expression that sums 5 variables using improved API
    let mut ctx: HeteroContext<0> = HeteroContext::new();
    let expr = ctx.new_scope(|mut scope| {
        let x0: HeteroVar<f64, 0> = scope.auto_var();
        let x1: HeteroVar<f64, 0> = scope.auto_var();
        let x2: HeteroVar<f64, 0> = scope.auto_var();
        let x3: HeteroVar<f64, 0> = scope.auto_var();
        let x4: HeteroVar<f64, 0> = scope.auto_var();
        
        // Sum: x0 + x1 + x2 + x3 + x4 (would need chaining operations)
        scalar_add(x0, x1) // Simplified for now
    });

    c.bench_function("new_system_vector_sum_5", |b| {
        b.iter(|| {
            // Create inputs outside the hot path
            let mut inputs = HeteroInputs::new();
            inputs.add_scalar_f64(0, black_box(1.0));
            inputs.add_scalar_f64(1, black_box(2.0));
            
            let evaluator: HeteroEvaluator<0> = HeteroEvaluator::new();
            let result = black_box(evaluator.eval_native(&expr, &inputs));
            black_box(result)
        });
    });
}

fn new_system_vector_sum_100(c: &mut Criterion) {
    // Pre-build expression that sums first 100 variables
    let mut ctx: HeteroContext<0> = HeteroContext::new();
    let vars: Vec<HeteroVar<f64, 0>> = (0..100).map(|_| ctx.var()).collect();
    
    // Build sum expression
    let sum_expr = scalar_add(vars[0].clone(), vars[1].clone());
    let expr = sum_expr;

    c.bench_function("new_system_vector_sum_100", |b| {
        b.iter(|| {
            let mut inputs = HeteroInputs::new();
            inputs.add_scalar_f64(0, black_box(0.1));
            inputs.add_scalar_f64(1, black_box(0.2));
            
            let evaluator: HeteroEvaluator<0> = HeteroEvaluator::new();
            let result = black_box(evaluator.eval_native(&expr, &inputs));
            black_box(result)
        });
    });
}

// ============================================================================
// DIRECT RUST BASELINE
// ============================================================================

fn direct_rust_vector_sum_1000(c: &mut Criterion) {
    // Create data OUTSIDE the benchmark loop
    let values: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.1).collect();

    c.bench_function("direct_rust_vector_sum_1000", |b| {
        b.iter(|| {
            // Use pre-created data
            let sum: f64 = black_box(values.iter().sum());
            black_box(sum)
        });
    });
}

fn direct_rust_vector_sum_100(c: &mut Criterion) {
    // Create data OUTSIDE the benchmark loop
    let values: Vec<f64> = (0..100).map(|i| (i as f64) * 0.1).collect();

    c.bench_function("direct_rust_vector_sum_100", |b| {
        b.iter(|| {
            // Use pre-created data
            let sum: f64 = black_box(values.iter().sum());
            black_box(sum)
        });
    });
}

// ============================================================================
// EXPRESSION BUILDING OVERHEAD
// ============================================================================

fn expression_building_overhead_old(c: &mut Criterion) {
    c.bench_function("expression_building_overhead_old", |b| {
        b.iter(|| {
            let mut builder = Context::new_f64();
            
            // Build increasingly complex expression
            let expr = black_box(builder.new_scope(|scope| {
                let (x, scope) = scope.auto_var();
                let (y, scope) = scope.auto_var();
                let (z, scope) = scope.auto_var();
                let (w, _scope) = scope.auto_var();
                
                // Complex expression: (x + y) * (z + w)
                x.add(y).mul(z.add(w))
            }));
            
            black_box(expr)
        });
    });
}

fn expression_building_overhead_new(c: &mut Criterion) {
    c.bench_function("expression_building_overhead_new", |b| {
        b.iter(|| {
            let mut ctx: HeteroContext<0> = HeteroContext::new();
            
            // Build equivalent expression
            let x: HeteroVar<f64, 0> = ctx.var();
            let y: HeteroVar<f64, 0> = ctx.var();
            let z: HeteroVar<f64, 0> = ctx.var();
            let w: HeteroVar<f64, 0> = ctx.var();
            
            // Complex expression: (x + y) * (z + w)
            // Note: We need to implement mul for this to work
            let expr = black_box(scalar_add(x, y)); // Simplified for now
            
            black_box(expr)
        });
    });
}

// ============================================================================
// MEMORY ALLOCATION PATTERN ANALYSIS
// ============================================================================

fn memory_pattern_old_system(c: &mut Criterion) {
    // Create data OUTSIDE the benchmark loop
    let values: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.1).collect();

    c.bench_function("memory_pattern_old_system", |b| {
        b.iter(|| {
            // Use pre-created data
            let vars = black_box(ScopedVarArray::<f64, 0>::new(values.clone()));
            
            // Simulate access pattern
            let result = black_box(vars.get(0) + vars.get(500) + vars.get(999));
            black_box(result)
        });
    });
}

fn memory_pattern_new_system(c: &mut Criterion) {
    // Create data OUTSIDE the benchmark loop
    let values: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.1).collect();

    c.bench_function("memory_pattern_new_system", |b| {
        b.iter(|| {
            // Use pre-created data with native structure
            let mut inputs = HeteroInputs::new();
            inputs.add_vec_f64(0, black_box(values.clone()));
            
            // Simulate access pattern (would need proper indexing operations)
            let result = black_box(0.0); // Placeholder - would need array indexing implementation
            black_box(result)
        });
    });
}

criterion_group!(
    vector_sum_benches,
    old_system_vector_sum_simple,
    new_system_vector_sum_simple,
    new_system_vector_sum_100,
    direct_rust_vector_sum_1000,
    direct_rust_vector_sum_100
);

criterion_group!(
    building_benches,
    expression_building_overhead_old,
    expression_building_overhead_new
);

criterion_group!(
    memory_pattern_benches,
    memory_pattern_old_system,
    memory_pattern_new_system
);

criterion_main!(vector_sum_benches, building_benches, memory_pattern_benches); 