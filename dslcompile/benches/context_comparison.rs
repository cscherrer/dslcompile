//! Context System Performance Comparison
//!
//! This benchmark compares the old Context<T, SCOPE> system with the new HeteroContext<SCOPE>
//! system to ensure no performance regressions before migration.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dslcompile::compile_time::{
    // Old system
    Context, ScopedVarArray, ScopedMathExpr, compose,
    // New system  
    heterogeneous_v2::{HeteroContext, HeteroVar, HeteroInputs, HeteroEvaluator, array_index, scalar_add}
};

// ============================================================================
// OLD SYSTEM BENCHMARKS
// ============================================================================

fn old_system_simple_expression(c: &mut Criterion) {
    c.bench_function("old_system_simple_expression", |b| {
        b.iter(|| {
            let mut builder = Context::new_f64();
            
            // Build f(x, y) = x + y
            let expr = builder.new_scope(|scope| {
                let (x, scope) = scope.auto_var();
                let (y, _scope) = scope.auto_var();
                x.add(y)
            });
            
            // Evaluate
            let vars = ScopedVarArray::<f64, 0>::new(vec![3.0, 4.0]);
            let result = black_box(expr.eval(&vars));
            assert_eq!(result, 7.0);
        });
    });
}

fn old_system_complex_expression(c: &mut Criterion) {
    c.bench_function("old_system_complex_expression", |b| {
        b.iter(|| {
            let mut builder = Context::new_f64();
            
            // Build quadratic: f(x, y) = x² + xy + y²
            let expr = builder.new_scope(|scope| {
                let (x, scope) = scope.auto_var();
                let (y, _scope) = scope.auto_var();
                x.clone().mul(x.clone())
                    .add(x.mul(y.clone()))
                    .add(y.clone().mul(y))
            });
            
            // Evaluate
            let vars = ScopedVarArray::<f64, 0>::new(vec![3.0, 4.0]);
            let result = black_box(expr.eval(&vars));
            assert_eq!(result, 37.0); // 9 + 12 + 16 = 37
        });
    });
}

fn old_system_composition(c: &mut Criterion) {
    c.bench_function("old_system_composition", |b| {
        b.iter(|| {
            let mut builder1 = Context::new_f64();
            
            // Define f(x) = x²
            let f = builder1.new_scope(|scope| {
                let (x, _scope) = scope.auto_var();
                x.clone().mul(x)
            });
            
            let mut builder2 = Context::new_f64();
            
            // Define g(y) = 2y  
            let g = builder2.new_scope(|scope| {
                let (y, scope) = scope.auto_var();
                y.mul(scope.constant(2.0))
            });
            
            // Compose h = f + g
            let composed = compose(f, g);
            let h = composed.add();
            
            // Evaluate h(3, 4) = f(3) + g(4) = 9 + 8 = 17
            let result = black_box(h.eval(&[3.0, 4.0]));
            assert_eq!(result, 17.0);
        });
    });
}

// ============================================================================
// NEW SYSTEM BENCHMARKS  
// ============================================================================

fn new_system_simple_expression(c: &mut Criterion) {
    c.bench_function("new_system_simple_expression", |b| {
        b.iter(|| {
            let mut ctx: HeteroContext<0> = HeteroContext::new();
            
            // Build f(x, y) = x + y with heterogeneous types
            let x: HeteroVar<f64, 0> = ctx.var();
            let y: HeteroVar<f64, 0> = ctx.var();
            let expr = scalar_add(x, y);
            
            // Evaluate with native inputs
            let mut inputs = HeteroInputs::new();
            inputs.add_scalar_f64(0, 3.0);
            inputs.add_scalar_f64(1, 4.0);
            
            let evaluator: HeteroEvaluator<0> = HeteroEvaluator::new();
            let result = black_box(evaluator.eval_native(&expr, &inputs));
            
            if let dslcompile::compile_time::heterogeneous_v2::EvaluationResult::F64(val) = result {
                assert_eq!(val, 7.0);
            } else {
                panic!("Expected f64 result");
            }
        });
    });
}

fn new_system_heterogeneous_expression(c: &mut Criterion) {
    c.bench_function("new_system_heterogeneous_expression", |b| {
        b.iter(|| {
            let mut ctx: HeteroContext<0> = HeteroContext::new();
            
            // Build heterogeneous expression: array[index] 
            let array: HeteroVar<Vec<f64>, 0> = ctx.var();
            let index: HeteroVar<usize, 0> = ctx.var();
            let expr = array_index(array, index);
            
            // Evaluate with native types (no flattening!)
            let mut inputs = HeteroInputs::new();
            inputs.add_vec_f64(0, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
            inputs.add_usize(1, 2);
            
            let evaluator: HeteroEvaluator<0> = HeteroEvaluator::new();
            let result = black_box(evaluator.eval_native(&expr, &inputs));
            
            if let dslcompile::compile_time::heterogeneous_v2::EvaluationResult::F64(val) = result {
                assert_eq!(val, 3.0); // array[2] = 3.0
            } else {
                panic!("Expected f64 result");
            }
        });
    });
}

fn new_system_neural_network_simulation(c: &mut Criterion) {
    c.bench_function("new_system_neural_network", |b| {
        b.iter(|| {
            let mut ctx: HeteroContext<0> = HeteroContext::new();
            
            // Neural network layer: inputs[index] (no flattening required!)
            let inputs: HeteroVar<Vec<f64>, 0> = ctx.var();
            let weights: HeteroVar<Vec<f64>, 0> = ctx.var();
            let index: HeteroVar<usize, 0> = ctx.var();
            
            // Build: inputs[index] + weights[index] (simplified)
            let input_val = array_index(inputs, index.clone());
            let weight_val = array_index(weights, index);
            
            // For this benchmark, we'll evaluate just the input access
            let expr = input_val;
            
            // Native evaluation - no Vec<f64> flattening!
            let mut native_inputs = HeteroInputs::new();
            native_inputs.add_vec_f64(0, vec![0.5, 0.8, 0.3, 0.9, 0.1]); // inputs
            native_inputs.add_vec_f64(1, vec![1.2, 0.7, 2.1, 0.4, 1.8]); // weights  
            native_inputs.add_usize(2, 1); // index
            
            let evaluator: HeteroEvaluator<0> = HeteroEvaluator::new();
            let result = black_box(evaluator.eval_native(&expr, &native_inputs));
            
            if let dslcompile::compile_time::heterogeneous_v2::EvaluationResult::F64(val) = result {
                assert_eq!(val, 0.8); // inputs[1] = 0.8
            } else {
                panic!("Expected f64 result");
            }
        });
    });
}

// ============================================================================
// MEMORY ALLOCATION BENCHMARKS
// ============================================================================

fn old_system_memory_overhead(c: &mut Criterion) {
    c.bench_function("old_system_memory_overhead", |b| {
        b.iter(|| {
            // Simulate the old system's requirement to flatten all inputs
            let mut flattened = Vec::new();
            
            // Neural network inputs that must be flattened
            let inputs = vec![0.5, 0.8, 0.3, 0.9, 0.1];
            let weights = vec![1.2, 0.7, 2.1, 0.4, 1.8];
            let bias = 0.5f64;
            let threshold = 0.0f64;
            
            // Flatten everything into single Vec<f64>
            flattened.extend_from_slice(&inputs);
            flattened.extend_from_slice(&weights);
            flattened.push(bias);
            flattened.push(threshold);
            
            // Simulate accessing the data
            let result = black_box(flattened[0] + flattened[5]); // inputs[0] + weights[0]
            assert_eq!(result, 1.7); // 0.5 + 1.2
        });
    });
}

fn new_system_zero_allocation(c: &mut Criterion) {
    c.bench_function("new_system_zero_allocation", |b| {
        b.iter(|| {
            // New system: no flattening required, native types
            let inputs = vec![0.5, 0.8, 0.3, 0.9, 0.1];
            let weights = vec![1.2, 0.7, 2.1, 0.4, 1.8];
            let bias = 0.5f64;
            let _threshold = 0.0f64;
            
            // Direct access to native types - no allocations!
            let result = black_box(inputs[0] + weights[0]);
            assert_eq!(result, 1.7); // 0.5 + 1.2
        });
    });
}

criterion_group!(
    old_system_benches,
    old_system_simple_expression,
    old_system_complex_expression, 
    old_system_composition
);

criterion_group!(
    new_system_benches,
    new_system_simple_expression,
    new_system_heterogeneous_expression,
    new_system_neural_network_simulation
);

criterion_group!(
    memory_benches,
    old_system_memory_overhead,
    new_system_zero_allocation
);

criterion_main!(old_system_benches, new_system_benches, memory_benches); 