//! Benchmark: Macro-Based vs Heterogeneous System
//!
//! This benchmark compares the macro-based zero-overhead approach
//! against the current heterogeneous system to quantify the performance difference.

use divan::Bencher;
use dslcompile::compile_time::heterogeneous::{HeteroContext, HeteroInputs, HeteroExpr, hetero_add};

/// Macro to build zero-overhead expressions with flexible arity
macro_rules! hetero_expr {
    // Simple binary operations
    (|$x:ident: f64, $y:ident: f64| $x_ref:ident + $y_ref:ident) => {
        |$x: f64, $y: f64| -> f64 { $x_ref + $y_ref }
    };
    
    // Ternary operations
    (|$x:ident: f64, $y:ident: f64, $z:ident: f64| $x_ref:ident + $y_ref:ident + $z_ref:ident) => {
        |$x: f64, $y: f64, $z: f64| -> f64 { $x_ref + $y_ref + $z_ref }
    };
    
    // Mixed types - array indexing
    (|$arr:ident: &[f64], $idx:ident: usize, $bias:ident: f64| $arr_ref:ident[$idx_ref:ident] + $bias_ref:ident) => {
        |$arr: &[f64], $idx: usize, $bias: f64| -> f64 { $arr_ref[$idx_ref] + $bias_ref }
    };
}

fn main() {
    divan::main();
}

/// Setup functions for comparison
fn setup_systems() -> (
    impl Fn(f64, f64) -> f64,  // Macro-based
    impl Fn(f64, f64) -> f64,  // Heterogeneous
    impl Fn(f64, f64) -> f64,  // Direct Rust baseline
) {
    // 1. Macro-based zero-overhead function
    let macro_fn = hetero_expr!(|x: f64, y: f64| x + y);
    
    // 2. Heterogeneous system setup
    let mut ctx = HeteroContext::<0, 8>::new();
    let x_var = ctx.var::<f64>();
    let y_var = ctx.var::<f64>();
    let hetero_expr = hetero_add::<f64, _, _, 0>(x_var, y_var);
    
    let hetero_fn = move |x: f64, y: f64| -> f64 {
        let mut inputs = HeteroInputs::<8>::new();
        inputs.add_f64(0, x);
        inputs.add_f64(1, y);
        hetero_expr.eval(&inputs)
    };
    
    // 3. Direct Rust baseline
    let direct_fn = |x: f64, y: f64| -> f64 { x + y };
    
    (macro_fn, hetero_fn, direct_fn)
}

#[divan::bench]
fn macro_based_system(bencher: Bencher) {
    let (macro_fn, _, _) = setup_systems();
    bencher.bench_local(|| macro_fn(3.0, 4.0));
}

#[divan::bench]
fn heterogeneous_system(bencher: Bencher) {
    let (_, hetero_fn, _) = setup_systems();
    bencher.bench_local(|| hetero_fn(3.0, 4.0));
}

#[divan::bench]
fn direct_rust_baseline(bencher: Bencher) {
    let (_, _, direct_fn) = setup_systems();
    bencher.bench_local(|| direct_fn(3.0, 4.0));
}

/// Test with more parameters to show scaling behavior
fn setup_ternary_systems() -> (
    impl Fn(f64, f64, f64) -> f64,  // Macro-based
    impl Fn(f64, f64, f64) -> f64,  // Heterogeneous
    impl Fn(f64, f64, f64) -> f64,  // Direct Rust baseline
) {
    // 1. Macro-based ternary function
    let macro_fn = hetero_expr!(|x: f64, y: f64, z: f64| x + y + z);
    
    // 2. Heterogeneous ternary system
    let mut ctx = HeteroContext::<0, 8>::new();
    let x_var = ctx.var::<f64>();
    let y_var = ctx.var::<f64>();
    let z_var = ctx.var::<f64>();
    let xy_expr = hetero_add::<f64, _, _, 0>(x_var, y_var);
    let xyz_expr = hetero_add::<f64, _, _, 0>(xy_expr, z_var);
    
    let hetero_fn = move |x: f64, y: f64, z: f64| -> f64 {
        let mut inputs = HeteroInputs::<8>::new();
        inputs.add_f64(0, x);
        inputs.add_f64(1, y);
        inputs.add_f64(2, z);
        xyz_expr.eval(&inputs)
    };
    
    // 3. Direct Rust baseline
    let direct_fn = |x: f64, y: f64, z: f64| -> f64 { x + y + z };
    
    (macro_fn, hetero_fn, direct_fn)
}

#[divan::bench]
fn macro_based_ternary(bencher: Bencher) {
    let (macro_fn, _, _) = setup_ternary_systems();
    bencher.bench_local(|| macro_fn(1.0, 2.0, 3.0));
}

#[divan::bench]
fn heterogeneous_ternary(bencher: Bencher) {
    let (_, hetero_fn, _) = setup_ternary_systems();
    bencher.bench_local(|| hetero_fn(1.0, 2.0, 3.0));
}

#[divan::bench]
fn direct_rust_ternary(bencher: Bencher) {
    let (_, _, direct_fn) = setup_ternary_systems();
    bencher.bench_local(|| direct_fn(1.0, 2.0, 3.0));
}

/// Test mixed types to show macro flexibility
fn setup_mixed_type_systems() -> (
    impl Fn(&[f64], usize, f64) -> f64,  // Macro-based
    impl Fn(&[f64], usize, f64) -> f64,  // Direct Rust baseline
) {
    // 1. Macro-based mixed types
    let macro_fn = hetero_expr!(|arr: &[f64], idx: usize, bias: f64| arr[idx] + bias);
    
    // 2. Direct Rust baseline
    let direct_fn = |arr: &[f64], idx: usize, bias: f64| -> f64 { arr[idx] + bias };
    
    (macro_fn, direct_fn)
}

#[divan::bench]
fn macro_based_mixed_types(bencher: Bencher) {
    let weights = [0.1, 0.2, 0.3, 0.4];
    let (macro_fn, _) = setup_mixed_type_systems();
    bencher.bench_local(|| macro_fn(&weights, 1, 0.5));
}

#[divan::bench]
fn direct_rust_mixed_types(bencher: Bencher) {
    let weights = [0.1, 0.2, 0.3, 0.4];
    let (_, direct_fn) = setup_mixed_type_systems();
    bencher.bench_local(|| direct_fn(&weights, 1, 0.5));
} 