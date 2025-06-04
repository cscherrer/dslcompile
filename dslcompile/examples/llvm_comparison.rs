//! LLVM IR Comparison for Heterogeneous System Optimization
//!
//! This example isolates the two heterogeneous implementations to compare
//! their generated LLVM IR and understand why the "optimized" version is slower.

use dslcompile::compile_time::heterogeneous::{HeteroContext, HeteroInputs, HeteroExpr, hetero_add};

/// Original implementation using method calls
#[inline(never)]
pub fn hetero_original(x: f64, y: f64) -> f64 {
    // Setup (this would be done once in real usage)
    let mut ctx = HeteroContext::<0, 8>::new();
    let x_var = ctx.var::<f64>();
    let y_var = ctx.var::<f64>();
    let expr = hetero_add::<f64, _, _, 0>(x_var, y_var);
    
    // The actual function call we're measuring
    let mut inputs = HeteroInputs::<8>::new();
    inputs.add_f64(0, x);
    inputs.add_f64(1, y);
    expr.eval(&inputs)
}

/// "Optimized" implementation using direct array access
#[inline(never)]
pub fn hetero_optimized(x: f64, y: f64) -> f64 {
    // Setup (this would be done once in real usage)
    let mut ctx = HeteroContext::<0, 8>::new();
    let x_var = ctx.var::<f64>();
    let y_var = ctx.var::<f64>();
    let expr = hetero_add::<f64, _, _, 0>(x_var, y_var);
    
    // The actual function call we're measuring - "optimized"
    let mut inputs = HeteroInputs::<8>::default();
    inputs.f64_values[0] = Some(x);
    inputs.f64_values[1] = Some(y);
    expr.eval(&inputs)
}

/// Direct Rust baseline
#[inline(never)]
pub fn direct_rust(x: f64, y: f64) -> f64 {
    x + y
}

fn main() {
    // Prevent dead code elimination
    let result1 = hetero_original(3.0, 4.0);
    let result2 = hetero_optimized(3.0, 4.0);
    let result3 = direct_rust(3.0, 4.0);
    
    println!("Original: {}", result1);
    println!("Optimized: {}", result2);
    println!("Direct: {}", result3);
    
    // Ensure all produce same result
    assert_eq!(result1, 7.0);
    assert_eq!(result2, 7.0);
    assert_eq!(result3, 7.0);
} 