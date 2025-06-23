//! Debug test to understand conversion issues

use dslcompile::prelude::*;
use frunk::hlist;


#[cfg(feature = "optimization")]
#[test]
fn debug_individual_sums() {
    let mut ctx = DynamicContext::new();
    let x = ctx.var::<f64>();

    // Test individual sums separately
    let data1 = vec![1.0, 2.0];
    let data2 = vec![3.0, 4.0];

    let sum1 = ctx.sum(data1.clone(), |item| &x * item);
    let sum2 = ctx.sum(data2.clone(), |item| &x * item);

    let test_x = 2.0;

    println!("=== INDIVIDUAL SUMS ===");

    // Test sum1 alone
    let sum1_ast = ctx.to_ast(&sum1);
    let sum1_original = ctx.eval(&sum1, hlist![test_x]);
    println!("Sum1 original: {}", sum1_original); // Should be 2 * (1+2) = 6

    // Optimization functionality removed
    let sum1_optimized = sum1_ast.eval_with_vars(&[test_x]);
    println!("Sum1 optimized: {}", sum1_optimized); // Should be 6

    // Test sum2 alone
    let sum2_ast = ctx.to_ast(&sum2);
    let sum2_original = ctx.eval(&sum2, hlist![test_x]);
    println!("Sum2 original: {}", sum2_original); // Should be 2 * (3+4) = 14

    // Optimization functionality removed
    let sum2_optimized = sum2_ast.eval_with_vars(&[test_x]);
    println!("Sum2 optimized: {}", sum2_optimized); // Should be 14

    // Both individual sums should work
    assert!(
        (sum1_optimized - 6.0).abs() < 1e-10,
        "Sum1 should be 6, got {}",
        sum1_optimized
    );
    assert!(
        (sum2_optimized - 14.0).abs() < 1e-10,
        "Sum2 should be 14, got {}",
        sum2_optimized
    );
}

#[cfg(feature = "optimization")]
#[test]
fn debug_addition_simple() {
    let mut ctx = DynamicContext::new();
    let x = ctx.var::<f64>();

    // Test simple addition of constants first
    let const1 = ctx.constant(6.0); // Simulating sum1 result
    let const2 = ctx.constant(14.0); // Simulating sum2 result  
    let simple_add = &const1 + &const2;

    let test_x = 2.0;
    let simple_add_ast = ctx.to_ast(&simple_add);
    let simple_add_original = ctx.eval(&simple_add, hlist![test_x]);
    println!("=== SIMPLE ADDITION ===");
    println!("Simple add original: {}", simple_add_original); // Should be 20

    // Optimization functionality removed
    let simple_add_optimized = simple_add_ast.eval_with_vars(&[test_x]);
    println!("Simple add optimized: {}", simple_add_optimized); // Should be 20

    assert!(
        (simple_add_optimized - 20.0_f64).abs() < 1e-10_f64,
        "Simple add should be 20, got {}",
        simple_add_optimized
    );
}
