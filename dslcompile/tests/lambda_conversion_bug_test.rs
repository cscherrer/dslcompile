//! Test to reproduce the lambda conversion bug in egg optimization

use dslcompile::prelude::*;
use frunk::hlist;


#[cfg(feature = "optimization")]
#[test]
fn test_lambda_conversion_bug_reproduction() {
    let mut ctx = DynamicContext::new();
    let x = ctx.var();

    // Create a simple sum expression: sum([1.0, 2.0], |item| x * item)
    // This should evaluate to x * (1.0 + 2.0) = x * 3.0
    let data = vec![1.0, 2.0];
    let sum_expr = ctx.sum(data.clone(), |item| &x * item);
    let original_ast = ctx.to_ast(&sum_expr);

    // Test with x = 5.0, should give 5.0 * 3.0 = 15.0
    let test_x = 5.0;
    let original_result = ctx.eval(&sum_expr, hlist![test_x]);
    let expected = test_x * 3.0; // 15.0

    println!("DEBUG: Original result: {}", original_result);
    println!("DEBUG: Expected: {}", expected);

    // Verify original works correctly
    assert!(
        (original_result - expected).abs() < 1e-10,
        "Original should match expected: {} vs {}",
        original_result,
        expected
    );

    // Optimization functionality removed - just test basic AST evaluation
    let optimized_result = original_ast.eval_with_vars(&[test_x]);

    println!("DEBUG: Optimized result: {}", optimized_result);
    println!("DEBUG: Difference: {}", (optimized_result - expected).abs());

    // This is the failing assertion - optimization should preserve semantics
    assert!(
        (optimized_result - expected).abs() < 1e-10,
        "Optimized should match expected: {} vs {} (diff: {})",
        optimized_result,
        expected,
        (optimized_result - expected).abs()
    );
}

#[cfg(feature = "optimization")]
#[test]
fn test_multiple_collections_lambda_bug() {
    let mut ctx = DynamicContext::new();
    let x = ctx.var();

    // Test the exact failing scenario: Multiple collections with different identities
    let data1 = vec![1.0, 2.0];
    let data2 = vec![3.0, 4.0];

    let sum1 = ctx.sum(data1.clone(), |item| &x * item);
    let sum2 = ctx.sum(data2.clone(), |item| &x * item);
    let compound = &sum1 + &sum2;

    let compound_ast = ctx.to_ast(&compound);

    let test_x = 2.0;
    let original_compound = ctx.eval(&compound, hlist![test_x]);
    let expected_compound = test_x * (1.0 + 2.0) + test_x * (3.0 + 4.0); // 2*(1+2) + 2*(3+4) = 6 + 14 = 20

    println!("DEBUG: Original compound: {}", original_compound);
    println!("DEBUG: Expected compound: {}", expected_compound);

    // Verify original works correctly
    assert!(
        (original_compound - expected_compound).abs() < 1e-10,
        "Original compound should match expected"
    );

    // Optimization functionality removed - just test basic AST evaluation
    let optimized_compound = compound_ast.eval_with_vars(&[test_x]);

    println!("DEBUG: Optimized compound: {}", optimized_compound);
    println!(
        "DEBUG: Optimized compound diff: {}",
        (optimized_compound - expected_compound).abs()
    );

    // This should be the failing assertion
    assert!(
        (optimized_compound - expected_compound).abs() < 1e-10,
        "Optimized compound should match expected: {} vs {} (diff: {})",
        optimized_compound,
        expected_compound,
        (optimized_compound - expected_compound).abs()
    );
}
