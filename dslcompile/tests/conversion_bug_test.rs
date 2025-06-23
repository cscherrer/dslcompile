//! Test to isolate the AST conversion bug

use dslcompile::prelude::*;
use frunk::hlist;


#[cfg(feature = "optimization")]
#[test]
fn test_simple_single_sum_conversion() {
    let mut ctx = DynamicContext::new();
    let x = ctx.var();

    // Test a single sum: sum([1.0, 2.0], |item| x * item)
    // Should evaluate to x * (1.0 + 2.0) = x * 3.0
    let data = vec![1.0, 2.0];
    let sum_expr = ctx.sum(data.clone(), |item| &x * item);
    let original_ast = ctx.to_ast(&sum_expr);

    let test_x = 2.0;
    let original_result = ctx.eval(&sum_expr, hlist![test_x]);
    let expected = test_x * 3.0; // 6.0

    println!("=== SINGLE SUM TEST ===");
    println!("DEBUG: Original result: {}", original_result);
    println!("DEBUG: Expected: {}", expected);

    // Verify original works
    assert!(
        (original_result - expected).abs() < 1e-10,
        "Original should match expected"
    );

    // Optimization functionality removed - test basic AST evaluation
    let optimized_result = original_ast.eval_with_vars(&[test_x]);

    println!("DEBUG: Optimized result: {}", optimized_result);
    println!("DEBUG: Difference: {}", (optimized_result - expected).abs());

    // This should work with no optimization rules
    assert!(
        (optimized_result - expected).abs() < 1e-10,
        "Single sum conversion should preserve semantics: {} vs {}",
        optimized_result,
        expected
    );
}

#[cfg(feature = "optimization")]
#[test]
fn test_two_separate_sums_conversion() {
    let mut ctx = DynamicContext::new();
    let x = ctx.var();

    // Test two separate sums:
    // sum1 = sum([1.0, 2.0], |item| x * item) = x * 3
    // sum2 = sum([3.0, 4.0], |item| x * item) = x * 7
    // total = sum1 + sum2 = x * 3 + x * 7 = x * 10
    let data1 = vec![1.0, 2.0];
    let data2 = vec![3.0, 4.0];

    let sum1 = ctx.sum(data1.clone(), |item| &x * item);
    let sum2 = ctx.sum(data2.clone(), |item| &x * item);
    let total = &sum1 + &sum2;

    let total_ast = ctx.to_ast(&total);

    let test_x = 2.0;
    let original_result = ctx.eval(&total, hlist![test_x]);
    let expected = test_x * 10.0; // 20.0

    println!("=== TWO SUMS TEST ===");
    println!("DEBUG: Original result: {}", original_result);
    println!("DEBUG: Expected: {}", expected);

    // Verify original works
    assert!(
        (original_result - expected).abs() < 1e-10,
        "Original should match expected"
    );

    // Optimization functionality removed - test basic AST evaluation
    println!("DEBUG: Using original AST: {:#?}", total_ast);
    let optimized_result = total_ast.eval_with_vars(&[test_x]);

    println!("DEBUG: Optimized result: {}", optimized_result);
    println!("DEBUG: Difference: {}", (optimized_result - expected).abs());

    // This should work with no optimization rules
    assert!(
        (optimized_result - expected).abs() < 1e-10,
        "Two sums conversion should preserve semantics: {} vs {}",
        optimized_result,
        expected
    );
}
