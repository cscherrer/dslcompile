//! Tests to reproduce and fix the variable indexing bug in summation lambdas
//!
//! These tests isolate the specific issue where external variables aren't properly
//! accessible within summation lambda expressions.

use dslcompile::prelude::*;
use dslcompile::contexts::dynamic::expression_builder::hlist_support::HListEval;
use frunk::hlist;

#[test]
fn test_simple_summation_no_external_vars() {
    // This should work - no external variables
    let mut ctx = DynamicContext::new();
    let data = vec![1.0, 2.0, 3.0];
    let sum_expr = ctx.sum(&data, |x| x * 2.0);
    
    let result = ctx.eval(&sum_expr, hlist![]);
    let expected = 2.0 * (1.0 + 2.0 + 3.0); // 12.0
    assert!((result - expected).abs() < 1e-10, "Expected {}, got {}", expected, result);
}

#[test]
fn test_summation_with_external_var_now_works() {
    // This should now work with the bug fixed
    let mut ctx = DynamicContext::new();
    let a: DynamicExpr<f64> = ctx.var(); // External variable, index 0
    
    let data = vec![1.0, 2.0, 3.0];
    let sum_expr = ctx.sum(&data, |x| &x + &a); // Lambda uses external variable 'a'
    
    println!("External variable 'a' has index: {}", a.var_id());
    
    // This should now work: (1+5) + (2+5) + (3+5) = 6 + 7 + 8 = 21
    let result = ctx.eval(&sum_expr, hlist![5.0]); // a = 5.0
    let expected = 21.0;
    assert!((result - expected).abs() < 1e-10, "Expected {}, got {}", expected, result);
}

#[test]
fn test_summation_with_two_external_vars_now_works() {
    // This should now work with multiple external variables
    let mut ctx = DynamicContext::new();
    let a: DynamicExpr<f64> = ctx.var(); // External variable, index 0
    let b: DynamicExpr<f64> = ctx.var(); // External variable, index 1
    
    let data = vec![1.0, 2.0];
    let sum_expr = ctx.sum(&data, |x| &x * &a + &b); // Lambda uses both external variables
    
    println!("External variables: a={}, b={}", a.var_id(), b.var_id());
    
    // This should now work: (1*2+10) + (2*2+10) = 12 + 14 = 26
    let result = ctx.eval(&sum_expr, hlist![2.0, 10.0]); // a = 2.0, b = 10.0
    let expected = 26.0;
    assert!((result - expected).abs() < 1e-10, "Expected {}, got {}", expected, result);
}

#[test]
fn test_basic_variable_access_works() {
    // Sanity check - basic variable access should work
    let mut ctx = DynamicContext::new();
    let a: DynamicExpr<f64> = ctx.var();
    let b: DynamicExpr<f64> = ctx.var();
    
    let expr = &a + &b;  // Simple expression with external variables
    let result = ctx.eval(&expr, hlist![3.0, 4.0]);
    assert!((result - 7.0).abs() < 1e-10, "Expected 7.0, got {}", result);
}

#[test]
fn test_debug_ast_structure() {
    // Debug test to examine the AST structure of summation with external variables
    let mut ctx = DynamicContext::new();
    let a: DynamicExpr<f64> = ctx.var(); // External variable, index 0
    
    let data = vec![1.0, 2.0];
    let sum_expr = ctx.sum(&data, |x| &x + &a);
    
    // Convert to AST and examine structure
    let ast = ctx.to_ast(&sum_expr);
    println!("AST structure: {:#?}", ast);
    
    // This test just prints the AST structure - doesn't evaluate
    // We can use this to understand how external variables are represented
}

#[test]
fn test_expected_behavior_now_works() {
    // This test shows that the expected behavior now works!
    let mut ctx = DynamicContext::new();
    let a: DynamicExpr<f64> = ctx.var(); // External variable 'a' with index 0
    
    let data = vec![1.0, 2.0, 3.0];
    let sum_expr = ctx.sum(&data, |x| &x + &a); // For each x in data, compute x + a
    
    // Expected behavior: When evaluating the summation
    // - The lambda should receive BOTH the loop variable (x) AND access to external variables
    // - For data=[1,2,3] and a=5, should compute: (1+5) + (2+5) + (3+5) = 6+7+8 = 21
    
    // This now works!
    let result = ctx.eval(&sum_expr, hlist![5.0]); // a = 5.0
    assert!((result - 21.0).abs() < 1e-10, "Expected 21.0, got {}", result);
    
    println!("The bug has been fixed! External variables now work in summation lambdas.");
}