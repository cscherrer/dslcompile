//! Test round-trip collection identity preservation

use dslcompile::prelude::*;
use frunk::hlist;

#[cfg(feature = "optimization")]
use dslcompile::symbolic::egg_optimizer::optimize_simple_sum_splitting;

#[cfg(feature = "optimization")]
#[test]
fn test_round_trip_collection_identity() {
    let mut ctx = DynamicContext::new();
    let x = ctx.var();
    
    // Test 1: Simple sum with collection identity
    let data = vec![1.0, 2.0, 3.0];
    let sum_expr = ctx.sum(data.clone(), |item| &x * item);
    let original_ast = ctx.to_ast(&sum_expr);
    
    // Perform round-trip: AST → MathLang → AST
    let result = optimize_simple_sum_splitting(&original_ast);
    assert!(result.is_ok(), "Round-trip should succeed");
    
    let optimized_ast = result.unwrap();
    
    // Test semantic equivalence
    let test_x = 2.0;
    let original_result = ctx.eval(&sum_expr, hlist![test_x]);
    let optimized_result = optimized_ast.eval_with_vars(&[test_x]);
    
    let expected = test_x * (1.0 + 2.0 + 3.0);
    assert!((original_result - expected).abs() < 1e-10, "Original should match expected");
    assert!((optimized_result - expected).abs() < 1e-10, "Optimized should match expected");
    assert!((original_result - optimized_result).abs() < 1e-10, "Round-trip should preserve semantics");
    
    println!("✅ Simple round-trip test passed: {} ≈ {} ≈ {}", original_result, optimized_result, expected);
}

#[cfg(feature = "optimization")]
#[test]
fn test_multiple_collection_identity() {
    let mut ctx = DynamicContext::new();
    let x = ctx.var();
    
    // Test 2: Multiple collections with different identities
    let data1 = vec![1.0, 2.0];
    let data2 = vec![3.0, 4.0];
    
    let sum1 = ctx.sum(data1.clone(), |item| &x * item);
    let sum2 = ctx.sum(data2.clone(), |item| &x * item);
    let compound = &sum1 + &sum2;
    
    let compound_ast = ctx.to_ast(&compound);
    let compound_result = optimize_simple_sum_splitting(&compound_ast);
    assert!(compound_result.is_ok(), "Compound round-trip should succeed");
    
    let compound_optimized = compound_result.unwrap();
    
    let test_x = 2.0;
    let original_compound = ctx.eval(&compound, hlist![test_x]);
    let optimized_compound = compound_optimized.eval_with_vars(&[test_x]);
    
    let expected_compound = test_x * (1.0 + 2.0) + test_x * (3.0 + 4.0);
    
    assert!((original_compound - expected_compound).abs() < 1e-10, "Original compound should match expected");
    assert!((optimized_compound - expected_compound).abs() < 1e-10, "Optimized compound should match expected");
    assert!((original_compound - optimized_compound).abs() < 1e-10, "Compound round-trip should preserve semantics");
    
    println!("✅ Multiple collection identity test passed: {} ≈ {} ≈ {}", original_compound, optimized_compound, expected_compound);
}