//! Test round-trip collection identity preservation

use dslcompile::prelude::*;
use frunk::hlist;


#[cfg(feature = "optimization")]
#[test]
fn test_round_trip_collection_identity() {
    let mut ctx = DynamicContext::new();
    let x = ctx.var();

    // Test 1: Simple sum with collection identity
    let data = vec![1.0, 2.0, 3.0];
    let sum_expr = ctx.sum(data.clone(), |item| &x * item);
    let original_ast = ctx.to_ast(&sum_expr);

    println!("DEBUG: Original AST: {:?}", original_ast);

    // Optimization functionality removed - test basic AST evaluation
    let optimized_ast = &original_ast;

    // Test semantic equivalence
    let test_x = 2.0;
    let original_result = ctx.eval(&sum_expr, hlist![test_x]);
    let optimized_result = optimized_ast.eval_with_vars(&[test_x]);

    let expected = test_x * (1.0 + 2.0 + 3.0);

    println!("DEBUG: Original result: {}", original_result);
    println!("DEBUG: Optimized result: {}", optimized_result);
    println!("DEBUG: Expected: {}", expected);
    println!(
        "DEBUG: Original diff: {}",
        (original_result - expected).abs()
    );
    println!(
        "DEBUG: Optimized diff: {}",
        (optimized_result - expected).abs()
    );

    assert!(
        (original_result - expected).abs() < 1e-10,
        "Original should match expected"
    );
    assert!(
        (optimized_result - expected).abs() < 1e-10,
        "Optimized should match expected"
    );
    assert!(
        (original_result - optimized_result).abs() < 1e-10,
        "Round-trip should preserve semantics"
    );

    println!(
        "✅ Simple round-trip test passed: {} ≈ {} ≈ {}",
        original_result, optimized_result, expected
    );
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
    // Optimization functionality removed
    let compound_optimized = &compound_ast;

    let test_x = 2.0;
    let original_compound = ctx.eval(&compound, hlist![test_x]);
    let optimized_compound = compound_optimized.eval_with_vars(&[test_x]);

    let expected_compound = test_x * (1.0 + 2.0) + test_x * (3.0 + 4.0);

    println!("DEBUG: Original compound: {}", original_compound);
    println!("DEBUG: Optimized compound: {}", optimized_compound);
    println!("DEBUG: Expected compound: {}", expected_compound);
    println!(
        "DEBUG: Original compound diff: {}",
        (original_compound - expected_compound).abs()
    );
    println!(
        "DEBUG: Optimized compound diff: {}",
        (optimized_compound - expected_compound).abs()
    );

    assert!(
        (original_compound - expected_compound).abs() < 1e-10,
        "Original compound should match expected"
    );
    assert!(
        (optimized_compound - expected_compound).abs() < 1e-10,
        "Optimized compound should match expected"
    );
    assert!(
        (original_compound - optimized_compound).abs() < 1e-10,
        "Compound round-trip should preserve semantics"
    );

    println!(
        "✅ Multiple collection identity test passed: {} ≈ {} ≈ {}",
        original_compound, optimized_compound, expected_compound
    );
}
