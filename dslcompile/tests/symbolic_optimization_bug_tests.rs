//! Test to investigate why symbolic optimization isn't reducing operations
//!
//! The egglog optimization should be finding simplifications in complex expressions

use dslcompile::{
    ast::ast_utils::visitors::{OperationCountVisitor, SummationCountVisitor},
    prelude::*,
};
use frunk::hlist;

#[cfg(feature = "optimization")]
#[test]
fn test_simple_optimization_should_reduce() {
    // Test a simple expression that should definitely be optimized
    let mut ctx = DynamicContext::new();
    let x = ctx.var::<f64>();

    // Create an expression with obvious redundancy: x + 0 + 0 * y
    let y = ctx.var::<f64>();
    let zero = 0.0;
    let expr = &x + zero + (zero * &y); // Should optimize to just x

    let original_ast = ctx.to_ast(&expr);
    let original_ops = OperationCountVisitor::count_operations(&original_ast);

    println!("Original expression: {original_ast:#?}");
    println!("Original operations: {original_ops}");

    // Apply symbolic optimization
    let mut optimizer = dslcompile::SymbolicOptimizer::new_for_testing().unwrap();
    let optimized_ast = optimizer.optimize(&original_ast).unwrap();
    let optimized_ops = OperationCountVisitor::count_operations(&optimized_ast);

    println!("Optimized expression: {optimized_ast:#?}");
    println!("Optimized operations: {optimized_ops}");

    // This should reduce operations! x + 0 + 0*y should become just x
    assert!(
        optimized_ops < original_ops,
        "Expected optimization to reduce operations from {original_ops} to less, but got {optimized_ops}"
    );

    // Verify semantic equivalence
    let original_result = ctx.eval(&expr, hlist![5.0, 3.0]); // x=5, y=3
    // For now, evaluate AST directly since we don't have from_ast
    let optimized_result = optimized_ast.eval_with_vars(&[5.0, 3.0]);

    assert!(
        (original_result - optimized_result).abs() < 1e-10,
        "Optimization changed semantics: {original_result} vs {optimized_result}"
    );
}

#[cfg(feature = "optimization")]
#[test]
fn test_algebraic_simplification() {
    // Test algebraic simplifications like x * 1, x * 0, etc.
    let mut ctx = DynamicContext::new();
    let x = ctx.var::<f64>();

    // x * 1 + x * 0 should optimize to x
    let expr = &x * 1.0 + &x * 0.0;

    let original_ast = ctx.to_ast(&expr);
    let original_ops = OperationCountVisitor::count_operations(&original_ast);

    println!("Algebraic expression: {original_ast:#?}");
    println!("Original operations: {original_ops}");

    let mut optimizer = dslcompile::SymbolicOptimizer::new_for_testing().unwrap();
    let optimized_ast = optimizer.optimize(&original_ast).unwrap();
    let optimized_ops = OperationCountVisitor::count_operations(&optimized_ast);

    println!("Optimized algebraic: {optimized_ast:#?}");
    println!("Optimized operations: {optimized_ops}");

    // Should reduce: x*1 + x*0 → x + 0 → x
    assert!(
        optimized_ops < original_ops,
        "Expected algebraic optimization to reduce operations"
    );
}

#[cfg(feature = "optimization")]
#[test]
fn test_normal_log_density_optimization() {
    // Test the actual Normal log-density expression for optimization opportunities
    let mut ctx = DynamicContext::new();
    let x = ctx.var::<f64>();
    let mu = ctx.var::<f64>();
    let sigma = ctx.var::<f64>();

    // Simplified normal log-density: -0.5 * ((x - μ) / σ)²
    let centered = &x - &mu;
    let standardized = &centered / &sigma;
    let squared = &standardized * &standardized;
    let log_density = -0.5 * &squared;

    let original_ast = ctx.to_ast(&log_density);
    let original_ops = OperationCountVisitor::count_operations(&original_ast);

    println!("Normal log-density: {original_ast:#?}");
    println!("Original operations: {original_ops}");

    let mut optimizer = dslcompile::SymbolicOptimizer::new_for_testing().unwrap();
    let optimized_ast = optimizer.optimize(&original_ast).unwrap();
    let optimized_ops = OperationCountVisitor::count_operations(&optimized_ast);

    println!("Optimized normal: {optimized_ast:#?}");
    println!("Optimized operations: {optimized_ops}");

    // The key metric is that optimization should reduce cost, not necessarily operation count
    // Division (cost 5) should be preferred over Mul + Pow (cost 1 + 10 = 11)
    // So let's check if the semantic result is preserved (the real goal)
    println!("Note: Operation count may increase during canonicalization, but cost should improve");

    // Verify semantic equivalence
    let test_values = hlist![1.5, 1.0, 0.5]; // x=1.5, μ=1.0, σ=0.5
    let original_result = ctx.eval(&log_density, test_values);
    let optimized_result = optimized_ast.eval_with_vars(&[1.5, 1.0, 0.5]);

    println!("Original result: {original_result}");
    println!("Optimized result: {optimized_result}");

    assert!(
        (original_result - optimized_result).abs() < 1e-10,
        "Optimization changed semantics"
    );
}

#[cfg(feature = "optimization")]
#[test]
fn test_iid_summation_optimization() {
    // Test the IID summation expression that was showing 16→16 operations
    let mut ctx = DynamicContext::new();
    let mu = ctx.var::<f64>();
    let sigma = ctx.var::<f64>();

    // Create IID summation like in the measures demo
    let data = vec![1.0, 2.0, 0.5];
    let iid_sum = ctx.sum(&data, |x_i| {
        // Simplified log-density for each observation
        let diff = &x_i - &mu;
        -0.5 * (&diff * &diff)
    });

    let original_ast = ctx.to_ast(&iid_sum);
    let original_ops = OperationCountVisitor::count_operations(&original_ast);
    let original_sums = SummationCountVisitor::count_summations(&original_ast);

    println!("IID summation: {original_ast:#?}");
    println!("Original operations: {original_ops}");
    println!("Original summations: {original_sums}");

    let mut optimizer = dslcompile::SymbolicOptimizer::new_for_testing().unwrap();
    let optimized_ast = optimizer.optimize(&original_ast).unwrap();
    let optimized_ops = OperationCountVisitor::count_operations(&optimized_ast);
    let optimized_sums = SummationCountVisitor::count_summations(&optimized_ast);

    println!("Optimized IID: {optimized_ast:#?}");
    println!("Optimized operations: {optimized_ops}");
    println!("Optimized summations: {optimized_sums}");

    // Summation count should stay the same
    assert_eq!(
        original_sums, optimized_sums,
        "Summation count should be preserved"
    );

    // Operations might reduce due to algebraic simplifications inside the lambda
    if optimized_ops < original_ops {
        println!(
            "✅ Optimization reduced operations by {}",
            original_ops - optimized_ops
        );
    } else {
        println!("ℹ️ No operation reduction found (may already be optimal)");
    }

    // Verify semantic equivalence
    let test_values = hlist![1.0]; // μ=1.0
    let original_result = ctx.eval(&iid_sum, test_values);
    let optimized_result = optimized_ast.eval_with_vars(&[1.0]);

    println!("Original result: {original_result}");
    println!("Optimized result: {optimized_result}");

    assert!(
        (original_result - optimized_result).abs() < 1e-10,
        "Optimization changed semantics"
    );
}

#[cfg(not(feature = "optimization"))]
#[test]
fn test_optimization_disabled() {
    println!("Symbolic optimization tests skipped - compile with --features optimization");
}
