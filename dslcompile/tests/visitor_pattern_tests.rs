//! Tests for the visitor pattern implementation
//!
//! These tests ensure that the visitor pattern correctly replaces recursive implementations
//! and handles edge cases like variable indexing in summations.

use dslcompile::{
    ast::ast_utils::visitors::{DepthVisitor, OperationCountVisitor, SummationCountVisitor},
    prelude::*,
};
use frunk::hlist;

#[test]
fn test_visitor_pattern_basic_operations() {
    let mut ctx = DynamicContext::new();
    let x: DynamicExpr<f64> = ctx.var();
    let y: DynamicExpr<f64> = ctx.var();

    // Create a simple expression: x + y * 2.0
    let expr = &x + &y * 2.0;
    let ast = ctx.to_ast(&expr);

    // Test operation counting with visitor
    let op_count = OperationCountVisitor::count_operations(&ast);
    assert!(op_count > 0, "Should count some operations");

    // Test depth computation with visitor
    let depth = DepthVisitor::compute_depth(&ast);
    assert!(depth > 1, "Should have depth > 1 for nested expression");

    // Test summation counting (should be 0 for basic arithmetic)
    let sum_count = SummationCountVisitor::count_summations(&ast);
    assert_eq!(sum_count, 0, "Basic arithmetic should have no summations");
}

#[test]
fn test_visitor_pattern_deep_expression() {
    let mut ctx = DynamicContext::new();
    let mut expr: DynamicExpr<f64> = ctx.var();

    // Create a deeply nested expression that would cause stack overflow with recursion
    for i in 0..100 {
        expr = expr + f64::from(i);
    }

    let ast = ctx.to_ast(&expr);

    // This should not panic or cause stack overflow
    let op_count = OperationCountVisitor::count_operations(&ast);
    assert!(op_count >= 100, "Should count at least 100 operations");

    let depth = DepthVisitor::compute_depth(&ast);
    assert!(depth > 50, "Should have significant depth");
}

#[test]
fn test_simple_summation_evaluation() {
    let mut ctx = DynamicContext::new();

    // Create simple summation over constant data
    let data = vec![1.0, 2.0, 3.0];
    let sum_expr = ctx.sum(&data, |x| x * 2.0);

    // Test AST analysis with visitors
    let ast = ctx.to_ast(&sum_expr);
    let sum_count = SummationCountVisitor::count_summations(&ast);
    assert_eq!(sum_count, 1, "Should have exactly one summation");

    // Test evaluation - this is where the panic occurs
    let result = ctx.eval(&sum_expr, hlist![]);
    let expected = 2.0 * (1.0 + 2.0 + 3.0); // 2 * 6 = 12
    assert!(
        (result - expected).abs() < 1e-10,
        "Expected {expected}, got {result}"
    );
}

#[test]
fn test_summation_with_external_variables() {
    let mut ctx = DynamicContext::new();

    // Create a variable for scaling
    let scale = ctx.var();

    // Create summation that uses both loop variable and external variable
    let data = vec![1.0, 2.0, 3.0];
    let sum_expr = ctx.sum(&data, |x| &x * &scale);

    // Test AST analysis
    let ast = ctx.to_ast(&sum_expr);
    let sum_count = SummationCountVisitor::count_summations(&ast);
    assert_eq!(sum_count, 1, "Should have exactly one summation");

    // Test evaluation with scale = 2.0
    // This might be where the variable index issue occurs
    let result = ctx.eval(&sum_expr, hlist![2.0]);
    let expected = 2.0 * (1.0 + 2.0 + 3.0); // 2 * 6 = 12
    assert!(
        (result - expected).abs() < 1e-10,
        "Expected {expected}, got {result}"
    );
}

#[test]
fn test_normal_distribution_simple() {
    // Simple Normal struct that mimics the demo
    #[derive(Clone)]
    struct Normal<Mean, StdDev> {
        mean: Mean,
        std_dev: StdDev,
    }

    impl<Mean, StdDev> Normal<Mean, StdDev> {
        fn new(mean: Mean, std_dev: StdDev) -> Self {
            Self { mean, std_dev }
        }
    }

    impl Normal<DynamicExpr<f64>, DynamicExpr<f64>> {
        fn log_density(&self, x: &DynamicExpr<f64>) -> DynamicExpr<f64> {
            // Simplified log-density: just (x - μ)²
            let centered = x - &self.mean;
            &centered * &centered
        }
    }

    let mut ctx = DynamicContext::new();
    let mu: DynamicExpr<f64> = ctx.var();
    let sigma: DynamicExpr<f64> = ctx.var();
    let x: DynamicExpr<f64> = ctx.var();

    let normal = Normal::new(mu.clone(), sigma.clone());
    let log_density = normal.log_density(&x);

    // Test evaluation
    let result = ctx.eval(&log_density, hlist![0.0, 1.0, 1.0]); // μ=0, σ=1, x=1
    assert!((result - 1.0).abs() < 1e-10, "Expected 1.0, got {result}"); // (1-0)² = 1
}

#[test]
fn test_iid_distribution_variable_indexing() {
    use std::marker::PhantomData;

    // IID struct that mimics the demo
    #[derive(Clone)]
    struct IID<Distribution> {
        base_distribution: Distribution,
        _phantom: PhantomData<Distribution>,
    }

    impl<Distribution> IID<Distribution> {
        fn new(base_distribution: Distribution) -> Self {
            Self {
                base_distribution,
                _phantom: PhantomData,
            }
        }
    }

    #[derive(Clone)]
    struct Normal<Mean, StdDev> {
        mean: Mean,
        std_dev: StdDev,
    }

    impl<Mean, StdDev> Normal<Mean, StdDev> {
        fn new(mean: Mean, std_dev: StdDev) -> Self {
            Self { mean, std_dev }
        }
    }

    impl Normal<DynamicExpr<f64>, DynamicExpr<f64>> {
        fn log_density(&self, x: &DynamicExpr<f64>) -> DynamicExpr<f64> {
            // Simplified: (x - μ)²
            let centered = x - &self.mean;
            &centered * &centered
        }
    }

    impl IID<Normal<DynamicExpr<f64>, DynamicExpr<f64>>> {
        fn log_density(&self, ctx: &mut DynamicContext, data: &[f64]) -> DynamicExpr<f64> {
            // This is where the variable indexing issue likely occurs
            ctx.sum(data, |x_i| self.base_distribution.log_density(&x_i))
        }
    }

    let mut ctx = DynamicContext::new();
    let mu = ctx.var(); // Variable 0
    let sigma = ctx.var(); // Variable 1

    let normal = Normal::new(mu.clone(), sigma.clone());
    let iid_normal = IID::new(normal);

    // Test with simple data
    let data = vec![1.0, 2.0];
    let iid_expr = iid_normal.log_density(&mut ctx, &data);

    // Test AST analysis first
    let ast = ctx.to_ast(&iid_expr);
    let sum_count = SummationCountVisitor::count_summations(&ast);
    assert_eq!(sum_count, 1, "Should have exactly one summation");

    // This is where the panic likely occurs - variable index mismatch
    println!("About to evaluate IID expression...");
    println!(
        "Variables in context: μ={}, σ={}",
        mu.var_id(),
        sigma.var_id()
    );

    // Try to evaluate - this should help us understand the variable indexing issue
    let result = ctx.eval(&iid_expr, hlist![0.0, 1.0]); // μ=0, σ=1

    // Expected: (1-0)² + (2-0)² = 1 + 4 = 5
    assert!((result - 5.0).abs() < 1e-10, "Expected 5.0, got {result}");
}

#[test]
fn test_variable_indexing_debug() {
    let mut ctx = DynamicContext::new();
    let a: DynamicExpr<f64> = ctx.var(); // Should be index 0
    let b: DynamicExpr<f64> = ctx.var(); // Should be index 1

    println!("Variable a has index: {}", a.var_id());
    println!("Variable b has index: {}", b.var_id());

    // Test simple summation with external variable
    let data = vec![1.0, 2.0];
    let sum_expr = ctx.sum(&data, |x| &x + &a); // x + a, where a is external variable 0

    println!("Created summation expression");

    // Convert to AST and examine structure
    let ast = ctx.to_ast(&sum_expr);
    println!("AST created successfully");

    // Try evaluation
    println!("About to evaluate with a=10.0");
    let result = ctx.eval(&sum_expr, hlist![10.0]); // a = 10.0
    println!("Evaluation result: {result}");

    // Expected: (1 + 10) + (2 + 10) = 11 + 12 = 23
    assert!((result - 23.0).abs() < 1e-10, "Expected 23.0, got {result}");
}
