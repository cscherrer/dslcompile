//! Tests to reproduce and fix the measures demo variable indexing bug
//!
//! The bug occurs when we create more variables than we provide in evaluation

use dslcompile::prelude::*;
use frunk::hlist;

#[test]
#[should_panic(expected = "Variable index 6 out of bounds")]
fn test_variable_count_mismatch_reproduces_bug() {
    // Reproduce the exact scenario from measures demo
    let mut ctx = DynamicContext::new();

    // Create 7 variables like in the demo
    let mu = ctx.var::<f64>(); // Variable 0
    let sigma = ctx.var::<f64>(); // Variable 1
    let x = ctx.var::<f64>(); // Variable 2
    let mu2 = ctx.var::<f64>(); // Variable 3
    let sigma2 = ctx.var::<f64>(); // Variable 4
    let x1 = ctx.var::<f64>(); // Variable 5
    let x2 = ctx.var::<f64>(); // Variable 6

    println!(
        "Variables created: mu={}, sigma={}, x={}, mu2={}, sigma2={}, x1={}, x2={}",
        mu.var_id(),
        sigma.var_id(),
        x.var_id(),
        mu2.var_id(),
        sigma2.var_id(),
        x1.var_id(),
        x2.var_id()
    );

    // Create expressions that reference external variables (simplified version of log-density)
    let expr1 = &x1 - &mu; // References variables 5 and 0
    let expr2 = &x2 - &mu2; // References variables 6 and 3
    let joint_expr = expr1 + expr2;

    // Try to evaluate with only 6 values - this should fail
    // Variable index 6 (x2) will be out of bounds
    let result = ctx.eval(&joint_expr, hlist![0.0, 1.0, 1.0, 2.0, 0.5, 3.0]); // 6 values for 7 variables
    println!("Result: {}", result);
}

#[test]
fn test_variable_count_mismatch_fixed() {
    // Same as above but with correct number of values
    let mut ctx = DynamicContext::new();

    let mu = ctx.var::<f64>(); // Variable 0
    let sigma = ctx.var::<f64>(); // Variable 1
    let x = ctx.var::<f64>(); // Variable 2
    let mu2 = ctx.var::<f64>(); // Variable 3
    let sigma2 = ctx.var::<f64>(); // Variable 4
    let x1 = ctx.var::<f64>(); // Variable 5
    let x2 = ctx.var::<f64>(); // Variable 6

    let expr1 = &x1 - &mu; // (3.0 - 0.0) = 3.0
    let expr2 = &x2 - &mu2; // (4.0 - 2.0) = 2.0
    let joint_expr = expr1 + expr2; // 3.0 + 2.0 = 5.0

    // Provide all 7 values needed
    let result = ctx.eval(&joint_expr, hlist![0.0, 1.0, 1.0, 2.0, 0.5, 3.0, 4.0]); // 7 values for 7 variables
    let expected = 5.0;
    assert!(
        (result - expected).abs() < 1e-10,
        "Expected {}, got {}",
        expected,
        result
    );
}

#[test]
fn test_complex_normal_distribution_simulation() {
    // More realistic test simulating the measures demo
    let mut ctx = DynamicContext::new();

    let mu = ctx.var::<f64>();
    let sigma = ctx.var::<f64>();
    let x = ctx.var::<f64>();

    // Simple log-density approximation: -0.5 * ((x - μ) / σ)²
    let centered = &x - &mu;
    let standardized = &centered / &sigma;
    let squared = &standardized * &standardized;
    let log_density = -0.5 * squared;

    // This should work with 3 variables
    let result = ctx.eval(&log_density, hlist![1.0, 0.5, 1.5]); // μ=1.0, σ=0.5, x=1.5

    // Expected: -0.5 * ((1.5 - 1.0) / 0.5)² = -0.5 * (0.5 / 0.5)² = -0.5 * 1² = -0.5
    let expected = -0.5;
    assert!(
        (result - expected).abs() < 1e-10,
        "Expected {}, got {}",
        expected,
        result
    );
}

#[test]
fn test_iid_with_external_variables() {
    // Test the IID summation pattern that was failing
    let mut ctx = DynamicContext::new();

    let mu = ctx.var::<f64>(); // Variable 0
    let sigma = ctx.var::<f64>(); // Variable 1

    // Create summation that uses external variables (like in measures demo)
    let data = vec![1.0, 2.0, 0.5];
    let iid_sum = ctx.sum(&data, |x_i| {
        // Simple log-density: -(x - μ)²
        let diff = &x_i - &mu;
        -(&diff * &diff)
    });

    // This should work - summing over data with external variable μ
    let result = ctx.eval(&iid_sum, hlist![1.5]); // μ = 1.5

    // Expected: -(1-1.5)² + -(2-1.5)² + -(0.5-1.5)² = -0.25 + -0.25 + -1.0 = -1.5
    let expected = -1.5;
    assert!(
        (result - expected).abs() < 1e-10,
        "Expected {}, got {}",
        expected,
        result
    );
}

#[test]
fn test_debug_variable_indices() {
    // Debug test to understand how variable indices are assigned
    let mut ctx = DynamicContext::new();

    let vars: Vec<DynamicExpr<f64>> = (0..10).map(|_| ctx.var()).collect();

    for (i, var) in vars.iter().enumerate() {
        println!("Variable {}: index = {}", i, var.var_id());
        assert_eq!(
            var.var_id(),
            i,
            "Variable index should match creation order"
        );
    }
}
