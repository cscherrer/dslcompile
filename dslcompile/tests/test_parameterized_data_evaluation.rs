//! Regression test for parameterized data evaluation bug
//!
//! This test captures the bug where Variable index 2 is out of bounds for evaluation
//! when using parameterized data variables. This ensures we don't regress after fixing it.

use dslcompile::prelude::*;
use frunk::hlist;

/// Test that demonstrates the parameterized data evaluation bug
/// 
/// Bug: When creating expressions with data as Variable(2), evaluation fails
/// with "Variable index 2 is out of bounds for evaluation" when only passing
/// 2 parameters in hlist![mu, sigma].
#[test]
fn test_parameterized_data_evaluation_bug() {
    let mut ctx = DynamicContext::new();
    
    // Create variables: μ, σ, data
    let mu = ctx.var();     // Variable(0)
    let sigma = ctx.var();  // Variable(1) 
    let data_var = ctx.var::<Vec<f64>>(); // Variable(2) - data parameter
    
    // Create a simple expression that uses all three variables
    // sum(data_var) where each element x is processed as (x - mu) / sigma
    let expr = data_var.map(|x| (x - &mu) / &sigma).sum();
    
    // Generate test data AFTER expression creation (proper parameterization)
    let test_data = vec![1.0, 2.0, 3.0];
    let test_mu = 0.0;
    let test_sigma = 1.0;
    
    // This should work but currently panics with "Variable index 2 is out of bounds"
    // when we only pass [test_mu, test_sigma] instead of [test_mu, test_sigma, test_data]
    let result = ctx.eval(&expr, hlist![test_mu, test_sigma, test_data.clone()]);
    
    // Expected: sum of (1-0)/1 + (2-0)/1 + (3-0)/1 = 1 + 2 + 3 = 6
    let expected = 6.0;
    assert!((result - expected).abs() < 1e-10);
    
    println!("✅ Parameterized data evaluation works correctly");
    println!("   Variables: μ={}, σ={}, data=Variable(2)", mu.var_id(), sigma.var_id());
    println!("   Result: {result}, Expected: {expected}");
}

/// Test the specific case from advanced_iid_normal_llvm_demo that was failing
#[test]
fn test_iid_normal_parameterized_data() {
    /// Normal distribution with embedded parameters
    #[derive(Debug)]
    struct Normal {
        mean: DynamicExpr<f64>,
        std_dev: DynamicExpr<f64>,
    }

    impl Normal {
        fn new(mean: DynamicExpr<f64>, std_dev: DynamicExpr<f64>) -> Self {
            Self { mean, std_dev }
        }

        /// Compute log-density: -0.5 * ln(2π) - ln(σ) - 0.5 * ((x-μ)/σ)²
        fn log_density(&self, x: DynamicExpr<f64>) -> DynamicExpr<f64> {
            let log_2pi = (2.0 * std::f64::consts::PI).ln();
            let neg_half = -0.5;

            let centered = x - &self.mean; // (x - μ)
            let standardized = &centered / &self.std_dev; // (x - μ) / σ
            let squared = &standardized * &standardized; // ((x - μ) / σ)²

            // Complete log-density formula
            neg_half * log_2pi - self.std_dev.clone().ln() + neg_half * &squared
        }
    }

    /// IID wrapper for any measure
    #[derive(Debug)]
    struct IID<T> {
        measure: T,
    }

    impl<T> IID<T> {
        fn new(measure: T) -> Self {
            Self { measure }
        }
    }

    impl IID<Normal> {
        /// Compute log-density for IID normal: Σ measure.log_density(xi) for xi in data
        fn log_density(&self, x: DynamicExpr<Vec<f64>>) -> DynamicExpr<f64> {
            x.map(|xi| self.measure.log_density(xi)).sum()
        }
    }

    let mut ctx = DynamicContext::new();

    // Create parameter variables
    let mu = ctx.var(); // Variable(0)
    let sigma = ctx.var(); // Variable(1)
    let data_var = ctx.var::<Vec<f64>>(); // Variable(2)

    // Create Normal and IID wrapper
    let normal = Normal::new(mu.clone(), sigma.clone());
    let iid_normal = IID::new(normal);

    // Create IID expression with parameterized data
    let iid_expr = iid_normal.log_density(data_var);

    // Generate test data AFTER expression compilation
    let sample_data = vec![1.0, 2.0, 0.5, 1.5, 0.8];
    let test_mu = 0.0;
    let test_sigma = 1.0;

    // This should work with all three parameters
    let result = ctx.eval(&iid_expr, hlist![test_mu, test_sigma, sample_data.clone()]);

    // Verify by manual computation
    let manual_sum: f64 = sample_data
        .iter()
        .map(|&x| {
            -0.5 * (2.0 * std::f64::consts::PI).ln() - test_sigma.ln() - 0.5 * ((x - test_mu) / test_sigma).powi(2)
        })
        .sum();

    assert!((result - manual_sum).abs() < 1e-10);
    
    println!("✅ IID Normal parameterized data evaluation works correctly");
    println!("   Result: {result:.6}, Manual: {manual_sum:.6}");
}

/// Test that shows the parameterized data approach (the main functionality we care about)
#[test] 
fn test_embedded_vs_parameterized_data() {
    let sample_data = vec![1.0, 2.0, 3.0];
    
    // NEW WAY: Parameterize data (proper compilation)
    let mut ctx = DynamicContext::new();
    let data_param = ctx.var::<Vec<f64>>();
    let parameterized_expr = data_param.map(|x| &x * &x).sum(); // x² for each element  
    
    // Data passed as parameter
    let parameterized_result = ctx.eval(&parameterized_expr, hlist![sample_data.clone()]);
    
    // Should give result: 1² + 2² + 3² = 14
    let expected = 14.0;
    assert!((parameterized_result - expected).abs() < 1e-10);
    
    println!("✅ Parameterized data approach works");
    println!("   Parameterized result: {parameterized_result}");
    println!("   Expected: {expected}");
    
    // Key advantage: parameterized expressions can be compiled once and reused with different data
    let different_data = vec![4.0, 5.0, 6.0];
    let new_result = ctx.eval(&parameterized_expr, hlist![different_data]);
    let new_expected = 4.0*4.0 + 5.0*5.0 + 6.0*6.0; // 16 + 25 + 36 = 77
    assert!((new_result - new_expected).abs() < 1e-10);
    
    println!("   Reused with different data: {new_result} (expected: {new_expected})");
    
    // Also test with empty data
    let empty_data: Vec<f64> = vec![];
    let empty_result = ctx.eval(&parameterized_expr, hlist![empty_data]);
    assert!((empty_result - 0.0).abs() < 1e-10);
    println!("   Empty data result: {empty_result} (expected: 0.0)");
}