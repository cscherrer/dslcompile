//! Test the heterogeneous fix for parameterized data evaluation
//!
//! This test verifies that the heterogeneous evaluation approach fixes
//! the variable indexing issues we were having with Vec<f64> data.

use dslcompile::prelude::*;

#[test]
fn test_heterogeneous_parameterized_data() {
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
    
    // Use heterogeneous evaluation instead of HList evaluation
    let storage = (test_mu, test_sigma, test_data.clone());
    let result = ctx.eval_heterogeneous(&expr, storage);
    
    // Expected: sum of (1-0)/1 + (2-0)/1 + (3-0)/1 = 1 + 2 + 3 = 6
    let expected = 6.0;
    assert!((result - expected).abs() < 1e-10);
    
    println!("✅ Heterogeneous parameterized data evaluation works correctly");
    println!("   Variables: μ={}, σ={}, data=Variable(2)", mu.var_id(), sigma.var_id());
    println!("   Result: {result}, Expected: {expected}");
}

#[test]
fn test_heterogeneous_simple_scalars() {
    let mut ctx = DynamicContext::new();
    
    let x = ctx.var();
    let y = ctx.var();
    let expr = &x + &y;
    
    // Test with heterogeneous storage
    let storage = (3.0_f64, 4.0_f32); // Different types, should auto-convert
    let result: f64 = ctx.eval_heterogeneous(&expr, storage);
    
    assert!((result - 7.0_f64).abs() < 1e-10);
    println!("✅ Heterogeneous scalar evaluation works: 3.0 + 4.0 = {}", result);
}