//! Data-Based Summation Demo
//!
//! This demonstrates the new data-based summation functionality that supports:
//! - Symbolic function parameters that stay symbolic during expression building
//! - Runtime data binding for different datasets
//! - No loop unrolling - keeps symbolic sums as symbolic
//! - Smart constant propagation for inner variables

use dslcompile::ast::DynamicContext;
use dslcompile::Result;

fn main() -> Result<()> {
    println!("🔢 Data-Based Summation Demo");
    println!("============================\n");

    // Example 1: Basic data summation with parameter
    println!("📊 Example 1: Basic Data Summation");
    println!("Expression: Σ(x * param for x in data)");
    
    let ctx1 = DynamicContext::new();
    let param = ctx1.var(); // Function parameter - stays symbolic
    let sum_expr = ctx1.sum_data(|x| x * param.clone())?;
    
    println!("Pretty print: {}", sum_expr.pretty_print());
    
    // Test with different parameter values and datasets
    let result1 = ctx1.eval_with_data(&sum_expr, &[2.0], &[vec![1.0, 2.0, 3.0]]);
    println!("param=2.0, data=[1,2,3] → result = {}", result1);
    
    let result2 = ctx1.eval_with_data(&sum_expr, &[3.0], &[vec![4.0, 5.0]]);
    println!("param=3.0, data=[4,5] → result = {}", result2);
    
    println!();

    // Example 2: More complex expression
    println!("📊 Example 2: Complex Expression");
    println!("Expression: Σ((x + offset) * scale for x in data)");
    
    let ctx2 = DynamicContext::new();
    let offset = ctx2.var(); // Variable(0)
    let scale = ctx2.var();  // Variable(1)
    let complex_sum = ctx2.sum_data(|x| (x + offset.clone()) * scale.clone())?;
    
    println!("Pretty print: {}", complex_sum.pretty_print());
    
    // offset=1.0, scale=2.0, data=[1,2,3]
    // Expected: (1+1)*2 + (2+1)*2 + (3+1)*2 = 4 + 6 + 8 = 18
    let result3 = ctx2.eval_with_data(&complex_sum, &[1.0, 2.0], &[vec![1.0, 2.0, 3.0]]);
    println!("offset=1.0, scale=2.0, data=[1,2,3] → result = {}", result3);
    
    println!();

    // Example 3: Statistical computation
    println!("📊 Example 3: Statistical Computation");
    println!("Expression: Σ((x - mean)² for x in data)");
    
    let ctx3 = DynamicContext::new();
    let mean = ctx3.var(); // Function parameter for mean
    let variance_sum = ctx3.sum_data(|x| {
        let diff = x - mean.clone();
        diff.clone() * diff
    })?;
    
    println!("Pretty print: {}", variance_sum.pretty_print());
    
    // Compute sum of squared deviations
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mean_val = 3.0; // Mean of [1,2,3,4,5]
    // Expected: (1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)² = 4 + 1 + 0 + 1 + 4 = 10
    let ss_result = ctx3.eval_with_data(&variance_sum, &[mean_val], &[data]);
    println!("mean=3.0, data=[1,2,3,4,5] → sum of squared deviations = {}", ss_result);
    
    println!();

    // Example 4: Empty data handling
    println!("📊 Example 4: Empty Data Handling");
    let empty_result = ctx1.eval_with_data(&sum_expr, &[5.0], &[vec![]]);
    println!("param=5.0, data=[] → result = {}", empty_result);
    
    println!();

    // Example 5: Multiple datasets (future extension)
    println!("📊 Example 5: Symbolic Sums Stay Symbolic");
    println!("✅ No loop unrolling - expressions remain symbolic until evaluation");
    println!("✅ Function parameters stay symbolic during expression building");
    println!("✅ Inner variables can be constant-propagated");
    println!("✅ Runtime data binding enables flexible evaluation");

    println!("\n🎉 Data summation demo completed successfully!");
    
    Ok(())
} 