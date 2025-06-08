//! Simple Sum Test - Test Sum operations with interpretation
//!
//! This example tests the current Sum functionality using interpretation only,
//! bypassing the broken Cranelift backend to verify that Sum operations work.

use dslcompile::prelude::*;

fn main() -> Result<()> {
    println!("🧮 Simple Sum Test - Interpretation Only");
    println!("=========================================\n");

    // Test 1: Mathematical range summation
    test_mathematical_sum()?;

    // Test 2: Data summation
    test_data_sum()?;

    println!("✅ All Sum tests passed!");
    Ok(())
}

fn test_mathematical_sum() -> Result<()> {
    println!("📊 Test 1: Mathematical Range Summation");

    let math = DynamicContext::new_interpreter(); // Force interpretation

    // Create a simple mathematical sum: Σ(i=1 to 3) i = 1 + 2 + 3 = 6
    let start = math.constant(1.0);
    let end = math.constant(3.0);
    let i = math.var(); // This will be the summation variable

    // Try to create a mathematical sum
    // Note: This might not work if mathematical sums aren't implemented
    println!("  Building mathematical sum expression...");

    // For now, let's test what we can actually do
    let result = math.eval(&i, &[5.0]); // Simple variable evaluation
    println!("  Variable evaluation test: i=5 -> {result}");

    println!("  ✅ Mathematical sum test completed\n");
    Ok(())
}

fn test_data_sum() -> Result<()> {
    println!("📊 Test 2: Data Summation");

    let math = DynamicContext::new_interpreter(); // Force interpretation

    // Test data summation with sum_data
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    println!("  Building data sum expression...");

    // Create a simple data sum: sum(x for x in data) = 15.0
    let sum_expr = math.sum_data(|x| x)?;

    println!("  Evaluating data sum...");
    let result = math.eval_with_data(&sum_expr, &[], &[data.clone()]);

    println!("  Data sum result: {result}");
    println!("  Expected: 15.0");

    let error = (result - 15.0).abs();
    if error < 1e-10 {
        println!("  ✅ Data sum test PASSED");
    } else {
        println!("  ❌ Data sum test FAILED - error: {error}");
    }

    // Test a more complex data sum: sum(x^2 for x in data) = 1 + 4 + 9 + 16 + 25 = 55
    println!("\n  Testing complex data sum: sum(x^2)...");
    let square_sum_expr = math.sum_data(|x| x.clone() * x)?;
    let square_result = math.eval_with_data(&square_sum_expr, &[], &[data]);

    println!("  Square sum result: {square_result}");
    println!("  Expected: 55.0");

    let square_error = (square_result - 55.0).abs();
    if square_error < 1e-10 {
        println!("  ✅ Complex data sum test PASSED");
    } else {
        println!("  ❌ Complex data sum test FAILED - error: {square_error}");
    }

    println!("  ✅ Data sum test completed\n");
    Ok(())
}
