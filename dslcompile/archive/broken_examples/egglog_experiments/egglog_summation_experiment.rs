//! Egglog Collection Summation Experiment
//!
//! This example tests the egglog-based collection summation system in isolation
//! by disabling the old range-based approach. We'll test:
//! 
//! 1. Basic mathematical range summation
//! 2. Bidirectional optimization capabilities
//! 3. Lambda calculus optimizations
//! 4. Collection operations

use dslcompile::prelude::*;

fn main() -> Result<()> {
    println!("🧪 Egglog Collection Summation Experiment");
    println!("=========================================\n");

    // Test 1: Basic mathematical range summation
    test_basic_range_summation()?;

    // Test 2: Linearity optimization
    test_linearity_optimization()?;

    // Test 3: Constant factor extraction
    test_constant_factor_extraction()?;

    // Test 4: Data iteration (converted to range)
    test_data_iteration()?;

    println!("✅ Egglog collection summation experiment completed!");
    Ok(())
}

fn test_basic_range_summation() -> Result<()> {
    println!("📊 Test 1: Basic Mathematical Range Summation");
    println!("==============================================");

    let ctx = DynamicContext::new();

    // Simple arithmetic series: Σ(i=1 to 10) i
    let sum_expr = ctx.sum(1..=10, |i| i)?;
    let result = ctx.eval(&sum_expr, &[]);
    
    println!("Expression: Σ(i=1 to 10) i");
    println!("Result: {} (expected: 55)", result);
    println!("Pretty print: {}", ctx.pretty_print(&sum_expr));
    println!();

    Ok(())
}

fn test_linearity_optimization() -> Result<()> {
    println!("📊 Test 2: Linearity Optimization");
    println!("==================================");

    let ctx = DynamicContext::new();

    // Test sum splitting: Σ(i + 2*i) should become Σ(i) + Σ(2*i)
    let sum_expr = ctx.sum(1..=5, |i| {
        i.clone() + i * 2.0
    })?;
    let result = ctx.eval(&sum_expr, &[]);
    
    println!("Expression: Σ(i=1 to 5) (i + 2*i) = Σ(i=1 to 5) 3*i");
    println!("Result: {} (expected: 45)", result); // 3 * (1+2+3+4+5) = 3 * 15 = 45
    println!("Pretty print: {}", ctx.pretty_print(&sum_expr));
    println!();

    Ok(())
}

fn test_constant_factor_extraction() -> Result<()> {
    println!("📊 Test 3: Constant Factor Extraction");
    println!("======================================");

    let ctx = DynamicContext::new();

    // Test constant factor: Σ(5*i) should become 5*Σ(i)
    let sum_expr = ctx.sum(1..=4, |i| {
        ctx.constant(5.0) * i
    })?;
    let result = ctx.eval(&sum_expr, &[]);
    
    println!("Expression: Σ(i=1 to 4) 5*i = 5*Σ(i=1 to 4) i");
    println!("Result: {} (expected: 50)", result); // 5 * (1+2+3+4) = 5 * 10 = 50
    println!("Pretty print: {}", ctx.pretty_print(&sum_expr));
    println!();

    Ok(())
}

fn test_data_iteration() -> Result<()> {
    println!("📊 Test 4: Data Iteration (Converted to Range)");
    println!("===============================================");

    let ctx = DynamicContext::new();

    // Test data iteration - this will be converted to range for the experiment
    let data = hlist![1.0, 2.0, 3.0];
    let sum_expr = ctx.sum(data, |x| {
        x * 2.0
    })?;
    let result = ctx.eval(&sum_expr, &[]);
    
    println!("Expression: Σ(x in [1,2,3]) 2*x (converted to range)");
    println!("Result: {} (note: this is experimental conversion)", result);
    println!("Pretty print: {}", ctx.pretty_print(&sum_expr));
    println!();

    Ok(())
} 