//! Unified Sum API Test
//!
//! This example demonstrates that the unified sum() method can handle both:
//! 1. Mathematical ranges (1..=10) with symbolic optimization  
//! 2. Data iteration (Vec<f64>) with numerical evaluation

use dslcompile::prelude::*;

fn main() -> Result<()> {
    println!("🚀 Unified Sum API Test");
    println!("=======================\n");

    let math = DynamicContext::new();

    // Test 1: Mathematical range with symbolic optimization
    println!("🎯 Test 1: Mathematical Range (1..=10)");
    let range_result = math.sum(1..=10, |i| {
        i * math.constant(2.0)  // Σ(2*i) = 2*Σ(i) = 2*55 = 110
    })?;
    let range_value = math.eval(&range_result, &[]);
    println!("   Σ(2*i) for i=1..=10");
    println!("   Expected: 110, Actual: {}", range_value);
    
    // Test 2: Data iteration with numerical evaluation
    println!("\n🎯 Test 2: Data Iteration");
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let data_result = math.sum(data, |x| {
        x * math.constant(3.0)  // Sum(3*x) for each x in data = 3*(1+2+3+4+5) = 45
    })?;
    let data_value = math.eval(&data_result, &[]);
    println!("   Σ(3*x) for x in [1,2,3,4,5]");
    println!("   Expected: 45, Actual: {}", data_value);

    // Test 3: More complex mathematical expression
    println!("\n🎯 Test 3: Complex Mathematical Expression");
    let complex_result = math.sum(1..=5, |i| {
        math.constant(2.0) * i.clone() + i.pow(math.constant(2.0))  // Σ(2*i + i²)
    })?;
    let complex_value = math.eval(&complex_result, &[]);
    // Expected: (2*1+1²) + (2*2+2²) + (2*3+3²) + (2*4+4²) + (2*5+5²) = 3+8+15+24+35 = 85
    println!("   Σ(2*i + i²) for i=1..=5");
    println!("   Expected: 85, Actual: {}", complex_value);

    println!("\n🎉 Summary:");
    let range_error = (range_value - 110.0).abs();
    let data_error = (data_value - 45.0).abs();
    let complex_error = (complex_value - 85.0).abs();
    
    println!("   - Mathematical range: Error = {:.2e}", range_error);
    println!("   - Data iteration: Error = {:.2e}", data_error);
    println!("   - Complex expression: Error = {:.2e}", complex_error);
    
    if range_error < 1e-10 && data_error < 1e-10 && complex_error < 1e-10 {
        println!("   🎯 UNIFIED SUM API: WORKING PERFECTLY!");
    } else {
        println!("   ❌ Some tests failed");
    }

    Ok(())
} 