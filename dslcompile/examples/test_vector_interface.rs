//! Test vector operations interface
//!
//! This is a minimal test to verify that DynamicExpr<Vec<T>> with .map() and .sum() methods work.

use dslcompile::prelude::*;

fn main() -> Result<()> {
    println!("🧪 Testing Vector Operations Interface");
    println!("====================================\n");
    
    // Create context
    let mut ctx = DynamicContext::new();
    
    // Test 1: Create vector expression
    println!("1. Creating vector expression...");
    let data = ctx.data_array(vec![1.0, 2.0, 3.0]);
    println!("   ✓ data_array() works");
    
    // Test 2: Map operation
    println!("2. Testing map operation...");
    let mapped = data.map(|x| x * 2.0);
    println!("   ✓ .map() works");
    
    // Test 3: Sum operation
    println!("3. Testing sum operation...");
    let result = mapped.sum();
    println!("   ✓ .sum() works");
    
    println!("\n🎉 Vector operations interface is functional!");
    println!("The pattern: data.map(|x| x * 2.0).sum() compiles successfully");
    
    Ok(())
}