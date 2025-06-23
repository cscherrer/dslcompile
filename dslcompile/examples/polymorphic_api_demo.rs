//! Polymorphic API Demo
//!
//! Demonstrates the clean polymorphic API:
//! - ctx.var::<T>() for any type T
//! - ctx.constant(value: T) for any type T
//!
//! This replaces the confusing mix of var(), data_array(), etc.

use dslcompile::prelude::*;

fn main() -> Result<()> {
    println!("🧪 Polymorphic API Demo");
    println!("=======================\n");

    let mut ctx = DynamicContext::new();

    // Test 1: Scalar variables and constants
    println!("1️⃣ Scalar Variables and Constants");
    println!("---------------------------------");
    
    let x = ctx.var::<f64>();  // Variable(0)
    let pi = ctx.constant(3.14159);  // Constant(3.14159) 
    let expr1 = &x + &pi;
    
    println!("✅ ctx.var::<f64>() - creates variable");
    println!("✅ ctx.constant(3.14159) - creates constant");
    println!("✅ Expression: x + π");
    
    // Test 2: Vector variables and constants  
    println!("\n2️⃣ Vector Variables and Constants");
    println!("----------------------------------");
    
    let data_var = ctx.var::<Vec<f64>>();  // Variable(1) - for runtime data
    let data_const = ctx.constant(vec![1.0, 2.0, 3.0]);  // Constant([1,2,3])
    
    println!("✅ ctx.var::<Vec<f64>>() - creates vector variable");
    println!("✅ ctx.constant(vec![1,2,3]) - creates vector constant");
    
    // Test 3: Integer types work too
    println!("\n3️⃣ Integer Types");
    println!("-----------------");
    
    let n = ctx.var::<i32>();
    let five = ctx.constant(5i32);
    
    println!("✅ ctx.var::<i32>() - creates integer variable");
    println!("✅ ctx.constant(5i32) - creates integer constant");
    
    // Test 4: Demonstrate map/sum works with both
    println!("\n4️⃣ Operations Work with Both");
    println!("-----------------------------");
    
    // Using constant data (embedding)
    let sum_const = data_const.map(|x| x * 2.0).sum();
    println!("✅ Constant data: [1,2,3].map(x → 2x).sum()");
    
    // Using variable data (parameterized) 
    let sum_var = data_var.map(|x| x * 2.0).sum();
    println!("✅ Variable data: data.map(x → 2x).sum()");
    
    println!("\n🎯 API Summary:");
    println!("   • ctx.var::<T>() - parameterized variable of type T");
    println!("   • ctx.constant(value: T) - embedded constant of type T");
    println!("   • Works for scalars, vectors, integers, any PartialOrd type");
    println!("   • No more confusing data_array(), special methods, magic types");
    
    println!("\n✨ Clean, polymorphic, and consistent!");

    Ok(())
}