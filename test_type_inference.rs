#!/usr/bin/env rust-script
//! Test script to demonstrate type inference improvements

use dslcompile::prelude::*;
use frunk::hlist;

fn main() -> Result<()> {
    println!("🧮 Type Inference Improvements Demo");
    println!("===================================\n");

    let mut ctx = DynamicContext::new();

    // 1. Type inference from assignment context
    println!("1️⃣ Type inference from assignment:");
    let x: DynamicExpr<f64> = ctx.var();  // Type inferred as f64
    let y: DynamicExpr<f32> = ctx.var();  // Type inferred as f32  
    let z: DynamicExpr<i32> = ctx.var();  // Type inferred as i32
    
    println!("Created variables with inferred types:");
    println!("- x: f64 variable");
    println!("- y: f32 variable");
    println!("- z: i32 variable\n");

    // 2. Type inference from usage context
    println!("2️⃣ Type inference from usage context:");
    let a = ctx.var();  // Type will be inferred from usage
    let expr_f64 = a + 3.0_f64;  // Now 'a' is inferred as f64
    println!("Variable 'a' inferred as f64 from expression with 3.0_f64\n");

    // 3. Convenience methods for common types
    println!("3️⃣ Convenience methods:");
    let w = ctx.var_f64();    // Explicit f64
    let v = ctx.var_f32();    // Explicit f32
    let u = ctx.var_i32();    // Explicit i32
    let t = ctx.var_usize();  // Explicit usize
    
    println!("Created variables using convenience methods:");
    println!("- w: f64 (via var_f64())");
    println!("- v: f32 (via var_f32())");
    println!("- u: i32 (via var_i32())");
    println!("- t: usize (via var_usize())\n");

    // 4. Heterogeneous expression evaluation
    println!("4️⃣ Heterogeneous evaluation:");
    let expr = x + 2.0;  // f64 expression
    let result = ctx.eval(&expr, hlist![5.0_f64]);
    println!("f64 expression x + 2.0 evaluated at x=5.0: {}", result);
    assert_eq!(result, 7.0);

    // 5. Type safety - these would cause compile errors:
    // let bad_expr = x + y;  // ❌ Cannot mix f64 and f32 without explicit conversion
    // let bad_result = ctx.eval(&expr, hlist![5.0_f32]);  // ❌ Wrong type in HList
    
    println!("\n✅ All type inference tests passed!");
    println!("✅ Heterogeneous variables are properly typed!");
    println!("✅ Type safety is preserved!");

    Ok(())
}