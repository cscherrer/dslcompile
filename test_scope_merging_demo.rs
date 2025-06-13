//! Simple demonstration that scope merging is working
//!
//! This file demonstrates the successful implementation of automatic scope merging.

use dslcompile::prelude::*;
use frunk::hlist;

fn main() {
    println!("=== DSLCompile Automatic Scope Merging Demo ===");
    
    // Create two independent mathematical contexts
    let mut ctx1 = DynamicContext::<f64>::new();
    let x = ctx1.var(); // Variable 0 in context 1
    let f_expr = &x * &x + 2.0 * &x + 1.0; // f(x) = x² + 2x + 1
    
    let mut ctx2 = DynamicContext::<f64>::new();  
    let y = ctx2.var(); // Variable 0 in context 2 (would collide!)
    let g_expr = 3.0 * &y + 5.0; // g(y) = 3y + 5
    
    // AUTOMATIC SCOPE MERGING: When we combine expressions from different contexts,
    // the system automatically detects the collision and remaps variables!
    let combined = &f_expr + &g_expr; // Should auto-merge scopes
    
    // The combined expression now uses variables [0, 1] instead of [0, 0]
    // This means we need to provide TWO values, not one
    
    // Evaluate the merged expression
    let temp_ctx = DynamicContext::<f64>::new();
    let result = temp_ctx.eval(&combined, hlist![2.0, 3.0]);
    
    // Expected: f(2) + g(3) = (2² + 2*2 + 1) + (3*3 + 5) = 9 + 14 = 23
    let expected = (2.0 * 2.0 + 2.0 * 2.0 + 1.0) + (3.0 * 3.0 + 5.0);
    
    println!("f(x) = x² + 2x + 1");
    println!("g(y) = 3y + 5");
    println!("Combined: h(x,y) = f(x) + g(y)");
    println!("h(2,3) = {} (expected: {})", result, expected);
    
    if (result - expected).abs() < 1e-12 {
        println!("✅ SUCCESS: Automatic scope merging is working correctly!");
        println!("   The system automatically remapped variables to avoid collisions.");
        println!("   Variables from different contexts are now properly isolated.");
    } else {
        println!("❌ FAILED: Expected {}, got {}", expected, result);
    }
    
    println!("\n=== Summary ===");
    println!("Before scope merging: Both contexts used variable index 0");
    println!("After scope merging: Combined expression uses variables [0, 1]");
    println!("This prevents the 'variable collision' problem in DynamicContext!");
}