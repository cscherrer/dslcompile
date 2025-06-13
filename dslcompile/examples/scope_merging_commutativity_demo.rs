//! Demonstration of Scope Merging Commutativity Fix
//!
//! This example demonstrates that the scope merging system now correctly handles
//! commutativity: expr1 + expr2 should equal expr2 + expr1 even when the expressions
//! come from different contexts.

use dslcompile::{
    contexts::DynamicContext,
    prelude::*,
};
use frunk::hlist;

fn main() {
    println!("=== Scope Merging Commutativity Demonstration ===\n");

    // Create two separate contexts with variables
    let mut ctx1 = DynamicContext::new();
    let x1 = ctx1.var();
    let expr1 = &x1 * 2.0; // 2 * x1

    let mut ctx2 = DynamicContext::new();
    let x2 = ctx2.var();
    let expr2 = &x2 + 1.0; // x2 + 1

    println!("Context 1: expr1 = x1 * 2.0");
    println!("Context 2: expr2 = x2 + 1.0");
    println!();

    // Test commutativity: expr1 + expr2 vs expr2 + expr1
    let combined1 = &expr1 + &expr2; // Should be: 2*x1 + x2 + 1
    let combined2 = &expr2 + &expr1; // Should be: x2 + 1 + 2*x1 = 2*x1 + x2 + 1

    println!("Combined expressions:");
    println!("combined1 = expr1 + expr2");
    println!("combined2 = expr2 + expr1");
    println!();

    // Test with specific values
    let test_values = vec![
        (3.0, 4.0),
        (1.5, 2.5),
        (0.0, -1.0),
        (-2.0, 5.0),
    ];

    println!("Testing commutativity with different values:");
    println!("Format: (x1, x2) -> combined1 result, combined2 result, difference");
    println!();

    let temp_ctx = DynamicContext::new();
    
    for (x1_val, x2_val) in test_values {
        let result1 = temp_ctx.eval(&combined1, hlist![x1_val, x2_val]);
        let result2 = temp_ctx.eval(&combined2, hlist![x1_val, x2_val]);
        let difference = (result1 - result2).abs();
        
        println!("({:4.1}, {:4.1}) -> {:8.3}, {:8.3}, {:e}", 
                 x1_val, x2_val, result1, result2, difference);
        
        // Verify commutativity (should be essentially zero)
        assert!(difference < 1e-12, 
                "Commutativity failed: {} vs {} (diff: {})", result1, result2, difference);
    }

    println!();
    println!("✅ All tests passed! Scope merging is now commutative.");
    println!();

    // Demonstrate the mathematical equivalence
    let x1_test = 3.0;
    let x2_test = 4.0;
    let expected = 2.0 * x1_test + x2_test + 1.0; // 2*3 + 4 + 1 = 11
    
    let result1 = temp_ctx.eval(&combined1, hlist![x1_test, x2_test]);
    let result2 = temp_ctx.eval(&combined2, hlist![x1_test, x2_test]);
    
    println!("Mathematical verification:");
    println!("Expected: 2*{} + {} + 1 = {}", x1_test, x2_test, expected);
    println!("Result 1: {}", result1);
    println!("Result 2: {}", result2);
    println!("Both match expected: {}", 
             (result1 - expected).abs() < 1e-12 && (result2 - expected).abs() < 1e-12);
} 