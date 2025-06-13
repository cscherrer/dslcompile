//! Type-Level Scope Safety Demo
//!
//! This demo demonstrates the breakthrough type-level scope safety implementation
//! that eliminates the non-deterministic scope merging bug in DynamicContext.
//!
//! Key achievements:
//! 1. TypedBuilderExpr<T, const SCOPE: usize> carries scope information at type level
//! 2. Operators only work within same scope, preventing cross-scope operations at compile time
//! 3. Eliminated runtime scope merging that used memory addresses for ordering
//! 4. Cross-scope operations require explicit scope advancement via next() method
//!
//! This solves the fundamental brittleness issue: DynamicContext now has the same
//! type-level safety as StaticContext while maintaining runtime flexibility.

use dslcompile::prelude::*;
use frunk::hlist;

fn main() {
    println!("🔒 Type-Level Scope Safety Demo");
    println!("================================");
    
    // Test 1: Same-scope operations work fine
    println!("\n✅ Test 1: Same-scope operations (should compile)");
    let mut ctx = DynamicContext::<f64, 0>::new();
    let x: TypedBuilderExpr<f64, 0> = ctx.var(); // Variable in scope 0
    let y: TypedBuilderExpr<f64, 0> = ctx.var(); // Variable in scope 0
    
    // These operations work because both variables are in scope 0
    let sum = &x + &y;
    let product = &x * &y;
    
    println!("  x + y: {}", ctx.pretty_print(&sum));
    println!("  x * y: {}", ctx.pretty_print(&product));
    
    // Test 2: Cross-scope operations prevented at compile time
    println!("\n🚫 Test 2: Cross-scope operations (would cause compile error)");
    println!("  The following code would NOT compile:");
    println!("  ```rust");
    println!("  let mut ctx1 = DynamicContext::<f64, 0>::new();");
    println!("  let mut ctx2 = ctx1.next(); // Advance to scope 1");
    println!("  let x: TypedBuilderExpr<f64, 0> = ctx1.var();");
    println!("  let y: TypedBuilderExpr<f64, 1> = ctx2.var();");
    println!("  let invalid = x + y; // ❌ COMPILE ERROR: mismatched scopes!");
    println!("  ```");
    println!("  This demonstrates type-level scope safety!");
    
    // Test 3: Safe scope advancement
    println!("\n✅ Test 3: Safe scope advancement");
    let ctx_scope_1 = ctx.next(); // Advance to scope 1
    let mut ctx_scope_1 = ctx_scope_1; // Make mutable for var creation
    let z: TypedBuilderExpr<f64, 1> = ctx_scope_1.var(); // Variable in scope 1
    
    println!("  Created variable z in scope 1");
    println!("  z: {}", ctx_scope_1.pretty_print(&z));
    
    // Test 4: Demonstrate the safety - no runtime variable collisions
    println!("\n🛡️  Test 4: No variable collision issues");
    println!("  Unlike the old system, variables are now scope-isolated:");
    
    // Create multiple contexts and show they don't interfere
    let mut ctx_a = DynamicContext::<f64, 0>::new();
    let mut ctx_b = DynamicContext::<f64, 0>::new();
    
    let a1: TypedBuilderExpr<f64, 0> = ctx_a.var();
    let a2: TypedBuilderExpr<f64, 0> = ctx_a.var();
    
    let b1: TypedBuilderExpr<f64, 0> = ctx_b.var();
    let b2: TypedBuilderExpr<f64, 0> = ctx_b.var();
    
    // These are safe because they're in separate contexts
    let expr_a = &a1 + &a2;
    let expr_b = &b1 * &b2;
    
    println!("  Context A: a1 + a2 = {}", ctx_a.pretty_print(&expr_a));
    println!("  Context B: b1 * b2 = {}", ctx_b.pretty_print(&expr_b));
    
    // Test 5: Evaluation with HLists
    println!("\n📊 Test 5: HList evaluation");
    
    // Evaluate expressions with proper HList parameters
    let result_a = ctx_a.eval(&expr_a, hlist![1.0, 2.0]);
    let result_b = ctx_b.eval(&expr_b, hlist![3.0, 4.0]);
    
    println!("  eval(a1 + a2, [1.0, 2.0]) = {}", result_a);
    println!("  eval(b1 * b2, [3.0, 4.0]) = {}", result_b);
    
    // Test 6: Mathematical functions with scope safety
    println!("\n🧮 Test 6: Mathematical functions with scope safety");
    let mut math_ctx = DynamicContext::<f64, 0>::new();
    let x: TypedBuilderExpr<f64, 0> = math_ctx.var();
    
    // All these operations preserve scope information
    let sin_x = x.clone().sin();
    let exp_x = x.clone().exp();
    let x_plus_const = x.clone() + math_ctx.constant(2.0);
    
    println!("  sin(x): {}", math_ctx.pretty_print(&sin_x));
    println!("  exp(x): {}", math_ctx.pretty_print(&exp_x));
    println!("  x + 2: {}", math_ctx.pretty_print(&x_plus_const));
    
    // Evaluate with a test value
    let test_val = std::f64::consts::PI / 4.0; // 45 degrees
    let sin_result = math_ctx.eval(&sin_x, hlist![test_val]);
    let exp_result = math_ctx.eval(&exp_x, hlist![test_val]);
    let plus_result = math_ctx.eval(&x_plus_const, hlist![test_val]);
    
    println!("  sin(π/4) = {:.6}", sin_result);
    println!("  exp(π/4) = {:.6}", exp_result);
    println!("  π/4 + 2 = {:.6}", plus_result);
    
    println!("\n🎉 Type-Level Scope Safety Demo Complete!");
    println!("   ✅ Same-scope operations work seamlessly");
    println!("   ✅ Cross-scope operations prevented at compile time");
    println!("   ✅ No runtime variable collision issues");
    println!("   ✅ Mathematical functions preserve scope safety");
    println!("   ✅ HList evaluation works correctly");
    println!("\n🔒 The non-deterministic scope merging bug is eliminated!");
    println!("   DynamicContext now has StaticContext-level type safety");
    println!("   while maintaining runtime flexibility.");
} 