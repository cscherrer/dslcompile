//! Demonstration of the new LambdaVar-unified architecture
//!
//! This example shows how both StaticContext and MathFunction now use
//! the superior LambdaVar approach for automatic scope management,
//! eliminating the variable collision issues of the old DynamicContext.

use dslcompile::{
    composition::MathFunction,
    contexts::{StaticConst, StaticContext},
    prelude::*,
};
use frunk::hlist;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== LambdaVar-Unified Architecture Demo ===\n");

    // ============================================================================
    // APPROACH 1: MathFunction with LambdaVar (Runtime Flexibility)
    // ============================================================================

    println!("🚀 APPROACH 1: MathFunction with LambdaVar");
    println!("   ✅ Automatic scope management");
    println!("   ✅ Safe composition");
    println!("   ✅ Natural mathematical syntax");
    println!();

    // Create reusable mathematical functions
    let square_plus_one = MathFunction::from_lambda("square_plus_one", |builder| {
        builder.lambda(|x| x.clone() * x + 1.0) // x² + 1
    });

    let linear = MathFunction::from_lambda("linear", |builder| {
        builder.lambda(|x| x * 2.0 + 3.0) // 2x + 3
    });

    // Safe function composition - no variable collisions!
    let f = square_plus_one.as_callable();
    let g = linear.as_callable();
    let composed = MathFunction::from_lambda("composed", |builder| {
        builder.lambda(|x| f.call(g.call(x))) // f(g(x)) = (2x + 3)² + 1
    });

    // Evaluate the functions
    let x_val = 2.0;
    println!("Input: x = {}", x_val);
    println!(
        "square_plus_one(x) = x² + 1 = {}",
        square_plus_one.eval(hlist![x_val])
    );
    println!("linear(x) = 2x + 3 = {}", linear.eval(hlist![x_val]));
    println!("composed(x) = f(g(x)) = {}", composed.eval(hlist![x_val]));
    println!("  Expected: (2*2 + 3)² + 1 = 7² + 1 = 50");
    println!();

    // ============================================================================
    // APPROACH 2: StaticContext with Lambda-Style Syntax (Compile-Time Optimization)
    // ============================================================================

    println!("🚀 APPROACH 2: StaticContext with Lambda-Style Syntax");
    println!("   ✅ Zero runtime overhead");
    println!("   ✅ No awkward scope threading");
    println!("   ✅ Compile-time optimization");
    println!();

    let mut ctx = StaticContext::new();

    // NEW: Clean lambda syntax - no scope threading!
    let static_square = ctx.lambda(|x| x.clone() * x);
    let result1 = static_square.eval(hlist![3.0]);
    println!("Static lambda: x² at x=3 = {}", result1);

    // Advance to next scope for composition safety
    let mut ctx = ctx.next();

    // Single-argument lambda (avoiding the removed lambda2 method)
    let static_double = ctx.lambda(|x| x * StaticConst::new(2.0));
    let result2 = static_double.eval(hlist![3.5]);
    println!("Static lambda: 2x at x=3.5 = {}", result2);
    println!();

    // ============================================================================
    // COMPARISON: Old vs New Approach
    // ============================================================================

    println!("📊 COMPARISON: Old vs New Approach");
    println!();

    // OLD APPROACH (deprecated - variable collision prone)
    println!("❌ OLD: DynamicContext (DEPRECATED)");
    println!("   Problems:");
    println!("   - Variable index collisions during composition");
    println!("   - Manual scope management required");
    println!("   - Runtime errors instead of compile-time safety");
    println!("   - 'Variable index out of bounds' errors");
    println!();

    // NEW APPROACH (recommended)
    println!("✅ NEW: LambdaVar Approach (RECOMMENDED)");
    println!("   Benefits:");
    println!("   - Automatic scope management");
    println!("   - Safe composition via function calls");
    println!("   - Natural mathematical syntax");
    println!("   - Compile-time safety");
    println!("   - Zero-cost abstractions when possible");
    println!();

    // ============================================================================
    // MIGRATION EXAMPLE
    // ============================================================================

    println!("🔄 MIGRATION EXAMPLE");
    println!();

    println!("// OLD: DynamicContext (collision-prone)");
    println!("let mut ctx = DynamicContext::new();  // ⚠️ DEPRECATED");
    println!("let x = ctx.var();  // Variable(0) - collision prone!");
    println!("let expr = x * x + 1.0;");
    println!();

    println!("// NEW: LambdaVar approach (safe composition)");
    println!("let f = MathFunction::from_lambda(\"square_plus_one\", |builder| {{");
    println!("    builder.lambda(|x| x * x + 1.0)  // Automatic scope management!");
    println!("}});");
    println!();

    // ============================================================================
    // ARCHITECTURAL BENEFITS
    // ============================================================================

    println!("🏗️ ARCHITECTURAL BENEFITS");
    println!();
    println!("1. **Unified Interface**: Both Static and Dynamic contexts use lambda syntax");
    println!("2. **Automatic Scoping**: No manual variable index management");
    println!("3. **Safe Composition**: Function calls prevent variable collisions");
    println!("4. **Natural Syntax**: Mathematical expressions look like math");
    println!("5. **Performance**: Zero-cost when possible, optimized when needed");
    println!();

    println!("✅ SUCCESS: LambdaVar-unified architecture eliminates variable collision issues!");
    println!("   Use MathFunction::from_lambda() or StaticContext::lambda() for new code.");

    Ok(())
}
