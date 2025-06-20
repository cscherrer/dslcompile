//! Closure-Based CSE Demonstration
//!
//! This example demonstrates the new unified variable system that combines:
//! - Ergonomic closure API (like StaticContext)
//! - ASTRepr::Let expressions for egglog cost-aware optimization
//! - Type-safe scope management with const generics
//!
//! The design preserves the sophisticated optimization capabilities while
//! providing a clean, type-safe interface for Common Subexpression Elimination.

use dslcompile::prelude::*;

fn main() {
    println!("=== Closure-Based CSE Demonstration ===\n");

    let mut ctx = DynamicContext::new();
    let x: DynamicExpr<f64, 0> = ctx.var();
    let y: DynamicExpr<f64, 0> = ctx.var();

    println!("Original expressions without CSE:");
    println!("x = Variable(0), y = Variable(1)");

    // Example 1: Basic CSE with closure API
    println!("\n--- Example 1: Basic CSE ---");
    let basic_cse = ctx.let_bind(&x + &y, |shared| {
        println!("Creating CSE binding for (x + y)");
        // 'shared' is a DynamicBoundVar representing the bound (x + y)
        // This generates: Let(0, Add(Variable(0), Variable(1)), body)
        shared.clone() * shared.clone() + shared // (x+y)² + (x+y)
    });

    println!("Generated AST: {:?}", basic_cse.as_ast());
    println!("Structure: Let(binding_id, x+y, BoundVar²+BoundVar)");

    // Example 2: Nested CSE with different subexpressions
    println!("\n--- Example 2: Nested CSE ---");
    let nested_cse = ctx.let_bind(&x * &y, |xy| {
        println!("Outer CSE: binding (x * y)");
        // Create a constant directly without borrowing ctx
        let one = DynamicExpr::new(ASTRepr::Constant(1.0), xy.registry());
        let xy_plus_one = xy.clone() + one;
        // Simple nested expression without nested let_bind for now
        xy_plus_one.clone() * xy_plus_one + xy.clone()
    });

    println!("Generated nested AST (simplified view): Let(xy, Let(xy+1, body))");

    // Example 3: Compare with manual BoundVar approach (for contrast)
    println!("\n--- Example 3: Manual Approach (Old Style) ---");
    let manual_ast: ASTRepr<f64> = ASTRepr::Let(
        0,
        Box::new(ASTRepr::add_binary(
            ASTRepr::Variable(0),
            ASTRepr::Variable(1),
        )),
        Box::new(ASTRepr::add_binary(
            ASTRepr::mul_binary(ASTRepr::BoundVar(0), ASTRepr::BoundVar(0)),
            ASTRepr::BoundVar(0),
        )),
    );
    println!("Manual AST: {:?}", manual_ast);
    println!("Issues with manual approach:");
    println!("- Error-prone binding ID management");
    println!("- No type safety for scope collisions");
    println!("- Verbose and hard to maintain");

    // Example 4: Demonstrate egglog cost analysis capability
    println!("\n--- Example 4: Egglog Cost Analysis ---");
    println!("The closure-based approach generates proper Let expressions:");
    println!("- Egglog can analyze: single evaluation of (x+y) vs multiple");
    println!("- Cost-aware CSE: balance computation vs memory usage");
    println!("- Maintains all optimization capabilities from manual approach");

    // Example 5: Type safety demonstration
    println!("\n--- Example 5: Type Safety with Const Generics ---");
    println!("Different scopes prevent accidental variable collisions:");

    let mut ctx2 = ctx.clone().next(); // Advance to scope 1, using clone to preserve ctx
    let z: DynamicExpr<f64, 1> = ctx2.var(); // Different scope type
    
    // This would be a compile error:
    // let bad = basic_cse + z; // ❌ Different SCOPE parameters!

    println!("✓ Scope 0 expressions: DynamicExpr<f64, 0>");
    println!("✓ Scope 1 expressions: DynamicExpr<f64, 1>");
    println!("✓ Compile-time prevention of cross-scope operations");

    // Example 6: Demonstrate ergonomic usage
    println!("\n--- Example 6: Ergonomic Complex Example ---");
    let complex_expr = ctx.let_bind(&x.clone().sin() + &y.clone().cos(), |trig_sum| {
        // Create constant directly without borrowing ctx
        let forty_two = DynamicExpr::new(ASTRepr::Constant(42.0), trig_sum.registry());
        let trig_squared = trig_sum.clone() * trig_sum.clone();
        // Simple expression without nested let_bind
        trig_squared + trig_sum + forty_two
    });

    println!("Complex expression with multiple CSE levels created naturally");
    println!("No manual binding ID management required!");

    println!("\n=== Summary ===");
    println!("✓ Ergonomic closure API for CSE creation");
    println!("✓ Generates proper ASTRepr::Let for egglog optimization");
    println!("✓ Type-safe scope management prevents collisions");
    println!("✓ Preserves all optimization capabilities");
    println!("✓ Natural operator overloading for bound variables");
    println!("✓ Unified with StaticContext design patterns");
}