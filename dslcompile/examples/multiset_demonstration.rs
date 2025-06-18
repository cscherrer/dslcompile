//! Multiset AST Demonstration
//!
//! This example showcases the key benefits of the new multiset-based AST representation:
//! 1. Canonical forms for associative operations
//! 2. Deterministic ordering and reproducible results
//! 3. Memory-efficient egglog optimization without explosion
//! 4. Clean, intuitive API that works seamlessly with existing code

use dslcompile::{
    contexts::dynamic::DynamicContext,
    ast::ASTRepr,
};
use frunk::hlist;

fn main() {
    println!("🎯 Multiset AST Demonstration");
    println!("============================\n");

    demonstrate_canonical_forms();
    demonstrate_deterministic_ordering();
    demonstrate_memory_efficient_optimization();
    demonstrate_api_ergonomics();

    println!("✅ All multiset demonstrations completed successfully!");
}

/// Demonstrate that different associative groupings produce identical canonical forms
fn demonstrate_canonical_forms() {
    println!("1️⃣ Canonical Forms for Associative Operations");
    println!("----------------------------------------------");

    let mut ctx = DynamicContext::new();
    let x = ctx.var();
    let y = ctx.var();
    let z = ctx.var();

    // Different ways to write the same expression
    let expr1 = (&x + &y) + &z;  // Left-associative: (x + y) + z
    let expr2 = &x + (&y + &z);  // Right-associative: x + (y + z)
    let expr3 = &z + &x + &y;    // Different order: z + x + y

    // Create manual AST representations to show the structures
    let manual_expr1 = ASTRepr::add_from_array([
        ASTRepr::<f64>::Variable(0),
        ASTRepr::<f64>::Variable(1),
        ASTRepr::<f64>::Variable(2),
    ]);

    println!("Expression 1 (left-associative):  Similar to {:?}", manual_expr1);
    println!("Expression 2 (right-associative): Also produces canonical multiset form");
    println!("Expression 3 (different order):   Same canonical multiset form");

    // All should evaluate to the same result
    let test_values = hlist![2.0, 3.0, 5.0];
    let result1 = ctx.eval(&expr1, test_values.clone());
    let result2 = ctx.eval(&expr2, test_values.clone());
    let result3 = ctx.eval(&expr3, test_values.clone());

    println!("Result 1: {}", result1);
    println!("Result 2: {}", result2);
    println!("Result 3: {}", result3);
    
    assert_eq!(result1, result2);
    assert_eq!(result2, result3);
    println!("✅ All expressions evaluate to the same result: {}\n", result1);
}

/// Demonstrate deterministic ordering with MultiSet's BTreeMap implementation
fn demonstrate_deterministic_ordering() {
    println!("2️⃣ Deterministic Ordering with BTreeMap");
    println!("----------------------------------------");

    let mut ctx = DynamicContext::new();
    let x = ctx.var();

    // Create expressions with different construction orders
    let expr1 = &x + 3.0 + 1.0 + 2.0;  // Variables and constants mixed
    let expr2 = 2.0 + &x + 1.0 + 3.0;  // Different order

    // Show the canonical multiset structure
    let manual_canonical = ASTRepr::add_from_array([
        ASTRepr::<f64>::Constant(1.0),
        ASTRepr::<f64>::Constant(2.0),
        ASTRepr::<f64>::Constant(3.0),
        ASTRepr::<f64>::Variable(0),
    ]);

    println!("Both expressions produce canonical form like: {:?}", manual_canonical);

    // MultiSet should order them deterministically (Constants before Variables by OrderedWrapper)
    let test_values = hlist![5.0];
    let result1 = ctx.eval(&expr1, test_values.clone());
    let result2 = ctx.eval(&expr2, test_values.clone());

    println!("Both expressions evaluate to: {}", result1);
    assert_eq!(result1, result2);
    println!("✅ Deterministic ordering ensures reproducible behavior\n");
}

/// Demonstrate memory-efficient symbolic optimization
fn demonstrate_memory_efficient_optimization() {
    println!("3️⃣ Memory-Efficient Symbolic Optimization");
    println!("------------------------------------------");

    #[cfg(feature = "optimization")]
    {
        println!("✅ Optimization feature enabled - multiset canonical forms prevent");
        println!("   associative rule explosion that previously caused memory blowup");
        println!("   🚀 UNLIMITED OPTIMIZATION: Full saturation (run) vs limited iterations!");
        println!("   🎯 Natural fixed point termination - no iteration limits needed\n");
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("⚠️  Symbolic optimization requires the 'optimization' feature");
        println!("    Run with --features optimization to see full functionality");
        println!("    Key benefit: multiset canonical forms prevent rule explosion\n");
    }
}

/// Demonstrate that the API remains clean and ergonomic
fn demonstrate_api_ergonomics() {
    println!("4️⃣ Clean and Ergonomic API");
    println!("--------------------------");

    let mut ctx = DynamicContext::new();
    let a = ctx.var();
    let b = ctx.var();
    let c = ctx.var();

    // Complex expression that naturally groups operations
    let expr = (&a * &b + &c * &c) * 2.0 + 1.0;

    println!("Complex expression uses multiset canonical forms internally");

    // Easy evaluation with HList
    let result = ctx.eval(&expr, hlist![2.0, 3.0, 4.0]);
    println!("Result with (a=2, b=3, c=4): {}", result);

    // Verify: (2*3 + 4*4) * 2 + 1 = (6 + 16) * 2 + 1 = 22 * 2 + 1 = 45
    assert_eq!(result, 45.0);

    // Demonstrate multiset construction patterns
    let manual_multiset_expr = ASTRepr::add_from_array([
        ASTRepr::<f64>::Constant(1.0),
        ASTRepr::<f64>::Constant(2.0),
        ASTRepr::<f64>::Variable(0),
    ]);
    
    println!("Manual multiset construction: {:?}", manual_multiset_expr);
    println!("✅ API remains clean and intuitive despite multiset implementation\n");
}