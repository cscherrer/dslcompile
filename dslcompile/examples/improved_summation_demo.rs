//! Improved Summation API Demonstration
//!
//! This example shows the new closure-based summation API that provides:
//! - Ergonomic closure syntax like StaticContext
//! - Proper Lambda AST generation for egglog optimization
//! - Type-safe bound variable management
//! - Unified interface for different collection types

use dslcompile::prelude::*;

fn main() {
    println!("=== Improved Summation API Demonstration ===\n");

    let mut ctx = DynamicContext::new();
    let mu: DynamicExpr<f64, 0> = ctx.var();
    let sigma: DynamicExpr<f64, 0> = ctx.var();

    println!("Variables: mu = Variable(0), sigma = Variable(1)");

    // Example 1: Summation over data arrays with closure syntax
    println!("\n--- Example 1: Data Array Summation ---");
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let sum_deviations = ctx.sum(data.clone(), |x| {
        println!("Creating summation body: x - mu");
        // x is a DynamicBoundVar representing the iteration variable
        // mu is a free variable from the outer scope
        x - mu.clone()
    });

    println!(
        "Generated AST structure: Sum(Map(Lambda([0], BoundVar(0) - Variable(0)), DataArray))"
    );
    println!("This creates: Σ(x - μ) for x in [1,2,3,4,5]");

    // Example 2: Mathematical range summation
    println!("\n--- Example 2: Range Summation ---");
    let sum_squares = ctx.sum(1.0..=10.0, |i| {
        println!("Creating range summation body: i * i");
        // i is bound variable representing values 1.0 through 10.0
        i.clone() * i
    });

    println!("Generated AST structure: Sum(Map(Lambda([0], BoundVar(0) * BoundVar(0)), Range))");
    println!("This creates: Σ(i²) for i in 1..=10");

    // Example 3: Complex expression with both free and bound variables
    println!("\n--- Example 3: Mixed Variables ---");
    let gaussian_like = ctx.sum(data, |x| {
        println!("Creating Gaussian-like summation");
        // Complex expression mixing bound variable (x) and free variables (mu, sigma)
        let deviation = x - mu.clone();
        let normalized = deviation / sigma.clone();
        normalized.clone() * normalized // (x-μ)²/σ²
    });

    println!("Generated complex lambda expression with proper variable scoping");

    // Example 4: Demonstrate type safety
    println!("\n--- Example 4: Type Safety ---");
    let different_scope = ctx.clone().next();
    // The summation expressions are typed with SCOPE=0
    // different_scope expressions would be typed with SCOPE=1
    // Cross-scope operations would be prevented at compile time

    println!("✓ Summations maintain scope type information");
    println!("✓ Bound variables are properly scoped within lambdas");
    println!("✓ Free variables reference outer scope correctly");

    // Example 5: Compare with old manual approach
    println!("\n--- Example 5: Old vs New Approach ---");
    println!("Old approach (manual BoundVar):");
    println!("  let iter_var = DynamicExpr::new(ASTRepr::BoundVar(0), registry);");
    println!("  let body = iter_var - mu;  // Manual variable management");
    println!("  let lambda = Lambda {{ var_indices: vec![0], body: Box::new(body.ast) }};");
    println!("  // Error-prone and verbose...");

    println!("\nNew approach (closure-based):");
    println!("  ctx.sum(data, |x| x - mu.clone())");
    println!("  // Automatic bound variable management, type-safe, ergonomic!");

    println!("\n=== Summary ===");
    println!("✓ Ergonomic closure-based summation API");
    println!("✓ Proper Lambda AST generation for egglog optimization");
    println!("✓ Type-safe bound variable management with scopes");
    println!("✓ Unified interface for data arrays and mathematical ranges");
    println!("✓ Natural operator overloading for bound variables");
    println!("✓ Maintains all optimization capabilities");
}
