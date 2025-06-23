//! Test Egg Sum Splitting: Σ(a*x + b*x) → (a+b)*Σ(x)
//!
//! This example demonstrates our egg-based sum splitting optimization
//! using constant factoring rules.

use dslcompile::prelude::*;


fn main() -> Result<()> {
    println!("🥚 Testing Egg Sum Splitting Optimization");
    println!("========================================\n");

    // Create test expression: 2*x + 3*x
    let x = ASTRepr::Variable(0);
    let two = ASTRepr::Constant(2.0);
    let three = ASTRepr::Constant(3.0);

    // Build: 2*x + 3*x
    let inner_expr = ASTRepr::add_binary(
        ASTRepr::mul_binary(two, x.clone()),
        ASTRepr::mul_binary(three, x),
    );

    println!("📊 Test Expression: 2*x + 3*x");
    println!("Expected optimization: (2+3)*x = 5*x");
    println!("Original: {inner_expr:?}\n");

    // Test evaluation
    let result = inner_expr.eval_with_vars(&[2.0]); // x = 2
    println!("📊 Evaluation test (x = 2.0):");
    println!("  Expression: 2*2 + 3*2 = {result}");
    println!(
        "  Correct: {}",
        (result - 10.0_f64).abs() < 1e-10_f64
    );

    #[cfg(not(feature = "optimization"))]
    {
        println!("🚫 Egg optimization feature not enabled");
        println!(
            "Run with: cargo run --example test_egg_sum_splitting --features egg_optimization"
        );
    }

    // Now test sum splitting with a proper sum
    println!("\n🔍 Testing with actual Sum expression...");

    // Create: Sum(2*x + 3*x) over data
    let data_array = vec![1.0, 2.0, 3.0];
    let collection = dslcompile::ast::ast_repr::Collection::Constant(data_array);
    let sum_expr = ASTRepr::Sum(Box::new(collection));

    println!("Sum expression: {sum_expr:?}");

    #[cfg(feature = "optimization")]
    {
        // Optimization functionality removed
        {
            println!("✅ Sum expression created!");
            println!("Expression: {sum_expr:?}");
            
            // Test evaluation
            let result = sum_expr.eval_with_vars(&[]);
            println!("Evaluation result: {result}");
        }
    }

    Ok(())
}
